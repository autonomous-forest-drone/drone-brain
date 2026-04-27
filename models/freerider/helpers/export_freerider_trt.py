#!/usr/bin/env python3
"""
Export the best PPO Freerider actor to ONNX then TensorRT FP16.

Usage (run on the Jetson — trtexec output is GPU-specific):
    python export_freerider_trt.py

Place the SB3 checkpoint zip inside freerider/model/ before running.
The script picks the single .zip found there and writes the outputs
alongside it:
    freerider/model/freerider_actor.onnx
    freerider/model/freerider_actor.trt

Architecture (v5 actor path):
    image (1, 3, 144, 256) ──► CNN ──► Linear(cnn_out, 256)+ReLU ──┐
    state (1, 1)  ─────────────────────────────────────────────────►cat(257)
                                                                     └──► MLP[64,64] ──► Linear(64,1) ──► action (1,1)

VecNormalize was trained with norm_obs=False — raw depth [0,1] is used
directly, no normalisation needed at inference time.
"""

import argparse
import glob
import os
import subprocess

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'model'))

# ---------------------------------------------------------------------------
# Architecture constants — must match the values used during training
# ---------------------------------------------------------------------------
CNN_FEATURES_DIM   = 256   # CNN head output dim
STATE_DIM          = 1     # accumulated action scalar
ACTOR_FEATURES_DIM = CNN_FEATURES_DIM + STATE_DIM   # 257
PRIVILEGED_DIM     = 45    # critic-only input; not used at inference


# ---------------------------------------------------------------------------
# v5 custom classes — embedded here so the Jetson does not need the
# training repo.  Architectures must exactly match the saved checkpoint.
# ---------------------------------------------------------------------------

class ActorFeaturesExtractor(BaseFeaturesExtractor):
    """CNN over stacked depth frames + accumulated action scalar → 257-dim."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = ACTOR_FEATURES_DIM):
        super().__init__(observation_space, features_dim=features_dim)

        n_ch = observation_space['image'].shape[0]   # = 3 (stacked frames)

        self.cnn = nn.Sequential(
            nn.Conv2d(n_ch, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            cnn_out = self.cnn(torch.zeros(1, *observation_space['image'].shape)).shape[1]

        self.cnn_head = nn.Sequential(nn.Linear(cnn_out, CNN_FEATURES_DIM), nn.ReLU())

    def forward(self, obs):
        cnn_features = self.cnn_head(self.cnn(obs['image']))   # (B, 256)
        return torch.cat([cnn_features, obs['state']], dim=1)  # (B, 257)


class AsymmetricActorCriticPolicy(MultiInputActorCriticPolicy):
    """Actor uses CNN + state; critic uses separate privileged MLP (not exported)."""

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.critic_net = nn.Sequential(
            nn.Linear(PRIVILEGED_DIM, 256), nn.ReLU(),
            nn.Linear(256, 256),           nn.ReLU(),
        ).to(self.device)
        self.value_net = nn.Linear(256, 1).to(self.device)
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _critic_value(self, privileged):
        return self.value_net(self.critic_net(privileged))

    def forward(self, obs, deterministic=False):
        features     = self.extract_features(obs)
        latent_pi    = self.mlp_extractor.forward_actor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions      = distribution.get_actions(deterministic=deterministic)
        log_prob     = distribution.log_prob(actions)
        values       = self._critic_value(obs['privileged'])
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        features     = self.extract_features(obs)
        latent_pi    = self.mlp_extractor.forward_actor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob     = distribution.log_prob(actions)
        entropy      = distribution.entropy()
        values       = self._critic_value(obs['privileged'])
        return values, log_prob, entropy

    def predict_values(self, obs):
        return self._critic_value(obs['privileged'])


# ---------------------------------------------------------------------------
# ONNX wrapper — actor path only
# ---------------------------------------------------------------------------

class FreeriderActorWrapper(nn.Module):
    """Wraps the actor path of AsymmetricActorCriticPolicy for ONNX tracing.

    Inputs
    ------
    image : (B, 3, 144, 256) float32 — stacked depth frames in [0, 1]
    state : (B, 1)           float32 — accumulated smoothed lateral action

    Output
    ------
    action : (B, 1) float32 — raw lateral command in ℝ  (clip to [-1, 1] at runtime)
    """

    def __init__(self, policy):
        super().__init__()
        self.features_extractor = policy.features_extractor
        self.policy_net         = policy.mlp_extractor.policy_net
        self.action_net         = policy.action_net

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        obs      = {'image': image, 'state': state}
        features = self.features_extractor(obs)   # (B, 257)
        latent   = self.policy_net(features)      # (B, 64)
        return self.action_net(latent)            # (B, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_checkpoint_in_model_dir() -> str:
    """Return the single .zip file in freerider/model/. Raises if none or many."""
    zips = glob.glob(os.path.join(MODEL_DIR, '*.zip'))
    if not zips:
        raise FileNotFoundError(
            f'No .zip checkpoint found in {MODEL_DIR}\n'
            'Copy your SB3 checkpoint zip there and re-run.'
        )
    if len(zips) > 1:
        raise RuntimeError(
            f'Multiple .zip files found in {MODEL_DIR}:\n'
            + '\n'.join(f'  {z}' for z in zips)
            + '\nLeave only one checkpoint zip and re-run.'
        )
    return zips[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Export Freerider actor → ONNX → TensorRT FP16')
    parser.add_argument('--out_dir', default=MODEL_DIR,
                        help='Output directory for ONNX and TRT files (default: freerider/model/)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Locate checkpoint — must be manually placed in freerider/model/
    # ------------------------------------------------------------------
    ckpt_path = find_checkpoint_in_model_dir()
    print(f'[export] Loading checkpoint: {ckpt_path}')

    # ------------------------------------------------------------------
    # Load SB3 policy on CPU (tracing does not require CUDA)
    #
    # custom_objects bypasses the Python 3.10 / 3.11 pickle mismatch:
    # the checkpoint was saved on Python 3.11+ (18-field code objects)
    # and cannot be unpickled on Python 3.10.  We supply policy_class
    # and policy_kwargs directly — identical to the training-time values.
    # ------------------------------------------------------------------
    model = PPO.load(
        ckpt_path,
        device='cpu',
        custom_objects={
            'policy_class': AsymmetricActorCriticPolicy,
            'policy_kwargs': dict(
                features_extractor_class=ActorFeaturesExtractor,
                features_extractor_kwargs=dict(features_dim=ACTOR_FEATURES_DIM),
                net_arch=dict(pi=[64, 64], vf=[]),
                normalize_images=False,
                share_features_extractor=True,
            ),
        },
    )
    policy = model.policy
    policy.eval()

    wrapper = FreeriderActorWrapper(policy)
    wrapper.eval()

    # Sanity-check forward pass
    with torch.no_grad():
        dummy_image  = torch.zeros(1, 3, 144, 256)
        dummy_state  = torch.zeros(1, 1)
        dummy_action = wrapper(dummy_image, dummy_state)
    print(f'[export] Sanity check passed — output shape: {dummy_action.shape}')

    # ------------------------------------------------------------------
    # ONNX export
    # ------------------------------------------------------------------
    onnx_path = os.path.join(args.out_dir, 'freerider_actor.onnx')
    print(f'[export] Tracing to ONNX → {onnx_path}')
    torch.onnx.export(
        wrapper,
        (dummy_image, dummy_state),
        onnx_path,
        input_names=['image', 'state'],
        output_names=['action'],
        dynamic_axes={
            'image':  {0: 'batch'},
            'state':  {0: 'batch'},
            'action': {0: 'batch'},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print('[export] ONNX export complete.')

    # ------------------------------------------------------------------
    # TensorRT FP16 conversion
    # ------------------------------------------------------------------
    trt_path = os.path.join(args.out_dir, 'freerider_actor.trt')
    print(f'[export] Converting to TensorRT FP16 → {trt_path}')
    cmd = [
        'trtexec',
        f'--onnx={onnx_path}',
        f'--saveEngine={trt_path}',
        '--fp16',
        '--shapes=image:1x3x144x256,state:1x1',
    ]
    print('[export] Running:', ' '.join(cmd))
    subprocess.check_call(cmd)
    print(f'[export] Done. TensorRT engine saved to: {trt_path}')


if __name__ == '__main__':
    main()
