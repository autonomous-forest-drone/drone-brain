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
import sys

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR     = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'model'))
V5_SRC_DEFAULT = os.path.abspath(
    os.path.join(SCRIPT_DIR, '..', '..', '..', 'ppo-cnn-drone-navigation', 'src', 'v5')
)


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
        self.features_extractor = policy.features_extractor   # ActorFeaturesExtractor
        self.policy_net         = policy.mlp_extractor.policy_net
        self.action_net         = policy.action_net

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        obs      = {'image': image, 'state': state}
        features = self.features_extractor(obs)   # (B, 257)
        latent   = self.policy_net(features)      # (B, 64)
        return self.action_net(latent)            # (B, 1)


def main():
    parser = argparse.ArgumentParser(description='Export Freerider actor → ONNX → TensorRT FP16')
    parser.add_argument('--v5_src',  default=V5_SRC_DEFAULT,
                        help='Path to ppo-cnn-drone-navigation/src/v5 (for custom class imports)')
    parser.add_argument('--out_dir', default=MODEL_DIR,
                        help='Output directory for ONNX and TRT files (default: freerider/model/)')
    args = parser.parse_args()

    # Add v5 source to path so SB3 can resolve custom classes on load
    sys.path.insert(0, os.path.abspath(args.v5_src))
    from stable_baselines3 import PPO  # noqa: E402

    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Locate checkpoint — must be manually placed in freerider/model/
    # ------------------------------------------------------------------
    ckpt_path = find_checkpoint_in_model_dir()
    print(f'[export] Loading checkpoint: {ckpt_path}')

    # ------------------------------------------------------------------
    # Load SB3 policy on CPU (tracing does not require CUDA)
    # ------------------------------------------------------------------
    model  = PPO.load(ckpt_path, device='cpu')
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
