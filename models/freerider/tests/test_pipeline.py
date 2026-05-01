#!/usr/bin/env python3
"""
test_pipeline.py — Ground test of the Freerider perception + inference pipeline.

Runs the full camera → depth → TRT policy loop on the ground without arming,
taking off, or flying.  Identical outputs to a real flight run:

    <log_dir>/frames/   — raw BGR frames (every step)
    <log_dir>/depth/    — depth maps as 8-bit greyscale JPEGs
    <log_dir>/flight.csv — t, raw_action, smoothed_action, forward_vel,
                           lateral_vel, step_latency_ms
    <log_dir>/flight.png — plots of the above

On exit syncs to Dropbox under dropbox:images/freerider_test_pipeline_<stamp>/

Usage:
    python models/freerider/tests/test_pipeline.py
    python models/freerider/tests/test_pipeline.py --engine path/to/engine.trt

Press Ctrl+C to stop.
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt
import torch
from PIL import Image as PILImage
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from tools.camera_imx219 import Camera

# ---------------------------------------------------------------------------
# Constants — kept identical to run_freerider.py
# ---------------------------------------------------------------------------
SETPOINT_HZ     = 10
FLIGHT_LOG_ROOT = os.path.expanduser('~/drone-brain/images')
DEPTH_MODEL_ID  = 'depth-anything/Depth-Anything-V2-Small-hf'
IMG_H, IMG_W    = 144, 256
N_FRAMES        = 3
FIXED_SPEED     = 1.0
MAX_LATERAL     = 0.8
ACTION_MOMENTUM = 0.3
SMOOTH_ALPHA    = 1.0 - ACTION_MOMENTUM   # 0.7

ENGINE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'model', 'freerider_actor.trt'
)


# ---------------------------------------------------------------------------
# TensorRT inference wrapper (identical to run_freerider.py)
# ---------------------------------------------------------------------------

class TRTEngine:

    def __init__(self, engine_path: str):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime      = trt.Runtime(logger)
            self._engine = runtime.deserialize_cuda_engine(f.read())
        self._context = self._engine.create_execution_context()
        self._stream  = cuda.Stream()

        self._bindings = {}
        for i in range(self._engine.num_io_tensors):
            name  = self._engine.get_tensor_name(i)
            shape = tuple(self._engine.get_tensor_shape(name))
            dtype = trt.nptype(self._engine.get_tensor_dtype(name))
            host  = cuda.pagelocked_empty(shape, dtype)
            dev   = cuda.mem_alloc(host.nbytes)
            self._bindings[name] = {'shape': shape, 'dtype': dtype, 'host': host, 'dev': dev}
            self._context.set_tensor_address(name, int(dev))

    def infer(self, image_np: np.ndarray, state_np: np.ndarray) -> float:
        img = self._bindings['image']
        st  = self._bindings['state']
        act = self._bindings['action']

        np.copyto(img['host'], image_np.reshape(img['shape']).astype(img['dtype']))
        np.copyto(st['host'],  state_np.reshape(st['shape']).astype(st['dtype']))

        cuda.memcpy_htod_async(img['dev'], img['host'], self._stream)
        cuda.memcpy_htod_async(st['dev'],  st['host'],  self._stream)
        self._context.execute_async_v3(self._stream.handle)
        cuda.memcpy_dtoh_async(act['host'], act['dev'], self._stream)
        self._stream.synchronize()

        return float(act['host'].flat[0])


# ---------------------------------------------------------------------------
# Depth estimator (identical to run_freerider.py)
# ---------------------------------------------------------------------------

class DepthEstimator:

    def __init__(self, device: str = 'cuda'):
        print(f'[DepthEstimator] Loading {DEPTH_MODEL_ID} on {device} ...')
        self._processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
        self._model     = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID)
        self._model.to(device).eval()
        self._device    = device
        print('[DepthEstimator] Ready.')

    def estimate(self, bgr: np.ndarray) -> np.ndarray:
        rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil    = PILImage.fromarray(rgb)
        inputs = self._processor(images=pil, return_tensors='pt')
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            raw = self._model(**inputs).predicted_depth
        depth = raw.squeeze().cpu().numpy()
        depth = cv2.resize(depth, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
        d_min, d_max = float(depth.min()), float(depth.max())
        if d_max > d_min:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)
        return depth.astype(np.float32)


# ---------------------------------------------------------------------------
# Post-run plot (identical to run_freerider.py)
# ---------------------------------------------------------------------------

def _plot_log(flight_dir: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    csv_path = os.path.join(flight_dir, 'flight.csv')
    if not os.path.exists(csv_path):
        return

    t, raw, smoothed, fwd, lat, latency = [], [], [], [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            t.append(float(row['t']))
            raw.append(float(row['raw_action']))
            smoothed.append(float(row['smoothed_action']))
            fwd.append(float(row['forward_vel']))
            lat.append(float(row['lateral_vel']))
            latency.append(float(row['step_latency_ms']))

    if not t:
        return

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(os.path.basename(flight_dir))

    axes[0].plot(t, raw,      color='steelblue', label='raw action')
    axes[0].axhline(0, color='gray', linewidth=0.5)
    axes[0].set_ylabel('Raw action')
    axes[0].set_ylim(-1.1, 1.1)
    axes[0].legend()

    axes[1].plot(t, smoothed, color='orange', label='smoothed action')
    axes[1].axhline(0, color='gray', linewidth=0.5)
    axes[1].set_ylabel('Smoothed action')
    axes[1].set_ylim(-1.1, 1.1)
    axes[1].legend()

    axes[2].plot(t, fwd, color='green', label='forward (m/s)')
    axes[2].plot(t, lat, color='red',   label='lateral (m/s)')
    axes[2].axhline(0, color='gray', linewidth=0.5)
    axes[2].set_ylabel('Velocity (m/s)')
    axes[2].legend()

    axes[3].plot(t, latency, color='purple', label='step latency (ms)')
    axes[3].set_ylabel('Latency (ms)')
    axes[3].set_xlabel('Time (s)')
    axes[3].legend()

    plt.tight_layout()
    out = os.path.join(flight_dir, 'flight.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'[plot] saved → {out}')


# ---------------------------------------------------------------------------
# Dropbox sync
# ---------------------------------------------------------------------------

def _sync_dropbox(flight_dir: str, stamp: str):
    dest = f'dropbox:images/freerider_test_pipeline_{stamp}'
    print(f'[dropbox] Syncing to {dest} ...')
    try:
        subprocess.run(['rclone', 'copy', flight_dir, dest], timeout=120)
        print('[dropbox] Sync complete.')
    except subprocess.TimeoutExpired:
        print('[dropbox] Sync timed out after 120s.')
    except Exception as e:
        print(f'[dropbox] Sync failed: {e}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Ground test of the Freerider perception + inference pipeline.'
    )
    parser.add_argument(
        '--engine',
        default=ENGINE_PATH,
        help='Path to TensorRT engine (.trt)',
    )
    args = parser.parse_args()

    stamp      = time.strftime('%Y-%m-%d_%H-%M-%S')
    flight_dir = os.path.join(FLIGHT_LOG_ROOT, f'freerider_test_pipeline_{stamp}')
    frames_dir = os.path.join(flight_dir, 'frames')
    depth_dir  = os.path.join(flight_dir, 'depth')
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(depth_dir,  exist_ok=True)

    print('=' * 60)
    print('Freerider — Ground Pipeline Test')
    print(f'Engine     : {args.engine}')
    print(f'Log dir    : {flight_dir}')
    print(f'Rate       : {SETPOINT_HZ} Hz')
    print('=' * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device     : {device}')

    depth_est  = DepthEstimator(device=device)
    trt_engine = TRTEngine(args.engine)
    print(f'[TRT] Engine loaded: {args.engine}')

    log_path   = os.path.join(flight_dir, 'flight.csv')
    log_file   = open(log_path, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        't', 'raw_action', 'smoothed_action',
        'forward_vel', 'lateral_vel', 'step_latency_ms',
    ])

    frame_stack = deque(maxlen=N_FRAMES)
    smoothed    = 0.0
    step_count  = 0
    latencies   = []
    t0          = time.monotonic()

    with Camera() as cam:
        print(f'\nCapturing. Press Ctrl+C to stop.\n')
        try:
            while True:
                t_step = time.monotonic()

                bgr = cam.capture()
                if bgr is None:
                    continue

                depth = depth_est.estimate(bgr)
                frame_stack.append(depth)
                while len(frame_stack) < N_FRAMES:
                    frame_stack.appendleft(frame_stack[0])

                image_np   = np.stack(list(frame_stack), axis=0)
                state_np   = np.array([smoothed], dtype=np.float32)
                raw_action = float(np.clip(trt_engine.infer(image_np, state_np), -1.0, 1.0))
                smoothed   = SMOOTH_ALPHA * raw_action + ACTION_MOMENTUM * smoothed

                fwd = FIXED_SPEED - MAX_LATERAL * abs(smoothed)
                lat = MAX_LATERAL * smoothed

                step_latency_ms = (time.monotonic() - t_step) * 1000.0
                t = time.monotonic() - t0
                latencies.append(step_latency_ms)

                log_writer.writerow([
                    f'{t:.3f}', f'{raw_action:.4f}', f'{smoothed:.4f}',
                    f'{fwd:.4f}', f'{lat:.4f}', f'{step_latency_ms:.1f}',
                ])
                log_file.flush()

                step_count += 1
                stamp_str  = f'{step_count:06d}'
                cv2.imwrite(os.path.join(frames_dir, f'{stamp_str}.jpg'), bgr)
                cv2.imwrite(
                    os.path.join(depth_dir, f'{stamp_str}.jpg'),
                    (depth * 255).astype(np.uint8),
                )

                print(
                    f'  [{stamp_str}]  raw={raw_action:+.3f}  '
                    f'smoothed={smoothed:+.3f}  '
                    f'fwd={fwd:.3f}  lat={lat:+.3f}  '
                    f'{step_latency_ms:.0f}ms'
                )

                elapsed   = time.monotonic() - t_step
                sleep_for = (1.0 / SETPOINT_HZ) - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)

        except KeyboardInterrupt:
            print(f'\nStopped. {step_count} steps recorded.')
        finally:
            log_file.close()

    if latencies:
        print(
            f'Latency — avg: {sum(latencies)/len(latencies):.1f} ms  '
            f'min: {min(latencies):.1f} ms  '
            f'max: {max(latencies):.1f} ms'
        )

    _plot_log(flight_dir)
    _sync_dropbox(flight_dir, stamp)


if __name__ == '__main__':
    main()
