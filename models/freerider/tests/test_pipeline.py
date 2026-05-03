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

import torch
torch.cuda.init()  # create PyTorch primary CUDA context first
# cuDNN conflicts with pycuda/TRT on Jetson — disable it so PyTorch uses
# its own CUDA kernels instead. Still runs on GPU, just without cuDNN.
torch.backends.cudnn.enabled = False

import atexit
import cv2
import numpy as np
import pycuda.driver as cuda

# Attach pycuda to the same primary context that PyTorch already created.
cuda.init()
_pycuda_ctx = cuda.Device(torch.cuda.current_device()).retain_primary_context()
_pycuda_ctx.push()
atexit.register(_pycuda_ctx.pop)  # keep the context stack clean on exit

import tensorrt as trt
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

DEPTH_ENGINE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'model',
    'depth_anything_v2_small_fp16.trt'
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
# Depth estimator — HuggingFace fallback (slow, ~550 ms/step without cuDNN)
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
# Depth estimator — TensorRT fast path (~30–80 ms/step target)
# ---------------------------------------------------------------------------

class DepthEstimatorTRT:
    """Runs DepthAnything V2 Small via a pre-built TRT engine.

    Preprocessing (resize + normalise) still runs on CPU via AutoImageProcessor
    (< 2 ms); only the forward pass goes through TRT on the GPU.
    """

    def __init__(self, engine_path: str):
        # Read normalisation stats from the processor config — don't use the
        # processor's resize logic (it ignores our target resolution).
        print(f'[DepthEstimatorTRT] Loading processor from {DEPTH_MODEL_ID} ...')
        _proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
        self._mean = np.array(_proc.image_mean, dtype=np.float32)  # [0.485, 0.456, 0.406]
        self._std  = np.array(_proc.image_std,  dtype=np.float32)  # [0.229, 0.224, 0.225]

        print(f'[DepthEstimatorTRT] Loading TRT engine: {engine_path}')
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
            self._bindings[name] = {
                'shape': shape, 'dtype': dtype, 'host': host, 'dev': dev
            }
            self._context.set_tensor_address(name, int(dev))

        _, _, self._in_h, self._in_w = self._bindings['pixel_values']['shape']
        print(f'[DepthEstimatorTRT] Input: {self._in_h}×{self._in_w}  Ready.')

    def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
        """Resize to engine input size and normalise → float32 NCHW."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self._in_w, self._in_h),
                         interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        rgb = (rgb - self._mean) / self._std          # HWC, float32
        return rgb.transpose(2, 0, 1)[np.newaxis]     # → NCHW [1,3,H,W]

    def estimate(self, bgr: np.ndarray) -> np.ndarray:
        pixel_values = self._preprocess(bgr)  # [1, 3, in_h, in_w]

        pv  = self._bindings['pixel_values']
        out = self._bindings['predicted_depth']

        np.copyto(pv['host'], pixel_values.reshape(pv['shape']).astype(pv['dtype']))
        cuda.memcpy_htod_async(pv['dev'],  pv['host'],  self._stream)
        self._context.execute_async_v3(self._stream.handle)
        cuda.memcpy_dtoh_async(out['host'], out['dev'], self._stream)
        self._stream.synchronize()

        depth = out['host'].squeeze().copy()
        depth = cv2.resize(depth.astype(np.float32), (IMG_W, IMG_H),
                           interpolation=cv2.INTER_LINEAR)
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
# HUD composite frame
# ---------------------------------------------------------------------------

# Display size for each panel in the HUD
_DISP_W, _DISP_H = 640, 360
_BAR_H            = 70   # height of the action strip below the panels


def _draw_hud(bgr: np.ndarray,
              depth: np.ndarray,        # float32 normalised 0-1, shape (H, W)
              raw_action: float,
              smoothed: float,
              fwd: float,
              lat: float,
              step: int,
              latency_ms: float) -> np.ndarray:
    """Build a 1280×(360+70) BGR composite frame for display / combined video."""
    W, H, BH = _DISP_W, _DISP_H, _BAR_H

    # --- left panel: camera ------------------------------------------------
    cam = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR)

    # Steering arrow: from bottom-centre, deflected left/right by smoothed
    cx = W // 2
    tip_x = cx + int(smoothed * W * 0.40)
    tip_y = H // 3
    base_y = H - 20
    mag = abs(smoothed)
    if mag < 0.3:
        arrow_color = (0, 220, 0)       # green
    elif mag < 0.6:
        arrow_color = (0, 200, 220)     # yellow
    else:
        arrow_color = (0, 60, 230)      # red
    cv2.arrowedLine(cam, (cx, base_y), (tip_x, tip_y),
                    arrow_color, 3, tipLength=0.25)

    # Step counter
    cv2.putText(cam, f'step {step:06d}', (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

    # --- right panel: depth (inferno colormap) -----------------------------
    depth_u8  = (depth * 255).astype(np.uint8)
    depth_col = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
    depth_pan = cv2.resize(depth_col, (W, H), interpolation=cv2.INTER_LINEAR)
    cv2.putText(depth_pan, 'depth', (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

    # --- bottom strip: action bar + telemetry text -------------------------
    bar = np.zeros((BH, W * 2, 3), dtype=np.uint8)

    # Background
    bar[:] = (30, 30, 30)

    # Horizontal action bar centred on the strip
    bar_cx  = W              # centre of the full-width strip
    bar_y0, bar_y1 = 12, 38
    bar_max_half = W - 40    # max pixel extent each side

    # Grey track
    cv2.rectangle(bar, (bar_cx - bar_max_half, bar_y0),
                       (bar_cx + bar_max_half, bar_y1), (70, 70, 70), -1)
    # Smoothed action fill
    fill_w = int(abs(smoothed) * bar_max_half)
    fill_color = (50, 180, 50) if smoothed >= 0 else (50, 50, 200)
    if smoothed >= 0:
        cv2.rectangle(bar, (bar_cx, bar_y0),
                           (bar_cx + fill_w, bar_y1), fill_color, -1)
    else:
        cv2.rectangle(bar, (bar_cx - fill_w, bar_y0),
                           (bar_cx, bar_y1), fill_color, -1)
    # Centre tick
    cv2.line(bar, (bar_cx, bar_y0 - 2), (bar_cx, bar_y1 + 2), (200, 200, 200), 2)

    # Telemetry text
    txt = (f'raw={raw_action:+.3f}  smooth={smoothed:+.3f}  '
           f'fwd={fwd:.3f}  lat={lat:+.3f}  {latency_ms:.0f}ms')
    cv2.putText(bar, txt, (10, BH - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

    # --- assemble ----------------------------------------------------------
    top  = np.hstack([cam, depth_pan])
    return np.vstack([top, bar])


# ---------------------------------------------------------------------------
# Video export
# ---------------------------------------------------------------------------

def _make_video(flight_dir: str, fps: int = SETPOINT_HZ):
    frames_dir = os.path.join(flight_dir, 'frames')
    depth_dir  = os.path.join(flight_dir, 'depth')

    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith('.jpg'))
    if not frame_files:
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    sample = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    fh, fw = sample.shape[:2]

    depth_sample = cv2.imread(os.path.join(depth_dir, frame_files[0]))
    dh, dw = depth_sample.shape[:2] if depth_sample is not None else (fh, fw)

    out_frames = os.path.join(flight_dir, 'frames.mp4')
    out_depth  = os.path.join(flight_dir, 'depth.mp4')
    vw_frames  = cv2.VideoWriter(out_frames, fourcc, fps, (fw, fh))
    vw_depth   = cv2.VideoWriter(out_depth,  fourcc, fps, (dw, dh))

    for fname in frame_files:
        bgr   = cv2.imread(os.path.join(frames_dir, fname))
        depth = cv2.imread(os.path.join(depth_dir,  fname))
        if bgr is not None:
            vw_frames.write(bgr)
        if depth is not None:
            vw_depth.write(depth)

    vw_frames.release()
    vw_depth.release()
    print(f'[video] saved → {out_frames}')
    print(f'[video] saved → {out_depth}')


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

    if os.path.exists(DEPTH_ENGINE_PATH):
        print(f'Depth TRT  : {DEPTH_ENGINE_PATH}')
        depth_est = DepthEstimatorTRT(DEPTH_ENGINE_PATH)
    else:
        print(f'Depth TRT  : not found — using HuggingFace model (~550 ms/step)')
        print(f'             Run export_depth_trt.py to build the engine.')
        depth_est = DepthEstimator(device=device)

    trt_engine = TRTEngine(args.engine)
    print(f'[TRT] Engine loaded: {args.engine}')

    log_path   = os.path.join(flight_dir, 'flight.csv')
    log_file   = open(log_path, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        't', 'raw_action', 'smoothed_action',
        'forward_vel', 'lateral_vel', 'step_latency_ms',
    ])

    # Combined HUD video writer (always on)
    hud_path   = os.path.join(flight_dir, 'combined.mp4')
    hud_writer = cv2.VideoWriter(
        hud_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        SETPOINT_HZ,
        (_DISP_W * 2, _DISP_H + _BAR_H),
    )

    frame_stack       = deque(maxlen=N_FRAMES)
    smoothed          = 0.0
    accumulated_state = 0.0   # running sum of smoothed actions — state input to actor
    step_count        = 0
    latencies         = []
    t0               = time.monotonic()
    last_beep        = t0

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

                image_np          = np.stack(list(frame_stack), axis=0)
                state_np          = np.array([accumulated_state], dtype=np.float32)
                raw_action        = float(np.clip(trt_engine.infer(image_np, state_np), -1.0, 1.0))
                smoothed          = SMOOTH_ALPHA * raw_action + ACTION_MOMENTUM * smoothed
                accumulated_state += smoothed

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

                now = time.monotonic()
                if now - last_beep >= 2.0:
                    subprocess.Popen(
                        ['ros2', 'topic', 'pub', '--once',
                         '/mavros/play_tune',
                         'mavros_msgs/msg/PlayTuneV2',
                         '{format: 1, tune: "MFT120L4 O6 CCC"}'],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    last_beep = now

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

                hud = _draw_hud(bgr, depth, raw_action, smoothed,
                                fwd, lat, step_count, step_latency_ms)
                hud_writer.write(hud)

                elapsed   = time.monotonic() - t_step
                sleep_for = (1.0 / SETPOINT_HZ) - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)

        except KeyboardInterrupt:
            print(f'\nStopped. {step_count} steps recorded.')
        finally:
            log_file.close()
            hud_writer.release()

    if latencies:
        print(
            f'Latency — avg: {sum(latencies)/len(latencies):.1f} ms  '
            f'min: {min(latencies):.1f} ms  '
            f'max: {max(latencies):.1f} ms'
        )

    _plot_log(flight_dir)
    _make_video(flight_dir)
    _sync_dropbox(flight_dir, stamp)


if __name__ == '__main__':
    main()
