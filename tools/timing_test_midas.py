#!/usr/bin/env python3
"""
Times each stage of the tree-avoidance perception loop on real hardware,
no MAVROS / arming / takeoff. Use this to calibrate sim-to-real: the sim
step is one forward pass, but the real loop also pays capture + MiDaS +
TRT policy + decision.
"""

import argparse
import os
import select
import sys
import time
from collections import defaultdict, deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

from jetson_camera import DEFAULT_FOCUS, GstJpegCapture, set_focus
from midas_trt import TRTMidas

# MiDaS blows up with CUDNN_STATUS_NOT_INITIALIZED on this Jetson unless cuDNN
# is off — same workaround as run_tree_avoid.py.
torch.backends.cudnn.enabled = False


ENGINE_PATH   = "/home/beetlesniffer/drone-brain/models/fortune_cookie/model/jetson_converted_model.trt"
MIDAS_TRT     = "/home/beetlesniffer/drone-brain/models/fortune_cookie/model/midas_small.trt"
MIDAS_WEIGHTS = "/home/beetlesniffer/.cache/torch/hub/checkpoints/midas_v21_small_256.pt"
MIDAS_REPO    = "/home/beetlesniffer/.cache/torch/hub/intel-isl_MiDaS_master"

# Policy expects (B, 3, 126, 224) — 3 stacked MiDaS depth frames at 126x224.
# Buffer size and stride match training: idxs[-11], [-6], [-1] → oldest → newest,
# spanning 1 s of motion at 10 Hz.
DEPTH_H        = 126
DEPTH_W        = 224
DEPTH_STACK_N  = 3
DEPTH_STRIDE   = 5
DEPTH_BUFFER_N = (DEPTH_STACK_N - 1) * DEPTH_STRIDE + 1   # 11
MIDAS_INPUT    = 256   # MiDaS was trained/exported at 256
N_ITERS        = 100
N_WARMUP       = 5
SNAPSHOT_DIR   = "/home/beetlesniffer/drone-brain/tools/snapshots"


def stdin_pressed():
    # non-blocking: True if a line is waiting on stdin (user hit Enter)
    if not sys.stdin.isatty():
        return False
    r, _, _ = select.select([sys.stdin], [], [], 0)
    if not r:
        return False
    sys.stdin.readline()
    return True


def save_snapshot(frame_bgr, rgb_small, depth2d):
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    raw_path   = os.path.join(SNAPSHOT_DIR, f"{ts}_frame.png")
    depth_path = os.path.join(SNAPSHOT_DIR, f"{ts}_depth.png")
    side_path  = os.path.join(SNAPSHOT_DIR, f"{ts}_side.png")

    cv2.imwrite(raw_path, frame_bgr)

    d8 = (np.clip(depth2d, 0.0, 1.0) * 255).astype(np.uint8)
    d_color = cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)
    cv2.imwrite(depth_path, d_color)

    h = frame_bgr.shape[0]
    d_resized = cv2.resize(d_color, (int(d_color.shape[1] * h / d_color.shape[0]), h))
    cv2.imwrite(side_path, np.hstack([frame_bgr, d_resized]))
    print(f"[snapshot] {raw_path}  +  {depth_path}  +  {side_path}")


class Timer:
    def __init__(self):
        self.data = defaultdict(list)

    def tick(self, name, t0):
        self.data[name].append((time.perf_counter() - t0) * 1000)

    def report(self):
        print("\n--- TIMING RESULTS (ms) ---")
        order = ["capture", "preprocess", "depth", "stack", "policy", "decision", "total"]
        for k in order:
            v = self.data.get(k)
            if not v:
                continue
            arr = np.array(v)
            print(
                f"{k:12s} n={len(arr):3d}  mean={arr.mean():6.2f}  "
                f"median={np.median(arr):6.2f}  p95={np.percentile(arr,95):6.2f}  "
                f"max={arr.max():6.2f}"
            )
        tot = self.data.get("total")
        if tot:
            print(f"\neffective rate: {1000.0 / np.mean(tot):.2f} Hz")

        # -------- estimated, NOT measured --------
        # These happen downstream of _publish_vel and can't be observed from the Jetson
        # alone. Numbers are rough priors — refine once you have bench data.
        downstream = [
            # (stage,                              low,  high, note)
            ("sensor + rolling shutter",            5,   15,  "photon → pixel ready on IMX477 / UVC cam"),
            ("MAVROS publish + DDS",                1,    2,  "rclpy → local DDS"),
            ("UART to PX4 @115200",                 4,    6,  "~58-byte MAVLink SET_POSITION_TARGET_LOCAL_NED"),
            ("PX4 pos-ctrl schedule (50 Hz)",       0,   20,  "avg ~10 — wait for next controller tick"),
            ("PX4 att-ctrl + mixer",                2,    5,  "250 Hz inner loop + mixer"),
            ("ESC + motor spin-up",                30,   60,  "to ~63% of commanded thrust; prop/battery dependent"),
        ]
        print("\n--- ESTIMATED DOWNSTREAM (not measured, ms) ---")
        lo_sum, hi_sum = 0, 0
        for name, lo, hi, note in downstream:
            print(f"{name:32s} {lo:3d}–{hi:3d}   {note}")
            lo_sum += lo
            hi_sum += hi
        print("-" * 72)
        print(f"{'downstream total':32s} {lo_sum:3d}–{hi_sum:3d}")

        if tot:
            meas = np.mean(tot)
            print(f"\n>> observation → motor response (est.): "
                  f"{meas + lo_sum:.0f}–{meas + hi_sum:.0f} ms "
                  f"(measured {meas:.0f} + downstream {lo_sum}–{hi_sum})")


# ---- MiDaS ------------------------------------------------------------------

def load_midas():
    if MIDAS_REPO not in sys.path:
        sys.path.insert(0, MIDAS_REPO)
    from midas.midas_net_custom import MidasNet_small
    from midas.transforms import Resize, NormalizeImage, PrepareForNet

    model = MidasNet_small(
        None, features=64, backbone="efficientnet_lite3",
        exportable=True, non_negative=True, blocks={"expand": True},
    )
    model.load_state_dict(torch.load(MIDAS_WEIGHTS, map_location="cpu", weights_only=False))
    model.eval().cuda()

    def transform(img):
        sample = {"image": img / 255.0}
        sample = Resize(256, 256, resize_target=None, keep_aspect_ratio=True,
                        ensure_multiple_of=32, resize_method="upper_bound",
                        image_interpolation_method=cv2.INTER_CUBIC)(sample)
        sample = NormalizeImage(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])(sample)
        sample = PrepareForNet()(sample)
        return torch.from_numpy(sample["image"]).unsqueeze(0)

    print("[midas] ready")
    return model, transform


class PyTorchMidas:
    """Fallback path — the slow one. Used if the TRT engine isn't built yet."""
    def __init__(self):
        model, transform = load_midas()
        self.model     = model
        self.transform = transform

    def infer(self, rgb, out_size=(DEPTH_H, DEPTH_W)):
        inp = self.transform(rgb).cuda()
        with torch.no_grad():
            pred = self.model(inp)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        pred = F.interpolate(pred, out_size, mode="bilinear", align_corners=False)
        pred = pred.squeeze(1)
        mn, mx = pred.min(), pred.max()
        depth = (pred - mn) / (mx - mn + 1e-8)
        torch.cuda.synchronize()
        return depth.cpu().numpy().astype(np.float32)   # (1, H, W)


# ---- camera -----------------------------------------------------------------

CAM_W, CAM_H, CAM_FPS = 640, 480, 30


def open_camera(focus):
    cap = GstJpegCapture(CAM_W, CAM_H, CAM_FPS)
    print(f"[camera] gst-launch nvarguscamerasrc → multifilesink at {CAM_W}x{CAM_H} @ {CAM_FPS} fps")
    if focus is not None:
        set_focus(focus)
        time.sleep(0.3)  # VCM lens travel
        print(f"[camera] focus locked at {focus}")
    return cap


def bgr_to_rgb_small(frame):
    # resize straight to MiDaS input size so TRTMidas.infer's internal resize is a no-op.
    # resize first (SIMD), cvtColor on the tiny output (near-free) — stride-reversed numpy
    # slices are ~10x slower than cvtColor on big frames.
    small = cv2.resize(frame, (MIDAS_INPUT, MIDAS_INPUT), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)


# ---- TRT policy -------------------------------------------------------------

class TRTPolicy:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        print(f"[trt] loading engine: {engine_path}")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")

        self.context = self.engine.create_execution_context()
        self.stream  = cuda.Stream()
        self.buffers = {}

        for i in range(self.engine.num_io_tensors):
            name  = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            mode  = self.engine.get_tensor_mode(name)
            host   = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            device = cuda.mem_alloc(host.nbytes)
            self.buffers[name] = {"h": host, "d": device, "mode": mode}
            if mode == trt.TensorIOMode.INPUT:
                print(f"[trt] input '{name}' shape={tuple(shape)} dtype={dtype.__name__}")

    def infer(self, x):
        x = x.astype(np.float32, copy=False)
        for name, buf in self.buffers.items():
            if buf["mode"] == trt.TensorIOMode.INPUT:
                np.copyto(buf["h"], x.ravel())
                cuda.memcpy_htod_async(buf["d"], buf["h"], self.stream)
            self.context.set_tensor_address(name, int(buf["d"]))

        self.context.execute_async_v3(self.stream.handle)

        for name, buf in self.buffers.items():
            if buf["mode"] == trt.TensorIOMode.OUTPUT:
                cuda.memcpy_dtoh_async(buf["h"], buf["d"], self.stream)
        self.stream.synchronize()

        for name, buf in self.buffers.items():
            if buf["mode"] == trt.TensorIOMode.OUTPUT:
                return float(np.clip(buf["h"][0], -1.0, 1.0))
        return 0.0


# ---- main loop --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focus", type=int, default=DEFAULT_FOCUS,
                        help=f"VCM focus value 0-1000 (default {DEFAULT_FOCUS})")
    args = parser.parse_args()

    cap = open_camera(focus=args.focus)
    if os.path.exists(MIDAS_TRT):
        midas = TRTMidas(MIDAS_TRT)
        print("[midas] path = TRT FP16")
    else:
        print(f"[midas] {MIDAS_TRT} not found — using PyTorch fallback")
        print("        run `python3 export_midas_trt.py` to build the TRT engine")
        midas = PyTorchMidas()
    policy = TRTPolicy(ENGINE_PATH)
    timer = Timer()
    prev_action = 0.0

    # 11-frame ring buffer of squeezed (H, W) depth maps. Stack indices [-11, -6, -1]
    # → 3 frames, oldest first, 1 s of motion at 10 Hz — matches training-time obs.
    depth_buffer = deque(maxlen=DEPTH_BUFFER_N)
    stack_idxs = [-1 - i * DEPTH_STRIDE for i in range(DEPTH_STACK_N)][::-1]   # [-11, -6, -1]

    def push_and_stack(d2):
        depth_buffer.append(d2)
        return np.stack([depth_buffer[i] for i in stack_idxs], axis=0)

    # warmup — first MiDaS + TRT calls are always slow (lazy init, kernel compile)
    print(f"[warmup] {N_WARMUP} iters...")
    shape_printed = False
    for w in range(N_WARMUP):
        ret, frame = cap.read()
        if not ret:
            continue
        if not shape_printed:
            print(f"[camera] actual frame shape: {frame.shape}  dtype={frame.dtype}")
            shape_printed = True
        rgb = bgr_to_rgb_small(frame)
        depth = midas.infer(rgb, out_size=(DEPTH_H, DEPTH_W))
        d2 = depth[0]
        if w == 0:
            # mimic env.reset(): pre-fill the whole buffer with the first depth
            for _ in range(DEPTH_BUFFER_N):
                depth_buffer.append(d2)
        else:
            depth_buffer.append(d2)
        obs = np.stack([depth_buffer[i] for i in stack_idxs], axis=0)
        _ = policy.infer(obs[np.newaxis])

    print(f"[run] {N_ITERS} iters... (press Enter at any time for a snapshot)")
    for i in range(N_ITERS):
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        ret, frame = cap.read()
        timer.tick("capture", t0)
        if not ret:
            continue

        t0 = time.perf_counter()
        rgb = bgr_to_rgb_small(frame)
        timer.tick("preprocess", t0)

        t0 = time.perf_counter()
        depth = midas.infer(rgb, out_size=(DEPTH_H, DEPTH_W))
        timer.tick("depth", t0)

        t0 = time.perf_counter()
        obs = push_and_stack(depth[0])
        timer.tick("stack", t0)

        t0 = time.perf_counter()
        action = policy.infer(obs[np.newaxis])   # (1, 3, DEPTH_H, DEPTH_W)
        timer.tick("policy", t0)

        t0 = time.perf_counter()
        smoothed    = 0.7 * action + 0.3 * prev_action
        prev_action = smoothed
        forward     = 1.5 * (1.0 - abs(smoothed))
        lateral     = 1.5 * smoothed
        timer.tick("decision", t0)

        timer.tick("total", t_total)

        if stdin_pressed():
            save_snapshot(frame, rgb, depth[0])

        if i % 20 == 0:
            print(f"iter {i:3d}  action={action:+.3f}  fwd={forward:.2f}  lat={lateral:+.2f}")

    timer.report()
    cap.release()


if __name__ == "__main__":
    main()
