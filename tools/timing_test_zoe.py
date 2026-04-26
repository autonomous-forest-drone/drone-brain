#!/usr/bin/env python3
"""
Same harness as timing_test_midas.py but the depth stage runs ZoeDepth (ZoeD_NK)
instead of MiDaS_small. Use this to compare ZoeDepth's per-frame cost on the
Jetson against MiDaS, sim-to-real-style.

Notes:
  * ZoeD_NK is BEiT-Large 384 underneath — much heavier than MiDaS_small.
    On a Jetson Orin in PyTorch FP32 expect several hundred ms per frame.
  * ZoeDepth returns metric depth (meters, closer = smaller). MiDaS returns
    inverse depth (closer = larger). The min-max normalization here flattens
    the two outputs to the same [0,1] shape so the existing avoidance_policy
    engine still ingests something — but the polarity is OPPOSITE to what the
    policy was trained on. Treat the policy stage here as timing only; the
    actions are not meaningful unless the policy is retrained on ZoeDepth.
  * Needs working torchvision + timm (ZoeDepth import chain). On the JetPack
    torch 2.5 build, torchvision is currently broken — fix that first.
"""

import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

# Same cuDNN workaround as run_tree_avoid.py / timing_test_midas.py.
torch.backends.cudnn.enabled = False


ENGINE_PATH = "/home/beetlesniffer/PythonProjects/DANI/models/avoidance_policy.trt"
ZOE_TRT     = "/home/beetlesniffer/PythonProjects/DANI/models/zoed_nk.trt"

DEPTH_SIZE = 192   # policy input — matches the deployed avoidance_policy.trt
ZOE_INPUT  = 384   # ZoeD_NK / BEiT-L native long side
N_ITERS    = 100
N_WARMUP   = 5


class Timer:
    def __init__(self):
        self.data = defaultdict(list)

    def tick(self, name, t0):
        self.data[name].append((time.perf_counter() - t0) * 1000)

    def report(self):
        print("\n--- TIMING RESULTS (ms) ---")
        order = ["capture", "preprocess", "depth", "policy", "decision", "total"]
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
        downstream = [
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


# ---- ZoeDepth (PyTorch fallback) -------------------------------------------

class PyTorchZoe:
    """Slow path. ZoeD_NK via torch.hub. Used if the TRT engine isn't built."""
    def __init__(self):
        print("[zoe] loading ZoeD_NK via torch.hub (first time will download)")
        # trust_repo=True silences the y/n prompt on first fetch
        self.model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK",
                                    pretrained=True, trust_repo=True)
        self.model.eval().cuda()
        print("[zoe] ready (PyTorch FP32)")

    def infer(self, rgb):
        # rgb is uint8 HxWx3 (RGB) at ZOE_INPUT — see bgr_to_rgb_small.
        x = torch.from_numpy(rgb).float().div_(255.0)        # (H,W,3) in [0,1]
        x = x.permute(2, 0, 1).unsqueeze(0).cuda()           # (1,3,H,W)
        with torch.no_grad():
            # with_flip_aug=False: skip the test-time horizontal-flip aug (2x speed)
            # pad_input=False: don't pad to multiple of 32 — we already gave it 384
            pred = self.model.infer(x, with_flip_aug=False, pad_input=False)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        pred = F.interpolate(pred, (DEPTH_SIZE, DEPTH_SIZE),
                             mode="bilinear", align_corners=False)
        pred = pred.squeeze(1)
        mn, mx = pred.min(), pred.max()
        depth = (pred - mn) / (mx - mn + 1e-8)
        torch.cuda.synchronize()
        return depth.cpu().numpy().astype(np.float32)        # (1, H, W)


# ---- ZoeDepth (TRT FP16) ----------------------------------------------------

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


class TRTZoe:
    """ZoeD_NK as a FP16 TensorRT engine (separate export script).

    Assumes the engine was exported expecting ImageNet-normalized NCHW float32
    input at ZOE_INPUT × ZOE_INPUT. If you bake normalization into the export,
    drop the (x - mean)/std line below.
    """
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        print(f"[trt] loading Zoe engine: {engine_path}")
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load TRT engine: {engine_path}")

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
            self.buffers[name] = {"h": host, "d": device, "mode": mode, "shape": tuple(shape)}
            print(f"[trt] zoe {mode.name.lower()} '{name}' shape={tuple(shape)}")

    def infer(self, rgb):
        x = cv2.resize(rgb, (ZOE_INPUT, ZOE_INPUT), interpolation=cv2.INTER_LINEAR)
        x = x.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None]
        x = (x - _IMAGENET_MEAN) / _IMAGENET_STD

        out_shape = None
        for name, buf in self.buffers.items():
            if buf["mode"] == trt.TensorIOMode.INPUT:
                np.copyto(buf["h"], x.ravel())
                cuda.memcpy_htod_async(buf["d"], buf["h"], self.stream)
            else:
                out_shape = buf["shape"]
            self.context.set_tensor_address(name, int(buf["d"]))

        self.context.execute_async_v3(self.stream.handle)

        for name, buf in self.buffers.items():
            if buf["mode"] == trt.TensorIOMode.OUTPUT:
                cuda.memcpy_dtoh_async(buf["h"], buf["d"], self.stream)
        self.stream.synchronize()

        for name, buf in self.buffers.items():
            if buf["mode"] == trt.TensorIOMode.OUTPUT:
                raw = buf["h"].reshape(out_shape).squeeze()   # → (H, W)
                break

        if raw.shape != (DEPTH_SIZE, DEPTH_SIZE):
            raw = cv2.resize(raw, (DEPTH_SIZE, DEPTH_SIZE), interpolation=cv2.INTER_LINEAR)
        mn, mx = raw.min(), raw.max()
        depth = (raw - mn) / (mx - mn + 1e-8)
        return depth.astype(np.float32)[None]   # (1, H, W)


# ---- camera -----------------------------------------------------------------

CAM_W, CAM_H, CAM_FPS = 640, 480, 60


def open_camera():
    # GStreamer first — on Jetson the v4l2 path silently ignores cap.set for
    # CSI cams (IMX477 stays at native 4032x3040 no matter what you ask).

    gst_csi = (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={CAM_W}, height={CAM_H}, framerate={CAM_FPS}/1 ! "
        f"nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink drop=1 max-buffers=1"
    )
    cap = cv2.VideoCapture(gst_csi, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print(f"[camera] GStreamer CSI (nvargus) at {CAM_W}x{CAM_H} @ {CAM_FPS} fps")
        return cap

    gst_v4l2 = (
        f"v4l2src device=/dev/video0 ! "
        f"video/x-raw, width={CAM_W}, height={CAM_H}, framerate={CAM_FPS}/1 ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink drop=1 max-buffers=1"
    )
    cap = cv2.VideoCapture(gst_v4l2, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print(f"[camera] GStreamer v4l2src at {CAM_W}x{CAM_H} @ {CAM_FPS} fps")
        return cap

    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        cap.set(cv2.CAP_PROP_FPS,          CAM_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[camera] /dev/video0 at {w}x{h} @ {fps:.1f} fps (requested {CAM_W}x{CAM_H}@{CAM_FPS})")
        if (w, h) != (CAM_W, CAM_H):
            print("         WARNING: driver ignored setters — preprocess will be slow")
        return cap

    raise RuntimeError("Camera could not be opened (CSI + v4l2 + /dev/video0 all failed)")


def bgr_to_rgb_small(frame):
    # Resize first (SIMD), cvtColor on the small output — same trick as the
    # MiDaS version. Just sized for ZoeDepth's 384.
    small = cv2.resize(frame, (ZOE_INPUT, ZOE_INPUT), interpolation=cv2.INTER_LINEAR)
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
    cap = open_camera()
    if os.path.exists(ZOE_TRT):
        zoe = TRTZoe(ZOE_TRT)
        print("[zoe] path = TRT FP16")
    else:
        print(f"[zoe] {ZOE_TRT} not found — using PyTorch fallback")
        print("       (export a TRT engine for real-time numbers)")
        zoe = PyTorchZoe()
    policy = TRTPolicy(ENGINE_PATH)
    timer = Timer()
    prev_action = 0.0

    print(f"[warmup] {N_WARMUP} iters...")
    shape_printed = False
    for _ in range(N_WARMUP):
        ret, frame = cap.read()
        if not ret:
            continue
        if not shape_printed:
            print(f"[camera] actual frame shape: {frame.shape}  dtype={frame.dtype}")
            shape_printed = True
        rgb = bgr_to_rgb_small(frame)
        depth = zoe.infer(rgb)
        _ = policy.infer(depth[np.newaxis])

    print(f"[run] {N_ITERS} iters...")
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
        depth = zoe.infer(rgb)
        timer.tick("depth", t0)

        t0 = time.perf_counter()
        action = policy.infer(depth[np.newaxis])   # (1,1,DEPTH_SIZE,DEPTH_SIZE)
        timer.tick("policy", t0)

        t0 = time.perf_counter()
        smoothed    = 0.7 * action + 0.3 * prev_action
        prev_action = smoothed
        forward     = 1.5 * (1.0 - abs(smoothed))
        lateral     = 1.5 * smoothed
        timer.tick("decision", t0)

        timer.tick("total", t_total)

        if i % 20 == 0:
            print(f"iter {i:3d}  action={action:+.3f}  fwd={forward:.2f}  lat={lateral:+.2f}")

    timer.report()
    cap.release()


if __name__ == "__main__":
    main()
