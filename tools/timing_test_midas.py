#!/usr/bin/env python3
"""
Times each stage of the tree-avoidance perception loop on real hardware,
no MAVROS / arming / takeoff. Use this to calibrate sim-to-real: the sim
step is one forward pass, but the real loop also pays capture + MiDaS +
TRT policy + decision.
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

# MiDaS blows up with CUDNN_STATUS_NOT_INITIALIZED on this Jetson unless cuDNN
# is off — same workaround as run_tree_avoid.py.
torch.backends.cudnn.enabled = False


ENGINE_PATH   = "/home/beetlesniffer/PythonProjects/DANI/models/avoidance_policy.trt"
MIDAS_TRT     = "/home/beetlesniffer/PythonProjects/DANI/models/midas_small.trt"
MIDAS_WEIGHTS = "/home/beetlesniffer/.cache/torch/hub/checkpoints/midas_v21_small_256.pt"
MIDAS_REPO    = "/home/beetlesniffer/.cache/torch/hub/intel-isl_MiDaS_master"

DEPTH_SIZE  = 192   # policy input size — matches the current avoidance_policy.trt
MIDAS_INPUT = 256   # MiDaS was trained/exported at 256
N_ITERS     = 100
N_WARMUP    = 5


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

    def infer(self, rgb):
        inp = self.transform(rgb).cuda()
        with torch.no_grad():
            pred = self.model(inp)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        pred = F.interpolate(pred, (DEPTH_SIZE, DEPTH_SIZE), mode="bilinear", align_corners=False)
        pred = pred.squeeze(1)
        mn, mx = pred.min(), pred.max()
        depth = (pred - mn) / (mx - mn + 1e-8)
        torch.cuda.synchronize()
        return depth.cpu().numpy().astype(np.float32)   # (1, H, W)


# ---- MiDaS (TRT FP16) -------------------------------------------------------

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


class TRTMidas:
    """MiDaS small as a FP16 TensorRT engine. Built by export_midas_trt.py."""
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        print(f"[trt] loading MiDaS engine: {engine_path}")
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
            print(f"[trt] midas {mode.name.lower()} '{name}' shape={tuple(shape)}")

    def infer(self, rgb):
        # preprocess on CPU: resize to MIDAS_INPUT, normalize, NCHW, float32
        x = cv2.resize(rgb, (MIDAS_INPUT, MIDAS_INPUT), interpolation=cv2.INTER_LINEAR)
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
    # Try GStreamer pipelines FIRST — on Jetson the v4l2 path silently ignores cap.set
    # for CSI cams (IMX477 exposes its native 4032x3040 no matter what you ask).
    # The resolution has to go into the pipeline caps.

    # CSI / IMX477 via libargus
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

    # USB cam via v4l2src with explicit caps (forces the driver to honor size/fps)
    gst_v4l2 = (
        f"v4l2src device=/dev/video0 ! "
        f"video/x-raw, width={CAM_W}, height={CAM_H}, framerate={CAM_FPS}/1 ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink drop=1 max-buffers=1"
    )
    cap = cv2.VideoCapture(gst_v4l2, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print(f"[camera] GStreamer v4l2src at {CAM_W}x{CAM_H} @ {CAM_FPS} fps")
        return cap

    # Last resort — plain /dev/video0 and hope the driver honors the setters
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
    cap = open_camera()
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

    # warmup — first MiDaS + TRT calls are always slow (lazy init, kernel compile)
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
        depth = midas.infer(rgb)
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
        depth = midas.infer(rgb)
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
