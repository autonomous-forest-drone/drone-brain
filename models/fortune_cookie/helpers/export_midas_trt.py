#!/usr/bin/env python3
"""
One-shot: export MiDaS_small to ONNX, then build a FP16 TensorRT engine.

Run once. Produces:
    models/midas_small.onnx
    models/midas_small.trt

After it finishes, point timing_test.py at midas_small.trt.
"""

import os
import subprocess
import sys

import torch

MIDAS_REPO    = "/home/beetlesniffer/.cache/torch/hub/intel-isl_MiDaS_master"
MIDAS_WEIGHTS = "/home/beetlesniffer/.cache/torch/hub/checkpoints/midas_v21_small_256.pt"
OUT_DIR       = "/home/beetlesniffer/drone-brain/models/fortune_cookie/model"
ONNX_PATH     = os.path.join(OUT_DIR, "midas_small.onnx")
TRT_PATH      = os.path.join(OUT_DIR, "midas_small.trt")
INPUT_SIZE    = 256   # MiDaS_small was trained at 256

if MIDAS_REPO not in sys.path:
    sys.path.insert(0, MIDAS_REPO)
from midas.midas_net_custom import MidasNet_small


def export_onnx():
    if os.path.exists(ONNX_PATH):
        print(f"[onnx] {ONNX_PATH} already exists — skipping export")
        return

    print(f"[onnx] loading weights: {MIDAS_WEIGHTS}")
    model = MidasNet_small(
        None, features=64, backbone="efficientnet_lite3",
        exportable=True, non_negative=True, blocks={"expand": True},
    )
    model.load_state_dict(torch.load(MIDAS_WEIGHTS, map_location="cpu", weights_only=False))
    model.eval()

    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    print(f"[onnx] exporting → {ONNX_PATH}  (fixed shape {INPUT_SIZE}x{INPUT_SIZE})")
    torch.onnx.export(
        model, dummy, ONNX_PATH,
        input_names=["input"], output_names=["depth"],
        opset_version=11,
    )
    print("[onnx] done")


def build_trt():
    if os.path.exists(TRT_PATH):
        print(f"[trt] {TRT_PATH} already exists — delete it to rebuild")
        return

    cmd = [
        "/usr/src/tensorrt/bin/trtexec",
        f"--onnx={ONNX_PATH}",
        f"--saveEngine={TRT_PATH}",
        "--fp16",
    ]
    print("[trt] " + " ".join(cmd))
    subprocess.check_call(cmd)
    print("[trt] done")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    export_onnx()
    build_trt()
    print(f"\nOK — engine at {TRT_PATH}")


if __name__ == "__main__":
    main()
