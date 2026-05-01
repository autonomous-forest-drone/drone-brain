#!/usr/bin/env python3
"""
export_depth_trt.py — Convert DepthAnything V2 Small to a TensorRT engine.

Steps:
  1. Load the HuggingFace model on CPU.
  2. Export to ONNX (fixed 518×518 input — the processor's native size).
  3. Build a TRT engine from the ONNX (FP16 by default).

Outputs (written to models/freerider/model/):
    depth_anything_v2_small.onnx
    depth_anything_v2_small_fp16.trt   ← default
    depth_anything_v2_small_fp32.trt   ← with --fp32

Usage:
    python models/freerider/export_depth_trt.py
    python models/freerider/export_depth_trt.py --fp32
    python models/freerider/export_depth_trt.py --onnx-only
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import torch
import torch.nn as nn

DEPTH_MODEL_ID = 'depth-anything/Depth-Anything-V2-Small-hf'
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')

# DepthAnything V2 processor always resizes to 518×518 (ViT-S patch stride 14)
INPUT_H = 518
INPUT_W = 518


class _DepthWrapper(nn.Module):
    """Minimal wrapper: pixel_values → predicted_depth (no HF dataclass)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).predicted_depth


def export_onnx(onnx_path: str) -> None:
    from transformers import AutoModelForDepthEstimation

    print(f'[export] Loading {DEPTH_MODEL_ID} on CPU ...')
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID)
    model.eval()

    wrapper = _DepthWrapper(model)
    dummy = torch.zeros(1, 3, INPUT_H, INPUT_W)

    print(f'[export] Exporting ONNX (input 1×3×{INPUT_H}×{INPUT_W}) ...')
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            onnx_path,
            input_names=['pixel_values'],
            output_names=['predicted_depth'],
            opset_version=17,
            do_constant_folding=True,
        )
    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f'[export] ONNX saved → {onnx_path}  ({size_mb:.1f} MB)')


def build_trt(onnx_path: str, trt_path: str, fp16: bool) -> None:
    import tensorrt as trt  # noqa: F401 — local import keeps top-level clean

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    print('[trt] Parsing ONNX ...')
    with open(onnx_path, 'rb') as f:
        data = f.read()
    if not parser.parse(data):
        for i in range(parser.num_errors):
            print(f'  error [{i}]: {parser.get_error(i)}')
        raise RuntimeError('ONNX parsing failed — see errors above')

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4 GiB
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print('[trt] FP16 mode enabled')

    print('[trt] Building engine — this will take several minutes on Jetson ...')
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError('TRT build returned None — check ONNX and TRT logs above')

    with open(trt_path, 'wb') as f:
        f.write(serialized)
    size_mb = os.path.getsize(trt_path) / 1e6
    print(f'[trt] Engine saved → {trt_path}  ({size_mb:.1f} MB)')


def main():
    parser = argparse.ArgumentParser(
        description='Export DepthAnything V2 Small to TensorRT'
    )
    parser.add_argument(
        '--fp32', action='store_true',
        help='Build FP32 engine (default: FP16)',
    )
    parser.add_argument(
        '--onnx-only', action='store_true',
        help='Stop after ONNX export, skip TRT build',
    )
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)

    onnx_path = os.path.join(MODEL_DIR, 'depth_anything_v2_small.onnx')
    precision = 'fp32' if args.fp32 else 'fp16'
    trt_path  = os.path.join(MODEL_DIR, f'depth_anything_v2_small_{precision}.trt')

    export_onnx(onnx_path)

    if args.onnx_only:
        print('[export] --onnx-only set, skipping TRT build.')
        return

    build_trt(onnx_path, trt_path, fp16=not args.fp32)
    print('\nDone.  Update DEPTH_ENGINE_PATH in test_pipeline.py to:')
    print(f'  {trt_path}')


if __name__ == '__main__':
    main()
