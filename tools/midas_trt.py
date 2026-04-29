"""MiDaS small as a FP16 TensorRT engine.

Built by models/fortune_cookie/helpers/export_midas_trt.py. Lets us run
depth inference on the Jetson without needing a CUDA-enabled torch wheel
— pycuda handles the GPU directly.
"""

import cv2
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda


MIDAS_INPUT = 256   # MiDaS small was trained / exported at 256x256

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


class TRTMidas:
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

    def infer(self, rgb, out_size=None):
        """Run depth inference. rgb: HxWx3 uint8. Returns (1, H, W) float32 in [0, 1].

        If out_size=(H, W) is given, the depth map is bilinearly resized to that
        shape (used by the policy path which needs a fixed obs size). Otherwise
        the raw engine output shape is returned. The leading dim matches the
        PyTorch MiDaS path so callers can index `[0]` uniformly.
        """
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
                raw = buf["h"].reshape(out_shape).squeeze()
                break

        if out_size is not None and raw.shape != out_size:
            raw = cv2.resize(raw, (out_size[1], out_size[0]), interpolation=cv2.INTER_LINEAR)

        mn, mx = raw.min(), raw.max()
        depth = (raw - mn) / (mx - mn + 1e-8)
        return depth.astype(np.float32)[None]   # (1, H, W)
