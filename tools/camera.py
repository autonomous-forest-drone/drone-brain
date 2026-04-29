"""
Arducam IMX477 camera utilities for Jetson Orin Nano.

Provides a Camera context manager that starts a persistent GStreamer pipeline
(keeping the camera powered so VCM focus commands work), plus set_focus() and
autofocus() functions.

Requires root (sudo) to run — i2cset needs elevated privileges.

Usage:
    from tools.camera import Camera

    with Camera() as cam:
        cam.set_focus(400)
        frame = cam.capture()          # returns BGR numpy array

        best = cam.autofocus()         # runs full scan, returns best focus value
        frame = cam.capture()
"""

import cv2
import glob
import numpy as np
import os
import subprocess
import time


# -------- CONFIG --------
FRAME_DIR = "/tmp/af_frames"
COARSE_STEP = 50    # step size for full-range coarse scan
FINE_STEP = 10      # step size for fine scan around coarse peak
FINE_RANGE = 60     # +/- around coarse peak for fine scan
SETTLE_TIME = 0.5   # seconds for lens to move and a fresh frame to arrive
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
I2C_BUS = 9         # VCM motor I2C bus (active only while camera streams)
VCM_ADDR = 0x0C     # Arducam VCM chip I2C address


class Camera:
    """
    Context manager for the IMX477 camera + VCM autofocus.

    Opens a persistent GStreamer pipeline on entry so the camera stays
    powered (required for i2cset VCM commands to work), and shuts it
    down cleanly on exit.

        with Camera() as cam:
            cam.set_focus(500)
            frame = cam.capture()
            best_focus = cam.autofocus()
    """

    def __init__(self,
                 width: int = CAPTURE_WIDTH,
                 height: int = CAPTURE_HEIGHT,
                 verbose: bool = False):
        self.width = width
        self.height = height
        self.verbose = verbose
        self._proc = None

    # ------------------------------------------------------------------ #
    # Context manager                                                       #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Explicit open / close (alternative to context manager)             #
    # ------------------------------------------------------------------ #

    def open(self):
        """Start the camera pipeline. Call close() when done."""
        self._start_pipeline()
        return self

    def close(self):
        """Stop the camera pipeline and release resources."""
        self._stop_pipeline()

    # ------------------------------------------------------------------ #
    # Context manager                                                     #
    # ------------------------------------------------------------------ #

    def __enter__(self):
        self._start_pipeline()
        return self

    def __exit__(self, *_):
        self._stop_pipeline()

    # ------------------------------------------------------------------ #
    # Pipeline                                                             #
    # ------------------------------------------------------------------ #

    def _start_pipeline(self):
        os.makedirs(FRAME_DIR, exist_ok=True)
        for f in glob.glob(f"{FRAME_DIR}/frame_*.jpg"):
            os.remove(f)

        cmd = (
            f"gst-launch-1.0 nvarguscamerasrc ! "
            f"'video/x-raw(memory:NVMM),width={self.width},height={self.height},"
            f"framerate=30/1' ! nvvidconv ! jpegenc ! "
            f'multifilesink location="{FRAME_DIR}/frame_%05d.jpg" max-files=5'
        )
        self._proc = subprocess.Popen(
            cmd, shell=True,
            stdout=subprocess.DEVNULL,
            stderr=None if self.verbose else subprocess.DEVNULL,
        )
        time.sleep(2.5)  # wait for Argus session to initialize

        if self.capture() is None:
            self._stop_pipeline()
            raise RuntimeError(
                "Camera pipeline started but no frames received. "
                "Check nvargus_nvraw --lps and that script runs as root."
            )

    def _stop_pipeline(self):
        if self._proc is not None:
            self._proc.terminate()
            self._proc.wait()
            self._proc = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def capture(self) -> np.ndarray | None:
        """Return the latest frame from the pipeline as a BGR numpy array."""
        files = sorted(glob.glob(f"{FRAME_DIR}/frame_*.jpg"))
        if not files:
            return None
        return cv2.imread(files[-1])

    def set_focus(self, value: int):
        """
        Move the VCM lens to the given focus position (0–1000).

        0   = closest focus (lens extended)
        1000 = furthest focus (lens retracted)

        The camera pipeline must be running (i.e. inside a `with Camera()`
        block) for the i2cset command to reach the VCM motor.
        """
        value = max(0, min(1000, int(value)))
        raw = (value << 4) & 0x3FF0
        data1 = (raw >> 8) & 0x3F
        data2 = raw & 0xF0
        os.system(f"i2cset -y {I2C_BUS} 0x{VCM_ADDR:02X} {data1} {data2}")

    def autofocus(self, verbose: bool = True,
                  step_callback: callable = None) -> int:
        """
        Run a two-pass autofocus scan and return the best focus value (0–1000).

        Pass 1 — coarse: scans 0→1000 in steps of COARSE_STEP.
        Pass 2 — fine:   scans ±FINE_RANGE around the coarse peak in steps
                         of FINE_STEP.

        Leaves the lens at the best focus position found.

        step_callback: optional callable invoked at each focus step, useful
                       for keeping an external loop alive (e.g. OFFBOARD
                       setpoint publishing in freerider).
        """
        # Reset to scan start and wait for VCM to fully settle from any
        # previous position (full travel can take ~1–2 s).
        self.set_focus(0)
        time.sleep(2.0)

        if verbose:
            print("  [coarse scan]")
        coarse_best, _ = self._scan(0, 1000, COARSE_STEP, verbose, step_callback)

        fine_start = max(0, coarse_best - FINE_RANGE)
        fine_stop = min(1000, coarse_best + FINE_RANGE)
        if verbose:
            print(f"  [fine scan around {coarse_best}]")
        fine_best, _ = self._scan(fine_start, fine_stop, FINE_STEP, verbose, step_callback)

        self.set_focus(fine_best)
        return fine_best

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _sharpness(self, image) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _scan(self, start: int, stop: int, step: int,
              verbose: bool = True,
              step_callback: callable = None) -> tuple[int, float]:
        best_focus = start
        best_score = -1.0
        for f in range(start, stop + step, step):
            f = max(0, min(1000, f))
            if step_callback is not None:
                step_callback()
            self.set_focus(f)
            time.sleep(SETTLE_TIME)
            frame = self.capture()
            if frame is None:
                continue
            score = self._sharpness(frame)
            if verbose:
                print(f"  focus={f:4d}  sharpness={score:.2f}")
            if score > best_score:
                best_score = score
                best_focus = f
        return best_focus, best_score
