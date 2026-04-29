"""
Arducam IMX477 camera — appsink variant for Jetson Orin Nano.

Drop-in replacement for camera.py that reads frames directly from the
GStreamer pipeline via OpenCV's appsink backend instead of writing JPEG
files to disk.

Why this exists
---------------
The original camera.py uses multifilesink: GStreamer writes JPEG files
continuously and capture() reads the latest one.  multifilesink does NOT
use atomic writes — it opens a file and streams data into it while encoding.
Reading the file mid-write yields a partial JPEG built from rows of two
different frames, causing the horizontal banding seen during drone flight.

With appsink, cap.read() returns a single complete frame that was fully
buffered by the ISP before the call.  No disk I/O, no race conditions.

Rolling shutter note
--------------------
The IMX477 reads rows sequentially top-to-bottom.  Even with a perfect
capture path there is a small inter-row time offset (rolling shutter).
Using a higher-framerate sensor mode reduces this because the sensor reads
all rows faster:

    sensor_mode=3  →  1332×990  @ 120 fps  (4× less rolling shutter)
    sensor_mode=0  →  1920×1080 @ 30 fps   (default)

Run ``v4l2-ctl --list-formats-ext`` on the Jetson to confirm mode numbers
for your firmware.

Usage:
    from tools.camera_appsink import Camera

    with Camera(sensor_mode=3) as cam:
        cam.set_focus(400)
        frame = cam.capture()

        best = cam.autofocus()
        frame = cam.capture()
"""

import cv2
import numpy as np
import os
import time


# -------- CONFIG --------
COARSE_STEP = 50
FINE_STEP = 10
FINE_RANGE = 60
SETTLE_TIME = 0.5
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
I2C_BUS = 9
VCM_ADDR = 0x0C


class Camera:
    """
    Context manager for the IMX477 camera + VCM autofocus (appsink variant).

    Identical public API to camera.Camera — swap the import and everything
    else stays the same.

        with Camera(sensor_mode=3) as cam:
            cam.set_focus(500)
            frame = cam.capture()
            best_focus = cam.autofocus()
    """

    def __init__(self,
                 width: int = CAPTURE_WIDTH,
                 height: int = CAPTURE_HEIGHT,
                 framerate: int = 60,
                 exposure_ns: int | None = None,
                 sensor_mode: int | None = None,
                 verbose: bool = False):
        """
        Parameters
        ----------
        width, height : int
            Capture resolution.
        framerate : int
            Sensor framerate.  Higher = shorter readout window = less rolling
            shutter.  This camera supports:
                mode 0  →  3840×2160 @ 30 fps
                mode 1  →  1920×1080 @ 60 fps  (default, use this for flight)
            Default is 60 to match mode 1.
        exposure_ns : int | None
            Fixed shutter speed in nanoseconds.  Omit for auto-exposure.
        sensor_mode : int | None
            nvarguscamerasrc sensor mode index.  None lets GStreamer pick.
        verbose : bool
            Print the GStreamer pipeline string before opening it.
        """
        self.width = width
        self.height = height
        self.framerate = framerate
        self.exposure_ns = exposure_ns
        self.sensor_mode = sensor_mode
        self.verbose = verbose
        self._cap = None

    # ------------------------------------------------------------------ #
    # Explicit open / close                                               #
    # ------------------------------------------------------------------ #

    def open(self):
        self._start_pipeline()
        return self

    def close(self):
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

    def _build_pipeline(self) -> str:
        src_props = []
        if self.sensor_mode is not None:
            src_props.append(f"sensor-mode={int(self.sensor_mode)}")
        if self.exposure_ns is not None:
            exp_ns = int(self.exposure_ns)
            src_props.append(f"aelock=true exposuretimerange=\"{exp_ns} {exp_ns}\"")

        src_props_str = " ".join(src_props)

        return (
            f"nvarguscamerasrc {src_props_str} ! "
            f"video/x-raw(memory:NVMM),width={self.width},height={self.height},framerate={self.framerate}/1 ! "
            f"nvvidconv ! "
            f"video/x-raw,format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw,format=BGR ! "
            f"appsink drop=1 max-buffers=1"
        )

    def _start_pipeline(self):
        pipeline = self._build_pipeline()
        if self.verbose:
            print(f"[camera] pipeline: {pipeline}")

        self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        time.sleep(2.5)  # wait for Argus session to initialize

        if not self._cap.isOpened() or self.capture() is None:
            self._stop_pipeline()
            raise RuntimeError(
                "Camera pipeline failed to open or no frames received. "
                "Check that the script runs as root and the sensor is connected."
            )

    def _stop_pipeline(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def capture(self) -> np.ndarray | None:
        """Return the latest frame as a BGR numpy array.

        Reads directly from the GStreamer appsink — no disk I/O, no partial
        frames, no race conditions.
        """
        if self._cap is None or not self._cap.isOpened():
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def set_focus(self, value: int):
        """Move the VCM lens to the given focus position (0–1000)."""
        value = max(0, min(1000, int(value)))
        raw = (value << 4) & 0x3FF0
        data1 = (raw >> 8) & 0x3F
        data2 = raw & 0xF0
        os.system(f"i2cset -y {I2C_BUS} 0x{VCM_ADDR:02X} {data1} {data2}")

    def autofocus(self, verbose: bool = True,
                  step_callback: callable = None) -> int:
        """Run a two-pass autofocus scan and return the best focus value (0–1000)."""
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
