"""
Arducam IMX219 camera — appsink variant for Jetson Orin Nano.

Fixed-focus camera connected to the camera1 CSI port (sensor-id=1).
Reads frames directly from the GStreamer pipeline via OpenCV's appsink
backend — no disk I/O, no partial frames.

No VCM / autofocus: the IMX219 has a fixed lens.

Usage:
    from tools.camera_imx219 import Camera

    with Camera() as cam:
        frame = cam.capture()   # returns BGR numpy array
"""

import cv2
import numpy as np
import time


CAPTURE_WIDTH  = 1920
CAPTURE_HEIGHT = 1080
SENSOR_ID      = 0      # source index 0 (camera1 CSI port maps to sensor-id=0 in this device tree)


class Camera:
    """
    Context manager for the IMX219 camera (appsink variant).

        with Camera() as cam:
            frame = cam.capture()
    """

    def __init__(self,
                 width: int = CAPTURE_WIDTH,
                 height: int = CAPTURE_HEIGHT,
                 framerate: int = 30,
                 exposure_ns: int | None = None,
                 verbose: bool = False):
        """
        Parameters
        ----------
        width, height : int
            Capture resolution.
        framerate : int
            Sensor framerate.
        exposure_ns : int | None
            Fixed shutter speed in nanoseconds.  Omit for auto-exposure.
        verbose : bool
            Print the GStreamer pipeline string before opening it.
        """
        self.width       = width
        self.height      = height
        self.framerate   = framerate
        self.exposure_ns = exposure_ns
        self.verbose     = verbose
        self._cap        = None

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
        src_props = [f"sensor-id={SENSOR_ID}"]
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
            print(f"[camera_imx219] pipeline: {pipeline}")

        self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        time.sleep(2.5)  # wait for Argus session to initialize

        if not self._cap.isOpened() or self.capture() is None:
            self._stop_pipeline()
            raise RuntimeError(
                "IMX219 pipeline failed to open or no frames received. "
                "Check that the sensor is connected to camera1 and the script runs as root."
            )

    def _stop_pipeline(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def capture(self) -> np.ndarray | None:
        """Return the latest frame as a BGR numpy array."""
        if self._cap is None or not self._cap.isOpened():
            return None
        ret, frame = self._cap.read()
        return frame if ret else None
