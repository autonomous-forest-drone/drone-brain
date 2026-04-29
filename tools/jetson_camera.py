"""Camera + focus helpers for the Jetson IMX477 (Arducam motorized).

OpenCV on this Jetson isn't built with GStreamer, so cv2.VideoCapture(...,
CAP_GSTREAMER) silently returns False. We wrap gst-launch-1.0 as a
subprocess writing JPEGs (same trick as tests/sensor_tests/capture_gst.py)
and serve frames via cv2.imread.

The Arducam VCM (focus motor) only accepts I2C writes while an Argus
session is active — set_focus() must be called AFTER the camera pipeline
is streaming.
"""

import glob
import os
import subprocess
import time

import cv2

# VCM focus motor — Arducam IMX477 on Jetson Orin
I2C_BUS  = 9
VCM_ADDR = 0x0C

# Default focus value used by both the timing test and the flight runner.
# Calibrated against trees ~5 m out with capture_gst.py — close to
# hyperfocal for this lens at typical forest distances. Override with
# --focus on the CLI if you re-AF against a different scene.
DEFAULT_FOCUS = 320  # 0–1000

GST_FRAME_DIR = "/tmp/drone_brain_frames"


def set_focus(value: int) -> None:
    """Write focus value (0–1000) to the VCM via i2cset.

    No-op-safe if i2c-tools isn't installed; the os.system call will just
    print an error. Camera must be actively streaming for this to take.
    """
    value = max(0, min(1000, int(value)))
    raw = (value << 4) & 0x3FF0
    data1 = (raw >> 8) & 0x3F
    data2 = raw & 0xF0
    os.system(f"i2cset -y {I2C_BUS} 0x{VCM_ADDR:02X} {data1} {data2}")


class GstJpegCapture:
    """Drop-in for cv2.VideoCapture: .read() / .release() / .isOpened()."""

    def __init__(self, width, height, fps, frame_dir=GST_FRAME_DIR):
        self.frame_dir = frame_dir
        os.makedirs(self.frame_dir, exist_ok=True)
        for f in glob.glob(f"{self.frame_dir}/frame_*.jpg"):
            os.remove(f)

        # nvarguscamerasrc only allows one open session — a leftover
        # gst-launch from a prior crashed run will block this one.
        try:
            stale = subprocess.check_output(
                ["pgrep", "-af", "gst-launch.*nvargus"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            if stale:
                print(f"[camera] WARNING stale gst-launch holding CSI camera:\n{stale}")
                print("[camera] kill with:  pkill -f 'gst-launch.*nvargus'")
        except subprocess.CalledProcessError:
            pass

        cmd = (
            f"gst-launch-1.0 nvarguscamerasrc ! "
            f"'video/x-raw(memory:NVMM),width={width},height={height},"
            f"framerate={fps}/1' ! nvvidconv ! jpegenc ! "
            f'multifilesink location="{self.frame_dir}/frame_%05d.jpg" max-files=5'
        )
        self._stderr_path = f"{self.frame_dir}/gst.stderr.log"
        self._stderr_fh = open(self._stderr_path, "wb")
        self.proc = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.DEVNULL, stderr=self._stderr_fh,
        )
        self._last_path = None

        # Argus init takes ~2 s on Orin; first frame typically lands by 3 s.
        deadline = time.time() + 8.0
        while time.time() < deadline:
            if glob.glob(f"{self.frame_dir}/frame_*.jpg"):
                return
            if self.proc.poll() is not None:
                self._stderr_fh.close()
                with open(self._stderr_path, "r", errors="replace") as f:
                    err = f.read().strip()
                raise RuntimeError(
                    f"gst-launch exited with code {self.proc.returncode}.\n"
                    f"stderr ({self._stderr_path}):\n{err or '<empty>'}"
                )
            time.sleep(0.1)

        self.release()
        with open(self._stderr_path, "r", errors="replace") as f:
            err = f.read().strip()
        raise RuntimeError(
            f"nvarguscamerasrc produced no frames within 8s. "
            f"Try: sudo systemctl restart nvargus-daemon\n"
            f"stderr ({self._stderr_path}):\n{err or '<empty>'}"
        )

    def isOpened(self):
        return self.proc is not None and self.proc.poll() is None

    def read(self):
        deadline = time.time() + 1.0
        while time.time() < deadline:
            files = sorted(glob.glob(f"{self.frame_dir}/frame_*.jpg"))
            if files and files[-1] != self._last_path:
                path = files[-1]
                # multifilesink can hand us a half-written file — imread
                # returns None in that case, just retry.
                frame = cv2.imread(path)
                if frame is not None:
                    self._last_path = path
                    return True, frame
            time.sleep(0.002)
        return False, None

    def release(self):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
        if getattr(self, "_stderr_fh", None) is not None:
            try:
                self._stderr_fh.close()
            except Exception:
                pass
            self._stderr_fh = None


def open_camera_with_focus(width, height, fps, focus=DEFAULT_FOCUS, settle_s=0.3):
    """Open the camera and lock the lens to `focus`. Returns the cap."""
    cap = GstJpegCapture(width, height, fps)
    if focus is not None:
        set_focus(focus)
        time.sleep(settle_s)  # VCM lens travel
    return cap
