"""
GStreamer-based capture + autofocus for Arducam IMX477 (Jetson)

Uses a persistent GStreamer pipeline (multifilesink) so the camera stays
powered throughout the autofocus scan. This is required because the VCM
motor (I2C 0x0C on bus 9) only responds to i2cset while the Argus session
is active — the camera must be streaming for the focus to change.

No GStreamer-enabled OpenCV build required.
"""

import cv2
import glob
import os
import subprocess
import time
from datetime import datetime


# -------- CONFIG --------
SAVE_DIR = "images"
FRAME_DIR = "/tmp/af_frames"
INITIAL_FOCUS = 300
FOCUS_STEP = 20
FOCUS_RANGE = 100   # +/- around current focus
SETTLE_TIME = 0.2   # seconds for lens to move and a fresh frame to arrive
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
I2C_BUS = 9         # VCM motor I2C bus (active only while camera streams)
VCM_ADDR = 0x0C     # Arducam VCM chip I2C address


# -------- PIPELINE --------
def start_pipeline():
    """Start a persistent GStreamer pipeline that continuously writes JPEG frames."""
    os.makedirs(FRAME_DIR, exist_ok=True)
    for f in glob.glob(f"{FRAME_DIR}/frame_*.jpg"):
        os.remove(f)

    cmd = (
        f"gst-launch-1.0 nvarguscamerasrc ! "
        f"'video/x-raw(memory:NVMM),width={CAPTURE_WIDTH},height={CAPTURE_HEIGHT},"
        f"framerate=30/1' ! nvvidconv ! jpegenc ! "
        f"multifilesink location={FRAME_DIR}/frame_%05d.jpg max-files=5"
    )
    proc = subprocess.Popen(cmd, shell=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2.5)  # wait for Argus session to initialize
    return proc


# -------- FOCUS CONTROL --------
def set_focus(value):
    """Send focus value (0–1000) to VCM motor via i2cset."""
    value = max(0, min(1000, value))
    raw = (value << 4) & 0x3FF0
    data1 = (raw >> 8) & 0x3F
    data2 = raw & 0xF0
    os.system(f"sudo i2cset -y {I2C_BUS} 0x{VCM_ADDR:02X} {data1} {data2}")


# -------- FRAME CAPTURE --------
def capture_frame():
    """Return the latest frame written by the persistent pipeline."""
    files = sorted(glob.glob(f"{FRAME_DIR}/frame_*.jpg"))
    if not files:
        return None
    return cv2.imread(files[-1])


# -------- UTIL --------
def sharpness(image):
    """Compute sharpness using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# -------- AUTOFOCUS --------
def autofocus(current_focus):
    """
    Scans +/- FOCUS_RANGE around current_focus and picks the sharpest position.
    The persistent pipeline keeps the camera powered so i2cset reaches the VCM.
    """
    best_focus = current_focus
    best_score = -1

    for f in range(current_focus - FOCUS_RANGE,
                   current_focus + FOCUS_RANGE + FOCUS_STEP,
                   FOCUS_STEP):

        set_focus(f)
        time.sleep(SETTLE_TIME)  # lens settles + fresh frame arrives at 30fps

        frame = capture_frame()
        if frame is None:
            continue
        score = sharpness(frame)
        print(f"  focus={f:4d}  sharpness={score:.1f}")

        if score > best_score:
            best_score = score
            best_focus = f

    set_focus(best_focus)
    return best_focus


# -------- MAIN --------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Starting camera pipeline...")
    proc = start_pipeline()

    try:
        frame = capture_frame()
        if frame is None:
            raise RuntimeError("Pipeline started but no frames received.")
        print("Camera OK.")

        print(f"Setting initial focus: {INITIAL_FOCUS}")
        set_focus(INITIAL_FOCUS)
        time.sleep(0.5)

        print("Running autofocus...")
        best_focus = autofocus(INITIAL_FOCUS)
        print(f"Best focus: {best_focus}")

        print("Capturing final image...")
        time.sleep(SETTLE_TIME)
        frame = capture_frame()
        if frame is None:
            raise RuntimeError("Failed to capture final frame.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SAVE_DIR, f"capture_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Image saved: {filename}")

    finally:
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    main()
