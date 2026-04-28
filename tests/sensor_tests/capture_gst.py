"""
GStreamer-based capture + autofocus for Arducam IMX477 (Jetson)

Uses subprocess GStreamer pipeline instead of OpenCV's GStreamer backend,
so no GStreamer-enabled OpenCV build is required.
"""

import cv2
import os
import subprocess
import time
from datetime import datetime


# -------- CONFIG --------
SAVE_DIR = "images"
INITIAL_FOCUS = 300
FOCUS_STEP = 20
FOCUS_RANGE = 100   # +/- around current focus
SETTLE_TIME = 0.1   # delay for lens to move and settle
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
I2C_BUS = 9         # camera I2C bus on Orin Nano
VCM_ADDR = 0x0C     # Arducam VCM chip I2C address
TMP_FRAME = "/tmp/af_frame.jpg"


# -------- FOCUS CONTROL --------
def set_focus(value):
    """Send focus value (0–1000) to VCM motor via i2cset."""
    value = max(0, min(1000, value))
    raw = (value << 4) & 0x3FF0
    data1 = (raw >> 8) & 0x3F
    data2 = raw & 0xF0
    os.system(f"i2cset -y {I2C_BUS} 0x{VCM_ADDR:02X} {data1} {data2}")


# -------- FRAME CAPTURE --------
def capture_frame(path=TMP_FRAME):
    """Capture a single frame via GStreamer and return it as a BGR numpy array."""
    cmd = (
        f"gst-launch-1.0 nvarguscamerasrc num-buffers=1 ! "
        f"\"video/x-raw(memory:NVMM),width={CAPTURE_WIDTH},height={CAPTURE_HEIGHT},"
        f"framerate=30/1\" ! nvvidconv ! jpegenc ! filesink location={path}"
    )
    result = subprocess.run(cmd, shell=True, capture_output=True)
    if result.returncode != 0 or not os.path.exists(path):
        return None
    return cv2.imread(path)


# -------- UTIL --------
def sharpness(image):
    """Compute sharpness using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# -------- AUTOFOCUS --------
def autofocus(current_focus):
    """Fast autofocus: scans +/- FOCUS_RANGE around current focus position."""
    best_focus = current_focus
    best_score = -1

    for f in range(current_focus - FOCUS_RANGE,
                   current_focus + FOCUS_RANGE + FOCUS_STEP,
                   FOCUS_STEP):

        set_focus(f)
        time.sleep(SETTLE_TIME)

        frame = capture_frame()
        if frame is None:
            continue
        score = sharpness(frame)

        if score > best_score:
            best_score = score
            best_focus = f

    set_focus(best_focus)
    return best_focus


# -------- MAIN --------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Verifying camera...")
    frame = capture_frame()
    if frame is None:
        raise RuntimeError("Failed to capture frame. Check nvargus_nvraw --lps.")

    print(f"Setting initial focus: {INITIAL_FOCUS}")
    set_focus(INITIAL_FOCUS)

    print("Running autofocus...")
    best_focus = autofocus(INITIAL_FOCUS)
    print(f"Best focus: {best_focus}")

    print("Capturing final image...")
    frame = capture_frame()
    if frame is None:
        raise RuntimeError("Failed to capture final frame.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(SAVE_DIR, f"capture_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    print(f"Image saved: {filename}")


if __name__ == "__main__":
    main()
