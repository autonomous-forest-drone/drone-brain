"""
Autofocus + Capture Script for Arducam IMX477 (Jetson)

Requirements BEFORE running:

1. Arducam Jetson Driver
   Install the matching .deb for your JetPack from:
   https://github.com/ArduCAM/MIPI_Camera/releases/tag/v0.0.1-orin-nx

   For Jetson Orin Nano / NX on JP6 (L4T R36.4.4):
   wget https://github.com/ArduCAM/MIPI_Camera/releases/download/v0.0.1-orin-nx/arducam-nvidia-l4t-kernel-t234-nx-5.15.148-tegra-36.4.4-20250704150251_arm64_imx477.deb
   sudo dpkg -i arducam-nvidia-l4t-kernel-t234-nx-*.deb
   sudo reboot

2. I2C tools (for focus motor)
   sudo apt install i2c-tools

3. Python packages
   pip install opencv-python numpy

4. Verify camera is detected
   nvargus_nvraw --lps

----------------------------------------

What this script does:
- Opens camera stream via GStreamer + OpenCV
- Sets focus via i2cset over I2C bus 7 (VCM motor at 0x0C)
- Runs a fast autofocus scan around INITIAL_FOCUS
- Captures one image and saves it to ./images/

----------------------------------------

Notes for drone usage:
- Uses small focus adjustments (fast)
- Avoids blocking delays
- Can be adapted into ROS node later

"""

import cv2
import os
import time
from datetime import datetime


# -------- CONFIG --------
SAVE_DIR = "images"
INITIAL_FOCUS = 300
FOCUS_STEP = 20
FOCUS_RANGE = 100   # +/- around current focus
SETTLE_TIME = 0.05  # small delay for lens to move
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
I2C_BUS = 7         # /dev/i2c-7 — VCM motor bus on Orin Nano
VCM_ADDR = 0x0C     # Arducam VCM chip I2C address


# -------- FOCUS CONTROL --------
def set_focus(value):
    """Send focus value (0–1000) to VCM motor via i2cset."""
    value = max(0, min(1000, value))
    raw = (value << 4) & 0x3FF0
    data1 = (raw >> 8) & 0x3F
    data2 = raw & 0xF0
    os.system(f"i2cset -y {I2C_BUS} 0x{VCM_ADDR:02X} {data1} {data2}")


# -------- GSTREAMER PIPELINE --------
def gstreamer_pipeline(width, height, framerate=30):
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink"
    )


# -------- UTIL --------
def sharpness(image):
    """Compute sharpness using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# -------- AUTOFOCUS --------
def autofocus(cap, current_focus):
    """
    Fast autofocus: scans +/- FOCUS_RANGE around current focus position.
    Keeps latency low by using a narrow search window.
    """
    best_focus = current_focus
    best_score = -1

    for f in range(current_focus - FOCUS_RANGE,
                   current_focus + FOCUS_RANGE + FOCUS_STEP,
                   FOCUS_STEP):

        set_focus(f)
        time.sleep(SETTLE_TIME)

        ret, frame = cap.read()
        if not ret:
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

    print("Initializing camera...")
    pipeline = gstreamer_pipeline(CAPTURE_WIDTH, CAPTURE_HEIGHT)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        raise RuntimeError("Failed to open camera. Check nvargus_nvraw --lps and driver install.")

    try:
        time.sleep(1)  # warm-up

        print(f"Setting initial focus: {INITIAL_FOCUS}")
        set_focus(INITIAL_FOCUS)

        print("Running autofocus...")
        best_focus = autofocus(cap, INITIAL_FOCUS)
        print(f"Best focus: {best_focus}")

        # Capture final image at best focus
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to capture final frame.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SAVE_DIR, f"capture_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Image saved: {filename}")

    finally:
        cap.release()


if __name__ == "__main__":
    main()
