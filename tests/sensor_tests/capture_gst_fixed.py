"""
GStreamer capture for Arducam IMX477 (Jetson) with fixed focus = 550.
Press Enter to capture an image, then exit.
"""

import cv2
import glob
import os
import subprocess
import time
from datetime import datetime


SAVE_DIR = "images"
FRAME_DIR = "/tmp/af_frames"
FOCUS = 550
SETTLE_TIME = 0.3
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
I2C_BUS = 9
VCM_ADDR = 0x0C


def start_pipeline():
    os.makedirs(FRAME_DIR, exist_ok=True)
    for f in glob.glob(f"{FRAME_DIR}/frame_*.jpg"):
        os.remove(f)

    cmd = (
        f"gst-launch-1.0 nvarguscamerasrc ! "
        f"'video/x-raw(memory:NVMM),width={CAPTURE_WIDTH},height={CAPTURE_HEIGHT},"
        f"framerate=30/1' ! nvvidconv ! jpegenc ! "
        f'multifilesink location="{FRAME_DIR}/frame_%05d.jpg" max-files=5'
    )
    proc = subprocess.Popen(cmd, shell=True,
                            stdout=subprocess.DEVNULL, stderr=None)
    time.sleep(2.5)
    return proc


def set_focus(value):
    value = max(0, min(1000, value))
    raw = (value << 4) & 0x3FF0
    data1 = (raw >> 8) & 0x3F
    data2 = raw & 0xF0
    os.system(f"i2cset -y {I2C_BUS} 0x{VCM_ADDR:02X} {data1} {data2}")


def capture_frame():
    files = sorted(glob.glob(f"{FRAME_DIR}/frame_*.jpg"))
    if not files:
        return None
    return cv2.imread(files[-1])


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Starting camera pipeline...")
    proc = start_pipeline()

    try:
        if capture_frame() is None:
            raise RuntimeError("Pipeline started but no frames received.")
        print("Camera OK.")

        print(f"Setting focus to {FOCUS}...")
        set_focus(FOCUS)
        time.sleep(SETTLE_TIME)

        input("Press Enter to capture image... ")

        frame = capture_frame()
        if frame is None:
            raise RuntimeError("Failed to capture frame.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SAVE_DIR, f"capture_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Image saved: {filename}")

    finally:
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    main()
