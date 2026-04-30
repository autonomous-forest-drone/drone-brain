"""
Continuous image capture script using the IMX219 camera (camera1 port).

Usage:
    sudo python tools/capture.py
    sudo python tools/capture.py --interval 0.2

Options:
    --interval SECS     Capture interval in seconds (default: 0.1).

Images are saved to images/test_YYYY-MM-DD_HH-MM-SS/frame_NNNNNN.jpg
relative to the repo root.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow running from repo root or from tools/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from tools.camera_imx219 import Camera


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture images from the IMX219 camera (camera1 port)."
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.1,
        metavar="SECS",
        help="Seconds between captures (default: 0.1).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    session_name = "test_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = repo_root / "images" / session_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving frames to: {out_dir}")

    with Camera() as cam:
        print(f"Capturing every {args.interval}s. Press Ctrl+C to stop.\n")
        frame_idx = 0
        try:
            while True:
                t_start = time.monotonic()

                frame = cam.capture()
                if frame is not None:
                    filename = out_dir / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(filename), frame)
                    print(f"  [{frame_idx:06d}] saved {filename.name}")
                    frame_idx += 1
                else:
                    print("  [warn] no frame received, skipping")

                elapsed = time.monotonic() - t_start
                sleep_for = args.interval - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)

        except KeyboardInterrupt:
            print(f"\nStopped. {frame_idx} frames saved to {out_dir}")


if __name__ == "__main__":
    main()
