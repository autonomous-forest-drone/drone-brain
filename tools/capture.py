"""
Continuous image capture script using the IMX477 camera.

Usage:
    sudo python tools/capture.py                  # autofocus, then capture
    sudo python tools/capture.py --focus 400      # fixed focus, then capture
    sudo python tools/capture.py --focus 400 --interval 0.2

Options:
    --focus FOCUS     Fixed VCM focus value (0–1000). If omitted, autofocus
                      is run before capturing begins.
    --interval SECS   Capture interval in seconds (default: 0.1).

Images are saved to images/test_YYYY-MM-DD_HH-MM-SS/frame_NNNNNN.jpg
relative to the repo root.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow running from repo root or from tools/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from tools.camera import Camera


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture images from the IMX477 camera."
    )
    parser.add_argument(
        "--focus",
        type=int,
        default=None,
        metavar="VALUE",
        help="Fixed focus value (0–1000). Omit to run autofocus first.",
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

    # Output directory: images/test_YYYY-MM-DD_HH-MM-SS/
    repo_root = Path(__file__).resolve().parent.parent
    session_name = "test_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = repo_root / "images" / session_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving frames to: {out_dir}")

    with Camera() as cam:
        if args.focus is not None:
            print(f"Setting fixed focus to {args.focus}...")
            cam.set_focus(args.focus)
            time.sleep(0.5)  # let the lens settle
        else:
            print("Running autofocus scan...")
            best = cam.autofocus()
            print(f"Autofocus complete — best focus value: {best}")

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
