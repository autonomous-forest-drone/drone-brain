"""
Continuous image capture script using the IMX477 camera.

Usage:
    sudo python tools/capture.py                        # autofocus, auto-exposure
    sudo python tools/capture.py --focus 400            # fixed focus
    sudo python tools/capture.py --shutter 500000       # 1/2000 s shutter (drone flight)
    sudo python tools/capture.py --focus 400 --shutter 1000000 --interval 0.2

Options:
    --focus FOCUS       Fixed VCM focus value (0–1000). If omitted, autofocus
                        is run before capturing begins.
    --shutter NS        Fixed shutter speed in nanoseconds.  Strongly recommended
                        during flight to reduce rolling-shutter artefacts.
                        Common values:
                            500000  → 1/2000 s  (fast flight)
                          1000000  → 1/1000 s  (moderate motion)
                          2000000  → 1/500 s   (slow / hover)
                        Omit to use ISP auto-exposure.
    --interval SECS     Capture interval in seconds (default: 0.1).

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
        "--shutter",
        type=int,
        default=None,
        metavar="NS",
        help=(
            "Fixed shutter speed in nanoseconds. "
            "E.g. 500000 = 1/2000 s, 1000000 = 1/1000 s. "
            "Omit for ISP auto-exposure."
        ),
    )
    parser.add_argument(
        "--sensor-mode",
        type=int,
        default=None,
        metavar="MODE",
        dest="sensor_mode",
        help=(
            "nvarguscamerasrc sensor mode index. Higher modes run at higher "
            "framerates (shorter sensor readout window = less rolling shutter). "
            "Mode 4 is typically 1332x990 @ 120fps on IMX477. "
            "Omit for default mode."
        ),
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
    if args.shutter is not None:
        print(f"Shutter speed: {args.shutter} ns  (1/{1e9/args.shutter:.0f} s)  — rolling-shutter mitigation active")
    else:
        print("Shutter speed: auto-exposure (consider --shutter for drone flight)")

    with Camera(exposure_ns=args.shutter, sensor_mode=args.sensor_mode) as cam:
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
