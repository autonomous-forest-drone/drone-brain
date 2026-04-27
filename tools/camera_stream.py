#!/usr/bin/env python3
"""Stream camera from Jetson via GStreamer over TCP (SSH tunnel) or UDP.

--- TCP mode (recommended when SSH'd from outside the local network) ---

  On Jetson:
      python3 camera_stream.py --tcp [--mjpeg] [--port 5000] [--width 640] [--height 480] [--fps 15]

  On your computer — open SSH tunnel in a separate terminal:
      ssh -L 5000:localhost:5000 <user>@<jetson_ip>

  Then play:
      ffplay -fflags nobuffer -flags low_delay -framedrop tcp://localhost:5000   # H.264
      ffplay -fflags nobuffer tcp://localhost:5000                               # MJPEG

--- UDP mode (same local network only) ---

  On Jetson:
      python3 camera_stream.py <receiver_ip> [--port 5000] [--width 640] [--height 480] [--fps 15]

  On your computer:
      ffplay -fflags nobuffer -flags low_delay -framedrop rtp://@:5000
"""

import argparse
import subprocess
import sys


_H264_ENCODERS = [
    ("nvv4l2h264enc", "nvv4l2h264enc bitrate=2000000 iframeinterval=15"),
    ("omxh264enc",    "omxh264enc bitrate=2000000"),
    ("x264enc",       "x264enc tune=zerolatency bitrate=2000 speed-preset=ultrafast key-int-max=15"),
    ("openh264enc",   "openh264enc bitrate=2000000"),
]


def detect_h264_encoder() -> tuple[str, str]:
    for name, element_str in _H264_ENCODERS:
        result = subprocess.run(["gst-inspect-1.0", name], capture_output=True)
        if result.returncode == 0:
            print(f"Using encoder: {name}")
            return name, element_str
    print("ERROR: no supported H.264 encoder found", file=sys.stderr)
    sys.exit(1)


def _raw_caps(encoder_name: str) -> str:
    if encoder_name in ("x264enc", "openh264enc", "omxh264enc", "jpegenc"):
        return "video/x-raw,format=I420"
    return ""


def _source(width: int, height: int, fps: int) -> str:
    return (
        f"nvarguscamerasrc sensor-id=0 ! "
        f"video/x-raw(memory:NVMM),width={width},height={height},framerate={fps}/1 ! "
    )


def build_mjpeg_tcp_pipeline(port: int, width: int, height: int, fps: int) -> str:
    return (
        _source(width, height, fps) +
        "nvvidconv ! video/x-raw,format=I420 ! "
        "queue max-size-buffers=1 leaky=downstream ! "
        "jpegenc quality=60 ! "
        "mpegtsmux ! "
        f"tcpserversink host=0.0.0.0 port={port} sync=false"
    )


def build_h264_tcp_pipeline(port: int, width: int, height: int, fps: int) -> str:
    enc_name, enc_str = detect_h264_encoder()
    caps = _raw_caps(enc_name)
    conv = f"nvvidconv ! {caps} ! " if caps else "nvvidconv ! "
    return (
        _source(width, height, fps) +
        conv +
        f"queue max-size-buffers=1 leaky=downstream ! {enc_str} ! "
        "h264parse config-interval=-1 ! "
        "queue max-size-buffers=1 leaky=downstream ! mpegtsmux ! "
        f"tcpserversink host=0.0.0.0 port={port} sync=false"
    )


def build_h264_udp_pipeline(host: str, port: int, width: int, height: int, fps: int) -> str:
    enc_name, enc_str = detect_h264_encoder()
    caps = _raw_caps(enc_name)
    conv = f"nvvidconv ! {caps} ! " if caps else "nvvidconv ! "
    return (
        _source(width, height, fps) +
        conv +
        f"queue max-size-buffers=1 leaky=downstream ! {enc_str} ! "
        "h264parse config-interval=-1 ! "
        "queue max-size-buffers=1 leaky=downstream ! rtph264pay config-interval=1 pt=96 ! "
        f"udpsink host={host} port={port} sync=false"
    )


def print_tcp_instructions(port: int, mjpeg: bool) -> None:
    play_cmd = (
        f"ffplay -fflags nobuffer tcp://localhost:{port}"
        if mjpeg else
        f"ffplay -fflags nobuffer -flags low_delay -framedrop tcp://localhost:{port}"
    )
    print("\n--- On your computer ---")
    print(f"\n1. SSH tunnel (if not already open):")
    print(f"     ssh -L {port}:localhost:{port} <user>@<jetson_ip>")
    print(f"\n2. Play:")
    print(f"     {play_cmd}")
    print()


def print_udp_instructions(host: str, port: int) -> None:
    print("\n--- On your computer ---")
    print(f"  ffplay -fflags nobuffer -flags low_delay -framedrop rtp://@:{port}")
    print()


def run_pipeline(pipeline: str) -> None:
    cmd = ["gst-launch-1.0", "-e"] + pipeline.split()
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nStream stopped.")
    except subprocess.CalledProcessError as e:
        print(f"GStreamer error: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream Jetson camera via GStreamer")
    parser.add_argument("receiver_ip", nargs="?", help="Receiver IP for UDP mode")
    parser.add_argument("--tcp", action="store_true", help="TCP server mode (use with SSH tunnel)")
    parser.add_argument("--mjpeg", action="store_true", help="Use MJPEG instead of H.264 (lower latency, higher bandwidth)")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    if args.tcp:
        if args.mjpeg:
            pipeline = build_mjpeg_tcp_pipeline(args.port, args.width, args.height, args.fps)
            codec = "MJPEG"
        else:
            pipeline = build_h264_tcp_pipeline(args.port, args.width, args.height, args.fps)
            codec = "H.264"
        print_tcp_instructions(args.port, args.mjpeg)
        print(f"TCP {codec} server on port {args.port}  [{args.width}x{args.height} @ {args.fps}fps]")
    else:
        if not args.receiver_ip:
            parser.error("receiver_ip is required in UDP mode (or use --tcp)")
        pipeline = build_h264_udp_pipeline(args.receiver_ip, args.port, args.width, args.height, args.fps)
        print_udp_instructions(args.receiver_ip, args.port)
        print(f"UDP H.264 → {args.receiver_ip}:{args.port}  [{args.width}x{args.height} @ {args.fps}fps]")

    print("Press Ctrl+C to stop.\n")
    run_pipeline(pipeline)


if __name__ == "__main__":
    main()
