#!/usr/bin/env python3

import os
import subprocess
import time
import sys

# source ROS humble setup and apply env vars to current process
ros_env = subprocess.run(
    ["bash", "-c", "source /opt/ros/humble/setup.bash && env"],
    capture_output=True, text=True
)
for line in ros_env.stdout.splitlines():
    if "=" in line:
        k, _, v = line.partition("=")
        os.environ[k] = v

import rclpy
from rclpy.node import Node
from mavros_msgs.msg import Tunnel
import cv2

CHUNK_SIZE = 128
FCU_URL = "/dev/ttyACM0"
MAVROS_CMD = ["ros2", "run", "mavros", "mavros_node", "--ros-args", "-p", f"fcu_url:={FCU_URL}"]


def start_mavros():
    proc = subprocess.Popen(MAVROS_CMD, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=os.environ)
    print("Starting MAVROS...")

    for _ in range(10):
        result = subprocess.run(["ros2", "topic", "list"], capture_output=True, text=True, timeout=10, env=os.environ)
        if "/mavros/state" in result.stdout:
            print("MAVROS ready.")
            return proc
        time.sleep(1)

    print("Warning: MAVROS not detected, proceeding anyway...")
    return proc


class ImageSender(Node):
    def __init__(self):
        super().__init__('image_sender')
        self.pub = self.create_publisher(Tunnel, '/mavros/tunnel/in', 10)

    def send(self, image_path=None):
        if image_path:
            img = cv2.imread(image_path)
        else:
            pipeline = (
                "nvarguscamerasrc sensor-id=0 ! "
                "video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1 ! "
                "nvvidconv ! video/x-raw,format=BGRx ! "
                "videoconvert ! video/x-raw,format=BGR ! appsink"
            )
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            ret, img = cap.read()
            cap.release()
            if not ret:
                print("ERROR: Failed to capture from IMX477")
                return

        if img is None:
            print("ERROR: Image is None, check your source")
            return

        print("Image captured.")

        img = cv2.resize(img, (128, 128))
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 60])
        data = buf.tobytes()
        total_chunks = (len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE

        print(f"Sending {len(data)} bytes in {total_chunks} chunks...")

        for i in range(total_chunks):
            chunk = data[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]

            msg = Tunnel()
            msg.target_system = 0     # broadcast to all systems
            msg.target_component = 0
            msg.payload_type = i          # chunk index
            msg.payload_length = len(chunk)
            msg.payload = list(chunk) + [0] * (128 - len(chunk))  # pad to 128
            self.pub.publish(msg)

            time.sleep(0.05)  # small delay to avoid flooding

        # send a final marker so GCS knows transmission is done
        end_msg = Tunnel()
        end_msg.target_system = 0
        end_msg.target_component = 0
        end_msg.payload_type = 0xFFFF     # end-of-image marker
        end_msg.payload_length = 0
        end_msg.payload = [0] * 128
        self.pub.publish(end_msg)

        print(f"Image sent — {len(data)} bytes, {total_chunks} chunks.")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None

    mavros_proc = start_mavros()

    rclpy.init()
    node = ImageSender()

    time.sleep(1.0)

    try:
        while rclpy.ok():
            node.send(image_path=path)
            print("Waiting 20 seconds before next capture...")
            time.sleep(20)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        mavros_proc.terminate()
        print("MAVROS stopped.")


if __name__ == '__main__':
    main()
