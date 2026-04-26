#!/usr/bin/env python3

import os
import select
import subprocess
import sys
import termios
import threading
import time
import tty

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from mavros_msgs.msg import State, StatusText
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode

SETPOINT_HZ    = 10
PRESTREAM_TIME = 2.0
VIDEO_DIR      = '/home/beetlesniffer/PythonProjects/DANI/debug_frames'
GST_PIPELINE   = (
    'nvarguscamerasrc ! '
    'video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1 ! '
    'nvvidconv ! video/x-raw, format=BGRx ! '
    'videoconvert ! video/x-raw, format=BGR ! appsink'
)
_CMD_INTERVAL  = 2.0
_ARM_TIMEOUT   = 30.0
_TAKEOFF_TIMEOUT  = 30.0
_OFFBOARD_TIMEOUT = 10.0
RC_OVERRIDE_MODES = {'ALTCTL', 'POSCTL'}


class TestForwardNode(Node):
    def __init__(self):
        super().__init__('test_forward')

        self.state          = State()
        self._left_rc_modes = False
        self._yaw           = 0.0
        self._recording     = False

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        state_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.create_subscription(State,     '/mavros/state',                 self._on_state,      state_qos)
        self.create_subscription(StatusText,'/mavros/statustext',            self._on_statustext, qos)
        self.create_subscription(Odometry,  '/mavros/local_position/odom',   self._on_odom,       qos)

        self.arm_client  = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode,     '/mavros/set_mode')
        self.land_client = self.create_client(CommandTOL,  '/mavros/cmd/land')

        self.vel_pub = self.create_publisher(TwistStamped, '/mavros/setpoint_velocity/cmd_vel', 10)

    def _on_state(self, msg):
        self.state = msg

    def _on_statustext(self, msg):
        self.get_logger().info(f'[PX4] {msg.text}')

    def _on_odom(self, msg):
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._yaw = np.arctan2(siny_cosp, cosy_cosp)

    def _rc_override(self):
        if self.state.mode not in RC_OVERRIDE_MODES:
            self._left_rc_modes = True
            return False
        return self._left_rc_modes

    def _publish_vel(self, vx=0.0, vy=0.0, vz=0.0):
        yaw = self._yaw
        msg = TwistStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.twist.linear.x  = vx * np.cos(yaw) - vy * np.sin(yaw)
        msg.twist.linear.y  = vx * np.sin(yaw) + vy * np.cos(yaw)
        msg.twist.linear.z  = vz
        self.vel_pub.publish(msg)

    def _arm(self):
        self.get_logger().info('Arming...')
        self.arm_client.wait_for_service()
        deadline, last_send = time.monotonic() + _ARM_TIMEOUT, 0.0
        while time.monotonic() < deadline:
            if time.monotonic() - last_send >= _CMD_INTERVAL:
                req = CommandBool.Request()
                req.value = True
                future = self.arm_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
                last_send = time.monotonic()
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.state.armed:
                self.get_logger().info('Armed.')
                return True
        return False

    def _takeoff(self):
        self.get_logger().info('AUTO.TAKEOFF...')
        self.mode_client.wait_for_service()
        deadline, last_send = time.monotonic() + _TAKEOFF_TIMEOUT, 0.0
        while time.monotonic() < deadline:
            if time.monotonic() - last_send >= _CMD_INTERVAL:
                req = SetMode.Request()
                req.custom_mode = 'AUTO.TAKEOFF'
                future = self.mode_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
                last_send = time.monotonic()
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._rc_override():
                return None
            if self.state.mode == 'AUTO.TAKEOFF':
                self.get_logger().info('AUTO.TAKEOFF confirmed.')
                return True
            if not self.state.armed:
                return False
        return False

    def _switch_offboard(self):
        dt = 1.0 / SETPOINT_HZ
        self.get_logger().info(f'Pre-streaming setpoints for {PRESTREAM_TIME:.0f}s...')
        deadline = time.monotonic() + PRESTREAM_TIME
        while time.monotonic() < deadline:
            self._publish_vel()
            rclpy.spin_once(self, timeout_sec=dt)

        self.get_logger().info('Switching to OFFBOARD...')
        self.mode_client.wait_for_service()
        deadline, last_send = time.monotonic() + _OFFBOARD_TIMEOUT, 0.0
        while time.monotonic() < deadline:
            self._publish_vel()
            if time.monotonic() - last_send >= _CMD_INTERVAL:
                req = SetMode.Request()
                req.custom_mode = 'OFFBOARD'
                future = self.mode_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
                last_send = time.monotonic()
            rclpy.spin_once(self, timeout_sec=dt)
            if self._rc_override():
                return None
            if self.state.mode == 'OFFBOARD':
                self.get_logger().info('OFFBOARD confirmed.')
                return True
        return False

    def _land(self):
        self.get_logger().info('Landing...')
        self.mode_client.wait_for_service()
        deadline, last_send = time.monotonic() + 30.0, 0.0
        while time.monotonic() < deadline:
            if time.monotonic() - last_send >= _CMD_INTERVAL:
                req = SetMode.Request()
                req.custom_mode = 'AUTO.LAND'
                future = self.mode_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
                last_send = time.monotonic()
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.state.mode == 'AUTO.LAND':
                self.get_logger().info('AUTO.LAND confirmed.')
                return
            if not self.state.armed:
                return

    def _start_recording(self):
        os.makedirs(VIDEO_DIR, exist_ok=True)
        self._recording = True
        threading.Thread(target=self._record_video, daemon=True).start()

    def _record_video(self):
        cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.get_logger().error('Cannot open camera for recording')
            return

        fname = os.path.join(VIDEO_DIR, f'{time.strftime("%H%M%S")}.mp4')
        writer = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1080))
        self.get_logger().info(f'Recording to {fname}')

        while self._recording:
            ret, frame = cap.read()
            if ret:
                writer.write(frame)

        writer.release()
        cap.release()
        self.get_logger().info('Recording saved.')
        subprocess.Popen(
            ['rclone', 'copy', VIDEO_DIR, 'dropbox:DANI/debug_frames'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def run(self):
        dt = 1.0 / SETPOINT_HZ
        self._left_rc_modes = False

        deadline = time.monotonic() + 2.0
        while not self.state.connected and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

        if not self.state.connected:
            self.get_logger().info('MAVROS not connected — launching...')
            subprocess.Popen(
                ['ros2', 'launch', 'mavros', 'px4.launch', 'fcu_url:=/dev/ttyTHS1:115200'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

        self.get_logger().info('Waiting for MAVROS connection...')
        while not self.state.connected:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info('Connected. Press any key to arm and take off.')
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                rclpy.spin_once(self, timeout_sec=0.05)
                if select.select([sys.stdin], [], [], 0)[0]:
                    sys.stdin.read(1)
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        if not self._arm():
            self.get_logger().error('Failed to arm — aborting.')
            return

        result = self._takeoff()
        if not result:
            self.get_logger().error('Takeoff failed — aborting.')
            return

        self.get_logger().info('Climbing...')
        while self.state.mode == 'AUTO.TAKEOFF':
            if self._rc_override():
                return
            if not self.state.armed:
                return
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info(f'Takeoff complete (now in {self.state.mode}).')
        self._start_recording()

        result = self._switch_offboard()
        if not result:
            self.get_logger().error('Failed to enter OFFBOARD — aborting.')
            return

        self.get_logger().info('Going forward at 1 m/s for 2 seconds...')
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            self._publish_vel(vx=1.0)
            rclpy.spin_once(self, timeout_sec=dt)
            if self._rc_override():
                return

        self.get_logger().info('Going lateral at -1 m/s for 1 second...')
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            self._publish_vel(vy=-1.0)
            rclpy.spin_once(self, timeout_sec=dt)
            if self._rc_override():
                return

        self.get_logger().info('Stopping...')
        stop_deadline = time.monotonic() + 1.0
        while time.monotonic() < stop_deadline:
            self._publish_vel()
            rclpy.spin_once(self, timeout_sec=dt)

        self._land()
        self._recording = False


def main():
    print('=' * 50)
    print('Test: arm → takeoff → forward 1m/s for 2s → land')
    print('RC override: switch to ALTCTL or POSCTL at any time')
    print('=' * 50)

    rclpy.init()
    node = TestForwardNode()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
    finally:
        node._recording = False
        time.sleep(0.5)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
