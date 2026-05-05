#!/usr/bin/env python3
"""
compass_monitor.py — Print compass heading and yaw from a running MAVROS instance.

  heading   — compass heading in degrees from /mavros/global_position/compass_hdg
              (0° = North, 90° = East, clockwise)
  yaw       — yaw extracted from IMU quaternion (/mavros/imu/data)
              (ENU convention: 0° = East, 90° = North, counter-clockwise)

Usage (MAVROS must already be running):
    python tools/compass_monitor.py
"""

import math
import os
import subprocess
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, MagneticField
from std_msgs.msg import Float64


FCU_URL   = '/dev/ttyTHS1:115200'
ROS_SETUP = '/opt/ros/humble/setup.bash'


def start_mavros():
    env = os.environ.copy()
    cmd = f'source {ROS_SETUP} && ros2 launch mavros px4.launch fcu_url:={FCU_URL}'
    proc = subprocess.Popen(
        ['bash', '-c', cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    print(f'MAVROS started (pid {proc.pid}), waiting for it to come up...')
    time.sleep(5)
    return proc


class CompassMonitor(Node):

    def __init__(self):
        super().__init__('compass_monitor')
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self._heading_deg = None   # GPS-fused heading (optional)
        self._mag_deg     = None   # raw magnetometer heading (no GPS needed)
        self._yaw_deg     = None   # EKF yaw from IMU quaternion
        self._last_print  = 0.0
        self._start_time  = time.monotonic()

        # GPS-fused heading — only available with GPS fix
        self.create_subscription(Float64,       '/mavros/global_position/compass_hdg', self._on_heading, qos)
        # Raw magnetometer — always available when FCU is connected
        self.create_subscription(MagneticField, '/mavros/imu/mag',                     self._on_mag,     qos)
        # EKF orientation (filtered) and raw IMU
        self.create_subscription(Imu,           '/mavros/imu/data',                    self._on_imu,     qos)
        self.create_subscription(Imu,           '/mavros/imu/data_raw',                self._on_imu,     qos)

    def _watchdog(self):
        if self._mag_deg is None and self._yaw_deg is None and self._heading_deg is None:
            elapsed = time.monotonic() - self._start_time
            print(f'  [{elapsed:.0f}s] no data yet — check: ros2 topic list | grep mavros', flush=True)

    def _on_heading(self, msg):
        self._heading_deg = msg.data
        self._maybe_print()

    def _on_mag(self, msg):
        # Heading from raw mag field vector (assumes drone is level — no tilt compensation)
        # atan2(-y, x) gives heading in ENU: 0=East, 90=North
        # Convert to NED compass convention: 0=North, 90=East
        mag_enu_deg = math.degrees(math.atan2(-msg.magnetic_field.y, msg.magnetic_field.x))
        self._mag_deg = (90.0 - mag_enu_deg) % 360.0
        self._maybe_print()

    def _on_imu(self, msg):
        q = msg.orientation
        # Skip if quaternion is identity (data_raw has no orientation estimate)
        if q.w == 0.0 and q.x == 0.0 and q.y == 0.0 and q.z == 0.0:
            return
        if abs(q.w) < 0.01 and abs(q.z) < 0.01:
            return
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw_rad   = math.atan2(siny_cosp, cosy_cosp)
        self._yaw_deg = math.degrees(yaw_rad)

    def _maybe_print(self):
        now = time.monotonic()
        if now - self._last_print < 0.2:   # 5 Hz
            return
        self._last_print = now

        mag = f'{self._mag_deg:5.1f}°'     if self._mag_deg     is not None else '  N/A '
        hdg = f'{self._heading_deg:5.1f}°' if self._heading_deg is not None else '  N/A '
        yaw = f'{self._yaw_deg:+6.1f}°'   if self._yaw_deg     is not None else '   N/A'

        print(f'mag={mag} (N=0,E=90)   gps_hdg={hdg}   ekf_yaw={yaw} (ENU)')


def main():
    mavros_proc = start_mavros()
    rclpy.init()
    node = CompassMonitor()
    print('Listening to MAVROS compass/IMU topics — Ctrl-C to stop', flush=True)

    # Watchdog runs in a plain thread — independent of rclpy so it always fires
    def _watchdog_loop():
        while True:
            time.sleep(3.0)
            node._watchdog()

    threading.Thread(target=_watchdog_loop, daemon=True).start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print()
    finally:
        node.destroy_node()
        rclpy.shutdown()
        mavros_proc.terminate()
        mavros_proc.wait()


if __name__ == '__main__':
    main()
