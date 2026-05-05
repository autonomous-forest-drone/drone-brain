#!/usr/bin/env python3
"""
altitude_monitor.py — Print altitude readings from a running MAVROS instance.

Shows odom.z (what the P-controller reads) alongside the absolute altitude
topics so you can spot sign convention issues and drift.

  odom.z    — z from /mavros/local_position/odom  (ENU or NED depending on setup)
  rel       — relative altitude from /mavros/altitude (above home, barometer)
  amsl      — above mean sea level from /mavros/altitude (GPS-fused)

Usage (MAVROS must already be running):
    python tools/altitude_monitor.py
"""

import os
import subprocess
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry

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

try:
    from mavros_msgs.msg import Altitude
    _HAS_ALTITUDE_MSG = True
except ImportError:
    _HAS_ALTITUDE_MSG = False


class AltitudeMonitor(Node):

    def __init__(self):
        super().__init__('altitude_monitor')
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self._odom_z     = None
        self._rel        = None
        self._amsl       = None
        self._last_print = 0.0
        self._start_time = time.monotonic()

        self.create_subscription(Odometry, '/mavros/local_position/odom', self._on_odom, qos)

        if _HAS_ALTITUDE_MSG:
            self.create_subscription(Altitude, '/mavros/altitude', self._on_altitude, qos)
        else:
            self.get_logger().warn('mavros_msgs.Altitude not available — only odom.z shown')

        # Watchdog: print status every 3 s until first message arrives
        self.create_timer(3.0, self._watchdog)

    def _watchdog(self):
        if self._odom_z is None:
            elapsed = time.monotonic() - self._start_time
            print(f'  [{elapsed:.0f}s] waiting for /mavros/local_position/odom ...')

    def _on_odom(self, msg):
        self._odom_z = msg.pose.pose.position.z
        self._maybe_print()

    def _on_altitude(self, msg):
        self._rel  = msg.relative
        self._amsl = msg.amsl

    def _maybe_print(self):
        now = time.monotonic()
        if now - self._last_print < 0.2:   # 5 Hz
            return
        self._last_print = now

        odom  = f'{self._odom_z:+7.3f}' if self._odom_z is not None else '    N/A'
        rel   = f'{self._rel:+7.3f}'   if self._rel   is not None else '    N/A'
        amsl  = f'{self._amsl:7.2f}'   if self._amsl  is not None else '   N/A'

        print(f'odom.z={odom} m   rel={rel} m   amsl={amsl} m')


def main():
    mavros_proc = start_mavros()
    rclpy.init()
    node = AltitudeMonitor()
    print('Listening to MAVROS altitude topics — Ctrl-C to stop')
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
