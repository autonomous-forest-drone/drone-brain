#!/usr/bin/env python3
"""
test_forward_flight.py — Forward flight test without model lateral corrections.

Same flight sequence as run_freerider (MAVROS, arm, AUTO.TAKEOFF, OFFBOARD,
altitude hold, RC override, AUTO.LAND) but replaces the avoidance loop with a
fixed-duration straight forward flight. Use this to validate takeoff, altitude
hold, and forward speed before enabling the model.

Flow:
    wait for MAVROS → keypress → arm → AUTO.TAKEOFF → wait climb
    → OFFBOARD → fly forward for FORWARD_TIME s → AUTO.LAND

RC override: switching to ALTCTL or POSCTL at any time exits the script.

Usage:
    python models/freerider/tests/test_forward_flight.py
    python models/freerider/tests/test_forward_flight.py --sim
"""

import argparse
import os
import select
import subprocess
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
import termios
import time
import tty

import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped
from mavros_msgs.msg import PlayTuneV2, State, StatusText
from mavros_msgs.srv import CommandBool, SetMode
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

# ---------------------------------------------------------------------------
# Constants  —  match run_freerider
# ---------------------------------------------------------------------------
SETPOINT_HZ     = 10
PRESTREAM_TIME  = 2.0
FORWARD_SPEED   = 0.6    # m/s — same as FIXED_SPEED in run_freerider
FORWARD_TIME    = 5.0    # seconds of straight forward flight before landing

_CMD_INTERVAL     = 2.0
_ARM_TIMEOUT      = 30.0
_TAKEOFF_TIMEOUT  = 30.0
_OFFBOARD_TIMEOUT = 10.0
RC_OVERRIDE_MODES = {'ALTCTL', 'POSCTL'}
SIM_FCU_URL       = 'udp://:14540@194.47.28.91:14580'


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class ForwardFlightNode(Node):

    def __init__(self, sim: bool = False):
        super().__init__('forward_flight_test')

        self._sim           = sim
        self.state          = State()
        self._left_rc_modes = False
        self._alt           = 0.0
        self._target_alt    = 0.0

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        state_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.create_subscription(State,      '/mavros/state',               self._on_state,      state_qos)
        self.create_subscription(StatusText, '/mavros/statustext',          self._on_statustext, qos)
        self.create_subscription(Odometry,   '/mavros/local_position/odom', self._on_odom,       qos)

        self.arm_client  = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode,     '/mavros/set_mode')
        self.vel_pub     = self.create_publisher(TwistStamped, '/mavros/setpoint_velocity/cmd_vel', 10)
        self.tune_pub    = self.create_publisher(PlayTuneV2,   '/mavros/play_tune', 10)

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def _on_state(self, msg):
        self.state = msg

    def _on_statustext(self, msg):
        self.get_logger().info(f'[PX4] {msg.text}')

    def _on_odom(self, msg):
        self._alt = msg.pose.pose.position.z

    # ------------------------------------------------------------------
    # RC override
    # ------------------------------------------------------------------

    def _rc_override(self) -> bool:
        if self.state.mode not in RC_OVERRIDE_MODES:
            self._left_rc_modes = True
            return False
        return self._left_rc_modes

    # ------------------------------------------------------------------
    # Velocity and buzzer
    # ------------------------------------------------------------------

    def _publish_vel(self, vx: float = 0.0, vy: float = 0.0, vz: float = 0.0):
        msg = TwistStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x  = vx
        msg.twist.linear.y  = vy
        msg.twist.linear.z  = vz
        self.vel_pub.publish(msg)

    def _play_tune(self, tune: str):
        msg = PlayTuneV2()
        msg.format = 1
        msg.tune   = tune
        self.tune_pub.publish(msg)

    # ------------------------------------------------------------------
    # Arm / takeoff / offboard / land
    # ------------------------------------------------------------------

    def _arm(self) -> bool:
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

    # ------------------------------------------------------------------
    # Main run sequence
    # ------------------------------------------------------------------

    def run(self):
        dt = 1.0 / SETPOINT_HZ
        self._left_rc_modes = False

        deadline = time.monotonic() + 2.0
        while not self.state.connected and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

        fcu_url = SIM_FCU_URL if self._sim else '/dev/ttyTHS1:115200'
        if not self.state.connected:
            self.get_logger().info('MAVROS not connected — launching...')
            subprocess.Popen(
                ['ros2', 'launch', 'mavros', 'px4.launch', f'fcu_url:={fcu_url}'],
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
                self.get_logger().info('RC override during climb — aborting.')
                return
            if not self.state.armed:
                self.get_logger().error('Disarmed during climb — aborting.')
                return
            rclpy.spin_once(self, timeout_sec=0.1)

        self._target_alt = self._alt
        self.get_logger().info(
            f'Takeoff complete (now in {self.state.mode}). '
            f'Altitude locked: {self._target_alt:.2f} m'
        )
        self._play_tune('MFT120L4 O6 CEG')   # ascending: takeoff done

        result = self._switch_offboard()
        if not result:
            self.get_logger().error('Failed to enter OFFBOARD — aborting.')
            return

        self.get_logger().info(
            f'OFFBOARD active. Flying forward {FORWARD_SPEED} m/s '
            f'for {FORWARD_TIME:.0f}s. Target alt: {self._target_alt:.2f} m'
        )
        self._play_tune('MFT120L4 O6 CCC')   # three beeps: OFFBOARD active

        deadline = time.monotonic() + FORWARD_TIME
        while time.monotonic() < deadline:
            alt_err = self._target_alt - self._alt
            vz = float(np.clip(1.0 * alt_err, -0.5, 0.5))
            self._publish_vel(vx=FORWARD_SPEED, vz=vz)
            rclpy.spin_once(self, timeout_sec=dt)

            remaining = deadline - time.monotonic()
            print(
                f'\r  t={FORWARD_TIME - remaining:.1f}s / {FORWARD_TIME:.0f}s  '
                f'alt={self._alt:.2f}m  vz={vz:+.2f}',
                end='', flush=True,
            )

            if self._rc_override():
                print()
                self.get_logger().info('RC override — stopping.')
                self._play_tune('MFT120L4 O6 GEC')
                return
            if not self.state.armed:
                print()
                self.get_logger().info('Disarmed — stopping.')
                return

        print()
        self.get_logger().info('Forward flight complete.')
        self._play_tune('MFT120L4 O6 GEC')   # descending: mission done, landing
        self._land()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Freerider forward flight test')
    parser.add_argument('--sim', action='store_true', help='Simulation mode')
    args = parser.parse_args()

    print('=' * 60)
    print('Freerider — Forward Flight Test')
    print(f'  Forward speed : {FORWARD_SPEED} m/s')
    print(f'  Forward time  : {FORWARD_TIME:.0f} s  (~{FORWARD_SPEED * FORWARD_TIME:.1f} m)')
    print(f'  Target alt    : captured from AUTO.TAKEOFF')
    print('  RC override   : switch to ALTCTL or POSCTL at any time')
    print('=' * 60)

    rclpy.init()
    node = ForwardFlightNode(sim=args.sim)
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
