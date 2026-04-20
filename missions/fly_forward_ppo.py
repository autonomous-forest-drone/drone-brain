"""
Forward flight with PPO lateral correction — no GPS required.

The drone takes off, locks its heading from the EKF yaw at OFFBOARD entry,
then flies forward at that heading for FLY_DURATION seconds.

A PPO model (or MockPPO) applies yaw corrections to avoid obstacles.
The accumulated lateral deviation from the original track is computed
from EKF pose and passed to the model so it can steer back to track
after an obstacle is cleared.

No GPS is needed after takeoff. Navigation relies on:
  - IMU (gyro) for heading hold
  - EKF local_position/pose for lateral deviation tracking

Coordinate conventions (EKF local frame, ENU):
  forward_dist  — metres travelled along original heading (positive = forward)
  lateral_dev   — metres off the original heading line
                  positive = drifted LEFT, negative = drifted RIGHT

PPO integration: replace the single line
    ppo_out = self._mock_ppo.get_output(self._lateral_dev)
with your model inference call.  The model receives lateral_dev in [-inf, +inf]
and returns a value in [-1, 1].

Usage:
  python3 fly_forward_ppo.py
  python3 fly_forward_ppo.py --duration 20   # fly for 20 seconds
  python3 fly_forward_ppo.py --speed 0.8     # 0.8 m/s forward
  python3 fly_forward_ppo.py --preview       # print PPO sequence and exit

RC override: switch to ALTCTL or POSCTL at any time.

Requires MAVROS:
  ros2 launch mavros px4.launch fcu_url:=/dev/ttyTHS1:115200
"""

import argparse
import math
import select
import subprocess
import sys
import termios
import time
import tty
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import TwistStamped, PoseStamped
from mavros_msgs.msg import State, StatusText
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode


# ------------------------------------------------------------------ PPO sequence
#
# Mock obstacle avoidance sequence.
# Each row: (start_sec, end_sec, ppo_output, label)
#   ppo_output in [-1, 1] — positive = yaw right, negative = yaw left
#   start/end are seconds since _goto starts
#
MOCK_SEQUENCE = [
    # start   end    ppo    label
    (  0.0,   5.0,   0.0,  'straight'),
    (  5.0,   5.5,   0.3,  'obstacle — slight right'),
    (  5.5,   6.0,   0.5,  'obstacle — right'),
    (  6.0,   6.5,   1.0,  'obstacle — hard right'),
    (  6.5,   7.0,   0.7,  'obstacle — tapering off'),
    (  7.0,  float('inf'), 0.0, 'obstacle cleared'),
]


class MockPPO:
    """
    Replays MOCK_SEQUENCE by elapsed time since reset().
    Receives lateral_dev (metres, positive=left) but ignores it in mock mode.
    Returns ppo_value in [-1, 1].
    """

    def __init__(self):
        self._t0    = None
        self._phase = 'not started'

    def reset(self):
        self._t0 = time.monotonic()

    def get_output(self, lateral_dev: float) -> tuple:
        """
        Returns (float ppo_output, str phase_label).
        lateral_dev is available for real model use.
        """
        if self._t0 is None:
            return 0.0, 'not started'
        elapsed = time.monotonic() - self._t0
        for start, end, value, label in MOCK_SEQUENCE:
            if start <= elapsed < end:
                return value, label
        return 0.0, 'sequence complete'


# ------------------------------------------------------------------ tuning

FLY_DURATION  = 15.0   # s   — how long to fly forward (overridden by --duration)
CRUISE_SPEED  = 1.0    # m/s — forward speed (overridden by --speed)
HOVER_TIME    = 3.0    # s   — hover after forward flight before landing

YAW_RATE_MAX  = 0.5    # rad/s — ppo ±1 maps to ±0.5 rad/s
YAW_KP        = 0.5    # bearing correction gain (used during yaw align only)
YAW_ALIGN_THRESH  = 0.087  # rad (~5°) — yaw align done threshold
YAW_ALIGN_TIMEOUT = 30.0   # s

PRESTREAM_TIME = 2.0
SETPOINT_HZ    = 20

RC_OVERRIDE_MODES = {'ALTCTL', 'POSCTL'}

_CMD_INTERVAL     = 2.0
_ARM_TIMEOUT      = 30.0
_TAKEOFF_TIMEOUT  = 30.0
_OFFBOARD_TIMEOUT = 10.0
_LAND_TIMEOUT     = 30.0


# ------------------------------------------------------------------ helpers

def preview_sequence():
    print()
    print('Mock PPO sequence')
    print('-' * 62)
    print(f'  {"Start":>6}  {"End":>6}  {"PPO":>5}  {"Yaw add":>10}  Label')
    print('-' * 62)
    for start, end, val, label in MOCK_SEQUENCE:
        end_str = f'{end:6.1f}' if end != float('inf') else '   ∞  '
        dir_str = '→' if val == 0.0 else ('↻ R' if val > 0 else '↺ L')
        print(f'  {start:6.1f}  {end_str}  {val:+5.2f}  '
              f'{val * YAW_RATE_MAX:+.3f} rad/s  {label}  [{dir_str}]')
    print('-' * 62)
    print(f'  YAW_RATE_MAX = {YAW_RATE_MAX} rad/s  ({math.degrees(YAW_RATE_MAX):.1f}°/s)')
    print(f'  lateral_dev passed to model every tick (positive = drifted left)')
    print()


# ------------------------------------------------------------------ node

class FlyForwardPPO(Node):
    def __init__(self, fly_duration: float, cruise_speed: float):
        super().__init__('fly_forward_ppo')

        self._fly_duration  = fly_duration
        self._cruise_speed  = cruise_speed
        self._mock_ppo      = MockPPO()

        # Set at OFFBOARD entry — defines the forward track
        self._locked_yaw    = None   # rad, ENU
        self._start_x       = None   # m, EKF local
        self._start_y       = None   # m, EKF local

        # Updated every tick
        self._forward_dist  = 0.0    # m, along original heading
        self._lateral_dev   = 0.0    # m, perpendicular (+ = left, - = right)

        qos = QoSProfile(depth=10,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        state_qos = QoSProfile(depth=10,
                               reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.state = State()
        self.pose  = None
        self._left_rc_modes = False

        self.create_subscription(State,       '/mavros/state',               self._on_state,      state_qos)
        self.create_subscription(StatusText,  '/mavros/statustext',          self._on_statustext, qos)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self._on_pose,       qos)

        self.arm_client  = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode,     '/mavros/set_mode')
        self.land_client = self.create_client(CommandTOL,  '/mavros/cmd/land')

        self.vel_pub = self.create_publisher(
            TwistStamped, '/mavros/setpoint_velocity/cmd_vel', 10)

    # ------------------------------------------------------------------ callbacks

    def _on_state(self, msg):      self.state = msg
    def _on_statustext(self, msg): self.get_logger().info(f'[PX4] {msg.text}')
    def _on_pose(self, msg):       self.pose = msg

    # ------------------------------------------------------------------ helpers

    def _rc_override(self):
        if self.state.mode not in RC_OVERRIDE_MODES:
            self._left_rc_modes = True
            return False
        return self._left_rc_modes

    def _yaw_from_pose(self) -> float:
        """ENU yaw (rad). 0 = East, CCW positive."""
        q = self.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    def _update_track(self):
        """
        Recompute forward_dist and lateral_dev from EKF pose.
        Call every tick after _locked_yaw and _start_x/y are set.
        """
        dx = self.pose.pose.position.x - self._start_x
        dy = self.pose.pose.position.y - self._start_y

        # Unit vectors along and perpendicular to locked heading
        fwd_e =  math.cos(self._locked_yaw)
        fwd_n =  math.sin(self._locked_yaw)
        lat_e = -math.sin(self._locked_yaw)   # left is CCW of forward
        lat_n =  math.cos(self._locked_yaw)

        self._forward_dist = dx * fwd_e + dy * fwd_n
        self._lateral_dev  = dx * lat_e + dy * lat_n

    def _publish_vel(self, vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0):
        """ENU world-frame velocity: x=East, y=North, z=up."""
        msg = TwistStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.twist.linear.x  = vx
        msg.twist.linear.y  = vy
        msg.twist.linear.z  = vz
        msg.twist.angular.z = yaw_rate
        self.vel_pub.publish(msg)

    # ------------------------------------------------------------------ commands

    def _arm(self):
        self.get_logger().info('Arming...')
        self.arm_client.wait_for_service()
        deadline, last_send = time.monotonic() + _ARM_TIMEOUT, 0.0
        while time.monotonic() < deadline:
            if time.monotonic() - last_send >= _CMD_INTERVAL:
                req = CommandBool.Request(); req.value = True
                fut = self.arm_client.call_async(req)
                rclpy.spin_until_future_complete(self, fut, timeout_sec=1.0)
                res = fut.result() if fut.done() else None
                self.get_logger().info('ARM: OK' if res and res.success
                                       else 'ARM: retrying...')
                last_send = time.monotonic()
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.state.armed:
                self.get_logger().info('Armed.'); return True
        return False

    def _takeoff(self):
        self.get_logger().info('AUTO.TAKEOFF...')
        self.mode_client.wait_for_service()
        deadline, last_send = time.monotonic() + _TAKEOFF_TIMEOUT, 0.0
        while time.monotonic() < deadline:
            if time.monotonic() - last_send >= _CMD_INTERVAL:
                req = SetMode.Request(); req.custom_mode = 'AUTO.TAKEOFF'
                fut = self.mode_client.call_async(req)
                rclpy.spin_until_future_complete(self, fut, timeout_sec=1.0)
                res = fut.result() if fut.done() else None
                self.get_logger().info('TAKEOFF: OK' if res and res.mode_sent
                                       else 'TAKEOFF: retrying...')
                last_send = time.monotonic()
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.state.mode == 'AUTO.TAKEOFF':
                self.get_logger().info('AUTO.TAKEOFF confirmed.'); return True
            if not self.state.armed:
                self.get_logger().error('Disarmed before takeoff.'); return False
        return False

    def _switch_offboard(self):
        """
        Pre-stream setpoints, switch to OFFBOARD, then lock heading and
        start position from the current EKF pose.
        """
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
            self._publish_vel(vz=0.0)
            if time.monotonic() - last_send >= _CMD_INTERVAL:
                req = SetMode.Request(); req.custom_mode = 'OFFBOARD'
                fut = self.mode_client.call_async(req)
                rclpy.spin_until_future_complete(self, fut, timeout_sec=1.0)
                res = fut.result() if fut.done() else None
                self.get_logger().info('OFFBOARD: OK' if res and res.mode_sent
                                       else 'OFFBOARD: retrying...')
                last_send = time.monotonic()
            rclpy.spin_once(self, timeout_sec=dt)
            if self._rc_override(): return None
            if self.state.mode == 'OFFBOARD':
                # Lock heading and origin from current EKF pose
                self._locked_yaw = self._yaw_from_pose()
                self._start_x    = self.pose.pose.position.x
                self._start_y    = self.pose.pose.position.y
                self.get_logger().info(
                    f'OFFBOARD confirmed. Heading locked: '
                    f'{math.degrees(self._locked_yaw):.1f}° ENU  '
                    f'({90.0 - math.degrees(self._locked_yaw):.1f}° NED compass)  '
                    f'origin EKF: ({self._start_x:.2f}, {self._start_y:.2f})')
                return True
        return False

    def _fly_forward(self):
        """
        Fly forward along the locked heading for _fly_duration seconds.

        Every tick:
          1. Update forward_dist and lateral_dev from EKF pose.
          2. Get PPO correction (replace mock with real model here).
          3. Publish velocity: forward in locked heading + yaw correction.

        Real PPO swap:
            ppo_out = self._mock_ppo.get_output(self._lateral_dev)
                   ↓
            ppo_out = model.predict(observation)  # float in [-1, 1]
            # lateral_dev is available as self._lateral_dev
        """
        dt       = 1.0 / SETPOINT_HZ
        deadline = time.monotonic() + self._fly_duration

        self.get_logger().info(
            f'Flying forward for {self._fly_duration:.0f}s at {self._cruise_speed} m/s. '
            f'PPO sim active.')
        self._mock_ppo.reset()

        # Forward velocity components (constant — locked heading)
        ve_fwd = self._cruise_speed * math.cos(self._locked_yaw)
        vn_fwd = self._cruise_speed * math.sin(self._locked_yaw)

        prev_phase = None

        while time.monotonic() < deadline:
            self._update_track()

            # PPO lateral correction
            # ← real model: ppo_out = model.predict(observation)
            ppo_out, phase = self._mock_ppo.get_output(self._lateral_dev)
            yaw_rate = ppo_out * YAW_RATE_MAX

            if phase != prev_phase:
                self.get_logger().info(
                    f'[PPO] → "{phase}"  ppo={ppo_out:+.2f}  '
                    f'lat_dev={self._lateral_dev:+.2f}m  '
                    f'yaw_rate={yaw_rate:+.3f} rad/s')
                prev_phase = phase

            self.get_logger().info(
                f'fwd={self._forward_dist:.1f}m  '
                f'lat_dev={self._lateral_dev:+.2f}m  '
                f'ppo={ppo_out:+.2f}  '
                f'yaw={math.degrees(self._yaw_from_pose()):.1f}°  '
                f'alt={self.pose.pose.position.z:.2f}m  '
                f'ekf=({self.pose.pose.position.x:.2f},{self.pose.pose.position.y:.2f})',
                throttle_duration_sec=1.0)

            self._publish_vel(vx=ve_fwd, vy=vn_fwd, yaw_rate=yaw_rate, vz=0.0)
            rclpy.spin_once(self, timeout_sec=dt)

            if self._rc_override():
                self.get_logger().info('RC override.'); return None
            if not self.state.armed:
                self.get_logger().error('Disarmed during flight.'); return False

        self.get_logger().info(
            f'Forward flight complete — '
            f'fwd={self._forward_dist:.1f}m  lat_dev={self._lateral_dev:+.2f}m')
        return True

    def _land(self):
        self.get_logger().info('AUTO.LAND...')
        self.land_client.wait_for_service()
        deadline, last_send = time.monotonic() + _LAND_TIMEOUT, 0.0
        while time.monotonic() < deadline:
            if time.monotonic() - last_send >= _CMD_INTERVAL:
                req = CommandTOL.Request()
                req.min_pitch = 0.0;  req.yaw = float('nan')
                req.latitude = req.longitude = req.altitude = 0.0
                fut = self.land_client.call_async(req)
                rclpy.spin_until_future_complete(self, fut, timeout_sec=1.0)
                res = fut.result() if fut.done() else None
                self.get_logger().info('LAND: OK' if res and res.success
                                       else 'LAND: retrying...')
                last_send = time.monotonic()
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._rc_override():             return None
            if self.state.mode == 'AUTO.LAND':
                self.get_logger().info('AUTO.LAND confirmed.'); return True
            if not self.state.armed:
                self.get_logger().info('Disarmed — landed.'); return True
        return False

    def _set_posctl(self):
        req = SetMode.Request(); req.custom_mode = 'POSCTL'
        self.mode_client.wait_for_service()
        rclpy.spin_until_future_complete(self,
            self.mode_client.call_async(req), timeout_sec=2.0)

    # ------------------------------------------------------------------ mission

    def run(self):
        self._left_rc_modes = False

        # MAVROS connection
        deadline = time.monotonic() + 2.0
        while not self.state.connected and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if not self.state.connected:
            self.get_logger().info('MAVROS not connected — launching...')
            subprocess.Popen(
                ['ros2', 'launch', 'mavros', 'px4.launch', 'fcu_url:=/dev/ttyTHS1:115200'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        self.get_logger().info('Waiting for MAVROS...')
        while not self.state.connected:
            rclpy.spin_once(self, timeout_sec=0.1)

        # EKF local position
        self.get_logger().info('Waiting for local position / EKF...')
        deadline = time.monotonic() + 30.0
        while self.pose is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.pose is None:
            self.get_logger().error('No local position after 30 s — aborting.'); return

        self.get_logger().info(
            f'EKF ready. Heading: {math.degrees(self._yaw_from_pose()):.1f}° ENU  '
            f'({90.0 - math.degrees(self._yaw_from_pose()):.1f}° NED compass)')
        self.get_logger().info(
            f'Will fly {self._fly_duration:.0f}s at {self._cruise_speed} m/s '
            f'(~{self._fly_duration * self._cruise_speed:.0f}m).')
        self.get_logger().info('Press Enter to arm and take off.')

        # Wait for Enter
        fd, old = sys.stdin.fileno(), termios.tcgetattr(sys.stdin.fileno())
        try:
            tty.setraw(fd)
            while True:
                rclpy.spin_once(self, timeout_sec=0.05)
                if select.select([sys.stdin], [], [], 0)[0]:
                    ch = sys.stdin.read(1)
                    if ch in ('\r', '\n'):
                        break
                    if ch == '\x03':
                        raise KeyboardInterrupt
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

        if not self._arm():
            self.get_logger().error('Failed to arm.'); return
        if not self._takeoff():
            self.get_logger().error('Failed AUTO.TAKEOFF.'); return

        self.get_logger().info('Climbing...')
        while self.state.mode == 'AUTO.TAKEOFF':
            if self._rc_override(): return
            if not self.state.armed:
                self.get_logger().error('Disarmed during climb.'); return
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info(f'Takeoff complete (now in {self.state.mode}).')

        result = self._switch_offboard()
        if result is None: return
        if not result:
            self.get_logger().error('Failed to enter OFFBOARD.'); return

        result = self._fly_forward()
        if result is None: return

        self.get_logger().info(f'Hovering for {HOVER_TIME:.0f}s...')
        dt = 1.0 / SETPOINT_HZ
        deadline = time.monotonic() + HOVER_TIME
        while time.monotonic() < deadline:
            self._publish_vel(vz=0.0)
            rclpy.spin_once(self, timeout_sec=dt)
            if self._rc_override(): return

        result = self._land()
        if result is None: return
        if not result:
            self.get_logger().error('Failed to land.'); return

        self.get_logger().info('Waiting until disarmed...')
        while self.state.armed:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info('Landed and disarmed.')
        self._set_posctl()


# ------------------------------------------------------------------ entry point

def main():
    parser = argparse.ArgumentParser(
        description='Forward flight with PPO lateral correction — no GPS required')
    parser.add_argument('--duration', type=float, default=FLY_DURATION,
                        metavar='SEC', help=f'seconds to fly forward (default {FLY_DURATION})')
    parser.add_argument('--speed', type=float, default=CRUISE_SPEED,
                        metavar='M/S', help=f'forward speed m/s (default {CRUISE_SPEED})')
    parser.add_argument('--preview', action='store_true',
                        help='print PPO sequence and exit')
    args = parser.parse_args()

    if args.preview:
        preview_sequence()
        return

    rclpy.init()

    preview_sequence()
    print('=' * 60)
    print('Forward flight — PPO obstacle avoidance (no GPS)')
    print()
    print(f'Duration        : {args.duration:.0f}s')
    print(f'Speed           : {args.speed} m/s  '
          f'(~{args.duration * args.speed:.0f}m total)')
    print(f'Yaw rate max    : {math.degrees(YAW_RATE_MAX):.1f}°/s  '
          f'(ppo ±1 → ±{YAW_RATE_MAX} rad/s)')
    print(f'Heading         : locked from EKF at OFFBOARD entry')
    print(f'Lateral track   : computed from EKF pose, passed to PPO every tick')
    print(f'GPS dependency  : none during flight')
    print('RC override     : switch to ALTCTL or POSCTL at any time')
    print()
    print('=' * 60)
    print()

    node = FlyForwardPPO(fly_duration=args.duration, cruise_speed=args.speed)
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
