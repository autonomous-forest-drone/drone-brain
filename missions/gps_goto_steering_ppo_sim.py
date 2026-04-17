"""
GPS goto with simulated PPO obstacle avoidance.

Identical to gps_goto_steering_v2.py except _goto_goal() adds a PPO lateral
correction on top of the bearing-to-goal controller:

    yaw_rate = bearing_correction + ppo_correction

When PPO outputs 0, the bearing controller runs alone and automatically steers
back toward the goal — no explicit recovery phase needed.

MockPPO replays a scripted sequence of values in [-1, 1] keyed on elapsed
seconds since the start of _goto_goal().  Positive = right, negative = left.

Edit MOCK_SEQUENCE below to test different scenarios.
Each entry: (start_s, end_s, ppo_value, label)

Real PPO integration: replace the single line
    ppo_out, phase = self._mock_ppo.get_output()
with your model inference call.

Usage:
  python3 gps_goto_steering_ppo_sim.py
  python3 gps_goto_steering_ppo_sim.py --lat 56.04 --lon 14.15
  python3 gps_goto_steering_ppo_sim.py --set-goal
  python3 gps_goto_steering_ppo_sim.py --preview   # print sequence and exit
"""

import argparse
import json
import math
import os
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
from mavros_msgs.msg import HomePosition, State, StatusText
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from sensor_msgs.msg import NavSatFix


# ------------------------------------------------------------------ PPO sequence
#
# Simulates an obstacle appearing at t=5s and requiring a rightward dodge.
# After PPO returns to 0, the bearing controller (always running) steers back.
#
# Each row: (start_sec, end_sec, ppo_output, label)
#   ppo_output in [-1, 1]  — positive = yaw right, negative = yaw left
#
MOCK_SEQUENCE = [
    # start   end    ppo    label
    (  0.0,   5.0,   0.0,  'straight'),
    (  5.0,   6.0,   0.3,  'obstacle — slight right'),
    (  6.0,   7.5,   0.5,  'obstacle — right'),
    (  7.5,   9.0,   0.7,  'obstacle — hard right'),
    (  9.0,  10.5,   0.3,  'obstacle — tapering off'),
    ( 10.5,  float('inf'), 0.0, 'obstacle cleared'),
]


class MockPPO:
    """
    Replays MOCK_SEQUENCE by elapsed time since reset().
    Returns (ppo_value, phase_label).
    """

    def __init__(self):
        self._t0 = None

    def reset(self):
        self._t0 = time.monotonic()

    def get_output(self) -> tuple:
        if self._t0 is None:
            return 0.0, 'not started'
        elapsed = time.monotonic() - self._t0
        for start, end, value, label in MOCK_SEQUENCE:
            if start <= elapsed < end:
                return value, label
        return 0.0, 'sequence complete'


# ------------------------------------------------------------------ tuning

CRUISE_SPEED  = 1.0    # m/s
GOAL_RADIUS   = 1.5    # m
HOVER_TIME    = 3.0    # s
GOTO_TIMEOUT  = 120.0  # s

YAW_RATE_MAX      = 0.5    # rad/s — ppo ±1 maps to ±0.5 rad/s
YAW_KP            = 0.5    # bearing proportional gain
YAW_ALIGN_THRESH  = 0.087  # rad (~5°)
YAW_ALIGN_TIMEOUT = 30.0   # s

GPS_AVG_SAMPLES     = 20
GPS_AVG_TIMEOUT     = 60.0
GPS_COVARIANCE_MAX  = 9.0
GPS_QUALITY_TIMEOUT = 120.0

SLOWDOWN_RADIUS    = 3.0   # m
MIN_APPROACH_SPEED = 0.15  # m/s

PRESTREAM_TIME = 2.0
SETPOINT_HZ    = 20

RC_OVERRIDE_MODES = {'ALTCTL', 'POSCTL'}

_CMD_INTERVAL     = 2.0
_ARM_TIMEOUT      = 30.0
_TAKEOFF_TIMEOUT  = 30.0
_OFFBOARD_TIMEOUT = 10.0
_LAND_TIMEOUT     = 30.0

EARTH_R   = 6_371_000.0
GOAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'goal.json')


# ------------------------------------------------------------------ goal persistence

def load_goal():
    if not os.path.exists(GOAL_FILE):
        return None
    try:
        with open(GOAL_FILE) as f:
            d = json.load(f)
        return float(d['lat']), float(d['lon'])
    except Exception as e:
        print(f'[WARN] Could not read {GOAL_FILE}: {e}')
        return None


def save_goal(lat: float, lon: float):
    with open(GOAL_FILE, 'w') as f:
        json.dump({'lat': lat, 'lon': lon}, f, indent=2)
    print(f'Goal saved: ({lat:.7f}, {lon:.7f})  →  {GOAL_FILE}')


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
    print(f'  After PPO → 0, bearing controller auto-steers back to goal.')
    print()


# ------------------------------------------------------------------ node

class GpsGotoPPOSim(Node):
    def __init__(self, goal_lat, goal_lon):
        super().__init__('gps_goto_ppo_sim')

        self._goal_lat     = goal_lat
        self._goal_lon     = goal_lon
        self._target_yaw   = None
        self._goal_local_e = None
        self._goal_local_n = None
        self._mock_ppo     = MockPPO()

        qos = QoSProfile(depth=10,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        state_qos = QoSProfile(depth=10,
                               reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.state = State()
        self.gps   = NavSatFix()
        self.gps.status.status = -1
        self.pose           = None
        self._home          = None
        self._left_rc_modes = False

        self.create_subscription(State,        '/mavros/state',                  self._on_state,      state_qos)
        self.create_subscription(StatusText,   '/mavros/statustext',             self._on_statustext, qos)
        self.create_subscription(NavSatFix,    '/mavros/global_position/global', self._on_gps,        qos)
        self.create_subscription(PoseStamped,  '/mavros/local_position/pose',    self._on_pose,       qos)
        self.create_subscription(HomePosition, '/mavros/home_position/home',     self._on_home,       state_qos)

        self.arm_client  = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode,     '/mavros/set_mode')
        self.land_client = self.create_client(CommandTOL,  '/mavros/cmd/land')

        self.vel_pub = self.create_publisher(
            TwistStamped, '/mavros/setpoint_velocity/cmd_vel', 10)

    # ------------------------------------------------------------------ callbacks

    def _on_state(self, msg):      self.state = msg
    def _on_statustext(self, msg): self.get_logger().info(f'[PX4] {msg.text}')
    def _on_gps(self, msg):        self.gps = msg
    def _on_pose(self, msg):       self.pose = msg

    def _on_home(self, msg):
        first = self._home is None
        self._home = msg
        if first and self._goal_lat is not None and self._goal_lon is not None:
            h_lat = self._home.geo.latitude
            h_lon = self._home.geo.longitude
            self._goal_local_n = math.radians(self._goal_lat - h_lat) * EARTH_R
            self._goal_local_e = math.radians(self._goal_lon - h_lon) * EARTH_R * math.cos(
                math.radians(h_lat))
            self.get_logger().info(
                f'Goal local ENU: E={self._goal_local_e:.2f} m  N={self._goal_local_n:.2f} m')

    # ------------------------------------------------------------------ helpers

    def _rc_override(self):
        if self.state.mode not in RC_OVERRIDE_MODES:
            self._left_rc_modes = True
            return False
        return self._left_rc_modes

    def _has_gps_fix(self):
        return self.gps.status.status >= 0

    def _gps_quality_ok(self):
        if not self._has_gps_fix():
            return False
        if self.gps.position_covariance_type == 0:
            return False
        return self.gps.position_covariance[0] < GPS_COVARIANCE_MAX

    def _yaw_from_pose(self) -> float:
        q = self.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    def _displacement_to_goal(self):
        if self._goal_local_e is not None and self.pose is not None:
            de = self._goal_local_e - self.pose.pose.position.x
            dn = self._goal_local_n - self.pose.pose.position.y
            return dn, de, math.sqrt(dn * dn + de * de)
        lat_rad = math.radians(self.gps.latitude)
        dn = math.radians(self._goal_lat - self.gps.latitude) * EARTH_R
        de = math.radians(self._goal_lon - self.gps.longitude) * EARTH_R * math.cos(lat_rad)
        return dn, de, math.sqrt(dn * dn + de * de)

    def _publish_vel(self, vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0):
        msg = TwistStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.twist.linear.x  = vx
        msg.twist.linear.y  = vy
        msg.twist.linear.z  = vz
        msg.twist.angular.z = yaw_rate
        self.vel_pub.publish(msg)

    def _avg_gps_samples(self, n: int):
        dt = 1.0 / SETPOINT_HZ
        lats, lons, prev = [], [], None
        deadline = time.monotonic() + GPS_AVG_TIMEOUT
        self.get_logger().info(f'Collecting {n} GPS fixes...')
        while len(lats) < n:
            if time.monotonic() > deadline:
                self.get_logger().error(f'GPS averaging timed out ({len(lats)}/{n}).')
                return None
            rclpy.spin_once(self, timeout_sec=dt)
            lat = self.gps.latitude
            if self._has_gps_fix() and lat != prev:
                lats.append(lat)
                lons.append(self.gps.longitude)
                prev = lat
        avg_lat = sum(lats) / n
        avg_lon = sum(lons) / n
        self.get_logger().info(f'GPS average ({n} fixes): ({avg_lat:.7f}, {avg_lon:.7f})')
        return avg_lat, avg_lon

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
                self.get_logger().info('ARM: OK' if res and res.success else 'ARM: retrying...')
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
                self.get_logger().info('TAKEOFF: OK' if res and res.mode_sent else 'TAKEOFF: retrying...')
                last_send = time.monotonic()
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.state.mode == 'AUTO.TAKEOFF':
                self.get_logger().info('AUTO.TAKEOFF confirmed.'); return True
            if not self.state.armed:
                self.get_logger().error('Disarmed before takeoff.'); return False
        return False

    def _switch_offboard(self):
        dt = 1.0 / SETPOINT_HZ
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
                self.get_logger().info('OFFBOARD: OK' if res and res.mode_sent else 'OFFBOARD: retrying...')
                last_send = time.monotonic()
            rclpy.spin_once(self, timeout_sec=dt)
            if self._rc_override(): return None
            if self.state.mode == 'OFFBOARD':
                self.get_logger().info('OFFBOARD confirmed.'); return True
        return False

    def _align_yaw(self):
        dt       = 1.0 / SETPOINT_HZ
        deadline = time.monotonic() + YAW_ALIGN_TIMEOUT
        target   = self._target_yaw
        self.get_logger().info(
            f'Aligning yaw — target {math.degrees(target):.1f}° ENU '
            f'({90.0 - math.degrees(target):.1f}° NED)...')
        while time.monotonic() < deadline:
            err = (target - self._yaw_from_pose() + math.pi) % (2 * math.pi) - math.pi
            if abs(err) < YAW_ALIGN_THRESH:
                self.get_logger().info(f'Yaw aligned ({math.degrees(err):.1f}°).'); return True
            self._publish_vel(
                yaw_rate=max(-YAW_RATE_MAX, min(YAW_RATE_MAX, err * YAW_KP)), vz=0.0)
            self.get_logger().info(
                f'yaw_err={math.degrees(err):.1f}°', throttle_duration_sec=0.5)
            rclpy.spin_once(self, timeout_sec=dt)
            if self._rc_override(): return None
            if not self.state.armed:
                self.get_logger().error('Disarmed during yaw align.'); return False
        self.get_logger().error('YAW_ALIGN_TIMEOUT — aborting.'); return False

    def _goto_goal(self):
        """
        Drive forward at CRUISE_SPEED.

        Every tick:
          1. Compute bearing correction toward goal (proportional, clamped).
          2. Add PPO lateral correction on top.
          3. Clamp combined yaw_rate to ±YAW_RATE_MAX.

        Because bearing is recomputed every tick, when PPO returns to 0
        the drone automatically steers back toward the goal with no explicit
        recovery logic.

        Real PPO swap:
            ppo_out, phase = self._mock_ppo.get_output()
                   ↓
            ppo_out = model.predict(observation)   # float in [-1, 1]
            phase   = 'live'
        """
        dt       = 1.0 / SETPOINT_HZ
        deadline = time.monotonic() + GOTO_TIMEOUT

        self.get_logger().info('Goto goal — PPO sim active.')
        self._mock_ppo.reset()
        prev_phase = None

        while time.monotonic() < deadline:
            dn, de, dist = self._displacement_to_goal()

            if dist <= GOAL_RADIUS:
                self.get_logger().info(f'Goal reached — {dist:.2f} m.'); return True

            # Bearing correction (runs every tick regardless of PPO)
            bearing      = math.atan2(dn, de)
            yaw_err      = (bearing - self._yaw_from_pose() + math.pi) % (2 * math.pi) - math.pi
            bearing_corr = max(-YAW_RATE_MAX, min(YAW_RATE_MAX, yaw_err * YAW_KP))

            # PPO lateral correction — replace this line with real model inference
            ppo_out, phase = self._mock_ppo.get_output()
            ppo_corr = ppo_out * YAW_RATE_MAX

            # Combined yaw rate, clamped to physical limit
            yaw_rate = max(-YAW_RATE_MAX, min(YAW_RATE_MAX, bearing_corr + ppo_corr))

            if phase != prev_phase:
                self.get_logger().info(
                    f'[PPO] → "{phase}"  ppo={ppo_out:+.2f}  '
                    f'bearing_corr={bearing_corr:+.3f}  ppo_corr={ppo_corr:+.3f}  '
                    f'yaw_rate={yaw_rate:+.3f} rad/s')
                prev_phase = phase

            speed = CRUISE_SPEED if dist > SLOWDOWN_RADIUS \
                else max(MIN_APPROACH_SPEED, CRUISE_SPEED * (dist / SLOWDOWN_RADIUS))

            yaw = self._yaw_from_pose()
            ve  = speed * math.cos(yaw)
            vn  = speed * math.sin(yaw)

            self.get_logger().info(
                f'dist={dist:.1f}m  spd={speed:.2f}  '
                f'yaw={math.degrees(self._yaw_from_pose()):.1f}°  '
                f'ppo={ppo_out:+.2f}  yaw_rate={yaw_rate:+.3f} rad/s  '
                f'alt={self.pose.pose.position.z:.2f}m  '
                f'ekf=({self.pose.pose.position.x:.2f},{self.pose.pose.position.y:.2f})  '
                f'gps=({self.gps.latitude:.7f},{self.gps.longitude:.7f})  [{phase}]',
                throttle_duration_sec=1.0)
            self._publish_vel(vx=ve, vy=vn, yaw_rate=yaw_rate, vz=0.0)
            rclpy.spin_once(self, timeout_sec=dt)

            if self._rc_override():
                self.get_logger().info('RC override.'); return None
            if not self.state.armed:
                self.get_logger().error('Disarmed during goto.'); return False

        self.get_logger().error(f'GOTO_TIMEOUT — landing.'); return False

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
                self.get_logger().info('LAND: OK' if res and res.success else 'LAND: retrying...')
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
        rclpy.spin_until_future_complete(self, self.mode_client.call_async(req), timeout_sec=2.0)

    def set_goal_mode(self):
        self.get_logger().info('Waiting for MAVROS...')
        while not self.state.connected:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info('Waiting for GPS fix...')
        deadline = time.monotonic() + 30.0
        while not self._has_gps_fix() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if not self._has_gps_fix():
            self.get_logger().error('No GPS fix after 30 s.'); return
        result = self._avg_gps_samples(GPS_AVG_SAMPLES)
        if result is None: return
        save_goal(*result)

    def run(self):
        self._left_rc_modes = False

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

        self.get_logger().info('Waiting for GPS fix...')
        deadline = time.monotonic() + 30.0
        while not self._has_gps_fix() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if not self._has_gps_fix():
            self.get_logger().error('No GPS fix.'); return

        self.get_logger().info(f'Waiting for GPS quality (cov < {GPS_COVARIANCE_MAX} m²)...')
        deadline = time.monotonic() + GPS_QUALITY_TIMEOUT
        while not self._gps_quality_ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            self.get_logger().info(
                f'GPS cov={self.gps.position_covariance[0]:.2f} m²',
                throttle_duration_sec=5.0)
        if not self._gps_quality_ok():
            self.get_logger().error('GPS quality insufficient.'); return

        self.get_logger().info('Waiting for local position...')
        deadline = time.monotonic() + 30.0
        while self.pose is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.pose is None:
            self.get_logger().error('No local position.'); return

        self.get_logger().info('Waiting for home position...')
        deadline = time.monotonic() + 30.0
        while self._home is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._home is None:
            self.get_logger().warn('No home position — using raw GPS for distance.')

        result = self._avg_gps_samples(GPS_AVG_SAMPLES)
        if result is None: return
        avg_lat, avg_lon = result
        lat_rad = math.radians(avg_lat)
        dn = math.radians(self._goal_lat - avg_lat) * EARTH_R
        de = math.radians(self._goal_lon - avg_lon) * EARTH_R * math.cos(lat_rad)
        dist = math.sqrt(dn * dn + de * de)
        self._target_yaw = math.atan2(dn, de)

        self.get_logger().info(
            f'Goal: ({self._goal_lat:.7f}, {self._goal_lon:.7f})  distance: {dist:.1f} m')
        self.get_logger().info(
            f'Bearing: {math.degrees(self._target_yaw):.1f}° ENU  '
            f'({90.0 - math.degrees(self._target_yaw):.1f}° NED)')
        self.get_logger().info('Press Enter to arm and take off.')

        fd, old = sys.stdin.fileno(), termios.tcgetattr(sys.stdin.fileno())
        try:
            tty.setraw(fd)
            while True:
                rclpy.spin_once(self, timeout_sec=0.05)
                if select.select([sys.stdin], [], [], 0)[0]:
                    ch = sys.stdin.read(1)
                    if ch in ('\r', '\n'):
                        break
                    if ch == '\x03':   # Ctrl+C in raw mode
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

        result = self._switch_offboard()
        if result is None: return
        if not result:
            self.get_logger().error('Failed OFFBOARD.'); return

        result = self._align_yaw()
        if result is None: return
        if not result:
            self._land(); return

        result = self._goto_goal()
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
        description='GPS goto — simulated PPO obstacle avoidance')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--set-goal', action='store_true')
    group.add_argument('--lat', type=float, metavar='DEG')
    group.add_argument('--preview', action='store_true',
                       help='print PPO sequence table and exit')
    parser.add_argument('--lon', type=float, metavar='DEG')
    args = parser.parse_args()

    if args.preview:
        preview_sequence()
        return

    rclpy.init()

    if args.set_goal:
        node = GpsGotoPPOSim(None, None)
        try:
            node.set_goal_mode()
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
        return

    if args.lat is not None:
        if args.lon is None:
            parser.error('--lon is required with --lat')
        goal_lat, goal_lon = args.lat, args.lon
        save_goal(goal_lat, goal_lon)
    else:
        result = load_goal()
        if result is None:
            parser.error('No goal set. Use --lat/--lon or --set-goal.')
        goal_lat, goal_lon = result

    preview_sequence()
    print('=' * 60)
    print('GPS goto — PPO obstacle avoidance simulation')
    print()
    print(f'Goal            : ({goal_lat:.7f}, {goal_lon:.7f})')
    print(f'Cruise speed    : {CRUISE_SPEED} m/s')
    print(f'Yaw rate max    : {math.degrees(YAW_RATE_MAX):.1f}°/s  (ppo ±1 → ±{YAW_RATE_MAX} rad/s)')
    print(f'Arrival radius  : {GOAL_RADIUS} m')
    print(f'Distance sensor : EKF local_position/pose')
    print('RC override     : switch to ALTCTL or POSCTL at any time')
    print()
    print('=' * 60)
    print()

    node = GpsGotoPPOSim(goal_lat, goal_lon)
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
