"""
GPS goto mission with continuous yaw steering.

arm → AUTO.TAKEOFF → OFFBOARD → fly to goal → hover → AUTO.LAND.

Goal position is stored in goal.json (same directory as this script).

Usage:
  python3 gps_goto_steering.py                          # fly to saved goal
  python3 gps_goto_steering.py --lat 56.04 --lon 14.15 # set new goal and fly
  python3 gps_goto_steering.py --set-goal               # average current GPS, save as goal, exit

Before takeoff the script averages GPS_AVG_SAMPLES GPS fixes (while the drone
sits stationary on the ground) to compute a stable bearing to the goal.

In flight the drone drives forward (body-frame vx) while a proportional yaw
rate keeps the nose pointed at the goal.

RC override: switching to ALTCTL or POSCTL hands control back to the RC.

Requires MAVROS:
  ros2 launch mavros px4.launch fcu_url:=/dev/ttyTHS1:115200
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
from mavros_msgs.msg import State, StatusText
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from sensor_msgs.msg import NavSatFix


# ------------------------------------------------------------------ tuning

CRUISE_SPEED  = 1.0    # m/s — forward speed (body-frame vx)
GOAL_RADIUS   = 1.5    # m   — arrival threshold (GPS accuracy is typically ≥0.8 m)
HOVER_TIME    = 3.0    # s   — hover at goal before landing
GOTO_TIMEOUT  = 120.0  # s

YAW_RATE_MAX      = 0.5    # rad/s — max yaw rate (~28°/s)
YAW_KP            = 0.5    # proportional gain — reduces overshoot
YAW_ALIGN_THRESH  = 0.087  # rad   — done when error < ~5°
YAW_ALIGN_TIMEOUT = 30.0   # s

GPS_AVG_SAMPLES  = 20   # unique GPS fixes to average (~4 s at 5 Hz GPS)

SLOWDOWN_RADIUS    = 3.0   # m   — start slowing down at this distance
MIN_APPROACH_SPEED = 0.15  # m/s — minimum speed near goal

ALT_KP    = 1.0
ALT_KI    = 0.3
ALT_I_MAX = 0.5

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
    """Return (lat, lon) from goal.json, or None if not found."""
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


# ------------------------------------------------------------------ node

class GpsGotoSteering(Node):
    def __init__(self, goal_lat, goal_lon):
        super().__init__('gps_goto_steering')

        self._goal_lat   = goal_lat
        self._goal_lon   = goal_lon
        self._target_yaw = None   # computed from averaged GPS before takeoff

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
        self.target_z       = None
        self._alt_integral  = 0.0
        self._left_rc_modes = False

        self.create_subscription(State,       '/mavros/state',                  self._on_state,      state_qos)
        self.create_subscription(StatusText,  '/mavros/statustext',             self._on_statustext, qos)
        self.create_subscription(NavSatFix,   '/mavros/global_position/global', self._on_gps,        qos)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose',    self._on_pose,       qos)

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

    # ------------------------------------------------------------------ helpers

    def _rc_override(self):
        if self.state.mode not in RC_OVERRIDE_MODES:
            self._left_rc_modes = True
            return False
        return self._left_rc_modes

    def _has_gps_fix(self):
        return self.gps.status.status >= 0

    def _yaw_from_pose(self) -> float:
        """ENU yaw (rad) from pose quaternion. 0=East, CCW positive."""
        q = self.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    def _displacement_to_goal(self):
        """(north_m, east_m, distance_m) from current GPS to goal."""
        lat_rad = math.radians(self.gps.latitude)
        dn = math.radians(self._goal_lat - self.gps.latitude) * EARTH_R
        de = math.radians(self._goal_lon - self.gps.longitude) * EARTH_R * math.cos(lat_rad)
        return dn, de, math.sqrt(dn * dn + de * de)

    def _vz_hold(self) -> float:
        if self.target_z is None or self.pose is None:
            return 0.0
        dt  = 1.0 / SETPOINT_HZ
        err = self.target_z - self.pose.pose.position.z
        self._alt_integral += err * dt
        self._alt_integral  = max(-ALT_I_MAX, min(ALT_I_MAX, self._alt_integral))
        return max(-1.0, min(1.0, ALT_KP * err + ALT_KI * self._alt_integral))

    def _publish_vel(self, vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0):
        """Body-frame velocity: x=forward, y=left, z=up."""
        msg = TwistStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x  = vx
        msg.twist.linear.y  = vy
        msg.twist.linear.z  = vz
        msg.twist.angular.z = yaw_rate
        self.vel_pub.publish(msg)

    def _avg_gps_samples(self, n: int) -> tuple:
        """
        Collect n unique GPS fixes and return (avg_lat, avg_lon).
        Only counts a reading when the GPS message latitude changes,
        so duplicate callbacks at 20 Hz don't inflate the sample count.
        """
        dt = 1.0 / SETPOINT_HZ
        lats, lons, prev = [], [], None
        self.get_logger().info(f'Collecting {n} GPS fixes (~{n // 5} s)...')
        while len(lats) < n:
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
        self.get_logger().info('Arming — will retry until confirmed...')
        self.arm_client.wait_for_service()
        deadline, last_send = time.monotonic() + _ARM_TIMEOUT, 0.0
        while time.monotonic() < deadline:
            if time.monotonic() - last_send >= _CMD_INTERVAL:
                req = CommandBool.Request(); req.value = True
                fut = self.arm_client.call_async(req)
                rclpy.spin_until_future_complete(self, fut, timeout_sec=1.0)
                res = fut.result() if fut.done() else None
                self.get_logger().info('ARM: ACK success' if res and res.success
                                       else 'ARM: no ACK — retrying...')
                last_send = time.monotonic()
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.state.armed:
                self.get_logger().info('Armed.')
                return True
        return False

    def _takeoff(self):
        self.get_logger().info('Commanding AUTO.TAKEOFF...')
        self.mode_client.wait_for_service()
        deadline, last_send = time.monotonic() + _TAKEOFF_TIMEOUT, 0.0
        while time.monotonic() < deadline:
            if time.monotonic() - last_send >= _CMD_INTERVAL:
                req = SetMode.Request(); req.custom_mode = 'AUTO.TAKEOFF'
                fut = self.mode_client.call_async(req)
                rclpy.spin_until_future_complete(self, fut, timeout_sec=1.0)
                res = fut.result() if fut.done() else None
                self.get_logger().info('SET_MODE AUTO.TAKEOFF: ACK success' if res and res.mode_sent
                                       else 'SET_MODE AUTO.TAKEOFF: no ACK — retrying...')
                last_send = time.monotonic()
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.state.mode == 'AUTO.TAKEOFF':
                self.get_logger().info('AUTO.TAKEOFF confirmed.')
                return True
            if not self.state.armed:
                self.get_logger().error('Disarmed before takeoff — aborting.')
                return False
        return False

    def _switch_offboard(self):
        dt = 1.0 / SETPOINT_HZ
        self.get_logger().info(f'Pre-streaming setpoints for {PRESTREAM_TIME:.0f}s...')
        deadline = time.monotonic() + PRESTREAM_TIME
        while time.monotonic() < deadline:
            self._publish_vel()
            rclpy.spin_once(self, timeout_sec=dt)

        # Drone has been in AUTO.LOITER for ~2 s and is fully settled — safe to lock altitude.
        if self.pose is not None:
            self.target_z      = self.pose.pose.position.z
            self._alt_integral = 0.0
            self.get_logger().info(f'Altitude hold target: {self.target_z:.2f} m (ENU)')

        self.get_logger().info('Switching to OFFBOARD...')
        self.mode_client.wait_for_service()
        deadline, last_send = time.monotonic() + _OFFBOARD_TIMEOUT, 0.0
        while time.monotonic() < deadline:
            self._publish_vel(vz=self._vz_hold())
            if time.monotonic() - last_send >= _CMD_INTERVAL:
                req = SetMode.Request(); req.custom_mode = 'OFFBOARD'
                fut = self.mode_client.call_async(req)
                rclpy.spin_until_future_complete(self, fut, timeout_sec=1.0)
                res = fut.result() if fut.done() else None
                self.get_logger().info('SET_MODE OFFBOARD: ACK success' if res and res.mode_sent
                                       else 'SET_MODE OFFBOARD: no ACK — retrying...')
                last_send = time.monotonic()
            rclpy.spin_once(self, timeout_sec=dt)
            if self._rc_override():
                self.get_logger().info('RC override during OFFBOARD switch.')
                return None
            if self.state.mode == 'OFFBOARD':
                self.get_logger().info('OFFBOARD confirmed.')
                return True
        return False

    def _align_yaw(self):
        """
        Rotate in place to face the goal.
        Uses self._target_yaw computed from averaged GPS before takeoff —
        NOT recomputed from noisy live GPS during flight.
        """
        dt       = 1.0 / SETPOINT_HZ
        deadline = time.monotonic() + YAW_ALIGN_TIMEOUT
        target   = self._target_yaw

        self.get_logger().info(
            f'Aligning yaw — target {math.degrees(target):.1f}° ENU '
            f'({90.0 - math.degrees(target):.1f}° NED compass)...'
        )

        while time.monotonic() < deadline:
            err = target - self._yaw_from_pose()
            err = (err + math.pi) % (2 * math.pi) - math.pi

            if abs(err) < YAW_ALIGN_THRESH:
                self.get_logger().info(f'Yaw aligned — error {math.degrees(err):.1f}°.')
                return True

            yaw_rate = max(-YAW_RATE_MAX, min(YAW_RATE_MAX, err * YAW_KP))
            self._publish_vel(yaw_rate=yaw_rate, vz=self._vz_hold())
            self.get_logger().info(
                f'yaw_err={math.degrees(err):.1f}°  rate={yaw_rate:.2f} rad/s',
                throttle_duration_sec=0.5)
            rclpy.spin_once(self, timeout_sec=dt)

            if self._rc_override():
                self.get_logger().info('RC override during yaw align.')
                return None
            if not self.state.armed:
                self.get_logger().error('Disarmed during yaw align — aborting.')
                return False

        self.get_logger().error('YAW_ALIGN_TIMEOUT — proceeding anyway.')
        return True

    def _goto_goal(self):
        """
        Drive forward at CRUISE_SPEED while steering yaw toward goal.
        Decelerates within SLOWDOWN_RADIUS.
        """
        dt       = 1.0 / SETPOINT_HZ
        deadline = time.monotonic() + GOTO_TIMEOUT

        self.get_logger().info(
            f'Steering to goal ({self._goal_lat:.7f}, {self._goal_lon:.7f}) '
            f'at {CRUISE_SPEED} m/s — timeout {GOTO_TIMEOUT:.0f}s')

        while time.monotonic() < deadline:
            dn, de, dist = self._displacement_to_goal()

            if dist <= GOAL_RADIUS:
                self.get_logger().info(f'Goal reached — {dist:.2f} m (threshold {GOAL_RADIUS} m).')
                return True

            bearing   = math.atan2(dn, de)
            yaw_err   = bearing - self._yaw_from_pose()
            yaw_err   = (yaw_err + math.pi) % (2 * math.pi) - math.pi
            yaw_rate  = max(-YAW_RATE_MAX, min(YAW_RATE_MAX, yaw_err * YAW_KP))

            speed = CRUISE_SPEED if dist > SLOWDOWN_RADIUS \
                else max(MIN_APPROACH_SPEED, CRUISE_SPEED * (dist / SLOWDOWN_RADIUS))

            self.get_logger().info(
                f'dist={dist:.1f}m  spd={speed:.2f}  '
                f'yaw_err={math.degrees(yaw_err):.1f}°  rate={yaw_rate:.2f} rad/s',
                throttle_duration_sec=1.0)
            self._publish_vel(vx=speed, yaw_rate=yaw_rate, vz=self._vz_hold())
            rclpy.spin_once(self, timeout_sec=dt)

            if self._rc_override():
                self.get_logger().info('RC override during goto.')
                return None
            if not self.state.armed:
                self.get_logger().error('Disarmed during goto — aborting.')
                return False

        self.get_logger().error(f'GOTO_TIMEOUT ({GOTO_TIMEOUT:.0f}s) — landing.')
        return False

    def _land(self):
        self.get_logger().info('Commanding AUTO.LAND...')
        self.land_client.wait_for_service()
        deadline, last_send = time.monotonic() + _LAND_TIMEOUT, 0.0
        while time.monotonic() < deadline:
            if time.monotonic() - last_send >= _CMD_INTERVAL:
                req = CommandTOL.Request()
                req.min_pitch = 0.0;  req.yaw = float('nan')
                req.latitude  = self.gps.latitude
                req.longitude = self.gps.longitude
                req.altitude  = 0.0
                fut = self.land_client.call_async(req)
                rclpy.spin_until_future_complete(self, fut, timeout_sec=1.0)
                res = fut.result() if fut.done() else None
                self.get_logger().info('cmd/land: ACK success' if res and res.success
                                       else 'cmd/land: no ACK — retrying...')
                last_send = time.monotonic()
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._rc_override():                        return None
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

    # ------------------------------------------------------------------ set-goal mode

    def set_goal_mode(self):
        """Average current GPS and save as goal. Does not fly."""
        self.get_logger().info('Waiting for MAVROS...')
        while not self.state.connected:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info('Waiting for GPS fix...')
        deadline = time.monotonic() + 30.0
        while not self._has_gps_fix() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if not self._has_gps_fix():
            self.get_logger().error('No GPS fix after 30 s — aborting.')
            return

        avg_lat, avg_lon = self._avg_gps_samples(GPS_AVG_SAMPLES)
        save_goal(avg_lat, avg_lon)

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
        else:
            self.get_logger().info('MAVROS already connected — reusing.')

        self.get_logger().info('Waiting for MAVROS connection...')
        while not self.state.connected:
            rclpy.spin_once(self, timeout_sec=0.1)

        # GPS fix
        self.get_logger().info('Waiting for GPS fix...')
        deadline = time.monotonic() + 30.0
        while not self._has_gps_fix() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if not self._has_gps_fix():
            self.get_logger().error('No GPS fix after 30 s — aborting.'); return

        # Local position / EKF
        self.get_logger().info('Waiting for local position / EKF...')
        deadline = time.monotonic() + 30.0
        while self.pose is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.pose is None:
            self.get_logger().error('No local position after 30 s — aborting.'); return

        # ---- Average GPS on the ground for a stable bearing ----
        avg_lat, avg_lon = self._avg_gps_samples(GPS_AVG_SAMPLES)
        lat_rad = math.radians(avg_lat)
        dn = math.radians(self._goal_lat - avg_lat) * EARTH_R
        de = math.radians(self._goal_lon - avg_lon) * EARTH_R * math.cos(lat_rad)
        dist = math.sqrt(dn * dn + de * de)
        self._target_yaw = math.atan2(dn, de)

        self.get_logger().info(
            f'Goal: ({self._goal_lat:.7f}, {self._goal_lon:.7f})  distance: {dist:.1f} m')
        self.get_logger().info(
            f'Bearing: {math.degrees(self._target_yaw):.1f}° ENU  '
            f'({90.0 - math.degrees(self._target_yaw):.1f}° NED compass)')
        self.get_logger().info('Press Enter to arm and take off.')

        # Wait for Enter
        fd, old = sys.stdin.fileno(), termios.tcgetattr(sys.stdin.fileno())
        try:
            tty.setraw(fd)
            while True:
                rclpy.spin_once(self, timeout_sec=0.05)
                if select.select([sys.stdin], [], [], 0)[0]:
                    if sys.stdin.read(1) in ('\r', '\n'):
                        break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

        self.get_logger().info('Arming and taking off...')

        if not self._arm():
            self.get_logger().error('Failed to arm — aborting.'); return
        if not self._takeoff():
            self.get_logger().error('Failed AUTO.TAKEOFF — aborting.'); return

        self.get_logger().info('Climbing...')
        while self.state.mode == 'AUTO.TAKEOFF':
            if self._rc_override(): return
            if not self.state.armed:
                self.get_logger().error('Disarmed during climb — aborting.'); return
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info(f'Takeoff complete (now in {self.state.mode}).')

        result = self._switch_offboard()
        if result is None: return
        if not result:
            self.get_logger().error('Failed to enter OFFBOARD — aborting.'); return

        result = self._align_yaw()
        if result is None: return
        if not result:     return

        result = self._goto_goal()
        if result is None: return

        self.get_logger().info(f'Hovering for {HOVER_TIME:.0f}s...')
        dt = 1.0 / SETPOINT_HZ
        deadline = time.monotonic() + HOVER_TIME
        while time.monotonic() < deadline:
            self._publish_vel(vz=self._vz_hold())
            rclpy.spin_once(self, timeout_sec=dt)
            if self._rc_override(): return

        result = self._land()
        if result is None: return
        if not result:
            self.get_logger().error('Failed to land — disarm manually.'); return

        self.get_logger().info('Waiting until disarmed...')
        while self.state.armed:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info('Landed and disarmed.')
        self._set_posctl()


# ------------------------------------------------------------------ entry point

def main():
    parser = argparse.ArgumentParser(description='GPS goto mission — continuous yaw steering')
    group  = parser.add_mutually_exclusive_group()
    group.add_argument('--set-goal', action='store_true',
                       help='average current GPS position and save as goal, then exit')
    group.add_argument('--lat', type=float, metavar='DEG',
                       help='goal latitude in degrees (saves to goal.json)')
    parser.add_argument('--lon', type=float, metavar='DEG',
                        help='goal longitude (required with --lat)')
    args = parser.parse_args()

    rclpy.init()

    # -- set-goal mode --
    if args.set_goal:
        node = GpsGotoSteering(None, None)
        try:
            node.set_goal_mode()
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
        return

    # -- resolve goal --
    if args.lat is not None:
        if args.lon is None:
            parser.error('--lon is required when using --lat')
        goal_lat, goal_lon = args.lat, args.lon
        save_goal(goal_lat, goal_lon)
    else:
        result = load_goal()
        if result is None:
            parser.error(
                'No goal set. Use --lat/--lon to set one, or '
                '--set-goal to capture the current GPS position.')
        goal_lat, goal_lon = result

    print('=' * 56)
    print('GPS goto — continuous yaw steering')
    print()
    print(f'Goal            : ({goal_lat:.7f}, {goal_lon:.7f})')
    print(f'Cruise speed    : {CRUISE_SPEED} m/s  (forward only)')
    print(f'Yaw rate max    : {math.degrees(YAW_RATE_MAX):.0f}°/s')
    print(f'Arrival radius  : {GOAL_RADIUS} m')
    print(f'Hover at goal   : {HOVER_TIME:.0f}s')
    print(f'Goto timeout    : {GOTO_TIMEOUT:.0f}s')
    print(f'GPS avg samples : {GPS_AVG_SAMPLES} (bearing computed on ground before arming)')
    print('Takeoff alt     : set via MIS_TAKEOFF_ALT in QGC')
    print('RC override     : switch to ALTCTL or POSCTL at any time')
    print()
    print('=' * 56)
    print()

    node = GpsGotoSteering(goal_lat, goal_lon)
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
