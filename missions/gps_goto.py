"""
GPS goto mission: arm → AUTO.TAKEOFF → OFFBOARD → fly to GOAL_LAT/GOAL_LON
at CRUISE_SPEED m/s → hover HOVER_TIME s → AUTO.LAND.

Edit GOAL_LAT and GOAL_LON at the top of the file to set the target.

At each control loop tick the script:
  1. Reads current GPS (lat/lon) and ENU yaw from local_position/pose.
  2. Computes the horizontal displacement vector to the goal.
  3. If distance < GOAL_RADIUS the drone is considered arrived.
  4. Otherwise publishes a body-frame velocity setpoint at CRUISE_SPEED
     pointing toward the goal (no altitude change — vz=0).

Body frame convention (base_link): x=forward, y=left, z=up.
Yaw in ENU: 0=East, CCW positive.  The ENU displacement is rotated into
body frame using the pose quaternion so heading does not matter.

RC override: switching to ALTCTL or POSCTL at any point hands control back
to the RC immediately and the script exits.

After landing the script switches to POSCTL so the next run starts from a
clean mode.

Requires MAVROS running:
  ros2 launch mavros px4.launch fcu_url:=/dev/ttyTHS1:115200
Run:
  python3 gps_goto.py
"""

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
from sensor_msgs.msg import NavSatFix


# ------------------------------------------------------------------ goal & tuning

# !! Set your target GPS coordinates here !!
GOAL_LAT = 56.0484350   # degrees — target latitude
GOAL_LON = 14.1484201   # degrees — target longitude

CRUISE_SPEED   = 1.0    # m/s — horizontal speed toward goal
GOAL_RADIUS    = 0.5    # m   — arrival threshold (horizontal distance)
HOVER_TIME     = 3.0    # s   — hover at goal before landing
GOTO_TIMEOUT   = 120.0  # s   — abort if goal not reached within this time

YAW_RATE_MAX      = 0.5    # rad/s — max yaw rate during alignment (~28°/s)
YAW_ALIGN_THRESH  = 0.087  # rad   — alignment done when error < ~5°
YAW_ALIGN_TIMEOUT = 30.0   # s     — abort if alignment not achieved in time

ALT_KP    = 1.0   # proportional gain for altitude hold (m/s per metre of error)
ALT_KI    = 0.3   # integral gain — eliminates steady-state drift
ALT_I_MAX = 0.5   # anti-windup: clamp on integral contribution (m/s)

PRESTREAM_TIME  = 2.0   # s   — stream zeros before switching to OFFBOARD
SETPOINT_HZ     = 20    # Hz  — must stay >2 Hz or PX4 exits OFFBOARD

RC_OVERRIDE_MODES = {'ALTCTL', 'POSCTL'}

_CMD_INTERVAL     = 2.0
_ARM_TIMEOUT      = 30.0
_TAKEOFF_TIMEOUT  = 30.0
_OFFBOARD_TIMEOUT = 10.0
_LAND_TIMEOUT     = 30.0

EARTH_R = 6_371_000.0   # m — mean Earth radius for small-angle approximation


# ------------------------------------------------------------------ node

class GpsGoto(Node):
    def __init__(self):
        super().__init__('gps_goto')

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        state_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.state = State()
        self.gps   = NavSatFix()
        self.gps.status.status = -1   # NavSatStatus.STATUS_NO_FIX
        self.pose          = None   # PoseStamped — None until first message
        self.target_z      = None   # altitude (m, ENU) to hold during OFFBOARD
        self._alt_integral = 0.0    # PI controller integral accumulator

        # Latch: True once we've seen a non-RC mode (prevents POSCTL start from
        # tripping the override check before AUTO.TAKEOFF is confirmed).
        self._left_rc_modes = False

        self.create_subscription(State,       '/mavros/state',                  self._on_state,      state_qos)
        self.create_subscription(StatusText,  '/mavros/statustext',             self._on_statustext, qos)
        self.create_subscription(NavSatFix,   '/mavros/global_position/raw/fix', self._on_gps,       qos)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose',    self._on_pose,       qos)

        self.arm_client  = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode,     '/mavros/set_mode')
        self.land_client = self.create_client(CommandTOL,  '/mavros/cmd/land')

        self.vel_pub = self.create_publisher(
            TwistStamped, '/mavros/setpoint_velocity/cmd_vel', 10
        )

    # ------------------------------------------------------------------ callbacks

    def _on_state(self, msg: State):
        self.state = msg

    def _on_statustext(self, msg: StatusText):
        self.get_logger().info(f'[PX4] {msg.text}')

    def _on_gps(self, msg: NavSatFix):
        self.gps = msg

    def _on_pose(self, msg: PoseStamped):
        self.pose = msg

    # ------------------------------------------------------------------ helpers

    def _rc_override(self):
        if self.state.mode not in RC_OVERRIDE_MODES:
            self._left_rc_modes = True
            return False
        return self._left_rc_modes

    def _has_gps_fix(self):
        return self.gps.status.status >= 0

    def _yaw_from_pose(self) -> float:
        """Extract ENU yaw (rad) from the current pose quaternion.
        ENU convention: 0 = East, CCW positive."""
        q = self.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    def _displacement_to_goal(self):
        """Return (north_m, east_m, distance_m) from current GPS to goal."""
        lat_rad = math.radians(self.gps.latitude)
        dn = math.radians(GOAL_LAT - self.gps.latitude) * EARTH_R
        de = math.radians(GOAL_LON - self.gps.longitude) * EARTH_R * math.cos(lat_rad)
        dist = math.sqrt(dn * dn + de * de)
        return dn, de, dist

    def _vel_toward_goal(self, dn: float, de: float, dist: float):
        """Body-frame (vx, vy) at CRUISE_SPEED pointing toward goal.
        ENU → body rotation uses current yaw from local_position/pose."""
        yaw = self._yaw_from_pose()
        # unit ENU vector toward goal (East, North)
        ue = de / dist
        un = dn / dist
        # rotate into body frame:  x=forward, y=left
        # forward = East*cos(yaw) + North*sin(yaw)
        # left    = -East*sin(yaw) + North*cos(yaw)
        vx = (ue * math.cos(yaw) + un * math.sin(yaw)) * CRUISE_SPEED
        vy = (-ue * math.sin(yaw) + un * math.cos(yaw)) * CRUISE_SPEED
        return vx, vy

    def _vz_hold(self) -> float:
        """PI vz correction to maintain target_z.
        P term responds immediately; I term eliminates steady-state drift.
        Returns 0 until both pose and target_z are set.  Clamped to ±1 m/s."""
        if self.target_z is None or self.pose is None:
            return 0.0
        dt  = 1.0 / SETPOINT_HZ
        err = self.target_z - self.pose.pose.position.z
        self._alt_integral += err * dt
        self._alt_integral  = max(-ALT_I_MAX, min(ALT_I_MAX, self._alt_integral))
        vz = ALT_KP * err + ALT_KI * self._alt_integral
        return max(-1.0, min(1.0, vz))

    def _publish_vel(self, vx: float = 0.0, vy: float = 0.0, vz: float = 0.0,
                     yaw_rate: float = 0.0):
        """Publish velocity in body frame: x=forward, y=left, z=up, yaw_rate=CCW."""
        msg = TwistStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x  = vx
        msg.twist.linear.y  = vy
        msg.twist.linear.z  = vz
        msg.twist.angular.z = yaw_rate
        self.vel_pub.publish(msg)

    # ------------------------------------------------------------------ commands

    def _arm(self):
        self.get_logger().info('Arming — will retry until confirmed...')
        self.arm_client.wait_for_service()
        deadline  = time.monotonic() + _ARM_TIMEOUT
        last_send = 0.0
        while time.monotonic() < deadline:
            if time.monotonic() - last_send >= _CMD_INTERVAL:
                req       = CommandBool.Request()
                req.value = True
                future    = self.arm_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
                result = future.result() if future.done() else None
                if result and result.success:
                    self.get_logger().info('ARM: ACK success')
                else:
                    self.get_logger().warn('ARM: no ACK — retrying...')
                last_send = time.monotonic()
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.state.armed:
                self.get_logger().info('Armed.')
                return True
        return False

    def _takeoff(self):
        self.get_logger().info('Commanding AUTO.TAKEOFF — will retry until mode confirms...')
        self.mode_client.wait_for_service()
        deadline  = time.monotonic() + _TAKEOFF_TIMEOUT
        last_send = 0.0
        while time.monotonic() < deadline:
            if time.monotonic() - last_send >= _CMD_INTERVAL:
                req             = SetMode.Request()
                req.custom_mode = 'AUTO.TAKEOFF'
                future          = self.mode_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
                result = future.result() if future.done() else None
                if result and result.mode_sent:
                    self.get_logger().info('SET_MODE AUTO.TAKEOFF: ACK success')
                else:
                    self.get_logger().warn('SET_MODE AUTO.TAKEOFF: no ACK — retrying...')
                last_send = time.monotonic()
            rclpy.spin_once(self, timeout_sec=0.1)
            # No RC override check here — drone may start in POSCTL (left by
            # _set_posctl).  The latch won't have fired yet so _rc_override()
            # would be safe, but skipping avoids confusion before AUTO.TAKEOFF confirms.
            if self.state.mode == 'AUTO.TAKEOFF':
                self.get_logger().info('AUTO.TAKEOFF mode confirmed.')
                return True
            if not self.state.armed:
                self.get_logger().error('Drone disarmed before takeoff confirmed — aborting.')
                return False
        return False

    def _switch_offboard(self):
        """Stream zero-velocity setpoints, then switch to OFFBOARD."""
        dt = 1.0 / SETPOINT_HZ

        self.get_logger().info(f'Pre-streaming setpoints for {PRESTREAM_TIME:.0f}s...')
        deadline = time.monotonic() + PRESTREAM_TIME
        while time.monotonic() < deadline:
            self._publish_vel()
            rclpy.spin_once(self, timeout_sec=dt)

        self.get_logger().info('Switching to OFFBOARD...')
        self.mode_client.wait_for_service()
        deadline  = time.monotonic() + _OFFBOARD_TIMEOUT
        last_send = 0.0
        while time.monotonic() < deadline:
            self._publish_vel()
            if time.monotonic() - last_send >= _CMD_INTERVAL:
                req             = SetMode.Request()
                req.custom_mode = 'OFFBOARD'
                future          = self.mode_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
                result = future.result() if future.done() else None
                if result and result.mode_sent:
                    self.get_logger().info('SET_MODE OFFBOARD: ACK success')
                else:
                    self.get_logger().warn('SET_MODE OFFBOARD: no ACK — retrying...')
                last_send = time.monotonic()
            rclpy.spin_once(self, timeout_sec=dt)
            if self._rc_override():
                self.get_logger().info(f'RC override ({self.state.mode}) during OFFBOARD switch — handing off.')
                return None
            if self.state.mode == 'OFFBOARD':
                self.get_logger().info('OFFBOARD confirmed.')
                return True
        return False

    def _align_yaw(self):
        """
        Rotate in place until the drone faces the goal (yaw error < YAW_ALIGN_THRESH).
        Uses proportional yaw rate clamped to YAW_RATE_MAX.
        Returns True on success, None on RC override, False on disarm/timeout.
        """
        dt       = 1.0 / SETPOINT_HZ
        deadline = time.monotonic() + YAW_ALIGN_TIMEOUT

        dn, de, _ = self._displacement_to_goal()
        # Target ENU yaw: angle from East axis toward goal direction.
        # ENU: 0=East, CCW positive.  atan2(North, East) gives the bearing.
        target_yaw = math.atan2(dn, de)

        self.get_logger().info(
            f'Aligning yaw to goal bearing {math.degrees(target_yaw):.1f}° (ENU)  '
            f'from gps=({self.gps.latitude:.7f},{self.gps.longitude:.7f})'
        )

        while time.monotonic() < deadline:
            current_yaw = self._yaw_from_pose()
            error = target_yaw - current_yaw
            # Normalise to (-π, π]
            error = (error + math.pi) % (2 * math.pi) - math.pi

            if abs(error) < YAW_ALIGN_THRESH:
                self.get_logger().info(
                    f'Yaw aligned — error {math.degrees(error):.1f}°.'
                )
                return True

            # Proportional control, clamped to YAW_RATE_MAX
            yaw_rate = max(-YAW_RATE_MAX, min(YAW_RATE_MAX, error))
            self._publish_vel(yaw_rate=yaw_rate, vz=self._vz_hold())
            self.get_logger().info(
                f'gps=({self.gps.latitude:.7f},{self.gps.longitude:.7f})  '
                f'yaw_err={math.degrees(error):.1f}°  rate={yaw_rate:.2f} rad/s',
                throttle_duration_sec=0.5,
            )
            rclpy.spin_once(self, timeout_sec=dt)

            if self._rc_override():
                self.get_logger().info(f'RC override ({self.state.mode}) during yaw align — handing off.')
                return None
            if not self.state.armed:
                self.get_logger().error('Drone disarmed during yaw align — aborting.')
                return False

        self.get_logger().error('YAW_ALIGN_TIMEOUT reached — proceeding anyway.')
        return True   # non-fatal: fly toward goal even if not perfectly aligned

    def _goto_goal(self):
        """
        Fly toward GOAL_LAT/GOAL_LON at CRUISE_SPEED until within GOAL_RADIUS.
        Returns True on arrival, None on RC override, False on disarm/timeout.
        """
        dt      = 1.0 / SETPOINT_HZ
        deadline = time.monotonic() + GOTO_TIMEOUT

        self.get_logger().info(
            f'Flying to goal ({GOAL_LAT:.7f}, {GOAL_LON:.7f}) '
            f'at {CRUISE_SPEED} m/s — timeout {GOTO_TIMEOUT:.0f}s'
        )

        while time.monotonic() < deadline:
            dn, de, dist = self._displacement_to_goal()

            if dist <= GOAL_RADIUS:
                self.get_logger().info(
                    f'Goal reached — distance {dist:.2f} m (threshold {GOAL_RADIUS} m).'
                )
                return True

            vx, vy = self._vel_toward_goal(dn, de, dist)
            self.get_logger().info(
                f'gps=({self.gps.latitude:.7f},{self.gps.longitude:.7f})  '
                f'dist={dist:.1f}m  vx={vx:.2f}  vy={vy:.2f}',
                throttle_duration_sec=1.0,
            )
            self._publish_vel(vx=vx, vy=vy, vz=self._vz_hold())
            rclpy.spin_once(self, timeout_sec=dt)

            if self._rc_override():
                self.get_logger().info(f'RC override ({self.state.mode}) during goto — handing off.')
                return None
            if not self.state.armed:
                self.get_logger().error('Drone disarmed during goto — aborting.')
                return False

        self.get_logger().error(
            f'GOTO_TIMEOUT ({GOTO_TIMEOUT:.0f}s) reached — goal not achieved. Landing.'
        )
        return False

    def _land(self):
        """Stop velocity stream and command AUTO.LAND via cmd/land."""
        self.get_logger().info('Stopping velocity stream and commanding AUTO.LAND...')
        self.land_client.wait_for_service()
        deadline  = time.monotonic() + _LAND_TIMEOUT
        last_send = 0.0
        while time.monotonic() < deadline:
            if time.monotonic() - last_send >= _CMD_INTERVAL:
                req           = CommandTOL.Request()
                req.min_pitch = 0.0
                req.yaw       = float('nan')
                req.latitude  = self.gps.latitude
                req.longitude = self.gps.longitude
                req.altitude  = 0.0
                future = self.land_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
                result = future.result() if future.done() else None
                if result and result.success:
                    self.get_logger().info('cmd/land: ACK success')
                else:
                    self.get_logger().warn('cmd/land: no ACK — retrying...')
                last_send = time.monotonic()
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._rc_override():
                self.get_logger().info(f'RC override ({self.state.mode}) during land — handing off.')
                return None
            if self.state.mode == 'AUTO.LAND':
                self.get_logger().info('AUTO.LAND mode confirmed.')
                return True
            if not self.state.armed:
                self.get_logger().info('Drone disarmed — already landed.')
                return True
        return False

    def _set_posctl(self):
        """Switch to POSCTL so the next run starts from a clean mode."""
        self.get_logger().info('Switching to POSCTL...')
        self.mode_client.wait_for_service()
        req = SetMode.Request()
        req.custom_mode = 'POSCTL'
        future = self.mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        result = future.result() if future.done() else None
        if result and result.mode_sent:
            self.get_logger().info('POSCTL: mode sent.')
        else:
            self.get_logger().warn('POSCTL: no ACK — leaving mode as-is.')

    # ------------------------------------------------------------------ mission

    def run(self):
        self._left_rc_modes = False

        # ---- MAVROS connection ----
        deadline = time.monotonic() + 2.0
        while not self.state.connected and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

        if not self.state.connected:
            self.get_logger().info('MAVROS not connected — launching...')
            subprocess.Popen(
                ['ros2', 'launch', 'mavros', 'px4.launch', 'fcu_url:=/dev/ttyTHS1:115200'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            self.get_logger().info('MAVROS already connected — reusing.')

        self.get_logger().info('Waiting for MAVROS connection...')
        while not self.state.connected:
            rclpy.spin_once(self, timeout_sec=0.1)

        # ---- wait for GPS fix ----
        self.get_logger().info('Waiting for GPS fix...')
        deadline = time.monotonic() + 30.0
        while not self._has_gps_fix() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if not self._has_gps_fix():
            self.get_logger().error('No GPS fix after 30 s — aborting.')
            return

        # ---- wait for pose (needed for yaw) ----
        self.get_logger().info('Waiting for local position / EKF...')
        deadline = time.monotonic() + 30.0
        while self.pose is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.pose is None:
            self.get_logger().error('No local position after 30 s — aborting.')
            return

        # ---- show goal distance ----
        _, _, dist = self._displacement_to_goal()
        self.get_logger().info(
            f'GPS fix OK. Goal: ({GOAL_LAT:.7f}, {GOAL_LON:.7f})  '
            f'distance: {dist:.1f} m'
        )
        self.get_logger().info('Press any key to arm and take off.')

        # ---- key press ----
        fd           = sys.stdin.fileno()
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

        self.get_logger().info('Key received — arming and taking off...')

        # ---- arm ----
        if not self._arm():
            self.get_logger().error('Failed to arm — aborting.')
            return

        # ---- AUTO.TAKEOFF ----
        if not self._takeoff():
            self.get_logger().error('Failed to set AUTO.TAKEOFF — aborting.')
            return

        self.get_logger().info('Climbing to takeoff altitude...')
        while self.state.mode == 'AUTO.TAKEOFF':
            if self._rc_override():
                self.get_logger().info(f'RC override ({self.state.mode}) during climb — handing off.')
                return
            if not self.state.armed:
                self.get_logger().error('Drone disarmed during climb — aborting.')
                return
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info(f'Takeoff complete (now in {self.state.mode}).')

        # ---- switch to OFFBOARD ----
        result = self._switch_offboard()
        if result is None:
            return  # RC override
        if not result:
            self.get_logger().error('Failed to enter OFFBOARD — aborting.')
            return

        # Record altitude target for hold controller.
        if self.pose is not None:
            self.target_z      = self.pose.pose.position.z
            self._alt_integral = 0.0
            self.get_logger().info(f'Altitude hold target: {self.target_z:.2f} m (ENU)')
        else:
            self.get_logger().warn('No pose yet — altitude hold disabled.')

        # ---- face the goal ----
        result = self._align_yaw()
        if result is None:
            return  # RC override
        if not result:
            return  # disarmed

        # ---- fly to goal ----
        result = self._goto_goal()
        if result is None:
            return  # RC override
        # On timeout (result=False) we still land — fall through

        # ---- hover at goal ----
        self.get_logger().info(f'Hovering for {HOVER_TIME:.0f}s...')
        dt = 1.0 / SETPOINT_HZ
        hover_deadline = time.monotonic() + HOVER_TIME
        while time.monotonic() < hover_deadline:
            self._publish_vel(vz=self._vz_hold())
            rclpy.spin_once(self, timeout_sec=dt)
            if self._rc_override():
                self.get_logger().info(f'RC override ({self.state.mode}) during hover — handing off.')
                return

        # ---- land ----
        result = self._land()
        if result is None:
            return  # RC override
        if not result:
            self.get_logger().error('Failed to land — drone left hovering. Disarm manually via RC.')
            return

        self.get_logger().info('AUTO.LAND confirmed. Waiting until disarmed...')
        while self.state.armed:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info('Landed and disarmed.')
        self._set_posctl()


def main():
    print("=" * 56)
    print("GPS goto mission")
    print()
    print(f"Goal            : ({GOAL_LAT:.7f}, {GOAL_LON:.7f})")
    print(f"Cruise speed    : {CRUISE_SPEED} m/s")
    print(f"Arrival radius  : {GOAL_RADIUS} m")
    print(f"Hover at goal   : {HOVER_TIME:.0f}s")
    print(f"Goto timeout    : {GOTO_TIMEOUT:.0f}s")
    print(f"Yaw align       : ±{math.degrees(YAW_ALIGN_THRESH):.0f}° threshold, "
          f"{math.degrees(YAW_RATE_MAX):.0f}°/s max rate")
    print("Takeoff alt     : set via MIS_TAKEOFF_ALT in QGC")
    print("Requires        : GPS lock")
    print("RC override     : switch to ALTCTL or POSCTL at any time")
    print()
    if GOAL_LAT == 0.0 and GOAL_LON == 0.0:
        print("WARNING: GOAL_LAT and GOAL_LON are both 0.0 — edit the file before flying!")
        print()
    print("=" * 56)
    print()

    rclpy.init()
    node = GpsGoto()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
