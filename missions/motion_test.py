"""
Full motion test: arm → AUTO.TAKEOFF → OFFBOARD → run all movement axes → AUTO.LAND.

Velocity setpoints use body frame (base_link): x = forward, y = left, z = up.
Angular z = yaw rate (rad/s, positive = CCW / left turn).
PX4 closes the velocity loop via GPS/EKF2, so GPS lock is required.

Movement sequence (with a brief stop between each segment):
  1. Forward
  2. Backward
  3. Left (strafe)
  4. Right (strafe)
  5. Yaw left
  6. Yaw right
  7. Mix: forward + yaw left
  8. Mix: forward + strafe right

RC override: switching to ALTCTL or POSCTL at any point hands control back
to the RC immediately and the script exits.

After landing the script switches to POSCTL so the next run starts from a
clean mode (PX4 would otherwise revert to OFFBOARD after disarm).

Requires MAVROS running:
  ros2 launch mavros px4.launch fcu_url:=/dev/ttyTHS1:115200
Run:
  python3 motion_test.py
"""

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


SPEED          = 1.0   # m/s — linear speed for all translational segments
YAW_RATE       = 0.5   # rad/s — ~28°/s, so MOVE_TIME gives ~57° of rotation
MOVE_TIME      = 2.0   # seconds per segment
PAUSE_TIME     = 0.5   # seconds of zero-velocity hold between segments
PRESTREAM_TIME = 2.0   # seconds to stream zeros before switching to OFFBOARD
SETPOINT_HZ    = 20    # Hz — must stay >2 Hz or PX4 exits OFFBOARD
ALT_KP         = 0.5   # proportional gain for altitude hold (m/s per metre of error)

# (label, duration, vx, vy, vz, yaw_rate)
# body frame: x=forward, y=left, z=up, yaw_rate positive=CCW
PHASES = [
    ('Forward',              MOVE_TIME,  SPEED,  0.0,   0.0,  0.0     ),
    ('Backward',             MOVE_TIME, -SPEED,  0.0,   0.0,  0.0     ),
    ('Strafe left',          MOVE_TIME,  0.0,    SPEED, 0.0,  0.0     ),
    ('Strafe right',         MOVE_TIME,  0.0,   -SPEED, 0.0,  0.0     ),
    ('Yaw left',             MOVE_TIME,  0.0,    0.0,   0.0,  YAW_RATE),
    ('Yaw right',            MOVE_TIME,  0.0,    0.0,   0.0, -YAW_RATE),
    ('Forward + yaw left',   MOVE_TIME,  SPEED,  0.0,   0.0,  YAW_RATE),
    ('Forward + strafe right', MOVE_TIME, SPEED, -SPEED, 0.0, 0.0     ),
]

RC_OVERRIDE_MODES = {'ALTCTL', 'POSCTL'}

_CMD_INTERVAL     = 2.0
_ARM_TIMEOUT      = 30.0
_TAKEOFF_TIMEOUT  = 30.0
_OFFBOARD_TIMEOUT = 10.0
_LAND_TIMEOUT     = 30.0


class MotionTest(Node):
    def __init__(self):
        super().__init__('motion_test')

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

        self.state        = State()
        self.gps_latitude  = 0.0
        self.gps_longitude = 0.0
        self.pose          = None   # PoseStamped — None until first message
        self.target_z      = None   # altitude (m, ENU) to hold during OFFBOARD
        # Latch: becomes True once we've seen a non-RC mode during the mission.
        # Only after that can a return to POSCTL/ALTCTL be a genuine pilot takeover.
        self._left_rc_modes = False
        self.create_subscription(State,       '/mavros/state',               self._on_state,      state_qos)
        self.create_subscription(StatusText,  '/mavros/statustext',          self._on_statustext, qos)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self._on_pose,       qos)

        self.arm_client  = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode,     '/mavros/set_mode')
        self.land_client = self.create_client(CommandTOL,  '/mavros/cmd/land')

        self.vel_pub = self.create_publisher(
            TwistStamped, '/mavros/setpoint_velocity/cmd_vel', 10
        )

    def _on_state(self, msg: State):
        self.state = msg

    def _on_statustext(self, msg: StatusText):
        self.get_logger().info(f'[PX4] {msg.text}')

    def _on_pose(self, msg: PoseStamped):
        self.pose = msg

    def _rc_override(self):
        if self.state.mode not in RC_OVERRIDE_MODES:
            self._left_rc_modes = True
            return False
        return self._left_rc_modes

    def _vz_hold(self) -> float:
        """Proportional vz correction to maintain target_z.  Returns 0 until
        both pose and target_z are set.  Clamped to ±1 m/s."""
        if self.target_z is None or self.pose is None:
            return 0.0
        err = self.target_z - self.pose.pose.position.z
        return max(-1.0, min(1.0, ALT_KP * err))

    def _publish_vel(self, vx: float = 0.0, vy: float = 0.0, vz: float = 0.0,
                     yaw_rate: float = 0.0):
        """Publish velocity in body frame: x=forward, y=left, z=up, yaw_rate=CCW."""
        msg = TwistStamped()
        msg.header.stamp     = self.get_clock().now().to_msg()
        msg.header.frame_id  = 'base_link'
        msg.twist.linear.x   = vx
        msg.twist.linear.y   = vy
        msg.twist.linear.z   = vz
        msg.twist.angular.z  = yaw_rate
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
            # No RC override check here — the drone may start in POSCTL (left by
            # _set_posctl at the end of the previous run).  The latch won't have
            # fired yet, so _rc_override() is already safe, but skipping the check
            # avoids any edge-case confusion before AUTO.TAKEOFF is confirmed.
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
            self._publish_vel()                          # keep stream alive
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

    def _move(self, label: str, duration: float,
              vx: float = 0.0, vy: float = 0.0, vz: float = 0.0,
              yaw_rate: float = 0.0):
        """
        Run a timed velocity phase.
        Returns True on success, None on RC override, False if disarmed.
        """
        self.get_logger().info(f'{label} for {duration:.1f}s...')
        dt       = 1.0 / SETPOINT_HZ
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            self._publish_vel(vx=vx, vy=vy, vz=vz + self._vz_hold(), yaw_rate=yaw_rate)
            rclpy.spin_once(self, timeout_sec=dt)
            if self._rc_override():
                self.get_logger().info(f'RC override ({self.state.mode}) during {label} — handing off.')
                return None
            if not self.state.armed:
                self.get_logger().error(f'Drone disarmed during {label} — aborting.')
                return False
        return True

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
                req.latitude  = self.gps_latitude
                req.longitude = self.gps_longitude
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
        self._left_rc_modes = False  # reset latch for this run

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

        self.get_logger().info('Connected. Press any key to arm and take off.')

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

        # Record the altitude to hold throughout the motion sequence.
        if self.pose is not None:
            self.target_z = self.pose.pose.position.z
            self.get_logger().info(f'Altitude hold target: {self.target_z:.2f} m (ENU)')
        else:
            self.get_logger().warn('No pose yet — altitude hold disabled.')

        # ---- motion sequence ----
        dt = 1.0 / SETPOINT_HZ
        for label, duration, vx, vy, vz, yaw_rate in PHASES:
            result = self._move(label, duration, vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate)
            if result is None:
                return  # RC override
            if not result:
                return  # disarmed

            # brief stop between segments
            stop_deadline = time.monotonic() + PAUSE_TIME
            while time.monotonic() < stop_deadline:
                self._publish_vel(vz=self._vz_hold())
                rclpy.spin_once(self, timeout_sec=dt)
                if self._rc_override():
                    self.get_logger().info(f'RC override ({self.state.mode}) during pause — handing off.')
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
    total = sum(d for _, d, *_ in PHASES) + PAUSE_TIME * len(PHASES)
    print("=" * 56)
    print("Full motion test — all axes + combos")
    print()
    print(f"Linear speed  : {SPEED} m/s")
    print(f"Yaw rate      : {YAW_RATE} rad/s  (~{YAW_RATE * MOVE_TIME * 57.3:.0f}° per yaw segment)")
    print(f"Segment time  : {MOVE_TIME}s  (~{SPEED * MOVE_TIME:.1f} m per linear segment)")
    print(f"Segments      : {len(PHASES)}  (~{total:.0f}s in OFFBOARD)")
    print("Takeoff alt   : set via MIS_TAKEOFF_ALT in QGC")
    print("Requires      : GPS lock")
    print("RC override   : switch to ALTCTL or POSCTL at any time")
    print()
    for i, (label, duration, vx, vy, vz, yaw_rate) in enumerate(PHASES, 1):
        print(f"  {i}. {label}")
    print("=" * 56)
    print()

    rclpy.init()
    node = MotionTest()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
