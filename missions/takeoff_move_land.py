"""
Key press → arm → AUTO.TAKEOFF → OFFBOARD (move forward 1 m/s for 1 s) → AUTO.LAND.

Velocity setpoints use body frame (base_link): x = forward, y = left, z = up.
PX4 closes the velocity loop via GPS/EKF2, so GPS lock is required.

Requires MAVROS running:
  ros2 launch mavros px4.launch fcu_url:=/dev/ttyTHS1:115200
Run:
  python3 takeoff_move_land.py
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
from geometry_msgs.msg import TwistStamped
from mavros_msgs.msg import State, StatusText
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode


FORWARD_SPEED   = 1.0   # m/s — body x (forward)
MOVE_TIME       = 2.0   # seconds → ~1 m
PRESTREAM_TIME  = 2.0   # seconds to stream zeros before switching to OFFBOARD
SETPOINT_HZ     = 20    # Hz — must stay >2 Hz or PX4 exits OFFBOARD

_CMD_INTERVAL    = 2.0
_ARM_TIMEOUT     = 30.0
_TAKEOFF_TIMEOUT = 30.0
_OFFBOARD_TIMEOUT = 10.0
_LAND_TIMEOUT    = 30.0


class TakeoffMoveLand(Node):
    def __init__(self):
        super().__init__('takeoff_move_land')

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

        self.state         = State()
        self.gps_latitude  = 0.0
        self.gps_longitude = 0.0
        self.create_subscription(State,      '/mavros/state',      self._on_state,      state_qos)
        self.create_subscription(StatusText, '/mavros/statustext', self._on_statustext, qos)

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

    def _publish_vel(self, vx: float = 0.0, vy: float = 0.0, vz: float = 0.0):
        """Publish velocity in body frame: x=forward, y=left, z=up."""
        msg = TwistStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x  = vx
        msg.twist.linear.y  = vy
        msg.twist.linear.z  = vz
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
            if self.state.mode == 'OFFBOARD':
                self.get_logger().info('OFFBOARD confirmed.')
                return True
        return False

    def _land(self):
        """Switch to AUTO.LAND while keeping the velocity stream alive."""
        self.get_logger().info('Commanding AUTO.LAND — will retry until mode confirms...')
        self.mode_client.wait_for_service()
        dt        = 1.0 / SETPOINT_HZ
        deadline  = time.monotonic() + _LAND_TIMEOUT
        last_send = 0.0
        while time.monotonic() < deadline:
            self._publish_vel()                          # keep OFFBOARD alive during switch
            if time.monotonic() - last_send >= _CMD_INTERVAL:
                req             = SetMode.Request()
                req.custom_mode = 'AUTO.LAND'
                future          = self.mode_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
                result = future.result() if future.done() else None
                if result and result.mode_sent:
                    self.get_logger().info('SET_MODE AUTO.LAND: ACK success')
                else:
                    self.get_logger().warn('SET_MODE AUTO.LAND: no ACK — retrying...')
                last_send = time.monotonic()
            rclpy.spin_once(self, timeout_sec=dt)
            if self.state.mode == 'AUTO.LAND':
                self.get_logger().info('AUTO.LAND mode confirmed.')
                return True
            if not self.state.armed:
                self.get_logger().info('Drone disarmed — already landed.')
                return True
        return False

    # ------------------------------------------------------------------ mission

    def run(self):
        dt = 1.0 / SETPOINT_HZ

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
            if not self.state.armed:
                self.get_logger().error('Drone disarmed during climb — aborting.')
                return
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info(f'Takeoff complete (now in {self.state.mode}).')

        # ---- switch to OFFBOARD ----
        if not self._switch_offboard():
            self.get_logger().error('Failed to enter OFFBOARD — aborting.')
            return

        # ---- move forward ----
        self.get_logger().info(
            f'Moving forward at {FORWARD_SPEED} m/s for {MOVE_TIME} s...'
        )
        deadline = time.monotonic() + MOVE_TIME
        while time.monotonic() < deadline:
            self._publish_vel(vx=FORWARD_SPEED)
            rclpy.spin_once(self, timeout_sec=dt)
            if not self.state.armed:
                self.get_logger().error('Drone disarmed during forward move — aborting.')
                return

        # ---- stop (brief hold before land) ----
        self.get_logger().info('Stopping...')
        stop_deadline = time.monotonic() + 0.5
        while time.monotonic() < stop_deadline:
            self._publish_vel()
            rclpy.spin_once(self, timeout_sec=dt)

        # ---- land ----
        if not self._land():
            self.get_logger().error('Failed to land — drone left hovering. Disarm manually via RC.')
            return

        self.get_logger().info('AUTO.LAND confirmed. Waiting until disarmed...')
        while self.state.armed:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info('Landed and disarmed.')


def main():
    print("=" * 50)
    print("Takeoff → move forward 1 m → land")
    print()
    print(f"Forward speed : {FORWARD_SPEED} m/s")
    print(f"Move duration : {MOVE_TIME} s  (~{FORWARD_SPEED * MOVE_TIME:.1f} m)")
    print("Takeoff altitude : set via MIS_TAKEOFF_ALT in QGC")
    print("Requires        : GPS lock")
    print("=" * 50)
    print()

    rclpy.init()
    node = TakeoffMoveLand()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
