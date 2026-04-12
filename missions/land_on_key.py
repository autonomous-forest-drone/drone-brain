"""
Waits for any key press, then commands AUTO.LAND immediately.

Requires MAVROS running:
  ros2 launch mavros px4.launch fcu_url:=/dev/ttyTHS1:921600
Run:
  python3 land_on_key.py
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
from mavros_msgs.msg import State, StatusText
from mavros_msgs.srv import CommandLong
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PoseStamped

_LAND_VERIFY_TIMEOUT = 4.0   # seconds to wait for mode flip after ACK (HEARTBEAT is ~1 Hz)
_EKF_WAIT_TIMEOUT   = 30.0  # seconds to wait for EKF local position before giving up
_GPS_FIX_TIMEOUT = 30.0  # seconds to wait for GPS fix before giving up


class LandOnKey(Node):
    def __init__(self):
        super().__init__('land_on_key')

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.state = State()
        self.gps = NavSatFix()
        self.gps.status.status = -1  # NavSatStatus.STATUS_NO_FIX
        self._local_pos_received_at = 0.0  # wall-clock time of last local_position/pose msg
        self._state_received_at = 0.0      # wall-clock time of last /mavros/state msg
        self.create_subscription(State, '/mavros/state', self._on_state, qos)
        self.create_subscription(NavSatFix, '/mavros/global_position/raw/fix', lambda msg: setattr(self, 'gps', msg), qos)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose',
            lambda msg: setattr(self, '_local_pos_received_at', time.monotonic()), qos)
        self.create_subscription(StatusText, '/mavros/statustext', self._on_statustext, qos)
        self.cmd_client = self.create_client(CommandLong, '/mavros/cmd/command')

    def _on_state(self, msg: State):
        self.state = msg
        self._state_received_at = time.monotonic()

    def _on_statustext(self, msg: StatusText):
        self.get_logger().info(f'[PX4] {msg.text}')

    def _has_fresh_state(self, max_age: float = 1.5) -> bool:
        return self._state_received_at > 0 and time.monotonic() - self._state_received_at < max_age

    def _has_gps_fix(self):
        # NavSatStatus: -1=no fix, 0=fix, 1=SBAS, 2=GBAS
        return self.gps.status.status >= 0

    def _has_local_position(self):
        # True if we received a local_position/pose message within the last 2s.
        # ALTCTL flies on baro alone, but AUTO.LAND requires EKF local_position_valid.
        return (self._local_pos_received_at > 0 and
                time.monotonic() - self._local_pos_received_at < 2.0)

    def _wait_for_local_position(self):
        if self._has_local_position():
            return True
        self.get_logger().warn(
            'EKF local position not yet valid — AUTO.LAND requires it. '
            f'Waiting up to {_EKF_WAIT_TIMEOUT:.0f}s...'
        )
        deadline = time.monotonic() + _EKF_WAIT_TIMEOUT
        last_log = time.monotonic()
        while not self._has_local_position() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.monotonic() - last_log >= 5.0:
                elapsed = _EKF_WAIT_TIMEOUT - (deadline - time.monotonic())
                self.get_logger().warn(f'Still waiting for EKF local position... ({elapsed:.0f}s elapsed)')
                last_log = time.monotonic()
        if self._has_local_position():
            self.get_logger().info('EKF local position valid.')
            return True
        self.get_logger().error(
            f'EKF local position not valid after {_EKF_WAIT_TIMEOUT:.0f}s — '
            'AUTO.LAND will be rejected by PX4. Land manually via RC.'
        )
        return False

    def _wait_for_gps(self):
        if self._has_gps_fix():
            return True
        self.get_logger().warn(
            'GPS has no fix yet — AUTO.LAND requires a position estimate. '
            f'Waiting up to {_GPS_FIX_TIMEOUT:.0f}s...'
        )
        deadline = self.get_clock().now().nanoseconds + int(_GPS_FIX_TIMEOUT * 1e9)
        last_log = self.get_clock().now().nanoseconds
        while not self._has_gps_fix() and self.get_clock().now().nanoseconds < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            now = self.get_clock().now().nanoseconds
            if now - last_log >= int(5.0 * 1e9):
                elapsed = (now - (deadline - int(_GPS_FIX_TIMEOUT * 1e9))) / 1e9
                self.get_logger().warn(f'Still waiting for GPS fix... ({elapsed:.0f}s elapsed)')
                last_log = now
        if self._has_gps_fix():
            self.get_logger().info('GPS fix acquired.')
            return True
        self.get_logger().error(
            f'No GPS fix after {_GPS_FIX_TIMEOUT:.0f}s — AUTO.LAND will likely be rejected by PX4. '
            'Land manually via RC.'
        )
        return False

    def _land(self):
        self.get_logger().info(
            f'Pre-land state: mode={self.state.mode} armed={self.state.armed} '
            f'gps_fix={"YES" if self._has_gps_fix() else "NO"} '
            f'ekf_local_pos={"YES" if self._has_local_position() else "NO"}'
        )
        if not self._has_fresh_state():
            self.get_logger().error(
                'State message is stale (>1.5s since last update) — '
                'FCU may have disconnected. Aborting land.'
            )
            return False
        if not self._wait_for_gps():
            return False
        if not self._wait_for_local_position():
            return False
        self.cmd_client.wait_for_service()
        req = CommandLong.Request()
        req.command = 176   # MAV_CMD_DO_SET_MODE
        req.param1  = float(0x01 | (0x80 if self.state.armed else 0))  # base_mode
        req.param2  = 4.0   # PX4_CUSTOM_MAIN_MODE_AUTO
        req.param3  = 6.0   # PX4_CUSTOM_SUB_MODE_AUTO_LAND
        future = self.cmd_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        result = future.result() if future.done() else None
        if result and result.success:
            self.get_logger().info('MAV_CMD_DO_SET_MODE: ACK received')
        else:
            self.get_logger().warn('MAV_CMD_DO_SET_MODE: no ACK — checking mode anyway')
        # mode_sent is unreliable — the mode flip is the authoritative check.
        # HEARTBEAT publishes at ~1 Hz so wait long enough to see several cycles.
        deadline = self.get_clock().now().nanoseconds + int(_LAND_VERIFY_TIMEOUT * 1e9)
        while self.state.mode != 'AUTO.LAND' and self.get_clock().now().nanoseconds < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.state.mode == 'AUTO.LAND':
            self.get_logger().info('SET_MODE AUTO.LAND: confirmed')
            return True
        return False

    def run(self):
        # Spin briefly to see if MAVROS is already connected. DDS service
        # discovery lingers after a process dies, so wait_for_service() gives
        # false positives — the connected flag is the only reliable health check.
        deadline = time.monotonic() + 2.0
        while not self.state.connected and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

        if not self.state.connected:
            self.get_logger().info('MAVROS not connected — launching...')
            subprocess.Popen(
                ['ros2', 'launch', 'mavros', 'px4.launch', 'fcu_url:=/dev/ttyTHS1:921600'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            self.get_logger().info('MAVROS already connected — reusing.')

        self.get_logger().info('Waiting for MAVROS connection...')
        while not self.state.connected:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info('Connected. Press any key to land.')

        # Put stdin in raw mode so we get keypresses immediately
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

        self.get_logger().info('Key received — switching to AUTO.LAND...')
        if not self._land():
            self.get_logger().error('Failed to set AUTO.LAND.')
            return

        # Wait for the mode to reflect AUTO.LAND
        deadline = self.get_clock().now().nanoseconds + int(5.0 * 1e9)
        while self.state.mode != 'AUTO.LAND' and self.get_clock().now().nanoseconds < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.state.mode != 'AUTO.LAND':
            self.get_logger().error(f'Mode did not switch — still in {self.state.mode}')
            return

        self.get_logger().info('AUTO.LAND confirmed. Waiting until disarmed...')
        while self.state.armed:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info('Landed and disarmed.')


def main():
    print("=" * 40)
    print("Land on key")
    print()
    print("=" * 40)
    print()

    rclpy.init()
    node = LandOnKey()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
