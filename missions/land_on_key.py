"""
Waits for any key press, then commands AUTO.LAND immediately.

Requires MAVROS running:
  ros2 launch mavros px4.launch fcu_url:=/dev/ttyTHS1:115200
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
from mavros_msgs.srv import CommandTOL, SetMode
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PoseStamped

_LAND_VERIFY_TIMEOUT = 10.0  # seconds to wait for mode flip after AUTO.LAND command
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
        state_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.state = State()
        self.gps = NavSatFix()
        self.gps.status.status = -1  # NavSatStatus.STATUS_NO_FIX
        self._local_pos_received_at = 0.0  # wall-clock time of last local_position/pose msg
        self._state_received_at = 0.0      # wall-clock time of last /mavros/state msg
        self.create_subscription(State, '/mavros/state', self._on_state, state_qos)
        self.create_subscription(NavSatFix, '/mavros/global_position/raw/fix', lambda msg: setattr(self, 'gps', msg), qos)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose',
            lambda msg: setattr(self, '_local_pos_received_at', time.monotonic()), qos)
        self.create_subscription(StatusText, '/mavros/statustext', self._on_statustext, qos)
        self.land_client = self.create_client(CommandTOL, '/mavros/cmd/land')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')

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
        if not self._wait_for_gps():
            return False
        if not self._wait_for_local_position():
            return False

        # Use cmd/land (MAV_CMD_NAV_LAND) — same as QGC's Land button.
        # This is a navigation command and is not overridden by the RC mode switch,
        # unlike SET_MODE which channel-5 ALTCTL immediately cancels.
        self.get_logger().info('Sending MAV_CMD_NAV_LAND via /mavros/cmd/land ...')
        self.land_client.wait_for_service()
        req = CommandTOL.Request()
        req.min_pitch = 0.0
        req.yaw = float('nan')
        req.latitude = self.gps.latitude
        req.longitude = self.gps.longitude
        req.altitude = 0.0
        future = self.land_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        result = future.result() if future.done() else None
        if result and result.success:
            self.get_logger().info('cmd/land: ACK success')
        else:
            # PX4 often routes COMMAND_ACK to TELEM1 (QGC) instead of TELEM2
            # (MAVROS). The command may still execute — fall through and check
            # the mode rather than bailing out here.
            self.get_logger().warn(
                'cmd/land: no ACK or result=False — '
                'PX4 may have routed ACK to QGC. Checking mode anyway...'
            )

        # HEARTBEAT is ~1 Hz; wait long enough to see several cycles.
        deadline = time.monotonic() + _LAND_VERIFY_TIMEOUT
        while self.state.mode != 'AUTO.LAND' and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.state.mode == 'AUTO.LAND':
            self.get_logger().info('AUTO.LAND mode confirmed.')
            return True

        # Fallback: try SET_MODE in case cmd/land was rejected.
        self.get_logger().warn(
            'cmd/land did not result in AUTO.LAND — falling back to SET_MODE...'
        )
        self.mode_client.wait_for_service()
        mode_req = SetMode.Request()
        mode_req.custom_mode = 'AUTO.LAND'
        mode_future = self.mode_client.call_async(mode_req)
        rclpy.spin_until_future_complete(self, mode_future, timeout_sec=3.0)
        mode_result = mode_future.result() if mode_future.done() else None
        if mode_result and mode_result.mode_sent:
            self.get_logger().info('SET_MODE AUTO.LAND: ACK received')
        else:
            self.get_logger().warn('SET_MODE AUTO.LAND: no ACK — checking mode anyway')
        deadline = time.monotonic() + _LAND_VERIFY_TIMEOUT
        while self.state.mode != 'AUTO.LAND' and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.state.mode == 'AUTO.LAND':
            self.get_logger().info('AUTO.LAND mode confirmed (via SET_MODE fallback).')
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
                ['ros2', 'launch', 'mavros', 'px4.launch', 'fcu_url:=/dev/ttyTHS1:115200'],
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
            self.get_logger().error('Failed to set AUTO.LAND — disarm manually via RC.')
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
