"""
Waits for any key press, then arms and commands AUTO.TAKEOFF. PX4 climbs to
the altitude configured in QGroundControl (MIS_TAKEOFF_ALT). Once the takeoff
is complete (mode transitions away from AUTO.TAKEOFF), the script holds and
listens for the next flight mode change from a remote command, then exits.

Requires MAVROS running:
  ros2 launch mavros px4.launch fcu_url:=/dev/ttyTHS1:115200
Run:
  python3 takeoff_on_key.py
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
from mavros_msgs.srv import CommandBool, CommandTOL


_CMD_INTERVAL  = 2.0   # seconds between command retries
_ARM_TIMEOUT   = 30.0  # seconds to wait for arm confirmation
_TAKEOFF_TIMEOUT = 30.0  # seconds to wait for AUTO.TAKEOFF mode confirmation


class TakeoffOnKey(Node):
    def __init__(self):
        super().__init__('takeoff_on_key')

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
        self.create_subscription(State, '/mavros/state', self._on_state, state_qos)
        self.create_subscription(StatusText, '/mavros/statustext', self._on_statustext, qos)
        self.arm_client    = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.takeoff_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

    def _on_state(self, msg: State):
        self.state = msg

    def _on_statustext(self, msg: StatusText):
        self.get_logger().info(f'[PX4] {msg.text}')

    def _arm(self):
        self.get_logger().info('Arming — will retry until confirmed...')
        self.arm_client.wait_for_service()
        deadline = time.monotonic() + _ARM_TIMEOUT
        last_send = 0.0
        while time.monotonic() < deadline:
            if time.monotonic() - last_send >= _CMD_INTERVAL:
                req = CommandBool.Request()
                req.value = True
                future = self.arm_client.call_async(req)
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
        self.takeoff_client.wait_for_service()
        deadline = time.monotonic() + _TAKEOFF_TIMEOUT
        last_send = 0.0
        while time.monotonic() < deadline:
            if time.monotonic() - last_send >= _CMD_INTERVAL:
                req = CommandTOL.Request()
                req.min_pitch = 0.0
                req.yaw = 0.0
                req.latitude = 0.0
                req.longitude = 0.0
                req.altitude = 0.0  # PX4 uses MIS_TAKEOFF_ALT from QGC
                future = self.takeoff_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
                result = future.result() if future.done() else None
                if result and result.success:
                    self.get_logger().info('cmd/takeoff: ACK success')
                else:
                    self.get_logger().warn('cmd/takeoff: no ACK — retrying...')
                last_send = time.monotonic()
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.state.mode == 'AUTO.TAKEOFF':
                self.get_logger().info('AUTO.TAKEOFF mode confirmed.')
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

        self.get_logger().info('Key received — arming and taking off...')

        if not self._arm():
            self.get_logger().error('Failed to arm — aborting.')
            return

        if not self._takeoff():
            self.get_logger().error('Failed to set AUTO.TAKEOFF — aborting.')
            return

        self.get_logger().info('Climbing to takeoff altitude...')
        while self.state.mode == 'AUTO.TAKEOFF':
            rclpy.spin_once(self, timeout_sec=0.1)

        hover_mode = self.state.mode
        self.get_logger().info(
            f'Takeoff complete — hovering in "{hover_mode}". '
            'Listening for remote flight mode change...'
        )

        while self.state.mode == hover_mode:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info(
            f'Flight mode changed to "{self.state.mode}" — handing off control.'
        )


def main():
    print("=" * 40)
    print("Takeoff on key")
    print()
    print("Takeoff altitude is set via MIS_TAKEOFF_ALT in QGroundControl.")
    print("=" * 40)
    print()

    rclpy.init()
    node = TakeoffOnKey()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
