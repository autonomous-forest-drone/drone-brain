"""
Waits for any key press, then commands AUTO.LAND immediately.

Triggers landing by overriding RC channel 5 to the land position (PWM 2000),
which is the same as physically flicking the land switch on the transmitter.
The override is released once the drone has disarmed.

Requires MAVROS running:
  ros2 launch mavros px4.launch fcu_url:=/dev/ttyTHS1:115200
Run:
  python3 land_on_key_v2.py
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
from mavros_msgs.msg import State, StatusText, OverrideRCIn
from mavros_msgs.srv import SetMode


_LAND_CHANNEL   = 5     # RC channel that controls land mode (1-indexed)
_LAND_PWM       = 2000  # PWM value that maps to AUTO.LAND on that channel
_RELEASE_PWM    = 0     # 0 = release override, hands control back to transmitter


class LandOnKey(Node):
    def __init__(self):
        super().__init__('land_on_key')

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.state = State()
        self.create_subscription(State, '/mavros/state', self._on_state, qos)
        self.create_subscription(StatusText, '/mavros/statustext', self._on_statustext, qos)
        reliable_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.rc_override_pub = self.create_publisher(OverrideRCIn, '/mavros/rc/override', reliable_qos)
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')

    def _on_statustext(self, msg: StatusText):
        self.get_logger().info(f'[PX4] {msg.text}')

    def _on_state(self, msg: State):
        prev = self.state
        self.state = msg
        if msg.mode != prev.mode or msg.armed != prev.armed:
            armed_str = 'ARMED' if msg.armed else 'DISARMED'
            self.get_logger().info(f'State: {msg.mode} / {armed_str}')

    def _send_rc_override(self, pwm: int):
        msg = OverrideRCIn()
        # 0 = don't touch this channel, only override channel 5
        msg.channels = [0] * 18
        msg.channels[_LAND_CHANNEL - 1] = pwm  # convert to 0-indexed
        self.rc_override_pub.publish(msg)

    def _release_rc_override(self):
        self._send_rc_override(_RELEASE_PWM)

    def _land(self):
        self.get_logger().info(
            f'Overriding RC ch{_LAND_CHANNEL} to {_LAND_PWM} (land position)'
        )
        # Fire SET_MODE as well — now that MAV_1_MODE=Normal this may work directly
        req = SetMode.Request()
        req.custom_mode = 'AUTO.LAND'
        req.base_mode = 0x01 | (0x80 if self.state.armed else 0)
        if self.mode_client.wait_for_service(timeout_sec=2.0):
            self.mode_client.call_async(req)

        # RC_CHANNELS_OVERRIDE times out in ~0.5s — keep sending at 10 Hz
        # until PX4 confirms the mode flip.
        deadline = time.monotonic() + 3.0
        while self.state.mode != 'AUTO.LAND' and time.monotonic() < deadline:
            self._send_rc_override(_LAND_PWM)
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.state.mode == 'AUTO.LAND':
            self.get_logger().info('AUTO.LAND confirmed.')
            return True

        self.get_logger().error(f'Mode did not flip — still in {self.state.mode}')
        self._release_rc_override()
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

        self.get_logger().info('Waiting until disarmed...')
        while self.state.armed:
            rclpy.spin_once(self, timeout_sec=0.1)

        self._release_rc_override()
        self.get_logger().info('Landed and disarmed. RC override released.')


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
        node._release_rc_override()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
