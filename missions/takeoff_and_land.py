"""
Waits for any key press, then arms and commands AUTO.TAKEOFF. PX4 climbs to
the altitude configured in QGroundControl (MIS_TAKEOFF_ALT). Once the takeoff
is complete the script immediately commands AUTO.LAND and waits until disarmed.

RC override: switching to ALTCTL or POSCTL at any point during the mission
hands control back to the RC immediately and the script exits.

Requires MAVROS running:
  ros2 launch mavros px4.launch fcu_url:=/dev/ttyTHS1:921600
Run:
  python3 takeoff_and_land.py
"""

import select
import subprocess
import sys
import termios
import tty
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode


RC_OVERRIDE_MODES = {'ALTCTL', 'POSCTL'}

_RETRIES = 5
_RETRY_DELAY = 1.0  # seconds


class TakeoffAndLand(Node):
    def __init__(self):
        super().__init__('takeoff_and_land')

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.state = State()
        self.create_subscription(State, '/mavros/state', lambda msg: setattr(self, 'state', msg), qos)
        self.arm_client  = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode,     '/mavros/set_mode')

    def _spin_for(self, seconds):
        deadline = self.get_clock().now().nanoseconds + int(seconds * 1e9)
        while self.get_clock().now().nanoseconds < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

    def _set_mode(self, mode):
        self.mode_client.wait_for_service()
        req = SetMode.Request()
        req.custom_mode = mode
        for attempt in range(1, _RETRIES + 1):
            future = self.mode_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            if not (future.result() and future.result().mode_sent):
                self.get_logger().warn(
                    f'SET_MODE {mode}: attempt {attempt}/{_RETRIES} rejected'
                    + (' — retrying' if attempt < _RETRIES else '')
                )
            else:
                # mode_sent is an ACK only — verify the mode actually flipped
                self._spin_for(_RETRY_DELAY)
                if self.state.mode == mode:
                    self.get_logger().info(f'SET_MODE {mode}: confirmed (attempt {attempt})')
                    return True
                self.get_logger().warn(
                    f'SET_MODE {mode}: ACKed but mode did not flip (attempt {attempt}/{_RETRIES})'
                    + (' — retrying' if attempt < _RETRIES else '')
                )
                continue  # delay already consumed above
            self._spin_for(_RETRY_DELAY)
        return False

    def _arm(self):
        self.arm_client.wait_for_service()
        req = CommandBool.Request()
        req.value = True
        for attempt in range(1, _RETRIES + 1):
            future = self.arm_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            if not (future.result() and future.result().success):
                self.get_logger().warn(
                    f'ARM: attempt {attempt}/{_RETRIES} rejected'
                    + (' — retrying' if attempt < _RETRIES else '')
                )
            else:
                self._spin_for(_RETRY_DELAY)
                if self.state.armed:
                    self.get_logger().info(f'ARM: confirmed (attempt {attempt})')
                    return True
                self.get_logger().warn(
                    f'ARM: ACKed but not armed yet (attempt {attempt}/{_RETRIES})'
                    + (' — retrying' if attempt < _RETRIES else '')
                )
                continue
            self._spin_for(_RETRY_DELAY)
        return False

    def _rc_override(self):
        """Returns True if the RC has switched to an override mode."""
        return self.state.mode in RC_OVERRIDE_MODES

    def run(self):
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

        self.get_logger().info('Key received — arming and switching to AUTO.TAKEOFF...')
        if not self._arm():
            self.get_logger().error('Failed to arm after all retries — aborting.')
            return
        if not self._set_mode('AUTO.TAKEOFF'):
            self.get_logger().error('Failed to set AUTO.TAKEOFF after all retries — aborting.')
            return

        # Wait for takeoff to complete, bail out on RC override
        self.get_logger().info('Climbing to takeoff altitude...')
        while self.state.mode == 'AUTO.TAKEOFF':
            if self._rc_override():
                self.get_logger().info(f'RC override ({self.state.mode}) during climb — handing off.')
                return
            rclpy.spin_once(self, timeout_sec=0.1)

        # AUTO.TAKEOFF transitions to hover momentarily — go straight to landing
        self.get_logger().info('Takeoff complete. Switching to AUTO.LAND...')
        if not self._set_mode('AUTO.LAND'):
            self.get_logger().error('Failed to set AUTO.LAND after all retries — drone left hovering.')
            return

        # Wait until disarmed, still watching for RC override
        while self.state.armed:
            if self._rc_override():
                self.get_logger().info(f'RC override ({self.state.mode}) during landing — handing off.')
                return
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info('Landed and disarmed.')


def main():
    print("=" * 40)
    print("Takeoff and land")
    print()
    print("Starting MAVROS...")
    subprocess.Popen(
        ['ros2', 'launch', 'mavros', 'px4.launch', 'fcu_url:=/dev/ttyTHS1:921600'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("Press any key to arm and trigger AUTO.TAKEOFF.")
    print("Takeoff altitude is set via MIS_TAKEOFF_ALT in QGroundControl.")
    print("The drone will AUTO.LAND as soon as it reaches altitude.")
    print("Switch to ALTCTL or POSCTL at any time for immediate RC override.")
    print("=" * 40)
    print()

    rclpy.init()
    node = TakeoffAndLand()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
