"""
Waits for any key press, then arms and commands AUTO.TAKEOFF. PX4 climbs to
the altitude configured in QGroundControl (MIS_TAKEOFF_ALT). The drone hovers
for HOVER_TIME seconds, then commands AUTO.LAND and waits until disarmed.

RC override: switching to ALTCTL or POSCTL at any point hands control back
to the RC immediately and the script exits.

Requires MAVROS running:
  ros2 launch mavros px4.launch fcu_url:=/dev/ttyTHS1:115200
Run:
  python3 takeoff_and_land.py
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
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode


HOVER_TIME       = 5.0   # seconds to hover between takeoff and land
_CMD_INTERVAL    = 2.0   # seconds between command retries
_ARM_TIMEOUT     = 30.0  # seconds to wait for arm confirmation
_TAKEOFF_TIMEOUT = 30.0  # seconds to wait for AUTO.TAKEOFF mode confirmation
_LAND_TIMEOUT    = 30.0  # seconds to wait for AUTO.LAND mode confirmation

RC_OVERRIDE_MODES = {'ALTCTL', 'POSCTL'}


class TakeoffAndLand(Node):
    def __init__(self):
        super().__init__('takeoff_and_land')

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
        self.gps_latitude  = 0.0
        self.gps_longitude = 0.0
        self.create_subscription(State, '/mavros/state', self._on_state, state_qos)
        self.create_subscription(StatusText, '/mavros/statustext', self._on_statustext, qos)
        self.arm_client  = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode,     '/mavros/set_mode')
        self.land_client = self.create_client(CommandTOL,  '/mavros/cmd/land')

    def _on_state(self, msg: State):
        self.state = msg

    def _on_statustext(self, msg: StatusText):
        self.get_logger().info(f'[PX4] {msg.text}')

    def _rc_override(self):
        return self.state.mode in RC_OVERRIDE_MODES

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
        self.mode_client.wait_for_service()
        deadline = time.monotonic() + _TAKEOFF_TIMEOUT
        last_send = 0.0
        while time.monotonic() < deadline:
            if time.monotonic() - last_send >= _CMD_INTERVAL:
                req = SetMode.Request()
                req.custom_mode = 'AUTO.TAKEOFF'
                future = self.mode_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
                result = future.result() if future.done() else None
                if result and result.mode_sent:
                    self.get_logger().info('SET_MODE AUTO.TAKEOFF: ACK success')
                else:
                    self.get_logger().warn('SET_MODE AUTO.TAKEOFF: no ACK — retrying...')
                last_send = time.monotonic()
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._rc_override():
                self.get_logger().info(f'RC override ({self.state.mode}) — handing off.')
                return None
            if self.state.mode == 'AUTO.TAKEOFF':
                self.get_logger().info('AUTO.TAKEOFF mode confirmed.')
                return True
            if not self.state.armed:
                self.get_logger().error('Drone disarmed before takeoff confirmed — aborting.')
                return False
        return False

    def _land(self):
        self.get_logger().info('Commanding AUTO.LAND — will retry until mode confirms...')
        self.land_client.wait_for_service()
        deadline = time.monotonic() + _LAND_TIMEOUT
        last_send = 0.0
        while time.monotonic() < deadline:
            if time.monotonic() - last_send >= _CMD_INTERVAL:
                req = CommandTOL.Request()
                req.min_pitch = 0.0
                req.yaw = float('nan')
                req.latitude = self.gps_latitude
                req.longitude = self.gps_longitude
                req.altitude = 0.0
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
                self.get_logger().info(f'RC override ({self.state.mode}) — handing off.')
                return None
            if self.state.mode == 'AUTO.LAND':
                self.get_logger().info('AUTO.LAND mode confirmed.')
                return True
            if not self.state.armed:
                self.get_logger().info('Drone disarmed — already landed.')
                return True
        return False

    def run(self):
        # Spin briefly to see if MAVROS is already connected.
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

        result = self._takeoff()
        if result is None:
            return  # RC override
        if not result:
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

        self.get_logger().info(f'Takeoff complete. Hovering for {HOVER_TIME:.0f}s...')
        hover_deadline = time.monotonic() + HOVER_TIME
        while time.monotonic() < hover_deadline:
            if self._rc_override():
                self.get_logger().info(f'RC override ({self.state.mode}) during hover — handing off.')
                return
            rclpy.spin_once(self, timeout_sec=0.1)

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


def main():
    print("=" * 40)
    print("Takeoff and land")
    print()
    print(f"Hover time: {HOVER_TIME:.0f}s")
    print("Takeoff altitude is set via MIS_TAKEOFF_ALT in QGroundControl.")
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
