"""
Arm → hover at 1.5m for 3s → land → disarm

Requires MAVROS running:
  ros2 launch mavros px4.launch fcu_url:=/dev/ttyTHS1

Run:
  python3 hover_test.py
"""

import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode


HOVER_ALTITUDE = 1.5   # metres
HOVER_SECONDS  = 3.0


class HoverTest(Node):
    def __init__(self):
        super().__init__('hover_test')

        qos_state = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.state = State()
        self.create_subscription(State, '/mavros/state', self._state_cb, qos_state)

        self.pose_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)

        self.arm_client  = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode,     '/mavros/set_mode')

    # ------------------------------------------------------------------ helpers

    def _state_cb(self, msg):
        self.state = msg

    def _publish_setpoint(self, z):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = 0.0
        msg.pose.position.y = 0.0
        msg.pose.position.z = z
        self.pose_pub.publish(msg)

    def _set_mode(self, mode):
        self.mode_client.wait_for_service()
        req = SetMode.Request()
        req.custom_mode = mode
        future = self.mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        ok = future.result() and future.result().mode_sent
        self.get_logger().info(f'Set mode {mode}: {"OK" if ok else "FAILED"}')
        return ok

    def _arm(self, value):
        self.arm_client.wait_for_service()
        req = CommandBool.Request()
        req.value = value
        future = self.arm_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        ok = future.result() and future.result().success
        label = 'ARMED' if value else 'DISARMED'
        self.get_logger().info(f'{label}: {"OK" if ok else "FAILED"}')
        return ok

    # ------------------------------------------------------------------ main sequence

    def run(self):
        self.get_logger().info('Waiting for MAVROS connection...')
        while not self.state.connected:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info('Connected. Pre-streaming setpoints...')
        # PX4 requires setpoints to be streaming BEFORE switching to OFFBOARD
        for _ in range(50):
            self._publish_setpoint(HOVER_ALTITUDE)
            time.sleep(0.05)

        self.get_logger().info('Switching to OFFBOARD mode...')
        self._set_mode('OFFBOARD')
        time.sleep(0.5)

        self.get_logger().info('Arming...')
        self._arm(True)
        time.sleep(0.5)

        # Keep publishing setpoint during ascent + hover
        self.get_logger().info(f'Climbing to {HOVER_ALTITUDE}m...')
        ascent_end = time.time() + 4.0          # allow 4s to reach altitude
        while time.time() < ascent_end:
            self._publish_setpoint(HOVER_ALTITUDE)
            rclpy.spin_once(self, timeout_sec=0.05)

        self.get_logger().info(f'Hovering for {HOVER_SECONDS}s...')
        hover_end = time.time() + HOVER_SECONDS
        while time.time() < hover_end:
            self._publish_setpoint(HOVER_ALTITUDE)
            rclpy.spin_once(self, timeout_sec=0.05)

        self.get_logger().info('Switching to AUTO.LAND...')
        self._set_mode('AUTO.LAND')

        # Wait for disarm (PX4 disarms automatically after landing)
        self.get_logger().info('Waiting for auto-disarm...')
        timeout = time.time() + 15.0
        while time.time() < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if not self.state.armed:
                break

        if self.state.armed:
            self.get_logger().warn('Auto-disarm timed out — disarming manually.')
            self._arm(False)

        self.get_logger().info('Done.')


def main():
    rclpy.init()
    node = HoverTest()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted — switching to LAND.')
        node._set_mode('AUTO.LAND')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
