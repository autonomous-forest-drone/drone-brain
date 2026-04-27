import sys
import time
import rclpy
from rclpy.node import Node
from mavros_msgs.msg import ActuatorControl
from mavros_msgs.srv import CommandBool, SetMode


class MotorController(Node):
    def __init__(self):
        super().__init__('motor_controller')

        self.pub = self.create_publisher(ActuatorControl, '/mavros/actuator_control', 10)

        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')

        self.get_logger().info('Motor controller ready.')

    def set_mode(self, mode: str) -> bool:
        if not self.mode_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error('set_mode service not available')
            return False
        req = SetMode.Request()
        req.custom_mode = mode
        future = self.mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() and future.result().mode_sent:
            self.get_logger().info(f'Mode set to {mode}')
            return True
        self.get_logger().error(f'Failed to set mode {mode}')
        return False

    def arm(self, do_arm: bool) -> bool:
        if not self.arm_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error('arming service not available')
            return False
        req = CommandBool.Request()
        req.value = do_arm
        future = self.arm_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() and future.result().success:
            state = 'ARMED' if do_arm else 'DISARMED'
            self.get_logger().info(f'Vehicle {state}')
            return True
        self.get_logger().error('Arming/disarming failed')
        return False

    def send_motors(self, throttles: list[float]):
        """
        throttles: list of up to 8 values in range [0.0, 1.0]
                   index 0 = motor 1, index 1 = motor 2, etc.
        Uses group_mix=3 (direct motor control, bypasses flight controller mixing).
        """
        controls = [0.0] * 8
        for i, val in enumerate(throttles[:8]):
            controls[i] = max(0.0, min(1.0, val))

        msg = ActuatorControl()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.group_mix = 3          # group 3 = direct motor control
        msg.controls = controls
        self.pub.publish(msg)

    def stop_all(self):
        self.send_motors([0.0] * 8)
        self.get_logger().info('All motors stopped.')


def print_usage():
    print(
        '\nUsage:\n'
        '  python3 motor_control.py manual         — switch to MANUAL mode\n'
        '  python3 motor_control.py arm           — arm the vehicle\n'
        '  python3 motor_control.py disarm        — disarm the vehicle\n'
        '  python3 motor_control.py spin <throttle>        — spin all motors at throttle (0.0–1.0)\n'
        '  python3 motor_control.py motor <index> <throttle> — spin one motor (index 0–7)\n'
        '  python3 motor_control.py stop          — set all motors to 0\n'
    )


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    cmd = sys.argv[1]

    rclpy.init()
    node = MotorController()

    try:
        if cmd == 'manual':
            node.set_mode('MANUAL')

        elif cmd == 'arm':
            node.arm(True)

        elif cmd == 'disarm':
            node.stop_all()
            time.sleep(0.5)
            node.arm(False)

        elif cmd == 'spin':
            if len(sys.argv) < 3:
                print('Usage: motor_control.py spin <throttle>')
                sys.exit(1)
            throttle = float(sys.argv[2])
            print(f'Spinning all motors at {throttle:.2f} for 3 seconds... (Ctrl+C to stop)')
            end = time.time() + 3.0
            while time.time() < end:
                node.send_motors([throttle] * 8)
                time.sleep(0.05)
            node.stop_all()

        elif cmd == 'motor':
            if len(sys.argv) < 4:
                print('Usage: motor_control.py motor <index> <throttle>')
                sys.exit(1)
            index = int(sys.argv[2])
            throttle = float(sys.argv[3])
            throttles = [0.0] * 8
            throttles[index] = throttle
            print(f'Spinning motor {index} at {throttle:.2f} for 3 seconds...')
            end = time.time() + 3.0
            while time.time() < end:
                node.send_motors(throttles)
                time.sleep(0.05)
            node.stop_all()

        elif cmd == 'stop':
            node.stop_all()

        else:
            print_usage()
            sys.exit(1)

    except KeyboardInterrupt:
        node.stop_all()
        print()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
