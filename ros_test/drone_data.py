

"""
The following commands were used to open the terminal, which would run in the background fetching the nodes to subscribe to the pixhawk, the ros library in this code subs from the other end reading the data

The command:
  source /opt/ros/humble/setup.bash
  ros2 launch mavros px4.launch fcu_url:=/dev/ttyACM0
"""

import sys


# ROS2 core libraries for creating nodes and managing communication
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

# Message types for GPS, IMU, and battery data from mavros
from sensor_msgs.msg import NavSatFix, Imu, BatteryState


# Maps GPS status codes to human-readable strings
STATUS_MAP = {-1: 'NO FIX', 0: 'FIX', 1: 'SBAS FIX', 2: 'GBAS FIX'}


class DroneDataSubscriber(Node):
    def __init__(self, mode):
        super().__init__('drone_data_subscriber')
        self.mode = mode

        # BEST_EFFORT QoS matches mavros sensor topics which don't guarantee delivery
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Subscribe only to the topic matching the requested argument
        if mode == 'battery':
            self.create_subscription(BatteryState, '/mavros/battery', self.battery_callback, qos)
            self.get_logger().info('Listening to /mavros/battery ...')
        elif mode == 'gps':
            self.create_subscription(NavSatFix, '/mavros/global_position/global', self.gps_callback, qos)
            self.get_logger().info('Listening to /mavros/global_position/global ...')
        elif mode == 'imu':
            self.create_subscription(Imu, '/mavros/imu/data_raw', self.imu_callback, qos)
            self.get_logger().info('Listening to /mavros/imu/data_raw ...')

    def battery_callback(self, msg):
        # Print voltage, charge percentage, and current draw on a single updating line
        print(f'\r⚡ Voltage: {msg.voltage:.2f}V  |  Charge: {msg.percentage * 100:.1f}%  |  Current: {msg.current:.2f}A    ', end='', flush=True)

    def gps_callback(self, msg):
        # Look up the fix status string, defaulting to UNKNOWN for unrecognised codes
        status = STATUS_MAP.get(msg.status.status, 'UNKNOWN')
        print(f'\r🛰  Lat: {msg.latitude:.6f}  Lon: {msg.longitude:.6f}  Alt: {msg.altitude:.2f}m  Status: {status}    ', end='', flush=True)

    def imu_callback(self, msg):
        # Unpack linear acceleration and angular velocity vectors for cleaner formatting
        a = msg.linear_acceleration
        g = msg.angular_velocity
        print(
            f'\r📐 Accel(m/s²) x={a.x:7.3f}  y={a.y:7.3f}  z={a.z:7.3f}  |  '
            f'Gyro(rad/s) x={g.x:7.3f}  y={g.y:7.3f}  z={g.z:7.3f}    ',
            end='', flush=True
        )


def main():
    valid_modes = ['battery', 'gps', 'imu']

    # Validate argument — strip leading dashes so both 'gps' and '-gps' work
    if len(sys.argv) < 2 or sys.argv[1].lstrip('-') not in valid_modes:
        print(f'Usage: python3 drone_data.py -battery | -gps | -imu')
        sys.exit(1)

    mode = sys.argv[1].lstrip('-')

    rclpy.init()
    node = DroneDataSubscriber(mode)
    try:
        # spin() blocks here, calling callbacks as messages arrive
        rclpy.spin(node)
    except KeyboardInterrupt:
        print()
    finally:
        # Always clean up the node and shut down rclpy on exit
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
