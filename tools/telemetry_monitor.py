

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
    def __init__(self, modes):
        super().__init__('telemetry_monitor_subscriber')

        # BEST_EFFORT QoS matches mavros sensor topics which don't guarantee delivery
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        if 'battery' in modes:
            self.create_subscription(BatteryState, '/mavros/battery', self.battery_callback, qos)
            self.get_logger().info('Listening to /mavros/battery ...')
        if 'gps' in modes:
            self.create_subscription(NavSatFix, '/mavros/global_position/global', self.gps_callback, qos)
            self.get_logger().info('Listening to /mavros/global_position/global ...')
        if 'imu' in modes:
            self.create_subscription(Imu, '/mavros/imu/data_raw', self.imu_callback, qos)
            self.get_logger().info('Listening to /mavros/imu/data_raw ...')

    def battery_callback(self, msg):
        print(f'[BAT]  Voltage: {msg.voltage:.2f}V  |  Charge: {msg.percentage * 100:.1f}%  |  Current: {msg.current:.2f}A')

    def gps_callback(self, msg):
        status = STATUS_MAP.get(msg.status.status, 'UNKNOWN')
        print(f'[GPS]  Lat: {msg.latitude:.6f}  Lon: {msg.longitude:.6f}  Alt: {msg.altitude:.2f}m  Status: {status}')

    def imu_callback(self, msg):
        a = msg.linear_acceleration
        g = msg.angular_velocity
        print(
            f'[IMU]  Accel(m/s²) x={a.x:7.3f}  y={a.y:7.3f}  z={a.z:7.3f}  |  '
            f'Gyro(rad/s) x={g.x:7.3f}  y={g.y:7.3f}  z={g.z:7.3f}'
        )


def main():
    valid_modes = ['battery', 'gps', 'imu']

    if len(sys.argv) < 2:
        modes = valid_modes
    else:
        modes = [a.lstrip('-') for a in sys.argv[1:]]
        invalid = [m for m in modes if m not in valid_modes]
        if invalid:
            print(f'Usage: python3 telemetry_monitor.py [-battery] [-gps] [-imu]')
            print(f'       (no arguments streams everything)')
            sys.exit(1)

    rclpy.init()
    node = DroneDataSubscriber(modes)
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
