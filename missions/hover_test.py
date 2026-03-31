"""
RC-triggered hover mission (real hardware) or auto-triggered (AirSim SITL).

RC ch5 has 3 positions: altitude / position / offboard.

The script streams setpoints at 20 Hz from the start so PX4 can accept an
OFFBOARD mode switch when the RC gets there (PX4 rejects the switch if there's
no active setpoint stream).

Switching ch5 to OFFBOARD triggers the mission:
  - arm
  - climb to HOVER_ALTITUDE and hold for HOVER_SECONDS
  - AUTO.LAND (PX4 descends at MPC_LAND_SPEED and cuts motors on touchdown)

Altitude is in the local EKF2 frame — z=0 is ground level at boot time.
PX4 closes the loop using barometer + IMU fusion; this script just sends the
target and waits.

After the mission ends the script waits for ch5 to leave OFFBOARD before
allowing a new run, so it won't restart on its own.

Switching ch5 away from OFFBOARD at any point during the mission hands
control back to the RC immediately.

--- Real hardware ---
Requires MAVROS running:
  ros2 launch mavros px4.launch fcu_url:=/dev/ttyACM0
Run:
  python3 hover_test.py

--- AirSim SITL ---
Requires AirSim (UE5), PX4 SITL, and MAVROS running:
  ros2 launch mavros px4.launch fcu_url:=udp://:14540@127.0.0.1:14580
Run:
  python3 hover_test.py --airsim
"""

import argparse
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode


HOVER_ALTITUDE = 1.5   # metres
HOVER_SECONDS  = 3.0
SETPOINT_HZ    = 20    # publish rate while streaming


class HoverMission(Node):
    def __init__(self):
        super().__init__('hover_test')

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.state = State()
        self.create_subscription(State, '/mavros/state', self._state_cb, qos)
        self.pose_pub  = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)
        self.arm_client  = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode,     '/mavros/set_mode')

    # ------------------------------------------------------------------ helpers

    def _state_cb(self, msg):
        self.state = msg

    def _is_offboard(self):
        return self.state.mode == 'OFFBOARD'

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
        self.get_logger().info(f'SET_MODE {mode}: {"OK" if ok else "FAILED"}')
        return ok

    def _arm(self, value):
        self.arm_client.wait_for_service()
        req = CommandBool.Request()
        req.value = value
        future = self.arm_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        ok = future.result() and future.result().success
        self.get_logger().info(f'{"ARM" if value else "DISARM"}: {"OK" if ok else "FAILED"}')
        return ok

    def _spin_hz(self, duration_sec, check_offboard=False):
        """Spin for duration_sec, publishing setpoints at SETPOINT_HZ.
        If check_offboard=True, returns False immediately if OFFBOARD is lost."""
        dt = 1.0 / SETPOINT_HZ
        end = time.time() + duration_sec
        while time.time() < end:
            if check_offboard and not self._is_offboard():
                return False
            self._publish_setpoint(HOVER_ALTITUDE)
            rclpy.spin_once(self, timeout_sec=dt)
        return True

    # ------------------------------------------------------------------ mission

    def _run_mission(self):
        self.get_logger().info('OFFBOARD active — arming...')
        self._arm(True)

        self.get_logger().info(f'Climbing to {HOVER_ALTITUDE} m...')
        if not self._spin_hz(4.0, check_offboard=True):
            self.get_logger().info('OFFBOARD lost during ascent — RC has control.')
            return

        self.get_logger().info(f'Hovering for {HOVER_SECONDS} s...')
        if not self._spin_hz(HOVER_SECONDS, check_offboard=True):
            self.get_logger().info('OFFBOARD lost during hover — RC has control.')
            return

        # Mission complete — land unless RC switches away first
        self.get_logger().info('Hover complete. Landing...')
        self._set_mode('AUTO.LAND')

        # Wait until disarmed (landed) or RC takes over
        while self.state.armed:
            if not self._is_offboard() and self.state.mode != 'AUTO.LAND':
                self.get_logger().info('RC has control.')
                return
            rclpy.spin_once(self, timeout_sec=1.0 / SETPOINT_HZ)

        self.get_logger().info('Landed and disarmed.')

    # ------------------------------------------------------------------ main loop

    def run(self, airsim=False):
        self.get_logger().info('Waiting for MAVROS connection...')
        while not self.state.connected:
            rclpy.spin_once(self, timeout_sec=0.1)

        if airsim:
            self.get_logger().info(
                'AirSim mode — streaming setpoints for 2 s, then switching to OFFBOARD...'
            )
            self._spin_hz(2.0)
            self._set_mode('OFFBOARD')
        else:
            self.get_logger().info(
                'Connected. Streaming setpoints — switch RC ch5 to OFFBOARD to start mission.'
            )

        dt = 1.0 / SETPOINT_HZ
        while True:
            # Keep streaming so PX4 can enter OFFBOARD the moment the RC switches
            self._publish_setpoint(HOVER_ALTITUDE)
            rclpy.spin_once(self, timeout_sec=dt)

            if self._is_offboard():
                self._run_mission()
                if airsim:
                    self.get_logger().info('Simulation mission complete.')
                    return
                if self.state.connected:
                    self.get_logger().info(
                        'Mission done. Switch RC ch5 out of OFFBOARD, then back to run again.'
                    )
                # Block re-trigger until the user explicitly leaves OFFBOARD
                while self._is_offboard():
                    self._publish_setpoint(HOVER_ALTITUDE)
                    rclpy.spin_once(self, timeout_sec=dt)


def main():
    parser = argparse.ArgumentParser(description='Hover mission — real hardware or AirSim SITL.')
    parser.add_argument(
        '--airsim',
        action='store_true',
        help='AirSim/SITL mode: auto-switch to OFFBOARD instead of waiting for RC ch5.',
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Hover mission")
    print()
    if args.airsim:
        print("Mode: AirSim simulation")
        print()
        print("Required (all on the GCS machine):")
        print("  1. AirSim/UE5 running with PX4 settings.json")
        print("  2. PX4 SITL:  make px4_sitl_default none_iris")
        print("  3. MAVROS:    ros2 launch mavros px4.launch \\")
        print("                  fcu_url:=udp://:14540@127.0.0.1:14580")
    else:
        print("Mode: Real hardware (run everything on the Jetson)")
        print()
        print("Required:")
        print("  MAVROS: ros2 launch mavros px4.launch fcu_url:=/dev/ttyACM0")
        print()
        print("RC ch5:  altitude | position | offboard")
        print("  Switch to OFFBOARD  →  mission starts (arm → climb → hover → land)")
        print("  Switch back anytime  →  RC takes over immediately")
    print("=" * 60)
    print()

    rclpy.init()
    node = HoverMission()
    try:
        node.run(airsim=args.airsim)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
