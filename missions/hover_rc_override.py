"""
Hover using RC override — pilot can reclaim control at any time.

The Jetson publishes RC override channels to hold neutral sticks at hover
throttle. When the real RC input deviates from centre by more than
STICK_THRESHOLD PWM units on any channel, overrides are released and the
pilot flies manually.

Requires MAVROS running and drone in AltHold / Loiter mode:
  ros2 launch mavros px4.launch fcu_url:=/dev/ttyTHS1

Run:
  python3 hover_rc_override.py
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from mavros_msgs.msg import OverrideRCIn, RCIn, State
from mavros_msgs.srv import CommandBool, SetMode

# PWM values
PWM_CENTER   = 1500   # neutral roll / pitch / yaw
PWM_HOVER    = 1500   # throttle at hover (tune for your aircraft)
CHAN_RELEASE  = 0     # release override on this channel
CHAN_NOCHANGE = 65535 # leave channel unchanged

# How far a stick must move from centre before we consider it pilot input
STICK_THRESHOLD = 50  # µs

# Channels (0-indexed) treated as sticks; anything outside threshold = pilot input.
# Throttle is intentionally excluded: it rests at the bottom (~1000 PWM), not at
# PWM_HOVER, so it would always trigger a false release on startup.
STICK_CHANNELS = {
    0: PWM_CENTER,  # Roll
    1: PWM_CENTER,  # Pitch
    3: PWM_CENTER,  # Yaw
}

OVERRIDE_HZ = 20  # publish rate while overriding
FLIGHT_MODE  = 'ALT_HOLD'  # ArduPilot mode name; use 'LOITER' if GPS available


class HoverRCOverride(Node):
    def __init__(self):
        super().__init__('hover_rc_override')

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.state           = State()
        self.rc_in_channels  = [PWM_CENTER] * 18
        self.overriding      = True

        self.create_subscription(State, '/mavros/state', self._state_cb, qos)
        self.create_subscription(RCIn,  '/mavros/rc/in', self._rc_in_cb, qos)
        self.override_pub = self.create_publisher(OverrideRCIn, '/mavros/rc/override', 10)
        self.create_timer(1.0 / OVERRIDE_HZ, self._timer_cb)

        self.arm_client  = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode,     '/mavros/set_mode')

    # ------------------------------------------------------------------ helpers

    def _state_cb(self, msg):
        self.state = msg

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

        self.get_logger().info('Connected. Setting flight mode...')
        self._set_mode(FLIGHT_MODE)

        self.get_logger().info('Arming...')
        self._arm(True)

        self.get_logger().info('Hover RC override started. Move any stick to reclaim control.')
        rclpy.spin(self)

    def _rc_in_cb(self, msg: RCIn):
        self.rc_in_channels = list(msg.channels)

        if not self.overriding:
            return

        for ch, centre in STICK_CHANNELS.items():
            if ch < len(self.rc_in_channels):
                if abs(self.rc_in_channels[ch] - centre) > STICK_THRESHOLD:
                    self.get_logger().info(
                        f'Pilot input detected on channel {ch+1} '
                        f'(value={self.rc_in_channels[ch]}). Releasing override.'
                    )
                    self._release_override()
                    return

    def _timer_cb(self):
        if not self.overriding:
            return
        self._publish_override()

    def _publish_override(self):
        msg = OverrideRCIn()
        channels = [CHAN_NOCHANGE] * 18
        channels[0] = PWM_CENTER   # Roll   — hold centre
        channels[1] = PWM_CENTER   # Pitch  — hold centre
        channels[2] = PWM_HOVER    # Throttle — hold hover
        channels[3] = PWM_CENTER   # Yaw    — hold centre
        msg.channels = channels
        self.override_pub.publish(msg)

    def _release_override(self):
        """Set all overridden channels to CHAN_RELEASE so FC uses real RC."""
        self.overriding = False
        msg = OverrideRCIn()
        channels = [CHAN_NOCHANGE] * 18
        for ch in STICK_CHANNELS:
            channels[ch] = CHAN_RELEASE
        # Also release throttle channel even though it's not in STICK_CHANNELS
        channels[2] = CHAN_RELEASE
        msg.channels = channels
        self.override_pub.publish(msg)
        self.get_logger().info('Override released. Pilot has full control.')


def main():
    rclpy.init()
    node = HoverRCOverride()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted — releasing override.')
        node._release_override()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
