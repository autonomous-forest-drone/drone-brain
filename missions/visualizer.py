"""
RViz visualizer for drone-brain missions.

Subscribes to MAVROS topics and publishes RViz-compatible messages:
  - Vehicle pose (PoseStamped)
  - EKF trajectory path (nav_msgs/Path)
  - Raw GPS trajectory path (nav_msgs/Path)  ← shows GPS noise vs EKF smoothing
  - Commanded velocity arrow (Marker)
  - Goal position marker (Marker)
  - Status text overlay (Marker)

Run alongside a mission script:
  ros2 run drone_brain visualizer
  # or directly:
  python3 missions/visualizer.py

Then open RViz2 and add the topics under /drone_viz/.

Topics published:
  /drone_viz/vehicle_pose       — current EKF pose
  /drone_viz/ekf_path           — EKF-smoothed trajectory (green)
  /drone_viz/gps_path           — raw GPS trajectory (red, shows noise)
  /drone_viz/velocity_cmd       — commanded velocity arrow (yellow)
  /drone_viz/goal_marker        — goal position sphere (blue)
  /drone_viz/status_text        — mode / distance text overlay
"""

import json
import math
import os
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import Point, PoseStamped, TwistStamped
from mavros_msgs.msg import HomePosition, State
from nav_msgs.msg import Path
from sensor_msgs.msg import NavSatFix
from visualization_msgs.msg import Marker, MarkerArray


EARTH_R   = 6_371_000.0
TRAIL_MAX = 2000           # maximum poses kept in each path
GOAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'goal.json')


def _load_goal():
    if not os.path.exists(GOAL_FILE):
        return None
    try:
        with open(GOAL_FILE) as f:
            d = json.load(f)
        return float(d['lat']), float(d['lon'])
    except Exception:
        return None


class DroneVisualizer(Node):

    def __init__(self):
        super().__init__('drone_visualizer')

        # ---- state ----
        self._pose          = None   # current EKF PoseStamped
        self._gps           = None   # current NavSatFix
        self._cmd_vel       = None   # current TwistStamped
        self._home          = None   # HomePosition
        self._state         = State()
        self._ekf_path      = Path()
        self._gps_path      = Path()
        self._goal          = _load_goal()  # (lat, lon) or None

        if self._goal:
            self.get_logger().info(
                f'Goal loaded from goal.json: ({self._goal[0]:.7f}, {self._goal[1]:.7f})')
        else:
            self.get_logger().info('No goal.json found — goal marker will not be shown.')

        # ---- QoS ----
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE)
        state_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)

        # ---- subscriptions ----
        self.create_subscription(PoseStamped,  '/mavros/local_position/pose',    self._on_pose,  sensor_qos)
        self.create_subscription(NavSatFix,    '/mavros/global_position/global', self._on_gps,   sensor_qos)
        self.create_subscription(TwistStamped, '/mavros/setpoint_velocity/cmd_vel', self._on_cmd_vel, 10)
        self.create_subscription(HomePosition, '/mavros/home_position/home',     self._on_home,  state_qos)
        self.create_subscription(State,        '/mavros/state',                  self._on_state, state_qos)

        # ---- publishers ----
        self._pub_pose    = self.create_publisher(PoseStamped,   '/drone_viz/vehicle_pose',   10)
        self._pub_ekf     = self.create_publisher(Path,          '/drone_viz/ekf_path',        10)
        self._pub_gps     = self.create_publisher(Path,          '/drone_viz/gps_path',        10)
        self._pub_markers = self.create_publisher(MarkerArray,   '/drone_viz/markers',         10)

        # ---- timer (20 Hz) ----
        self.create_timer(0.05, self._publish)

        self.get_logger().info('Drone visualizer started.')

    # ------------------------------------------------------------------ callbacks

    def _on_pose(self, msg):
        self._pose = msg

    def _on_gps(self, msg):
        if msg.status.status < 0:
            return
        self._gps = msg

    def _on_cmd_vel(self, msg):
        self._cmd_vel = msg

    def _on_home(self, msg):
        self._home = msg

    def _on_state(self, msg):
        self._state = msg

    # ------------------------------------------------------------------ helpers

    def _gps_to_local(self, lat, lon):
        """Convert GPS lat/lon to local ENU (x=East, y=North) relative to home."""
        if self._home is None:
            return None
        h_lat = self._home.geo.latitude
        h_lon = self._home.geo.longitude
        dn = math.radians(lat - h_lat) * EARTH_R
        de = math.radians(lon - h_lon) * EARTH_R * math.cos(math.radians(h_lat))
        return de, dn   # (east, north) = (x, y) in ENU

    def _distance_to_goal(self):
        if self._goal is None or self._gps is None:
            return None
        dn = math.radians(self._goal[0] - self._gps.latitude) * EARTH_R
        de = math.radians(self._goal[1] - self._gps.longitude) * EARTH_R * math.cos(
            math.radians(self._gps.latitude))
        return math.sqrt(dn * dn + de * de)

    def _append_path(self, path_msg, pose_stamped):
        path_msg.poses.append(pose_stamped)
        if len(path_msg.poses) > TRAIL_MAX:
            del path_msg.poses[0]

    # ------------------------------------------------------------------ publish

    def _publish(self):
        now = self.get_clock().now().to_msg()

        # -- EKF pose & path --
        if self._pose is not None:
            self._pub_pose.publish(self._pose)

            self._ekf_path.header.stamp    = now
            self._ekf_path.header.frame_id = 'map'
            self._append_path(self._ekf_path, self._pose)
            self._pub_ekf.publish(self._ekf_path)

        # -- raw GPS path (converted to local frame) --
        if self._gps is not None and self._home is not None:
            local = self._gps_to_local(self._gps.latitude, self._gps.longitude)
            if local is not None:
                gps_pose = PoseStamped()
                gps_pose.header.stamp    = now
                gps_pose.header.frame_id = 'map'
                gps_pose.pose.position.x = local[0]   # East
                gps_pose.pose.position.y = local[1]   # North
                gps_pose.pose.position.z = self._pose.pose.position.z if self._pose else 0.0
                gps_pose.pose.orientation.w = 1.0

                self._gps_path.header.stamp    = now
                self._gps_path.header.frame_id = 'map'
                self._append_path(self._gps_path, gps_pose)
                self._pub_gps.publish(self._gps_path)

        # -- markers --
        markers = MarkerArray()
        mid = 0

        # velocity command arrow
        if self._cmd_vel is not None and self._pose is not None:
            m = Marker()
            m.header.stamp    = now
            m.header.frame_id = 'map'
            m.ns     = 'velocity'
            m.id     = mid; mid += 1
            m.type   = Marker.ARROW
            m.action = Marker.ADD
            m.scale.x = 0.08   # shaft diameter
            m.scale.y = 0.16   # head diameter
            m.scale.z = 0.0
            m.color.r = 1.0
            m.color.g = 0.8
            m.color.a = 1.0
            p = self._pose.pose.position
            tail = Point(x=p.x, y=p.y, z=p.z)
            dt = 0.5   # scale: 1 m/s → 0.5 m arrow
            v = self._cmd_vel.twist.linear
            head = Point(x=p.x + dt * v.x, y=p.y + dt * v.y, z=p.z + dt * v.z)
            m.points = [tail, head]
            markers.markers.append(m)

        # goal sphere
        if self._goal is not None and self._home is not None:
            local = self._gps_to_local(self._goal[0], self._goal[1])
            if local is not None:
                m = Marker()
                m.header.stamp    = now
                m.header.frame_id = 'map'
                m.ns     = 'goal'
                m.id     = mid; mid += 1
                m.type   = Marker.CYLINDER
                m.action = Marker.ADD
                m.pose.position.x = local[0]
                m.pose.position.y = local[1]
                m.pose.position.z = 0.0
                m.pose.orientation.w = 1.0
                m.scale.x = 3.0   # goal radius visualisation (= GOAL_RADIUS * 2)
                m.scale.y = 3.0
                m.scale.z = 0.1
                m.color.r = 0.0
                m.color.g = 0.4
                m.color.b = 1.0
                m.color.a = 0.5
                markers.markers.append(m)

        # status text
        mode     = self._state.mode if self._state else '—'
        armed    = 'ARMED' if (self._state and self._state.armed) else 'DISARMED'
        dist_str = f'{self._distance_to_goal():.1f} m' if self._distance_to_goal() is not None else '—'
        z_str    = f'{self._pose.pose.position.z:.2f} m' if self._pose else '—'

        m = Marker()
        m.header.stamp    = now
        m.header.frame_id = 'map'
        m.ns     = 'status'
        m.id     = mid; mid += 1
        m.type   = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = 0.0
        m.pose.position.y = 0.0
        m.pose.position.z = 3.5
        m.pose.orientation.w = 1.0
        m.scale.z = 0.3
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 1.0
        m.color.a = 1.0
        m.text = f'{armed}  |  {mode}  |  alt: {z_str}  |  dist: {dist_str}'
        markers.markers.append(m)

        if markers.markers:
            self._pub_markers.publish(markers)


# ------------------------------------------------------------------ entry point

def main(args=None):
    rclpy.init(args=args)
    node = DroneVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
