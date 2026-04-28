#!/usr/bin/env python3
"""
FREERIDER — PPO obstacle avoidance flight script.

Mirrors the DANI run_tree_avoid.py pattern, adapted for the Freerider policy:
  • Depth Anything V2 Small (HuggingFace transformers) for monocular depth
  • Freerider actor (TensorRT FP16) with two inputs: image stack + state
  • 3-frame depth stack  (3 × 144 × 256)
  • Action smoothing matching training: smoothed = 0.7 * raw + 0.3 * prev

Velocity (body frame, matches avoidance_env.py):
    fwd = 1.0 - 0.8 * |smoothed|   (always positive, slows near obstacles)
    lat = 0.8 * smoothed

Flow:
    wait for MAVROS → keypress → arm → AUTO.TAKEOFF → wait climb
    → OFFBOARD → avoidance loop until RC override → AUTO.LAND

RC override: switching to ALTCTL or POSCTL at any time exits the loop.

Usage:
    python run_freerider.py --engine ~/freerider/model/freerider_actor.trt

Requirements (Jetson):
    pip install transformers torch torchvision pillow pycuda
    sudo apt install tensorrt python3-tensorrt
    ROS 2 Humble + MAVROS
"""

import argparse
import csv
import os
import select
import subprocess
import sys
import termios
import threading
import time
import tty
from collections import deque

import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401 — initialises CUDA context
import pycuda.driver as cuda
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped
from mavros_msgs.msg import State, StatusText
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from nav_msgs.msg import Odometry
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image as RosImage
import tensorrt as trt
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SETPOINT_HZ     = 10
PRESTREAM_TIME  = 2.0
LOG_DIR         = os.path.expanduser('~/freerider/logs')
VIDEO_DIR       = os.path.expanduser('~/freerider/debug_frames')
DEPTH_MODEL_ID  = 'depth-anything/Depth-Anything-V2-Small-hf'
IMG_H, IMG_W    = 144, 256
N_FRAMES        = 3
FIXED_SPEED     = 1.0   # m/s forward (matches avoidance_env fixed_speed)
MAX_LATERAL     = 0.8   # m/s lateral (matches avoidance_env max_lateral)
ACTION_MOMENTUM = 0.3   # matches avoidance_env ACTION_MOMENTUM
SMOOTH_ALPHA    = 1.0 - ACTION_MOMENTUM  # 0.7
GST_PIPELINE = (
    'nvarguscamerasrc ! '
    'video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1 ! '
    'nvvidconv ! video/x-raw, format=BGRx ! '
    'videoconvert ! video/x-raw, format=BGR ! appsink'
)
_CMD_INTERVAL     = 2.0
_ARM_TIMEOUT      = 30.0
_TAKEOFF_TIMEOUT  = 30.0
_OFFBOARD_TIMEOUT = 10.0
RC_OVERRIDE_MODES = {'ALTCTL', 'POSCTL'}
SIM_FCU_URL       = 'udp://:14540@194.47.28.91:14580'
SIM_IMAGE_TOPIC   = '/airsim_node/drone_1/front_center_custom/Scene'


# ---------------------------------------------------------------------------
# TensorRT inference wrapper
# ---------------------------------------------------------------------------

class TRTEngine:
    """Minimal TensorRT FP16 inference wrapper for the Freerider policy actor.

    Expects the engine to have exactly three tensors named:
        'image'  — input  (1, 3, 144, 256) float32
        'state'  — input  (1, 1)           float32
        'action' — output (1, 1)           float32
    """

    def __init__(self, engine_path: str):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime       = trt.Runtime(logger)
            self._engine  = runtime.deserialize_cuda_engine(f.read())
        self._context = self._engine.create_execution_context()
        self._stream  = cuda.Stream()

        self._bindings = {}
        for i in range(self._engine.num_io_tensors):
            name  = self._engine.get_tensor_name(i)
            shape = tuple(self._engine.get_tensor_shape(name))
            dtype = trt.nptype(self._engine.get_tensor_dtype(name))
            host  = cuda.pagelocked_empty(shape, dtype)
            dev   = cuda.mem_alloc(host.nbytes)
            self._bindings[name] = {'shape': shape, 'dtype': dtype, 'host': host, 'dev': dev}
            self._context.set_tensor_address(name, int(dev))

    def infer(self, image_np: np.ndarray, state_np: np.ndarray) -> float:
        """Run one deterministic forward pass.

        Parameters
        ----------
        image_np : (3, 144, 256) float32 — stacked depth frames in [0, 1]
        state_np : (1,)          float32 — previous smoothed lateral action

        Returns
        -------
        float — raw lateral action (caller should clip to [-1, 1])
        """
        img = self._bindings['image']
        st  = self._bindings['state']
        act = self._bindings['action']

        np.copyto(img['host'], image_np.reshape(img['shape']).astype(img['dtype']))
        np.copyto(st['host'],  state_np.reshape(st['shape']).astype(st['dtype']))

        cuda.memcpy_htod_async(img['dev'], img['host'], self._stream)
        cuda.memcpy_htod_async(st['dev'],  st['host'],  self._stream)

        self._context.execute_async_v3(self._stream.handle)

        cuda.memcpy_dtoh_async(act['host'], act['dev'], self._stream)
        self._stream.synchronize()

        return float(act['host'].flat[0])


# ---------------------------------------------------------------------------
# Depth estimator (Depth Anything V2 Small)
# ---------------------------------------------------------------------------

class DepthEstimator:
    """Thin wrapper around Depth Anything V2 Small.

    Output: (IMG_H, IMG_W) float32, per-frame min-max normalised to [0, 1].
    Higher values = closer obstacles (disparity convention).
    """

    def __init__(self, device: str = 'cuda'):
        print(f'[DepthEstimator] Loading {DEPTH_MODEL_ID} on {device} ...')
        self._processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
        self._model     = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID)
        self._model.to(device).eval()
        self._device    = device
        print('[DepthEstimator] Ready.')

    def estimate(self, bgr: np.ndarray) -> np.ndarray:
        rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil    = PILImage.fromarray(rgb)
        inputs = self._processor(images=pil, return_tensors='pt')
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            raw = self._model(**inputs).predicted_depth  # (1, H, W) disparity
        depth = raw.squeeze().cpu().numpy()
        depth = cv2.resize(depth, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
        d_min, d_max = float(depth.min()), float(depth.max())
        if d_max > d_min:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)
        return depth.astype(np.float32)


# ---------------------------------------------------------------------------
# ROS 2 node
# ---------------------------------------------------------------------------

class FreeriderNode(Node):

    def __init__(self, engine_path: str, sim: bool = False, sim_image_topic: str = SIM_IMAGE_TOPIC):
        super().__init__('freerider')

        self._sim            = sim
        self.state           = State()
        self._left_rc_modes  = False
        self._yaw            = 0.0
        self._latest_bgr     = None
        self._frame_lock     = threading.Lock()
        self._recording      = False

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        state_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.create_subscription(State,      '/mavros/state',               self._on_state,      state_qos)
        self.create_subscription(StatusText, '/mavros/statustext',          self._on_statustext, qos)
        self.create_subscription(Odometry,   '/mavros/local_position/odom', self._on_odom,       qos)

        if sim:
            self._bridge = CvBridge()
            self.create_subscription(RosImage, sim_image_topic, self._on_sim_image, qos)
            self.get_logger().info(f'[Freerider] Sim mode — camera from {sim_image_topic}')

        self.arm_client  = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode,     '/mavros/set_mode')
        self.land_client = self.create_client(CommandTOL,  '/mavros/cmd/land')

        self.vel_pub = self.create_publisher(
            TwistStamped, '/mavros/setpoint_velocity/cmd_vel', 10
        )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._depth = DepthEstimator(device=device)
        self._trt   = TRTEngine(engine_path)
        self.get_logger().info(f'[Freerider] Engine loaded: {engine_path}')

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def _on_state(self, msg):
        self.state = msg

    def _on_statustext(self, msg):
        self.get_logger().info(f'[PX4] {msg.text}')

    def _on_odom(self, msg):
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._yaw = np.arctan2(siny_cosp, cosy_cosp)

    def _on_sim_image(self, msg):
        """Receive a ROS Image from the simulator and store it as BGR."""
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self._frame_lock:
                self._latest_bgr = bgr
        except Exception as e:
            self.get_logger().warn(f'[Freerider] sim image conversion failed: {e}')

    # ------------------------------------------------------------------
    # RC override
    # ------------------------------------------------------------------

    def _rc_override(self) -> bool:
        """Returns True once the pilot has switched to an RC mode after
        having been in a non-RC mode (i.e. after OFFBOARD has been active)."""
        if self.state.mode not in RC_OVERRIDE_MODES:
            self._left_rc_modes = True
            return False
        return self._left_rc_modes

    # ------------------------------------------------------------------
    # Velocity publishing (body-frame → ENU via yaw rotation)
    # ------------------------------------------------------------------

    def _publish_vel(self, vx: float = 0.0, vy: float = 0.0, vz: float = 0.0):
        yaw = self._yaw
        msg = TwistStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.twist.linear.x  = vx * np.cos(yaw) - vy * np.sin(yaw)
        msg.twist.linear.y  = vx * np.sin(yaw) + vy * np.cos(yaw)
        msg.twist.linear.z  = vz
        self.vel_pub.publish(msg)

    # ------------------------------------------------------------------
    # Arm / takeoff / offboard / land
    # ------------------------------------------------------------------

    def _arm(self) -> bool:
        self.get_logger().info('Arming...')
        self.arm_client.wait_for_service()
        deadline, last_send = time.monotonic() + _ARM_TIMEOUT, 0.0
        while time.monotonic() < deadline:
            if time.monotonic() - last_send >= _CMD_INTERVAL:
                req = CommandBool.Request()
                req.value = True
                future = self.arm_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
                last_send = time.monotonic()
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.state.armed:
                self.get_logger().info('Armed.')
                return True
        return False

    def _takeoff(self):
        self.get_logger().info('AUTO.TAKEOFF...')
        self.mode_client.wait_for_service()
        deadline, last_send = time.monotonic() + _TAKEOFF_TIMEOUT, 0.0
        while time.monotonic() < deadline:
            if time.monotonic() - last_send >= _CMD_INTERVAL:
                req = SetMode.Request()
                req.custom_mode = 'AUTO.TAKEOFF'
                future = self.mode_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
                last_send = time.monotonic()
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._rc_override():
                return None
            if self.state.mode == 'AUTO.TAKEOFF':
                self.get_logger().info('AUTO.TAKEOFF confirmed.')
                return True
            if not self.state.armed:
                return False
        return False

    def _switch_offboard(self):
        dt = 1.0 / SETPOINT_HZ
        self.get_logger().info(f'Pre-streaming setpoints for {PRESTREAM_TIME:.0f}s...')
        deadline = time.monotonic() + PRESTREAM_TIME
        while time.monotonic() < deadline:
            self._publish_vel()
            rclpy.spin_once(self, timeout_sec=dt)

        self.get_logger().info('Switching to OFFBOARD...')
        self.mode_client.wait_for_service()
        deadline, last_send = time.monotonic() + _OFFBOARD_TIMEOUT, 0.0
        while time.monotonic() < deadline:
            self._publish_vel()
            if time.monotonic() - last_send >= _CMD_INTERVAL:
                req = SetMode.Request()
                req.custom_mode = 'OFFBOARD'
                future = self.mode_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
                last_send = time.monotonic()
            rclpy.spin_once(self, timeout_sec=dt)
            if self._rc_override():
                return None
            if self.state.mode == 'OFFBOARD':
                self.get_logger().info('OFFBOARD confirmed.')
                return True
        return False

    def _land(self):
        self.get_logger().info('Landing...')
        self.mode_client.wait_for_service()
        deadline, last_send = time.monotonic() + 30.0, 0.0
        while time.monotonic() < deadline:
            if time.monotonic() - last_send >= _CMD_INTERVAL:
                req = SetMode.Request()
                req.custom_mode = 'AUTO.LAND'
                future = self.mode_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
                last_send = time.monotonic()
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.state.mode == 'AUTO.LAND':
                self.get_logger().info('AUTO.LAND confirmed.')
                return
            if not self.state.armed:
                return

    # ------------------------------------------------------------------
    # Camera thread (GStreamer → raw BGR, + MP4 recording)
    # ------------------------------------------------------------------

    def _start_camera(self):
        if self._sim:
            self.get_logger().info('[Freerider] Sim mode — using ROS image topic, skipping GStreamer.')
            return
        threading.Thread(target=self._camera_reader, daemon=True).start()

    def _camera_reader(self):
        cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.get_logger().error('Cannot open camera')
            return

        os.makedirs(VIDEO_DIR, exist_ok=True)
        stamp  = time.strftime('%Y%m%d_%H%M%S')
        fname  = os.path.join(VIDEO_DIR, f'{stamp}.mp4')
        writer = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1080))
        self.get_logger().info(f'Recording to {fname}')
        self._recording = True

        while self._recording:
            ret, frame = cap.read()
            if ret:
                writer.write(frame)
                with self._frame_lock:
                    self._latest_bgr = frame

        writer.release()
        cap.release()
        self.get_logger().info('Recording saved.')
        if not self._sim:
            subprocess.Popen(
                ['rclone', 'copy', VIDEO_DIR, 'dropbox:freerider/debug_frames'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

    def _get_frame(self):
        with self._frame_lock:
            return self._latest_bgr.copy() if self._latest_bgr is not None else None

    # ------------------------------------------------------------------
    # One policy step
    # ------------------------------------------------------------------

    def _avoidance_step(self, frame_stack: deque, smoothed_action: float):
        """Estimate depth, run TRT policy, publish velocity.

        Returns (frame_stack, raw_action, new_smoothed_action, lateral_vel, step_latency_ms).
        Returns unchanged smoothed_action, 0.0 lateral and 0.0 latency if no frame yet.
        """
        bgr = self._get_frame()
        if bgr is None:
            return frame_stack, smoothed_action, smoothed_action, 0.0, 0.0

        t_step_start = time.monotonic()

        depth = self._depth.estimate(bgr)   # (144, 256) float32 [0, 1]
        frame_stack.append(depth)

        # Pad with the oldest frame until the stack is full
        while len(frame_stack) < N_FRAMES:
            frame_stack.appendleft(frame_stack[0])

        image_np = np.stack(list(frame_stack), axis=0)         # (3, 144, 256)
        state_np = np.array([smoothed_action], dtype=np.float32)  # (1,)

        raw_action  = float(np.clip(self._trt.infer(image_np, state_np), -1.0, 1.0))
        new_smoothed = SMOOTH_ALPHA * raw_action + ACTION_MOMENTUM * smoothed_action

        fwd = FIXED_SPEED - MAX_LATERAL * abs(new_smoothed)
        lat = MAX_LATERAL * new_smoothed
        self._publish_vel(vx=fwd, vy=lat)

        step_latency_ms = (time.monotonic() - t_step_start) * 1000.0

        return frame_stack, raw_action, new_smoothed, lat, step_latency_ms

    # ------------------------------------------------------------------
    # Main run sequence
    # ------------------------------------------------------------------

    def run(self):
        dt = 1.0 / SETPOINT_HZ
        self._left_rc_modes = False

        # Wait briefly for MAVROS, then try to launch it if absent
        deadline = time.monotonic() + 2.0
        while not self.state.connected and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

        fcu_url = SIM_FCU_URL if self._sim else '/dev/ttyTHS1:115200'
        if not self.state.connected:
            self.get_logger().info('MAVROS not connected — launching...')
            subprocess.Popen(
                ['ros2', 'launch', 'mavros', 'px4.launch', f'fcu_url:={fcu_url}'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

        self.get_logger().info('Waiting for MAVROS connection...')
        while not self.state.connected:
            rclpy.spin_once(self, timeout_sec=0.1)

        if self._sim:
            self.get_logger().info('Sim mode — starting automatically.')
        else:
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

        if not self._arm():
            self.get_logger().error('Failed to arm — aborting.')
            return

        result = self._takeoff()
        if not result:
            self.get_logger().error('Takeoff failed — aborting.')
            return

        self.get_logger().info('Climbing...')
        while self.state.mode == 'AUTO.TAKEOFF':
            if self._rc_override():
                return
            if not self.state.armed:
                return
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info(f'Takeoff complete (now in {self.state.mode}).')
        self._start_camera()

        result = self._switch_offboard()
        if not result:
            self.get_logger().error('Failed to enter OFFBOARD — aborting.')
            return

        # CSV log
        os.makedirs(LOG_DIR, exist_ok=True)
        stamp    = time.strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(LOG_DIR, f'flight_{stamp}.csv')
        log_file = open(log_path, 'w', newline='')
        csv_writer = csv.writer(log_file)
        csv_writer.writerow(['t', 'raw_action', 'smoothed_action', 'lateral_vel_ms', 'step_latency_ms'])

        self.get_logger().info('Freerider avoidance active. RC override to exit.')
        frame_stack     = deque(maxlen=N_FRAMES)
        smoothed_action = 0.0
        t0              = time.monotonic()
        latencies       = []

        try:
            while True:
                rclpy.spin_once(self, timeout_sec=dt)
                if self._rc_override():
                    self.get_logger().info('RC override detected — stopping.')
                    break
                if not self.state.armed:
                    self.get_logger().info('Disarmed — stopping.')
                    break
                frame_stack, raw_action, smoothed_action, lat, latency_ms = self._avoidance_step(
                    frame_stack, smoothed_action,
                )
                if latency_ms > 0.0:
                    latencies.append(latency_ms)
                csv_writer.writerow([
                    f'{time.monotonic() - t0:.3f}',
                    f'{raw_action:.4f}',
                    f'{smoothed_action:.4f}',
                    f'{lat:.4f}',
                    f'{latency_ms:.1f}',
                ])
        finally:
            log_file.close()
            if latencies:
                avg_ms = sum(latencies) / len(latencies)
                self.get_logger().info(
                    f'Step latency — avg: {avg_ms:.1f} ms  '
                    f'min: {min(latencies):.1f} ms  '
                    f'max: {max(latencies):.1f} ms  '
                    f'({len(latencies)} steps)'
                )
            if not self._sim:
                subprocess.Popen(
                    ['rclone', 'copy', LOG_DIR, 'dropbox:freerider/logs'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )

        self._land()
        self._recording = False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Freerider PPO avoidance flight')
    parser.add_argument(
        '--engine',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'freerider_actor.trt'),
        help='Path to TensorRT engine (.trt)',
    )
    parser.add_argument(
        '--sim',
        action='store_true',
        help='Simulation mode: connect via UDP, skip keypress, use ROS image topic for camera.',
    )
    parser.add_argument(
        '--sim-image-topic',
        default=SIM_IMAGE_TOPIC,
        help=f'ROS image topic to use in sim mode (default: {SIM_IMAGE_TOPIC})',
    )
    args = parser.parse_args()

    print('=' * 60)
    print('Freerider — PPO Obstacle Avoidance')
    print(f'Engine : {args.engine}')
    if args.sim:
        print(f'Mode   : Simulation (PX4 SITL via {SIM_FCU_URL})')
        print(f'Camera : {args.sim_image_topic}')
    else:
        print('Mode   : Real hardware')
        print('RC override: switch to ALTCTL or POSCTL at any time')
    print('=' * 60)

    rclpy.init()
    node = FreeriderNode(args.engine, sim=args.sim, sim_image_topic=args.sim_image_topic)
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
    finally:
        node._recording = False
        time.sleep(0.5)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
