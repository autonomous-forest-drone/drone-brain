#!/usr/bin/env python3
"""
FREERIDER — PPO obstacle avoidance flight script.

  • Depth Anything V2 Small (HuggingFace transformers) for monocular depth
  • Freerider actor (TensorRT FP16) with two inputs: image stack + state
  • 3-frame depth stack  (3 × 144 × 256)
  • Action smoothing matching training: smoothed = 0.7 * raw + 0.3 * prev
  • Per-flight logs: images/<timestamp>/{frames/, depth/, flight.csv, flight.png}

Camera (real hardware):
  IMX219 fixed-focus camera on camera1 CSI port (sensor-id=0), read via appsink.

Velocity (body frame):
    fwd = FIXED_SPEED - MAX_LATERAL * |smoothed|
    lat = MAX_LATERAL * smoothed

Flow:
    wait for MAVROS → keypress → arm → AUTO.TAKEOFF → wait climb
    → start camera pipeline → OFFBOARD → avoidance loop until RC override → AUTO.LAND

RC override: switching to ALTCTL or POSCTL at any time exits the loop.

Usage:
    python run_freerider.py
    python run_freerider.py --sim
"""

import argparse
import atexit
import csv
import os
import select
import subprocess
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import termios
import threading
import time
import tty
from collections import deque

import torch
torch.cuda.init()  # create PyTorch primary CUDA context first
torch.backends.cudnn.enabled = False  # cuDNN conflicts with pycuda/TRT on Jetson

import cv2
import numpy as np
import pycuda.driver as cuda

# Attach pycuda to the same primary context that PyTorch already created.
cuda.init()
_pycuda_ctx = cuda.Device(torch.cuda.current_device()).retain_primary_context()
_pycuda_ctx.push()
atexit.register(_pycuda_ctx.pop)

import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped
from mavros_msgs.msg import PlayTuneV2, State, StatusText
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from nav_msgs.msg import Odometry
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image as RosImage
import tensorrt as trt
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from tools.camera_imx219 import Camera

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SETPOINT_HZ     = 10
PRESTREAM_TIME  = 2.0
FLIGHT_LOG_ROOT = os.path.expanduser('~/drone-brain/images')
DEPTH_MODEL_ID  = 'depth-anything/Depth-Anything-V2-Small-hf'
IMG_H, IMG_W    = 144, 256
N_FRAMES        = 3
FIXED_SPEED     = 0.6
MAX_LATERAL     = 1.0
ACTION_MOMENTUM = 0.3
SMOOTH_ALPHA    = 1.0 - ACTION_MOMENTUM   # 0.7
RGB_SAVE_EVERY  = 1                       # save every frame

DEPTH_ENGINE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'model',
    'depth_anything_v2_small_fp16.trt'
)

# HUD composite video dimensions
_DISP_W, _DISP_H = 640, 360
_BAR_H            = 70

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
            raw = self._model(**inputs).predicted_depth
        depth = raw.squeeze().cpu().numpy()
        depth = cv2.resize(depth, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
        d_min, d_max = float(depth.min()), float(depth.max())
        if d_max > d_min:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)
        return depth.astype(np.float32)


# ---------------------------------------------------------------------------
# Depth estimator — TensorRT fast path
# ---------------------------------------------------------------------------

class DepthEstimatorTRT:

    def __init__(self, engine_path: str):
        print(f'[DepthEstimatorTRT] Loading processor from {DEPTH_MODEL_ID} ...')
        _proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
        self._mean = np.array(_proc.image_mean, dtype=np.float32)
        self._std  = np.array(_proc.image_std,  dtype=np.float32)

        print(f'[DepthEstimatorTRT] Loading TRT engine: {engine_path}')
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime      = trt.Runtime(logger)
            self._engine = runtime.deserialize_cuda_engine(f.read())
        self._context = self._engine.create_execution_context()
        self._stream  = cuda.Stream()

        self._bindings = {}
        for i in range(self._engine.num_io_tensors):
            name  = self._engine.get_tensor_name(i)
            shape = tuple(self._engine.get_tensor_shape(name))
            dtype = trt.nptype(self._engine.get_tensor_dtype(name))
            host  = cuda.pagelocked_empty(shape, dtype)
            dev   = cuda.mem_alloc(host.nbytes)
            self._bindings[name] = {
                'shape': shape, 'dtype': dtype, 'host': host, 'dev': dev
            }
            self._context.set_tensor_address(name, int(dev))

        _, _, self._in_h, self._in_w = self._bindings['pixel_values']['shape']
        print(f'[DepthEstimatorTRT] Input: {self._in_h}×{self._in_w}  Ready.')

    def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self._in_w, self._in_h),
                         interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        rgb = (rgb - self._mean) / self._std
        return rgb.transpose(2, 0, 1)[np.newaxis]

    def estimate(self, bgr: np.ndarray) -> np.ndarray:
        pixel_values = self._preprocess(bgr)

        pv  = self._bindings['pixel_values']
        out = self._bindings['predicted_depth']

        np.copyto(pv['host'], pixel_values.reshape(pv['shape']).astype(pv['dtype']))
        cuda.memcpy_htod_async(pv['dev'],  pv['host'],  self._stream)
        self._context.execute_async_v3(self._stream.handle)
        cuda.memcpy_dtoh_async(out['host'], out['dev'], self._stream)
        self._stream.synchronize()

        depth = out['host'].squeeze().copy()
        depth = cv2.resize(depth.astype(np.float32), (IMG_W, IMG_H),
                           interpolation=cv2.INTER_LINEAR)
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

    def __init__(self, engine_path: str, flight_dir: str, sim: bool = False,
                 sim_image_topic: str = SIM_IMAGE_TOPIC, no_save: bool = False):
        super().__init__('freerider')

        self._sim        = sim
        self._no_save    = no_save
        self._flight_dir = flight_dir
        self._frames_dir = os.path.join(flight_dir, 'frames')
        self._depth_dir  = os.path.join(flight_dir, 'depth')
        os.makedirs(self._frames_dir, exist_ok=True)
        os.makedirs(self._depth_dir,  exist_ok=True)

        self.state          = State()
        self._left_rc_modes = False
        self._yaw           = 0.0
        self._alt           = 0.0    # altitude from odometry (ENU, metres above home)
        self._target_alt    = None   # set after offboard entry; P controller holds this altitude
        self._latest_bgr    = None   # used in sim mode only
        self._frame_lock    = threading.Lock()
        self._camera        = None   # Camera instance (real hardware)
        self._step_count    = 0

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
        self.vel_pub     = self.create_publisher(TwistStamped, '/mavros/setpoint_velocity/cmd_vel', 10)
        self.tune_pub    = self.create_publisher(PlayTuneV2,   '/mavros/play_tune', 10)

        if os.path.exists(DEPTH_ENGINE_PATH):
            self.get_logger().info(f'[Freerider] Depth TRT: {DEPTH_ENGINE_PATH}')
            self._depth = DepthEstimatorTRT(DEPTH_ENGINE_PATH)
        else:
            self.get_logger().warn(
                f'[Freerider] Depth TRT engine not found — using HuggingFace (~700ms/step). '
                f'Run export_depth_trt.py to build it.'
            )
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._depth = DepthEstimator(device=device)
        self._trt = TRTEngine(engine_path)
        self.get_logger().info(f'[Freerider] Engine loaded: {engine_path}')

        log_path         = os.path.join(flight_dir, 'flight.csv')
        self._log_file   = open(log_path, 'w', newline='')
        self._log_writer = csv.writer(self._log_file)
        self._log_writer.writerow([
            't', 'raw_action', 'smoothed_action',
            'forward_vel', 'lateral_vel', 'step_latency_ms',
        ])

        if not no_save:
            hud_path         = os.path.join(flight_dir, 'combined.mp4')
            self._hud_writer = cv2.VideoWriter(
                hud_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                SETPOINT_HZ,
                (_DISP_W * 2, _DISP_H + _BAR_H),
            )
        else:
            self._hud_writer = None

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
        self._alt = msg.pose.pose.position.z

    def _on_sim_image(self, msg):
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
        if self.state.mode not in RC_OVERRIDE_MODES:
            self._left_rc_modes = True
            return False
        return self._left_rc_modes

    # ------------------------------------------------------------------
    # Velocity publishing
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

    def _play_tune(self, tune: str):
        """Play a tune on the PX4 buzzer via MAVROS (QB MML format)."""
        msg = PlayTuneV2()
        msg.format = 1   # QBasic 1.1 Music Macro Language
        msg.tune   = tune
        self.tune_pub.publish(msg)

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _start_camera(self):
        if self._sim:
            self.get_logger().info('[Freerider] Sim mode — using ROS image topic.')
            return
        self.get_logger().info('[Freerider] Starting GStreamer camera pipeline...')
        self._camera = Camera()
        self._camera.open()
        self.get_logger().info('[Freerider] Camera pipeline ready.')

    def _get_frame(self) -> np.ndarray | None:
        if self._sim:
            with self._frame_lock:
                return self._latest_bgr.copy() if self._latest_bgr is not None else None
        return self._camera.capture()

    def _stop_camera(self):
        if self._camera is not None:
            self._camera.close()
            self._camera = None

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
    # One policy step
    # ------------------------------------------------------------------

    def _avoidance_step(self, frame_stack: deque, smoothed_action: float,
                        accumulated_state: float, t0: float):
        bgr = self._get_frame()
        if bgr is None:
            return frame_stack, smoothed_action, accumulated_state, 0.0, 0.0, 0.0

        t_step_start = time.monotonic()

        depth = self._depth.estimate(bgr)
        frame_stack.append(depth)
        while len(frame_stack) < N_FRAMES:
            frame_stack.appendleft(frame_stack[0])

        image_np          = np.stack(list(frame_stack), axis=0)
        state_np          = np.array([accumulated_state], dtype=np.float32)

        raw_action        = float(np.clip(self._trt.infer(image_np, state_np), -1.0, 1.0))
        new_smoothed      = SMOOTH_ALPHA * raw_action + ACTION_MOMENTUM * smoothed_action
        new_accumulated   = accumulated_state + new_smoothed

        fwd = FIXED_SPEED - MAX_LATERAL * abs(new_smoothed)
        lat = MAX_LATERAL * new_smoothed
        # Altitude P-controller: correct vertical drift during avoidance.
        if self._target_alt is not None:
            alt_err = self._target_alt - self._alt
            vz = float(np.clip(1.0 * alt_err, -0.5, 0.5))
        else:
            vz = 0.0
        self._publish_vel(vx=fwd, vy=lat, vz=vz)

        step_latency_ms = (time.monotonic() - t_step_start) * 1000.0
        t = time.monotonic() - t0

        self._log_writer.writerow([
            f'{t:.3f}', f'{raw_action:.4f}', f'{new_smoothed:.4f}',
            f'{fwd:.4f}', f'{lat:.4f}', f'{step_latency_ms:.1f}',
        ])
        self._log_file.flush()

        self._step_count += 1
        stamp = f'{self._step_count:06d}'
        print(
            f'  [{stamp}]  raw={raw_action:+.3f}  smooth={new_smoothed:+.3f}  '
            f'acc={new_accumulated:+.2f}  fwd={fwd:.3f}  lat={lat:+.3f}  {step_latency_ms:.0f}ms'
        )
        if not self._no_save:
            if self._step_count % RGB_SAVE_EVERY == 0:
                cv2.imwrite(os.path.join(self._frames_dir, f'{stamp}.jpg'), bgr)
            cv2.imwrite(
                os.path.join(self._depth_dir, f'{stamp}.jpg'),
                (depth * 255).astype(np.uint8),
            )

        if self._hud_writer is not None:
            hud = _draw_hud(bgr, depth, raw_action, new_smoothed,
                            fwd, lat, self._step_count, step_latency_ms)
            self._hud_writer.write(hud)

        return frame_stack, raw_action, new_smoothed, new_accumulated, fwd, lat, step_latency_ms

    # ------------------------------------------------------------------
    # Main run sequence
    # ------------------------------------------------------------------

    def run(self):
        dt = 1.0 / SETPOINT_HZ
        self._left_rc_modes = False

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
                self.get_logger().info('RC override during climb — aborting.')
                return
            if not self.state.armed:
                self.get_logger().error('Disarmed during climb (motors not spinning? PDB off?) — aborting.')
                return
            rclpy.spin_once(self, timeout_sec=0.1)

        self._target_alt = self._alt
        self.get_logger().info(
            f'Takeoff complete (now in {self.state.mode}). '
            f'Altitude locked: {self._target_alt:.2f} m'
        )
        self._play_tune('MFT120L4 O6 CEG')   # ascending 3-note: takeoff done
        self._start_camera()

        result = self._switch_offboard()
        if not result:
            self.get_logger().error('Failed to enter OFFBOARD — aborting.')
            return

        self._play_tune('MFT120L4 O6 CCC')   # three beeps: OFFBOARD active

        latencies         = []
        frame_stack       = deque(maxlen=N_FRAMES)
        smoothed          = 0.0
        accumulated_state = 0.0   # running sum of smoothed actions — state input to actor
        t0                = time.monotonic()

        self.get_logger().info('Freerider avoidance active. RC override to exit.')
        try:
            while True:
                rclpy.spin_once(self, timeout_sec=dt)
                if self._rc_override():
                    self.get_logger().info('RC override detected — stopping.')
                    self._play_tune('MFT120L4 O6 GEC')   # descending 3-note: RC override / stopping
                    break
                if not self.state.armed:
                    self.get_logger().info('Disarmed — stopping.')
                    break
                frame_stack, raw, smoothed, accumulated_state, fwd, lat, latency_ms = self._avoidance_step(
                    frame_stack, smoothed, accumulated_state, t0,
                )
                if latency_ms > 0.0:
                    latencies.append(latency_ms)
        finally:
            self._log_file.close()
            self._stop_camera()
            if latencies:
                self.get_logger().info(
                    f'Step latency — avg: {sum(latencies)/len(latencies):.1f} ms  '
                    f'min: {min(latencies):.1f} ms  '
                    f'max: {max(latencies):.1f} ms  '
                    f'({len(latencies)} steps)'
                )

        self._land()


# ---------------------------------------------------------------------------
# Post-flight plot
# ---------------------------------------------------------------------------

def _plot_flight_log(flight_dir: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    csv_path = os.path.join(flight_dir, 'flight.csv')
    if not os.path.exists(csv_path):
        return

    t, raw, smoothed, fwd, lat, latency = [], [], [], [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            t.append(float(row['t']))
            raw.append(float(row['raw_action']))
            smoothed.append(float(row['smoothed_action']))
            fwd.append(float(row['forward_vel']))
            lat.append(float(row['lateral_vel']))
            latency.append(float(row['step_latency_ms']))

    if not t:
        return

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(os.path.basename(flight_dir))

    axes[0].plot(t, raw,      color='steelblue', label='raw action')
    axes[0].axhline(0, color='gray', linewidth=0.5)
    axes[0].set_ylabel('Raw action')
    axes[0].set_ylim(-1.1, 1.1)
    axes[0].legend()

    axes[1].plot(t, smoothed, color='orange', label='smoothed action')
    axes[1].axhline(0, color='gray', linewidth=0.5)
    axes[1].set_ylabel('Smoothed action')
    axes[1].set_ylim(-1.1, 1.1)
    axes[1].legend()

    axes[2].plot(t, fwd, color='green', label='forward (m/s)')
    axes[2].plot(t, lat, color='red',   label='lateral (m/s)')
    axes[2].axhline(0, color='gray', linewidth=0.5)
    axes[2].set_ylabel('Velocity (m/s)')
    axes[2].legend()

    axes[3].plot(t, latency, color='purple', label='step latency (ms)')
    axes[3].set_ylabel('Latency (ms)')
    axes[3].set_xlabel('Time (s)')
    axes[3].legend()

    plt.tight_layout()
    out = os.path.join(flight_dir, 'flight.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'[plot] saved → {out}')


# ---------------------------------------------------------------------------
# HUD composite frame
# ---------------------------------------------------------------------------

def _draw_hud(bgr: np.ndarray,
              depth: np.ndarray,
              raw_action: float,
              smoothed: float,
              fwd: float,
              lat: float,
              step: int,
              latency_ms: float) -> np.ndarray:
    W, H, BH = _DISP_W, _DISP_H, _BAR_H

    cam = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR)
    cx = W // 2
    tip_x = cx + int(smoothed * W * 0.40)
    tip_y = H // 3
    base_y = H - 20
    mag = abs(smoothed)
    arrow_color = (0, 220, 0) if mag < 0.3 else (0, 200, 220) if mag < 0.6 else (0, 60, 230)
    cv2.arrowedLine(cam, (cx, base_y), (tip_x, tip_y), arrow_color, 3, tipLength=0.25)
    cv2.putText(cam, f'step {step:06d}', (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

    depth_u8  = (depth * 255).astype(np.uint8)
    depth_pan = cv2.resize(cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO), (W, H))
    cv2.putText(depth_pan, 'depth', (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

    bar = np.full((BH, W * 2, 3), 30, dtype=np.uint8)
    bar_cx = W
    bar_y0, bar_y1 = 12, 38
    bar_max_half = W - 40
    cv2.rectangle(bar, (bar_cx - bar_max_half, bar_y0),
                       (bar_cx + bar_max_half, bar_y1), (70, 70, 70), -1)
    fill_w = int(abs(smoothed) * bar_max_half)
    fill_color = (50, 180, 50) if smoothed >= 0 else (50, 50, 200)
    x0 = bar_cx if smoothed >= 0 else bar_cx - fill_w
    cv2.rectangle(bar, (x0, bar_y0), (x0 + fill_w, bar_y1), fill_color, -1)
    cv2.line(bar, (bar_cx, bar_y0 - 2), (bar_cx, bar_y1 + 2), (200, 200, 200), 2)
    txt = (f'raw={raw_action:+.3f}  smooth={smoothed:+.3f}  '
           f'fwd={fwd:.3f}  lat={lat:+.3f}  {latency_ms:.0f}ms')
    cv2.putText(bar, txt, (10, BH - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

    return np.vstack([np.hstack([cam, depth_pan]), bar])


# ---------------------------------------------------------------------------
# Video export
# ---------------------------------------------------------------------------

def _make_video(flight_dir: str, fps: int = SETPOINT_HZ):
    frames_dir = os.path.join(flight_dir, 'frames')
    depth_dir  = os.path.join(flight_dir, 'depth')

    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith('.jpg'))
    if not frame_files:
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    sample = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    fh, fw = sample.shape[:2]

    depth_sample = cv2.imread(os.path.join(depth_dir, frame_files[0]))
    dh, dw = depth_sample.shape[:2] if depth_sample is not None else (fh, fw)

    out_frames = os.path.join(flight_dir, 'frames.mp4')
    out_depth  = os.path.join(flight_dir, 'depth.mp4')
    vw_frames  = cv2.VideoWriter(out_frames, fourcc, fps, (fw, fh))
    vw_depth   = cv2.VideoWriter(out_depth,  fourcc, fps, (dw, dh))

    for fname in frame_files:
        bgr   = cv2.imread(os.path.join(frames_dir, fname))
        depth = cv2.imread(os.path.join(depth_dir,  fname))
        if bgr is not None:
            vw_frames.write(bgr)
        if depth is not None:
            vw_depth.write(depth)

    vw_frames.release()
    vw_depth.release()
    print(f'[video] saved → {out_frames}')
    print(f'[video] saved → {out_depth}')


# ---------------------------------------------------------------------------
# Dropbox sync
# ---------------------------------------------------------------------------

def _sync_dropbox(flight_dir: str, stamp: str):
    dest = f'dropbox:images/flight_freerider_{stamp}'
    print(f'[dropbox] Syncing to {dest} ...')
    try:
        subprocess.run(
            ['rclone', 'copy', flight_dir, dest],
            timeout=120,
        )
        print('[dropbox] Sync complete.')
    except subprocess.TimeoutExpired:
        print('[dropbox] Sync timed out after 120s.')
    except Exception as e:
        print(f'[dropbox] Sync failed: {e}')


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
    parser.add_argument('--sim', action='store_true',
                        help='Sim mode: connect via UDP, skip keypress, use ROS image topic.')
    parser.add_argument('--sim-image-topic', default=SIM_IMAGE_TOPIC,
                        help=f'ROS image topic in sim mode (default: {SIM_IMAGE_TOPIC})')
    parser.add_argument('--no-save', action='store_true',
                        help='Skip saving frames and depth JPEGs (measures pure pipeline latency)')
    args = parser.parse_args()

    stamp      = time.strftime('%Y-%m-%d_%H-%M-%S')
    flight_dir = os.path.join(FLIGHT_LOG_ROOT, stamp)
    os.makedirs(flight_dir, exist_ok=True)

    print('=' * 60)
    print('Freerider — PPO Obstacle Avoidance')
    print(f'Engine     : {args.engine}')
    print(f'Flight log : {flight_dir}')
    if args.sim:
        print(f'Mode       : Simulation (PX4 SITL via {SIM_FCU_URL})')
        print(f'Camera     : {args.sim_image_topic}')
    else:
        print('Mode       : Real hardware')
        print('Camera     : IMX219 fixed-focus (sensor-id=0)')
        print('RC override: switch to ALTCTL or POSCTL at any time')
    print('=' * 60)

    rclpy.init()
    node = FreeriderNode(args.engine, flight_dir, sim=args.sim,
                         sim_image_topic=args.sim_image_topic,
                         no_save=args.no_save)
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
    finally:
        if not node._log_file.closed:
            node._log_file.close()
        if node._hud_writer is not None:
            node._hud_writer.release()
            print(f'[video] saved → {os.path.join(flight_dir, "combined.mp4")}')
        node._stop_camera()
        time.sleep(0.5)
        node.destroy_node()
        rclpy.shutdown()
        _plot_flight_log(flight_dir)
        if not args.no_save:
            _make_video(flight_dir)
        if not args.sim:
            _sync_dropbox(flight_dir, stamp)


if __name__ == '__main__':
    main()
