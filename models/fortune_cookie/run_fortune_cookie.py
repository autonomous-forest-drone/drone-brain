#!/usr/bin/env python3

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

import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from mavros_msgs.msg import State, StatusText
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode

# Shared camera + focus + MiDaS-TRT helpers — see tools/{jetson_camera,midas_trt}.py
sys.path.insert(0, '/home/beetlesniffer/drone-brain/tools')
from jetson_camera import DEFAULT_FOCUS, GstJpegCapture, set_focus
from midas_trt import TRTMidas

ENGINE_PATH    = '/home/beetlesniffer/drone-brain/models/fortune_cookie/model/jetson_converted_model.trt'
MIDAS_TRT      = '/home/beetlesniffer/drone-brain/models/fortune_cookie/model/midas_small.trt'
SETPOINT_HZ    = 10
PRESTREAM_TIME = 2.0

# Camera + depth pipeline — must match timing_test_midas.py and the policy's
# training-time obs shape: (B, 3, 126, 224), 3 stacked depth frames at 126x224
# spanning 1 s of motion at 10 Hz (idxs [-11, -6, -1] over an 11-frame deque).
CAM_W, CAM_H, CAM_FPS = 640, 480, 30
MIDAS_INPUT    = 256
DEPTH_H        = 126
DEPTH_W        = 224
DEPTH_STACK_N  = 3
DEPTH_STRIDE   = 5
DEPTH_BUFFER_N = (DEPTH_STACK_N - 1) * DEPTH_STRIDE + 1   # 11
STACK_IDXS     = [-1 - i * DEPTH_STRIDE for i in range(DEPTH_STACK_N)][::-1]   # [-11, -6, -1]

RC_OVERRIDE_MODES = {'ALTCTL', 'POSCTL'}
_CMD_INTERVAL     = 2.0
_ARM_TIMEOUT      = 30.0
_TAKEOFF_TIMEOUT  = 30.0
_OFFBOARD_TIMEOUT = 10.0

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def load_engine(path: str) -> trt.ICudaEngine:
    with open(path, 'rb') as f, trt.Runtime(TRT_LOGGER) as rt:
        return rt.deserialize_cuda_engine(f.read())


class AvoidanceNode(Node):
    def __init__(self, focus=DEFAULT_FOCUS):
        super().__init__('avoidance_node')

        self.focus = focus
        self.cap   = None     # opened in _init_camera() before takeoff

        self._init_trt()
        self.midas = TRTMidas(MIDAS_TRT)
        self.latest_rgb       = None
        self.latest_depth_2d  = None       # most recent (DEPTH_H, DEPTH_W) for debug
        self.latest_obs       = None       # (3, DEPTH_H, DEPTH_W) — fed to policy
        self.depth_buffer     = deque(maxlen=DEPTH_BUFFER_N)
        self.delayed_action   = 0.0
        self.prev_action      = 0.0

        self.state           = State()
        self._left_rc_modes  = False
        self._yaw            = 0.0
        self._alt            = None
        self._cruise_alt     = None

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        state_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.create_subscription(State,    '/mavros/state',                    self._on_state,      state_qos)
        self.create_subscription(StatusText, '/mavros/statustext',             self._on_statustext, qos)
        self.create_subscription(Odometry, '/mavros/local_position/odom',      self._on_odom,       qos)

        self.arm_client  = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode,     '/mavros/set_mode')
        self.land_client = self.create_client(CommandTOL,  '/mavros/cmd/land')

        self.vel_pub = self.create_publisher(TwistStamped, '/mavros/setpoint_velocity/cmd_vel', 10)

        self._debug_dir  = '/home/beetlesniffer/drone-brain/images'
        self._flight_dir = os.path.join(self._debug_dir, f'flight_fortune_cookie{time.strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(self._flight_dir, exist_ok=True)
        self._flight_t0  = None
        threading.Thread(target=self._debug_image_saver, daemon=True).start()
        self._log_path   = os.path.join(self._flight_dir, 'flight.csv')
        self._log_file   = open(self._log_path, 'w', newline='')
        self._log_writer = csv.writer(self._log_file)
        self._log_writer.writerow(['t', 'raw_action', 'smoothed', 'forward', 'lateral'])

    # ------------------------------------------------------------------ init

    def _init_trt(self):
        self.engine    = load_engine(ENGINE_PATH)
        self.trt_ctx   = self.engine.create_execution_context()  # renamed — 'context' is reserved by Node
        self.stream    = cuda.Stream()
        self._cuda_ctx = cuda.Context.get_current()

        self._bufs = {}
        for i in range(self.engine.num_io_tensors):
            name  = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            mode  = self.engine.get_tensor_mode(name)
            h_buf = cuda.pagelocked_empty(int(trt.volume(shape)), dtype=dtype)
            d_buf = cuda.mem_alloc(h_buf.nbytes)
            self._bufs[name] = {'h': h_buf, 'd': d_buf, 'mode': mode}
            # Log shapes so a drift between engine and obs shape surfaces immediately
            # (cryptic np.copyto length errors otherwise).
            self.get_logger().info(f"[trt] {mode.name.lower()} '{name}' shape={tuple(shape)} dtype={dtype.__name__}")

        self.get_logger().info(f'TRT engine loaded: {ENGINE_PATH}')

    def _debug_image_saver(self):
        while True:
            time.sleep(1.0)
            if self.latest_depth_2d is None:
                continue
            fname = os.path.join(self._flight_dir, f'{time.strftime("%H%M%S")}.jpg')
            cv2.imwrite(fname, (self.latest_depth_2d * 255).astype(np.uint8))

    # --------------------------------------------------------------- callbacks

    def _camera_reader(self):
        if self.cap is None or not self.cap.isOpened():
            self.get_logger().error('Camera not opened — _init_camera must be called first.')
            return

        last_depth   = 0.0
        depth_period = 1.0 / SETPOINT_HZ
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            # resize to MiDaS input then BGR→RGB on the small image (cheaper)
            small = cv2.resize(frame, (MIDAS_INPUT, MIDAS_INPUT), interpolation=cv2.INTER_LINEAR)
            self.latest_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            now = time.monotonic()
            if now - last_depth >= depth_period:
                d2 = self.midas.infer(self.latest_rgb, out_size=(DEPTH_H, DEPTH_W))[0]   # (DEPTH_H, DEPTH_W)
                self.latest_depth_2d = d2
                # pre-fill the buffer the first time so stacking works immediately
                if not self.depth_buffer:
                    for _ in range(DEPTH_BUFFER_N):
                        self.depth_buffer.append(d2)
                else:
                    self.depth_buffer.append(d2)
                self.latest_obs = np.stack([self.depth_buffer[i] for i in STACK_IDXS], axis=0)
                last_depth = now

    def _on_state(self, msg: State):
        self.state = msg

    def _on_statustext(self, msg: StatusText):
        self.get_logger().info(f'[PX4] {msg.text}')

    def _on_odom(self, msg: Odometry):
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._yaw = np.arctan2(siny_cosp, cosy_cosp)
        self._alt = msg.pose.pose.position.z

    # --------------------------------------------------------------- RC override

    def _rc_override(self) -> bool:
        if self.state.mode not in RC_OVERRIDE_MODES:
            self._left_rc_modes = True
            return False
        return self._left_rc_modes

    # --------------------------------------------------------------- velocity

    def _publish_vel(self, vx=0.0, vy=0.0, vz=0.0):
        """Body frame input — rotated to ENU before publishing."""
        yaw = self._yaw
        msg = TwistStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.twist.linear.x  = vx * np.cos(yaw) - vy * np.sin(yaw)
        msg.twist.linear.y  = vx * np.sin(yaw) + vy * np.cos(yaw)
        msg.twist.linear.z  = vz
        self.vel_pub.publish(msg)

    # --------------------------------------------------------------- inference

    def _policy_infer(self, obs_stack: np.ndarray) -> float:
        """obs_stack shape: (DEPTH_STACK_N, DEPTH_H, DEPTH_W) → batch to (1, ...)."""
        self._cuda_ctx.push()
        try:
            obs = obs_stack[np.newaxis].astype(np.float32, copy=False)
            for name, buf in self._bufs.items():
                if buf['mode'] == trt.TensorIOMode.INPUT:
                    np.copyto(buf['h'], obs.ravel())
                    cuda.memcpy_htod_async(buf['d'], buf['h'], self.stream)
                self.trt_ctx.set_tensor_address(name, int(buf['d']))

            self.trt_ctx.execute_async_v3(self.stream.handle)

            for name, buf in self._bufs.items():
                if buf['mode'] == trt.TensorIOMode.OUTPUT:
                    cuda.memcpy_dtoh_async(buf['h'], buf['d'], self.stream)
            self.stream.synchronize()

            for name, buf in self._bufs.items():
                if buf['mode'] == trt.TensorIOMode.OUTPUT:
                    return float(np.clip(buf['h'][0], -1.0, 1.0))
        finally:
            self._cuda_ctx.pop()
        return 0.0

    # --------------------------------------------------------------- avoidance step

    def _avoidance_step(self):
        if self.latest_obs is None:
            self._publish_vel()
            return

        if self._flight_t0 is None:
            self._flight_t0 = time.monotonic()
        if self._cruise_alt is None and self._alt is not None:
            self._cruise_alt = self._alt

        new_action = self._policy_infer(self.latest_obs)

        smoothed         = 0.7 * new_action + 0.3 * self.prev_action
        self.prev_action = smoothed

        forward = 1.5 * (1.0 - abs(smoothed))
        lateral = 1.5 * smoothed

        vz = 0.0

        t = time.monotonic() - self._flight_t0
        self._log_writer.writerow([f'{t:.3f}', f'{new_action:.4f}', f'{smoothed:.4f}',
                                   f'{forward:.4f}', f'{lateral:.4f}'])
        self._log_file.flush()

        self._publish_vel(vx=forward, vy=lateral, vz=vz)

    # --------------------------------------------------------------- mission commands

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
                result = future.result() if future.done() else None
                if not (result and result.success):
                    self.get_logger().warn('ARM: no ACK — retrying...')
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
                result = future.result() if future.done() else None
                if not (result and result.mode_sent):
                    self.get_logger().warn('SET_MODE AUTO.TAKEOFF: no ACK — retrying...')
                last_send = time.monotonic()
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._rc_override():
                self.get_logger().info(f'RC override ({self.state.mode}) during takeoff.')
                return None
            if self.state.mode == 'AUTO.TAKEOFF':
                self.get_logger().info('AUTO.TAKEOFF confirmed.')
                return True
            if not self.state.armed:
                self.get_logger().error('Disarmed before takeoff confirmed.')
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
                result = future.result() if future.done() else None
                if not (result and result.mode_sent):
                    self.get_logger().warn('SET_MODE OFFBOARD: no ACK — retrying...')
                last_send = time.monotonic()
            rclpy.spin_once(self, timeout_sec=dt)
            if self._rc_override():
                self.get_logger().info(f'RC override ({self.state.mode}) during OFFBOARD switch.')
                return None
            if self.state.mode == 'OFFBOARD':
                self.get_logger().info('OFFBOARD confirmed.')
                return True
        return False

    def _set_posctl(self):
        req = SetMode.Request()
        req.custom_mode = 'POSCTL'
        future = self.mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

    # --------------------------------------------------------------- camera

    def _init_camera(self):
        """Open the CSI camera and lock focus. Must run before _camera_reader.
        VCM only accepts I2C writes while Argus is streaming, so the focus-set
        comes after the pipeline is up."""
        self.cap = GstJpegCapture(CAM_W, CAM_H, CAM_FPS)
        set_focus(self.focus)
        time.sleep(0.3)   # VCM lens travel
        self.get_logger().info(
            f'Camera streaming {CAM_W}x{CAM_H}@{CAM_FPS} fps  |  focus locked at {self.focus}'
        )

    # --------------------------------------------------------------- mission

    def run(self):
        dt = 1.0 / SETPOINT_HZ
        self._left_rc_modes = False

        deadline = time.monotonic() + 2.0
        while not self.state.connected and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

        if not self.state.connected:
            self.get_logger().info('MAVROS not connected — launching...')
            subprocess.Popen(
                ['ros2', 'launch', 'mavros', 'px4.launch', 'fcu_url:=/dev/ttyTHS1:115200'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

        self.get_logger().info('Waiting for MAVROS connection...')
        while not self.state.connected:
            rclpy.spin_once(self, timeout_sec=0.1)

        # Bring the camera up before the operator commits to arm — gives them
        # a chance to bail if the focus or pipeline is bad, and pre-warms the
        # depth ring buffer so the policy has a full obs by the time we hit OFFBOARD.
        self._init_camera()
        threading.Thread(target=self._camera_reader, daemon=True).start()

        print('=' * 60)
        print(f'  Camera   : streaming {CAM_W}x{CAM_H} @ {CAM_FPS} fps')
        print(f'  Focus    : {self.focus} (locked, override with --focus N)')
        print(f'  Engine   : {os.path.basename(ENGINE_PATH)}')
        print(f'  Obs shape: (1, {DEPTH_STACK_N}, {DEPTH_H}, {DEPTH_W})')
        print('=' * 60)
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
        if result is None:
            return
        if not result:
            self.get_logger().error('Failed to set AUTO.TAKEOFF — aborting.')
            return

        self.get_logger().info('Climbing...')
        while self.state.mode == 'AUTO.TAKEOFF':
            if not self.state.armed:
                self.get_logger().error('Disarmed during climb — aborting.')
                return
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info(f'Takeoff complete (now in {self.state.mode}).')

        result = self._switch_offboard()
        if result is None:
            return
        if not result:
            self.get_logger().error('Failed to enter OFFBOARD — aborting.')
            return

        self._left_rc_modes = False
        self.get_logger().info('Running avoidance policy indefinitely. RC override to stop.')
        while rclpy.ok():
            self._avoidance_step()
            rclpy.spin_once(self, timeout_sec=dt)
            if self._rc_override():
                self.get_logger().info(f'RC override ({self.state.mode}) — handing off.')
                return
            if not self.state.armed:
                self.get_logger().info('Disarmed — exiting.')
                return

        self._set_posctl()


def _plot_flight_log(csv_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not os.path.exists(csv_path):
        return

    t, raw, smoothed, forward, lateral = [], [], [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(float(row['t']))
            raw.append(float(row['raw_action']))
            smoothed.append(float(row['smoothed']))
            forward.append(float(row['forward']))
            lateral.append(float(row['lateral']))

    if not t:
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(os.path.basename(os.path.dirname(csv_path)))

    axes[0].plot(t, raw, label='raw action', color='steelblue')
    axes[0].axhline(0, color='gray', linewidth=0.5)
    axes[0].set_ylabel('Raw action')
    axes[0].set_ylim(-1.1, 1.1)
    axes[0].legend()

    axes[1].plot(t, smoothed, label='smoothed action', color='orange')
    axes[1].axhline(0, color='gray', linewidth=0.5)
    axes[1].set_ylabel('Smoothed action')
    axes[1].set_ylim(-1.1, 1.1)
    axes[1].legend()

    axes[2].plot(t, forward, label='forward (m/s)', color='green')
    axes[2].plot(t, lateral, label='lateral (m/s)', color='red')
    axes[2].axhline(0, color='gray', linewidth=0.5)
    axes[2].set_ylabel('Velocity (m/s)')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend()

    plt.tight_layout()
    plot_path = csv_path.replace('.csv', '.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f'Plot saved: {plot_path}')


def _sync_dropbox(debug_dir):
    print('Syncing to Dropbox...')
    try:
        subprocess.run(
            ['rclone', 'copy', debug_dir, 'dropbox:fortune_cookie/images'],
            timeout=120,
        )
        print('Dropbox sync complete.')
    except subprocess.TimeoutExpired:
        print('Dropbox sync timed out after 120s.')
    except Exception as e:
        print(f'Dropbox sync failed: {e}')


def main():
    parser = argparse.ArgumentParser(
        description='Avoidance policy: arm → takeoff 1.5 m → OFFBOARD → run until RC override.'
    )
    parser.add_argument('--focus', type=int, default=DEFAULT_FOCUS,
                        help=f'VCM focus value 0-1000 (default {DEFAULT_FOCUS}).')
    args = parser.parse_args()

    print('=' * 60)
    print('Avoidance policy — arm → takeoff 1.5 m → OFFBOARD → run until RC override')
    print('RC override : switch to ALTCTL or POSCTL at any time')
    print('=' * 60)

    rclpy.init()
    node = AvoidanceNode(focus=args.focus)
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
    finally:
        if node.cap is not None:
            node.cap.release()
        node._log_file.close()
        _plot_flight_log(node._log_path)
        _sync_dropbox(node._debug_dir)
        time.sleep(0.5)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
