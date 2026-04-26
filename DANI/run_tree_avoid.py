#!/usr/bin/env python3

import csv
import os
import select
import subprocess
import sys
import termios
import threading
import time
import tty

import numpy as np
import cv2
import torch
import torch.nn.functional as F
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

ENGINE_PATH    = '/home/beetlesniffer/PythonProjects/DANI/models/avoidance_policy.trt'
SETPOINT_HZ    = 10
PRESTREAM_TIME = 2.0

GST_PIPELINE = (
    'nvarguscamerasrc ! '
    'video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1 ! '
    'nvvidconv ! video/x-raw, format=BGRx ! '
    'videoconvert ! video/x-raw, format=BGR ! appsink'
)

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
    def __init__(self):
        super().__init__('avoidance_node')

        torch.backends.cudnn.enabled = False
        self._init_trt()
        self._init_midas()
        self.latest_rgb     = None
        self.latest_depth   = None
        self.delayed_action = 0.0
        self.prev_action    = 0.0

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

        self._debug_dir  = '/home/beetlesniffer/PythonProjects/DANI/debug_frames'
        self._flight_dir = os.path.join(self._debug_dir, f'flight_{time.strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(self._flight_dir, exist_ok=True)
        self._recording  = False
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

        self.get_logger().info(f'TRT engine loaded: {ENGINE_PATH}')

    def _init_midas(self):
        import sys
        MIDAS_PATH = '/home/beetlesniffer/.cache/torch/hub/intel-isl_MiDaS_master'
        if MIDAS_PATH not in sys.path:
            sys.path.insert(0, MIDAS_PATH)

        self.get_logger().info('Loading MiDaS_small...')
        from midas.midas_net_custom import MidasNet_small
        from midas.transforms import Resize, NormalizeImage, PrepareForNet

        self.midas = MidasNet_small(
            None, features=64, backbone='efficientnet_lite3',
            exportable=True, non_negative=True, blocks={'expand': True}
        )
        weights_path = '/home/beetlesniffer/.cache/torch/hub/checkpoints/midas_v21_small_256.pt'
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
        self.midas.load_state_dict(state_dict)
        self.midas.eval().cuda()

        self.midas_transform = self._make_small_transform(Resize, NormalizeImage, PrepareForNet)
        self.get_logger().info('MiDaS ready.')

    @staticmethod
    def _make_small_transform(Resize, NormalizeImage, PrepareForNet):
        import cv2
        def transform(img):
            sample = {"image": img / 255.0}
            sample = Resize(256, 256, resize_target=None, keep_aspect_ratio=True,
                            ensure_multiple_of=32, resize_method="upper_bound",
                            image_interpolation_method=cv2.INTER_CUBIC)(sample)
            sample = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(sample)
            sample = PrepareForNet()(sample)
            return torch.from_numpy(sample["image"]).unsqueeze(0)
        return transform

    def _debug_image_saver(self):
        while True:
            time.sleep(1.0)
            if self.latest_depth is None:
                continue
            fname = os.path.join(self._flight_dir, f'{time.strftime("%H%M%S")}.jpg')
            cv2.imwrite(fname, (self.latest_depth[0] * 255).astype(np.uint8))

    def _start_recording(self):
        self._recording = True
        threading.Thread(target=self._record_video, daemon=True).start()

    def _record_video(self):
        cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.get_logger().error('Cannot open camera for recording')
            return
        fname = os.path.join(self._debug_dir, f'{time.strftime("%H%M%S")}.mp4')
        writer = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1080))
        self.get_logger().info(f'Recording to {fname}')
        while self._recording:
            ret, frame = cap.read()
            if ret:
                writer.write(frame)
        writer.release()
        cap.release()
        self.get_logger().info('Recording saved.')
        subprocess.Popen(
            ['rclone', 'copy', self._debug_dir, 'dropbox:DANI/debug_frames'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    # --------------------------------------------------------------- callbacks

    def _camera_reader(self):
        cap = None
        while cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.get_logger().warn('Camera not available — retrying in 1s...')
                time.sleep(1.0)
        self.get_logger().info('Camera opened.')

        last_depth    = 0.0
        depth_period  = 1.0 / SETPOINT_HZ
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            rgb = frame[:, :, ::-1].copy()
            self.latest_rgb = cv2.resize(rgb, (192, 192), interpolation=cv2.INTER_AREA)
            now = time.monotonic()
            if now - last_depth >= depth_period:
                self.latest_depth = self._depth_from_rgb(self.latest_rgb.copy())
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

    def _depth_from_rgb(self, rgb: np.ndarray) -> np.ndarray:
        inp  = self.midas_transform(rgb).cuda()  # transform returns CPU tensor, move to GPU
        with torch.no_grad():
            pred = self.midas(inp)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        pred  = F.interpolate(pred, (192, 192), mode='bilinear', align_corners=False)
        pred  = pred.squeeze(1)
        mn, mx = pred.min(), pred.max()
        depth = (pred - mn) / (mx - mn + 1e-8)
        arr = depth.cpu().numpy().astype(np.float32)
        self.get_logger().info(f'depth min={arr.min():.3f} max={arr.max():.3f} mean={arr.mean():.3f}')
        return arr

    def _policy_infer(self, depth: np.ndarray) -> float:
        self._cuda_ctx.push()
        try:
            obs = depth[np.newaxis]
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
        if self.latest_rgb is None:
            self._publish_vel()
            return

        if self._flight_t0 is None:
            self._flight_t0 = time.monotonic()
        if self._cruise_alt is None and self._alt is not None:
            self._cruise_alt = self._alt

        new_action = self._policy_infer(self.latest_depth)

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
        threading.Thread(target=self._camera_reader, daemon=True).start()
        # self._start_recording()

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
            ['rclone', 'copy', debug_dir, 'dropbox:DANI/debug_frames'],
            timeout=120,
        )
        print('Dropbox sync complete.')
    except subprocess.TimeoutExpired:
        print('Dropbox sync timed out after 120s.')
    except Exception as e:
        print(f'Dropbox sync failed: {e}')


def main():
    print('=' * 50)
    print('Avoidance policy — arm → takeoff 1.5 m → OFFBOARD → run until RC override')
    print('RC override : switch to ALTCTL or POSCTL at any time')
    print('=' * 50)

    rclpy.init()
    node = AvoidanceNode()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
    finally:
        node._recording = False
        node._log_file.close()
        _plot_flight_log(node._log_path)
        _sync_dropbox(node._debug_dir)
        time.sleep(0.5)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
