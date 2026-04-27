# Drone Brain

Autonomous navigation system for forest drone operations. Runs on NVIDIA Jetson Orin Nano for real-time vision processing, PPO-based obstacle avoidance, and flight control via PX4 / MAVROS.

## Hardware

| Component | Part |
|---|---|
| Companion computer | NVIDIA Jetson Orin Nano 8 GB |
| Frame | Holybro S500 V2 Quadcopter |
| Flight controller | Pixhawk 6C |
| Camera | Arducam IMX477 (12 MP, 100° FOV) |

**Communication:** MAVLink over UART (Jetson ↔ Pixhawk, `/dev/ttyTHS1` at 115200 baud). ROS 2 Humble + MAVROS for flight control. Telemetry radio for ground station.

## Repository Structure

```
drone-brain/
├── models/
│   ├── freerider/                 # Depth 2.0 based tree-avoidance policy
│   │   ├── helpers/
│   │   │   └── export_freerider_trt.py
│   │   └── run_freerider.py
│   └── fortune_cookie/            # MiDaS-based tree-avoidance policy
│       ├── helpers/
│       │   └── export_midas_trt.py
│       ├── model/
│       │   ├── avoidance_policy.onnx
│       │   ├── avoidance_policy.onnx.data
│       │   └── avoidance_policy.trt
│       ├── run_tree_avoid.py
│       └── tests/
│           └── test_forward.py
├── tests/
│   ├── flight_tests/              # Scripts that perform actual flight commands
│   │   ├── takeoff.py
│   │   ├── land.py
│   │   ├── takeoff_hover_land.py
│   │   ├── takeoff_move_land.py
│   │   ├── pitch_roll_yaw.py
│   │   ├── gps_goto.py
│   │   └── goal.json
│   └── sensor_tests/              # Hardware and model verification (no flight)
│       ├── autofocus_test.py
│       ├── capture_test.py
│       ├── motor_test.py
│       └── probe_baud.py
├── tools/                         # Utilities to run alongside missions
│   ├── telemetry_monitor.py
│   ├── image_sender.py
│   ├── camera_stream.py
│   ├── midas_viewer.py
│   ├── timing_test_midas.py
│   ├── timing_test_zoe.py
│   ├── visualizer.py
│   ├── visualizer.rviz
│   └── vnc_rviz_setup.md
└── airsim/
    └── settings.json              # AirSim config for PX4 SITL
```

---

## Models

### Freerider (`models/freerider/`)

PPO-based obstacle avoidance policy trained in AirSim. Uses Depth Anything V2 Small for monocular depth estimation and a TensorRT FP16 engine for inference on the Jetson.

| File | Description |
|---|---|
| `helpers/export_freerider_trt.py` | Converts a training `.zip` checkpoint to ONNX then TRT |
| `run_freerider.py` | Main flight script — real hardware or `--sim` mode |


### Fortune Cookie (`models/fortune_cookie/`)

MiDaS-based tree-avoidance policy. Pre-built TRT engine included.

| File | Description |
|---|---|
| `helpers/export_midas_trt.py` | Converts MiDaS ONNX to TRT on the Jetson |
| `run_tree_avoid.py` | Flight script for the MiDaS avoidance policy |
| `tests/test_forward.py` | Forward-pass sanity check for the ONNX model |

---

## Tests

### Flight tests (`tests/flight_tests/`)

Scripts that arm the drone and perform actual flight manoeuvres. All require MAVROS running.

| Script | Description |
|---|---|
| `takeoff.py` | Arms and climbs to target altitude |
| `land.py` | Initiates AUTO.LAND and waits for disarm |
| `takeoff_hover_land.py` | Full arm → climb → hover → land sequence |
| `takeoff_move_land.py` | Takeoff, fly to a waypoint, land |
| `pitch_roll_yaw.py` | Open-loop attitude command test |
| `gps_goto.py` | Fly to GPS coordinates with EKF local position and quality gate |
| `goal.json` | Target coordinates used by `gps_goto.py` |

### Sensor tests (`tests/sensor_tests/`)

Hardware and model verification — no motors armed.

| Script | Description |
|---|---|
| `autofocus_test.py` | Checks Arducam IMX477 autofocus over I2C |
| `capture_test.py` | Grabs a frame via GStreamer and saves it to disk |
| `motor_test.py` | Spins individual motors via MAVROS for direction/order check |
| `probe_baud.py` | Scans UART baud rates to find the Pixhawk connection |

---

## Tools

Standalone utilities to run in a separate terminal alongside flight scripts.

| Script | Description |
|---|---|
| `telemetry_monitor.py` | Live GPS, IMU, or battery readout from MAVROS (`-gps`, `-imu`, `-battery`) |
| `image_sender.py` | Captures an IMX477 frame and sends it to the GCS over MAVROS Tunnel messages |
| `camera_stream.py` | Streams the IMX477 feed over a network socket for remote viewing |
| `midas_viewer.py` | Displays live MiDaS depth output on-screen for visual inspection |
| `timing_test_midas.py` | Benchmarks MiDaS inference latency on the Jetson |
| `timing_test_zoe.py` | Benchmarks ZoeDepth inference latency on the Jetson |
| `visualizer.py` | RViz helper — publishes sensor data as ROS topics for the `.rviz` config |
| `visualizer.rviz` | RViz layout for the visualizer |
| `vnc_rviz_setup.md` | Guide for running RViz remotely over VNC |

---

## Prerequisites

### Hardware setup

1. **Physical connections:**
   - Jetson UART1 (`/dev/ttyTHS1`) → Pixhawk TELEM2
   - Camera CSI ribbon → Jetson CSI port
   - Power from drone battery bus → Jetson power input

2. **Pixhawk configuration:**
   - Enable MAVLink on TELEM2: `MAV_1_CONFIG = TELEM2`
   - Set baud rate: `SER_TEL2_BAUD = 115200`

### Software

- NVIDIA Jetson Orin Nano with JetPack 6.x (Ubuntu 22.04)
- ROS 2 Humble
- MAVROS for ROS 2

```bash
sudo apt update
sudo apt install ros-humble-mavros ros-humble-mavros-extras
wget https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh
sudo bash ./install_geographiclib_datasets.sh
```

**UART permissions:**

```bash
sudo usermod -a -G dialout $USER
# Log out and back in for the group change to take effect
```

**Python dependencies:**

```bash
pip install -r requirements.txt
```

### MAVROS — real hardware

```bash
ros2 launch mavros px4.launch fcu_url:=/dev/ttyTHS1:115200
```

### MAVROS — AirSim simulation

```bash
ros2 launch mavros px4.launch fcu_url:=udp://:14540@127.0.0.1:14580
```

---

## AirSim Settings

AirSim reads `~/Documents/AirSim/settings.json`. Copy the repo config before opening AirSim:

```bash
cp ~/drone-brain/airsim/settings.json ~/Documents/AirSim/settings.json
```

This enables the front-center camera used by the freerider policy.

---

## Related Repositories

- **[ground-control-station](https://github.com/autonomous-forest-drone-/ground-control-station)**: Telemetry monitoring on the GCS

---

## Contributing

Thesis project — Kristianstad University.

- **Authors**: Elina Rosato, Danielis Maizelis
- **Supervisor**: Kristianstad University, Sweden
- **Thesis**: "Autonomous Drone Navigation in Forest Environments Using PPO Reinforcement Learning"

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
