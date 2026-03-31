# Drone Brain

Autonomous navigation system for forest drone operations. Runs on NVIDIA Jetson Orin Nano for real-time vision processing, PPO-based obstacle avoidance, and flight control.

## Overview

This repository contains the onboard companion computer software for an autonomous drone capable of navigating forest environments. The system uses Proximal Policy Optimization (PPO) reinforcement learning for collision avoidance and autonomous navigation.

**Hardware:**
- NVIDIA Jetson Orin Nano (8GB)
- Holybro S500 V2 Quadcopter Frame
- Pixhawk 6C Flight Controller
- Arducam IMX477 Camera (12MP, 100° FOV)

**Communication:**
- MAVLink protocol via UART (Jetson ↔ Pixhawk)
- ROS 2 / MAVROS for flight control abstraction
- Telemetry radio for ground station communication

## Repository Structure

```
drone-brain/
├── missions/          # Standalone flight scripts — one file per mission
├── helpers/           # Utilities to run alongside missions in separate terminals
└── docs/              # Documentation for each mission
```

## Missions

Each mission is a self-contained script that manages arming, flight, and landing. Full setup and usage instructions for each mission are in `docs/`.

| Script | Description | Docs |
|---|---|---|
| `missions/hover_test.py` | Arms, climbs to 1.5 m, hovers for 3 s, lands via `AUTO.LAND`. Triggered by RC ch5 on real hardware or `--airsim` flag in simulation. | [docs/hover_test.md](docs/hover_test.md) |
| `missions/hover_rc_override.py` | Holds hover by publishing RC override channels. Pilot reclaims control by moving any stick past the threshold. | — |

### Running a mission

All missions follow the same pattern.

**Real hardware** (all terminals on the Jetson):

```bash
# Terminal 1
ros2 launch mavros px4.launch fcu_url:=/dev/ttyACM0

# Terminal 2
python3 missions/<mission>.py
```

**AirSim simulation** (all terminals on the GCS machine — where UE5 and AirSim are installed):

```bash
# Terminal 1 — PX4 SITL (from PX4-Autopilot/)
make px4_sitl_default none_iris

# Terminal 2 — MAVROS
ros2 launch mavros px4.launch fcu_url:=udp://:14540@127.0.0.1:14580

# Terminal 3
python3 missions/<mission>.py --airsim
```

See the relevant doc in `docs/` for mission-specific details and what to expect.

## Helpers

Helpers are standalone tools you run in a separate terminal alongside a mission. They do not control flight — they monitor, test, or send data.

---

### `helpers/telemetry_monitor.py`

Live readout of GPS, IMU, or battery data from the Pixhawk via MAVROS.

**Requires MAVROS running.**

```bash
python3 helpers/telemetry_monitor.py -gps
python3 helpers/telemetry_monitor.py -imu
python3 helpers/telemetry_monitor.py -battery
```

Output updates in place on a single line, e.g.:

```
🛰  Lat: 55.123456  Lon: 14.654321  Alt: 12.34m  Status: FIX
```

---

### `helpers/motor_control.py`

Direct motor commands for bench testing. Bypasses the flight controller mixing (group 3 — direct motor control). Useful for verifying motor direction and order without flying.

**Requires MAVROS running.**

```bash
python3 helpers/motor_control.py manual              # switch to MANUAL mode
python3 helpers/motor_control.py arm                 # arm the vehicle
python3 helpers/motor_control.py disarm              # stop motors and disarm
python3 helpers/motor_control.py spin <throttle>     # spin all motors (0.0–1.0) for 3 s
python3 helpers/motor_control.py motor <idx> <throttle>  # spin one motor (index 0–7) for 3 s
python3 helpers/motor_control.py stop                # set all motors to 0
```

---

### `helpers/image_sender.py`

Captures a frame from the Arducam IMX477 via GStreamer, resizes it to 128×128, and transmits it to the ground station in 128-byte chunks over MAVROS Tunnel messages. Repeats every 20 seconds.

Starts its own MAVROS instance automatically if one is not already running.

```bash
# Capture from the IMX477 camera
python3 helpers/image_sender.py

# Send a local image file instead (for testing without hardware)
python3 helpers/image_sender.py path/to/image.jpg
```

---

## Prerequisites

### Hardware setup

1. **Physical connections:**
   - Jetson UART → Pixhawk TELEM2 (57600 baud)
   - Camera CSI connector → Jetson CSI port
   - Power: input to Jetson from drone battery

2. **Pixhawk configuration:**
   - Enable MAVLink on TELEM2: `MAV_1_CONFIG = TELEM2`
   - Set baud rate: `SER_TEL2_BAUD = 57600`

### Software

- NVIDIA Jetson Orin Nano with JetPack 5.1+ (Ubuntu 22.04)
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

## AirSim settings

AirSim reads `~/Documents/AirSim/settings.json`. Use the following config:

| File | Used for |
|---|---|
| `~airsim/settings.json` | PX4 SITL / mission testing |

Update before opening AirSim:

```bash
cp ~/drone-brain/airsim/settings.json ~/Documents/AirSim/settings.px4.json
```

## Related Repositories

- **[ground-control-station](https://github.com/autonomous-forest-drone-/ground-control-station)**: Telemetry monitoring
- **[ppo-training](https://github.com/autonomous-forest-drone-/ppo-training)**: Model training pipeline with AirSim

## Contributing

This is a thesis project by Elina Rosato and Danielis Maizelis (Kristianstad University). For questions or collaboration:

- **Authors**: Elina Rosato, Danielis Maizelis
- **University**: Kristianstad University, Sweden
- **Thesis**: "Autonomous Drone Navigation in Forest Environments Using PPO Reinforcement Learning"
- **Industry Partner**: BeetleSense / Christo Van Zyl (Lund University)

## License

Apache 2 License — see LICENSE for details.

## Acknowledgments

- Kristianstad University for resources and support
- BeetleSense for domain expertise and motivation
- PX4 and MAVROS communities for open-source flight control tools
