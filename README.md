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
- ROS/MAVROS for flight control abstraction
- Telemetry radio for ground station communication

## Repository Structure

```
drone-brain/
├── mavros_image_sender/       # Vision processing and image streaming
├── ros_test/                  # ROS node testing and utilities
├── scripts/
│   ├── takeoff_hover.py      # Simple takeoff and hover test
│   └── autonomous_mission.py  # Autonomous waypoint mission
├── models/                    # Trained PPO models (download separately)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Features

- **Vision Processing**: Real-time image capture and preprocessing from Arducam IMX477
- **MAVROS Integration**: ROS wrapper for MAVLink communication with Pixhawk
- **Autonomous Flight**: Takeoff, hover, waypoint navigation, and landing
- **PPO Inference**: Real-time obstacle avoidance using trained reinforcement learning model
- **Safety Features**: Geofencing, battery monitoring, failsafe behaviors

## Prerequisites

### System Requirements

- NVIDIA Jetson Orin Nano with JetPack 5.1+ (Ubuntu 20.04)
- Python 3.8+
- ROS Noetic
- CUDA 11.4+ (included with JetPack)

### Hardware Setup

1. **Physical Connections:**
   - Jetson UART → Pixhawk TELEM2 (57600 baud)
   - Camera CSI connector → Jetson CSI port
   - Power: 12V input to Jetson from drone battery via buck converter

2. **Pixhawk Configuration:**
   - Enable MAVLink on TELEM2: `MAV_1_CONFIG = TELEM2`
   - Set baud rate: `SER_TEL2_BAUD = 57600`
   - Enable companion computer mode: `SYS_COMPANION = Companion Link`

## Installation

### 1. Clone Repository

```bash
cd ~
git clone https://github.com/forest-drone-rl/onboard-stack.git
cd onboard-stack
```

### 2. Install System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install ROS Noetic (if not already installed)
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt install curl
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt update
sudo apt install ros-noetic-desktop-full

# Install MAVROS
sudo apt install ros-noetic-mavros ros-noetic-mavros-extras
wget https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh
sudo bash ./install_geographiclib_datasets.sh

# Install camera utilities
sudo apt install v4l-utils python3-opencv
```

### 3. Install Python Dependencies

```bash
pip3 install -r requirements.txt
```

**requirements.txt includes:**
- `torch` (PyTorch for PPO inference)
- `torchvision`
- `opencv-python`
- `numpy`
- `pymavlink`
- `rospy`
- `stable-baselines3` (PPO implementation)

### 4. Download Trained Models

```bash
# Models are stored separately due to size
# Download from release or Google Drive link
mkdir -p models
cd models
wget https://github.com/forest-drone-rl/onboard-stack/releases/download/v1.0/ppo_forest_nav.zip
unzip ppo_forest_nav.zip
cd ..
```

### 5. Configure MAVROS

Edit `config/mavros_config.yaml` to match your setup:

```yaml
# UART connection to Pixhawk
fcu_url: "/dev/ttyTHS0:57600"
gcs_url: ""
target_system_id: 1
target_component_id: 1
```

**Find your UART device:**
```bash
ls /dev/ttyTHS*  # For Jetson UART
# or
ls /dev/ttyUSB*  # If using USB-to-serial adapter
```

### 6. Set Permissions

```bash
# Allow non-root access to UART
sudo usermod -a -G dialout $USER
sudo chmod 666 /dev/ttyTHS0

# Reboot for changes to take effect
sudo reboot
```

## Usage

### Initialize ROS Environment

```bash
source /opt/ros/noetic/setup.bash
cd ~/onboard-stack
source devel/setup.bash  # If using catkin workspace
```

### 1. Test MAVROS Connection

Start MAVROS and verify connection to Pixhawk:

```bash
roslaunch mavros px4.launch fcu_url:="/dev/ttyTHS0:57600"
```

**Verify connection:**
```bash
# In another terminal
rostopic echo /mavros/state
# Should show: connected: True
```

### 2. Test Camera

```bash
# Check camera detection
v4l2-ctl --list-devices

# Run image sender node
rosrun mavros_image_sender image_capture.py
```

### 3. Simple Takeoff and Hover

**⚠️ Safety: Remove propellers for first test, fly in open area**

```bash
cd ~/onboard-stack/scripts
python3 takeoff_hover.py
```

**What it does:**
1. Arms the drone
2. Takes off to 2 meters altitude
3. Hovers for 30 seconds
4. Lands autonomously

**Parameters (edit in script):**
- `TARGET_ALTITUDE`: Height in meters (default: 2.0)
- `HOVER_DURATION`: Time to hover in seconds (default: 30)

### 4. Autonomous Mission

```bash
python3 autonomous_mission.py
```

**What it does:**
1. Arms and takes off
2. Navigates to predefined waypoints
3. Uses PPO model for obstacle avoidance
4. Returns to launch point
5. Lands

**Mission configuration:**
Edit waypoints in `autonomous_mission.py`:
```python
waypoints = [
    (47.397742, 8.545594, 5),  # lat, lon, alt (meters)
    (47.397850, 8.545700, 5),
    # Add more waypoints
]
```

### 5. Run Full Autonomous Navigation Stack

```bash
# Terminal 1: Start MAVROS
roslaunch mavros px4.launch fcu_url:="/dev/ttyTHS0:57600"

# Terminal 2: Start vision processing
rosrun mavros_image_sender vision_node.py

# Terminal 3: Start autonomous navigation
python3 scripts/autonomous_mission.py --model models/ppo_forest_nav.zip
```

## Configuration

### Camera Settings

Edit `config/camera_config.yaml`:

```yaml
camera:
  width: 640
  height: 480
  framerate: 30
  format: "RGB"
  
preprocessing:
  resize: [128, 128]      # Input size for PPO model
  normalize: true
  grayscale: false
```

### Flight Parameters

Edit in respective scripts or create `config/flight_params.yaml`:

```yaml
takeoff:
  altitude: 2.0           # meters
  ascent_rate: 0.5        # m/s
  
navigation:
  max_speed: 3.0          # m/s
  obstacle_distance: 5.0  # meters (detection range)
  safety_margin: 1.5      # meters (minimum clearance)
  
landing:
  descent_rate: 0.3       # m/s
```

## File Descriptions

### `mavros_image_sender/`

Vision processing module that captures images from the camera and publishes them to ROS topics.

**Key files:**
- `image_capture.py`: Captures frames from CSI camera
- `image_preprocessor.py`: Resizes, normalizes for PPO input
- `vision_node.py`: ROS node for vision pipeline

### `ros_test/`

Testing utilities and example ROS nodes.

**Key files:**
- `mavlink_test.py`: Test MAVLink connection
- `topic_monitor.py`: Monitor ROS topics
- `sensor_check.py`: Verify GPS, IMU, battery data

### `scripts/takeoff_hover.py`

Simple autonomous takeoff and hover script.

**Usage:**
```bash
python3 takeoff_hover.py [--altitude 2.0] [--duration 30]
```

### `scripts/autonomous_mission.py`

Full autonomous mission with PPO-based navigation.

**Usage:**
```bash
python3 autonomous_mission.py --model models/ppo_forest_nav.zip --waypoints mission1.yaml
```

## Safety Features

### Pre-flight Checks

The system performs automatic pre-flight checks:
- ✅ GPS 3D fix (>10 satellites)
- ✅ Battery voltage >14.5V (for 4S LiPo)
- ✅ Pixhawk armed and ready
- ✅ Camera functional
- ✅ PPO model loaded successfully
- ✅ Geofence configured

### Failsafes

- **Low battery**: Automatic return-to-launch (RTL) at 20% remaining
- **GPS loss**: Hover in place, switch to Altitude mode
- **Communication loss**: RTL after 10 seconds
- **Obstacle detection failure**: Immediate landing
- **Manual RC override**: Always available via transmitter

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

Apache 2 License - See LICENSE file for details.

## Acknowledgments

- Kristianstad University for resources and support
- BeetleSense for domain expertise and motivation
- PX4 and MAVROS communities for open-source flight control tools

## Citation

If you use this work in research, please cite:

```bibtex
@mastersthesis{rosato2026forest,
  title={Autonomous Drone Navigation in Forest Environments Using PPO Reinforcement Learning},
  author={Rosato, Elina and Maizelis, Danielis},
  year={2026},
  school={Kristianstad University},
  address={Kristianstad, Sweden}
}
```

---

**Last Updated**: March 2026  
**Version**: 1.0.0
