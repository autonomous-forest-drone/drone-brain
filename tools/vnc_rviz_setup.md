# VNC + RViz Setup — Jetson → Mac

Visualize drone trajectory in RViz on a Mac by connecting to a VNC session
running on the Jetson.

---

## One-Time Setup (Jetson)

### 1. Install packages
```bash
sudo apt update && sudo apt install x11vnc xvfb -y
```

### 2. Set VNC password
```bash
x11vnc -storepasswd
```

### 3. Create startup script
Run each line separately:
```bash
echo '#!/bin/bash' > ~/start_vnc.sh
echo 'pkill Xvfb 2>/dev/null; pkill x11vnc 2>/dev/null; sleep 1' >> ~/start_vnc.sh
echo 'Xvfb :1 -screen 0 1280x800x24 &' >> ~/start_vnc.sh
echo 'sleep 2' >> ~/start_vnc.sh
echo 'x11vnc -display :1 -rfbauth ~/.vnc/passwd -forever -bg -quiet' >> ~/start_vnc.sh
echo "echo \"VNC started. Connect to: vnc://\$(hostname -I | awk '{print \$1}')\"" >> ~/start_vnc.sh
chmod +x ~/start_vnc.sh
```

---

## Every Flight

### Step 1 — Start VNC (Jetson SSH terminal)
```bash
~/start_vnc.sh
```
It prints the IP, e.g. `vnc://194.47.32.39`.
The DPMS warning is harmless.

### Step 2 — Connect from Mac
- Finder → **Go** → **Connect to Server...**
- Enter: `vnc://194.47.32.39`
- Enter VNC password

You should see `beetlesniffer-desktop:1 opened`.

### Step 3 — Open a terminal inside VNC (Jetson SSH terminal)
```bash
export DISPLAY=:1
xterm &
```
An xterm window will appear in the VNC session on your Mac.

### Step 4 — Launch RViz (inside the xterm in VNC)
```bash
source /opt/ros/humble/setup.bash
rviz2 -d ~/Sites/drone-brain/missions/visualizer.rviz
```

### Step 5 — Start the visualizer (Jetson SSH terminal)
```bash
export DISPLAY=:1
source /opt/ros/humble/setup.bash
python3 ~/Sites/drone-brain/missions/visualizer.py
```

### Step 6 — Run the mission (Jetson SSH terminal)
```bash
source /opt/ros/humble/setup.bash
python3 ~/Sites/drone-brain/missions/gps_goto_steering_v2.py
```

---

## What You See in RViz

| Display | Colour | Source |
|---------|--------|--------|
| EKF path | Green | `/mavros/local_position/pose` — IMU-smoothed |
| GPS path | Red | `/mavros/global_position/global` — raw, shows noise |
| Velocity arrow | Yellow | `/mavros/setpoint_velocity/cmd_vel` |
| Goal marker | Blue cylinder | `goal.json` |
| Status text | White | Mode / altitude / distance to goal |

---

## Troubleshooting

**VNC window is black with no xterm:**
Run `export DISPLAY=:1 && xterm &` from an SSH terminal.

**RViz shows "Fixed Frame [map] does not exist":**
Normal until MAVROS publishes the first pose — it clears once the drone has a GPS fix.

**Topics not visible (`ros2 topic list` is empty):**
Make sure `ROS_DOMAIN_ID` matches on all terminals:
```bash
export ROS_DOMAIN_ID=0
```

**VNC already running from a previous session:**
```bash
pkill Xvfb; pkill x11vnc
~/start_vnc.sh
```
