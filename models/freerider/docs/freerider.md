# Freerider — PPO Forest Navigation

`run_freerider.py` is the onboard flight script for the Freerider policy: a PPO actor trained with Depth Anything V2 depth perception to navigate through forest environments.

Supports two modes:

| Mode | Flag | Camera source |
|---|---|---|
| Real hardware | _(none)_ | Arducam IMX477 via GStreamer (Jetson) |
| AirSim simulation | `--sim` | AirSim ROS image topic (GCS machine) |

---

## 1. Export the model (run on the Jetson)

TensorRT engines are GPU-architecture-specific — they must be compiled on the machine that will run inference. The workflow is:

1. Copy the training checkpoint (`.zip`) from the training machine to the Jetson.
2. Run `export_freerider_trt.py` on the Jetson to produce the `.trt` engine.

### Prerequisites

```bash
pip install stable-baselines3 torch torchvision
# trtexec must be on PATH — it ships with TensorRT / JetPack
```

### Steps

**Copy the checkpoint to the Jetson:**

```bash
# On the training machine
scp best_avoidance_v5_*.zip user@jetson:~/freerider/checkpoints/
```

**Run the export script on the Jetson:**

```bash
cd ~/drone-brain/models/freerider/helpers

# Auto-discovers the latest best_avoidance_v5_*.zip in the default search path
python3 export_freerider_trt.py

# Or point to a specific checkpoint
python3 export_freerider_trt.py --checkpoint ~/freerider/checkpoints/best_avoidance_v5_1000000_steps.zip
```

The script:
1. Loads the SB3 PPO checkpoint and extracts the actor subgraph (`features_extractor` → `policy_net` → `action_net`).
2. Exports it to `~/freerider/models/freerider_actor.onnx` (two inputs: `image 1×3×144×256`, `state 1×1`).
3. Calls `trtexec --fp16` to compile the ONNX to `~/freerider/models/freerider_actor.trt`.

**Expected output:**

```
[export] Loading checkpoint: best_avoidance_v5_1000000_steps.zip
[export] Exporting ONNX → ~/freerider/models/freerider_actor.onnx
[export] Running trtexec (FP16) ...
[export] Done. Engine saved to ~/freerider/models/freerider_actor.trt
```

The `.trt` file is what `run_freerider.py` loads at runtime.

---

## 2. Real hardware flight

Everything runs on the **Jetson Orin Nano**.

### Prerequisites

- MAVROS running on the Jetson.
- `.trt` engine built (see section 1).
- Arducam IMX477 connected via CSI.
- Pixhawk connected via `/dev/ttyTHS1` at 115200 baud.

### Terminal 1 — MAVROS

```bash
ros2 launch mavros px4.launch fcu_url:=/dev/ttyTHS1:921600
```

### Terminal 2 — Freerider

```bash
cd ~/drone-brain/models/freerider

python3 run_freerider.py --engine ~/freerider/models/freerider_actor.trt
```

The script:
1. Loads the TRT engine and the Depth Anything V2 depth estimator.
2. Waits for MAVROS connection then arms and climbs to 1.5 m.
3. Runs a two-pass autofocus scan (coarse 0–1000, then fine around the peak) while hovering.
4. Streams velocity commands at ~5 Hz, blending raw policy output with the previous action (`momentum = 0.3`).
5. Monitors RC: switching from OFFBOARD to ALTCTL / POSCTL hands control back to the pilot immediately.
6. On landing, prints step-latency statistics.

#### Skipping autofocus with a known focus value

If you already know the best focus value for the current location (e.g. from a previous flight), pass it with `--focus` to skip the autofocus scan and start flying immediately:

```bash
python3 run_freerider.py --engine ~/freerider/models/freerider_actor.trt --focus 550
```

The focus value (0–1000) is set once on hover and re-applied every avoidance step to counteract vibration drift. To find a good value for a new location, run `capture_gst.py` on the ground first:

```bash
cd ~/drone-brain
sudo python3 tests/sensor_tests/capture_gst.py
# Best focus: 550  ← use this with --focus
```

**Triggering:** switch RC ch5 to OFFBOARD. Switch back at any time to reclaim control.

### Terminal 3 — (optional) telemetry monitor

```bash
python3 ~/drone-brain/tools/telemetry_monitor.py -gps
```

### CSV log

After each flight, a timestamped CSV is written to `~/freerider/logs/`:

```
t, raw_action, smoothed_action, lateral_vel_ms, step_latency_ms
```

A latency summary (average / min / max per step) is printed to the console on exit.

---

## 3. Simulation with AirSim

Everything runs on the **ground control station** (the machine where UE5 + AirSim are installed). The Jetson is not involved in simulation.

### AirSim settings

Copy the repo settings before opening AirSim:

```bash
cp ~/drone-brain/airsim/settings.json ~/Documents/AirSim/settings.json
```

This enables the front-center camera published on `/airsim_node/drone_1/front_center_custom/Scene`.

### Terminal 1 — UE5 / AirSim

Launch the AirSim Unreal project or prebuilt binary. AirSim listens on TCP 4560 for PX4 SITL. Leave it running in the background.

### Terminal 2 — PX4 SITL

From the root of the `PX4-Autopilot` repository:

```bash
make px4_sitl_default none_iris
```

Wait for:

```
[AirSim] Connected to AirSim!
INFO  [commander] Ready for takeoff!
```

### Terminal 3 — MAVROS

```bash
ros2 launch mavros px4.launch fcu_url:=udp://:14540@127.0.0.1:14580
```

| Port | Role |
|---|---|
| `14540` | MAVROS listens — receives telemetry from PX4 SITL |
| `14580` | PX4 SITL listens — receives commands from MAVROS |

### Terminal 4 — Freerider (sim mode)

```bash
cd ~/drone-brain/models/freerider

python3 run_freerider.py --sim
```

With `--sim` the script:
- Reads camera frames from the AirSim ROS image topic (via `cv_bridge`) instead of GStreamer.
- Switches to OFFBOARD automatically — no RC input required.
- Skips rclone Dropbox upload after landing.

To override the camera topic (e.g. a different AirSim vehicle or camera name):

```bash
python3 run_freerider.py --sim --sim-image-topic /airsim_node/drone_1/front_center_custom/Scene
```

### Expected console output

```
============================================================
Freerider — PPO forest navigation

Mode: AirSim simulation
Image topic: /airsim_node/drone_1/front_center_custom/Scene
Engine: ~/freerider/models/freerider_actor.trt
============================================================

[DepthEstimator] Loading depth-anything/Depth-Anything-V2-Small-hf on cuda ...
[DepthEstimator] Ready.
[TRTEngine] Loaded: ~/freerider/models/freerider_actor.trt
[freerider] Waiting for MAVROS connection...
[freerider] Connected. Arming...
[freerider] ARM: OK
[freerider] Climbing to 1.5 m...
[freerider] OFFBOARD — avoidance loop running.
...
[freerider] Landed. Steps: 312  Avg latency: 47.3 ms  Min: 38.1 ms  Max: 201.4 ms
```
