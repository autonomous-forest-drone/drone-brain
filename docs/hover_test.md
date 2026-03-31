# hover_test.py

Autonomous hover mission using ROS 2 and MAVROS. The drone arms, climbs to
**1.5 m**, hovers for **3 seconds**, then lands automatically via `AUTO.LAND`.

Supports two modes:

| Mode | Flag | Trigger |
|---|---|---|
| Real hardware | _(none)_ | RC ch5 switched to OFFBOARD position |
| AirSim simulation | `--airsim` | Script switches to OFFBOARD automatically |

---

## How it works

1. Starts streaming position setpoints at 20 Hz immediately on launch.
   PX4 requires an active setpoint stream before it will accept an OFFBOARD
   mode switch — this keeps the stream alive the whole time.
2. Waits for OFFBOARD mode (via RC in real life, or auto-triggered in sim).
3. Arms the drone.
4. Sends a setpoint at 1.5 m for 4 seconds (ascent window).
5. Holds at 1.5 m for 3 seconds (hover).
6. Switches to `AUTO.LAND` — PX4 descends and disarms on touchdown.

If OFFBOARD mode is lost at any point during the mission, the script hands
control back immediately and does not resume the mission on its own.

---

## Real hardware

Everything runs on the **Jetson Orin Nano**. Use three separate terminals.

### What needs to be running

**Terminal 1 — MAVROS**

Bridges ROS 2 to the Pixhawk over the serial connection:

```bash
ros2 launch mavros px4.launch fcu_url:=/dev/ttyACM0
```

**Terminal 2 — hover_test.py**

```bash
python3 missions/hover_test.py
```

The script prints `Connected. Streaming setpoints...` when MAVROS is up and
the FCU is responding.

**Terminal 3 — (optional) telemetry monitor**

```bash
python3 helpers/telemetry_monitor.py -gps
```

### Triggering the mission

Flip **RC ch5** to the **OFFBOARD** position. The mission starts immediately.

Flip ch5 back at any time to hand control to the RC.

After landing, flip ch5 out of OFFBOARD and back again to run a second mission.

---

## AirSim simulation

Everything runs on the **ground control station** (the machine where UE5 and
AirSim are installed). Use three separate terminals.

### AirSim settings

AirSim must be configured for PX4 before launching UE5. Two settings files are
kept in `~/Documents/AirSim/`:

| File | Used for |
|---|---|
| `settings.simplflight.json` | PPO training pipeline |
| `settings.px4.json` | PX4 SITL / hover_test |

Swap to the PX4 settings before opening AirSim:

```bash
cp ~/Documents/AirSim/settings.px4.json ~/Documents/AirSim/settings.json
```

Restore for training afterwards:

```bash
cp ~/Documents/AirSim/settings.simplflight.json ~/Documents/AirSim/settings.json
```

### What needs to be running

**UE5 / AirSim**

Launch the AirSim Unreal project (or prebuilt binary). AirSim will listen on
TCP port **4560** for PX4 SITL. Leave it running in the background.

**Terminal 1 — PX4 SITL**

From the root of the PX4-Autopilot repository:

```bash
make px4_sitl_default none_iris
```

This compiles PX4 (first run only) and starts the SITL process. PX4 will
connect to AirSim over TCP 4560. Wait for the console to show:

```
[AirSim] Connected to AirSim!
INFO  [commander] Ready for takeoff!
```

**Terminal 2 — MAVROS**

Connects to PX4 SITL over UDP instead of serial:

```bash
ros2 launch mavros px4.launch fcu_url:=udp://:14540@127.0.0.1:14580
```

| Port | Role |
|---|---|
| `14540` | MAVROS listens — receives telemetry from PX4 SITL |
| `14580` | PX4 SITL listens — receives commands from MAVROS |

**Terminal 3 — hover_test.py**

```bash
python3 missions/hover_test.py --airsim
```

The `--airsim` flag makes the script:
1. Stream setpoints for 2 seconds after MAVROS connects (so PX4 is ready to
   accept OFFBOARD).
2. Call `SetMode('OFFBOARD')` automatically — no RC input needed.
3. Run the full mission and exit cleanly when done.

### Expected console output

```
============================================================
Hover mission

Mode: AirSim simulation

Required (all on the GCS machine):
  1. AirSim/UE5 running with PX4 settings.json
  2. PX4 SITL:  make px4_sitl_default none_iris
  3. MAVROS:    ros2 launch mavros px4.launch \
                  fcu_url:=udp://:14540@127.0.0.1:14580
============================================================

[hover_test] Waiting for MAVROS connection...
[hover_test] AirSim mode — streaming setpoints for 2 s, then switching to OFFBOARD...
[hover_test] SET_MODE OFFBOARD: OK
[hover_test] OFFBOARD active — arming...
[hover_test] ARM: OK
[hover_test] Climbing to 1.5 m...
[hover_test] Hovering for 3.0 s...
[hover_test] Hover complete. Landing...
[hover_test] Landed and disarmed.
[hover_test] Simulation mission complete.
```
