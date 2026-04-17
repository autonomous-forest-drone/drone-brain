# GPS, EKF & Offboard Mission — Improvements Guide

## Module: Holybro M10 GPS Standard

**Chipset:** u-blox M10 (SAM-M10Q)
**Max update rate:** 18 Hz
**Constellations:** GPS L1 + GLONASS L1 + Galileo E1 + BeiDou B1 (all 4 simultaneously)
**Accuracy:** 1.5 m CEP (open sky) — ~2.5 m at 95%
**Default baud rate (Holybro firmware):** 38400
**Compass:** IST8310

The M10 is a significant improvement over M8N. Its main limitation for our use case is
still single-band (L1 only) — multipath near trees affects L1 more than dual-band modules.

---

## Background

During test flights we observed two GPS-related failure modes:

1. **Low update rate (~1.3 Hz)** — GPS fixes arrived only every ~750 ms instead of every 200 ms
   (5 Hz), causing the bearing-to-goal to stall and then jump 40–50° when the next fix arrived.
   One test ran correctly at 5 Hz; another dropped to 1.3 Hz — indicating an intermittent
   configuration or signal quality issue rather than a hardware fault.
2. **Position noise (~1–3 m)** — at close range (5 m from goal), a 1 m GPS error produces
   ~11° bearing error; multipath near trees can exceed 2–3 m, pushing errors past 30°.

---

## 1. GPS Update Rate

### Diagnosis
In the flight logs the line:
```
Collecting 20 GPS fixes... took 15.2 s  →  1.3 Hz
```
indicates the M10 is running at 1–2 Hz, far below its capable 5–18 Hz.

**Root cause:** The u-blox M10's dynamic model defaults to "portable" or "stationary" when
`GPS_UBX_DYNMODEL` is not set to airborne. These models throttle the output rate automatically.

### Fix — QGC / PX4 Parameters

| Parameter | Recommended | Notes |
|---|---|---|
| `GPS_UBX_DYNMODEL` | **7** (Airborne <2g) | **Most important.** Stationary/portable modes throttle to 1–2 Hz |
| `GPS_1_PROTOCOL` | **1** (u-blox) | Must be u-blox, not NMEA |
| `SER_GPS1_BAUD` | **115200** | Holybro M10 default is 38400 — too slow for 4-constellation data at 5+ Hz. PX4 reconfigures the module baud at startup if this is set higher |

> The baud rate mismatch is a likely cause of the 1.3 Hz issue. At 38400 baud with all 4
> constellations enabled, the serial buffer can overflow, causing PX4's UBX driver to drop
> messages and the effective rate to fall below 2 Hz.

> After changing these parameters and restarting, verify "Collecting 20 GPS fixes" completes
> in ≤ 4 s (5 Hz) or ≤ 2 s (10 Hz).

### Fix — u-blox module direct configuration (u-center)
If the PX4 parameter alone is insufficient (PX4 may not fully reconfigure M10 at startup):

**u-center 2** (for M10 generation):
- **UART1 baudrate** → 115200
- **GNSS config** → enable GPS + GLONASS + Galileo + BeiDou
- **Navigation rate** → 5 Hz or 10 Hz (`measRate = 200 ms` or `100 ms`)
- **Save to flash** via CFG-VALSET with layer = Flash

> Note: The M10 uses UBX **generation 9** configuration protocol (CFG-VALSET/CFG-VALGET)
> instead of the older UBX-CFG-* messages used on M8N. Use **u-center 2** (not classic u-center)
> for the M10.

---

## 2. GPS Accuracy & Multipath

### Constellation selection

| Parameter | Recommended | Notes |
|---|---|---|
| `GPS_1_GNSS` | **0** (auto) or **67** | M10 supports all 4 simultaneously; auto lets PX4/u-blox choose |
| `GPS_UBX_CFG_SBAS` | **1** (enabled) | EGNOS covers Sweden; improves to ~0.5 m from ~2 m |

`GPS_1_GNSS` bit mask:
- Bit 0 = GPS (1)
- Bit 1 = SBAS (2)
- Bit 2 = Galileo (4)
- Bit 5 = GLONASS (32)
- Bit 6 = BeiDou (64)
→ **67 = 1+2+4+64** (GPS + SBAS + Galileo + BeiDou)

### Antenna placement
- Keep antenna ≥ 15 cm above all carbon-fibre structures (carbon is conductive → multipath)
- Use a **ground plane** ≥ 10 cm diameter under the patch antenna
- Route GPS cable away from ESC power wires
- After physically moving the GPS module → recalibrate compass and update:
  - `EKF2_GPS_POS_X` / `_Y` / `_Z` (antenna offset from IMU, in metres NED body frame)

---

## 3. EKF2 Parameters for GPS Fusion

### GPS noise model
These tell the EKF how much to trust each GPS measurement:

| Parameter | Default | Forest/multipath recommendation | Effect |
|---|---|---|---|
| `EKF2_GPS_P_NOISE` | 0.5 m | **1.0–2.0 m** | Higher value → EKF trusts GPS less → smoother position estimate |
| `EKF2_GPS_V_NOISE` | 0.3 m/s | 0.5 m/s | Reduce velocity jump sensitivity |
| `EKF2_GPS_DELAY_MS` | 110 ms | measure actual | Mis-set delay degrades fusion quality |

> Setting `EKF2_GPS_P_NOISE = 1.5` in a tree-covered area effectively acts as a low-pass filter
> on the position estimate at the EKF level — no code changes required and the improvement
> applies to all subscribers of `/mavros/global_position/global`.

### GPS quality checks (arm-prevention)

| Parameter | Bits | Notes |
|---|---|---|
| `EKF2_GPS_CHECK` | default 245 | Bit 1 = horizontal speed; Bit 4 = HDOP; Bit 6 = horizontal position noise check |
| `EKF2_REQ_HDOP` | 2.5 | Lower → stricter; consider 3.0 near trees |
| `EKF2_REQ_NSATS` | 6 | Minimum satellites before EKF trusts GPS |
| `EKF2_REQ_SACC` | 0.5 m/s | Speed accuracy requirement |

### Height sensor fusion

| Parameter | Value | Notes |
|---|---|---|
| `EKF2_HGT_REF` | **1** (barometric) | Use barometer as primary height reference, not GPS |
| `EKF2_BARO_NOISE` | 3.5 Pa | Increase if baro is noisy from prop wash |

> GPS vertical accuracy is typically 2–5× worse than horizontal. Always use barometer as the
> height reference for low-altitude flight. GPS altitude should only supplement, not replace it.

### Magnetic declination
| Parameter | Value | Notes |
|---|---|---|
| `EKF2_DECL_TYPE` | **2** (automatic) | Uses GPS position to look up magnetic declination |

---

## 4. Hardware Upgrade Path

| Tier | Module | Accuracy | Notes |
|---|---|---|---|
| **Current** | Holybro M10 (u-blox M10) | 1.5 m CEP | Single-band L1, 4-constellation, 18 Hz max |
| Next step | Holybro H-RTK F9P | 1–2 cm CEP (RTK) / 1 m (standalone) | Dual-band L1+L2, requires base station for RTK |
| Alternative | u-blox F9P (e.g. SparkFun GPS-RTK2) | 1–2 cm CEP (RTK) | Same F9P chipset, cheaper base option |

**Current status:** The M10 is a capable module. With `GPS_UBX_DYNMODEL=7`, correct baud rate,
and EGNOS enabled, it should deliver 1–1.5 m CEP consistently. This translates to ~15° bearing
error at 5 m range — still significant. The 5 m arrival radius (`GOAL_RADIUS = 1.5 m`) is
already set conservatively to account for this.

For sub-metre accuracy at 5 m range (required for PPO model observation quality matching
simulation), RTK GPS is the correct long-term hardware upgrade. The Holybro H-RTK F9P
is a direct drop-in replacement for the M10 mounting footprint.

---

## 5. Code Architecture Improvements

### 5.1 Replace spin-loop with ROS2 timer-based design
**Current:** Mission logic runs in a blocking while-loop calling `rclpy.spin_once()`.
**Industry standard:** Each concern is a separate timer callback at a defined rate.

```python
# Current pattern (fragile — spin_once timing not guaranteed)
while condition:
    self._publish_vel(...)
    rclpy.spin_once(self, timeout_sec=dt)

# Better pattern (timer-driven)
self.timer = self.create_timer(1.0 / SETPOINT_HZ, self._control_callback)
```

Benefits: predictable timing, cleaner state machine, proper callback scheduling.

### 5.2 Explicit OffboardControlMode
When using `setpoint_velocity/cmd_vel`, MAVROS implicitly sends an OFFBOARD keep-alive.
The px4-offboard reference (ETH Zurich) explicitly publishes `OffboardControlMode` at the
same rate as setpoints, declaring exactly which control channels are active.

```python
# Declare velocity control mode explicitly (px4_msgs approach)
msg = OffboardControlMode()
msg.position    = False
msg.velocity    = True
msg.acceleration = False
msg.attitude    = False
```

Our scripts achieve this implicitly through MAVROS — acceptable but less transparent.

### 5.3 rosbag2 recording of every flight
Add to the launch/startup script:
```bash
ros2 bag record -o flight_$(date +%Y%m%d_%H%M%S) \
  /mavros/local_position/pose \
  /mavros/global_position/global \
  /mavros/setpoint_velocity/cmd_vel \
  /mavros/state \
  /mavros/statustext
```
This captures every flight for post-flight analysis in RViz or Python.

### 5.4 QoS profile alignment
The px4-offboard reference uses specific QoS profiles. For MAVROS topics:

```python
# For state/status topics (reliable, transient-local)
state_qos = QoSProfile(depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL)

# For sensor topics (best-effort, volatile)
sensor_qos = QoSProfile(depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE)
```
Our scripts already do this correctly. ✓

### 5.5 Use setpoint_raw for mixed position+velocity control
`setpoint_raw/local` allows commanding position Z (altitude hold by PX4's own position
controller) while commanding velocity X/Y (horizontal navigation). This avoids the need for
a custom altitude PI controller entirely:

```python
from mavros_msgs.msg import PositionTarget

msg = PositionTarget()
msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
# Use velocity for XY, position for Z:
msg.type_mask = (PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY |
                 PositionTarget.IGNORE_VZ | PositionTarget.IGNORE_AFX |
                 PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
                 PositionTarget.IGNORE_YAW)
msg.velocity.x = vn_ned   # North in NED
msg.velocity.y = ve_ned   # East in NED
msg.position.z = -target_altitude_enu  # NED Z is negative-up
msg.yaw_rate   = -yaw_rate_enu  # NED yaw is negative of ENU
```

This is the recommended approach for altitude-holding velocity missions.

### 5.6 GPS-based bearing: use EKF local position instead of raw GPS
`/mavros/local_position/pose` (EKF fused, IMU-smoothed) gives a better position estimate
between GPS updates than `/mavros/global_position/global` (GPS rate-limited).

Convert goal once to local frame at startup, then use local position for bearing:
```python
# At startup: convert goal GPS to local ENU offset from home
home_lat, home_lon = get_home_from_mavros()
goal_de = (goal_lon - home_lon) * EARTH_R * math.cos(math.radians(home_lat))
goal_dn = (goal_lat - home_lat) * EARTH_R
# goal in ENU local frame: (goal_de, goal_dn)

# In flight: use EKF pose for bearing (smoother than raw GPS)
dx = goal_de - self.pose.pose.position.x  # East
dy = goal_dn - self.pose.pose.position.y  # North
bearing = math.atan2(dy, dx)
```

### 5.7 Setpoint publishing rate
PX4 requires setpoints at ≥ 2 Hz to maintain OFFBOARD mode (times out at 500 ms).
The official PX4 C++ reference (`px4_ros_com`) publishes at **10 Hz** as a safe minimum.
Our scripts publish at **20 Hz** which is well above the threshold.

**Important:** both the `OffboardControlMode` keep-alive AND the setpoint message must be
published together on every cycle. Our scripts implicitly satisfy this via MAVROS's internal
keep-alive, but explicit pairing is safer.

### 5.8 MAVROS vs px4_msgs — architecture tradeoff

| | MAVROS (current) | px4_msgs / XRCE-DDS |
|---|---|---|
| Transport | MAVLink serial/UDP | DDS via micro-XRCE-DDS agent |
| Frame | Converts to ENU automatically | Raw NED — manual conversion required |
| Latency | Higher (MAVLink serialization) | Lower (direct uORB bridge) |
| Arming | Service calls | Topic-based `VehicleCommand` |
| Complexity | Lower — works out of the box | Higher — requires px4_msgs package |

**For the current phase** (GPS navigation + PPO integration), MAVROS is the right choice:
automatic frame conversion, well-documented, and battle-tested on Jetson hardware.

If latency becomes a bottleneck for high-rate PPO control (>20 Hz actions), migrating the
setpoint publisher to `px4_msgs` + XRCE-DDS is the industry upgrade path, while keeping
MAVROS for state subscriptions and MAVROS services for arming/mode switching.

### 5.9 Pre-arm warmup (already correct)
The official PX4 example streams setpoints for 10 cycles before issuing the arm command.
Our `_switch_offboard()` pre-streams for **2 seconds at 20 Hz = 40 cycles** — well above
the requirement. ✓

---

## 6. Parameters Checklist — Before Every Test

```
[ ] GPS_UBX_DYNMODEL    = 7      (airborne)
[ ] GPS_1_GNSS          = 67     (GPS+SBAS+Galileo+BeiDou)
[ ] GPS_UBX_CFG_SBAS    = 1      (EGNOS enabled)
[ ] SER_GPS1_BAUD       = 57600+ (sufficient for multi-constellation 5 Hz)
[ ] EKF2_GPS_P_NOISE    = 1.0    (higher if near trees)
[ ] EKF2_GPS_DELAY_MS   = 110    (verify with actual module)
[ ] EKF2_HGT_REF        = 1      (barometer as height reference)
[ ] EKF2_DECL_TYPE      = 2      (auto magnetic declination)
[ ] EKF2_GPS_POS_X/Y/Z  = offset from IMU (update after physical GPS move)
[ ] MIS_TAKEOFF_ALT     = desired takeoff height in metres
[ ] MC_YAWRATE_MAX      >= 30    (allow our 0.5 rad/s ≈ 28°/s commands)
[ ] MPC_XY_VEL_MAX      >= 2.0   (allow 1 m/s cruise + margin)
[ ] COM_OF_LOSS_T       = 1.0    (OFFBOARD timeout — switch to POSCTL on link loss)
[ ] COM_RC_OVERRIDE      = 1      (allow RC takeover in OFFBOARD)
```