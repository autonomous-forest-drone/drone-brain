"""
Probes a serial port at each common PX4 baud rate and reports
which ones receive valid MAVLink heartbeats (RX) and whether
PX4 responds to commands (TX).

Usage (run on the Jetson):
  python3 tools/probe_baud.py             # tests /dev/ttyTHS1
  python3 tools/probe_baud.py /dev/ttyTHS0
"""

import sys
import time
from pymavlink import mavutil

PORT = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyTHS1'
TIMEOUT = 3.0  # seconds to wait per check

BAUD_RATES = [9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600]

print(f'Probing {PORT} — waiting {TIMEOUT:.0f}s per baud rate\n')
print(f'  {"baud":>7}   {"RX (heartbeat)":22} {"TX (command response)"}')
print(f'  {"-"*7}   {"-"*22} {"-"*21}')

for baud in BAUD_RATES:
    rx_ok = False
    tx_ok = False
    rx_detail = ''

    try:
        conn = mavutil.mavlink_connection(PORT, baud=baud, source_system=255)

        # RX check: wait for a heartbeat from PX4
        msg = conn.recv_match(type='HEARTBEAT', blocking=True, timeout=TIMEOUT)
        if msg:
            rx_ok = True
            rx_detail = f'autopilot={msg.autopilot}'
            conn.target_system = msg.get_srcSystem()
            conn.target_component = msg.get_srcComponent()

            # TX check: request autopilot capabilities and wait for response.
            # This is the same request MAVROS makes on startup — if PX4 receives
            # our transmission it will reply with AUTOPILOT_VERSION.
            conn.mav.command_long_send(
                conn.target_system,
                conn.target_component,
                mavutil.mavlink.MAV_CMD_REQUEST_AUTOPILOT_CAPABILITIES,
                0,   # confirmation
                1,   # param1: request capabilities
                0, 0, 0, 0, 0, 0,
            )
            response = conn.recv_match(
                type=['AUTOPILOT_VERSION', 'COMMAND_ACK'],
                blocking=True,
                timeout=TIMEOUT,
            )
            tx_ok = response is not None

        conn.close()
    except Exception as e:
        rx_detail = f'error: {e}'

    rx_str = f'OK  {rx_detail}' if rx_ok else 'no heartbeat'
    tx_str = 'OK' if tx_ok else ('FAIL — no response' if rx_ok else 'n/a')
    print(f'  {baud:>7}   {rx_str:22} {tx_str}')

print('\nRX=OK means PX4 is sending on this port at this baud rate.')
print('TX=OK means PX4 is receiving and responding — this baud rate is fully working.')
print('TX=FAIL means wiring or config issue: check TX/RX pin swap on the cable.')
