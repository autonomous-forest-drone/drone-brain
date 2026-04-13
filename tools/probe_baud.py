"""
Probes a serial port at each common PX4 baud rate and reports
which ones receive valid MAVLink heartbeats.

Usage (run on the Jetson):
  python3 tools/probe_baud.py             # tests /dev/ttyTHS1
  python3 tools/probe_baud.py /dev/ttyTHS0
"""

import sys
import time
from pymavlink import mavutil

PORT = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyTHS1'
TIMEOUT = 3.0  # seconds to wait for a heartbeat at each baud rate

BAUD_RATES = [9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600]

print(f'Probing {PORT} — waiting {TIMEOUT:.0f}s per baud rate\n')

for baud in BAUD_RATES:
    print(f'  {baud:>7} baud ... ', end='', flush=True)
    try:
        conn = mavutil.mavlink_connection(PORT, baud=baud)
        msg = conn.recv_match(type='HEARTBEAT', blocking=True, timeout=TIMEOUT)
        if msg:
            print(f'HEARTBEAT received  autopilot={msg.autopilot} type={msg.type}')
        else:
            print('no heartbeat')
        conn.close()
    except Exception as e:
        print(f'error: {e}')

print('\nDone. The baud rate that received a heartbeat is what PX4 TELEM2 is set to.')
print('Set SER_TEL2_BAUD in QGC to match, or update the MAVROS fcu_url to match.')
