#!/usr/bin/env python3

import os
import subprocess
import sys
import time

import numpy as np
import cv2

import pycuda.autoinit  # noqa: F401  — must come before TRT engine load

from jetson_camera import DEFAULT_FOCUS, GstJpegCapture, set_focus
from midas_trt import TRTMidas

DEBUG_DIR = '/home/beetlesniffer/drone-brain/models/fortune_cookie/images'
MIDAS_TRT = '/home/beetlesniffer/drone-brain/models/fortune_cookie/model/midas_small.trt'

CAM_W, CAM_H, CAM_FPS = 1920, 1080, 30   # full res for the viewer
VIEWER_FOCUS = 550   # tuned for the viewer's typical scene; override per-run if needed


def open_camera(focus=VIEWER_FOCUS):
    cap = GstJpegCapture(CAM_W, CAM_H, CAM_FPS)
    set_focus(focus)
    time.sleep(0.3)   # VCM lens travel
    print(f'Camera streaming {CAM_W}x{CAM_H}@{CAM_FPS}  |  focus locked at {focus}')
    return cap


def sync_dropbox():
    subprocess.Popen(
        ['rclone', 'copy', DEBUG_DIR, 'dropbox:fortune_cookie/images'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def main():
    print()
    print('Select mode:')
    print('  1) Raw full resolution')
    print('  2) Raw compressed (192x192)')
    print('  3) MiDaS depth')
    print()
    choice = input('Enter 1, 2 or 3: ').strip()
    if choice not in ('1', '2', '3'):
        print('Invalid choice.')
        return

    os.makedirs(DEBUG_DIR, exist_ok=True)

    midas = None
    if choice == '3':
        if not os.path.exists(MIDAS_TRT):
            print(f'ERROR: MiDaS TRT engine not found at {MIDAS_TRT}')
            print('       Build it with: python3 models/fortune_cookie/helpers/export_midas_trt.py')
            return
        print('Loading MiDaS TRT engine...')
        midas = TRTMidas(MIDAS_TRT)
        print('MiDaS ready.')

    cap = open_camera()
    if cap is None:
        return

    mode_names = {'1': 'raw full res', '2': 'raw compressed', '3': 'MiDaS depth'}
    print(f'Saving {mode_names[choice]} images to {DEBUG_DIR} every second. Ctrl+C to stop.')
    last_save = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Failed to read frame')
            time.sleep(0.1)
            continue

        now = time.monotonic()
        if now - last_save < 1.0:
            continue
        last_save = now

        fname = os.path.join(DEBUG_DIR, f'{time.strftime("%H%M%S")}.jpg')

        if choice == '1':
            cv2.imwrite(fname, frame)

        elif choice == '2':
            rgb192 = cv2.resize(frame, (192, 192), interpolation=cv2.INTER_AREA)
            cv2.imwrite(fname, rgb192)

        elif choice == '3':
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            depth = midas.infer(rgb)[0]   # (H, W) float32 in [0, 1]
            depth_u8 = (depth * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
            cv2.imwrite(fname, depth_color)

        print(f'Saved {fname}')
        sync_dropbox()


if __name__ == '__main__':
    main()
