#!/usr/bin/env python3

import os
import subprocess
import sys
import time

import numpy as np
import cv2
import torch
import torch.nn.functional as F

DEBUG_DIR  = '/home/beetlesniffer/PythonProjects/DANI/debug_frames'
MIDAS_PATH = '/home/beetlesniffer/.cache/torch/hub/intel-isl_MiDaS_master'


def load_midas():
    if MIDAS_PATH not in sys.path:
        sys.path.insert(0, MIDAS_PATH)

    from midas.midas_net_custom import MidasNet_small
    from midas.transforms import Resize, NormalizeImage, PrepareForNet

    model = MidasNet_small(
        None, features=64, backbone='efficientnet_lite3',
        exportable=True, non_negative=True, blocks={'expand': True}
    )
    weights_path = '/home/beetlesniffer/.cache/torch/hub/checkpoints/midas_v21_small_256.pt'
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict)
    model.eval().cuda()

    def transform(img):
        sample = {"image": img / 255.0}
        sample = Resize(256, 256, resize_target=None, keep_aspect_ratio=True,
                        ensure_multiple_of=32, resize_method="upper_bound",
                        image_interpolation_method=cv2.INTER_CUBIC)(sample)
        sample = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(sample)
        sample = PrepareForNet()(sample)
        return torch.from_numpy(sample["image"]).unsqueeze(0)

    return model, transform


def open_camera():
    gst_pipeline = (
        'nvarguscamerasrc ! '
        'video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1 ! '
        'nvvidconv ! video/x-raw, format=BGRx ! '
        'videoconvert ! video/x-raw, format=BGR ! appsink'
    )
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print('GStreamer failed, falling back to v4l2...')
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('ERROR: cannot open camera')
        return None
    return cap


def sync_dropbox():
    subprocess.Popen(
        ['rclone', 'copy', DEBUG_DIR, 'dropbox:DANI/debug_frames'],
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

    midas, transform = None, None
    if choice == '3':
        torch.zeros(1).cuda()
        torch.backends.cudnn.enabled = False
        print('Loading MiDaS...')
        midas, transform = load_midas()
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
            rgb = frame[:, :, ::-1].copy()
            rgb = cv2.resize(rgb, (192, 192), interpolation=cv2.INTER_AREA)
            inp = transform(rgb).cuda()
            with torch.no_grad():
                pred = midas(inp)
            if pred.dim() == 3:
                pred = pred.unsqueeze(1)
            pred  = F.interpolate(pred, (192, 192), mode='bilinear', align_corners=False).squeeze(1)
            mn, mx = pred.min(), pred.max()
            depth = ((pred - mn) / (mx - mn + 1e-8)).cpu().numpy().astype(np.float32)
            cv2.imwrite(fname, (depth[0] * 255).astype(np.uint8))

        print(f'Saved {fname}')
        sync_dropbox()


if __name__ == '__main__':
    main()
