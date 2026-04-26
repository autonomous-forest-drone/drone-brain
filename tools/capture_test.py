#!/usr/bin/env python3

import cv2

pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1 ! "
    "nvvidconv ! video/x-raw,format=BGRx ! "
    "videoconvert ! video/x-raw,format=BGR ! appsink"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
ret, frame = cap.read()
cap.release()

if not ret or frame is None:
    print("ERROR: failed to capture frame")
else:
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite("capture_test.jpg", frame)
    print(f"Saved capture_test.jpg  ({frame.shape[1]}x{frame.shape[0]})")
