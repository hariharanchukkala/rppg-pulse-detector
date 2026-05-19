# step6_buffer.py
# This code collects green values in a buffer.
# Buffer = a list that stores last 150 green values.
# 150 frames at 30fps = about 5 seconds of signal.
# Press 'q' to quit.

import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os
from collections import deque  # deque = a list with max size

# Download model if needed
model_path = "face_landmarker.task"

if not os.path.exists(model_path):
    print("Downloading model... please wait...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Model downloaded!")

# Set up face landmarker
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Cheek landmark numbers
LEFT_CHEEK  = [116, 123, 147, 213, 192, 214, 212, 216, 206, 203]
RIGHT_CHEEK = [345, 352, 376, 433, 416, 434, 432, 436, 426, 423]

# Buffer to store last 150 green values
# When buffer is full, oldest value is removed automatically
BUFFER_SIZE = 150
green_buffer = deque(maxlen=BUFFER_SIZE)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

print("Signal Buffering started! Press 'q' to quit.")
print(f"Collecting {BUFFER_SIZE} frames before BPM calculation...")
print("---")

# Force create window
cv2.namedWindow("Signal Buffer", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Signal Buffer", 640, 480)
cv2.moveWindow("Signal Buffer", 100, 100)

with FaceLandmarker.create_from_options(options) as landmarker:

    while True:
        success, frame = cap.read()

        if not success:
            print("ERROR: Could not read frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame)

        result = landmarker.detect(mp_image)

        h, w, _ = frame.shape

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            # Get cheek points
            left_points = []
            for idx in LEFT_CHEEK:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                left_points.append([x, y])

            right_points = []
            for idx in RIGHT_CHEEK:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                right_points.append([x, y])

            left_pts  = np.array(left_points,  dtype=np.int32)
            right_pts = np.array(right_points, dtype=np.int32)

            # Create mask for cheek area
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [left_pts],  255)
            cv2.fillPoly(mask, [right_pts], 255)

            # Get green channel mean
            green_channel = frame[:, :, 1]
            green_mean = cv2.mean(green_channel, mask=mask)[0]

            # Add green value to buffer
            green_buffer.append(green_mean)

            # How many values collected so far
            collected = len(green_buffer)

            # Print progress every 30 frames
            if collected % 30 == 0:
                print(f"Collected {collected}/{BUFFER_SIZE} values | Latest green: {green_mean:.2f}")

            # Draw cheek area
            cv2.fillPoly(frame, [left_pts],  (0, 255, 0))
            cv2.fillPoly(frame, [right_pts], (0, 255, 0))

            # Show buffer progress on screen
            cv2.putText(frame, f"Green: {green_mean:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.putText(frame, f"Buffer: {collected}/{BUFFER_SIZE}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Show ready message when buffer is full
            if collected == BUFFER_SIZE:
                cv2.putText(frame, "Buffer Full! Ready for BPM!", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        else:
            cv2.putText(frame, "No Face Detected", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Signal Buffer", frame)

        key = cv2.waitKey(30)
        if key & 0xFF == ord('q'):
            print("Closing.")
            break

cap.release()
cv2.destroyAllWindows()
print("Done!")