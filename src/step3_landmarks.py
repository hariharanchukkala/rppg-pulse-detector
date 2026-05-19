# step3_landmarks.py
# Draws landmark dots on your face using new MediaPipe version.
# Press 'q' to quit.

import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os

# Download the face landmark model if not already downloaded
model_path = "face_landmarker.task"

if not os.path.exists(model_path):
    print("Downloading face landmark model... please wait...")
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

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

print("Face Landmarks started! Press 'q' to quit.")

# Force create window before the loop
cv2.namedWindow("Face Landmarks", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face Landmarks", 640, 480)
cv2.moveWindow("Face Landmarks", 100, 100)

with FaceLandmarker.create_from_options(options) as landmarker:

    while True:
        success, frame = cap.read()

        if not success:
            print("ERROR: Could not read frame.")
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wrap in MediaPipe image format
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame)

        # Detect landmarks
        result = landmarker.detect(mp_image)

        # Get frame size
        h, w, _ = frame.shape

        # If face found
        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            # Draw every landmark dot on the face
            for landmark in landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            cv2.putText(frame, "Face Found! Landmarks detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        else:
            cv2.putText(frame, "No Face Detected", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show frame
        cv2.imshow("Face Landmarks", frame)

        # This forces the window to update and show
        key = cv2.waitKey(30)
        if key & 0xFF == ord('q'):
            print("Closing.")
            break

cap.release()
cv2.destroyAllWindows()
print("Done!")