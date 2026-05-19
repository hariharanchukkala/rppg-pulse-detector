# step4_roi.py
# This code selects the cheek region (ROI) from your face.
# ROI = Region of Interest = the area we check for pulse.
# Press 'q' to quit.

import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os

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

# These are the landmark numbers for left and right cheek
# These numbers come from the MediaPipe Face Mesh map
LEFT_CHEEK  = [116, 123, 147, 213, 192, 214, 212, 216, 206, 203]
RIGHT_CHEEK = [345, 352, 376, 433, 416, 434, 432, 436, 426, 423]

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

print("ROI Selection started! Press 'q' to quit.")

# Force create window
cv2.namedWindow("ROI Selection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ROI Selection", 640, 480)
cv2.moveWindow("ROI Selection", 100, 100)

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

            # Get left cheek points in pixels
            left_points = []
            for idx in LEFT_CHEEK:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                left_points.append([x, y])

            # Get right cheek points in pixels
            right_points = []
            for idx in RIGHT_CHEEK:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                right_points.append([x, y])

            # Convert to numpy array
            left_pts  = np.array(left_points, dtype=np.int32)
            right_pts = np.array(right_points, dtype=np.int32)

            # Draw filled green polygon on left cheek
            cv2.fillPoly(frame, [left_pts], (0, 255, 0))

            # Draw filled green polygon on right cheek
            cv2.fillPoly(frame, [right_pts], (0, 255, 0))

            cv2.putText(frame, "ROI Selected - Cheeks", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        else:
            cv2.putText(frame, "No Face Detected", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("ROI Selection", frame)

        key = cv2.waitKey(30)
        if key & 0xFF == ord('q'):
            print("Closing.")
            break

cap.release()
cv2.destroyAllWindows()
print("Done!")