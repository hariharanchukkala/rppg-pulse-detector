# step2_face_detection.py
# This code opens webcam and draws a box around your face.
# Works with new MediaPipe version.
# Press 'q' to quit.

import cv2
import mediapipe as mp

# Load MediaPipe face detector (new way for version 0.10+)
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Download the face detection model first
import urllib.request
import os

model_path = "face_detection_short_range.tflite"

# Download model if not already downloaded
if not os.path.exists(model_path):
    print("Downloading face detection model... please wait...")
    url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    urllib.request.urlretrieve(url, model_path)
    print("Model downloaded!")

# Set up face detector options
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    min_detection_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

print("Face Detection started! Press 'q' to quit.")

# Start the face detector
with FaceDetector.create_from_options(options) as detector:

    while True:
        success, frame = cap.read()

        if not success:
            print("ERROR: Could not read frame.")
            break

        # Convert frame from BGR to RGB (MediaPipe needs RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wrap frame in MediaPipe image format
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame)

        # Detect faces
        result = detector.detect(mp_image)

        # Get frame size
        h, w, _ = frame.shape

        # If faces found, draw box
        if result.detections:
            for face in result.detections:
                box = face.bounding_box
                x = box.origin_x
                y = box.origin_y
                bw = box.width
                bh = box.height

                # Draw green box around face
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

                # Show confidence score
                score = int(face.categories[0].score * 100)
                cv2.putText(frame, f"Face: {score}%", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Face Detected", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show frame
        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Closing.")
            break

cap.release()
cv2.destroyAllWindows()
print("Done!")