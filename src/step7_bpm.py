# step7_bpm.py
# This code calculates BPM from the green signal buffer.
# FFT finds the pulse frequency from the green signal.
# Normal human BPM is between 60 and 150.
# Press 'q' to quit.

import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os
from collections import deque
from scipy.signal import butter, filtfilt

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

# Buffer settings
BUFFER_SIZE = 150
FPS         = 30

# Buffer to store green values
green_buffer = deque(maxlen=BUFFER_SIZE)

# Filter function to remove noise
def bandpass_filter(signal, lowcut=0.8, highcut=3.0, fs=30, order=3):
    nyq  = fs / 2
    low  = lowcut  / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# BPM calculation function
def calculate_bpm(buffer, fps=30):
    signal = np.array(buffer)
    signal = signal - np.mean(signal)
    try:
        filtered = bandpass_filter(signal, fs=fps)
    except:
        return 0
    fft_result    = np.fft.rfft(filtered)
    fft_freqs     = np.fft.rfftfreq(len(filtered), d=1.0/fps)
    valid         = (fft_freqs >= 0.8) & (fft_freqs <= 3.0)
    if not np.any(valid):
        return 0
    fft_magnitude = np.abs(fft_result)
    peak_freq     = fft_freqs[valid][np.argmax(fft_magnitude[valid])]
    bpm           = peak_freq * 60
    return round(bpm)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

print("BPM Calculation started! Press 'q' to quit.")
print("Please sit still and wait for buffer to fill...")
print("---")

# Force create window
cv2.namedWindow("BPM Calculator", cv2.WINDOW_NORMAL)
cv2.resizeWindow("BPM Calculator", 640, 480)
cv2.moveWindow("BPM Calculator", 100, 100)

# Store current BPM
current_bpm = 0

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

            # Get left cheek points
            left_points = []
            for idx in LEFT_CHEEK:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                left_points.append([x, y])

            # Get right cheek points
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
            green_mean    = cv2.mean(green_channel, mask=mask)[0]

            # Add to buffer
            green_buffer.append(green_mean)
            collected = len(green_buffer)

            # Calculate BPM when buffer is full
            if collected == BUFFER_SIZE:
                current_bpm = calculate_bpm(green_buffer, fps=FPS)
                print(f"BPM calculated: {current_bpm}")

            # Draw cheek area on frame
            cv2.fillPoly(frame, [left_pts],  (0, 255, 0))
            cv2.fillPoly(frame, [right_pts], (0, 255, 0))

            # Show green value
            cv2.putText(frame, f"Green: {green_mean:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Show buffer progress
            cv2.putText(frame, f"Buffer: {collected}/{BUFFER_SIZE}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Show BPM
            if current_bpm > 0:
                cv2.putText(frame, f"BPM: {current_bpm}", (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            else:
                cv2.putText(frame, "Collecting data...", (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        else:
            cv2.putText(frame, "No Face Detected", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("BPM Calculator", frame)

        key = cv2.waitKey(30)
        if key & 0xFF == ord('q'):
            print("Closing.")
            break

cap.release()
cv2.destroyAllWindows()
print("Done!")