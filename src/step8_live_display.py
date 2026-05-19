# step8_live_display.py
# This is the FINAL complete program.
# It shows webcam + green cheeks + BPM + live pulse graph.
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

# Filter function
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
        return 0, []
    fft_result    = np.fft.rfft(filtered)
    fft_freqs     = np.fft.rfftfreq(len(filtered), d=1.0/fps)
    valid         = (fft_freqs >= 0.8) & (fft_freqs <= 3.0)
    if not np.any(valid):
        return 0, filtered.tolist()
    fft_magnitude = np.abs(fft_result)
    peak_freq     = fft_freqs[valid][np.argmax(fft_magnitude[valid])]
    bpm           = peak_freq * 60
    return round(bpm), filtered.tolist()

# This function draws the pulse graph on a black panel
def draw_graph(signal, width=640, height=150):
    # Create black panel for graph
    graph = np.zeros((height, width, 3), dtype=np.uint8)

    if len(signal) < 2:
        return graph

    # Normalize signal between 0 and height
    sig = np.array(signal)
    min_val = sig.min()
    max_val = sig.max()

    if max_val == min_val:
        return graph

    # Scale signal to fit graph height
    sig_scaled = (sig - min_val) / (max_val - min_val)
    sig_scaled = (sig_scaled * (height - 20) + 10).astype(int)

    # Calculate x positions for each point
    x_step = width / len(sig_scaled)

    # Draw green line connecting all points
    for i in range(1, len(sig_scaled)):
        x1 = int((i - 1) * x_step)
        x2 = int(i * x_step)
        y1 = height - sig_scaled[i - 1]
        y2 = height - sig_scaled[i]
        cv2.line(graph, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Add label
    cv2.putText(graph, "Pulse Signal", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return graph

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

print("Live Display started! Press 'q' to quit.")
print("Sit still and wait for BPM to appear...")
print("---")

# Force create window
cv2.namedWindow("Contactless Pulse Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Contactless Pulse Detection", 640, 650)
cv2.moveWindow("Contactless Pulse Detection", 100, 50)

# Store current BPM and filtered signal
current_bpm      = 0
filtered_signal  = []

with FaceLandmarker.create_from_options(options) as landmarker:

    while True:
        success, frame = cap.read()

        if not success:
            print("ERROR: Could not read frame.")
            break

        # Resize frame to fixed size
        frame = cv2.resize(frame, (640, 480))

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

            # Create mask
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [left_pts],  255)
            cv2.fillPoly(mask, [right_pts], 255)

            # Get green mean
            green_channel = frame[:, :, 1]
            green_mean    = cv2.mean(green_channel, mask=mask)[0]

            # Add to buffer
            green_buffer.append(green_mean)
            collected = len(green_buffer)

            # Calculate BPM when buffer is full
            if collected == BUFFER_SIZE:
                current_bpm, filtered_signal = calculate_bpm(
                    green_buffer, fps=FPS)

            # Draw cheek area
            cv2.fillPoly(frame, [left_pts],  (0, 255, 0))
            cv2.fillPoly(frame, [right_pts], (0, 255, 0))

            # Draw top info bar background
            cv2.rectangle(frame, (0, 0), (640, 50), (0, 0, 0), -1)

            # Show green value
            cv2.putText(frame, f"Green: {green_mean:.1f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Show buffer
            cv2.putText(frame, f"Buffer: {collected}/{BUFFER_SIZE}", (200, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Show BPM
            if current_bpm > 0:
                cv2.putText(frame, f"BPM: {current_bpm}", (400, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                cv2.putText(frame, "Collecting...", (400, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Show title
            cv2.putText(frame, "Contactless Pulse Detection", (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        else:
            cv2.putText(frame, "No Face Detected - Please face the camera",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

        # Draw pulse graph below the video
        graph = draw_graph(filtered_signal if filtered_signal
                           else list(green_buffer))

        # Combine video frame and graph vertically
        combined = np.vstack([frame, graph])

        cv2.imshow("Contactless Pulse Detection", combined)

        key = cv2.waitKey(30)
        if key & 0xFF == ord('q'):
            print("Closing.")
            break

cap.release()
cv2.destroyAllWindows()
print("Final BPM:", current_bpm)
print("Done!")