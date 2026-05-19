# app.py
# This is the web version of our pulse detection project.
# It runs in browser and works on mobile too!
# Uses phone camera when opened on mobile.

import streamlit as st
import cv2
import numpy as np
from collections import deque
from scipy.signal import butter, filtfilt
import mediapipe as mp
import urllib.request
import os

# Page settings
st.set_page_config(
    page_title="Pulse Detector",
    page_icon="❤️",
    layout="centered")

st.title("❤️ Contactless Pulse Detector")
st.write("Sit still and look at the camera. Your BPM will appear in 5 seconds!")

# Download model if needed
model_path = "face_landmarker.task"

if not os.path.exists(model_path):
    st.info("Downloading face model... please wait...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, model_path)

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

# Cheek landmarks
LEFT_CHEEK  = [116, 123, 147, 213, 192, 214, 212, 216, 206, 203]
RIGHT_CHEEK = [345, 352, 376, 433, 416, 434, 432, 436, 426, 423]

BUFFER_SIZE = 150
FPS         = 30

# Filter function
def bandpass_filter(signal, lowcut=0.8, highcut=3.0, fs=30, order=3):
    nyq  = fs / 2
    low  = lowcut  / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# BPM calculation
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

# Start button
start = st.button("▶ Start Pulse Detection")

if start:
    # Green buffer
    green_buffer = deque(maxlen=BUFFER_SIZE)
    current_bpm  = 0

    # Placeholders for live update
    frame_placeholder = st.empty()
    bpm_placeholder   = st.empty()
    info_placeholder  = st.empty()

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Cannot open camera. Please allow camera access.")
        st.stop()

    with FaceLandmarker.create_from_options(options) as landmarker:

        # Run for 300 frames then stop
        for i in range(300):
            success, frame = cap.read()

            if not success:
                st.error("Cannot read camera frame.")
                break

            frame     = cv2.resize(frame, (640, 480))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame)

            result = landmarker.detect(mp_image)

            h, w, _ = frame.shape

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]

                left_points = []
                for idx in LEFT_CHEEK:
                    x = int(landmarks[idx].x * w)
                    y = int(landmarks[idx].y * h)
                    left_points.append([x, y])

                right_points = []
                for idx in RIGHT_CHEEK:
                    x = int(landmarks[idx].x * w)
                    y = int(landmarks[idx].y * h)