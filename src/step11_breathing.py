# step11_breathing.py
# Breathing Rate Detection using rPPG signal
# Normal breathing rate = 12-20 breaths per minute
# We find the slow wave component from the green signal
# Press 'q' to quit

import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os
from collections import deque
from scipy.signal import butter, filtfilt

os.environ["GLOG_minloglevel"]     = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Download model if needed
model_path = "face_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading model... please wait...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Model downloaded!")

# MediaPipe setup
BaseOptions           = mp.tasks.BaseOptions
FaceLandmarker        = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# ROI landmarks
LEFT_CHEEK  = [116, 123, 147, 213, 192, 214, 212, 216, 206, 203]
RIGHT_CHEEK = [345, 352, 376, 433, 416, 434, 432, 436, 426, 423]
FOREHEAD    = [10, 67, 69, 104, 108, 151, 299, 337, 338]

# Buffer - need more data for breathing (30 seconds)
# 30 sec x 30 fps = 900 frames
BUFFER_SIZE = 900
FPS         = 30

# Buffers
r_buffer = deque(maxlen=BUFFER_SIZE)
g_buffer = deque(maxlen=BUFFER_SIZE)
b_buffer = deque(maxlen=BUFFER_SIZE)

# History for smoothing
bpm_history  = []
br_history   = []

# Bandpass filter function
def bandpass_filter(signal, lowcut, highcut, fs=30, order=4):
    nyq  = fs / 2
    low  = lowcut  / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# POS algorithm for heart rate
def pos_bpm(r_buf, g_buf, b_buf, fps=30):
    r = np.array(r_buf, dtype=float)
    g = np.array(g_buf, dtype=float)
    b = np.array(b_buf, dtype=float)

    r_mean = np.mean(r)
    g_mean = np.mean(g)
    b_mean = np.mean(b)

    if r_mean == 0 or g_mean == 0 or b_mean == 0:
        return 0

    rn = r / r_mean
    gn = g / g_mean
    bn = b / b_mean

    S1 = rn - gn
    S2 = rn + gn - 2 * bn

    std_s2 = np.std(S2)
    if std_s2 == 0:
        return 0

    alpha      = np.std(S1) / std_s2
    pos_signal = S1 + alpha * S2
    pos_signal = pos_signal - np.mean(pos_signal)

    try:
        filtered = bandpass_filter(pos_signal, 0.7, 3.5, fps)
    except:
        return 0

    fft_result    = np.fft.rfft(filtered)
    fft_freqs     = np.fft.rfftfreq(len(filtered), d=1.0/fps)
    valid         = (fft_freqs >= 0.7) & (fft_freqs <= 3.5)

    if not np.any(valid):
        return 0

    fft_magnitude = np.abs(fft_result)
    peak_freq     = fft_freqs[valid][np.argmax(fft_magnitude[valid])]
    return round(peak_freq * 60)

# Breathing rate from slow wave
def breathing_rate(g_buf, fps=30):
    g = np.array(g_buf, dtype=float)
    g = g - np.mean(g)

    # Breathing frequency range
    # 12 breaths/min = 0.2 Hz
    # 20 breaths/min = 0.33 Hz
    # We use wider range 0.1 - 0.5 Hz to be safe
    try:
        filtered = bandpass_filter(g, 0.1, 0.5, fps)
    except:
        return 0

    # FFT to find breathing frequency
    fft_result    = np.fft.rfft(filtered)
    fft_freqs     = np.fft.rfftfreq(len(filtered), d=1.0/fps)
    valid         = (fft_freqs >= 0.1) & (fft_freqs <= 0.5)

    if not np.any(valid):
        return 0

    fft_magnitude = np.abs(fft_result)
    peak_freq     = fft_freqs[valid][np.argmax(fft_magnitude[valid])]

    # Convert to breaths per minute
    br = peak_freq * 60
    return round(br)

# Get RGB mean from masked region
def get_rgb_mean(frame, mask):
    b_val = cv2.mean(frame[:,:,0], mask=mask)[0]
    g_val = cv2.mean(frame[:,:,1], mask=mask)[0]
    r_val = cv2.mean(frame[:,:,2], mask=mask)[0]
    return r_val, g_val, b_val

# Get polygon points from landmarks
def get_points(landmarks, indices, w, h):
    pts = []
    for idx in indices:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        pts.append([x, y])
    return np.array(pts, dtype=np.int32)

# Draw signal graph
def draw_graph(signal, width=640, height=100,
               color=(0,255,128), label="Signal"):
    graph = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(graph, label, (10,16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (150,150,150), 1)
    if len(signal) < 2:
        return graph
    sig    = np.array(signal[-300:])  # show last 300 points
    mn, mx = sig.min(), sig.max()
    if mx == mn:
        return graph
    scaled = ((sig-mn)/(mx-mn)*(height-24)+12).astype(int)
    xstep  = width / len(scaled)
    for i in range(1, len(scaled)):
        x1 = int((i-1) * xstep)
        x2 = int(i     * xstep)
        y1 = height - scaled[i-1]
        y2 = height - scaled[i]
        cv2.line(graph, (x1,y1), (x2,y2), color, 2)
    return graph

# Breathing status message
def breathing_status(br):
    if br == 0:
        return "Measuring...", (200,200,200)
    elif br < 10:
        return "Very Slow",    (255,100,  0)
    elif br < 12:
        return "Slow",         (  0,200,200)
    elif br <= 20:
        return "Normal",       (  0,255,  0)
    elif br <= 25:
        return "Fast",         (  0,165,255)
    else:
        return "Very Fast",    (  0,  0,255)

# Heart rate zone
def get_bpm_zone(bpm):
    if bpm < 60:
        return "Resting",  (  0,200,200)
    elif bpm <= 100:
        return "Normal",   (  0,255,  0)
    elif bpm <= 130:
        return "Elevated", (  0,165,255)
    else:
        return "High",     (  0,  0,255)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

print("Breathing Rate Detection started!")
print("Heart Rate: ready in 8 seconds")
print("Breathing Rate: ready in 30 seconds")
print("Sit still and breathe normally...")
print("Press 'q' to quit")
print("---")

cv2.namedWindow("Health Monitor - BPM + Breathing", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Health Monitor - BPM + Breathing", 640, 680)
cv2.moveWindow("Health Monitor - BPM + Breathing", 100, 20)

current_bpm = 0
current_br  = 0
br_signal   = []
pulse_signal = []

# Minimum frames needed
BPM_MIN_FRAMES = 256
BR_MIN_FRAMES  = 900

with FaceLandmarker.create_from_options(options) as landmarker:

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame     = cv2.resize(frame, (640, 480))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame)

        result  = landmarker.detect(mp_image)
        h, w, _ = frame.shape

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            left_pts     = get_points(landmarks, LEFT_CHEEK,  w, h)
            right_pts    = get_points(landmarks, RIGHT_CHEEK, w, h)
            forehead_pts = get_points(landmarks, FOREHEAD,    w, h)

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [left_pts],     255)
            cv2.fillPoly(mask, [right_pts],    255)
            cv2.fillPoly(mask, [forehead_pts], 255)

            r, g, b = get_rgb_mean(frame, mask)

            r_buffer.append(r)
            g_buffer.append(g)
            b_buffer.append(b)
            collected = len(g_buffer)

            # Calculate heart rate (needs 256 frames)
            if collected >= BPM_MIN_FRAMES:
                raw_bpm = pos_bpm(
                    list(r_buffer)[-BPM_MIN_FRAMES:],
                    list(g_buffer)[-BPM_MIN_FRAMES:],
                    list(b_buffer)[-BPM_MIN_FRAMES:],
                    fps=FPS)

                if 45 <= raw_bpm <= 150:
                    bpm_history.append(raw_bpm)
                if len(bpm_history) > 7:
                    bpm_history.pop(0)
                if len(bpm_history) > 0:
                    current_bpm = round(
                        sum(bpm_history) / len(bpm_history))

            # Calculate breathing rate (needs 900 frames)
            if collected >= BR_MIN_FRAMES:
                raw_br = breathing_rate(g_buffer, fps=FPS)

                if 8 <= raw_br <= 30:
                    br_history.append(raw_br)
                if len(br_history) > 5:
                    br_history.pop(0)
                if len(br_history) > 0:
                    current_br = round(
                        sum(br_history) / len(br_history))

                print(f"BPM: {current_bpm} | "
                      f"Breathing: {current_br} breaths/min")

            # Prepare graph signals
            g_list = list(g_buffer)
            if len(g_list) > 10:
                g_arr = np.array(g_list, dtype=float)
                g_arr = g_arr - np.mean(g_arr)
                try:
                    pulse_signal = bandpass_filter(
                        g_arr, 0.7, 3.5, FPS).tolist()
                    br_signal = bandpass_filter(
                        g_arr, 0.1, 0.5, FPS).tolist()
                except:
                    pulse_signal = g_list
                    br_signal    = g_list

            # Draw ROI overlay
            overlay = frame.copy()
            cv2.fillPoly(overlay, [left_pts],     (  0,255,  0))
            cv2.fillPoly(overlay, [right_pts],    (  0,255,  0))
            cv2.fillPoly(overlay, [forehead_pts], (  0,200,255))
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

            # Top info bar
            cv2.rectangle(frame, (0,0), (640,55), (15,15,15), -1)
            cv2.putText(frame,
                f"R:{r:.0f} G:{g:.0f} B:{b:.0f}",
                (10,18), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0,255,255), 1)

            # Progress info
            bpm_prog = min(100, int(collected/BPM_MIN_FRAMES*100))
            br_prog  = min(100, int(collected/BR_MIN_FRAMES *100))
            cv2.putText(frame,
                f"BPM ready:{bpm_prog}%  BR ready:{br_prog}%",
                (10,38), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (255,255,0), 1)

            # BPM display
            if current_bpm > 0:
                zone, zcolor = get_bpm_zone(current_bpm)
                cv2.putText(frame,
                    f"Heart: {current_bpm} BPM",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0,80,255), 3)
                cv2.putText(frame,
                    f"Heart: {current_bpm} BPM",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0,140,255), 1)
                cv2.putText(frame,
                    zone, (10,125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, zcolor, 2)
            else:
                cv2.putText(frame,
                    f"Heart Rate: measuring... {bpm_prog}%",
                    (10,100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (200,200,200), 1)

            # Breathing display
            br_status, br_color = breathing_status(current_br)
            if current_br > 0:
                cv2.putText(frame,
                    f"Breathing: {current_br} breaths/min",
                    (10, 165), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0,200,100), 3)
                cv2.putText(frame,
                    f"Breathing: {current_br} breaths/min",
                    (10, 165), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0,255,128), 1)
                cv2.putText(frame,
                    br_status, (10,190),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, br_color, 2)
            else:
                cv2.putText(frame,
                    f"Breathing: measuring... {br_prog}%",
                    (10,165), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (200,200,200), 1)
                cv2.putText(frame,
                    "Breathe normally and stay still",
                    (10,190), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (150,150,150), 1)

            # Normal ranges reference
            cv2.putText(frame,
                "Normal: Heart 60-100 BPM | Breathing 12-20/min",
                (10,472), cv2.FONT_HERSHEY_SIMPLEX,
                0.43, (120,120,120), 1)

        else:
            cv2.rectangle(frame, (0,0), (640,480), (20,20,20), -1)
            cv2.putText(frame,
                "No Face — Please face camera",
                (130,240), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0,0,200), 2)

        # Draw two graphs
        pulse_graph = draw_graph(
            pulse_signal, height=100,
            color=(0,140,255),
            label="Pulse signal (0.7-3.5 Hz)")

        br_graph = draw_graph(
            br_signal, height=100,
            color=(0,255,128),
            label="Breathing signal (0.1-0.5 Hz)")

        # Stack everything
        combined = np.vstack([frame, pulse_graph, br_graph])
        cv2.imshow("Health Monitor - BPM + Breathing", combined)

        key = cv2.waitKey(30)
        if key & 0xFF == ord('q'):
            print("Closing.")
            break

cap.release()
cv2.destroyAllWindows()
print(f"Final Heart Rate:    {current_bpm} BPM")
print(f"Final Breathing Rate: {current_br} breaths/min")
print("Done!")