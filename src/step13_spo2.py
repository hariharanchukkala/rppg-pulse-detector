# step13_spo2.py
# SpO2 (Blood Oxygen) Estimation using webcam
# Uses ratio of Red channel to Green channel
# Normal SpO2 = 95-100%
# WARNING: This is an ESTIMATE only - not medical grade!
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

# Buffer settings
BUFFER_SIZE = 512
FPS         = 30

# Buffers
r_buffer = deque(maxlen=BUFFER_SIZE)
g_buffer = deque(maxlen=BUFFER_SIZE)
b_buffer = deque(maxlen=BUFFER_SIZE)

# History for smoothing
bpm_history  = []
spo2_history = []

# Bandpass filter
def bandpass_filter(signal, lowcut, highcut, fs=30, order=4):
    nyq  = fs / 2
    low  = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# POS algorithm for BPM
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

    S1    = rn - gn
    S2    = rn + gn - 2 * bn
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

# SpO2 estimation
def estimate_spo2(r_buf, g_buf, fps=30):
    r = np.array(r_buf, dtype=float)
    g = np.array(g_buf, dtype=float)

    # Remove mean
    r = r - np.mean(r)
    g = g - np.mean(g)

    # Filter both channels to pulse frequency
    try:
        r_filtered = bandpass_filter(r, 0.7, 3.5, fps)
        g_filtered = bandpass_filter(g, 0.7, 3.5, fps)
    except:
        return 0

    # Calculate AC component (pulse variation)
    r_ac = np.std(r_filtered)
    g_ac = np.std(g_filtered)

    # Calculate DC component (average value)
    r_dc = np.mean(np.abs(r + np.mean(np.array(r_buf))))
    g_dc = np.mean(np.abs(g + np.mean(np.array(g_buf))))

    if g_ac == 0 or r_dc == 0 or g_dc == 0:
        return 0

    # Perfusion index ratio (R value)
    # R = (AC_red / DC_red) / (AC_green / DC_green)
    r_perfusion = r_ac / r_dc
    g_perfusion = g_ac / g_dc

    if g_perfusion == 0:
        return 0

    R = r_perfusion / g_perfusion

    # Convert R to SpO2 using empirical formula
    # SpO2 = 110 - 25 * R (simplified Beer-Lambert law)
    # This is calibrated for webcam use
    spo2 = 110 - 25 * R

    # Clamp to realistic range
    spo2 = max(85, min(100, spo2))

    return round(spo2, 1)

# SpO2 status
def spo2_status(spo2):
    if spo2 == 0:
        return "Measuring...", (200,200,200)
    elif spo2 >= 97:
        return "Excellent",    (0,255,100)
    elif spo2 >= 95:
        return "Normal",       (0,220, 80)
    elif spo2 >= 92:
        return "Low-Normal",   (0,165,255)
    elif spo2 >= 90:
        return "Low - Rest!",  (0, 80,255)
    else:
        return "Critical!",    (0,  0,255)

# Get RGB mean
def get_rgb_mean(frame, mask):
    b_val = cv2.mean(frame[:,:,0], mask=mask)[0]
    g_val = cv2.mean(frame[:,:,1], mask=mask)[0]
    r_val = cv2.mean(frame[:,:,2], mask=mask)[0]
    return r_val, g_val, b_val

# Get polygon points
def get_points(landmarks, indices, w, h):
    pts = []
    for idx in indices:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        pts.append([x, y])
    return np.array(pts, dtype=np.int32)

# Draw graph
def draw_graph(signal, width=640, height=100,
               color=(0,255,128), label="Signal"):
    graph = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(graph, label, (10,16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (150,150,150), 1)
    if len(signal) < 2:
        return graph
    sig    = np.array(signal[-300:])
    mn, mx = sig.min(), sig.max()
    if mx == mn:
        return graph
    scaled = ((sig-mn)/(mx-mn)*(height-24)+12).astype(int)
    xstep  = width / len(scaled)
    for i in range(1, len(scaled)):
        x1 = int((i-1)*xstep)
        x2 = int(i*xstep)
        y1 = height - scaled[i-1]
        y2 = height - scaled[i]
        cv2.line(graph, (x1,y1), (x2,y2), color, 2)
    return graph

# Draw SpO2 gauge
def draw_spo2_gauge(frame, spo2, x=350, y=120):
    if spo2 == 0:
        return
    # Background circle
    cv2.circle(frame, (x, y), 60, (40,40,40), -1)
    cv2.circle(frame, (x, y), 60, (80,80,80),  2)

    # Color based on value
    if spo2 >= 97:
        color = (0,255,100)
    elif spo2 >= 95:
        color = (0,220, 80)
    elif spo2 >= 92:
        color = (0,165,255)
    else:
        color = (0,  0,255)

    # SpO2 value
    cv2.putText(frame, f"{spo2}%",
                (x-35, y+8),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, color, 2)
    cv2.putText(frame, "SpO2",
                (x-22, y+28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (150,150,150), 1)

# Heart rate zone
def get_bpm_zone(bpm):
    if bpm < 60:
        return "Resting",  (0,200,200)
    elif bpm <= 100:
        return "Normal",   (0,255,  0)
    elif bpm <= 130:
        return "Elevated", (0,165,255)
    else:
        return "High",     (0,  0,255)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

print("SpO2 Estimation started!")
print("Heart Rate: ready in 8 seconds")
print("SpO2: ready in 17 seconds")
print("Sit still and breathe normally...")
print("Press 'q' to quit")
print("---")

cv2.namedWindow("SpO2 + Heart Monitor", cv2.WINDOW_NORMAL)
cv2.resizeWindow("SpO2 + Heart Monitor", 640, 700)
cv2.moveWindow("SpO2 + Heart Monitor", 100, 20)

current_bpm  = 0
current_spo2 = 0
r_signal     = []
g_signal     = []

BPM_MIN  = 256
SPO2_MIN = 512

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

            # Calculate BPM
            if collected >= BPM_MIN:
                raw_bpm = pos_bpm(
                    list(r_buffer)[-BPM_MIN:],
                    list(g_buffer)[-BPM_MIN:],
                    list(b_buffer)[-BPM_MIN:],
                    fps=FPS)
                if 45 <= raw_bpm <= 150:
                    bpm_history.append(raw_bpm)
                if len(bpm_history) > 7:
                    bpm_history.pop(0)
                if len(bpm_history) > 0:
                    current_bpm = round(
                        sum(bpm_history)/len(bpm_history))

            # Calculate SpO2
            if collected >= SPO2_MIN:
                raw_spo2 = estimate_spo2(
                    list(r_buffer)[-SPO2_MIN:],
                    list(g_buffer)[-SPO2_MIN:],
                    fps=FPS)
                if raw_spo2 > 0:
                    spo2_history.append(raw_spo2)
                if len(spo2_history) > 10:
                    spo2_history.pop(0)
                if len(spo2_history) > 0:
                    current_spo2 = round(
                        sum(spo2_history)/len(spo2_history), 1)
                    status, scolor = spo2_status(current_spo2)
                    print(f"BPM: {current_bpm} | "
                          f"SpO2: {current_spo2}% "
                          f"({status})")

            # Prepare signals for graphs
            g_list = list(g_buffer)
            r_list = list(r_buffer)
            if len(g_list) > 10:
                g_arr = np.array(g_list, dtype=float)
                r_arr = np.array(r_list, dtype=float)
                g_arr = g_arr - np.mean(g_arr)
                r_arr = r_arr - np.mean(r_arr)
                try:
                    g_signal = bandpass_filter(
                        g_arr, 0.7, 3.5, FPS).tolist()
                    r_signal = bandpass_filter(
                        r_arr, 0.7, 3.5, FPS).tolist()
                except:
                    g_signal = g_list
                    r_signal = r_list

            # Draw ROI overlay
            overlay = frame.copy()
            cv2.fillPoly(overlay, [left_pts],     (  0,255,  0))
            cv2.fillPoly(overlay, [right_pts],    (  0,255,  0))
            cv2.fillPoly(overlay, [forehead_pts], (  0,200,255))
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

            # Top bar
            cv2.rectangle(frame, (0,0), (640,55), (15,15,15), -1)
            cv2.putText(frame,
                f"R:{r:.0f} G:{g:.0f} B:{b:.0f}",
                (10,18), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0,255,255), 1)

            bpm_prog  = min(100, int(collected/BPM_MIN *100))
            spo2_prog = min(100, int(collected/SPO2_MIN*100))
            cv2.putText(frame,
                f"BPM:{bpm_prog}%  SpO2:{spo2_prog}%",
                (10,38), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (255,255,0), 1)

            # BPM display
            if current_bpm > 0:
                zone, zcolor = get_bpm_zone(current_bpm)
                cv2.putText(frame,
                    f"Heart: {current_bpm} BPM",
                    (10,100), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (0,100,255), 3)
                cv2.putText(frame,
                    zone, (10,128),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, zcolor, 2)
            else:
                cv2.putText(frame,
                    f"Heart Rate: measuring... {bpm_prog}%",
                    (10,100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (200,200,200), 1)

            # SpO2 display
            status, scolor = spo2_status(current_spo2)
            if current_spo2 > 0:
                draw_spo2_gauge(frame, current_spo2, x=530, y=160)
                cv2.putText(frame,
                    f"Oxygen: {status}",
                    (10,175), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, scolor, 2)

                # Warning if low
                if current_spo2 < 95:
                    cv2.putText(frame,
                        "LOW OXYGEN - Please rest!",
                        (10,210), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,0,255), 2)
                else:
                    cv2.putText(frame,
                        "Oxygen level is healthy",
                        (10,210), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,200,100), 1)
            else:
                cv2.putText(frame,
                    f"SpO2: measuring... {spo2_prog}%",
                    (10,175), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (200,200,200), 1)
                cv2.rectangle(frame,
                    (10,190), (620,205), (40,40,40), -1)
                fill = int(6.1 * spo2_prog)
                cv2.rectangle(frame,
                    (10,190), (10+fill,205),
                    (0,200,100), -1)

            # Disclaimer
            cv2.putText(frame,
                "SpO2 is estimated - not medical grade",
                (10,468), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (100,100,100), 1)

        else:
            cv2.rectangle(frame,
                (0,0), (640,480), (20,20,20), -1)
            cv2.putText(frame,
                "No Face — Please face camera",
                (130,240), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0,0,200), 2)

        # Draw two channel graphs
        g_graph = draw_graph(
            g_signal if g_signal else list(g_buffer),
            height=110, color=(0,200,100),
            label="Green channel (pulse)")

        r_graph = draw_graph(
            r_signal if r_signal else list(r_buffer),
            height=110, color=(0,80,255),
            label="Red channel (oxygen)")

        combined = np.vstack([frame, g_graph, r_graph])
        cv2.imshow("SpO2 + Heart Monitor", combined)

        key = cv2.waitKey(30)
        if key & 0xFF == ord('q'):
            print("Closing.")
            break

cap.release()
cv2.destroyAllWindows()
print(f"Final BPM:  {current_bpm}")
print(f"Final SpO2: {current_spo2}%")
print("Done!")