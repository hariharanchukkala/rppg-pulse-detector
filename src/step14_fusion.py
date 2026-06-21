# step14_fusion.py
# FINAL Multi-Parameter Health Fusion Dashboard
# Combines: BPM + Breathing + HRV + Stress + SpO2
# Shows everything in one beautiful dashboard
# Press 'q' to quit

import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os
from collections import deque
from scipy.signal import butter, filtfilt, find_peaks

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

# Buffer sizes
BUFFER_SIZE = 1800  # 60 seconds
FPS         = 30

# RGB buffers
r_buffer = deque(maxlen=BUFFER_SIZE)
g_buffer = deque(maxlen=BUFFER_SIZE)
b_buffer = deque(maxlen=BUFFER_SIZE)

# History for smoothing
bpm_history    = []
br_history     = []
hrv_history    = []
spo2_history   = []

# ── Signal Processing Functions ───────────────────────────

def bandpass_filter(signal, lowcut, highcut, fs=30, order=4):
    nyq  = fs / 2
    low  = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# POS Algorithm for BPM
def pos_bpm(r_buf, g_buf, b_buf, fps=30):
    r = np.array(r_buf, dtype=float)
    g = np.array(g_buf, dtype=float)
    b = np.array(b_buf, dtype=float)

    r_mean = np.mean(r)
    g_mean = np.mean(g)
    b_mean = np.mean(b)

    if r_mean == 0 or g_mean == 0 or b_mean == 0:
        return 0, []

    rn = r / r_mean
    gn = g / g_mean
    bn = b / b_mean

    S1 = rn - gn
    S2 = rn + gn - 2 * bn

    std_s2 = np.std(S2)
    if std_s2 == 0:
        return 0, []

    alpha      = np.std(S1) / std_s2
    pos_signal = S1 + alpha * S2
    pos_signal = pos_signal - np.mean(pos_signal)

    try:
        filtered = bandpass_filter(pos_signal, 0.7, 3.5, fps)
    except:
        return 0, []

    fft_result    = np.fft.rfft(filtered)
    fft_freqs     = np.fft.rfftfreq(len(filtered), d=1.0/fps)
    valid         = (fft_freqs >= 0.7) & (fft_freqs <= 3.5)

    if not np.any(valid):
        return 0, filtered.tolist()

    fft_magnitude = np.abs(fft_result)
    peak_freq     = fft_freqs[valid][np.argmax(fft_magnitude[valid])]
    return round(peak_freq * 60), filtered.tolist()

# Breathing Rate
def breathing_rate(g_buf, fps=30):
    g = np.array(g_buf, dtype=float)
    g = g - np.mean(g)
    try:
        filtered = bandpass_filter(g, 0.1, 0.5, fps)
    except:
        return 0
    fft_result    = np.fft.rfft(filtered)
    fft_freqs     = np.fft.rfftfreq(len(filtered), d=1.0/fps)
    valid         = (fft_freqs >= 0.1) & (fft_freqs <= 0.5)
    if not np.any(valid):
        return 0
    fft_magnitude = np.abs(fft_result)
    peak_freq     = fft_freqs[valid][np.argmax(fft_magnitude[valid])]
    return round(peak_freq * 60)

# HRV Calculation
def calculate_hrv(g_buf, fps=30):
    g = np.array(g_buf, dtype=float)
    g = g - np.mean(g)
    try:
        filtered = bandpass_filter(g, 0.7, 3.5, fps)
    except:
        return 0, 0
    min_distance = int(fps * 0.4)
    peaks, _     = find_peaks(filtered, distance=min_distance)
    if len(peaks) < 4:
        return 0, 0
    rr_intervals    = np.diff(peaks) / fps * 1000
    if len(rr_intervals) < 3:
        return 0, 0
    successive_diff = np.diff(rr_intervals)
    rmssd           = np.sqrt(np.mean(successive_diff ** 2))
    sdnn            = np.std(rr_intervals)
    return round(rmssd, 1), round(sdnn, 1)

# SpO2 Estimation
def estimate_spo2(r_buf, g_buf, fps=30):
    r = np.array(r_buf, dtype=float)
    g = np.array(g_buf, dtype=float)
    r = r - np.mean(r)
    g = g - np.mean(g)
    try:
        r_filtered = bandpass_filter(r, 0.7, 3.5, fps)
        g_filtered = bandpass_filter(g, 0.7, 3.5, fps)
    except:
        return 0
    r_ac = np.std(r_filtered)
    g_ac = np.std(g_filtered)
    r_dc = np.mean(np.abs(r + np.mean(np.array(r_buf))))
    g_dc = np.mean(np.abs(g + np.mean(np.array(g_buf))))
    if g_ac == 0 or r_dc == 0 or g_dc == 0:
        return 0
    R        = (r_ac / r_dc) / (g_ac / g_dc)
    spo2     = 115 - 20 * R
    spo2     = max(85, min(100, spo2))
    return round(spo2, 1)

# Signal Quality
def signal_quality(filtered_signal):
    if len(filtered_signal) < 10:
        return 0
    sig     = np.array(filtered_signal)
    fft_mag = np.abs(np.fft.rfft(sig))
    if np.mean(fft_mag) == 0:
        return 0
    snr = np.max(fft_mag) / np.mean(fft_mag)
    return min(100, int(snr * 5))

# Overall Health Score
def health_score(bpm, br, rmssd, spo2):
    score = 0
    total = 0

    # BPM score (60-100 = perfect)
    if bpm > 0:
        if 60 <= bpm <= 100:
            score += 25
        elif 50 <= bpm <= 110:
            score += 15
        else:
            score += 5
        total += 25

    # Breathing score (12-20 = perfect)
    if br > 0:
        if 12 <= br <= 20:
            score += 25
        elif 10 <= br <= 24:
            score += 15
        else:
            score += 5
        total += 25

    # HRV score (higher = better)
    if rmssd > 0:
        if rmssd > 50:
            score += 25
        elif rmssd > 30:
            score += 18
        elif rmssd > 20:
            score += 10
        else:
            score += 5
        total += 25

    # SpO2 score
    if spo2 > 0:
        if spo2 >= 97:
            score += 25
        elif spo2 >= 95:
            score += 20
        elif spo2 >= 92:
            score += 10
        else:
            score += 5
        total += 25

    if total == 0:
        return 0
    return round((score / total) * 100)

# ── Helper Functions ──────────────────────────────────────

def get_rgb_mean(frame, mask):
    b_val = cv2.mean(frame[:,:,0], mask=mask)[0]
    g_val = cv2.mean(frame[:,:,1], mask=mask)[0]
    r_val = cv2.mean(frame[:,:,2], mask=mask)[0]
    return r_val, g_val, b_val

def get_points(landmarks, indices, w, h):
    pts = []
    for idx in indices:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        pts.append([x, y])
    return np.array(pts, dtype=np.int32)

def draw_graph(signal, width=860, height=80,
               color=(0,255,128), label=""):
    graph = np.zeros((height, width, 3), dtype=np.uint8)
    if label:
        cv2.putText(graph, label, (8,14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (120,120,120), 1)
    if len(signal) < 2:
        return graph
    sig    = np.array(signal[-400:])
    mn, mx = sig.min(), sig.max()
    if mx == mn:
        return graph
    scaled = ((sig-mn)/(mx-mn)*(height-20)+10).astype(int)
    xstep  = width / len(scaled)
    for i in range(1, len(scaled)):
        x1 = int((i-1)*xstep)
        x2 = int(i*xstep)
        y1 = height - scaled[i-1]
        y2 = height - scaled[i]
        cv2.line(graph, (x1,y1), (x2,y2), color, 2)
    return graph

# ── Dashboard Drawing Functions ───────────────────────────

def draw_panel(dashboard, title, value, unit,
               status, color, x, y, w=200, h=130):
    # Panel background
    cv2.rectangle(dashboard, (x,y), (x+w,y+h),
                  (25,25,25), -1)
    cv2.rectangle(dashboard, (x,y), (x+w,y+h),
                  (60,60,60), 1)
    # Colored top bar
    cv2.rectangle(dashboard, (x,y), (x+w,y+4),
                  color, -1)
    # Title
    cv2.putText(dashboard, title,
                (x+8, y+22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (160,160,160), 1)
    # Value
    if value > 0:
        cv2.putText(dashboard, str(value),
                    (x+8, y+72),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.6, color, 3)
        # Unit
        cv2.putText(dashboard, unit,
                    (x+8, y+95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (120,120,120), 1)
        # Status
        cv2.putText(dashboard, status,
                    (x+8, y+118),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1)
    else:
        cv2.putText(dashboard, "...",
                    (x+8, y+72),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.6, (60,60,60), 3)
        cv2.putText(dashboard, "measuring",
                    (x+8, y+95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (80,80,80), 1)

def draw_health_score(dashboard, score, x, y):
    # Background circle
    cv2.circle(dashboard, (x,y), 55, (30,30,30), -1)
    cv2.circle(dashboard, (x,y), 55, (60,60,60),  2)

    if score >= 80:
        color = (0,255,100)
        label = "Excellent"
    elif score >= 60:
        color = (0,220, 80)
        label = "Good"
    elif score >= 40:
        color = (0,165,255)
        label = "Fair"
    else:
        color = (0, 80,255)
        label = "Poor"

    if score > 0:
        cv2.putText(dashboard, str(score),
                    (x-22, y+8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, color, 2)
        cv2.putText(dashboard, label,
                    (x-28, y+26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1)
    else:
        cv2.putText(dashboard, "?",
                    (x-10, y+10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (60,60,60), 2)

    cv2.putText(dashboard, "Health",
                (x-22, y-38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (120,120,120), 1)
    cv2.putText(dashboard, "Score",
                (x-18, y-24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (120,120,120), 1)

def get_bpm_status(bpm):
    if bpm <= 0:
        return "---", (100,100,100)
    elif bpm < 50:
        return "Too Low",   (255,100,  0)
    elif bpm < 60:
        return "Resting",   (  0,200,200)
    elif bpm <= 100:
        return "Normal",    (  0,255,  0)
    elif bpm <= 120:
        return "Elevated",  (  0,165,255)
    else:
        return "High",      (  0,  0,255)

def get_br_status(br):
    if br <= 0:
        return "---", (100,100,100)
    elif br < 12:
        return "Slow",   (  0,200,200)
    elif br <= 20:
        return "Normal", (  0,255,  0)
    else:
        return "Fast",   (  0,165,255)

def get_stress_status(rmssd):
    if rmssd <= 0:
        return "---", (100,100,100)
    elif rmssd > 50:
        return "Very Relaxed", (  0,255,100)
    elif rmssd > 35:
        return "Relaxed",      (  0,220, 80)
    elif rmssd > 20:
        return "Moderate",     (  0,165,255)
    else:
        return "Stressed",     (  0,  0,255)

def get_spo2_status(spo2):
    if spo2 <= 0:
        return "---", (100,100,100)
    elif spo2 >= 97:
        return "Excellent", (  0,255,100)
    elif spo2 >= 95:
        return "Normal",    (  0,220, 80)
    elif spo2 >= 92:
        return "Low-Normal",(  0,165,255)
    else:
        return "Low",       (  0,  0,255)

# ── Open Webcam ───────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

print("Multi-Parameter Fusion Dashboard started!")
print("Timeline:")
print("  8 seconds  → BPM appears")
print("  30 seconds → Breathing + HRV + Stress appear")
print("  60 seconds → All parameters stable")
print("Sit still and breathe normally...")
print("Press 'q' to quit")
print("---")

# Window
cv2.namedWindow("Health Fusion Dashboard", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Health Fusion Dashboard", 860, 900)
cv2.moveWindow("Health Fusion Dashboard", 50, 20)

# Current values
current_bpm    = 0
current_br     = 0
current_rmssd  = 0
current_sdnn   = 0
current_spo2   = 0
current_stress = 0
quality        = 0
pos_signal_disp = []
br_signal_disp  = []

# Minimum frames
BPM_MIN = 256
BR_MIN  = 900
HRV_MIN = 900
SPO2_MIN = 512

frame_count = 0

with FaceLandmarker.create_from_options(options) as landmarker:

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame      = cv2.resize(frame, (860, 480))
        rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image   = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame)

        result  = landmarker.detect(mp_image)
        h, w, _ = frame.shape
        frame_count += 1

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

            # BPM every 30 frames
            if collected >= BPM_MIN and frame_count % 30 == 0:
                raw_bpm, pos_signal_disp = pos_bpm(
                    list(r_buffer)[-BPM_MIN:],
                    list(g_buffer)[-BPM_MIN:],
                    list(b_buffer)[-BPM_MIN:],
                    fps=FPS)
                quality = signal_quality(pos_signal_disp)
                if 45 <= raw_bpm <= 150:
                    bpm_history.append(raw_bpm)
                if len(bpm_history) > 7:
                    bpm_history.pop(0)
                if len(bpm_history) > 0:
                    current_bpm = round(
                        sum(bpm_history)/len(bpm_history))

            # Breathing every 60 frames
            if collected >= BR_MIN and frame_count % 60 == 0:
                raw_br = breathing_rate(
                    list(g_buffer)[-BR_MIN:], fps=FPS)
                if 8 <= raw_br <= 30:
                    br_history.append(raw_br)
                if len(br_history) > 5:
                    br_history.pop(0)
                if len(br_history) > 0:
                    current_br = round(
                        sum(br_history)/len(br_history))

                # Breathing signal for graph
                g_arr = np.array(
                    list(g_buffer)[-BR_MIN:], dtype=float)
                g_arr = g_arr - np.mean(g_arr)
                try:
                    br_signal_disp = bandpass_filter(
                        g_arr, 0.1, 0.5, FPS).tolist()
                except:
                    br_signal_disp = []

            # HRV every 90 frames
            if collected >= HRV_MIN and frame_count % 90 == 0:
                rmssd, sdnn = calculate_hrv(
                    list(g_buffer)[-HRV_MIN:], fps=FPS)
                if rmssd > 0:
                    hrv_history.append(rmssd)
                if len(hrv_history) > 5:
                    hrv_history.pop(0)
                if len(hrv_history) > 0:
                    current_rmssd = round(
                        sum(hrv_history)/len(hrv_history), 1)
                    current_sdnn  = sdnn

            # SpO2 every 60 frames
            if collected >= SPO2_MIN and frame_count % 60 == 0:
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

            # Draw ROI on frame
            overlay = frame.copy()
            cv2.fillPoly(overlay, [left_pts],     (  0,255,  0))
            cv2.fillPoly(overlay, [right_pts],    (  0,255,  0))
            cv2.fillPoly(overlay, [forehead_pts], (  0,200,255))
            frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

            # Info bar on frame
            cv2.rectangle(frame, (0,0), (860,38), (15,15,15), -1)
            cv2.putText(frame,
                f"R:{r:.0f} G:{g:.0f} B:{b:.0f}  "
                f"Quality:{quality}%  "
                f"Buffer:{collected}/{BUFFER_SIZE}",
                (10,22), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0,255,255), 1)

        else:
            cv2.rectangle(frame, (0,0), (860,480), (20,20,20), -1)
            cv2.putText(frame,
                "No Face Detected — Please face camera",
                (200,240), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0,0,200), 2)

        # ── Build Dashboard ───────────────────────────────
        dashboard = np.zeros((900, 860, 3), dtype=np.uint8)

        # Paste webcam frame on top
        dashboard[0:480, 0:860] = frame

        # Dashboard background panel
        cv2.rectangle(dashboard,
            (0,482), (860,900), (18,18,18), -1)

        # Title bar
        cv2.rectangle(dashboard,
            (0,482), (860,505), (30,30,30), -1)
        cv2.putText(dashboard,
            "Multi-Parameter Health Fusion Dashboard",
            (10,498), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (200,200,200), 1)
        cv2.putText(dashboard,
            "rPPG Technology",
            (700,498), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (80,80,80), 1)

        # ── 4 Parameter Panels ────────────────────────────
        collected = len(g_buffer)

        # BPM Panel
        bpm_status, bpm_color = get_bpm_status(current_bpm)
        draw_panel(dashboard,
            "HEART RATE",
            current_bpm, "BPM",
            bpm_status, bpm_color,
            x=10, y=510)

        # Breathing Panel
        br_status, br_color = get_br_status(current_br)
        draw_panel(dashboard,
            "BREATHING",
            current_br, "breaths/min",
            br_status, br_color,
            x=220, y=510)

        # Stress Panel (show RMSSD as HRV)
        stress_status, stress_color = get_stress_status(
            current_rmssd)
        draw_panel(dashboard,
            "STRESS / HRV",
            int(current_rmssd), "ms RMSSD",
            stress_status, stress_color,
            x=430, y=510)

        # SpO2 Panel
        spo2_status, spo2_color = get_spo2_status(current_spo2)
        draw_panel(dashboard,
            "SpO2",
            current_spo2, "% oxygen",
            spo2_status, spo2_color,
            x=640, y=510)

        # ── Health Score Circle ───────────────────────────
        h_score = health_score(
            current_bpm, current_br,
            current_rmssd, current_spo2)
        draw_health_score(dashboard, h_score, x=790, y=560)

        # ── Progress bars ─────────────────────────────────
        y_prog = 650
        cv2.putText(dashboard, "Data collection progress:",
                    (10, y_prog),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (100,100,100), 1)

        labels  = ["BPM", "Breathing", "HRV", "SpO2"]
        mins    = [BPM_MIN, BR_MIN, HRV_MIN, SPO2_MIN]
        colors  = [(0,140,255),(0,200,100),
                   (0,165,255),(0,255,200)]

        for i, (lbl, mn, clr) in enumerate(
                zip(labels, mins, colors)):
            xp   = 10 + i * 210
            yp   = y_prog + 18
            prog = min(100, int(collected/mn*100))
            cv2.putText(dashboard, f"{lbl}:{prog}%",
                        (xp, yp),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (120,120,120), 1)
            cv2.rectangle(dashboard,
                (xp, yp+4), (xp+190, yp+12),
                (40,40,40), -1)
            fill = int(1.9 * prog)
            cv2.rectangle(dashboard,
                (xp, yp+4), (xp+fill, yp+12),
                clr, -1)

        # ── Pulse Signal Graph ────────────────────────────
        pulse_graph = draw_graph(
            pos_signal_disp if pos_signal_disp
            else list(g_buffer),
            width=860, height=75,
            color=(0,140,255),
            label="Pulse signal (POS algorithm)")
        dashboard[690:765, 0:860] = pulse_graph

        # ── Breathing Signal Graph ────────────────────────
        br_graph = draw_graph(
            br_signal_disp if br_signal_disp
            else list(g_buffer),
            width=860, height=75,
            color=(0,200,100),
            label="Breathing signal (0.1-0.5 Hz)")
        dashboard[765:840, 0:860] = br_graph

        # ── Alert system ──────────────────────────────────
        alerts = []
        if current_bpm > 0 and current_bpm > 130:
            alerts.append("HIGH HEART RATE!")
        if current_bpm > 0 and current_bpm < 45:
            alerts.append("LOW HEART RATE!")
        if current_br > 0 and current_br > 25:
            alerts.append("HIGH BREATHING RATE!")
        if current_spo2 > 0 and current_spo2 < 92:
            alerts.append("LOW SpO2!")

        if alerts:
            alert_text = " | ".join(alerts)
            cv2.rectangle(dashboard,
                (0,840), (860,870), (0,0,150), -1)
            cv2.putText(dashboard,
                f"ALERT: {alert_text}",
                (10,860), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0,0,255), 2)
        else:
            cv2.rectangle(dashboard,
                (0,840), (860,870), (0,40,0), -1)
            cv2.putText(dashboard,
                "All parameters normal",
                (300,860), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0,200,0), 1)

        # Bottom disclaimer
        cv2.putText(dashboard,
            "For educational use only — Not a medical device",
            (240,890), cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (60,60,60), 1)

        # Print to console every 90 frames
        if frame_count % 90 == 0 and current_bpm > 0:
            print(f"BPM:{current_bpm} | "
                  f"BR:{current_br} | "
                  f"HRV:{current_rmssd}ms | "
                  f"SpO2:{current_spo2}% | "
                  f"Score:{h_score}")

        cv2.imshow("Health Fusion Dashboard", dashboard)

        key = cv2.waitKey(30)
        if key & 0xFF == ord('q'):
            print("Closing.")
            break

cap.release()
cv2.destroyAllWindows()
print("="*50)
print("FINAL RESULTS:")
print(f"Heart Rate:    {current_bpm} BPM")
print(f"Breathing:     {current_br} breaths/min")
print(f"HRV (RMSSD):   {current_rmssd} ms")
print(f"SpO2:          {current_spo2}%")
print(f"Health Score:  {health_score(current_bpm, current_br, current_rmssd, current_spo2)}/100")
print("="*50)
print("Done!")