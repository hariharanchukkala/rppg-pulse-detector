# step12_hrv_stress.py
# HRV (Heart Rate Variability) and Stress Detection
# HRV measures variation between heartbeats
# High HRV = relaxed and healthy
# Low HRV  = stressed or tired
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

# Buffer - need 60 seconds for good HRV
BUFFER_SIZE = 1800   # 60 sec x 30 fps
FPS         = 30

# Buffers
r_buffer = deque(maxlen=BUFFER_SIZE)
g_buffer = deque(maxlen=BUFFER_SIZE)
b_buffer = deque(maxlen=BUFFER_SIZE)

# History for smoothing
bpm_history    = []
hrv_history    = []
stress_history = []

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

# HRV calculation
def calculate_hrv(g_buf, fps=30):
    g = np.array(g_buf, dtype=float)
    g = g - np.mean(g)

    # Filter to get pulse signal
    try:
        filtered = bandpass_filter(g, 0.7, 3.5, fps)
    except:
        return 0, 0, []

    # Find peaks (each peak = one heartbeat)
    # min distance between peaks = 0.4 sec = 12 frames (150 BPM max)
    min_distance = int(fps * 0.4)
    peaks, _     = find_peaks(filtered, distance=min_distance)

    if len(peaks) < 4:
        return 0, 0, []

    # Calculate RR intervals (time between peaks) in milliseconds
    rr_intervals = np.diff(peaks) / fps * 1000

    if len(rr_intervals) < 3:
        return 0, 0, []

    # RMSSD = Root Mean Square of Successive Differences
    # This is the most important HRV measure
    # Higher RMSSD = higher HRV = more relaxed
    successive_diff = np.diff(rr_intervals)
    rmssd           = np.sqrt(np.mean(successive_diff ** 2))

    # SDNN = Standard Deviation of NN intervals
    # Another important HRV measure
    sdnn = np.std(rr_intervals)

    return round(rmssd, 1), round(sdnn, 1), rr_intervals.tolist()

# Stress level from HRV
def calculate_stress(rmssd, sdnn):
    if rmssd == 0:
        return 0, "Measuring...", (200,200,200)

    # Higher RMSSD = lower stress
    # Typical RMSSD values:
    # >50ms = very relaxed
    # 20-50ms = normal
    # <20ms  = stressed

    if rmssd > 50:
        stress = 10
        label  = "Very Relaxed"
        color  = (0, 255, 100)
    elif rmssd > 35:
        stress = 30
        label  = "Relaxed"
        color  = (0, 220, 80)
    elif rmssd > 25:
        stress = 50
        label  = "Moderate"
        color  = (0, 165, 255)
    elif rmssd > 15:
        stress = 70
        label  = "Stressed"
        color  = (0,  80, 255)
    else:
        stress = 90
        label  = "High Stress"
        color  = (0,   0, 255)

    return stress, label, color

# Get RGB mean from mask
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

# Draw signal graph
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
        x1 = int((i-1) * xstep)
        x2 = int(i     * xstep)
        y1 = height - scaled[i-1]
        y2 = height - scaled[i]
        cv2.line(graph, (x1,y1), (x2,y2), color, 2)
    return graph

# Draw stress bar
def draw_stress_bar(frame, stress, x=10, y=250):
    cv2.putText(frame, "Stress Index:", (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (180,180,180), 1)
    # Background
    cv2.rectangle(frame, (x, y), (x+300, y+16),
                  (50,50,50), -1)
    # Fill
    fill  = int(stress * 3)
    color = (0,255,0) if stress < 30 else \
            (0,165,255) if stress < 60 else (0,0,255)
    cv2.rectangle(frame, (x, y), (x+fill, y+16),
                  color, -1)
    cv2.putText(frame, f"{stress}%", (x+305, y+13),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200,200,200), 1)

# Draw HRV bar
def draw_hrv_bar(frame, rmssd, x=10, y=310):
    cv2.putText(frame, f"HRV (RMSSD): {rmssd} ms",
                (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (180,180,180), 1)
    # Background
    cv2.rectangle(frame, (x, y), (x+300, y+16),
                  (50,50,50), -1)
    # Fill - higher is better for HRV
    fill  = min(300, int(rmssd * 4))
    color = (0,255,0) if rmssd > 35 else \
            (0,165,255) if rmssd > 20 else (0,0,255)
    cv2.rectangle(frame, (x, y), (x+fill, y+16),
                  color, -1)

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

print("HRV + Stress Detection started!")
print("Heart Rate: ready in 8 seconds")
print("HRV + Stress: ready in 30 seconds")
print("Sit still and relax...")
print("Press 'q' to quit")
print("---")

cv2.namedWindow("HRV + Stress Monitor", cv2.WINDOW_NORMAL)
cv2.resizeWindow("HRV + Stress Monitor", 640, 700)
cv2.moveWindow("HRV + Stress Monitor", 100, 20)

current_bpm    = 0
current_rmssd  = 0
current_sdnn   = 0
current_stress = 0
stress_label   = "Measuring..."
stress_color   = (200,200,200)
pulse_signal   = []
rr_intervals   = []

# Minimum frames needed
BPM_MIN  = 256
HRV_MIN  = 900

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

            # Calculate HRV and Stress
            if collected >= HRV_MIN:
                rmssd, sdnn, rr_intervals = calculate_hrv(
                    list(g_buffer)[-HRV_MIN:], fps=FPS)

                if rmssd > 0:
                    hrv_history.append(rmssd)
                if len(hrv_history) > 5:
                    hrv_history.pop(0)
                if len(hrv_history) > 0:
                    current_rmssd = round(
                        sum(hrv_history)/len(hrv_history), 1)
                    current_sdnn  = sdnn
                    current_stress, stress_label, stress_color = \
                        calculate_stress(current_rmssd, current_sdnn)

                    print(f"BPM: {current_bpm} | "
                          f"RMSSD: {current_rmssd}ms | "
                          f"SDNN: {current_sdnn}ms | "
                          f"Stress: {current_stress}% "
                          f"({stress_label})")

            # Prepare pulse graph signal
            g_list = list(g_buffer)
            if len(g_list) > 10:
                g_arr = np.array(g_list, dtype=float)
                g_arr = g_arr - np.mean(g_arr)
                try:
                    pulse_signal = bandpass_filter(
                        g_arr, 0.7, 3.5, FPS).tolist()
                except:
                    pulse_signal = g_list

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

            bpm_prog = min(100, int(collected/BPM_MIN*100))
            hrv_prog = min(100, int(collected/HRV_MIN*100))
            cv2.putText(frame,
                f"BPM:{bpm_prog}%  HRV:{hrv_prog}%",
                (10,38), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (255,255,0), 1)

            # BPM display
            if current_bpm > 0:
                zone, zcolor = get_bpm_zone(current_bpm)
                cv2.putText(frame,
                    f"Heart: {current_bpm} BPM  {zone}",
                    (10,95), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, (0,140,255), 2)
            else:
                cv2.putText(frame,
                    f"Heart Rate: measuring... {bpm_prog}%",
                    (10,95), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (200,200,200), 1)

            # Stress display
            if current_stress > 0:
                cv2.putText(frame,
                    f"Stress: {stress_label}",
                    (10,145), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, stress_color, 2)
                draw_stress_bar(frame, current_stress)
                draw_hrv_bar(frame, current_rmssd)

                cv2.putText(frame,
                    f"SDNN: {current_sdnn} ms",
                    (340, 323), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (150,150,150), 1)

                # HRV interpretation
                if current_rmssd > 50:
                    hrv_msg = "Excellent HRV - Very healthy!"
                    hclr    = (0,255,100)
                elif current_rmssd > 35:
                    hrv_msg = "Good HRV - Relaxed state"
                    hclr    = (0,220, 80)
                elif current_rmssd > 20:
                    hrv_msg = "Moderate HRV - Some stress"
                    hclr    = (0,165,255)
                else:
                    hrv_msg = "Low HRV - High stress detected"
                    hclr    = (0,  0,255)

                cv2.putText(frame, hrv_msg,
                    (10,360), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, hclr, 1)

            else:
                cv2.putText(frame,
                    f"HRV + Stress: measuring... {hrv_prog}%",
                    (10,145), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (200,200,200), 1)
                cv2.putText(frame,
                    "Relax and sit still for 30 seconds...",
                    (10,175), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (150,150,150), 1)
                # Progress bar for HRV
                cv2.rectangle(frame,
                    (10,190), (620,205), (40,40,40), -1)
                fill = int(6.1 * hrv_prog)
                cv2.rectangle(frame,
                    (10,190), (10+fill,205),
                    (0,200,100), -1)

            # Normal ranges reference
            cv2.putText(frame,
                "Normal: BPM 60-100 | RMSSD >20ms | Stress <50%",
                (10,468), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (100,100,100), 1)

        else:
            cv2.rectangle(frame, (0,0), (640,480),
                          (20,20,20), -1)
            cv2.putText(frame,
                "No Face — Please face camera",
                (130,240), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0,0,200), 2)

        # Draw pulse graph
        pulse_graph = draw_graph(
            pulse_signal if pulse_signal else list(g_buffer),
            height=110, color=(0,140,255),
            label="Pulse signal (POS filtered)")

        # Stack frame and graph
        combined = np.vstack([frame, pulse_graph])
        cv2.imshow("HRV + Stress Monitor", combined)

        key = cv2.waitKey(30)
        if key & 0xFF == ord('q'):
            print("Closing.")
            break

cap.release()
cv2.destroyAllWindows()
print(f"Final BPM:    {current_bpm}")
print(f"Final RMSSD:  {current_rmssd} ms")
print(f"Final SDNN:   {current_sdnn} ms")
print(f"Final Stress: {current_stress}% ({stress_label})")
print("Done!")