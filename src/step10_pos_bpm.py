# step10_pos_bpm.py
# POS Algorithm - most accurate rPPG method
# Press 'q' to quit

import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os
from collections import deque
from scipy.signal import butter, filtfilt

os.environ["GLOG_minloglevel"] = "3"
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
BUFFER_SIZE = 256
FPS         = 30

# RGB buffers
r_buffer = deque(maxlen=BUFFER_SIZE)
g_buffer = deque(maxlen=BUFFER_SIZE)
b_buffer = deque(maxlen=BUFFER_SIZE)

# BPM history
bpm_history = []

# Bandpass filter
def bandpass_filter(signal, lowcut=0.7, highcut=3.5, fs=30, order=4):
    nyq  = fs / 2
    low  = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# POS Algorithm
def pos_bpm(r_buf, g_buf, b_buf, fps=30):
    r = np.array(r_buf, dtype=float)
    g = np.array(g_buf, dtype=float)
    b = np.array(b_buf, dtype=float)

    r_mean = np.mean(r)
    g_mean = np.mean(g)
    b_mean = np.mean(b)

    if r_mean == 0 or g_mean == 0 or b_mean == 0:
        return 0, []

    # Normalize channels
    rn = r / r_mean
    gn = g / g_mean
    bn = b / b_mean

    # POS projection onto skin plane
    S1 = rn - gn
    S2 = rn + gn - 2 * bn

    std_s1 = np.std(S1)
    std_s2 = np.std(S2)

    if std_s2 == 0:
        return 0, []

    alpha = std_s1 / std_s2

    # Final POS signal
    pos_signal = S1 + alpha * S2
    pos_signal = pos_signal - np.mean(pos_signal)

    # Bandpass filter
    try:
        filtered = bandpass_filter(pos_signal, fs=fps)
    except:
        return 0, []

    # FFT
    fft_result    = np.fft.rfft(filtered)
    fft_freqs     = np.fft.rfftfreq(len(filtered), d=1.0/fps)
    valid         = (fft_freqs >= 0.7) & (fft_freqs <= 3.5)

    if not np.any(valid):
        return 0, filtered.tolist()

    fft_magnitude = np.abs(fft_result)
    peak_freq     = fft_freqs[valid][np.argmax(fft_magnitude[valid])]
    bpm           = peak_freq * 60
    return round(bpm), filtered.tolist()

# Signal quality score
def signal_quality(filtered_signal):
    if len(filtered_signal) < 10:
        return 0
    sig     = np.array(filtered_signal)
    fft_mag = np.abs(np.fft.rfft(sig))
    if np.mean(fft_mag) == 0:
        return 0
    snr     = np.max(fft_mag) / np.mean(fft_mag)
    quality = min(100, int(snr * 5))
    return quality

# Heart rate zone
def get_zone(bpm):
    if bpm < 50:
        return "Too Low",   (255, 100,   0)
    elif bpm < 60:
        return "Resting",   (  0, 200, 200)
    elif bpm <= 100:
        return "Normal",    (  0, 255,   0)
    elif bpm <= 120:
        return "Elevated",  (  0, 165, 255)
    elif bpm <= 140:
        return "High",      (  0,  80, 255)
    else:
        return "Very High", (  0,   0, 255)

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
def draw_graph(signal, width=640, height=120, color=(0,255,128), label="POS Signal"):
    graph = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(graph, label, (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)
    if len(signal) < 2:
        return graph
    sig    = np.array(signal)
    mn, mx = sig.min(), sig.max()
    if mx == mn:
        return graph
    scaled = ((sig - mn) / (mx - mn) * (height - 24) + 12).astype(int)
    xstep  = width / len(scaled)
    for i in range(1, len(scaled)):
        x1 = int((i-1) * xstep)
        x2 = int(i * xstep)
        y1 = height - scaled[i-1]
        y2 = height - scaled[i]
        cv2.line(graph, (x1,y1), (x2,y2), color, 2)
    return graph

# Draw quality bar
def draw_quality_bar(frame, quality, x=460, y=80):
    cv2.putText(frame, "Quality:", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
    cv2.rectangle(frame, (x, y), (x+150, y+12), (50,50,50), -1)
    fill  = int(quality * 1.5)
    color = (0,255,0) if quality > 60 else (0,165,255) if quality > 30 else (0,0,255)
    cv2.rectangle(frame, (x, y), (x+fill, y+12), color, -1)
    cv2.putText(frame, f"{quality}%", (x+155, y+11), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

print("POS Algorithm started! Press 'q' to quit.")
print("Sit very still — buffer needs 8.5 seconds...")
print("---")

cv2.namedWindow("POS Health Monitor", cv2.WINDOW_NORMAL)
cv2.resizeWindow("POS Health Monitor", 640, 640)
cv2.moveWindow("POS Health Monitor", 100, 30)

current_bpm     = 0
pos_signal_disp = []
quality         = 0

# Main loop
with FaceLandmarker.create_from_options(options) as landmarker:

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame     = cv2.resize(frame, (640, 480))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

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

            # Calculate BPM when buffer is full
            if collected == BUFFER_SIZE:
                raw_bpm, pos_signal_disp = pos_bpm(r_buffer, g_buffer, b_buffer, fps=FPS)
                quality = signal_quality(pos_signal_disp)

                if 45 <= raw_bpm <= 150:
                    bpm_history.append(raw_bpm)
                if len(bpm_history) > 7:
                    bpm_history.pop(0)
                if len(bpm_history) > 0:
                    current_bpm = round(sum(bpm_history) / len(bpm_history))
                    print(f"POS BPM: {current_bpm} | Quality: {quality}% | Raw: {raw_bpm}")

            # Draw ROI overlay
            overlay = frame.copy()
            cv2.fillPoly(overlay, [left_pts],     (  0, 255,   0))
            cv2.fillPoly(overlay, [right_pts],    (  0, 255,   0))
            cv2.fillPoly(overlay, [forehead_pts], (  0, 200, 255))
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

            # Top info bar
            cv2.rectangle(frame, (0,0), (640,60), (15,15,15), -1)
            cv2.putText(frame, f"R:{r:.0f}  G:{g:.0f}  B:{b:.0f}", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            cv2.putText(frame, f"Buffer: {collected}/{BUFFER_SIZE}", (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            cv2.putText(frame, "POS Algorithm", (490, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)

            # BPM display
            if current_bpm > 0:
                zone, zcolor = get_zone(current_bpm)
                cv2.putText(frame, f"BPM: {current_bpm}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 1.9, (0,60,255), 4)
                cv2.putText(frame, f"BPM: {current_bpm}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 1.9, (0,100,255), 2)
                cv2.putText(frame, f"Zone: {zone}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, zcolor, 2)
                draw_quality_bar(frame, quality)
            else:
                progress = int((collected / BUFFER_SIZE) * 100)
                cv2.putText(frame, f"Initializing... {progress}%", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,200), 2)
                cv2.rectangle(frame, (20,130), (620,145), (40,40,40), -1)
                fill = int(6 * progress)
                cv2.rectangle(frame, (20,130), (20+fill,145), (0,200,100), -1)

            cv2.putText(frame, "Green=Cheeks  Cyan=Forehead", (20, 472), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150,150,150), 1)

        else:
            cv2.rectangle(frame, (0,0), (640,480), (20,20,20), -1)
            cv2.putText(frame, "No Face Detected", (170, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,200), 3)
            cv2.putText(frame, "Please face the camera", (180, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150,150,150), 1)

        # Signal graph
        graph    = draw_graph(pos_signal_disp if pos_signal_disp else list(g_buffer))
        combined = np.vstack([frame, graph])
        cv2.imshow("POS Health Monitor", combined)

        key = cv2.waitKey(30)
        if key & 0xFF == ord('q'):
            print("Closing.")
            break

cap.release()
cv2.destroyAllWindows()
print(f"Final BPM: {current_bpm}")
print(f"Signal Quality: {quality}%")
print("Done!")