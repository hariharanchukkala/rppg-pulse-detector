# step9_chrom_bpm.py
# PHASE 1 — CHROM Algorithm for better BPM accuracy
# Uses all 3 color channels (R,G,B) instead of just green
# Much more accurate than green channel only method
# Press 'q' to quit

import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os
from collections import deque
from scipy.signal import butter, filtfilt

# ── Download model if needed ──────────────────────────────
model_path = "face_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading model... please wait...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Model downloaded!")

# ── MediaPipe setup ───────────────────────────────────────
BaseOptions      = mp.tasks.BaseOptions
FaceLandmarker   = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# ── Cheek landmark numbers ────────────────────────────────
LEFT_CHEEK  = [116, 123, 147, 213, 192, 214, 212, 216, 206, 203]
RIGHT_CHEEK = [345, 352, 376, 433, 416, 434, 432, 436, 426, 423]
FOREHEAD    = [10, 67, 69, 104, 108, 151, 299, 337, 338]

# ── Buffer settings ───────────────────────────────────────
BUFFER_SIZE = 150
FPS         = 30

# ── Buffers for R, G, B signals ──────────────────────────
r_buffer = deque(maxlen=BUFFER_SIZE)
g_buffer = deque(maxlen=BUFFER_SIZE)
b_buffer = deque(maxlen=BUFFER_SIZE)

# ── BPM history for smoothing ─────────────────────────────
bpm_history = []

# ── Bandpass filter ───────────────────────────────────────
def bandpass_filter(signal, lowcut=0.8, highcut=3.0, fs=30, order=3):
    nyq  = fs / 2
    low  = lowcut  / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# ── CHROM Algorithm ───────────────────────────────────────
def chrom_bpm(r_buf, g_buf, b_buf, fps=30):
    # Convert buffers to numpy arrays
    r = np.array(r_buf, dtype=float)
    g = np.array(g_buf, dtype=float)
    b = np.array(b_buf, dtype=float)

    # Step 1: Normalize each channel
    # This removes lighting changes
    r_mean = np.mean(r)
    g_mean = np.mean(g)
    b_mean = np.mean(b)

    # Avoid division by zero
    if r_mean == 0 or g_mean == 0 or b_mean == 0:
        return 0

    # Normalized channels
    rn = r / r_mean
    gn = g / g_mean
    bn = b / b_mean

    # Step 2: CHROM signals
    # Xs removes skin color effect
    # Xy removes more noise
    Xs = 3 * rn - 2 * gn
    Xy = 1.5 * rn + gn - 1.5 * bn

    # Step 3: Combine signals
    # Alpha balances the two signals
    std_xs = np.std(Xs)
    std_xy = np.std(Xy)

    if std_xy == 0:
        return 0

    alpha  = std_xs / std_xy
    signal = Xs - alpha * Xy

    # Step 4: Remove mean (detrend)
    signal = signal - np.mean(signal)

    # Step 5: Bandpass filter
    try:
        filtered = bandpass_filter(signal, fs=fps)
    except:
        return 0

    # Step 6: FFT to find dominant frequency
    fft_result    = np.fft.rfft(filtered)
    fft_freqs     = np.fft.rfftfreq(len(filtered), d=1.0/fps)
    valid         = (fft_freqs >= 0.8) & (fft_freqs <= 3.0)

    if not np.any(valid):
        return 0

    fft_magnitude = np.abs(fft_result)
    peak_freq     = fft_freqs[valid][np.argmax(fft_magnitude[valid])]

    # Step 7: Convert to BPM
    bpm = peak_freq * 60
    return round(bpm)

# ── Helper: get region mean for R,G,B ────────────────────
def get_rgb_mean(frame, mask):
    b_mean = cv2.mean(frame[:,:,0], mask=mask)[0]  # Blue
    g_mean = cv2.mean(frame[:,:,1], mask=mask)[0]  # Green
    r_mean = cv2.mean(frame[:,:,2], mask=mask)[0]  # Red
    return r_mean, g_mean, b_mean

# ── Helper: get landmark points ──────────────────────────
def get_points(landmarks, indices, w, h):
    pts = []
    for idx in indices:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        pts.append([x, y])
    return np.array(pts, dtype=np.int32)

# ── Helper: draw signal graph ─────────────────────────────
def draw_graph(signal, width=640, height=120, color=(0,255,0)):
    graph = np.zeros((height, width, 3), dtype=np.uint8)
    if len(signal) < 2:
        return graph
    sig = np.array(signal)
    mn, mx = sig.min(), sig.max()
    if mx == mn:
        return graph
    scaled = ((sig - mn) / (mx - mn) * (height - 20) + 10).astype(int)
    xstep  = width / len(scaled)
    for i in range(1, len(scaled)):
        x1 = int((i-1) * xstep)
        x2 = int(i     * xstep)
        y1 = height - scaled[i-1]
        y2 = height - scaled[i]
        cv2.line(graph, (x1,y1), (x2,y2), color, 2)
    cv2.putText(graph, "CHROM Pulse Signal",
                (10, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (180,180,180), 1)
    return graph

# ── Open webcam ───────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

print("CHROM Algorithm started! Press 'q' to quit.")
print("Sit still and wait 5 seconds for BPM...")
print("---")

# Force create window
cv2.namedWindow("CHROM Health Monitor", cv2.WINDOW_NORMAL)
cv2.resizeWindow("CHROM Health Monitor", 640, 620)
cv2.moveWindow("CHROM Health Monitor", 100, 50)

current_bpm    = 0
chrom_signal   = []

# ── Main loop ─────────────────────────────────────────────
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

        result = landmarker.detect(mp_image)
        h, w, _ = frame.shape

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            # Get landmark points
            left_pts     = get_points(landmarks, LEFT_CHEEK,  w, h)
            right_pts    = get_points(landmarks, RIGHT_CHEEK, w, h)
            forehead_pts = get_points(landmarks, FOREHEAD,    w, h)

            # Create combined mask
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [left_pts],     255)
            cv2.fillPoly(mask, [right_pts],    255)
            cv2.fillPoly(mask, [forehead_pts], 255)

            # Get R,G,B means from all regions
            r, g, b = get_rgb_mean(frame, mask)

            # Add to buffers
            r_buffer.append(r)
            g_buffer.append(g)
            b_buffer.append(b)

            collected = len(g_buffer)

            # Calculate BPM when buffer full
            if collected == BUFFER_SIZE:
                raw_bpm    = chrom_bpm(
                    r_buffer, g_buffer, b_buffer, fps=FPS)
                chrom_signal = list(
                    np.array(r_buffer) / np.mean(r_buffer) * 100)

                # Accept only valid readings
                if 45 <= raw_bpm <= 150:
                    bpm_history.append(raw_bpm)

                # Keep last 5 readings
                if len(bpm_history) > 5:
                    bpm_history.pop(0)

                # Smooth BPM
                if len(bpm_history) > 0:
                    current_bpm = round(
                        sum(bpm_history) / len(bpm_history))

                    print(f"CHROM BPM: {current_bpm} "
                          f"(raw={raw_bpm}, "
                          f"history={bpm_history})")

            # Draw ROI regions on frame
            overlay = frame.copy()
            cv2.fillPoly(overlay, [left_pts],     (0,255,0))
            cv2.fillPoly(overlay, [right_pts],    (0,255,0))
            cv2.fillPoly(overlay, [forehead_pts], (0,200,255))
            frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

            # ── Info panel ────────────────────────────────
            cv2.rectangle(frame, (0,0), (640,55), (20,20,20), -1)

            # Green value
            cv2.putText(frame,
                f"G:{g:.0f} R:{r:.0f} B:{b:.0f}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0,255,255), 1)

            # Buffer progress
            cv2.putText(frame,
                f"Buffer: {collected}/{BUFFER_SIZE}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255,255,0), 1)

            # Algorithm label
            cv2.putText(frame,
                "CHROM Algorithm",
                (440, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (180,180,180), 1)

            # BPM display
            if current_bpm > 0:
                # Heart rate zone
                if current_bpm < 60:
                    zone = "Low"
                    zclr = (255,165,0)
                elif current_bpm <= 100:
                    zone = "Normal"
                    zclr = (0,255,0)
                elif current_bpm <= 130:
                    zone = "Elevated"
                    zclr = (0,165,255)
                else:
                    zone = "High"
                    zclr = (0,0,255)

                # Big BPM number
                cv2.putText(frame,
                    f"BPM: {current_bpm}",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX,
                    1.8, (0,0,255), 3)

                # Zone label
                cv2.putText(frame,
                    f"Zone: {zone}",
                    (20, 145), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, zclr, 2)
            else:
                cv2.putText(frame,
                    "Collecting data...",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255,255,255), 2)

            # ROI legend
            cv2.putText(frame,
                "Green=Cheeks  Yellow=Forehead",
                (20, 475), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (180,180,180), 1)

        else:
            cv2.putText(frame,
                "No Face — Please face the camera",
                (80, 240), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0,0,255), 2)

        # Draw pulse graph below
        graph = draw_graph(
            chrom_signal if chrom_signal else list(g_buffer))
        combined = np.vstack([frame, graph])

        cv2.imshow("CHROM Health Monitor", combined)

        key = cv2.waitKey(30)
        if key & 0xFF == ord('q'):
            print("Closing.")
            break

cap.release()
cv2.destroyAllWindows()
print(f"Final BPM: {current_bpm}")
print("Done!")