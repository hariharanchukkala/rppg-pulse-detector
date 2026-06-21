\# ❤️ Contactless Pulse Detection Using Face Video



A college project that detects heart rate and other vital signs

from a webcam video using Remote Photoplethysmography (rPPG).

No physical contact needed!



\---



\## 🎯 What This Project Does



This project measures vital signs by analyzing tiny color changes

in face skin caused by blood flow, captured through a webcam.



\### Parameters detected:

\- ❤️ Heart Rate (BPM)

\- 🫁 Breathing Rate (breaths/min)

\- 😰 Stress Level (HRV analysis)

\- 🩸 SpO2 (Blood Oxygen estimation)

\- 📊 Overall Health Score



\---



\## 🧠 How It WorksWebcam → Face Detection → Landmarks → ROI Selection



→ Green Signal → Buffer → FFT → BPM



1\. Webcam captures live video of face

2\. MediaPipe detects face and finds 478 landmark points

3\. Cheek and forehead regions selected as ROI

4\. Green channel value extracted from ROI every frame

5\. 150 frames stored in buffer (5 seconds of data)

6\. Bandpass filter removes noise

7\. FFT finds dominant pulse frequency

8\. Frequency converted to BPM

9\. CHROM and POS algorithms improve accuracy



\---



\## 🛠️ Technologies Used



\- Python 3.14

\- OpenCV — webcam and image processing

\- MediaPipe — face detection and landmarks

\- NumPy — signal processing

\- SciPy — bandpass filter

\- Streamlit — web interface



\---



\## 📁 Project Structure

rppg-pulse-detector/



├── main.py                      # Run this to start



├── requirements.txt             # All libraries needed



├── README.md                    # This file



├── .gitignore                   # Files to ignore



└── src/



├── step1\_webcam.py          # Open webcam



├── step2\_face\_detection.py  # Detect face



├── step3\_landmarks.py       # Face landmarks



├── step4\_roi.py             # Select cheek ROI



├── step5\_signal.py          # Extract green signal



├── step6\_buffer.py          # Buffer signal



├── step7\_bpm.py             # Calculate BPM



├── step8\_live\_display.py    # Live display



├── step9\_chrom\_bpm.py       # CHROM algorithm



├── step10\_pos\_bpm.py        # POS algorithm



├── step11\_breathing.py      # Breathing rate



├── step12\_hrv\_stress.py     # HRV and stress



├── step13\_spo2.py           # SpO2 estimation



├── step14\_fusion.py         # Final dashboard



└── app.py                   # Web version



\---



\## ⚙️ Installation



\### Step 1 — Clone the repository

git clone https://github.com/hariharanchukkala/rppg-pulse-detector.git



cd rppg-pulse-detector



\### Step 2 — Install libraries

pip install -r requirements.txt



\### Step 3 — Run the project

python main.py



\---



\## 📊 Results



| Parameter | Normal Range | Our Result |

|-----------|-------------|------------|

| Heart Rate | 60-100 BPM | 60-80 BPM |

| Breathing | 12-20/min | 14/min |

| SpO2 | 95-100% | \~97% |

| Stress | Low is good | Very Relaxed |

| Health Score | 80+ is good | 80/100 |



\---



\## 📈 Algorithms Used



| Algorithm | Accuracy | Method |

|-----------|----------|--------|

| Green channel | Basic | Single channel |

| CHROM | Better | 3 channel chrominance |

| POS | Best | Skin plane projection |



\---



\## ⚠️ Limitations



\- Requires good lighting on face

\- Movement reduces accuracy

\- SpO2 is estimated — not medical grade

\- Not a medical device — educational only



\---



\## 👨‍💻 Author



\- Name: Hariharan Chukkala

\- Project: B.Tech College Project

\- Domain: Computer Vision and Signal Processing

\- GitHub: github.com/hariharanchukkala



\---



\## 📚 References



1\. Verkruysse et al. (2008) — Remote plethysmographic imaging

2\. MediaPipe Face Landmarker — Google AI

3\. de Haan \& Jeanne (2013) — CHROM algorithm

4\. Wang et al. (2016) — POS algorithm

5\. OpenCV Documentation

6\. SciPy Signal Processing Documentation



\---



\## ⚠️ Disclaimer



This project is for educational purposes only.

It is NOT a medical device.

Always consult a doctor for health concerns.

