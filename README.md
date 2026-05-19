\# ❤️ Contactless Pulse Detector Using Facial Video



A college project that detects heart rate (BPM) from a webcam video

using Remote Photoplethysmography (rPPG) — no physical contact needed!



\---



\## 🎯 What This Project Does



This project measures your pulse by analyzing tiny color changes

in your face skin that happen every time your heart beats.

The camera detects these changes and converts them into BPM.



\---



\## 🧠 How It Works



1\. Webcam captures live video of your face

2\. MediaPipe detects your face and finds landmark points

3\. Cheek region (ROI) is selected using landmark points

4\. Average green channel value is extracted from cheeks every frame

5\. 150 frames of green values are stored in a buffer

6\. FFT (Fast Fourier Transform) finds the dominant pulse frequency

7\. Frequency is converted to BPM and displayed live



\---



\## 🛠️ Technologies Used



\- Python 3.14

\- OpenCV — webcam and image processing

\- MediaPipe — face detection and landmarks

\- NumPy — signal processing and arrays

\- SciPy — bandpass filter for noise removal

\- Matplotlib — signal visualization

\- Streamlit — web interface



\---



\## 📁 Project Structure

rppg-pulse-detector/

├── main.py                    # Run this to start!

├── requirements.txt           # All libraries

├── README.md                  # This file

├── .gitignore                 # Files to ignore

└── src/

├── step1\_webcam.py        # Step 1: Open webcam

├── step2\_face\_detection.py # Step 2: Detect face

├── step3\_landmarks.py     # Step 3: Face landmarks

├── step4\_roi.py           # Step 4: Select cheek ROI

├── step5\_signal.py        # Step 5: Extract green signal

├── step6\_buffer.py        # Step 6: Buffer signal

├── step7\_bpm.py           # Step 7: Calculate BPM

├── step8\_live\_display.py  # Step 8: Live display

└── app.py                 # Web version



\---



\## ⚙️ Installation



1\. Clone this repository:

git clone https://github.com/hariharanchukkala/rppg-pulse-detector.git

2\. Install required libraries:

pip install -r requirements.txt

3\. Run the project:

python main.py

\---



\## 📊 Results



\- Detects BPM in approximately 5 seconds

\- Normal range: 60-100 BPM

\- Accuracy improves when sitting still with good lighting



\---



\## ⚠️ Limitations



\- Requires good lighting on face

\- Movement reduces accuracy

\- Not a medical device — for educational purposes only



\---



\## 👨‍💻 Author



\- Name: Harih

\- College Project — B.Tech

\- Domain: Computer Vision and Signal Processing



\---



\## 📚 References



\- MediaPipe Face Landmarker — Google

\- Remote Photoplethysmography (rPPG) research

\- Samuel Pröll — Extracting heartbeat signals from webcam video

\- OpenCV Documentation

