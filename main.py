# main.py
# This is the main file to run the project.
# Run this file to start the pulse detector!
# Usage: python main.py

import subprocess
import sys
import os

print("="*50)
print("  Contactless Pulse Detector")
print("  Using Facial Video Analysis")
print("="*50)
print()
print("Choose what to run:")
print("1 - Run on laptop (webcam window)")
print("2 - Run as website (browser version)")
print()

choice = input("Enter 1 or 2: ").strip()

if choice == "1":
    print("\nStarting laptop version...")
    src = os.path.join(os.path.dirname(__file__), "src", "step8_live_display.py")
    subprocess.run([sys.executable, src])

elif choice == "2":
    print("\nStarting web version...")
    src = os.path.join(os.path.dirname(__file__), "src", "app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", src])

else:
    print("Invalid choice! Please enter 1 or 2.")