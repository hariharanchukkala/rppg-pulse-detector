# run_app.py
# This script starts streamlit AND ngrok together automatically!

import subprocess
import time
import sys
from pyngrok import ngrok

# Step 1 - Start streamlit in background
print("Starting Streamlit...")
streamlit_process = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "app.py",
     "--server.headless", "true"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE)

# Step 2 - Wait for streamlit to start
print("Waiting for Streamlit to start...")
time.sleep(5)

# Step 3 - Start ngrok tunnel
print("Starting ngrok tunnel...")
tunnel = ngrok.connect(8501)

# Step 4 - Print the public link
print("\n" + "="*50)
print("SUCCESS! Share this link with anyone!")
print(f"Link: {tunnel.public_url}")
print("="*50)
print("\nKeep this window open while sharing!")
print("Press Ctrl+C to stop.")

# Step 5 - Keep running until user stops
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopping...")
    ngrok.disconnect(tunnel.public_url)
    streamlit_process.terminate()
    print("Done!")