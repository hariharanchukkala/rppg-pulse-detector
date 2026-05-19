# step1_webcam.py
# This code opens your webcam and shows the video on screen.
# Press 'q' on your keyboard to quit.

import cv2  # OpenCV library - used for webcam and images

# Open the webcam (0 = default/first webcam on your laptop)
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    print("ERROR: Cannot open webcam. Check if it is connected.")
    exit()

print("Webcam opened! Press 'q' to quit.")

# Keep showing video until user presses 'q'
while True:
    # Read one frame (one picture) from the webcam
    success, frame = cap.read()

    # If reading failed, stop
    if not success:
        print("ERROR: Could not read frame from webcam.")
        break

    # Show the frame in a window called "Webcam Test"
    cv2.imshow("Webcam Test", frame)

    # Wait 1 millisecond. If user presses 'q', exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("You pressed 'q'. Closing webcam.")
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
print("Done!")