import cv2
import time

# Open the camera
camera = cv2.VideoCapture(0)  # Change the index to 1 if you're using the Raspberry Pi Camera Module v2

# Create a window to display the feed
cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)

# Set desired frame rate
frame_rate = 24
delay = 1 / frame_rate

while True:
    # Read a frame from the camera
    ret, frame = camera.read()

    # Display the frame
    cv2.imshow("Camera Feed", frame)

    # Check if a key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Delay to achieve desired frame rate
    time.sleep(delay)

# Release the camera and close the window
camera.release()
cv2.destroyAllWindows()
