import cv2
import numpy as np
import time

# ---------------- Setup Webcam ----------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW for Windows
if not cap.isOpened():
    print("ERROR: Could not access webcam.")
    exit()

# Reduce resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(2)  # Warm-up camera

# ---------------- Capture Background ----------------
print("Capturing background... Stay out of frame.")
background_frames = []
for i in range(30):  # fewer frames = faster
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    background_frames.append(frame)

# Take median for stability
background = np.median(background_frames, axis=0).astype(np.uint8)

# ---------------- HSV Color Range ----------------
# Use trackbars for perfect tuning if needed
lower1 = np.array([0, 120, 70])    # red lower range 1
upper1 = np.array([10, 255, 255])  # red upper range 1
lower2 = np.array([170, 120, 70])  # red lower range 2
upper2 = np.array([180, 255, 255]) # red upper range 2

# ---------------- Invisibility Loop ----------------
print("Starting invisibility cloak effect... Press 'q' to quit.")
kernel = np.ones((3, 3), np.uint8)  # smaller kernel for speed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mask for cloak color
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = mask1 + mask2

    # Refine mask (open → dilate → smooth)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Inverse mask
    mask_inv = cv2.bitwise_not(mask)

    # Replace cloak area with background
    cloak_area = cv2.bitwise_and(background, background, mask=mask)
    rest_area = cv2.bitwise_and(frame, frame, mask=mask_inv)
    final_output = cv2.add(cloak_area, rest_area)

    cv2.imshow("Invisibility Cloak", final_output)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
