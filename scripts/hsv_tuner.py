"""HSV tuning tool. Opens camera with sliders to find color ranges.
Works with ZED 2 via OpenCV (left eye crop) or any webcam."""

import cv2
import numpy as np

cam_index = 0  # ZED 2
cap = cv2.VideoCapture(cam_index)

cv2.namedWindow("HSV Tuner")
cv2.createTrackbar("H_lo", "HSV Tuner", 0, 180, lambda x: None)
cv2.createTrackbar("S_lo", "HSV Tuner", 120, 255, lambda x: None)
cv2.createTrackbar("V_lo", "HSV Tuner", 70, 255, lambda x: None)
cv2.createTrackbar("H_hi", "HSV Tuner", 180, 180, lambda x: None)
cv2.createTrackbar("S_hi", "HSV Tuner", 255, 255, lambda x: None)
cv2.createTrackbar("V_hi", "HSV Tuner", 255, 255, lambda x: None)

print("Adjust sliders until ONLY the target color shows white on the right.")
print("Press 'q' to quit and print values.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Crop ZED side-by-side if needed
    if frame.shape[1] > 1300:
        frame = frame[:, :frame.shape[1]//2, :]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lo = np.array([cv2.getTrackbarPos("H_lo", "HSV Tuner"),
                   cv2.getTrackbarPos("S_lo", "HSV Tuner"),
                   cv2.getTrackbarPos("V_lo", "HSV Tuner")])
    hi = np.array([cv2.getTrackbarPos("H_hi", "HSV Tuner"),
                   cv2.getTrackbarPos("S_hi", "HSV Tuner"),
                   cv2.getTrackbarPos("V_hi", "HSV Tuner")])
    mask = cv2.inRange(hsv, lo, hi)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("HSV Tuner", np.hstack([frame, result]))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"\nHSV range: (({lo[0]}, {lo[1]}, {lo[2]}), ({hi[0]}, {hi[1]}, {hi[2]}))")
        print("Copy this into HSV_CONFIG in zed_perception.py")
        break

cap.release()
cv2.destroyAllWindows()

