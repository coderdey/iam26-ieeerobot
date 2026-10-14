"""Take snapshots from camera for YOLO training data or reference photos."""

import cv2
import os

os.makedirs("../data/yolo_images", exist_ok=True)

cam_index = 0  # ZED 2 = 0, wrist = 2 (check with lerobot-find-cameras)
cap = cv2.VideoCapture(cam_index)

if not cap.isOpened():
    print(f"Camera {cam_index} failed")
    exit(1)

i = 0
print("Press 's' to save snapshot, 'q' to quit")
print("Move objects/bins between snapshots for variety!")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # If ZED gives 2560x720 side-by-side, crop to left eye
    if frame.shape[1] > 1300:
        frame = frame[:, :frame.shape[1]//2, :]

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        path = f"../data/yolo_images/img_{i:04d}.png"
        cv2.imwrite(path, frame)
        print(f"Saved {path} ({i+1} total)")
        i += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Done. Saved {i} images.")

