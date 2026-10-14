"""Auto-calibrate camera-to-robot transform using QR codes on the workspace.
Print QR codes labeled 'qr_1' through 'qr_4', tape at known positions,
measure their positions relative to arm base with a ruler."""

import cv2
import numpy as np


# FILL IN: measured positions of QR codes relative to robot base (meters)
# Measure with ruler from arm base center to each QR code center
KNOWN_QR_POSITIONS = {
    "qr_1": np.array([0.10, 0.10, 0.0]),   # UPDATE with real measurements
    "qr_2": np.array([0.20, 0.10, 0.0]),   # UPDATE
    "qr_3": np.array([0.10, 0.20, 0.0]),   # UPDATE
    "qr_4": np.array([0.20, 0.20, 0.0]),   # UPDATE
}


def solve_rigid_transform(A, B):
    """Find 4x4 transform T such that B ~ T @ A. A,B are Nx3."""
    assert A.shape == B.shape
    cA = np.mean(A, axis=0)
    cB = np.mean(B, axis=0)
    H = (A - cA).T @ (B - cB)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = cB - R @ cA
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def calibrate_from_zed(zed_perception):
    """Run QR detection on live ZED feed and compute transform."""
    detector = cv2.QRCodeDetector()

    print("Point ZED 2 at workspace with all 4 QR codes visible.")
    print("Press 'c' to capture and calibrate, 'q' to quit.")

    while True:
        bgr, pc = zed_perception.grab()
        if bgr is None:
            continue

        cv2.imshow("QR Calibration", bgr)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            retval, decoded_info, points, _ = detector.detectAndDecodeMulti(bgr)
            if not retval or len(decoded_info) < 4:
                print(f"Only found {len(decoded_info) if retval else 0} QR codes. Need 4.")
                continue

            cam_points = []
            robot_points = []
            for info, pts in zip(decoded_info, points):
                if info not in KNOWN_QR_POSITIONS:
                    print(f"Unknown QR: '{info}', skipping")
                    continue
                cx, cy = pts.mean(axis=0)
                cam_xyz = zed_perception.pixel_to_robot_xyz(int(cx), int(cy))
                if cam_xyz is None:
                    print(f"No depth for QR '{info}', skipping")
                    continue
                cam_points.append(cam_xyz)
                robot_points.append(KNOWN_QR_POSITIONS[info])
                print(f"  {info}: pixel=({cx:.0f},{cy:.0f}) cam={cam_xyz} robot={KNOWN_QR_POSITIONS[info]}")

            if len(cam_points) >= 4:
                T = solve_rigid_transform(np.array(cam_points), np.array(robot_points))
                np.save("../config/camera_transform.npy", T)
                print(f"\nCalibration saved to config/camera_transform.npy")
                print(f"Transform:\n{T}")
                zed_perception.T_cam_to_robot = T
                break
            else:
                print("Not enough valid QR detections. Try again.")

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
