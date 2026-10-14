"""ZED 2 perception module — object detection + 3D positions.
Uses ZED SDK for proper depth. Used at RUNTIME, not for LeRobot training."""

import pyzed.sl as sl
import numpy as np
import cv2


class ZED2Perception:
    def __init__(self):
        self.zed = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.depth_mode = sl.DEPTH_MODE.ULTRA
        init.coordinate_units = sl.UNIT.METER
        init.depth_minimum_distance = 0.15

        status = self.zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {status}")

        self.image = sl.Mat()
        self.point_cloud = sl.Mat()
        self.runtime = sl.RuntimeParameters()

        # Camera-to-robot transform — set by QR calibration
        self.T_cam_to_robot = np.eye(4)

    def grab(self):
        """Get RGB image and point cloud."""
        if self.zed.grab(self.runtime) != sl.ERROR_CODE.SUCCESS:
            return None, None
        self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
        self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)
        bgr = self.image.get_data()[:, :, :3]
        return bgr, self.point_cloud

    def pixel_to_robot_xyz(self, px, py):
        """Convert pixel (px, py) to robot base frame XYZ using depth."""
        err, pt = self.point_cloud.get_value(int(px), int(py))
        if not np.isfinite(pt[0]):
            return None
        cam = np.array([pt[0], pt[1], pt[2], 1.0])
        robot = self.T_cam_to_robot @ cam
        return robot[:3]

    def detect_by_hsv(self, bgr, color_name, hsv_ranges):
        """Detect objects of a specific color using HSV."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in hsv_ranges:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5)))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            xyz = self.pixel_to_robot_xyz(cx, cy)
            detections.append({
                "color": color_name,
                "center_px": (cx, cy),
                "position_3d": xyz,
                "area": area,
            })
        return detections

    def load_calibration(self, filepath):
        """Load camera-to-robot transform from file."""
        self.T_cam_to_robot = np.load(filepath)
        print(f"Loaded calibration from {filepath}")

    def close(self):
        self.zed.close()


# --- HSV CONFIG ---
# These are YOUR tuned values from the slider tool. Update as needed.
HSV_CONFIG = {
    "red": [((119, 41, 0), (180, 255, 255))],
    "blue": [((97, 64, 13), (131, 255, 255))],
    "green": [((41, 60, 0), (75, 255, 255))],
    "yellow": [((18, 97, 190), (47, 255, 255))],
}

