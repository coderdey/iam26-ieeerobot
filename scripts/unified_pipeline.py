"""
Unified Pick-and-Place Pipeline
================================
ZED 2 (HSV detection + depth) → IK (approach object) → ACT policy (grasp + place)

This is the competition-ready pipeline. Classical perception decides WHAT to pick
and WHERE to place. IK moves the arm close. ACT policy does the fine manipulation.

Usage:
  python unified_pipeline.py                    # run color sort
  python unified_pipeline.py --task shape       # run shape sort
  python unified_pipeline.py --test-zed         # test ZED perception only
  python unified_pipeline.py --test-ik          # test IK only (no camera)
  python unified_pipeline.py --test-act         # test ACT policy only

Requirements:
  pip install pyzed2 pybullet numpy opencv-python torch
  LeRobot installed with feetech SDK
"""

import numpy as np
import cv2
import time
import sys
import pybullet as p
import pybullet_data
import os

# =============================================================================
# CONFIG
# =============================================================================

URDF_PATH = os.path.expanduser("~/.cache/huggingface/lerobot/lerobot/SO-ARM100/Simulation/SO101/so101_new_calib.urdf")
SERIAL_PORT = "/dev/ttyACM1"
BAUDRATE = 1_000_000

SERVO_IDS = [1, 2, 3, 4, 5]
GRIPPER_ID = 6
SERVO_CENTER = 2048
COUNTS_PER_DEG = 4096 / 360.0
GRIPPER_OPEN = 2500
GRIPPER_CLOSE = 1400

EE_INDEX = 5
EDGE_OFFSET = 0.05

APPROACH_HEIGHT = 0.15
GRASP_HEIGHT = 0.025
LIFT_HEIGHT = 0.18
DROP_HEIGHT = 0.06

# ACT policy checkpoint path — UPDATE after training
ACT_CHECKPOINT = "/home/cris/.cache/huggingface/lerobot/outputs/train/act_coderdey_my_dataset_20261020_191616_2026-10-20_20-19-55/run/checkpoints/last/pretrained_model"

# Camera-to-robot transform — loaded from QR calibration or set manually
CAM_TO_ROBOT = np.eye(4)  # UPDATE after calibration

# HSV color ranges — your tuned values
HSV_CONFIG = {
    "red": {
        "hsv_ranges": [((119, 41, 0), (180, 255, 255))],
        "bin_color": "red",
    },
    "blue": {
        "hsv_ranges": [((97, 64, 13), (131, 255, 255))],
        "bin_color": "blue",
    },
    "green": {
        "hsv_ranges": [((41, 60, 0), (75, 255, 255))],
        "bin_color": "green",
    },
    "yellow": {
        "hsv_ranges": [((18, 97, 190), (47, 255, 255))],
        "bin_color": "yellow",
    },
}

# Shape-to-bin mapping for shape sort (rule 5.8.1.40)
SHAPE_TO_BIN = {
    "cube": "red",
    "cylinder": "blue",
    "triangular_prism": "green",
    "hexagonal_prism": "yellow",
}


# =============================================================================
# ZED 2 PERCEPTION
# =============================================================================

class ZED2:
    def __init__(self):
        import pyzed.sl as sl
        self.sl = sl
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
        self.T_cam_to_robot = CAM_TO_ROBOT

    def grab(self):
        if self.zed.grab(self.runtime) != self.sl.ERROR_CODE.SUCCESS:
            return None, None
        self.zed.retrieve_image(self.image, self.sl.VIEW.LEFT)
        self.zed.retrieve_measure(self.point_cloud, self.sl.MEASURE.XYZRGBA)
        bgr = self.image.get_data()[:, :, :3]
        return bgr, self.point_cloud

    def pixel_to_robot(self, px, py):
        err, pt = self.point_cloud.get_value(int(px), int(py))
        if not np.isfinite(pt[0]):
            return None
        cam = np.array([pt[0], pt[1], pt[2], 1.0])
        robot = self.T_cam_to_robot @ cam
        return robot[:3]

    def detect_objects_by_color(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        detections = []
        for color_name, cfg in HSV_CONFIG.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in cfg["hsv_ranges"]:
                mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5)))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5)))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 500:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                detections.append({
                    "color": color_name, "center_px": (cx, cy),
                    "area": area, "contour": cnt,
                })
        return detections

    def detect_bins_by_color(self, bgr):
        detections = self.detect_objects_by_color(bgr)
        bins = {}
        for d in detections:
            if d["area"] > 3000:
                xyz = self.pixel_to_robot(d["center_px"][0], d["center_px"][1])
                if xyz is not None:
                    bins[d["color"]] = xyz
        return bins

    def load_calibration(self, path):
        self.T_cam_to_robot = np.load(path)

    def close(self):
        self.zed.close()


# =============================================================================
# IK ENGINE (PyBullet headless)
# =============================================================================

class IKEngine:
    def __init__(self, urdf_path=URDF_PATH):
        self.physics = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(urdf_path, basePosition=[EDGE_OFFSET, 0, 0], useFixedBase=True)
        self.gripper_down = p.getQuaternionFromEuler([np.pi, 0, 0])
        self.movable_joints = [
            i for i in range(p.getNumJoints(self.robot_id))
            if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED
        ]

    def solve(self, target_xyz):
        angles = p.calculateInverseKinematics(
            self.robot_id, EE_INDEX, target_xyz,
            targetOrientation=self.gripper_down,
            maxNumIterations=500, residualThreshold=1e-5)
        for i, joint_idx in enumerate(self.movable_joints):
            p.resetJointState(self.robot_id, joint_idx, angles[i])
        actual = p.getLinkState(self.robot_id, EE_INDEX)[0]
        error = np.linalg.norm(np.array(actual) - np.array(target_xyz))
        return angles, error

    def close(self):
        p.disconnect(self.physics)


# =============================================================================
# PHYSICAL ARM
# =============================================================================

class Arm:
    def __init__(self, port=SERIAL_PORT):
        from scservo_sdk import PortHandler, PacketHandler
        self.ph = PortHandler(port)
        self.pk = PacketHandler(0)
        if not self.ph.openPort():
            raise RuntimeError(f"Cannot open {port}")
        self.ph.setBaudRate(BAUDRATE)

    def send_ik_angles(self, angles_rad, speed=300):
        for i, rad in enumerate(angles_rad[:5]):
            deg = np.degrees(rad)
            pos = int(SERVO_CENTER + deg * COUNTS_PER_DEG)
            pos = max(0, min(4095, pos))
            self.pk.write2ByteTxRx(self.ph, SERVO_IDS[i], 46, speed)
            self.pk.write2ByteTxRx(self.ph, SERVO_IDS[i], 42, pos)

    def set_gripper(self, state, speed=100):
        pos = GRIPPER_OPEN if state == "open" else GRIPPER_CLOSE
        self.pk.write2ByteTxRx(self.ph, GRIPPER_ID, 46, speed)
        self.pk.write2ByteTxRx(self.ph, GRIPPER_ID, 42, pos)

    def move_to(self, ik_engine, xyz, speed=300):
        angles, error = ik_engine.solve(xyz)
        if error > 0.05:
            print(f"  WARNING: IK error {error*1000:.0f}mm for {xyz}")
        self.send_ik_angles(angles, speed)
        return angles, error

    def home(self):
        home_deg = [0, -30, 60, -30, 0]
        for i, deg in enumerate(home_deg):
            pos = int(SERVO_CENTER + deg * COUNTS_PER_DEG)
            self.pk.write2ByteTxRx(self.ph, SERVO_IDS[i], 46, 100)
            self.pk.write2ByteTxRx(self.ph, SERVO_IDS[i], 42, pos)
        time.sleep(1.5)
        self.set_gripper("open")

    def disconnect(self):
        self.set_gripper("open")
        time.sleep(0.3)
        self.ph.closePort()


# =============================================================================
# ACT POLICY — loads trained checkpoint for grasp + place
# =============================================================================

class ACTPolicy:
    def __init__(self, checkpoint_path=ACT_CHECKPOINT, device="cuda"):
        self.policy = None
        self.device = device
        try:
            from lerobot.policies.act.modeling_act import ACTPolicy as _ACT
            from lerobot.policies.act.configuration_act import ACTConfig
            import torch

            self.policy = _ACT.from_pretrained(checkpoint_path)
            self.policy.to(device)
            self.policy.eval()
            print(f"ACT policy loaded from {checkpoint_path}")
        except Exception as e:
            print(f"WARNING: Could not load ACT policy: {e}")
            print("  Falling back to classical place (no learned grasping)")

    def is_loaded(self):
        return self.policy is not None

    def run_grasp_and_place(self, robot_config, duration_s=10):
        """
        Run the ACT policy for a fixed duration. The policy reads the wrist
        camera and outputs joint actions through LeRobot's interface.

        This takes over servo control from scservo_sdk — don't mix them.
        """
        if not self.is_loaded():
            print("  ACT not loaded, skipping learned phase")
            return False

        try:
            from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
            from lerobot.cameras.opencv import OpenCVCameraConfig
            import torch
            print({k: (v.shape, v.dtype) for k, v in obs.items()})
            config = SO100FollowerConfig(
                port=robot_config["port"],
                id=robot_config["id"],
                cameras={
                    "wrist": OpenCVCameraConfig(
                        index_or_path=robot_config["wrist_cam_index"],
                        width=640, height=480, fps=25
                    ),
                },
            )

            robot = SO100Follower(config)
            robot.connect()

            fps = 30
            start = time.time()
            step = 0

            while time.time() - start < duration_s:
                t0 = time.perf_counter()
                obs = robot.get_observation()

                obs_tensor = {}
                for key, val in obs.items():
                    if isinstance(val, np.ndarray):
                        obs_tensor[key] = torch.from_numpy(val).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    action = self.policy.select_action(obs_tensor)

                action_dict = {}
                action_np = action.squeeze(0).cpu().numpy()
                for i, name in enumerate(robot.action_features):
                    action_dict[name] = float(action_np[i])

                robot.send_action(action_dict)
                step += 1

                elapsed = time.perf_counter() - t0
                sleep_time = (1.0 / fps) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            robot.disconnect()
            print(f"  ACT ran {step} steps in {duration_s}s")
            return True

        except Exception as e:
            print(f"  ACT execution failed: {e}")
            return False


# =============================================================================
# CLASSICAL FALLBACK — if ACT not available, do pick+place classically
# =============================================================================

def classical_pick(arm, ik_engine, object_xyz):
    print("  Classical pick...")
    arm.set_gripper("open")
    time.sleep(0.3)
    arm.move_to(ik_engine, [object_xyz[0], object_xyz[1], APPROACH_HEIGHT], speed=300)
    time.sleep(0.8)
    arm.move_to(ik_engine, [object_xyz[0], object_xyz[1], GRASP_HEIGHT], speed=100)
    time.sleep(0.6)
    arm.set_gripper("close")
    time.sleep(0.5)
    arm.move_to(ik_engine, [object_xyz[0], object_xyz[1], LIFT_HEIGHT], speed=300)
    time.sleep(0.6)


def classical_place(arm, ik_engine, bin_xyz):
    print("  Classical place...")
    arm.move_to(ik_engine, [bin_xyz[0], bin_xyz[1], LIFT_HEIGHT], speed=300)
    time.sleep(0.8)
    arm.move_to(ik_engine, [bin_xyz[0], bin_xyz[1], DROP_HEIGHT], speed=100)
    time.sleep(0.5)
    arm.set_gripper("open")
    time.sleep(0.3)
    arm.move_to(ik_engine, [bin_xyz[0], bin_xyz[1], LIFT_HEIGHT], speed=300)
    time.sleep(0.4)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_color_sort(zed, ik_engine, arm, act_policy, robot_config):
    """Game 1 Round 1: sort objects by color into matching bins."""
    print("\n=== COLOR SORT ===\n")
    arm.home()
    time.sleep(1)

    while True:
        bgr, pc = zed.grab()
        if bgr is None:
            continue

        objects = zed.detect_objects_by_color(bgr)
        bins = zed.detect_bins_by_color(bgr)

        # Show detection
        vis = bgr.copy()
        for obj in objects:
            cv2.circle(vis, obj["center_px"], 8, (0, 255, 0), -1)
            cv2.putText(vis, obj["color"], (obj["center_px"][0]+10, obj["center_px"][1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Detection", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # Filter to small objects (not bins)
        small_objects = [o for o in objects if o["area"] < 3000]
        if not small_objects:
            continue

        target = max(small_objects, key=lambda o: o["area"])
        target_xyz = zed.pixel_to_robot(target["center_px"][0], target["center_px"][1])
        if target_xyz is None:
            print(f"  No depth for {target['color']} object, skipping")
            continue

        bin_color = target["color"]
        if bin_color not in bins:
            print(f"  Can't find {bin_color} bin, skipping")
            continue
        bin_xyz = bins[bin_color]

        print(f"\nPicking {target['color']} object at {np.round(target_xyz, 3)}")
        print(f"  → {bin_color} bin at {np.round(bin_xyz, 3)}")

        # PHASE 1: IK moves arm near object
        arm.move_to(ik_engine, [target_xyz[0], target_xyz[1], APPROACH_HEIGHT], speed=300)
        time.sleep(0.8)

        # PHASE 2: ACT policy takes over for grasp + place
        if act_policy.is_loaded():
            print("  Handing off to ACT policy for grasp + place...")
            # Disconnect scservo before ACT takes over LeRobot's interface
            arm.disconnect()
            act_policy.run_grasp_and_place(robot_config, duration_s=8)
            # Reconnect scservo after ACT is done
            arm.__init__(port=robot_config["port"])
        else:
            # Fallback: fully classical
            classical_pick(arm, ik_engine, target_xyz)
            classical_place(arm, ik_engine, bin_xyz)

        arm.home()
        time.sleep(0.5)

    cv2.destroyAllWindows()


def run_shape_sort(zed, ik_engine, arm, act_policy, robot_config):
    """Game 1 Round 2: sort objects by shape into mapped bins."""
    print("\n=== SHAPE SORT ===")
    print("NOTE: needs YOLO for shape detection. Using color fallback for now.\n")
    run_color_sort(zed, ik_engine, arm, act_policy, robot_config)


# =============================================================================
# TEST MODES
# =============================================================================

def test_zed():
    print("\n=== ZED 2 TEST ===")
    zed = ZED2()
    print("Press 's' to snapshot, 'd' to detect, 'q' to quit")
    i = 0
    while True:
        bgr, pc = zed.grab()
        if bgr is None:
            continue
        cv2.imshow("ZED 2", bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            cv2.imwrite(f"zed_snap_{i:04d}.png", bgr)
            print(f"Saved zed_snap_{i:04d}.png")
            i += 1
        elif key == ord('d'):
            objects = zed.detect_objects_by_color(bgr)
            bins = zed.detect_bins_by_color(bgr)
            print(f"  Found {len(objects)} objects, {len(bins)} bins")
            for o in objects:
                xyz = zed.pixel_to_robot(o["center_px"][0], o["center_px"][1])
                print(f"    {o['color']} at pixel {o['center_px']}, 3D={xyz}")
            for color, xyz in bins.items():
                print(f"    BIN {color} at 3D={np.round(xyz, 3)}")
        elif key == ord('q'):
            break
    zed.close()
    cv2.destroyAllWindows()


def test_ik():
    print("\n=== IK TEST ===")
    ik = IKEngine()
    arm = Arm()
    arm.home()
    print("Type 'x y z' or 'q' to quit, 'home' to home, 'open'/'close' for gripper")
    while True:
        val = input("XYZ > ").strip()
        if val == 'q':
            break
        if val == 'home':
            arm.home()
            continue
        if val == 'open':
            arm.set_gripper("open")
            continue
        if val == 'close':
            arm.set_gripper("close")
            continue
        try:
            parts = val.replace(',', ' ').split()
            target = [float(parts[0]), float(parts[1]), float(parts[2])]
            angles, error = arm.move_to(ik, target)
            print(f"  Error: {error*1000:.1f}mm  Angles: {[f'{np.degrees(a):.1f}' for a in angles[:5]]}")
        except (ValueError, IndexError):
            print("  Format: x y z")
    arm.home()
    arm.disconnect()
    ik.close()


def test_act():
    print("\n=== ACT POLICY TEST ===")
    act = ACTPolicy()
    if not act.is_loaded():
        print("Cannot test — policy failed to load")
        return
    robot_config = {
        "port": SERIAL_PORT,
        "id": "follower_left",
        "wrist_cam_index": 2,
    }
    print("Running ACT policy for 15 seconds...")
    #print([attr for attr in dir(act.policy.config) if not attr.startswith('_')])
    print(act.policy.config.input_features)
    print(act.policy.config.output_features)
    act.run_grasp_and_place(robot_config, duration_s=15)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    args = set(sys.argv[1:])

    if "--test-zed" in args:
        test_zed()
    elif "--test-ik" in args:
        test_ik()
    elif "--test-act" in args:
        test_act()
    else:
        task = "color"
        if "--task" in args:
            idx = sys.argv.index("--task")
            if idx + 1 < len(sys.argv):
                task = sys.argv[idx + 1]

        robot_config = {
            "port": SERIAL_PORT,
            "id": "follower_left",
            "wrist_cam_index": 2,
        }

        print("Initializing...")
        zed = ZED2()
        ik_engine = IKEngine()
        arm = Arm()
        act_policy = ACTPolicy()

        try:
            if task == "color":
                run_color_sort(zed, ik_engine, arm, act_policy, robot_config)
            elif task == "shape":
                run_shape_sort(zed, ik_engine, arm, act_policy, robot_config)
            else:
                print(f"Unknown task: {task}")
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            arm.home()
            arm.disconnect()
            ik_engine.close()
            zed.close()
            cv2.destroyAllWindows()
