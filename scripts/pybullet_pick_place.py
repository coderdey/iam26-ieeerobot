"""
SO-101 Pick and Place — Simulation + Physical Arm
Based on pybulletintro.py (working sim) + so100_classical_pick_place.py (physical arm)

Usage:
  python pybullet_pick_place.py              # sim only (default)
  python pybullet_pick_place.py --real       # sim + send to physical arm
  python pybullet_pick_place.py -i           # interactive: type XYZ, watch arm move
  python pybullet_pick_place.py -i --real    # interactive + physical arm follows
"""
import pybullet as p
import pybullet_data
import time
import numpy as np
import matplotlib.pyplot as plt
import sys

# =============================================================================
# CONFIG — UPDATE THESE FOR YOUR SETUP
# =============================================================================
URDF_PATH = "C:/Users/mithi/lerobot/sim/so101_new_calib.urdf"
SERIAL_PORT = "/dev/ttyACM1"   # Linux desktop. Windows: "COM3"
BAUDRATE = 1_000_000

edge_offset = 0.05
arm_spacing = 0.5
table_depth = 0.8
table_width = 1.6
spacing_mid = arm_spacing / 2

ee_index = 5
grip_index = 6

APPROACH_HEIGHT = 0.15
GRASP_HEIGHT = 0.025
LIFT_HEIGHT = 0.18
DROP_HEIGHT = 0.06

SERVO_IDS = [1, 2, 3, 4, 5]
GRIPPER_ID = 6
SERVO_CENTER = 2048
COUNTS_PER_DEG = 4096 / 360.0
GRIPPER_OPEN = 1400
GRIPPER_CLOSE = 4000


# =============================================================================
# PHYSICAL ARM — sends joint angles from sim to real servos
# =============================================================================

class PhysicalArm:
    def __init__(self, port=SERIAL_PORT):
        from scservo_sdk import PortHandler, PacketHandler
        self.ph = PortHandler(port)
        self.pk = PacketHandler(0)
        if not self.ph.openPort():
            raise RuntimeError(f"Cannot open {port}")
        self.ph.setBaudRate(BAUDRATE)
        print(f"Physical arm connected on {port}")

    def send_angles_rad(self, angles_rad, speed=300):
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
        print("Physical arm disconnected")


# =============================================================================
# HELPER: solve IK in sim and optionally send to physical arm
# =============================================================================

GRIPPER_DOWN_ORN = None

def get_gripper_down_orn():
    global GRIPPER_DOWN_ORN
    if GRIPPER_DOWN_ORN is None:
        GRIPPER_DOWN_ORN = p.getQuaternionFromEuler([np.pi, 0, 0])
    return GRIPPER_DOWN_ORN

def solve_and_move(robot_id, target, movable_joints, real_arm=None, speed=300, label=""):
    joint_angles = p.calculateInverseKinematics(
        robot_id, ee_index, target,
        targetOrientation=get_gripper_down_orn(),
        maxNumIterations=500, residualThreshold=1e-5)

    for i, joint_idx in enumerate(movable_joints):
        p.setJointMotorControl2(robot_id, joint_idx, p.POSITION_CONTROL,
            targetPosition=joint_angles[i], force=50, maxVelocity=2.0)

    for _ in range(200):
        p.stepSimulation()
        time.sleep(1./240.)

    actual = p.getLinkState(robot_id, ee_index)[0]
    error = sum((a-b)**2 for a,b in zip(target, actual))**0.5
    degs = [f"{np.degrees(a):7.1f}" for a in joint_angles[:5]]
    print(f"  {label:20s} target={[round(t,3) for t in target]} error={error:.3f}m angles=[{', '.join(degs)}]")

    if real_arm:
        real_arm.send_angles_rad(joint_angles, speed)
        time.sleep(1.0)

    return joint_angles


def pixel_to_world(px, py, depth_buffer, w, h, view_mat, proj_mat):
    depth = depth_buffer[py][px]
    x_ndc = (2.0 * px) / w - 1.0
    y_ndc = 1.0 - (2.0 * py) / h
    view = np.array(view_mat).reshape(4, 4, order='F')
    proj = np.array(proj_mat).reshape(4, 4, order='F')
    tran_pix_world = np.linalg.inv(np.matmul(proj, view))
    z_ndc = 2.0 * depth - 1.0
    pixel_pos = np.array([x_ndc, y_ndc, z_ndc, 1.0])
    world_pos = np.matmul(tran_pix_world, pixel_pos)
    world_pos /= world_pos[3]
    return world_pos[:3]


# =============================================================================
# INTERACTIVE MODE: type XYZ, watch sim arm move (+ physical if --real)
# =============================================================================

def run_interactive():
    print("\n=== INTERACTIVE IK TESTER ===")
    print("Type 'x y z' in meters (e.g. '0.15 0.10 0.10')")
    print("Type 'home' to go to home position")
    print("Type 'open' or 'close' for gripper (physical arm only)")
    print("Type 'q' to quit\n")

    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF(URDF_PATH, basePosition=[edge_offset, 0, 0], useFixedBase=True)
    movable_joints = [i for i in range(p.getNumJoints(robot_id))
        if p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED]

    real_arm = None
    if "--real" in sys.argv:
        real_arm = PhysicalArm()
        real_arm.home()

    while True:
        val = input("\nXYZ > ").strip()
        if val.lower() == 'q':
            break
        if val.lower() == 'home':
            if real_arm:
                real_arm.home()
            print("  Home position sent")
            continue
        if val.lower() == 'open':
            if real_arm:
                real_arm.set_gripper("open")
            print("  Gripper opened")
            continue
        if val.lower() == 'close':
            if real_arm:
                real_arm.set_gripper("close")
            print("  Gripper closed")
            continue
        try:
            parts = val.replace(',', ' ').split()
            target = [float(parts[0]), float(parts[1]), float(parts[2])]
            solve_and_move(robot_id, target, movable_joints, real_arm, label="manual")
            actual = p.getLinkState(robot_id, ee_index)
            orn_euler = p.getEulerFromQuaternion(actual[1])
            orn_deg = [np.degrees(o) for o in orn_euler]
            print(f"  Gripper orientation (deg): [{orn_deg[0]:.0f}, {orn_deg[1]:.0f}, {orn_deg[2]:.0f}]")
        except (ValueError, IndexError):
            print("  Format: x y z (e.g. '0.15 0.10 0.10')")
          
    if real_arm:
        real_arm.home()
        real_arm.disconnect()
    p.disconnect()


# =============================================================================
# MAIN — full pick-and-place sequence
# =============================================================================

if "-i" in sys.argv or "--interactive" in sys.argv:
    run_interactive()
    sys.exit(0)

USE_REAL = "--real" in sys.argv

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)

# 1. Workspace setup
zed_x = table_depth / 2
zed_y = spacing_mid
zed_height = 1.0
fov = 90
zed_target = [zed_x, zed_y, 0]

plane_id = p.loadURDF("plane.urdf")
robot_id_1 = p.loadURDF(URDF_PATH, basePosition=[edge_offset, 0, 0], useFixedBase=True)
robot_id_2 = p.loadURDF(URDF_PATH, basePosition=[edge_offset, arm_spacing, 0], useFixedBase=True)

movable_joints = [i for i in range(p.getNumJoints(robot_id_1))
    if p.getJointInfo(robot_id_1, i)[2] != p.JOINT_FIXED]

print(f"Movable joints: {len(movable_joints)}")
for i in movable_joints:
    info = p.getJointInfo(robot_id_1, i)
    print(f"  Joint {i}: {info[1].decode()} limits=[{info[8]:.2f}, {info[9]:.2f}]")
'''
# 2. Reachability test
test_points = [
    [0.15, 0.05, 0.1],
    [0.5, 0.05, 0.1],
    [0.35, 0.25, 0.1],
    [0.35, -0.25, 0.1],
    [0.35, 0.05, 0.3]]

print("\n--- REACHABILITY TEST ---")
for target in test_points:
    joint_angles = p.calculateInverseKinematics(robot_id_1, ee_index, target,
        targetOrientation=get_gripper_down_orn(),
        maxNumIterations=500, residualThreshold=1e-5)
    for i, joint_idx in enumerate(movable_joints):
        p.resetJointState(robot_id_1, joint_idx, joint_angles[i])
    for _ in range(50):
        p.stepSimulation()
    actual_pos = p.getLinkState(robot_id_1, ee_index)[0]
    error = sum((a-b)**2 for a,b in zip(target, actual_pos))**0.5
    status = "OK" if error < 0.02 else "OUT OF REACH"
    print(f"  target {target} -> error = {error:.3f}m  {status}")
'''
# 3. ZED camera setup
view_matrix = p.computeViewMatrix(
    cameraEyePosition=[zed_x, zed_y, zed_height],
    cameraTargetPosition=zed_target, cameraUpVector=[1, 0, 0])
projection_matrix = p.computeProjectionMatrixFOV(
    fov=fov, aspect=1.0, nearVal=0.1, farVal=2.0)

table_visual = p.createVisualShape(p.GEOM_BOX,
    halfExtents=[table_depth/2, table_width/2, 0.001], rgbaColor=[1, 0, 0, 0.3])
p.createMultiBody(baseMass=0, baseVisualShapeIndex=table_visual, basePosition=[zed_x, zed_y, 0])

# 4. Object + bin
object_start_pos = [table_depth * 0.5, spacing_mid + 0.1, 0.02]
obj_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02], rgbaColor=[0, 1, 0, 1])
obj_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
object_id = p.createMultiBody(baseMass=0.05, baseCollisionShapeIndex=obj_collision,
    baseVisualShapeIndex=obj_visual, basePosition=object_start_pos)

bin_pos = [0.25, -0.10, 0.02]
bin_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.03], rgbaColor=[1, 0, 0, 0.5])
p.createMultiBody(baseMass=0, baseVisualShapeIndex=bin_visual, basePosition=bin_pos)

# 5. Camera detect object
width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
    width=640, height=640, viewMatrix=view_matrix, projectionMatrix=projection_matrix)
rgb_array = np.reshape(rgb_img, (height, width, 4))[:, :, :3]
depth_array = np.reshape(depth_img, (height, width))
green_mask = (rgb_array[:, :, 1] > 150) & (rgb_array[:, :, 0] < 100) & (rgb_array[:, :, 2] < 100)
ys, xs = np.where(green_mask)

if len(xs) == 0:
    print("Object not visible in frame! Using known position.")
    detected_pos = object_start_pos
else:
    pixel_x = int(np.mean(xs))
    pixel_y = int(np.mean(ys))
    print(f"Object found at pixel: ({pixel_x}, {pixel_y})")
    detected_pos = pixel_to_world(pixel_x, pixel_y, depth_array, width, height,
        view_matrix, projection_matrix)
    actual_pos = p.getBasePositionAndOrientation(object_id)[0]
    print(f"Detected world position: {np.round(detected_pos, 3)}")
    print(f"Actual object position:  {np.round(actual_pos, 3)}")

plt.imshow(rgb_array)
plt.title("Overhead Camera View")
plt.savefig("camera_view.png")
plt.show(block=False)
plt.pause(1)
plt.close()

# =============================================================================
# 6. PICK AND PLACE SEQUENCE — sim + optional physical arm
# =============================================================================

print(f"\n--- PICK AND PLACE ---")
print(f"Object at: {np.round(detected_pos, 3)}")
print(f"Bin at:    {bin_pos}")

real_arm = None
if USE_REAL:
    real_arm = PhysicalArm()
    real_arm.home()
    input("Physical arm ready. Press Enter to start (Ctrl+C to abort)...")

# PICK
print("\nPICK:")
solve_and_move(robot_id_1, [detected_pos[0], detected_pos[1], APPROACH_HEIGHT],
    movable_joints, real_arm, speed=100, label="above object")
if real_arm:
    real_arm.set_gripper("open")
    time.sleep(0.5)

solve_and_move(robot_id_1, [detected_pos[0], detected_pos[1], GRASP_HEIGHT],
    movable_joints, real_arm, speed=50, label="grasp position")
if real_arm:
    real_arm.set_gripper("close")
    time.sleep(0.8)

solve_and_move(robot_id_1, [detected_pos[0], detected_pos[1], LIFT_HEIGHT],
    movable_joints, real_arm, speed=100, label="lift")

# PLACE
print("\nPLACE:")
solve_and_move(robot_id_1, [bin_pos[0], bin_pos[1], LIFT_HEIGHT],
    movable_joints, real_arm, speed=100, label="above bin")

solve_and_move(robot_id_1, [bin_pos[0], bin_pos[1], DROP_HEIGHT],
    movable_joints, real_arm, speed=50, label="drop position")
if real_arm:
    real_arm.set_gripper("open")
    time.sleep(0.5)

solve_and_move(robot_id_1, [bin_pos[0], bin_pos[1], LIFT_HEIGHT],
    movable_joints, real_arm, speed=100, label="retreat")

print("\n--- DONE ---")
if real_arm:
    real_arm.home()
    real_arm.disconnect()

print("Simulation running. Close window or Ctrl+C to exit.")
try:
    while True:
        p.stepSimulation()
        time.sleep(1./240.)
except KeyboardInterrupt:
    pass
p.disconnect()
