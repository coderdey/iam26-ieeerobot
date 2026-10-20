"""Test IK using URDF. Run on desktop to verify arm reaches target positions."""

import ikpy.chain
import numpy as np

# Load chain from URDF — no manual measurements needed
chain = ikpy.chain.Chain.from_urdf_file(
    "/home/cris/.cache/huggingface/lerobot/lerobot/SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
    active_links_mask=[False, True, True, True, True, True, False]
)

# Test positions (x=forward, y=left, z=up, in meters from base)
test_points = [
    [0.10, 0.00, 0.10],   # front center, 10cm up
    [0.15, 0.00, 0.05],   # further forward, lower
    [0.10, 0.10, 0.05],   # front right
    [0.10, -0.10, 0.05],  # front left
]

for target in test_points:
    angles = chain.inverse_kinematics(target)
    fk = chain.forward_kinematics(angles)
    reached = fk[:3, 3]
    error_mm = np.linalg.norm(reached - target) * 1000

    print(f"Target: {target}")
    print(f"  Angles (deg): {np.round(np.degrees(angles[1:6]), 1)}")
    print(f"  Reached: {np.round(reached, 3)}")
    print(f"  Error: {error_mm:.1f} mm")
    print(f"  {'OK' if error_mm < 10 else 'BAD — check URDF'}")
    print()
    
    '''import ikpy.chain
import numpy as np

chain = ikpy.chain.Chain.from_urdf_file(
    "/home/cris/.cache/huggingface/lerobot/lerobot/SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
    active_links_mask=[False, True, True, True, True, True, False]
)

def solve_ik(target, tries=30):
    best, best_err = None, 1e9
    lo = [b[0] if b[0] is not None else -np.pi for b in (l.bounds for l in chain.links)]
    hi = [b[1] if b[1] is not None else np.pi for b in (l.bounds for l in chain.links)]
    rng = np.random.default_rng(0)
    for i in range(tries):
        if i == 0:
            init = np.zeros(len(chain.links))
        else:
            init = rng.uniform(lo, hi)
            init[0] = init[-1] = 0  # fixed links
        sol = chain.inverse_kinematics(target, initial_position=init)
        err = np.linalg.norm(chain.forward_kinematics(sol)[:3, 3] - target)
        if err < best_err:
            best, best_err = sol, err
        if err < 0.002:
            break
    return best, best_err * 1000
    
    chain = ikpy.chain.Chain.from_urdf_file(
    URDF,
    active_links_mask=[False, True, True, True, True, False, False]  # roll locked
)

angles = chain.inverse_kinematics(
    target,
    target_orientation=[0, 0, -1],
    orientation_mode="Z",
)'''
