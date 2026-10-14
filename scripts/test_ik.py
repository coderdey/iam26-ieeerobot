"""Test IK using URDF. Run on desktop to verify arm reaches target positions."""

import ikpy.chain
import numpy as np

# Load chain from URDF — no manual measurements needed
chain = ikpy.chain.Chain.from_urdf_file(
    "../urdf/so100.urdf",
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
