import sys
import os
import numpy as np
import pinocchio

# Add paths
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'manipulator_grasp'))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from manipulator_grasp.path_plan.dual_set_plan import DualArmPlanner

def main():
    """Quick test to find collision-prone configurations."""
    
    urdf_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "manipulator_grasp/robot_description/urdf/ur3e_ag95_dual.urdf"
    )
    
    print("Initializing planner...")
    planner = DualArmPlanner(urdf_path)
    nq = planner.model.nq
    
    print(f"Model has {nq} DOF\n")
    
    # Test various configurations to find collisions
    test_configs = [
        {
            "name": "Both arms forward, elbows bent inward",
            "q": np.array([0, -1.2, 1.5, -1.5, 0, 0,  # Left
                          0, -1.2, -1.5, -1.5, 0, 0]) # Right
        },
        {
            "name": "Arms crossing at center, same height",
            "q": np.array([-0.3, -1.0, 1.0, -1.5, 0, 0,  # Left reaching right
                          0.3, -1.0, -1.0, -1.5, 0, 0])  # Right reaching left
        },
        {
            "name": "Both elbows pointing at each other",
            "q": np.array([0, -1.3, 1.8, -2.0, 0, 0,  # Left
                          0, -1.3, -1.8, -2.0, 0, 0]) # Right
        },
        {
            "name": "Extreme crossing", 
            "q": np.array([-0.8, -0.9, 2.0, -2.5, 0, 0,  # Left stretched right
                          0.8, -0.9, -2.0, -2.5, 0, 0])  # Right stretched left
        }
    ]
    
    for config in test_configs:
        print(f"Testing: {config['name']}")
        q = config['q']
        
        pinocchio.computeCollisions(
            planner.model, planner.data,
            planner.collision_model, planner.collision_data,
            q, False
        )
        
        collisions = []
        for k in range(len(planner.collision_model.collisionPairs)):
            cr = planner.collision_data.collisionResults[k]
            if cr.isCollision():
                pair = planner.collision_model.collisionPairs[k]
                name1 = planner.collision_model.geometryObjects[pair.first].name
                name2 = planner.collision_model.geometryObjects[pair.second].name
                
                # Only report left-right collisions
                if ('left_' in name1 and 'right_' in name2) or ('right_' in name1 and 'left_' in name2):
                    collisions.append((name1, name2))
        
        if collisions:
            print(f"  ✓ COLLISION FOUND! {len(collisions)} collision pairs:")
            for c in collisions[:3]:  # Show first 3
                print(f"    - {c[0]} <-> {c[1]}")
        else:
            print(f"  ✗ No collision")
        print()

if __name__ == "__main__":
    main()
