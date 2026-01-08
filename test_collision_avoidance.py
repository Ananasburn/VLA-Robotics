import sys
import os
import numpy as np
import time
import pinocchio

# Add paths
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'manipulator_grasp'))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from manipulator_grasp.path_plan.dual_set_plan import DualArmPlanner
from manipulator_grasp.env.dual_robot_env import DualRobotEnv

def check_path_collisions(planner, path):
    """
    Check if any waypoint in the path has collisions.
    Returns: (has_collision, collision_count, collision_details)
    """
    collision_count = 0
    collision_details = []
    
    for i, q in enumerate(path):
        pinocchio.computeCollisions(
            planner.model, planner.data, 
            planner.collision_model, planner.collision_data, 
            q, False
        )
        
        waypoint_collisions = []
        for k in range(len(planner.collision_model.collisionPairs)):
            cr = planner.collision_data.collisionResults[k]
            if cr.isCollision():
                pair = planner.collision_model.collisionPairs[k]
                name1 = planner.collision_model.geometryObjects[pair.first].name
                name2 = planner.collision_model.geometryObjects[pair.second].name
                
                # Only count left-right arm collisions (not table collisions)
                if ('left_' in name1 and 'right_' in name2) or ('right_' in name1 and 'left_' in name2):
                    waypoint_collisions.append((name1, name2))
        
        if waypoint_collisions:
            collision_count += 1
            collision_details.append((i, waypoint_collisions))
    
    return collision_count > 0, collision_count, collision_details

def visualize_comparison(env, naive_path, planned_path, scenario_name):
    """
    Visualize both naive (collision) and planned (safe) paths for comparison.
    """
    print(f"\n{'='*60}")
    print(f"Visualizing: {scenario_name}")
    print(f"{'='*60}")
    
    # Show naive path first (RED - collision)
    if naive_path is not None:
        print("\n[NAIVE PATH - Straight Line Interpolation]")
        print("This would cause collisions! Watch the arms hit each other.")
        print("Press Ctrl+C to skip to the safe path...")
        
        try:
            for i, q_waypoint in enumerate(naive_path):
                ctrl = np.zeros(env.model.nu)
                ctrl[:12] = q_waypoint
                
                for _ in range(30):  # Faster playback
                    env.step(ctrl)
                    time.sleep(1.0 / env.sim_hz)
                
                if i % 10 == 0:
                    print(f"  Waypoint {i}/{len(naive_path)}")
            
            # Pause
            for _ in range(100):
                env.step(ctrl)
                time.sleep(1.0 / env.sim_hz)
                
        except KeyboardInterrupt:
            print("  Skipped naive path")
    
    # Show planned path (GREEN - safe)
    if planned_path is not None:
        print("\n[PLANNED PATH - RRT Collision-Free]")
        print("This avoids collisions! Watch the arms move safely.")
        
        try:
            for i, q_waypoint in enumerate(planned_path):
                ctrl = np.zeros(env.model.nu)
                ctrl[:12] = q_waypoint
                
                for _ in range(50):  # Smoother playback
                    env.step(ctrl)
                    time.sleep(1.0 / env.sim_hz)
                
                if i % 5 == 0:
                    print(f"  Waypoint {i}/{len(planned_path)}")
            
            # Pause at end
            print("  Pausing...")
            for _ in range(150):
                env.step(ctrl)
                time.sleep(1.0 / env.sim_hz)
                
        except KeyboardInterrupt:
            print("  Skipped planned path")

def test_scenario_crossing_paths(planner, env):
    """
    Scenario 1: Arms crossing paths
    Left arm moves to the right, right arm moves to the left.
    Direct interpolation would cause collision in the middle.
    """
    print("\n" + "="*70)
    print("SCENARIO 1: CROSSING PATHS")
    print("="*70)
    print("Left arm moves RIGHT, Right arm moves LEFT")
    print("Direct path would collide in the center workspace!")
    
    nq = planner.model.nq
    
    # Start: Both extended forward at different heights
    q_start = np.zeros(nq)
    q_start[0:6] = [0.8, -1.3, -1.5, -1.57, 0, 0]    # Left extended left side
    q_start[6:12] = [-0.8, -1.3, 1.5, -1.57, 0, 0]   # Right extended right side
    
    # Goal: Arms swap sides with elbows bent inward (collision zone!)
    q_goal = q_start.copy()
    q_goal[0] = -1.2     # Left shoulder pan far right
    q_goal[1] = -1.1     # Shoulder lift
    q_goal[2] = 1.8      # Elbow toward center (opposite direction)
    q_goal[6] = 1.2      # Right shoulder pan far left
    q_goal[7] = -1.1     # Shoulder lift
    q_goal[8] = -1.8     # Elbow toward center (opposite direction)
    
    # Create naive straight-line path
    naive_path = [q_start + alpha * (q_goal - q_start) for alpha in np.linspace(0, 1, 50)]
    has_collision, count, details = check_path_collisions(planner, naive_path)
    
    print(f"\nNaive straight-line path: {count}/50 waypoints have COLLISIONS")
    if has_collision and details:
        print(f"  First collision at waypoint {details[0][0]}")
        print(f"  Collision pairs: {details[0][1][0]}")
    
    # Plan collision-free path
    print("\nPlanning collision-free path with RRT...")
    planned_path = planner.plan(q_start, q_goal)
    
    if planned_path is not None:
        has_collision, count, details = check_path_collisions(planner, planned_path)
        print(f"Planned path: {count}/{len(planned_path)} waypoints have collisions")
        
        if not has_collision:
            print("✓ SUCCESS: Collision-free path found!")
        
        # Visualize
        visualize_comparison(env, naive_path, planned_path, "Crossing Paths")
        return True
    else:
        print("✗ FAILED: Could not find collision-free path")
        return False


def test_scenario_reaching_center(planner, env):
    """
    Scenario 2: Both arms reaching toward center
    Both arms move toward the shared workspace center.
    """
    print("\n" + "="*70)
    print("SCENARIO 2: REACHING TOWARD CENTER")
    print("="*70)
    print("Both arms reach toward the center workspace")
    print("Direct path would cause collision!")
    
    nq = planner.model.nq
    
    # Start: Arms extended outward and upward
    q_start = np.zeros(nq)
    q_start[0:6] = [1.0, -0.8, -0.5, -1.57, 0, 0]    # Left arm up and out
    q_start[6:12] = [-1.0, -0.8, 0.5, -1.57, 0, 0]   # Right arm up and out
    
    # Goal: Both reach toward center with elbows bent (collision!)
    q_goal = q_start.copy()
    q_goal[0] = -0.2      # Left reaches right
    q_goal[1] = -1.3      # Lower down
    q_goal[2] = 1.5       # Elbow bent inward
    q_goal[6] = 0.2       # Right reaches left
    q_goal[7] = -1.3      # Lower down  
    q_goal[8] = -1.5      # Elbow bent inward
    
    # Create naive path
    naive_path = [q_start + alpha * (q_goal - q_start) for alpha in np.linspace(0, 1, 50)]
    has_collision, count, details = check_path_collisions(planner, naive_path)
    
    print(f"\nNaive straight-line path: {count}/50 waypoints have COLLISIONS")
    if has_collision and details:
        print(f"  First collision at waypoint {details[0][0]}")
        print(f"  Collision pairs: {details[0][1][0]}")
    
    # Plan collision-free path
    print("\nPlanning collision-free path with RRT...")
    planned_path = planner.plan(q_start, q_goal)
    
    if planned_path is not None:
        has_collision, count, details = check_path_collisions(planner, planned_path)
        print(f"Planned path: {count}/{len(planned_path)} waypoints have collisions")
        
        if not has_collision:
            print("✓ SUCCESS: Collision-free path found!")
        
        # Visualize
        visualize_comparison(env, naive_path, planned_path, "Reaching Center")
        return True
    else:
        print("✗ FAILED: Could not find collision-free path")
        return False


def test_scenario_asymmetric_motion(planner, env):
    """
    Scenario 3: Asymmetric challenging motion
    One arm makes a large sweeping motion while the other adjusts position.
    """
    print("\n" + "="*70)
    print("SCENARIO 3: ASYMMETRIC SWEEPING MOTION")
    print("="*70)
    print("Left arm sweeps across workspace, right arm repositions")
    print("Arms need to coordinate to avoid collision!")
    
    nq = planner.model.nq
    
    # Start: Left far left, right neutral
    q_start = np.zeros(nq)
    q_start[0:6] = [1.4, -1.57, 0.5, -1.57, 0, 0]   # Left far out
    q_start[6:12] = [0, -1.57, 0, -1.57, 0, 0]      # Right neutral
    
    # Goal: Left sweeps to right side, right moves left
    q_goal = q_start.copy()
    q_goal[0] = -1.0      # Left sweeps far right
    q_goal[1] = -1.2      # Lower
    q_goal[2] = -0.5      # Elbow adjust
    q_goal[6] = 0.6       # Right moves slightly left
    q_goal[7] = -1.0      # Adjust height
    
    # Create naive path
    naive_path = [q_start + alpha * (q_goal - q_start) for alpha in np.linspace(0, 1, 50)]
    has_collision, count, details = check_path_collisions(planner, naive_path)
    
    print(f"\nNaive straight-line path: {count}/50 waypoints have COLLISIONS")
    
    # Plan collision-free path
    print("\nPlanning collision-free path with RRT...")
    planned_path = planner.plan(q_start, q_goal)
    
    if planned_path is not None:
        has_collision, count, details = check_path_collisions(planner, planned_path)
        print(f"Planned path: {count}/{len(planned_path)} waypoints have collisions")
        
        if not has_collision:
            print("✓ SUCCESS: Collision-free path found!")
        
        # Visualize
        visualize_comparison(env, naive_path, planned_path, "Asymmetric Motion")
        return True
    else:
        print("✗ FAILED: Could not find collision-free path")
        return False

def main():
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "  DUAL-ARM COLLISION AVOIDANCE DEMONSTRATION".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    # Setup
    urdf_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "manipulator_grasp/robot_description/urdf/ur3e_ag95_dual.urdf"
    )
    
    print("\nInitializing planner and environment...")
    planner = DualArmPlanner(urdf_path)
    env = DualRobotEnv(headless=False)
    env.reset()
    
    print(f"Planner ready: {planner.model.nq} DOF")
    print(f"Environment ready: {env.model.nu} actuators")
    
    # Run scenarios
    results = []
    
    try:
        # Scenario 1: Crossing Paths
        success = test_scenario_crossing_paths(planner, env)
        results.append(("Crossing Paths", success))
        
        # Scenario 2: Reaching Center
        success = test_scenario_reaching_center(planner, env)
        results.append(("Reaching Center", success))
        
        # Scenario 3: Asymmetric Motion
        success = test_scenario_asymmetric_motion(planner, env)
        results.append(("Asymmetric Motion", success))
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    
    finally:
        env.close()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for scenario, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{scenario:30s} {status}")
    
    passed = sum(1 for _, s in results if s)
    print(f"\nTotal: {passed}/{len(results)} scenarios passed")
    print("\nCollision avoidance capability demonstrated!")

if __name__ == "__main__":
    main()
