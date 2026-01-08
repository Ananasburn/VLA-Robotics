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

def visualize_path(env, path, scenario_name, repeat=1):
    """Visualize trajectory in MuJoCo."""
    print(f"\n{'='*60}")
    print(f"Visualizing: {scenario_name}")
    print(f"Path has {len(path)} waypoints")
    print(f"{'='*60}\n")
    
    try:
        for rep in range(repeat):
            if repeat > 1:
                print(f"Execution {rep + 1}/{repeat}")
            
            for i, q_waypoint in enumerate(path):
                ctrl = np.zeros(env.model.nu)
                ctrl[:12] = q_waypoint
                
                for _ in range(40):  # Smooth playback
                    env.step(ctrl)
                    time.sleep(1.0 / env.sim_hz)
                
                if i % 5 == 0:
                    print(f"  Waypoint {i}/{len(path)}")
            
            # Pause at end
            print("  Pausing...")
            for _ in range(150):
                env.step(ctrl)
                time.sleep(1.0 / env.sim_hz)
                
    except KeyboardInterrupt:
        print("  Skipped")

def demo_choreographed_motion(planner, env):
    """
    Demonstrate complex dual-arm choreography with collision awareness.
    The arms perform a coordinated dance, avoiding each other.
    """
    print("\n" + "="*70)
    print("DUAL-ARM COORDINATED CHOREOGRAPHY")
    print("="*70)
    print("This demo shows complex dual-arm motion planning where both arms")
    print("move through coordinated trajectories while avoiding collisions.")
    print("The planner ensures they never interfere with each other!\n")
    
    nq = planner.model.nq
    
    # Define a sequence of waypoints for a choreographed motion
    waypoints = []
    
    # Waypoint 1: Home position
    q1 = np.zeros(nq)
    q1[0:6] = [0, -1.57, 0, -1.57, 0, 0]
    q1[6:12] = [0, -1.57, 0, -1.57, 0, 0]
    waypoints.append(("Home Position", q1))
    
    # Waypoint 2: Both arms reach forward
    q2 = q1.copy()
    q2[1] = -1.2  # Left shoulder lift
    q2[2] = -0.5  # Left elbow
    q2[7] = -1.2  # Right shoulder lift
    q2[8] = 0.5   # Right elbow (opposite direction)
    waypoints.append(("Reaching Forward", q2))
    
    # Waypoint 3: Left sweeps right, Right goes up
    q3 = q2.copy()
    q3[0] = -0.5  # Left pans right
    q3[1] = -1.0  # Left higher
    q3[7] = -0.8  # Right much higher
    waypoints.append(("Left Sweeps Right, Right Rises", q3))
    
    # Waypoint 4: Mirror swap
    q4 = q3.copy()
    q4[0] = 0.5   # Left goes back left
    q4[1] = -1.4  # Left lowers
    q4[6] = -0.5  # Right pans left
    q4[7] = -1.3  # Right lowers
    waypoints.append(("Mirror Configuration", q4))
    
    # Waypoint 5: Both retract
    q5 = q1.copy()
    q5[1] = -1.8  # Left down
    q5[2] = 1. # Left elbow bent
    q5[7] = -1.8  # Right down
    q5[8] = -1.5  # Right elbow bent
    waypoints.append(("Retracted Position", q5))
    
    # Plan connection between consecutive waypoints
    full_path = []
    
    for i in range(len(waypoints) - 1):
        name_start, q_start = waypoints[i]
        name_goal, q_goal = waypoints[i + 1]
        
        print(f"\nPlanning segment {i+1}/{len(waypoints)-1}:")
        print(f"  From: {name_start}")
        print(f"  To: {name_goal}")
        
        path_segment = planner.plan(q_start, q_goal)
        
        if path_segment is None:
            print(f"  ✗ FAILED to plan this segment!")
            return False
        
        print(f"  ✓ Success! {len(path_segment)} waypoints")
        
        # Add to full path (avoid duplicating waypoints at joints)
        if len(full_path) == 0:
            full_path.extend(path_segment)
        else:
            full_path.extend(path_segment[1:])  # Skip first waypoint (duplicate)
    
    # Return to home
    print(f"\nPlanning return to home...")
    path_home = planner.plan(waypoints[-1][1], waypoints[0][1])
    if path_home:
        full_path.extend(path_home[1:])
        print(f"  ✓ Success! {len(path_home)} waypoints")
    
    print(f"\n{'='*70}")
    print(f"Complete choreography: {len(full_path)} total waypoints")
    print(f"{'='*70}")
    
    # Visualize the complete sequence
    visualize_path(env, full_path, "Complete Choreography", repeat=2)
    
    return True

def demo_independent_tasks(planner, env):
    """
    Demonstrate two arms working independently in shared workspace.
    Left arm picks from one location while right picks from another.
    """
    print("\n" + "="*70)
    print("INDEPENDENT TASK EXECUTION")
    print("="*70)
    print("Both arms execute independent tasks simultaneously.")
    print("They coordinate to avoid collisions in the shared workspace.\n")
    
    nq = planner.model.nq
    
    # Start: Both at home
    q_start = np.zeros(nq)
    q_start[0:6] = [0, -1.57, 0, -1.57, 0, 0]
    q_start[6:12] = [0, -1.57, 0, -1.57, 0, 0]
    
    # Goal: Left reaches forward-right, Right reaches forward-left
    # (simulating picking objects from shared table area)
    q_goal = q_start.copy()
    q_goal[0] = -0.3   # Left pan toward center
    q_goal[1] = -1.3   # Left reach down
    q_goal[2] = -0.8   # Left elbow
    q_goal[6] = 0.3    # Right pan toward center  
    q_goal[7] = -1.3   # Right reach down
    q_goal[8] = 0.8    # Right elbow
    
    print("Planning pick positions...")
    path = planner.plan(q_start, q_goal)
    
    if path is None:
        print("✗ Planning failed!")
        return False
    
    print(f"✓ Success! Planning found collision-free path with {len(path)} waypoints")
    
    # Visualize
    visualize_path(env, path, "Independent Task Execution", repeat=2)
    
    # Plan return
    print("\nPlanning return to home...")
    path_return = planner.plan(q_goal, q_start)
    if path_return:
        print(f"✓ Return path: {len(path_return)} waypoints")
        visualize_path(env, path_return, "Returning to Home", repeat=1)
    
    return True

def demo_sequential_handoff(planner, env):
    """
    Demonstrate a handoff-like scenario where arms take turns in shared space.
    """
    print("\n" + "="*70)
    print("SEQUENTIAL WORKSPACE SHARING")
    print("="*70)
    print("Arms take turns using the shared workspace.")
    print("One retracts while the other extends.\n")
    
    nq = planner.model.nq
    
    # Phase 1: Left extended, right retracted
    q1 = np.zeros(nq)
    q1[0:6] = [-0.2, -1.1, -1.0, -1.57, 0, 0]   # Left forward
    q1[6:12] = [0.3, -1.8, 0.5, -1.57, 0, 0]    # Right back
    
    # Phase 2: Right extended, left retracted
    q2 = np.zeros(nq)
    q2[0:6] = [0.3, -1.8, -0.5, -1.57, 0, 0]    # Left back
    q2[6:12] = [0.2, -1.1, 1.0, -1.57, 0, 0]    # Right forward
    
    # Phase 3: Both neutral
    q3 = np.zeros(nq)
    q3[0:6] = [0, -1.57, 0, -1.57, 0, 0]
    q3[6:12] = [0, -1.57, 0, -1.57, 0, 0]
    
    # Plan the sequence
    print("Planning Phase 1 → Phase 2 (handoff)...")
    path1 = planner.plan(q1, q2)
    
    if path1 is None:
        print("✗ Planning failed!")
        return False
    
    print(f"✓ Handoff path: {len(path1)} waypoints")
    visualize_path(env, path1, "Left to Right Handoff", repeat=2)
    
    print("\nPlanning Phase 2 → Neutral...")
    path2 = planner.plan(q2, q3)
    if path2:
        print(f"✓ Return path: {len(path2)} waypoints")
        visualize_path(env, path2, "Return to Neutral", repeat=1)
    
    return True

def main():
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "  DUAL-ARM COLLISION AVOIDANCE CAPABILITY DEMONSTRATION".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    print("\nThis demonstration showcases the dual-arm planner's ability to:")
    print("  1. Plan complex coordinated choreographies")
    print("  2. Handle independent tasks in shared workspace")
    print("  3. Enable sequential workspace sharing")
    print("\nAll while ensuring zero collisions between the arms!")
    
    # Setup
    urdf_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "manipulator_grasp/robot_description/urdf/ur3e_ag95_dual.urdf"
    )
    
    print("\nInitializing planner and environment...")
    planner = DualArmPlanner(urdf_path)
    env = DualRobotEnv(headless=False)
    env.reset()
    
    print(f"✓ Planner ready: {planner.model.nq} DOF")
    print(f"✓ Environment ready: {env.model.nu} actuators\n")
    print("="*70)
    
    # Run demonstrations
    results = []
    
    try:
        # Demo 1: Choreographed motion
        success = demo_choreographed_motion(planner, env)
        results.append(("Coordinated Choreography", success))
        
        input("\nPress Enter to continue to next demo...")
        
        # Demo 2: Independent tasks
        success = demo_independent_tasks(planner, env)
        results.append(("Independent Tasks", success))
        
        input("\nPress Enter to continue to final demo...")
        
        # Demo 3: Sequential handoff
        success = demo_sequential_handoff(planner, env)
        results.append(("Sequential Handoff", success))
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    
    finally:
        input("\nPress Enter to close...")
        env.close()
    
    # Summary
    print("\n" + "="*70)
    print("DEMONSTRATION SUMMARY")
    print("="*70)
    for demo, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{demo:35s} {status}")
    
    passed = sum(1 for _, s in results if s)
    print(f"\nTotal: {passed}/{len(results)} demonstrations successful")
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "  Collision avoidance capability demonstrated!".center(68) + "#")
    print("#" + "  The planner successfully coordinates dual-arm motion".center(68) + "#")
    print("#" + "  while respecting collision constraints.".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70 + "\n")

if __name__ == "__main__":
    main()
