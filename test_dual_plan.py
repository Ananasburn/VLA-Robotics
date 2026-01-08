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

def test_dual_planning():
    # ========== Visualization Parameters ==========
    VISUALIZE = True          # Set to False for headless testing
    NUM_REPETITIONS = 2       # How many times to repeat the trajectory
    STEPS_PER_WAYPOINT = 50   # Simulation steps per waypoint (higher = slower, smoother)
    REAL_TIME = True          # If True, runs at real-time speed; if False, runs as fast as possible
    # ==============================================
    
    # 1. Setup URDF path
    urdf_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "manipulator_grasp/robot_description/urdf/ur3e_ag95_dual.urdf"
    )
    
    # 2. Initialize Planner
    print("Initializing Planner...")
    planner = DualArmPlanner(urdf_path)
    
    # Check dimensions
    nq = planner.model.nq
    print(f"Planner Model nq: {nq}")
    print(f"Lower Limits: {planner.model.lowerPositionLimit}")
    print(f"Upper Limits: {planner.model.upperPositionLimit}")
    
    # 3. Define Start and Goal (Joint Space)
    # We need valid joint configurations.
    # UR3e Home: [0, -pi/2, 0, -pi/2, 0, 0] roughly.
    # 12 DoF for arms + 2 grippers? 
    # Pinocchio model from URDF will have joints. 
    # ur3e_ag95_dual.urdf has 2 arms. 
    # Let's inspect q vector size logic:
    # 2 x (6 arm joints + gripper joints). 
    # AG95 is often MIMO or simplified. 
    # Let's read nq at runtime. assuming first half is Left, second is Right.
    
    # Start: Neutral
    q_start = np.zeros(nq)
    # Left Arm
    q_start[0:6] = [0, -1.57, 0, -1.57, 0, 0] 
    # Right Arm
    q_start[6:12] = [0, -1.57, 0, -1.57, 0, 0] 
    
    # Goal: Raise both arms (More complex test)
    q_goal = q_start.copy()
    q_goal[1] = -0.8 # Left Lift up (from -1.57 to -0.8)
    q_goal[7] = -0.8 # Right Lift up (from -1.57 to -0.8)
    
    # Debug Collisions
    print("Checking collisions for q_start...")
    pinocchio.computeCollisions(planner.model, planner.data, planner.collision_model, planner.collision_data, q_start, False)
    for k in range(len(planner.collision_model.collisionPairs)):
        cr = planner.collision_data.collisionResults[k]
        if cr.isCollision():
            pair = planner.collision_model.collisionPairs[k]
            name1 = planner.collision_model.geometryObjects[pair.first].name
            name2 = planner.collision_model.geometryObjects[pair.second].name
            print(f"Collision detected: {name1} vs {name2}")

    print("Checking collisions for q_goal...")
    pinocchio.computeCollisions(planner.model, planner.data, planner.collision_model, planner.collision_data, q_goal, False)
    for k in range(len(planner.collision_model.collisionPairs)):
        cr = planner.collision_data.collisionResults[k]
        if cr.isCollision():
            pair = planner.collision_model.collisionPairs[k]
            name1 = planner.collision_model.geometryObjects[pair.first].name
            name2 = planner.collision_model.geometryObjects[pair.second].name
            print(f"Goal Collision detected: {name1} vs {name2}")

    # Manual Interpolation Check
    print("Checking interpolation...")
    t0 = time.time()
    steps = 100 # Increase steps for better average
    for i in range(steps + 1):
        alpha = i / steps
        q_interp = (1 - alpha) * q_start + alpha * q_goal
        
        pinocchio.computeCollisions(planner.model, planner.data, planner.collision_model, planner.collision_data, q_interp, False)
        for k in range(len(planner.collision_model.collisionPairs)):
            if planner.collision_data.collisionResults[k].isCollision():
                 break
    t1 = time.time()
    print(f"Interpolation (100 checks) took {t1-t0:.4f}s")
    print(f"Avg time per check: {(t1-t0)/100:.6f}s")

    # 4. Plan
    print(f"Starting RRT Plan at {time.strftime('%H:%M:%S')}")
    path = planner.plan(q_start, q_goal)
    print(f"Finished RRT Plan at {time.strftime('%H:%M:%S')}")
    
    if path is not None:
        print("Plan found! Visualizing...")
        
        # 5. Visualize in MuJoCo Env
        # Note: Env uses MuJoCo, Planner uses Pinocchio. 
        # Pinocchio q is 12-DOF (6 per arm after locking grippers)
        # MuJoCo has more actuators (12 arm joints + grippers)
        
        env = DualRobotEnv(headless=not VISUALIZE)
        env.reset()
        
        print(f"Env nq: {env.model.nq}, Env nu: {env.model.nu}, Planner nq: {nq}")
        print(f"Path length: {len(path)} waypoints")
        
        if VISUALIZE:
            # Execute the planned trajectory
            print("Executing trajectory in MuJoCo viewer...")
            print("Press Ctrl+C to stop early")
            
            try:
                # Repeat the trajectory for better visualization
                for repeat in range(NUM_REPETITIONS):
                    print(f"Execution {repeat + 1}/{NUM_REPETITIONS}")
                    
                    for i, q_waypoint in enumerate(path):
                        # Map Pinocchio q (12-DOF) to MuJoCo ctrl
                        # First 12 actuators are the arm joints (6 left + 6 right)
                        # Rest are gripper actuators (set to 0 or keep current)
                        ctrl = np.zeros(env.model.nu)
                        
                        # Set arm joint positions
                        ctrl[:12] = q_waypoint
                        
                        # Execute this waypoint for multiple steps to allow settling
                        for _ in range(STEPS_PER_WAYPOINT):
                            env.step(ctrl)
                            if REAL_TIME:
                                time.sleep(1.0 / env.sim_hz)  # Real-time visualization
                        
                        if i % 5 == 0:
                            print(f"  Waypoint {i}/{len(path)}")
                    
                    # Pause at end
                    print("  Pausing at end...")
                    for _ in range(100):
                        env.step(ctrl)
                        if REAL_TIME:
                            time.sleep(1.0 / env.sim_hz)
                
                print("\nVisualization complete!")
                print("Keeping viewer open for 5 more seconds...")
                for _ in range(int(5 * env.sim_hz)):
                    env.step(ctrl)
                    if REAL_TIME:
                        time.sleep(1.0 / env.sim_hz)
                    
            except KeyboardInterrupt:
                print("\nVisualization interrupted by user")
            
            finally:
                env.close()
                print("Success.")
        else:
            print("Headless mode - skipping visualization")
            env.close()
            print("Success.")
        
    else:
        print("Planning failed.")
        sys.exit(1)

if __name__ == "__main__":
    test_dual_planning()
