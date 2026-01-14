
import os
import sys
import numpy as np
import time

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from manipulator_grasp.rl_path_planner.rl_planner import RLPathPlanner

def main():
    print("🚀 Initializing RL Path Planner...")
    
    # Path to your successful model
    model_path = os.path.join(ROOT_DIR, "manipulator_grasp/rl_path_planner/models/task_space_v5_8_collision_check/best_model.zip")
    
    # Environment arguments (same as training)
    # Important: visualize=True to see the result!
    env_kwargs = {
        "max_steps": 200,
        "visualize": True, 
    }
    
    planner = RLPathPlanner(
        model_path=model_path,
        env_kwargs=env_kwargs
    )
    
    print("✅ Planner loaded successfully!")
    print("\n-----------------------------------")
    print("🧪 Test Case 1: Simple Reaching")
    
    # Define start (Home position)
    # Roughly [0, -1.57, 1.57, -1.57, -1.57, 0]
    start_qpos = np.array([0, -1.57, 1.57, -1.57, -1.57, 0])
    
    # Define target (Drop zone center)
    # [0.6, 0.2, 0.83]
    target_pos = np.array([0.6, 0.2, 0.83])
    
    print(f"Start Joint Pos: {start_qpos}")
    print(f"Target EE Pos: {target_pos}")
    
    success, trajectory, final_dist = planner.plan(
        start_joint_pos=start_qpos,
        target_ee_pos=target_pos,
        visualize=True
    )
    
    print("\n📊 Result:")
    print(f"Success: {success}")
    print(f"Steps taken: {len(trajectory)}")
    print(f"Final Distance: {final_dist:.4f} m")
    
    if success:
        print("🎉 Goal Reached!")
    else:
        print("❌ Failed to reach goal.")
        
    # Wait to see result
    time.sleep(2)
    
    print("\n-----------------------------------")
    print("🧪 Test Case 2: Different Start Position")
    
    # Slightly different start
    start_qpos_2 = np.array([0.5, -1.2, 1.8, -1.5, -1.5, 0])
    
    print(f"Start Joint Pos: {start_qpos_2}")
    
    success, trajectory, final_dist = planner.plan(
        start_joint_pos=start_qpos_2,
        target_ee_pos=target_pos,
        visualize=True
    )
    
    print("\n📊 Result:")
    print(f"Success: {success}")
    print(f"Steps taken: {len(trajectory)}")
    
    if success:
        print("🎉 Goal Reached!")
        
if __name__ == "__main__":
    main()
