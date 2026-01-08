import sys
import os
import cv2
import numpy as np

# Add manipulator_grasp to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'manipulator_grasp'))

from manipulator_grasp.env.dual_robot_env import DualRobotEnv

def test_dual_env():
    print("Initializing DualRobotEnv (Generic)...")
    env = DualRobotEnv(headless=True)
    env.reset()
    
    nu = env.model.nu
    print(f"Model nu (Actuators): {nu}")
    
    print("Running simulation loop...")
    for i in range(100):
        action = np.zeros(nu)
        # Apply some random movements if we have actuators
        if nu >= 14:
             # Just an example for UR3e, assuming we know indices. 
             # In a generic env, we might not know indices, but this test is specifically checking the setup.
            action[1] = -1.57 # Left lift
            action[8] = -1.57 # Right lift
        
        env.step(action)
    
    print("Rendering frame...")
    ret = env.render()
    img = ret['img']
    
    # Convert RGB to BGR for cv2
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('dual_setup_test_v2.png', img_bgr)
    print("Saved dual_setup_test_v2.png")
    
    env.close()
    print("Test passed!")

if __name__ == "__main__":
    test_dual_env()
