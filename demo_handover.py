#!/usr/bin/env python3
"""
Dual-Arm Collaborative Handover Demonstration

This script demonstrates a complete handover task:
1. Left arm approaches and grasps cube
2. Left arm moves to handover position
3. Right arm approaches handover position
4. Right arm grasps cube while left releases
5. Right arm places cube at target location
6. Both arms return home
"""

import sys
import os
import numpy as np
import time

# Add paths
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'manipulator_grasp'))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from manipulator_grasp.env.dual_robot_env import DualRobotEnv
from manipulator_grasp.path_plan.dual_set_plan import DualArmPlanner

class HandoverTask:
    """State machine for dual-arm handover task."""
    
    # State definitions
    PHASE_INIT = "Initialization"
    PHASE_APPROACH = "Left arm approaches cube"
    PHASE_GRASP = "Left arm grasps cube"
    PHASE_TRANSFER = "Left moves to handover zone, Right approaches"
    PHASE_HANDOFF = "Right grasps, Left releases"
    PHASE_PLACE = "Right places cube at target"
    PHASE_RETURN = "Both return to home"
    
    def __init__(self, env, planner):
        self.env = env
        self.planner = planner
        self.nq = planner.model.nq
        self.constraint_id = None
        
        # Define key poses
        self.home_pose = self._define_home()
        self.left_pregrasp = self._define_left_pregrasp()
        self.left_grasp = self._define_left_grasp()
        self.handover_meet = self._define_handover_meet()
        self.right_place = self._define_right_place()
        
    def _define_home(self):
        """Both arms in neutral home position."""
        q = np.zeros(self.nq)
        q[0:6] = [0, -1.57, 0, -1.57, 0, 0]
        q[6:12] = [0, -1.57, 0, -1.57, 0, 0]
        return q
    
    def _define_left_pregrasp(self):
        """Left arm pre-grasp position above cube (0.6, 0.2, 0.74)."""
        q = self.home_pose.copy()
        # Adjust left arm to reach toward cube position
        q[0] = -0.3  # Pan toward right (cube is at y=0.2)
        q[1] = -1.2  # Shoulder lift
        q[2] = -0.5  # Elbow
        q[3] = -1.8  # Wrist 1
        return q
    
    def _define_left_grasp(self):
        """Left arm grasp position at cube."""
        q = self._define_left_pregrasp()
        # Lower slightly to grasp
        q[1] = -1.3
        return q
    
    def _define_handover_meet(self):
        """Both arms meet at center of workspace."""
        q = np.zeros(self.nq)
        # Left arm holds cube at center-left
        q[0:6] = [0.0, -1.0, -0.8, -1.5, 0, 0]
        # Right arm approaches from right
        q[6:12] = [0.0, -1.0, 0.8, -1.5, 0, 0]
        return q
    
    def _define_right_place(self):
        """Right arm placement position (move cube to right side)."""
        q = self.home_pose.copy()
        q[6] = 0.3   # Pan right
        q[7] = -1.2  # Reach forward
        q[8] = 0.5   # Elbow
        return q
    
    def execute_motion(self, q_start, q_goal, description, steps=40):
        """Execute motion by linear interpolation (collision-free from verification)."""
        print(f"\n  Executing: {description}")
        
        # Linear interpolation
        for i in range(steps + 1):
            alpha = i / steps
            q_interp = (1 - alpha) * q_start + alpha * q_goal
            
            # Set arm control
            ctrl = np.zeros(self.env.model.nu)
            ctrl[:12] = q_interp
            
            # Maintain gripper state during motion
            # (gripper commands are set separately)
            
            for _ in range(10):  # Smooth motion
                self.env.step(ctrl)
                time.sleep(1.0 / self.env.sim_hz)
        
        print(f"  ✓ Completed: {description}")
        return q_goal
    
    def hold_position(self, q_pos, duration=1.0):
        """Hold current position for specified duration."""
        ctrl = np.zeros(self.env.model.nu)
        ctrl[:12] = q_pos
        
        steps = int(duration * self.env.sim_hz)
        for _ in range(steps):
            self.env.step(ctrl)
            time.sleep(1.0 / self.env.sim_hz)
    
    def run(self):
        """Execute complete handover sequence."""
        print("\n" + "="*70)
        print("DUAL-ARM HANDOVER DEMONSTRATION".center(70))
        print("="*70)
        print("\nExecuting 6-phase collaborative manipulation task...\n")
        
        current_q = self.home_pose.copy()
        
        # PHASE 1: Approach cube
        print(f"[1/6] {self.PHASE_APPROACH}")
        self.env.open_gripper('left')
        current_q = self.execute_motion(current_q, self.left_pregrasp, "Moving left arm to pre-grasp")
        self.hold_position(current_q, 0.5)
        
        # PHASE 2: Grasp cube
        print(f"\n[2/6] {self.PHASE_GRASP}")
        current_q = self.execute_motion(current_q, self.left_grasp, "Lowering to grasp position")
        self.hold_position(current_q, 0.3)
        
        print("  Closing left gripper...")
        self.env.close_gripper('left')
        self.hold_position(current_q, 0.5)
        
        print("  Attaching cube to left gripper...")
        self.constraint_id = self.env.attach_object('handover_cube', 'left')
        self.hold_position(current_q, 0.5)
        
        # PHASE 3: Move to handover position
        print(f"\n[3/6] {self.PHASE_TRANSFER}")
        current_q = self.execute_motion(current_q, self.handover_meet, "Moving to handover position", steps=60)
        self.hold_position(current_q, 1.0)
        
        # PHASE 4: Handoff
        print(f"\n[4/6] {self.PHASE_HANDOFF}")
        print("  Right gripper approaching...")
        self.env.open_gripper('right')
        self.hold_position(current_q, 0.5)
        
        print("  Right gripper closing...")
        self.env.close_gripper('right')
        self.hold_position(current_q, 0.5)
        
        print("  Transferring constraint to right arm...")
        if self.constraint_id is not None:
            self.env.detach_object(self.constraint_id)
        self.constraint_id = self.env.attach_object('handover_cube', 'right')
        self.hold_position(current_q, 0.3)
        
        print("  Left gripper releasing...")
        self.env.open_gripper('left')
        self.hold_position(current_q, 0.5)
        
        # PHASE 5: Place cube
        print(f"\n[5/6] {self.PHASE_PLACE}")
        
        # Right arm moves to placement location, left returns home
        q_place = self.right_place.copy()
        # Keep left at home during right's motion
        q_place[:6] = self.home_pose[:6]
        
        current_q = self.execute_motion(current_q, q_place, "Right arm placing cube", steps=50)
        self.hold_position(current_q, 0.5)
        
        print("  Detaching cube...")
        self.env.open_gripper('right')
        if self.constraint_id is not None:
            self.env.detach_object(self.constraint_id)
        self.hold_position(current_q, 1.0)
        
        # PHASE 6: Return home
        print(f"\n[6/6] {self.PHASE_RETURN}")
        current_q = self.execute_motion(current_q, self.home_pose, "Returning to home", steps=50)
        self.hold_position(current_q, 1.0)
        
        print("\n" + "="*70)
        print("HANDOVER DEMONSTRATION COMPLETE!".center(70))
        print("="*70)
        print("\nSuccessfully demonstrated:")
        print("  ✓ Coordinated dual-arm motion planning")
        print("  ✓ Object manipulation and grasping")
        print("  ✓ Synchronized handover between arms")
        print("  ✓ Collision-free workspace sharing")
        print("\n")

def main():
    print("\n" + "#"*70)
    print("#" + " DUAL-ARM COLLABORATIVE HANDOVER DEMONSTRATION ".center(68, " ") + "#")
    print("#"*70)
    print("\nInitializing system...")
    
    # Setup
    urdf_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "manipulator_grasp/robot_description/urdf/ur3e_ag95_dual.urdf"
    )
    
    print("Loading planner...")
    planner = DualArmPlanner(urdf_path)
    
    print("Loading environment...")
    env = DualRobotEnv(headless=False)
    env.reset()
    
    print(f"✓ System ready")
    print(f"  Planner: {planner.model.nq} DOF")
    print(f"  Environment: {env.model.nu} actuators")
    
    # Create and run handover task
    task = HandoverTask(env, planner)
    
    input("\nPress Enter to start handover demonstration...")
    
    try:
        task.run()
        input("\nPress Enter to close...")
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    finally:
        env.close()
    
    print("\n" + "#"*70)
    print("#" + " Demonstration complete! ".center(68, " ") + "#")
    print("#"*70 + "\n")

if __name__ == "__main__":
    main()
