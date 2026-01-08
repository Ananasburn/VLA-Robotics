import os.path
import sys
import numpy as np
import mujoco
import mujoco.viewer
import cv2

# Ensure manipulator_grasp is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

class DualRobotEnv:
    """
    A generic environment for dual-robot manipulation.
    Supports loading different scene XMLs (e.g., dual UR3e, dual Panda).
    """

    def __init__(self, scene_xml_path=None, headless=False):
        self.sim_hz = 500
        self.model: mujoco.MjModel = None
        self.data: mujoco.MjData = None
        self.headless = headless

        # If no scene provided, default to the dual UR3e scene
        # ---------------------------------------------------------------------------
        # TO CHANGE ROBOT (e.g., to Franka Panda):
        # 1. Create a new XML file (e.g., 'assets/scenes/scene_panda_dual.xml') 
        #    that includes the Panda robot definition instead of UR3e.
        # 2. Instantiate this class with the path to your new XML:
        #    env = DualRobotEnv(scene_xml_path='/path/to/scene_panda_dual.xml')
        # ---------------------------------------------------------------------------
        if scene_xml_path is None:
            self.scene_xml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'scenes', 'scene_dual.xml')
        else:
            self.scene_xml_path = scene_xml_path

        self.renderer: mujoco.Renderer = None
        self.depth_renderer: mujoco.Renderer = None
        self.viewer: mujoco.viewer.Handle = None

        self.height = 640
        self.width = 640
        self.fovy = np.pi / 4 

    def reset(self):
        if not os.path.exists(self.scene_xml_path):
             raise FileNotFoundError(f"Scene file not found: {self.scene_xml_path}")
             
        self.model = mujoco.MjModel.from_xml_path(self.scene_xml_path)
        self.data = mujoco.MjData(self.model)

        # Forward simulation to settle
        mujoco.mj_forward(self.model, self.data)

        # Initialize Renderers
        self.renderer = mujoco.renderer.Renderer(self.model, height=self.height, width=self.width)
        self.depth_renderer = mujoco.renderer.Renderer(self.model, height=self.height, width=self.width)
        self.renderer.update_scene(self.data, 0)
        self.depth_renderer.update_scene(self.data, 0)
        self.depth_renderer.enable_depth_rendering()
        
        # Initialize Viewer if not headless
        if not self.headless:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            # Adjust viewer to look at the center of the table
            self.viewer.cam.lookat[:] = [0.8, 0, 0.7] 
            self.viewer.cam.azimuth = 180
            self.viewer.cam.elevation = -30
            self.viewer.cam.distance = 2.0
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        if self.renderer is not None:
            self.renderer.close()
        if self.depth_renderer is not None:
            self.depth_renderer.close()

    def step(self, action=None):
        """
        Generic step function.
        action: np.array of shape (nu,) where nu is the number of actuators.
        """
        if action is not None:
            # Safety check for action dimension
            if len(action) != self.model.nu:
                print(f"[WARNING] Action dim {len(action)} != Model nu {self.model.nu}")
            
            # Apply action (clip to control range if needed, here we blindly apply)
            self.data.ctrl[:] = action
            
        mujoco.mj_step(self.model, self.data)
        
        if self.viewer is not None:
            self.viewer.sync()

    def open_gripper(self, arm='left'):
        """Open gripper to release/prepare for grasp."""
        if arm == 'left':
            # Left gripper actuator is index 12
            self.data.ctrl[12] = 0.0  # Open position
        elif arm == 'right':
            # Right gripper actuator is index 13  
            self.data.ctrl[13] = 0.0  # Open position
        else:
            raise ValueError(f"Unknown arm: {arm}")
    
    def close_gripper(self, arm='left'):
        """Close gripper to grasp object."""
        if arm == 'left':
            self.data.ctrl[12] = 0.8  # Close position (adjust as needed)
        elif arm == 'right':
            self.data.ctrl[13] = 0.8
        else:
            raise ValueError(f"Unknown arm: {arm}")
    
    def get_object_pose(self, object_name='handover_cube'):
        """Get position and orientation of an object."""
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_name)
            pos = self.data.xpos[body_id].copy()
            quat = self.data.xquat[body_id].copy()
            return pos, quat
        except:
            return None, None
    
    def attach_object(self, object_name, arm='left'):
        """
        Attach object to gripper using equality constraint (weld).
        Returns constraint id or None if failed.
        """
        try:
            # Get gripper body (using wrist_3 or tool0)
            if arm == 'left':
                gripper_body = 'left_tool0'
            else:
                gripper_body = 'right_tool0'
            
            body1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, gripper_body)
            body2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_name)
            
            # Find free equality constraint slot
            for i in range(self.model.neq):
                if self.model.eq_active[i] == 0:
                    # Configure weld constraint
                    self.model.eq_type[i] = mujoco.mjtEq.mjEQ_WELD
                    self.model.eq_obj1id[i] = body1_id
                    self.model.eq_obj2id[i] = body2_id
                    self.model.eq_active[i] = 1
                    
                    # Relative pose (gripper -> object)
                    # Compute current relative transform
                    obj_pos, obj_quat = self.get_object_pose(object_name)
                    gripper_id = body1_id
                    gripper_pos = self.data.xpos[gripper_id]
                    
                    # Store relative position
                    self.model.eq_data[i, :3] = obj_pos - gripper_pos
                    
                    print(f"Attached {object_name} to {arm} arm (constraint {i})")
                    return i
            
            print(f"No free equality constraints available!")
            return None
            
        except Exception as e:
            print(f"Failed to attach object: {e}")
            return None
    
    def detach_object(self, constraint_id):
        """Detach object by deactivating equality constraint."""
        if constraint_id is not None and constraint_id < self.model.neq:
            self.model.eq_active[constraint_id] = 0
            print(f"Detached object (constraint {constraint_id})")
            return True
        return False

    def render(self):
        self.renderer.update_scene(self.data, 0)
        self.depth_renderer.update_scene(self.data, 0)
        return {
            'img': self.renderer.render(),
            'depth': self.depth_renderer.render()
        }

if __name__ == '__main__':
    # Test with default UR3e
    env = DualRobotEnv()
    env.reset()
    print(f"Loaded model from: {env.scene_xml_path}")
    print(f"Model nu (Actuators): {env.model.nu}")
    env.close()
