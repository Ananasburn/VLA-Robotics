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

class DualUR3eGraspEnv:

    def __init__(self, headless=False):
        self.sim_hz = 500
        self.model: mujoco.MjModel = None
        self.data: mujoco.MjData = None
        self.headless = headless

        self.renderer: mujoco.Renderer = None
        self.depth_renderer: mujoco.Renderer = None
        self.viewer: mujoco.viewer.Handle = None

        self.height = 640
        self.width = 640
        self.fovy = np.pi / 4 # Approximate fov

    def reset(self):
        # Load the dual-arm scene
        # Path: manipulator_grasp/assets/scenes/scene_dual.xml
        filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'scenes', 'scene_dual.xml')
        if not os.path.exists(filename):
             raise FileNotFoundError(f"Scene file not found: {filename}")
             
        self.model = mujoco.MjModel.from_xml_path(filename)
        self.data = mujoco.MjData(self.model)

        # Set initial joint positions
        # Left Arm (indices 0-5), Left Gripper (6)
        # Right Arm (indices 7-12), Right Gripper (13)
        # Indices depend on the order in XML. 
        # In ur3e_ag95_dual.xml:
        # Actuators: 
        # 0-5: Left Arm
        # 6: Left Gripper
        # 7-12: Right Arm
        # 13: Right Gripper
        
        # Initial pose: Upright or home
        # qpos structure includes joint positions. 
        # UR3e joints: 6 per arm. Gripper joints: 6 mimic joints? 
        # Let's just set ctrl (actuators) and let physics settle, or set qpos if we know indices.
        # For simplicity in Phase 1, we just run forward.
        
        mujoco.mj_forward(self.model, self.data)

        # Initialize Renderers
        self.renderer = mujoco.renderer.Renderer(self.model, height=self.height, width=self.width)
        self.depth_renderer = mujoco.renderer.Renderer(self.model, height=self.height, width=self.width)
        self.renderer.update_scene(self.data, 0)
        self.depth_renderer.update_scene(self.data, 0)
        self.depth_renderer.enable_depth_rendering()
        
        # Initialize Viewer
        if not self.headless:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.lookat[:] = [0.8, 0, 0.75]
            self.viewer.cam.azimuth = 180
            self.viewer.cam.elevation = -30
            self.viewer.cam.distance = 1.5
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
        action: np.array of shape (14,) -> [Left(7), Right(7)]
        """
        if action is not None:
            self.data.ctrl[:] = action
            
        mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

    def render(self):
        self.renderer.update_scene(self.data, 0)
        self.depth_renderer.update_scene(self.data, 0)
        return {
            'img': self.renderer.render(),
            'depth': self.depth_renderer.render()
        }

if __name__ == '__main__':
    env = DualUR3eGraspEnv()
    env.reset()
    print(f"Model nq (Generalized Coordinates): {env.model.nq}")
    print(f"Model nv (Generalized Velocities): {env.model.nv}")
    print(f"Model nu (Actuators): {env.model.nu}")
    
    # Simple loop
    for i in range(2000):
        # Small sine wave motion for testing
        action = np.zeros(14)
        action[1] = -1.57 + 0.5 * np.sin(i * 0.01) # Left Shoulder Lift
        action[8] = -1.57 + 0.5 * np.cos(i * 0.01) # Right Shoulder Lift
        env.step(action)
        
    env.close()
