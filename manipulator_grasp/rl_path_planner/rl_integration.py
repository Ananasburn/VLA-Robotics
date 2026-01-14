"""
RL Path Planner Integration Module
Provides an interface to use trained PPO model for path planning in grasp_process.py
"""

import os
import numpy as np
import pinocchio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import mujoco


class RLPathPlanner:
    """
    Wrapper class for using a trained PPO model for path planning.
    
    The model is trained in task-space: it takes observation containing
    (ee_pos, ee_quat, joint_pos, target_pos) and outputs joint position deltas.
    
    IMPORTANT: This model was trained with a FIXED target (drop_zone_center).
    It does not generalize to arbitrary targets.
    """
    
    # Default model path
    DEFAULT_MODEL_DIR = os.path.join(
        os.path.dirname(__file__), 
        'models', 
        'task_space_v5_8_collision_check'
    )
    
    # Training target (MUST match rl_task_space_env.py drop_zone_center)
    TRAINING_DROP_ZONE = np.array([0.6, 0.2, 0.83])
    
    def __init__(self, model_path: str = None, vecnormalize_path: str = None):
        """
        Initialize the RL planner.
        
        Args:
            model_path: Path to the .zip model file. If None, uses best_model.zip
            vecnormalize_path: Path to vecnormalize .pkl file. If None, auto-inferred.
        """
        if model_path is None:
            model_path = os.path.join(self.DEFAULT_MODEL_DIR, 'best_model.zip')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Infer vecnormalize path if not provided
        if vecnormalize_path is None:
            vecnormalize_path = os.path.join(
                os.path.dirname(model_path), 
                'final_model_vecnormalize.pkl'
            )
        
        print(f"[RL Planner] Loading model from {model_path}")
        self.model = PPO.load(model_path)
        
        # Load normalization statistics
        self.obs_mean = None
        self.obs_var = None
        if os.path.exists(vecnormalize_path):
            print(f"[RL Planner] Loading normalization from {vecnormalize_path}")
            import pickle
            with open(vecnormalize_path, 'rb') as f:
                vecnorm_data = pickle.load(f)
                if hasattr(vecnorm_data, 'obs_rms'):
                    self.obs_mean = vecnorm_data.obs_rms.mean
                    self.obs_var = vecnorm_data.obs_rms.var
                    print(f"[RL Planner] Loaded obs normalization: mean shape {self.obs_mean.shape}")
        else:
            print(f"[RL Planner] Warning: VecNormalize not found at {vecnormalize_path}")
        
        # RL environment parameters (must match training config)
        self.action_scale = 0.3
        self.success_threshold = 0.10  # 10cm (more lenient for fallback to RRT)
        self.max_steps = 200
        self.clip_obs = 10.0  # VecNormalize uses clip_obs=10.0
        
        # Pinocchio model for FK (will be set from env)
        self.pin_model = None
        self.pin_data = None
        self.ee_frame_id = None
        
    def _setup_pinocchio(self, env):
        """Setup Pinocchio model from the grasp environment."""
        if self.pin_model is None and hasattr(env, 'model_roboplan'):
            self.pin_model = env.model_roboplan
            self.pin_data = self.pin_model.createData()
            self.ee_frame_id = self.pin_model.getFrameId("grasp_center")
            print(f"[RL Planner] Setup Pinocchio, EE frame ID: {self.ee_frame_id}")
    
    def _get_ee_position(self, q: np.ndarray) -> np.ndarray:
        """Compute end-effector position using Pinocchio FK."""
        if self.pin_model is None:
            raise RuntimeError("Pinocchio model not initialized. Call _setup_pinocchio first.")
        
        q_full = np.zeros(self.pin_model.nq)
        q_full[:6] = q
        pinocchio.forwardKinematics(self.pin_model, self.pin_data, q_full)
        pinocchio.updateFramePlacements(self.pin_model, self.pin_data)
        return self.pin_data.oMf[self.ee_frame_id].translation.copy()
    
    def _get_ee_orientation(self, q: np.ndarray) -> np.ndarray:
        """Compute end-effector orientation as quaternion."""
        if self.pin_model is None:
            raise RuntimeError("Pinocchio model not initialized.")
        
        q_full = np.zeros(self.pin_model.nq)
        q_full[:6] = q
        pinocchio.forwardKinematics(self.pin_model, self.pin_data, q_full)
        pinocchio.updateFramePlacements(self.pin_model, self.pin_data)
        
        rotation = self.pin_data.oMf[self.ee_frame_id].rotation
        quat = pinocchio.Quaternion(rotation).coeffs()  # [x, y, z, w]
        return quat.copy()
    
    def _make_observation(self, env, target_pos: np.ndarray) -> np.ndarray:
        """
        Construct observation vector matching the RL training environment.
        
        Observation (16-dim):
            - ee_pos (3)
            - ee_quat (4)
            - joint_pos (6)
            - target_pos (3)
        """
        q_current = env.data.qpos[:6].copy()
        ee_pos = self._get_ee_position(q_current)
        ee_quat = self._get_ee_orientation(q_current)
        
        obs = np.concatenate([
            ee_pos,       # 3
            ee_quat,      # 4
            q_current,    # 6
            target_pos,   # 3
        ]).astype(np.float32)
        
        return obs
    
    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Apply observation normalization if available (matching VecNormalize)."""
        if self.obs_mean is not None and self.obs_var is not None:
            normalized = (obs - self.obs_mean) / np.sqrt(self.obs_var + 1e-8)
            # VecNormalize clips to [-clip_obs, clip_obs]
            return np.clip(normalized, -self.clip_obs, self.clip_obs)
        return obs
    
    def run_plan(
        self, 
        env, 
        target_pos: np.ndarray,
        visualize: bool = True
    ) -> tuple:
        """
        Execute RL policy to move the robot to target position.
        
        Args:
            env: UR3eGraspEnv instance
            target_pos: Target end-effector position (3,)
            visualize: Whether to sync viewer during execution
            
        Returns:
            success: Whether the target was reached
            trajectory: List of joint configurations visited
        """
        self._setup_pinocchio(env)
        
        # IMPORTANT: Use the exact training target, not the requested target
        # The model was trained with a fixed target and doesn't generalize
        actual_target = self.TRAINING_DROP_ZONE.copy()
        requested_target = np.array(target_pos)
        
        if np.linalg.norm(actual_target - requested_target) > 0.15:
            print(f"[RL Planner] WARNING: Requested target {requested_target} differs from training target {actual_target}")
        
        trajectory = []
        
        print(f"[RL Planner] Starting plan to training target: {actual_target}")
        print(f"[RL Planner] (Requested target was: {requested_target})")
        
        for step in range(self.max_steps):
            # Get current state
            q_current = env.data.qpos[:6].copy()
            trajectory.append(q_current.copy())
            
            # Check if reached training target
            ee_pos = self._get_ee_position(q_current)
            dist = np.linalg.norm(ee_pos - actual_target)
            
            if step % 20 == 0:
                print(f"[RL Planner] Step {step}: dist={dist:.4f}")
            
            if dist < self.success_threshold:
                print(f"[RL Planner] Target reached in {step} steps!")
                return True, trajectory
            
            # Construct observation with TRAINING target (not requested)
            obs = self._make_observation(env, actual_target)
            obs_normalized = self._normalize_obs(obs)
            
            # Get action from policy
            action, _ = self.model.predict(obs_normalized, deterministic=True)
            
            # Scale action to joint deltas
            scaled_action = action * self.action_scale
            
            # Apply action
            q_target = q_current + scaled_action
            q_target = np.clip(q_target, -2*np.pi, 2*np.pi)  # Joint limits
            
            # Set control and step simulation
            env.data.ctrl[:6] = q_target
            
            # Multiple substeps for stability
            for _ in range(10):
                mujoco.mj_step(env.model, env.data)
            
            if visualize and hasattr(env, 'viewer') and env.viewer is not None:
                env.viewer.sync()
        
        print(f"[RL Planner] Timeout after {self.max_steps} steps (dist={dist:.4f})")
        return False, trajectory


# Singleton instance for caching
_rl_planner_instance = None

def get_rl_planner(model_path: str = None) -> RLPathPlanner:
    """Get or create a cached RLPathPlanner instance."""
    global _rl_planner_instance
    if _rl_planner_instance is None:
        _rl_planner_instance = RLPathPlanner(model_path)
    return _rl_planner_instance
