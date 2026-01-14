
import os
import sys
import numpy as np
from typing import Tuple, List, Optional
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from manipulator_grasp.env.rl_task_space_env import RLTaskSpaceEnv

class RLPathPlanner:
    """
    Wrapper for deploying the trained RL PPO model for path planning.
    """
    
    def __init__(
        self, 
        model_path: str, 
        env_kwargs: Optional[dict] = None,
        device: str = "cpu"
    ):
        """
        Initialize the planner.
        
        Args:
            model_path: Path to the .zip model file
            env_kwargs: Arguments to pass to RLTaskSpaceEnv constructor
            device: 'cpu' or 'cuda'
        """
        self.device = device
        
        # 1. Create Environment
        # We use DummyVecEnv because SB3 models expect vectorized environments
        def make_env():
            return RLTaskSpaceEnv(**(env_kwargs or {}))
            
        self.env = DummyVecEnv([make_env])
        
        # 2. Load Normalization Statistics
        # This is CRITICAL because the model was trained with normalized observations
        vecnormalize_path = model_path.replace(".zip", "") + "_vecnormalize.pkl"
        first_try_path = vecnormalize_path
        
        # Handle case where model name is 'best_model.zip' but norm file is 'final_model_vecnormalize.pkl'
        # or if they are stored in the same directory but with slightly different naming conventions
        if not os.path.exists(vecnormalize_path):
            dir_name = os.path.dirname(model_path)
            vecnormalize_path = os.path.join(dir_name, "final_model_vecnormalize.pkl")
            
        if not os.path.exists(vecnormalize_path):
             # Try one more guess: model_dir/vecnormalize.pkl
             raise FileNotFoundError(f"Could not find VecNormalize stats at {first_try_path} or {vecnormalize_path}")
            
        print(f"Loading normalization stats from: {vecnormalize_path}")
        self.env = VecNormalize.load(vecnormalize_path, self.env)
        
        # Important: Turn off training and reward normalization during inference
        self.env.training = False
        self.env.norm_reward = False
        
        # 3. Load Model
        print(f"Loading model from: {model_path}")
        self.model = PPO.load(model_path, device=self.device)
        
    def plan(
        self, 
        start_joint_pos: np.ndarray, 
        target_ee_pos: np.ndarray, 
        max_steps: int = 100,
        visualize: bool = False
    ) -> Tuple[bool, List[np.ndarray], float]:
        """
        Plan a path from start joint configuration to target end-effector position.
        
        Args:
            start_joint_pos: Initial joint angles (6,)
            target_ee_pos: Target end-effector position (3,)
            max_steps: Maximum steps allowed
            visualize: Whether to render the planning process (requires env with visualize=True)
            
        Returns:
            success: Boolean indicating if target was reached
            trajectory: List of joint positions
            final_distance: Final distance to target
        """
        
        # Configure reset options
        options = {
            "initial_qpos": start_joint_pos,
            "target_ee_pos": target_ee_pos
        }
        
        # Reset environment
        obs = self.env.reset()
        
        # Hack: VecEnv reset doesn't accept options in older SB3 versions or wrappers
        # We need to manually inject the options into the underlying env
        # Access the underlying UnvecEnv -> RLTaskSpaceEnv
        real_env = self.env.envs[0]
        real_env.reset(options=options)
        
        # Re-get observation after manual reset to ensure it matches the new state
        # But since we use VecNormalize, we need to manually normalize it
        raw_obs = real_env._get_observation()
        # VecNormalize expects (n_envs, obs_dim)
        obs = self.env.normalize_obs(raw_obs)
        
        trajectory = [start_joint_pos]
        success = False
        final_dist = 100.0
        
        print(f"Planning to target: {target_ee_pos}...")
        
        for step in range(max_steps):
            # Predict action
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, done, info = self.env.step(action)
            
            # Record state
            # info is a list of dicts for VecEnv
            current_info = info[0]
            
            if "dist_to_target" in current_info:
                final_dist = current_info["dist_to_target"]
                
            # SB3 VecEnv automatically resets on done, so we need to check info
            # In our case, we want to stop.
            if done[0]:
                # Check success flag
                if current_info.get("success", False):
                    success = True
                    print(f"Success! Reached target in {step+1} steps. Final dist: {final_dist:.4f}")
                else:
                    print(f"Failed (terminated). Final dist: {final_dist:.4f}")
                break
                
            # If not done, append current joint pos (approximate, or get from env)
            q_curr = real_env.mj_data.qpos[:6].copy()
            trajectory.append(q_curr)
            
            if visualize:
                real_env.render()
                
        if not success and final_dist < 0.05:
            print("Technically within threshold but env didn't flag success yet.")
            success = True
            
        return success, trajectory, final_dist
