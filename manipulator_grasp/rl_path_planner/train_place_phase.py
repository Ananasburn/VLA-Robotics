#!/usr/bin/env python3
"""
Training Script for Place Phase RL Model

This script trains a PPO model for the place phase task using
diverse starting configurations for better generalization.

Usage:
    python3 train_place_phase.py --name place_v1 --timesteps 5000000 --envs 8
"""

import os
import sys

# Prevent memory explosion due to thread duplication in subprocesses
os.environ['OMP_NUM_THREADS'] = '1' 
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import time
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import (
    EvalCallback, 
    CheckpointCallback,
    BaseCallback
)
from stable_baselines3.common.monitor import Monitor

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.rl_place_env import RLPlaceEnv, make_place_env
from utils.hybrid_vec_env import HybridVecEnv


class SuccessRateCallback(BaseCallback):
    """Callback to track and log success rate during training."""
    
    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.successes = []
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # Check if any episode finished
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_count += 1
                success = info.get("success", False)
                self.successes.append(int(success))
                
                # Log every check_freq episodes
                if self.episode_count % self.check_freq == 0:
                    recent_successes = self.successes[-self.check_freq:]
                    success_rate = np.mean(recent_successes) * 100
                    self.logger.record("custom/success_rate", success_rate)
                    self.logger.record("custom/episodes", self.episode_count)
                    if self.verbose:
                        print(f"Episode {self.episode_count}: Success Rate = {success_rate:.1f}%")
        return True


def make_env(rank: int, seed: int = 0, max_steps: int = 200, visualize: bool = False):
    """Create a single environment instance."""
    def _init():
        # Only visualize the first environment (rank 0) if requested
        should_visualize = visualize and (rank == 0)
        env = RLPlaceEnv(max_steps=max_steps, visualize=should_visualize)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def train(args):
    """Main training function."""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.name or f"place_phase_{timestamp}"
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models",
        model_name
    )
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=" * 60)
    print(f"Training Place Phase RL Model")
    print(f"=" * 60)
    print(f"Model name: {model_name}")
    print(f"Output directory: {output_dir}")
    print(f"Total timesteps: {args.timesteps:,}")
    print(f"Number of environments: {args.envs}")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Visualization: {'Enabled (Env 0)' if args.visualize else 'Disabled'}")
    print(f"=" * 60)
    
    # Create vectorized environment
    print("\nCreating training environments...")
    env_fns = [
        make_env(i, seed=args.seed, max_steps=args.max_steps, visualize=args.visualize) 
        for i in range(args.envs)
    ]
    
    # Use HybridVecEnv for memory efficiency (env_0 in main process, rest in subprocesses)
    if args.visualize and args.envs > 1:
        print(f"🚀 Using HybridVecEnv: env_0 visualized + {args.envs-1} in subprocesses")
        env = HybridVecEnv(env_fns)
    elif args.visualize or args.envs == 1:
        env = DummyVecEnv(env_fns)
    else:
        env = SubprocVecEnv(env_fns)
    
    # Add monitoring
    env = VecMonitor(env)
    
    # Wrap with VecNormalize for observation normalization
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )
    
    # Create evaluation environment (no visualization, single env)
    print("Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env(100, seed=args.seed + 100, max_steps=args.max_steps, visualize=False)])
    
    # Add monitoring (same as training env)
    eval_env = VecMonitor(eval_env)
    
    eval_env = VecNormalize(
        eval_env, 
        norm_obs=True, 
        norm_reward=False, 
        clip_obs=10.0,
        training=False
    )
    
    # Create PPO model
    print("\nInitializing PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=os.path.join(output_dir, "tb_logs"),
        device="cpu",
    )
    
    # Setup callbacks
    callbacks = [
        CheckpointCallback(
            save_freq=max(50000 // args.envs, 1000),
            save_path=os.path.join(output_dir, "checkpoints"),
            name_prefix="place_model"
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=output_dir,
            log_path=os.path.join(output_dir, "eval_logs"),
            eval_freq=max(10000 // args.envs, 500),
            n_eval_episodes=20,
            deterministic=True,
        ),
        SuccessRateCallback(check_freq=100),
    ]
    
    # Train
    print(f"\nStarting training for {args.timesteps:,} timesteps...")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/3600:.2f} hours")
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model")
    model.save(final_model_path)
    env.save(os.path.join(output_dir, "final_model_vecnormalize.pkl"))
    
    print(f"\nModel saved to: {final_model_path}")
    print(f"VecNormalize saved to: {output_dir}/final_model_vecnormalize.pkl")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    print("\nTraining complete!")
    print(f"To evaluate, run:")
    print(f"  python3 evaluate_place_phase.py --model {output_dir}/best_model.zip")


def main():
    parser = argparse.ArgumentParser(description="Train Place Phase RL Model")
    parser.add_argument("--name", type=str, default=None, help="Model name")
    parser.add_argument("--timesteps", type=int, default=5_000_000, help="Total timesteps")
    parser.add_argument("--envs", type=int, default=60, help="Number of parallel environments")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization (rank 0 only)")
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
