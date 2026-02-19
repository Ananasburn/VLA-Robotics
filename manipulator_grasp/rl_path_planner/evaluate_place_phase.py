#!/usr/bin/env python3
"""
Evaluate Place Phase RL Agent
加载训练好的模型并可视化评估

Usage:
    python3 evaluate_place_phase.py --model logs/place_with_object_v1/best_model.zip
    python3 evaluate_place_phase.py --model logs/place_with_object_v1  # Auto-finds best model
"""

import os
import sys
import argparse
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.rl_place_env import RLPlaceEnv, make_place_env
import model_config


def evaluate(
    model_path: str,
    vecnormalize_path: str,
    n_episodes: int = 10,
    visualize: bool = True,
):
    """
    Evaluate the trained place phase model.
    
    Args:
        model_path: Path to the trained PPO model (.zip)
        vecnormalize_path: Path to VecNormalize stats (.pkl)
        n_episodes: Number of episodes to evaluate
        visualize: Whether to render the environment
    """
    print(f"Loading model from {model_path}")
    print(f"Loading normalization info from {vecnormalize_path}")
    
    # 1. Create environment
    env = make_place_env(visualize=visualize, max_steps=500)
    
    # 2. Wrap as VecEnv (required by VecNormalize)
    vec_env = DummyVecEnv([lambda: env])
    
    # 3. Load VecNormalize statistics
    if os.path.exists(vecnormalize_path):
        vec_env = VecNormalize.load(vecnormalize_path, vec_env)
        vec_env.training = False  # Evaluation mode, don't update stats
        vec_env.norm_reward = False 
        print("✓ VecNormalize loaded successfully")
    else:
        print("⚠ Warning: VecNormalize path not found, running without normalization!")
        
    # 4. Load model
    model = PPO.load(model_path)
    print("✓ Model loaded successfully")
    
    # 5. Run evaluation
    print(f"\n{'='*60}")
    print(f"Starting evaluation for {n_episodes} episodes")
    print(f"{'='*60}\n")
    
    results = {
        "successes": [],
        "rewards": [],
        "episode_lengths": [],
        "final_distances": [],
    }
    
    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = False
        step = 0
        total_reward = 0
        
        # Get environment info
        unwrapped_env = vec_env.envs[0].env if hasattr(vec_env.envs[0], 'env') else vec_env.envs[0]
        
        print(f"\n{'─'*60}")
        print(f"Episode {ep+1}/{n_episodes}")
        print(f"{'─'*60}")
        print(f"Target (drop zone): {unwrapped_env.drop_zone_center}")
        print(f"Attached object: {unwrapped_env.attached_object_name}")
        
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Execute action
            obs, reward, done, info = vec_env.step(action)
            total_reward += reward[0]
            step += 1
            
            # Get real environment info (VecEnv returns lists)
            real_info = info[0]
            dist = real_info.get('dist_to_target', 0.0)
            
            if step % 50 == 0:
                print(f"  Step {step:3d}: dist={dist:.4f}, reward={reward[0]:7.2f}")
                
            if visualize:
                time.sleep(0.02)  # Slow down for visualization
                
        # Record results
        success = real_info.get('success', False)
        results["successes"].append(int(success))
        results["rewards"].append(total_reward)
        results["episode_lengths"].append(step)
        results["final_distances"].append(dist)
        
        # Print episode summary
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"\n{status}")
        print(f"  Total reward:   {total_reward:8.2f}")
        print(f"  Episode length: {step:3d} steps")
        print(f"  Final distance: {dist:.4f}m")

    # Print overall summary
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Episodes:        {n_episodes}")
    print(f"Success rate:    {np.mean(results['successes'])*100:.1f}%")
    print(f"Avg reward:      {np.mean(results['rewards']):.2f} ± {np.std(results['rewards']):.2f}")
    print(f"Avg ep. length:  {np.mean(results['episode_lengths']):.1f} ± {np.std(results['episode_lengths']):.1f}")
    print(f"Avg final dist:  {np.mean(results['final_distances']):.4f}m")
    print(f"{'='*60}\n")

    vec_env.close()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Place Phase RL Model")
    parser.add_argument("--model", type=str, help="Path to .zip model file or directory. Defaults to model_config.py")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    args = parser.parse_args()
    
    # 0. Load defaults from config if not provided
    norm_path = None
    
    if args.model is None:
        print("Using default model from model_config.py...")
        config = model_config.get_place_phase_config()
        args.model = config['model_path']
        norm_path = config['vecnormalize_path']
        print(f"Loaded config: {args.model}")
        
    # If the user provides a directory (or config returned a directory? Unlikely but check)
    if os.path.isdir(args.model):
        if os.path.exists(os.path.join(args.model, "best_model.zip")):
             args.model = os.path.join(args.model, "best_model.zip")
             print(f"Auto-selected model: {args.model}")
        elif os.path.exists(os.path.join(args.model, "final_model.zip")):
             args.model = os.path.join(args.model, "final_model.zip")
             print(f"Auto-selected model: {args.model}")
        else:
             # Try to find the latest checkpoint
             import glob
             checkpoints = glob.glob(os.path.join(args.model, "place_phase_*_steps.zip"))
             if checkpoints:
                 checkpoints.sort(key=lambda x: int(x.split("_")[-2]))
                 args.model = checkpoints[-1]
                 print(f"Auto-selected latest checkpoint: {args.model}")
             else:
                 print(f"Error: No model found in directory {args.model}")
                 sys.exit(1)

    # Infer vecnormalize path (if not already set from config)
    if not args.model.endswith(".zip"):
        args.model += ".zip"
        
    if norm_path is None:
        norm_path = args.model.replace(".zip", "_vecnormalize.pkl")
        base_dir = os.path.dirname(args.model)
    
        if not os.path.exists(norm_path):
        # 1. Try common vecnormalize filenames
        possible_names = [
            "final_model_vecnormalize.pkl",
            "best_model_vecnormalize.pkl",
            "vecnormalize.pkl",
        ]
        for name in possible_names:
            p = os.path.join(base_dir, name)
            if os.path.exists(p):
                norm_path = p
                break
        else:
            # 2. If not found, try to find latest checkpoint vecnormalize
            import glob
            checkpoint_norms = glob.glob(os.path.join(base_dir, "place_phase_vecnormalize_*_steps.pkl"))
            if checkpoint_norms:
                # Sort by step count
                checkpoint_norms.sort(key=lambda x: int(x.split("_")[-2]))
                norm_path = checkpoint_norms[-1]
                print(f"Using checkpoint vecnormalize: {norm_path}")
        
    evaluate(args.model, norm_path, n_episodes=args.episodes, visualize=not args.no_viz)
