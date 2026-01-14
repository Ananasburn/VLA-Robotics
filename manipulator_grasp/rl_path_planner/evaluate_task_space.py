"""
Evaluate Task-Space RL Agent
加载训练好的模型并可视化评估
"""

import os
import sys
import argparse
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# 添加项目路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from manipulator_grasp.env.rl_task_space_env import make_task_space_env

def evaluate(
    model_path: str,
    vecnormalize_path: str,
    n_episodes: int = 5,
):
    print(f"Loading model from {model_path}")
    print(f"Loading normalization info from {vecnormalize_path}")
    
    # 1. 创建环境
    env = make_task_space_env(visualize=True, max_steps=200)
    # 2. 包装为 VecEnv (因为 VecNormalize 需要)
    vec_env = DummyVecEnv([lambda: env])
    
    # 3. 加载 VecNormalize 统计数据
    if os.path.exists(vecnormalize_path):
        vec_env = VecNormalize.load(vecnormalize_path, vec_env)
        vec_env.training = False  # 测试模式，不更新统计数据
        vec_env.norm_reward = False 
    else:
        print("Warning: VecNormalize path not found, running without normalization!")
        
    # 4. 加载模型
    model = PPO.load(model_path)
    
    # 5. 运行评估
    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = False
        step = 0
        total_reward = 0
        
        print(f"\nEpisode {ep+1} started")
        print(f"Target: {vec_env.envs[0].drop_zone_center}")
        
        while not done:
            # 获取动作
            action, _ = model.predict(obs, deterministic=True)
            
            # 执行动作
            obs, reward, done, info = vec_env.step(action)
            total_reward += reward
            step += 1
            
            # 获取真实环境信息 (VecEnv 返回的是列表)
            real_info = info[0]
            dist = real_info.get('dist_to_target', 0.0)
            
            if step % 20 == 0:
                print(f"Step {step}: dist={dist:.4f}")
                
            time.sleep(0.05)  # 减慢显示
            
        print(f"Episode finished. Reward: {total_reward[0]:.2f}")
        print(f"Result: Success={real_info.get('success', False)}")

    vec_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to .zip model file")
    args = parser.parse_args()
    
    # If the user provides a directory, try to find the best model automatically
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
             checkpoints = glob.glob(os.path.join(args.model, "task_space_rl_*_steps.zip"))
             if checkpoints:
                 checkpoints.sort(key=lambda x: int(x.split("_")[-2]))
                 args.model = checkpoints[-1]
                 print(f"Auto-selected latest checkpoint: {args.model}")
             else:
                 print(f"Error: No model found in directory {args.model}")
                 sys.exit(1)

    # 推断 vecnormalize 路径
    if not args.model.endswith(".zip"):
        args.model += ".zip"
        
    norm_path = args.model.replace(".zip", "_vecnormalize.pkl")
    base_dir = os.path.dirname(args.model)
    
    if not os.path.exists(norm_path):
        # 1. 首先尝试常见的 vecnormalize 文件名
        possible_names = [
            "final_model_vecnormalize.pkl",
            "best_model_vecnormalize.pkl",
        ]
        for name in possible_names:
            p = os.path.join(base_dir, name)
            if os.path.exists(p):
                norm_path = p
                break
        else:
            # 2. 如果还没找到，尝试查找最新的 checkpoint vecnormalize
            import glob
            checkpoint_norms = glob.glob(os.path.join(base_dir, "task_space_rl_vecnormalize_*_steps.pkl"))
            if checkpoint_norms:
                # 取最新的 (按步数排序)
                checkpoint_norms.sort(key=lambda x: int(x.split("_")[-2]))
                norm_path = checkpoint_norms[-1]
                print(f"Using checkpoint vecnormalize: {norm_path}")
        
    evaluate(args.model, norm_path)
