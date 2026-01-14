"""
Task-Space RL Path Planner Training Script
训练机械臂末端从 pickup zone 到 drop zone 的路径规划

使用新的 RLTaskSpaceEnv 环境
"""

import os
import sys
import argparse
import yaml
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
    BaseCallback,
)

# 添加项目路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from manipulator_grasp.env.rl_task_space_env import RLTaskSpaceEnv, make_task_space_env
from manipulator_grasp.rl_path_planner.utils.hybrid_vec_env import HybridVecEnv


class SuccessRateCallback(BaseCallback):
    """记录成功率的回调"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.successes = []
        self.distances = []
        
    def _on_step(self) -> bool:
        # 检查是否有 episode 结束
        for info in self.locals.get("infos", []):
            if "success" in info:
                self.successes.append(float(info["success"]))
            if "dist_to_target" in info:
                self.distances.append(info["dist_to_target"])
                
        # 每 1000 步记录一次
        if self.n_calls % 1000 == 0 and len(self.successes) > 0:
            success_rate = np.mean(self.successes[-100:]) if len(self.successes) >= 100 else np.mean(self.successes)
            avg_dist = np.mean(self.distances[-100:]) if len(self.distances) >= 100 else np.mean(self.distances)
            
            self.logger.record("custom/success_rate", success_rate)
            self.logger.record("custom/avg_dist_to_target", avg_dist)
            
        return True


def make_env(
    rank: int,
    seed: int = 0,
    max_steps: int = 200,
    visualize: bool = False,
):
    """环境工厂函数"""
    def _init():
        # 只有第一个环境进行可视化
        viz = visualize and (rank == 0)
        env = make_task_space_env(
            max_steps=max_steps,
            visualize=viz,
        )
        env.reset(seed=seed + rank)
        return env
        
    set_random_seed(seed)
    return _init


def create_vec_env(
    n_envs: int,
    seed: int = 0,
    max_steps: int = 200,
    visualize: bool = False,
    use_subprocess: bool = True,
) -> VecNormalize:
    """创建向量化环境"""
    
    env_fns = [
        make_env(i, seed, max_steps, visualize)
        for i in range(n_envs)
    ]
    
    # 混合模式：env_0 可视化 + 其余子进程（最快）
    if visualize and n_envs > 1 and use_subprocess:
        print(f"🚀 Using HybridVecEnv: env_0 visualized + {n_envs-1} in subprocesses")
        vec_env = HybridVecEnv(env_fns)
    # 纯可视化模式：所有env在主进程（较慢）
    elif visualize or n_envs == 1:
        vec_env = DummyVecEnv(env_fns)
    # 纯性能模式：所有env在子进程（无可视化）
    else:
        vec_env = SubprocVecEnv(env_fns)
        
    # 添加 Monitor
    vec_env = VecMonitor(vec_env)
    
    # 添加归一化
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )
    
    return vec_env


def train(
    experiment_name: str = "task_space_v1",
    total_timesteps: int = 1_000_000,
    n_envs: int = 8,
    max_steps: int = 200,
    visualize: bool = False,
    resume_from: Optional[str] = None,
):
    """主训练函数"""
    
    # 设置目录
    log_dir = os.path.join(ROOT_DIR, "manipulator_grasp/rl_path_planner/logs", experiment_name)
    model_dir = os.path.join(ROOT_DIR, "manipulator_grasp/rl_path_planner/models", experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🚀 Task-Space RL Path Planner Training")
    print(f"{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"Log dir: {log_dir}")
    print(f"Model dir: {model_dir}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"Visualization: {visualize}")
    print(f"{'='*60}\n")
    
    # 创建训练环境
    print("Creating training environment...")
    train_env = create_vec_env(
        n_envs=n_envs,
        seed=42,
        max_steps=max_steps,
        visualize=visualize,
        use_subprocess=not visualize,
    )
    
    # 创建评估环境
    print("Creating evaluation environment...")
    eval_env = create_vec_env(
        n_envs=1,
        seed=1000,
        max_steps=max_steps,
        visualize=False,
        use_subprocess=False,
    )
    
    # 创建或加载模型
    if resume_from is not None and os.path.exists(resume_from):
        print(f"Resuming from: {resume_from}")
        model = PPO.load(resume_from, env=train_env, tensorboard_log=log_dir)
        
        # 加载归一化统计
        norm_path = resume_from.replace(".zip", "_vecnormalize.pkl")
        if os.path.exists(norm_path):
            train_env = VecNormalize.load(norm_path, train_env.venv)
    else:
        print("Creating new model...")
        
        # PPO 超参数
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
            activation_fn=torch.nn.ReLU,
        )
        
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            vf_coef=0.5,
            ent_coef=0.005,  # 小一点的熵系数
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_dir,
            verbose=1,
            device="cpu", #  if torch.cuda.is_available() else "cpu",
        )
        
    # 设置回调
    print("Setting up callbacks...")
    
    callbacks = []
    
    # 成功率回调
    success_callback = SuccessRateCallback(verbose=1)
    callbacks.append(success_callback)
    
    # 评估回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=10000 // n_envs,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )
    callbacks.append(eval_callback)
    
    # 检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // n_envs,
        save_path=model_dir,
        name_prefix="task_space_rl",
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    
    # 开始训练
    print(f"\n🏋️ Starting training for {total_timesteps:,} timesteps...")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks),
            progress_bar=True,
            reset_num_timesteps=resume_from is None,
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        
    # 保存最终模型
    final_model_path = os.path.join(model_dir, "final_model")
    model.save(final_model_path)
    train_env.save(os.path.join(model_dir, "final_model_vecnormalize.pkl"))
    
    print(f"\n💾 Final model saved to: {final_model_path}.zip")
    
    # 清理
    train_env.close()
    eval_env.close()
    
    print(f"\n{'='*60}")
    print(f"✅ Training completed!")
    print(f"   Run 'tensorboard --logdir {log_dir}' to view curves")
    print(f"{'='*60}\n")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Task-Space RL Path Planner")
    parser.add_argument(
        "--name", type=str, default="task_space_v1",
        help="Experiment name"
    )
    parser.add_argument(
        "--timesteps", type=int, default=1_000_000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--envs", type=int, default=8,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--max-steps", type=int, default=200,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Enable MuJoCo visualization during training"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to model to resume from"
    )
    
    args = parser.parse_args()
    
    train(
        experiment_name=args.name,
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        max_steps=args.max_steps,
        visualize=args.visualize,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
