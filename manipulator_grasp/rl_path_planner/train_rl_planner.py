"""
RL Path Planner Training Script
强化学习路径规划器训练脚本

使用Stable-Baselines3的PPO算法训练机器人手臂路径规划策略
支持课程学习、TensorBoard日志、模型检查点等功能
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
from stable_baselines3.common.callbacks import CheckpointCallback

# 添加项目路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from manipulator_grasp.env.rl_path_env import RLPathEnv, make_rl_path_env
from manipulator_grasp.rl_path_planner.utils.rl_callbacks import create_training_callbacks


def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def make_env(
    rank: int,
    seed: int = 0,
    max_steps: int = 500,
    curriculum_level: int = 0,
    randomize_obstacles: bool = False,
):
    """
    创建环境的工厂函数 (用于并行环境)
    
    Args:
        rank: 环境索引
        seed: 随机种子
        max_steps: 最大步数
        curriculum_level: 课程级别
        randomize_obstacles: 是否随机化障碍物
    """
    def _init():
        env = make_rl_path_env(
            render_mode=None,
            max_steps=max_steps,
            curriculum_level=curriculum_level,
            randomize_obstacles=randomize_obstacles,
        )
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def create_vec_env(
    n_envs: int,
    seed: int = 0,
    max_steps: int = 500,
    curriculum_level: int = 0,
    randomize_obstacles: bool = False,
    use_subprocess: bool = True,
) -> VecMonitor:
    """
    创建向量化环境
    
    Args:
        n_envs: 并行环境数量
        seed: 随机种子
        max_steps: 最大步数
        curriculum_level: 课程级别
        randomize_obstacles: 是否随机化障碍物
        use_subprocess: 是否使用子进程并行
        
    Returns:
        向量化环境
    """
    env_fns = [
        make_env(i, seed, max_steps, curriculum_level, randomize_obstacles)
        for i in range(n_envs)
    ]
    
    if use_subprocess and n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)
        
    # 添加Monitor用于统计
    vec_env = VecMonitor(vec_env)
    
    # 添加归一化Wrapper (关键改进: 解决Value Loss过大问题)
    # norm_obs=True: 归一化观测值
    # norm_reward=True: 归一化奖励
    # clip_obs=10.: 裁剪观测值
    # clip_reward=10.: 裁剪奖励
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)
    
    return vec_env


def create_model(
    env,
    config: Dict[str, Any],
    tensorboard_log: str,
) -> PPO:
    """
    创建PPO模型
    
    Args:
        env: 训练环境
        config: 配置字典
        tensorboard_log: TensorBoard日志目录
        
    Returns:
        PPO模型
    """
    ppo_config = config.get("ppo", {})
    policy_config = config.get("policy", {})
    
    # 策略网络架构
    policy_kwargs = {}
    if "net_arch" in policy_config:
        policy_kwargs["net_arch"] = policy_config["net_arch"]
    if "activation_fn" in policy_config:
        activation_fn = policy_config["activation_fn"]
        if activation_fn == "tanh":
            policy_kwargs["activation_fn"] = torch.nn.Tanh
        elif activation_fn == "relu":
            policy_kwargs["activation_fn"] = torch.nn.ReLU
            
    # 设备选择
    device = config.get("training", {}).get("device", "auto")
    
    model = PPO(
        policy=policy_config.get("type", "MlpPolicy"),
        env=env,
        learning_rate=ppo_config.get("learning_rate", 3e-4),
        n_steps=ppo_config.get("n_steps", 2048),
        batch_size=ppo_config.get("batch_size", 64),
        n_epochs=ppo_config.get("n_epochs", 10),
        gamma=ppo_config.get("gamma", 0.99),
        gae_lambda=ppo_config.get("gae_lambda", 0.95),
        clip_range=ppo_config.get("clip_range", 0.2),
        vf_coef=ppo_config.get("vf_coef", 0.5),
        ent_coef=ppo_config.get("ent_coef", 0.01),
        max_grad_norm=ppo_config.get("max_grad_norm", 0.5),
        policy_kwargs=policy_kwargs if policy_kwargs else None,
        tensorboard_log=tensorboard_log,
        verbose=config.get("logging", {}).get("verbose", 1),
        seed=config.get("training", {}).get("seed", None),
        device=device,
    )
    
    return model


def train(
    config_path: str,
    resume_from: Optional[str] = None,
    experiment_name: Optional[str] = None,
):
    """
    主训练函数
    
    Args:
        config_path: 配置文件路径
        resume_from: 继续训练的模型路径 (可选)
        experiment_name: 实验名称 (可选)
    """
    # 加载配置
    config = load_config(config_path)
    
    # 设置实验名称和目录
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    log_config = config.get("logging", {})
    base_log_dir = log_config.get("log_dir", "manipulator_grasp/rl_path_planner/logs")
    base_model_dir = log_config.get("model_dir", "manipulator_grasp/rl_path_planner/models")
    
    log_dir = os.path.join(ROOT_DIR, base_log_dir, experiment_name)
    model_dir = os.path.join(ROOT_DIR, base_model_dir, experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存配置副本
    config_save_path = os.path.join(log_dir, "config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
        
    print(f"\n{'='*60}")
    print(f"🚀 RL Path Planner Training")
    print(f"{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"Log dir: {log_dir}")
    print(f"Model dir: {model_dir}")
    print(f"{'='*60}\n")
    
    # 获取训练参数
    train_config = config.get("training", {})
    env_config = config.get("environment", {})
    curriculum_config = config.get("curriculum", {})
    
    total_timesteps = train_config.get("total_timesteps", 1000000)
    n_envs = train_config.get("n_envs", 4)
    seed = train_config.get("seed", 42)
    
    # 创建训练环境
    print("Creating training environment...")
    train_env = create_vec_env(
        n_envs=n_envs,
        seed=seed,
        max_steps=env_config.get("max_steps", 500),
        curriculum_level=0,  # 从Level 0开始
        randomize_obstacles=env_config.get("randomize_obstacles", False),
        use_subprocess=n_envs > 1,
    )
    
    # 创建评估环境
    print("Creating evaluation environment...")
    eval_env = create_vec_env(
        n_envs=1,
        seed=seed + 1000,
        max_steps=env_config.get("max_steps", 500),
        curriculum_level=0,
        randomize_obstacles=False,
        use_subprocess=False,
    )
    
    # 创建或加载模型
    if resume_from is not None:
        print(f"Resuming from: {resume_from}")
        print(f"Resuming from: {resume_from}")
        model = PPO.load(resume_from, env=train_env)
        
        # 尝试加载归一化统计数据 (如果存在)
        norm_path = resume_from.replace(".zip", "_vecnormalize.pkl")
        if os.path.exists(norm_path):
            print(f"Loading VecNormalize stats from: {norm_path}")
            train_env = VecNormalize.load(norm_path, train_env.venv)  # 注意需要传入venv
        else:
            print("Warning: No VecNormalize stats found, starting with fresh normalization stats")
    else:
        print("Creating new model...")
        model = create_model(
            train_env,
            config,
            tensorboard_log=log_dir,
        )
        
    # 创建回调
    print("Setting up callbacks...")
    curriculum_stages = curriculum_config.get("stages", None)
    callbacks = create_training_callbacks(
        log_dir=log_dir,
        model_dir=model_dir,
        eval_env=eval_env,
        curriculum_stages=curriculum_stages,
        verbose=log_config.get("verbose", 1),
    )
    
    # 添加检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=train_config.get("save_freq", 50000) // n_envs,
        save_path=model_dir,
        name_prefix="rl_path_planner",
        save_replay_buffer=log_config.get("save_replay_buffer", False),
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    
    # 开始训练
    print(f"\n🏋️ Starting training for {total_timesteps:,} timesteps...")
    print(f"   Using {n_envs} parallel environments")
    print(f"   Curriculum learning: {'enabled' if curriculum_config.get('enabled', True) else 'disabled'}")
    print()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=resume_from is None,
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        
    # 保存最终模型
    final_model_path = os.path.join(model_dir, "final_model")
    model.save(final_model_path)
    
    # 保存归一化统计数据
    train_env.save(os.path.join(model_dir, "final_model_vecnormalize.pkl"))
    print(f"\n💾 Final model saved to: {final_model_path}.zip")
    
    # 清理
    train_env.close()
    eval_env.close()
    
    print(f"\n{'='*60}")
    print(f"✅ Training completed!")
    print(f"   Models saved in: {model_dir}")
    print(f"   Logs saved in: {log_dir}")
    print(f"   Run 'tensorboard --logdir {log_dir}' to view training curves")
    print(f"{'='*60}\n")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train RL Path Planner")
    parser.add_argument(
        "--config",
        type=str,
        default="manipulator_grasp/rl_path_planner/configs/rl_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to model to resume training from"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (default: timestamp)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total timesteps from config"
    )
    
    args = parser.parse_args()
    
    # 处理配置路径
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(ROOT_DIR, config_path)
        
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
        
    # 覆盖配置参数
    if args.timesteps is not None:
        config = load_config(config_path)
        config["training"]["total_timesteps"] = args.timesteps
        # 临时保存修改后的配置
        temp_config_path = "/tmp/rl_config_temp.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        config_path = temp_config_path
        
    # 开始训练
    train(config_path, args.resume, args.name)


if __name__ == "__main__":
    main()
