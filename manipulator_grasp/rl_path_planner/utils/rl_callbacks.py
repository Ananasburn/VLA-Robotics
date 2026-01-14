"""
RL Training Callbacks
强化学习训练回调函数

提供训练过程中的自定义回调，包括：
- 成功率记录
- 课程学习调度
- 评估回调
"""

import os
import numpy as np
from typing import Dict, List, Optional, Any

from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)


class SuccessRateCallback(BaseCallback):
    """记录成功率和碰撞率的回调"""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.successes = []
        self.collisions = []
        self.distances = []
        
    def _on_step(self) -> bool:
        # 从 info 中提取数据
        for info in self.locals.get("infos", []):
            if "success" in info:
                self.successes.append(float(info["success"]))
            if "collision" in info:
                self.collisions.append(float(info["collision"]))
            if "dist_to_target" in info:
                self.distances.append(info["dist_to_target"])
            elif "dist_to_goal" in info:
                self.distances.append(info["dist_to_goal"])
                
        # 每 1000 步记录一次
        if self.n_calls % 1000 == 0 and len(self.successes) > 0:
            window = 100
            success_rate = np.mean(self.successes[-window:]) if len(self.successes) >= window else np.mean(self.successes)
            collision_rate = np.mean(self.collisions[-window:]) if len(self.collisions) >= window else np.mean(self.collisions) if self.collisions else 0.0
            avg_dist = np.mean(self.distances[-window:]) if len(self.distances) >= window else np.mean(self.distances) if self.distances else 0.0
            
            self.logger.record("custom/success_rate", success_rate)
            self.logger.record("custom/collision_rate", collision_rate)
            self.logger.record("custom/avg_dist_to_target", avg_dist)
            
            if self.verbose > 0:
                print(f"Step {self.n_calls}: success_rate={success_rate:.3f}, collision_rate={collision_rate:.3f}")
                
        return True


class CurriculumCallback(BaseCallback):
    """课程学习回调"""
    
    def __init__(
        self,
        stages: List[Dict],
        verbose: int = 0
    ):
        """
        Args:
            stages: 课程阶段列表，每个阶段包含 min_sr (最小成功率) 和 level (课程级别)
        """
        super().__init__(verbose)
        self.stages = stages
        self.current_stage = 0
        self.success_history = []
        
    def _on_step(self) -> bool:
        # 收集成功率
        for info in self.locals.get("infos", []):
            if "success" in info:
                self.success_history.append(float(info["success"]))
                
        # 每 5000 步检查是否需要提升课程级别
        if self.n_calls % 5000 == 0 and len(self.success_history) > 100:
            current_sr = np.mean(self.success_history[-100:])
            
            if self.current_stage < len(self.stages) - 1:
                next_stage = self.stages[self.current_stage + 1]
                if current_sr >= next_stage.get("min_sr", 0.5):
                    self.current_stage += 1
                    new_level = next_stage.get("level", self.current_stage)
                    
                    # 更新所有环境的课程级别
                    try:
                        self.training_env.env_method("set_curriculum_level", new_level)
                        if self.verbose > 0:
                            print(f"\n🎓 Curriculum: Advanced to stage {self.current_stage} (level={new_level})")
                    except Exception as e:
                        if self.verbose > 0:
                            print(f"Warning: Could not set curriculum level: {e}")
                            
            self.logger.record("curriculum/stage", self.current_stage)
            self.logger.record("curriculum/success_rate", current_sr)
            
        return True


class BestModelCallback(BaseCallback):
    """保存最佳模型的回调"""
    
    def __init__(
        self,
        save_path: str,
        metric: str = "success_rate",
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.metric = metric
        self.best_value = -np.inf
        self.metric_history = []
        
    def _on_step(self) -> bool:
        # 收集指标
        for info in self.locals.get("infos", []):
            if "success" in info:
                self.metric_history.append(float(info["success"]))
                
        # 每 10000 步检查
        if self.n_calls % 10000 == 0 and len(self.metric_history) > 100:
            current_value = np.mean(self.metric_history[-100:])
            
            if current_value > self.best_value:
                self.best_value = current_value
                save_path = os.path.join(self.save_path, "best_model")
                self.model.save(save_path)
                
                if self.verbose > 0:
                    print(f"\n💾 New best model saved! {self.metric}={current_value:.4f}")
                    
            self.logger.record(f"best/{self.metric}", self.best_value)
            
        return True


def create_training_callbacks(
    log_dir: str,
    model_dir: str,
    eval_env: Optional[Any] = None,
    curriculum_stages: Optional[List[Dict]] = None,
    verbose: int = 1,
) -> List[BaseCallback]:
    """
    创建训练回调列表
    
    Args:
        log_dir: 日志目录
        model_dir: 模型保存目录
        eval_env: 评估环境
        curriculum_stages: 课程学习阶段配置
        verbose: 日志详细程度
        
    Returns:
        回调列表
    """
    callbacks = []
    
    # 成功率记录
    success_callback = SuccessRateCallback(verbose=verbose)
    callbacks.append(success_callback)
    
    # 课程学习
    if curriculum_stages is not None:
        curriculum_callback = CurriculumCallback(curriculum_stages, verbose=verbose)
        callbacks.append(curriculum_callback)
        
    # 最佳模型保存
    best_model_callback = BestModelCallback(model_dir, verbose=verbose)
    callbacks.append(best_model_callback)
    
    # 评估回调
    if eval_env is not None:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=model_dir,
            log_path=log_dir,
            eval_freq=10000,
            deterministic=True,
            render=False,
            n_eval_episodes=10,
        )
        callbacks.append(eval_callback)
        
    return callbacks
