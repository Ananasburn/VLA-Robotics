"""
Hybrid Vectorized Environment
混合向量化环境：结合 DummyVecEnv (可视化) 和 SubprocVecEnv (性能)

用途：
- env_0: 在主进程运行，支持 MuJoCo 可视化
- env_1~n: 在子进程运行，提高训练速度
"""

import numpy as np
from typing import List, Callable, Optional, Tuple, Any
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvObs


class HybridVecEnv(VecEnv):
    """
    混合向量化环境
    
    将第一个环境放在主进程（DummyVecEnv）以支持可视化，
    将其余环境放在子进程（SubprocVecEnv）以提高性能。
    """
    
    def __init__(self, env_fns: List[Callable]):
        """
        Args:
            env_fns: 环境工厂函数列表
        """
        if len(env_fns) < 1:
            raise ValueError("At least one environment is required")
            
        # 创建第一个环境（主进程，可视化）
        self.dummy_env = DummyVecEnv([env_fns[0]])
        
        # 创建其余环境（子进程，性能）
        if len(env_fns) > 1:
            self.subprocess_env = SubprocVecEnv(env_fns[1:])
        else:
            self.subprocess_env = None
            
        # 从第一个环境获取空间信息
        dummy = self.dummy_env.envs[0]
        observation_space = dummy.observation_space
        action_space = dummy.action_space
        
        # 初始化父类
        num_envs = len(env_fns)
        super().__init__(num_envs, observation_space, action_space)
        
    def reset(self) -> VecEnvObs:
        """重置所有环境"""
        obs_dummy = self.dummy_env.reset()
        
        if self.subprocess_env is not None:
            obs_subprocess = self.subprocess_env.reset()
            # 合并观测
            obs = np.vstack([obs_dummy, obs_subprocess])
        else:
            obs = obs_dummy
            
        return obs
        
    def step_async(self, actions: np.ndarray) -> None:
        """异步执行动作"""
        # 分割动作
        action_dummy = actions[0:1]
        
        self.dummy_env.step_async(action_dummy)
        
        if self.subprocess_env is not None:
            action_subprocess = actions[1:]
            self.subprocess_env.step_async(action_subprocess)
            
    def step_wait(self) -> VecEnvStepReturn:
        """等待并获取 step 结果"""
        obs_dummy, rewards_dummy, dones_dummy, infos_dummy = self.dummy_env.step_wait()
        
        if self.subprocess_env is not None:
            obs_sub, rewards_sub, dones_sub, infos_sub = self.subprocess_env.step_wait()
            
            # 合并结果
            obs = np.vstack([obs_dummy, obs_sub])
            rewards = np.concatenate([rewards_dummy, rewards_sub])
            dones = np.concatenate([dones_dummy, dones_sub])
            infos = list(infos_dummy) + list(infos_sub)
        else:
            obs = obs_dummy
            rewards = rewards_dummy
            dones = dones_dummy
            infos = infos_dummy
            
        return obs, rewards, dones, infos
        
    def close(self) -> None:
        """关闭所有环境"""
        self.dummy_env.close()
        if self.subprocess_env is not None:
            self.subprocess_env.close()
            
    def get_attr(self, attr_name: str, indices=None):
        """获取环境属性"""
        if indices is None:
            indices = range(self.num_envs)
            
        results = []
        for idx in indices:
            if idx == 0:
                results.append(self.dummy_env.get_attr(attr_name, [0])[0])
            else:
                results.append(self.subprocess_env.get_attr(attr_name, [idx - 1])[0])
        return results
        
    def set_attr(self, attr_name: str, value, indices=None):
        """设置环境属性"""
        if indices is None:
            indices = range(self.num_envs)
            
        for idx in indices:
            if idx == 0:
                self.dummy_env.set_attr(attr_name, value, [0])
            else:
                self.subprocess_env.set_attr(attr_name, value, [idx - 1])
                
    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        """调用环境方法"""
        if indices is None:
            indices = range(self.num_envs)
            
        results = []
        for idx in indices:
            if idx == 0:
                result = self.dummy_env.env_method(method_name, *method_args, indices=[0], **method_kwargs)
                results.append(result[0])
            else:
                result = self.subprocess_env.env_method(method_name, *method_args, indices=[idx - 1], **method_kwargs)
                results.append(result[0])
        return results
        
    def seed(self, seed: Optional[int] = None):
        """设置随机种子"""
        seeds = []
        seeds.extend(self.dummy_env.seed(seed))
        if self.subprocess_env is not None:
            if seed is not None:
                seeds.extend(self.subprocess_env.seed(seed + 1))
            else:
                seeds.extend(self.subprocess_env.seed(None))
        return seeds
        
    def env_is_wrapped(self, wrapper_class, indices=None):
        """检查环境是否被包装"""
        if indices is None:
            indices = range(self.num_envs)
            
        results = []
        for idx in indices:
            if idx == 0:
                results.append(self.dummy_env.env_is_wrapped(wrapper_class, [0])[0])
            else:
                results.append(self.subprocess_env.env_is_wrapped(wrapper_class, [idx - 1])[0])
        return results
