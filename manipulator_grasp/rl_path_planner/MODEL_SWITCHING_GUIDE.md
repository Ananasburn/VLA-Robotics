# RL Path Planner - 模型切换指南

本指南说明如何优雅地选择和加载不同的训练模型。

## 🎯 快速开始

### 1. 使用默认最佳模型（推荐）

最简单的方式是什么都不改，系统会自动加载 `model_config.py` 中配置的默认模型：

```bash
python3 main_vlm.py --planner rl_ppo
```

### 2. 切换到不同的训练模型

有三种方式可以切换模型：

#### 方法 A：修改配置文件（推荐用于永久切换）

编辑 `manipulator_grasp/rl_path_planner/model_config.py`：

```python
PLACE_PHASE_CONFIG = {
    # 选项 1: 使用 best_model （推荐）
    'model_path': os.path.join(MODELS_DIR, 'place_with_object_v1', 'best_model.zip'),
    'vecnormalize_path': os.path.join(MODELS_DIR, 'place_with_object_v1', 'best_model_vecnormalize.pkl'),
    
    # 选项 2: 使用 final_model
    # 'model_path': os.path.join(MODELS_DIR, 'place_with_object_v1', 'final_model.zip'),
    # 'vecnormalize_path': os.path.join(MODELS_DIR, 'place_with_object_v1', 'final_model_vecnormalize.pkl'),
    
    # 选项 3: 使用特定 checkpoint
    # 'model_path': os.path.join(MODELS_DIR, 'place_with_object_v1', 'place_phase_2500000_steps.zip'),
    # 'vecnormalize_path': os.path.join(MODELS_DIR, 'place_with_object_v1', 'place_phase_vecnormalize_2500000_steps.pkl'),
}
```

#### 方法 B：使用环境变量（推荐用于临时切换）

```bash
# 指定模型和 VecNormalize 路径
export RL_PLACE_MODEL=/path/to/your/model.zip
export RL_PLACE_VECNORM=/path/to/your/vecnormalize.pkl

# 运行
python3 main_vlm.py --planner rl_ppo
```

#### 方法 C：在代码中直接指定（高级用法）

如果你需要在代码中动态切换模型：

```python
from manipulator_grasp.rl_path_planner.rl_integration import get_rl_planner

# 使用自定义路径
rl_planner = get_rl_planner(
    model_path='/path/to/your/model.zip',
    phase='place'
)
```

---

## 📁 模型文件组织

推荐的目录结构：

```
manipulator_grasp/rl_path_planner/
├── models/                          # 所有训练模型的默认位置
│   ├── place_with_object_v1/        # Place phase 模型版本 1
│   │   ├── best_model.zip           # 最佳模型（基于 eval success rate）
│   │   ├── best_model_vecnormalize.pkl
│   │   ├── final_model.zip          # 最终模型（训练结束时）
│   │   ├── final_model_vecnormalize.pkl
│   │   └── checkpoints/             # 中间 checkpoint (可选)
│   ├── place_with_object_v2/        # 另一个训练版本
│   └── task_space_v5_8_collision_check/  # 旧版 approach 模型
└── model_config.py                  # 配置文件
```

或者使用 `logs/` 目录：

```
logs/
├── place_with_object_v1/
│   ├── best_model.zip
│   └── best_model_vecnormalize.pkl
└── place_with_object_v2/
    └── ...
```

---

## 🔧 配置说明

### `model_config.py` 配置项

```python
PLACE_PHASE_CONFIG = {
    'model_path': str,              # .zip 模型文件路径
    'vecnormalize_path': str,       # .pkl 归一化统计文件路径
    'drop_zone_center': [x, y, z],  # 训练时使用的目标位置
    'success_threshold': float,     # 成功判定阈值（米）
    'max_steps': int,               # 最大步数
}
```

### 自动检测功能

如果配置的路径不存在，系统会自动在 `logs/` 目录下搜索包含 `place` 的目录，并尝试加载其中的 `best_model.zip`。

---

## 📊 模型选择建议

| 模型类型 | 何时使用 | 特点 |
|---------|---------|------|
| `best_model.zip` | **生产环境（推荐）** | 基于评估成功率选出的最佳模型 |
| `final_model.zip` | 测试/对比 | 训练结束时的模型（可能不是最佳） |
| `checkpoint_*.zip` | 调试/回滚 | 特定训练步数的模型（用于调试） |

---

## 🚀 完整使用示例

### 示例 1：使用默认最佳模型

```bash
# 1. 确保 model_config.py 配置正确（通常已经默认配置好）
# 2. 直接运行
python3 main_vlm.py --planner rl_ppo
```

### 示例 2：快速切换到另一个训练版本

```bash
# 使用环境变量临时切换
export RL_PLACE_MODEL=logs/place_with_object_v2/best_model.zip
export RL_PLACE_VECNORM=logs/place_with_object_v2/best_model_vecnormalize.pkl

python3 main_vlm.py --planner rl_ppo
```

### 示例 3：对比不同模型

```bash
# 测试 v1
python3 main_vlm.py --planner rl_ppo  # 使用默认 v1

# 测试 v2 (使用环境变量)
RL_PLACE_MODEL=logs/place_with_object_v2/best_model.zip \
RL_PLACE_VECNORM=logs/place_with_object_v2/best_model_vecnormalize.pkl \
python3 main_vlm.py --planner rl_ppo
```

---

## ❓ 常见问题

### Q: 如何知道当前使用的是哪个模型？

A: 运行时会打印日志：
```
[RL Planner] Creating new planner instance for phase=place
[RL Planner] Loading model from /path/to/model.zip
[execute_grasp] Using RL PPO planner for place phase
[execute_grasp] Model target: [0.6 0.2 0.83]
```

### Q: 模型文件放在哪里？

A: 优先级顺序：
1. `model_config.py` 中配置的路径
2. 环境变量 `RL_PLACE_MODEL`
3. `logs/` 目录下的自动检测

### Q: 如何使用 RRT-Connect（禁用 RL）？

A: 不传 `--planner` 参数，或显式指定：
```bash
python3 main_vlm.py --planner rrtconnect
```

### Q: 可以同时使用多个不同的模型吗？

A: 是的！系统会为每个 phase 和自定义路径分别缓存模型实例。例如：
```python
place_planner_v1 = get_rl_planner(phase='place')  # 使用默认配置
place_planner_v2 = get_rl_planner('/path/to/v2.zip')  # 使用自定义路径
```

---

## 📝 训练新模型后的集成步骤

当你完成一次新的训练后：

1. **复制模型文件到 models 目录**：
   ```bash
   mkdir -p manipulator_grasp/rl_path_planner/models/place_with_object_v2
   cp logs/place_with_object_v2/best_model.zip manipulator_grasp/rl_path_planner/models/place_with_object_v2/
   cp logs/place_with_object_v2/best_model_vecnormalize.pkl manipulator_grasp/rl_path_planner/models/place_with_object_v2/
   ```

2. **更新 `model_config.py`**：
   ```python
   PLACE_PHASE_CONFIG = {
       'model_path': os.path.join(MODELS_DIR, 'place_with_object_v2', 'best_model.zip'),
       'vecnormalize_path': os.path.join(MODELS_DIR, 'place_with_object_v2', 'best_model_vecnormalize.pkl'),
       # ... other config
   }
   ```

3. **测试**：
   ```bash
   python3 main_vlm.py --planner rl_ppo
   ```

---

## ✅ 验证模型加载

运行评估脚本确认模型正常加载：

```bash
python3 manipulator_grasp/rl_path_planner/evaluate_place_phase.py \
    --model logs/place_with_object_v1 \
    --episodes 10
```

---

**就这么简单！** 现在你可以轻松地在不同的训练模型之间切换了。🎉
