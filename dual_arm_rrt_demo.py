"""
双臂系统采样规划演示 - Dual-Arm Sampling-Based Planning Demonstration
场景 S1: 狭窄通道中的换手/交接

实现特性:
- RRT-Connect 双向采样规划
- 自适应扩展步长和边缘检查
- 目标偏置 + 任务空间偏置采样
- MuJoCo 碰撞检测(两级检查)
- Shortcut平滑
- 20种子统计评估
"""

import numpy as np
import mujoco
import mujoco.viewer
from dataclasses import dataclass
from typing import List, Optional, Tuple
import time
import random

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PlannerConfig:
    """RRT-Connect planner configuration"""
    # Joint limits (12 DOF: 6 left + 6 right arm)
    q_min: np.ndarray
    q_max: np.ndarray
    
    # Planning parameters    
    base_step_size: float = 0.15          # Base step size in radians
    adaptive_step: bool = True             # Enable adaptive step sizing
    min_step_size: float = 0.05            # Min step when near obstacles
    max_step_size: float = 0.3             # Max step when clear
    
    # Sampling
    goal_bias: float = 0.1                 # Probability of sampling goal
    task_space_bias: float = 0.2           # Probability of task-space biased sample
    handoff_zone: np.ndarray = None        # [x, y, z] handoff target position
    
    # Tree connection
    max_iters: int = 5000                  # Maximum iterations
    connection_threshold: float = 0.3      # Threshold for tree connection (rad)
    
    # Edge checking
    base_edge_resolution: float = 0.05     # Base resolution for edge checking
    adaptive_edge: bool = True             # Adaptive edge check resolution
    
    # Random seed
    seed: int = 42
    
    # Collision checking
    cheap_check_threshold: float = 0.1     # Distance for cheap collision check
    

@dataclass
class Node:
    """RRT tree node"""
    q: np.ndarray                          # Joint configuration
    parent: Optional['Node'] = None        # Parent node
    cost: float = 0.0                      # Cost from start

# ============================================================================
# Scene Setup
# ============================================================================

def build_scene() -> Tuple[mujoco.MjModel, str]:
    """
    构建场景S1: 狭窄通道 + 换手区域
    
    Returns:
        model: MuJoCo model
        xml_path: Path to generated XML
    """
    xml_content = """
<mujoco model="dual_arm_handoff_s1">
    <compiler angle="radian" autolimits="true"/>
    <option timestep="0.002" integrator="implicitfast"/>
    
    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/>
        <rgba haze="0.15 0.25 0.35 1"/>
        <global azimuth="180" elevation="-20"/>
    </visual>
    
    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" width="300" height="300"/>
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5"/>
        <material name="table_mat" rgba="0.6 0.4 0.2 1"/>
        <material name="passage_mat" rgba="0.3 0.3 0.3 1"/>
        <material name="target_mat" rgba="0.2 0.8 0.2 0.3"/>
    </asset>
    
    <worldbody>
        <light pos="0.8 0 2.5" dir="0 0 -1" directional="true"/>
        <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
        
        <!-- Table -->
        <body name="table" pos="0.8 0.0 0.69">
            <geom name="table_top" type="box" size="0.8 0.6 0.05" material="table_mat"/>
        </body>
        
        <!-- Narrow Passage (3 boxes forming gate) -->
        <body name="passage_left" pos="0.8 -0.05 0.85">
            <geom name="gate_left" type="box" size="0.4 0.02 0.08" material="passage_mat"/>
        </body>
        <body name="passage_right" pos="0.8 0.05 0.85">
            <geom name="gate_right" type="box" size="0.4 0.02 0.08" material="passage_mat"/>
        </body>
        <body name="passage_top" pos="0.8 0.0 0.95">
            <geom name="gate_top" type="box" size="0.4 0.05 0.02" material="passage_mat"/>
        </body>
        
        <!-- Handoff Zone Marker (visual only) -->
        <body name="handoff_marker" pos="0.8 0.0 0.85">
            <geom name="handoff_vis" type="sphere" size="0.03" rgba="1 0 0 0.5" contype="0" conaffinity="0"/>
        </body>
        
        <!-- Target Zone (right side) -->
        <body name="target_zone" pos="0.8 -0.4 0.75">
            <geom name="target_vis" type="cylinder" size="0.08 0.01" rgba="0.2 0.8 0.2 0.3" contype="0" conaffinity="0"/>
        </body>
        
        <!-- Objects (apple, banana) on left -->
        <body name="apple" pos="0.6 0.3 0.76" quat="1 0 0 0">
            <geom name="apple_geom" type="sphere" size="0.03" rgba="1 0 0 1" mass="0.05"/>
            <joint name="apple_free" type="free"/>
        </body>
        <body name="banana" pos="0.6 0.35 0.76" quat="1 0 0 0">
            <geom name="banana_geom" type="box" size="0.04 0.015 0.015" rgba="1 1 0 1" mass="0.05"/>
            <joint name="banana_free" type="free"/>
        </body>
        
        <!-- LEFT ARM (UR3e-like, simplified) -->
        <body name="left_base" pos="0.8 0.3 0.745">
            <geom name="left_base_geom" type="cylinder" size="0.05 0.08" rgba="0.2 0.2 0.2 1"/>
            
            <body name="left_link1" pos="0 0 0.08">
                <joint name="left_j1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
                <geom name="left_l1" type="capsule" size="0.04" fromto="0 0 0 0 0 0.15" rgba="0.4 0.6 0.8 1"/>
                
                <body name="left_link2" pos="0 0 0.15">
                    <joint name="left_j2" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
                    <geom name="left_l2" type="capsule" size="0.035" fromto="0 0 0 0 0 0.24" rgba="0.4 0.6 0.8 1"/>
                    
                    <body name="left_link3" pos="0 0 0.24">
                        <joint name="left_j3" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
                        <geom name="left_l3" type="capsule" size="0.03" fromto="0 0 0 0 0 0.21" rgba="0.4 0.6 0.8 1"/>
                        
                        <body name="left_link4" pos="0 0 0.21">
                            <joint name="left_j4" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
                            <geom name="left_l4" type="capsule" size="0.025" fromto="0 0 0 0 0 0.08" rgba="0.4 0.6 0.8 1"/>
                            
                            <body name="left_link5" pos="0 0 0.08">
                                <joint name="left_j5" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
                                <geom name="left_l5" type="capsule" size="0.025" fromto="0 0 0 0 0 0.08" rgba="0.4 0.6 0.8 1"/>
                                
                                <body name="left_link6" pos="0 0 0.08">
                                    <joint name="left_j6" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
                                    <geom name="left_ee" type="cylinder" size="0.03 0.02" rgba="0.8 0.4 0.4 1"/>
                                    <site name="left_ee_site" pos="0 0 0.02" size="0.01"/>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>
        
        <!-- RIGHT ARM (UR3e-like, simplified) -->
        <body name="right_base" pos="0.8 -0.3 0.745">
            <geom name="right_base_geom" type="cylinder" size="0.05 0.08" rgba="0.2 0.2 0.2 1"/>
            
            <body name="right_link1" pos="0 0 0.08">
                <joint name="right_j1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
                <geom name="right_l1" type="capsule" size="0.04" fromto="0 0 0 0 0 0.15" rgba="0.6 0.8 0.4 1"/>
                
                <body name="right_link2" pos="0 0 0.15">
                    <joint name="right_j2" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
                    <geom name="right_l2" type="capsule" size="0.035" fromto="0 0 0 0 0 0.24" rgba="0.6 0.8 0.4 1"/>
                    
                    <body name="right_link3" pos="0 0 0.24">
                        <joint name="right_j3" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
                        <geom name="right_l3" type="capsule" size="0.03" fromto="0 0 0 0 0 0.21" rgba="0.6 0.8 0.4 1"/>
                        
                        <body name="right_link4" pos="0 0 0.21">
                            <joint name="right_j4" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
                            <geom name="right_l4" type="capsule" size="0.025" fromto="0 0 0 0 0 0.08" rgba="0.6 0.8 0.4 1"/>
                            
                            <body name="right_link5" pos="0 0 0.08">
                                <joint name="right_j5" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
                                <geom name="right_l5" type="capsule" size="0.025" fromto="0 0 0 0 0 0.08" rgba="0.6 0.8 0.4 1"/>
                                
                                <body name="right_link6" pos="0 0 0.08">
                                    <joint name="right_j6" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
                                    <geom name="right_ee" type="cylinder" size="0.03 0.02" rgba="0.8 0.4 0.4 1"/>
                                    <site name="right_ee_site" pos="0 0 0.02" size="0.01"/>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>
    
    <actuator>
        <!-- Left arm actuators -->
        <position name="left_a1" joint="left_j1" kp="100"/>
        <position name="left_a2" joint="left_j2" kp="100"/>
        <position name="left_a3" joint="left_j3" kp="100"/>
        <position name="left_a4" joint="left_j4" kp="100"/>
        <position name="left_a5" joint="left_j5" kp="100"/>
        <position name="left_a6" joint="left_j6" kp="100"/>
        
        <!-- Right arm actuators -->
        <position name="right_a1" joint="right_j1" kp="100"/>
        <position name="right_a2" joint="right_j2" kp="100"/>
        <position name="right_a3" joint="right_j3" kp="100"/>
        <position name="right_a4" joint="right_j4" kp="100"/>
        <position name="right_a5" joint="right_j5" kp="100"/>
        <position name="right_a6" joint="right_j6" kp="100"/>
    </actuator>
</mujoco>
"""
    
    # Save to temp file
    import tempfile
    import os
    
    temp_dir = tempfile.gettempdir()
    xml_path = os.path.join(temp_dir, "dual_arm_s1.xml")
    
    with open(xml_path, 'w') as f:
        f.write(xml_content)
    
    model = mujoco.MjModel.from_xml_path(xml_path)
    
    return model, xml_path


# ============================================================================
# Collision Detection (Two-Level)
# ============================================================================

class CollisionChecker:
    """两级碰撞检测: cheap check + precise MuJoCo contact check"""
    
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)
        
        # Statistics
        self.num_cheap_checks = 0
        self.num_precise_checks = 0
        self.num_collisions = 0
        
        # Identify important geom IDs for cheap check
        self.left_geom_ids = []
        self.right_geom_ids = []
        self.env_geom_ids = []
        
        for i in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and 'left_' in name:
                self.left_geom_ids.append(i)
            elif name and 'right_' in name:
                self.right_geom_ids.append(i)
            elif name and ('gate_' in name or 'table' in name):
                self.env_geom_ids.append(i)
    
    def cheap_check(self, q: np.ndarray, threshold: float = 0.1) -> bool:
        """
        快速碰撞检查: 使用关键几何体的距离阈值
        
        Returns:
            True if potential collision (needs precise check)
        """
        self.num_cheap_checks += 1
        
        # Set configuration
        self.data.qpos[:12] = q
        mujoco.mj_forward(self.model, self.data)
        
        # Check distances between left and right arm geoms
        for left_id in self.left_geom_ids[:3]:  # Check first 3 links only
            for right_id in self.right_geom_ids[:3]:
                left_pos = self.data.geom_xpos[left_id]
                right_pos = self.data.geom_xpos[right_id]
                dist = np.linalg.norm(left_pos - right_pos)
                
                if dist < threshold:
                    return True  # Potential collision
        
        return False  # Likely safe
    
    def precise_check(self, q: np.ndarray) -> bool:
        """
        精确碰撞检测: 使用 MuJoCo contact
        
        Returns:
            True if collision detected
        """
        self.num_precise_checks += 1
        
        # Set configuration
        self.data.qpos[:12] = q
        mujoco.mj_forward(self.model, self.data)
        
        # Check for contacts
        if self.data.ncon > 0:
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                
                # Get geom names
                geom1_id = contact.geom1
                geom2_id = contact.geom2
                
                geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id) or ""
                geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id) or ""
                
                # Check for meaningful collisions
                # Left-Right arm collision
                if ('left_' in geom1_name and 'right_' in geom2_name) or \
                   ('right_' in geom1_name and 'left_' in geom2_name):
                    self.num_collisions += 1
                    return True
                
                # Arm-Environment collision (gate, table)
                if ('left_' in geom1_name or 'right_' in geom1_name) and \
                   ('gate_' in geom2_name or 'table' in geom2_name):
                    self.num_collisions += 1
                    return True
                
                if ('left_' in geom2_name or 'right_' in geom2_name) and \
                   ('gate_' in geom1_name or 'table' in geom1_name):
                    self.num_collisions += 1
                    return True
        
        return False
    
    def check_collision(self, q: np.ndarray, use_cheap: bool = True) -> bool:
        """
        两级检测策略
        
        Returns:
            True if collision
        """
        if use_cheap:
            if not self.cheap_check(q, threshold=0.15):
                return False  # Cheap check passed, assume safe
        
        # Do precise check
        return self.precise_check(q)
    
    def reset_stats(self):
        """Reset statistics"""
        self.num_cheap_checks = 0
        self.num_precise_checks = 0
        self.num_collisions = 0


# ============================================================================
# RRT-Connect Implementation
# ============================================================================

class RRTConnect:
    """RRT-Connect with adaptive sampling and step sizing"""
    
    def __init__(self, config: PlannerConfig, collision_checker: CollisionChecker):
        self.config = config
        self.collision_checker = collision_checker
        
        # Trees
        self.tree_start: List[Node] = []
        self.tree_goal: List[Node] = []
        
        # Statistics
        self.num_samples = 0
        self.planning_time = 0.0
        self.path_found = False
        
        # Seed random
        np.random.seed(config.seed)
        random.seed(config.seed)
    
    def sample(self) -> np.ndarray:
        """
        采样策略: 均匀 + goal bias + task-space bias
        """
        self.num_samples += 1
        
        # Goal biasing
        if np.random.rand() < self.config.goal_bias:
            return self.config.q_max  # Sample goal
        
        # Task-space biasing (向换手区域偏置)
        if self.config.task_space_bias > 0 and np.random.rand() < self.config.task_space_bias:
            # Sample and check if end-effector is near handoff zone
            max_attempts = 10
            for _ in range(max_attempts):
                q_sample = self._uniform_sample()
                
                # Compute end-effector position (simplified FK)
                ee_pos = self._compute_ee_position(q_sample)
                
                # Accept if reasonably close to handoff zone
                if self.config.handoff_zone is not None:
                    dist = np.linalg.norm(ee_pos - self.config.handoff_zone)
                    if dist < 0.3:  # Within 30cm
                        return q_sample
            
            # Fallback to uniform
            return self._uniform_sample()
        
        # Uniform sampling
        return self._uniform_sample()
    
    def _uniform_sample(self) -> np.ndarray:
        """均匀随机采样"""
        return np.random.uniform(self.config.q_min, self.config.q_max)
    
    def _compute_ee_position(self, q: np.ndarray) -> np.ndarray:
        """
        简化的末端位置计算 (用MuJoCo forward kinematics)
        只计算左臂末端 (用于task-space bias)
        """
        self.collision_checker.data.qpos[:12] = q
        mujoco.mj_forward(self.collision_checker.model, self.collision_checker.data)
        
        # Get left EE site position
        site_id = mujoco.mj_name2id(self.collision_checker.model, mujoco.mjtObj.mjOBJ_SITE, "left_ee_site")
        if site_id >= 0:
            return self.collision_checker.data.site_xpos[site_id].copy()
        
        return np.zeros(3)
    
    def nearest(self, tree: List[Node], q: np.ndarray) -> Node:
        """找最近节点 (线性搜索)"""
        min_dist = float('inf')
        nearest_node = tree[0]
        
        for node in tree:
            dist = np.linalg.norm(node.q - q)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        
        return nearest_node
    
    def adaptive_step_size(self, q_from: np.ndarray) -> float:
        """
        自适应步长: 离障碍远则大步,近则小步
        """
        if not self.config.adaptive_step:
            return self.config.base_step_size
        
        # Check clearance (cheap)
        if self.collision_checker.cheap_check(q_from, threshold=0.2):
            return self.config.min_step_size  # Near obstacle
        else:
            return self.config.max_step_size  # Clear space
    
    def extend(self, tree: List[Node], q_target: np.ndarray) -> Tuple[str, Optional[Node]]:
        """
        扩展树向目标
        
        Returns:
            status: 'reached', 'advanced', 'trapped'
            new_node: 新节点 (if advanced/reached)
        """
        q_near_node = self.nearest(tree, q_target)
        q_near = q_near_node.q
        
        # Direction and distance
        direction = q_target - q_near
        dist = np.linalg.norm(direction)
        
        if dist < 1e-6:
            return 'reached', q_near_node
        
        direction = direction / dist
        
        # Adaptive step
        step_size = self.adaptive_step_size(q_near)
        
        # Determine step
        if dist <= step_size:
            q_new = q_target
            status = 'reached'
        else:
            q_new = q_near + direction * step_size
            status = 'advanced'
        
        # Edge validity check
        if not self._edge_valid(q_near, q_new):
            return 'trapped', None
        
        # Create new node
        new_node = Node(q=q_new, parent=q_near_node, cost=q_near_node.cost + step_size)
        tree.append(new_node)
        
        return status, new_node
    
    def connect(self, tree: List[Node], q_target: np.ndarray) -> Tuple[str, Optional[Node]]:
        """
        连接到目标 (重复extend直到reached或trapped)
        """
        status = 'advanced'
        last_node = None
        
        max_steps = 50  # Prevent infinite loop
        steps = 0
        
        while status == 'advanced' and steps < max_steps:
            status, last_node = self.extend(tree, q_target)
            steps += 1
        
        return status, last_node
    
    def _edge_valid(self, q_from: np.ndarray, q_to: np.ndarray) -> bool:
        """
        边缘有效性检查: 自适应插值碰撞检测
        """
        dist = np.linalg.norm(q_to - q_from)
        
        # Adaptive resolution
        if self.config.adaptive_edge:
            # More checks near obstacles
            if self.collision_checker.cheap_check(q_from, threshold=0.2):
                resolution = self.config.base_edge_resolution / 2
            else:
                resolution = self.config.base_edge_resolution
        else:
            resolution = self.config.base_edge_resolution
        
        num_checks = int(np.ceil(dist / resolution)) + 1
        
        for i in range(num_checks):
            alpha = i / max(num_checks - 1, 1)
            q_check = q_from + alpha * (q_to - q_from)
            
            if self.collision_checker.check_collision(q_check):
                return False
        
        return True
    
    def trees_can_connect(self) -> Optional[Tuple[Node, Node]]:
        """
        检查两树是否可以连接
        
        Returns:
            (node_start, node_goal) if connectable, else None
        """
        for node_s in self.tree_start:
            for node_g in self.tree_goal:
                dist = np.linalg.norm(node_s.q - node_g.q)
                
                if dist < self.config.connection_threshold:
                    # Check if edge is valid
                    if self._edge_valid(node_s.q, node_g.q):
                        return (node_s, node_g)
        
        return None
    
    def plan(self, q_start: np.ndarray, q_goal: np.ndarray) -> Optional[List[np.ndarray]]:
        """
        RRT-Connect主循环
        
        Returns:
            path: List of configurations, or None if failed
        """
        start_time = time.time()
        
        # Initialize trees
        self.tree_start = [Node(q=q_start)]
        self.tree_goal = [Node(q=q_goal)]
        
        # Planning loop
        for iteration in range(self.config.max_iters):
            # Sample
            q_rand = self.sample()
            
            # Extend start tree
            status_s, node_s = self.extend(self.tree_start, q_rand)
            
            if status_s != 'trapped':
                # Try to connect goal tree to q_new
                status_g, node_g = self.connect(self.tree_goal, node_s.q)
                
                if status_g == 'reached':
                    # Trees connected!
                    self.planning_time = time.time() - start_time
                    self.path_found = True
                    return self._extract_path(node_s, node_g)
            
            # Swap trees
            self.tree_start, self.tree_goal = self.tree_goal, self.tree_start
        
        # Failed
        self.planning_time = time.time() - start_time
        self.path_found = False
        return None
    
    def _extract_path(self, node_start: Node, node_goal: Node) -> List[np.ndarray]:
        """提取路径从两个树"""
        path_start = []
        node = node_start
        while node is not None:
            path_start.append(node.q)
            node = node.parent
        path_start.reverse()
        
        path_goal = []
        node = node_goal
        while node is not None:
            path_goal.append(node.q)
            node = node.parent
        
        # Combine
        path = path_start + path_goal
        return path


# ============================================================================
# Path Post-Processing
# ============================================================================

def shortcut_smoothing(path: List[np.ndarray], collision_checker: CollisionChecker, 
                       num_attempts: int = 100) -> List[np.ndarray]:
    """
    Shortcut平滑: 随机选两点尝试直连
    """
    if len(path) < 3:
        return path
    
    smoothed_path = path.copy()
    
    for _ in range(num_attempts):
        if len(smoothed_path) < 3:
            break
        
        # Random two indices
        i = random.randint(0, len(smoothed_path) - 1)
        j = random.randint(0, len(smoothed_path) - 1)
        
        if abs(i - j) <= 1:
            continue
        
        if i > j:
            i, j = j, i
        
        # Try to connect directly
        q_i = smoothed_path[i]
        q_j = smoothed_path[j]
        
        # Check edge validity
        dist = np.linalg.norm(q_j - q_i)
        num_checks = int(np.ceil(dist / 0.05)) + 1
        
        valid = True
        for k in range(num_checks):
            alpha = k / max(num_checks - 1, 1)
            q_check = q_i + alpha * (q_j - q_i)
            
            if collision_checker.check_collision(q_check, use_cheap=False):
                valid = False
                break
        
        if valid:
            # Remove intermediate waypoints
            smoothed_path = smoothed_path[:i+1] + smoothed_path[j:]
    
    return smoothed_path


def compute_path_length(path: List[np.ndarray]) -> float:
    """计算路径长度 (关节空间)"""
    length = 0.0
    for i in range(len(path) - 1):
        length += np.linalg.norm(path[i+1] - path[i])
    return length


# ============================================================================
# Visualization
# ============================================================================

def playback_trajectory(model: mujoco.MjModel, path: List[np.ndarray], 
                       fps: int = 30, interpolate: bool = True):
    """
    在MuJoCo viewer中播放轨迹
    """
    data = mujoco.MjData(model)
    
    # Interpolate for smoother playback
    if interpolate and len(path) > 1:
        interpolated_path = []
        for i in range(len(path) - 1):
            q_start = path[i]
            q_end = path[i + 1]
            
            # 10 steps between waypoints
            for alpha in np.linspace(0, 1, 10):
                q_interp = q_start + alpha * (q_end - q_start)
                interpolated_path.append(q_interp)
        
        interpolated_path.append(path[-1])
        path = interpolated_path
    
    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.8, 0.0, 0.85]
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -20
        
        print("\n播放轨迹...")
        print("关闭窗口继续\n")
        
        dt = 1.0 / fps
        
        for i, q in enumerate(path):
            # Set configuration
            data.qpos[:12] = q
            data.ctrl[:12] = q  # Position control
            
            # Step simulation
            mujoco.mj_step(model, data)
            
            # Sync viewer
            viewer.sync()
            
            # Sleep
            time.sleep(dt)
            
            # Print progress
            if i % 10 == 0:
                print(f"  Waypoint {i}/{len(path)}")
        
        print("播放完成. 按任意键关闭...")
        input()


# ============================================================================
# Evaluation Framework
# ============================================================================

@dataclass
class PlanningResult:
    """单次规划结果"""
    seed: int
    success: bool
    planning_time: float
    num_nodes: int
    num_samples: int
    num_cheap_checks: int
    num_precise_checks: int
    raw_path_length: float
    smoothed_path_length: float
    path: Optional[List[np.ndarray]] = None


def evaluate_planner(num_trials: int = 20) -> List[PlanningResult]:
    """
    评估规划器: 20次随机种子
    """
    print("\n" + "="*70)
    print("评估模式: 运行 {} 次试验".format(num_trials))
    print("="*70)
    
    results = []
    
    # Build scene
    model, _ = build_scene()
    
    # Joint limits (12 DOF)
    q_min = np.array([-3.14] * 12)
    q_max = np.array([3.14] * 12)
    
    # Handoff zone
    handoff_zone = np.array([0.8, 0.0, 0.85])
    
    # Start and goal configurations
    q_start = np.array([0, -1.57, 0, -1.57, 0, 0,     # Left arm neutral
                        0, -1.57, 0, -1.57, 0, 0])    # Right arm neutral
    
    q_goal = np.array([-0.5, -1.2, 1.0, -1.8, 0, 0,   # Left arm reaching
                       0.5, -1.2, -1.0, -1.8, 0, 0])  # Right arm reaching
    
    for trial in range(num_trials):
        seed = 42 + trial
        
        # Create config
        config = PlannerConfig(
            q_min=q_min,
            q_max=q_max,
            seed=seed,
            goal_bias=0.1,
            task_space_bias=0.2,
            handoff_zone=handoff_zone,
            max_iters=3000
        )
        
        # Create collision checker
        collision_checker = CollisionChecker(model)
        collision_checker.reset_stats()
        
        # Create planner
        planner = RRTConnect(config, collision_checker)
        
        # Plan
        path = planner.plan(q_start, q_goal)
        
        if path is not None:
            raw_length = compute_path_length(path)
            
            # Smooth
            smoothed_path = shortcut_smoothing(path, collision_checker, num_attempts=50)
            smoothed_length = compute_path_length(smoothed_path)
            
            result = PlanningResult(
                seed=seed,
                success=True,
                planning_time=planner.planning_time,
                num_nodes=len(planner.tree_start) + len(planner.tree_goal),
                num_samples=planner.num_samples,
                num_cheap_checks=collision_checker.num_cheap_checks,
                num_precise_checks=collision_checker.num_precise_checks,
                raw_path_length=raw_length,
                smoothed_path_length=smoothed_length,
                path=smoothed_path
            )
        else:
            result = PlanningResult(
                seed=seed,
                success=False,
                planning_time=planner.planning_time,
                num_nodes=len(planner.tree_start) + len(planner.tree_goal),
                num_samples=planner.num_samples,
                num_cheap_checks=collision_checker.num_cheap_checks,
                num_precise_checks=collision_checker.num_precise_checks,
                raw_path_length=0.0,
                smoothed_path_length=0.0
            )
        
        results.append(result)
        
        # Progress
        status = "✓" if result.success else "✗"
        print(f"Trial {trial+1:2d}/{ num_trials}: Seed={seed}, {status}, Time={result.planning_time:.3f}s")
    
    return results


def print_statistics(results: List[PlanningResult]):
    """打印统计表格"""
    print("\n" + "="*70)
    print("统计结果")
    print("="*70)
    
    # Success rate
    num_success = sum(1 for r in results if r.success)
    success_rate = num_success / len(results) * 100
    
    print(f"\n成功率: {num_success}/{len(results)} ({success_rate:.1f}%)")
    
    if num_success == 0:
        print("所有试验失败!")
        return
    
    # Time statistics
    success_times = [r.planning_time for r in results if r.success]
    time_p50 = np.percentile(success_times, 50)
    time_p90 = np.percentile(success_times, 90)
    time_mean = np.mean(success_times)
    
    print(f"\n规划时间 (成功案例):")
    print(f"  平均: {time_mean:.3f}s")
    print(f"  P50:  {time_p50:.3f}s")
    print(f"  P90:  {time_p90:.3f}s")
    
    # Nodes
    success_nodes = [r.num_nodes for r in results if r.success]
    nodes_mean = np.mean(success_nodes)
    
    print(f"\n节点数 (平均): {nodes_mean:.0f}")
    
    # Collision checks
    cheap_mean = np.mean([r.num_cheap_checks for r in results if r.success])
    precise_mean = np.mean([r.num_precise_checks for r in results if r.success])
    
    print(f"\n碰撞检测次数 (平均):")
    print(f"  Cheap:   {cheap_mean:.0f}")
    print(f"  Precise: {precise_mean:.0f}")
    
    # Path length
    raw_mean = np.mean([r.raw_path_length for r in results if r.success])
    smoothed_mean = np.mean([r.smoothed_path_length for r in results if r.success])
    improvement = (raw_mean - smoothed_mean) / raw_mean * 100
    
    print(f"\n路径长度 (平均):")
    print(f"  原始:   {raw_mean:.3f} rad")
    print(f"  平滑后: {smoothed_mean:.3f} rad")
    print(f"  改进:   {improvement:.1f}%")
    
    print("\n" + "="*70)


# ============================================================================
# Main
# ============================================================================

def main():
    """主函数: 评估 + 可视化"""
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "  双臂系统采样规划演示 - Dual-Arm Sampling-Based Planning".center(68) + "#")
    print("#" + "  场景 S1: 狭窄通道中的换手/交接".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    # Step 1: Evaluation
    results = evaluate_planner(num_trials=20)
    
    # Step 2: Statistics
    print_statistics(results)
    
    # Step 3: Visualization (pick a successful case)
    successful_results = [r for r in results if r.success]
    
    if len(successful_results) > 0:
        print("\n" + "="*70)
        print("可视化模式")
        print("="*70)
        
        # Pick random successful case
        result = random.choice(successful_results)
        
        print(f"\n选择案例: Seed={result.seed}")
        print(f"  规划时间: {result.planning_time:.3f}s")
        print(f"  路径长度: {result.smoothed_path_length:.3f} rad")
        print(f"  节点数: {result.num_nodes}")
        
        # Build scene
        model, _ = build_scene()
        
        # Playback
        print("\n准备播放轨迹...")
        input("按 Enter 启动 MuJoCo viewer...")
        
        playback_trajectory(model, result.path, fps=30, interpolate=True)
        
        print("\n演示完成!")
    else:
        print("\n警告: 没有成功案例可以可视化!")
    
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "  演示完成!".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()
