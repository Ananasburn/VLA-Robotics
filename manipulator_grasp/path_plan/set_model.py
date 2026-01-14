import os
import coal
import pinocchio
import numpy as np

from pyroboplan.core.utils import set_collisions
from pyroboplan.ik.differential_ik import DifferentialIk, DifferentialIkOptions
from pyroboplan.planning.rrt import RRTPlannerOptions


def load_models(urdf_path):
    package_dirs = os.path.dirname(urdf_path)
    model = pinocchio.buildModelFromUrdf(urdf_path)
    collision_model = pinocchio.buildGeomFromUrdf(model, urdf_path, pinocchio.GeometryType.COLLISION, package_dirs=package_dirs)
    visual_model = pinocchio.buildGeomFromUrdf(model, urdf_path, pinocchio.GeometryType.VISUAL, package_dirs=package_dirs)

    return model, collision_model, visual_model

def add_self_collisions(model, collision_model, srdf_path):
    if os.path.exists(srdf_path):
        collision_model.addAllCollisionPairs()
        pinocchio.removeCollisionPairs(model, collision_model, srdf_path)
    else:
        print(f"警告: SRDF文件不存在 {srdf_path}")

def add_object_collisions(model, collision_model, visual_model, inflation_radius=0.0):
    # Add the collision objects
    # 降低地面碰撞体高度，避免与夹爪在抓取位置碰撞
    # z=0.60, 厚度0.05 -> 碰撞范围0.575-0.625（远离物体高度0.735）
    ground_plane = pinocchio.GeometryObject(
        "ground_plane",
        0,
        pinocchio.SE3(np.eye(3), np.array([0.8, 0.6, 0.60])),
        coal.Box(1.6, 1.2, 0.05),
    )
    ground_plane.meshColor = np.array([0.5, 0.5, 0.5, 0.5])
    visual_model.addGeometryObject(ground_plane)
    collision_model.addGeometryObject(ground_plane)

    obstacle_sphere_1 = pinocchio.GeometryObject(
        "obstacle_sphere_1",
        0,
        pinocchio.SE3(np.eye(3), np.array([0.6, 0.6, 0.85])),
        coal.Sphere(0.1 + inflation_radius),
    )
    obstacle_sphere_1.meshColor = np.array([0.0, 1.0, 0.0, 0.5])
    visual_model.addGeometryObject(obstacle_sphere_1)
    collision_model.addGeometryObject(obstacle_sphere_1)

    obstacle_sphere_2 = pinocchio.GeometryObject(
        "obstacle_sphere_2",
        0,
        pinocchio.SE3(np.eye(3), np.array([0.6, 0.6, 1.5])),
        coal.Sphere(0.15 + inflation_radius),
    )
    obstacle_sphere_2.meshColor = np.array([1.0, 1.0, 0.0, 0.5])
    visual_model.addGeometryObject(obstacle_sphere_2)
    collision_model.addGeometryObject(obstacle_sphere_2)

    obstacle_sphere_3= pinocchio.GeometryObject(
        "obstacle_sphere_3",
        0,
        pinocchio.SE3(np.eye(3), np.array([0.8, 1.0, 1.0])),
        coal.Sphere(0.1 + inflation_radius),
    )
    obstacle_sphere_3.meshColor = np.array([1.0, 1.0, 0.0, 0.5])
    visual_model.addGeometryObject(obstacle_sphere_3)
    collision_model.addGeometryObject(obstacle_sphere_3)

    obstacle_box_1 = pinocchio.GeometryObject(
        "obstacle_box_1",
        0,
        pinocchio.SE3(np.eye(3), np.array([1.35, 0.2, 1.0])),
        coal.Box(
            0.5 + 2.0 * inflation_radius,
            0.1 + 2.0 * inflation_radius,
            0.52 + 2.0 * inflation_radius,
        ),
    )
    obstacle_box_1.meshColor = np.array([1.0, 0.0, 0.0, 0.5])
    visual_model.addGeometryObject(obstacle_box_1)
    collision_model.addGeometryObject(obstacle_box_1)

    obstacle_box_2 = pinocchio.GeometryObject(
        "obstacle_box_2",
        0,
        pinocchio.SE3(np.eye(3), np.array([0.4, 0.6, 1.3])),
        coal.Box(
            0.8 + 2.0 * inflation_radius,
            1.2 + 2.0 * inflation_radius,
            0.08 + 2.0 * inflation_radius,
        ),
    )
    obstacle_box_2.meshColor = np.array([0.0, 0.0, 1.0, 0.5])
    visual_model.addGeometryObject(obstacle_box_2)
    collision_model.addGeometryObject(obstacle_box_2)

    obstacle_box_3 = pinocchio.GeometryObject(
        "obstacle_box_3",
        0,
        pinocchio.SE3(np.eye(3), np.array([1.15, 0.2, 0.9])),
        coal.Box(
            0.1 + 2.0 * inflation_radius,
            0.4 + 2.0 * inflation_radius,
            0.32 + 2.0 * inflation_radius,
        ),
    )
    obstacle_box_3.meshColor = np.array([0.0, 0.0, 1.0, 0.5])
    visual_model.addGeometryObject(obstacle_box_3)
    collision_model.addGeometryObject(obstacle_box_3)

    # Define the active collision pairs between the robot and obstacle links.
    robot_links = [
        cobj.name for cobj in collision_model.geometryObjects 
        if any(name in cobj.name for name in ['ag95', 'link', 'left', 'right'])
    ]
    obstacle_names = [
        "ground_plane",
        "obstacle_box_1",
        # "obstacle_box_2",
        # "obstacle_box_3",
        "obstacle_sphere_1",
        "obstacle_sphere_2",
        "obstacle_sphere_3",
    ]
    for obstacle_name in obstacle_names:
        for link_name in robot_links:
            set_collisions(model, collision_model, obstacle_name, link_name, True)

    # Exclude the collision between the ground and the base link
    set_collisions(model, collision_model, "base_link_inertia", "ground_plane", False)
    set_collisions(model, collision_model, "left_finger_pad", "ground_plane", False)
    set_collisions(model, collision_model, "right_finger_pad", "ground_plane", False)
    set_collisions(model, collision_model, "left_finger", "ground_plane", False)
    set_collisions(model, collision_model, "right_finger", "ground_plane", False)
    set_collisions(model, collision_model, "left_outer_knuckle", "ground_plane", False)
    set_collisions(model, collision_model, "right_outer_knuckle", "ground_plane", False)
    set_collisions(model, collision_model, "left_inner_knuckle", "ground_plane", False)
    set_collisions(model, collision_model, "right_inner_knuckle", "ground_plane", False)
    set_collisions(model, collision_model, "left_inner_finger", "ground_plane", False)
    set_collisions(model, collision_model, "right_inner_finger", "ground_plane", False)
    set_collisions(model, collision_model, "ag95_base_link", "ground_plane", False)
    set_collisions(model, collision_model, "grasp_center", "ground_plane", False)

def load_path_planner(model_roboplan, data_roboplan, collision_model):
    target_frame = "grasp_center"
    ignore_joint_indices = [
        model_roboplan.getJointId("left_inner_knuckle_joint") - 1,
        model_roboplan.getJointId("left_outer_knuckle_joint") - 1,
        model_roboplan.getJointId("left_finger_joint") - 1,
        model_roboplan.getJointId("right_inner_knuckle_joint") - 1,
        model_roboplan.getJointId("right_outer_knuckle_joint") - 1,
        model_roboplan.getJointId("right_finger_joint") - 1,
    ]

    # Set up the IK solver
    # 优化IK参数加快求解速度
    ik_options = DifferentialIkOptions(
        max_iters=200,          # 减少迭代次数加快速度
        max_retries=10,         # 减少重试次数
        damping=0.0001,         # 稍微增加阻尼提高稳定性
        min_step_size=0.05,     # 增大最小步长加快收敛
        max_step_size=0.5,      # 增大最大步长
        ignore_joint_indices=ignore_joint_indices,
        rng_seed=42,            # 固定种子提高可重复性
    )
    ik = DifferentialIk(
        model_roboplan,
        data=data_roboplan,
        collision_model=collision_model,
        options=ik_options,
        visualizer=None,
    )

    # 优化RRT规划参数以更快找到路径
    rrt_options = RRTPlannerOptions(
            max_step_size=0.3,              # 增大步长加快探索
            max_connection_dist=1.0,        # 增大连接距离
            rrt_connect=True,               # 启用RRT-Connect（双向搜索更快）
            bidirectional_rrt=True,         # 启用双向RRT
            rrt_star=False,                 # 关闭RRT*（优化路径但更慢）
            max_rewire_dist=0.3,
            max_planning_time=2.0,          # 减少最大规划时间
            fast_return=True,
            goal_biasing_probability=0.5,   # 增加目标偏向概率加快收敛
            collision_distance_padding=0.001,
        )
    
    return target_frame, ik, rrt_options