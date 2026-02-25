import pinocchio
import numpy as np

from pyroboplan.planning.rrt import RRTPlanner
from pyroboplan.trajectory.trajectory_optimization import (
    CubicTrajectoryOptimization,
    CubicTrajectoryOptimizationOptions,
)

def getIk(env, init_state, T_target):
    rotation_matrix = T_target.R
    translation_vector = T_target.t

    target_tform = pinocchio.SE3(rotation_matrix, np.array(translation_vector))
    q_sol = env.ik.solve(
        env.target_frame,
        target_tform,
        init_state=init_state.copy(),
        verbose=False,  # 关闭详细输出，避免干扰
    )
    return q_sol
        
def get_traj(env, q_start, q_goal, override_planner_options=None): 
    # Search for a path
    print("")
    print(f"Planning a path...")
    
    # 允许在外部覆盖原有的 RRT 配置或 collision_model（若需要，不过 env.collision_model 是引用，这里只需确保 RRTplanner 拿到最新的组合即可）
    # 当动态添加了被夹取物体后，由于 collision_model 被修改了，
    # 我们应该重新生成一个 RRTPlanner 以确保它加载了新的碰撞检测树
    options = override_planner_options if override_planner_options is not None else env.rrt_options
    planner = RRTPlanner(env.model_roboplan, env.collision_model, options=options)
    
    q_path = planner.plan(q_start, q_goal)
    if q_path is not None and len(q_path) > 0:
        print(f"Got a path with {len(q_path)} waypoints")
        if len(q_path) > 100:
            print("Path is too long, skipping...")
            return None
        else:
            print(q_path)
            # Perform trajectory optimization.
            dt = 0.007
            traj_options = CubicTrajectoryOptimizationOptions(
                num_waypoints=len(q_path),
                samples_per_segment=1,
                min_segment_time=0.5,
                max_segment_time=10.0,
                min_vel=-1.0,
                max_vel=1.0,
                min_accel=-0.5,
                max_accel=0.5,
                min_jerk=-0.5,
                max_jerk=0.5,
                max_planning_time=3.0,
                check_collisions=False,
                min_collision_dist=0.001,
                collision_influence_dist=0.05,
                collision_avoidance_cost_weight=0.0,
                collision_link_list=[
                    # "obstacle_box_1",
                    # "obstacle_box_2",
                    # "obstacle_sphere_1",
                    # "obstacle_sphere_2",
                    # "obstacle_sphere_3",
                    # "ground_plane",
                    "grasp_center",
                ],
            )         
            print("Optimizing the path...")
            optimizer = CubicTrajectoryOptimization(env.model_roboplan, env.collision_model, traj_options)
            print("Retrying with all the RRT waypoints...")
            traj = optimizer.plan(q_path, init_path=q_path)
            if traj is not None:
                print("Trajectory optimization successful")
                traj_gen = traj.generate(dt)
                q_vec = traj_gen[1]
                print(f"path has {q_vec.shape[1]} points")
                return q_vec
            else:
                return None
    else:
        print("Failed to plan.")
        return None

             




