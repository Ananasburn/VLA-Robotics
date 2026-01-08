import numpy as np
import pinocchio
from pyroboplan.planning.rrt import RRTPlanner, RRTPlannerOptions
from pyroboplan.trajectory.trajectory_optimization import CubicTrajectoryOptimization, CubicTrajectoryOptimizationOptions

# Import our new model loader
from manipulator_grasp.path_plan.dual_set_model import load_dual_model, add_dual_collisions

class DualArmPlanner:
    def __init__(self, urdf_path):
        self.urdf_path = urdf_path
        
        # Load Model
        self.model, self.collision_model, self.visual_model = load_dual_model(urdf_path)
        
        # Setup Collisions
        add_dual_collisions(self.model, self.collision_model, self.visual_model)
        
        # Setup Data
        self.data = self.model.createData()
        self.collision_data = self.collision_model.createData()
        
        # Setup RRT Options
        # Dual UR3e = 12 DoF (plus grippers 2*? -> URDF has them).
        # We need appropriate step sizes.
        self.rrt_options = RRTPlannerOptions(
            max_step_size=0.1,         # Radians per step (smaller for better collision avoidance)
            max_connection_dist=0.5,   # Max distance to connect tree
            rrt_connect=True,          # RRT-Connect is usually faster for simple query
            bidirectional_rrt=True,
            rrt_star=False,            # Optimization later
            max_planning_time=60.0,    # Increased timeout for complex dual-arm planning
            goal_biasing_probability=0.1,  # Guide search toward goal
            collision_distance_padding=0.0 # Reduced padding
        )
        
        self.planner = RRTPlanner(
            self.model, 
            self.collision_model, 
            options=self.rrt_options
        )

    def plan(self, q_start, q_goal):
        """
        Plans a path from q_start to q_goal.
        Both q must be of dimension model.nq.
        """
        print(f"Planning dual-arm path from {q_start.shape} to {q_goal.shape}...")
        print(f"Distance between start and goal: {np.linalg.norm(q_goal - q_start):.4f} rad")
        
        # Plan
        path = self.planner.plan(q_start, q_goal)

        
        if path is not None and len(path) > 0:
            print(f"RRT found path with {len(path)} waypoints. Smoothing...")
            
            # Trajectory Optimization (Smoothing)
            # Use simplified options for now
            traj_options = CubicTrajectoryOptimizationOptions(
                num_waypoints=len(path),
                min_segment_time=0.1,
                max_segment_time=5.0,
                check_collisions=True, # Ensure smooth path is valid
                max_planning_time=2.0 
            )
            
            optimizer = CubicTrajectoryOptimization(
                self.model, 
                self.collision_model, 
                traj_options
            )
            
            # Seed with RRT path
            traj = optimizer.plan(path, init_path=path)
            
            if traj is not None:
                # Discretize
                dt = 0.01
                traj_gen = traj.generate(dt)
                q_vec = traj_gen[1] # [nq, T]
                return q_vec.T # Return [T, nq]
            else:
                print("Smoothing failed, returning raw path.")
                return np.array(path)
                
        else:
            print("RRT failed to find a path.")
            return None
