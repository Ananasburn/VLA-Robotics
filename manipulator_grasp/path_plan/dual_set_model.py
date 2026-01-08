import os
import hppfcl as coal
import pinocchio
import numpy as np

from pyroboplan.core.utils import set_collisions

def load_dual_model(urdf_path):
    package_dirs = os.path.dirname(os.path.dirname(urdf_path)) # Go up to robot_description usually, or let pinocchio handle relative
    # Actually, urdf is in robot_description/urdf. Meshes are in robot_description/meshes.
    # Pinocchio package_dirs should usually point to the directory containing the 'meshes' folder if referenced as package:// or relative.
    # The generated urdf uses "../meshes/...".
    # So if we load from "robot_description/urdf/file.urdf", base dir is "robot_description/urdf".
    # package_dirs argument helps resolve "package://" urls, but for relative paths, buildModelFromUrdf usually works relative to the file.
    
    model = pinocchio.buildModelFromUrdf(urdf_path)
    collision_model = pinocchio.buildGeomFromUrdf(model, urdf_path, pinocchio.GeometryType.COLLISION, package_dirs=package_dirs) 
    visual_model = pinocchio.buildGeomFromUrdf(model, urdf_path, pinocchio.GeometryType.VISUAL, package_dirs=package_dirs)

    #Create Reduced Model (Lock gripper joints usually)
    joints_to_lock = []
    for name in model.names:
        if "knuckle" in name or "finger" in name:
            if model.existJointName(name):
                joints_to_lock.append(model.getJointId(name))
    
    if len(joints_to_lock) > 0:
        # Reference configuration for locked joints (neutral)
        q_ref = pinocchio.neutral(model)
        print(f"DEBUG: model.nq={model.nq}, q_ref.shape={q_ref.shape}, locking {len(joints_to_lock)} joints")
        
        # Build reduced models
        r_model, r_visual_model = pinocchio.buildReducedModel(model, visual_model, joints_to_lock, q_ref)
        # Note: collision model needs separate reduction call? 
        # API: buildReducedModel(model, geom_model, joints_to_lock, q_ref)
        # We need to be careful. The first call returns a NEW model. 
        # The second call must use the ORIGINAL model to identify joints, but we need the NEW collision model consistent with NEW model?
        # Actually pinocchio.buildReducedModel(model, list_of_geom_models, ...) is not the signature.
        # It's (model, geom_model, ...).
        # We must regenerate collision model using the reduced model? No, buildReducedModel handles it.
        # But we need to use the SAME reduced model for both.
        
        # Correct pattern for multiple geom models:
        # reduced_model = pinocchio.buildReducedModel(model, joints_to_lock, q_ref)
        # reduced_visual = ...
        # But buildReducedModel is overloaded.
        
        # Let's use the version that returns (model, geom_model).
        # model_reduced, visual_mid = pinocchio.buildReducedModel(model, visual_model, joints_to_lock, q_ref)
        # model_reduced_2, collision_mid = pinocchio.buildReducedModel(model, collision_model, joints_to_lock, q_ref)
        # model_reduced and model_reduced_2 should be identical in structure. 
        
        # Let's do:
        r_model, r_visual_model = pinocchio.buildReducedModel(model, visual_model, joints_to_lock, q_ref)
        _, r_collision_model = pinocchio.buildReducedModel(model, collision_model, joints_to_lock, q_ref)
        
        model = r_model
        visual_model = r_visual_model
        collision_model = r_collision_model
        
    return model, collision_model, visual_model

def add_dual_collisions(model, collision_model, visual_model):
    """
    Adds the shared table and enables collisions:
    1. Table vs All Robot Links
    2. Left Arm vs Right Arm
    """
    
    # 1. Add Table (Scene Object)
    # Matches scene_dual.xml: pos="0.8 0.0 0.69", size="0.8 0.8 0.05"
    table_pose = pinocchio.SE3(np.eye(3), np.array([0.8, 0.0, 0.69]))
    table_geom = pinocchio.GeometryObject(
        "table",
        0,
        coal.Box(1.6, 1.6, 0.1), 
        table_pose,
    )
    table_geom.meshColor = np.array([0.5, 0.5, 0.5, 0.5])
    
    # Check if table already exists (semantics?)
    # Just add it.
    visual_model.addGeometryObject(table_geom)
    collision_model.addGeometryObject(table_geom)

    # 2. Identify Links
    # model.names supports getting link names. collision_model.geometryObjects are the collision geometries attached to links.
    # We want to enable collision for the collision geometries.
    
    left_geoms = []
    right_geoms = []
    
    for cobj in collision_model.geometryObjects:
        # Check the name of the link this geometry is attached to (parentJoint -> matches link? No, cobj.name usually derived from link/geom name)
        # Actually in Pinocchio URDF loader, geom names are often "link_name_0".
        # Let's filter by string "left_" and "right_" in the collision object name.
        if cobj.name.startswith("left_"):
            left_geoms.append(cobj.name)
        elif cobj.name.startswith("right_"):
            right_geoms.append(cobj.name)

    # 3. Set Collisions
    # Table vs All
    all_robot_geoms = left_geoms + right_geoms
    for geom_name in all_robot_geoms:
        if "base_link" in geom_name:
             set_collisions(model, collision_model, "table", geom_name, False)
        else:
             set_collisions(model, collision_model, "table", geom_name, True)
    
    # Left vs Right
    for l_name in left_geoms:
        for r_name in right_geoms:
            # We enable collision between every pair of left and right arm parts
            set_collisions(model, collision_model, l_name, r_name, True)
            
    print(f"Dual Collisions setup: {len(left_geoms)} Left objects, {len(right_geoms)} Right objects.")
