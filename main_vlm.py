import os
import sys
import cv2
import mujoco
import matplotlib.pyplot as plt 
import time
import torch
import numpy as np
import argparse

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'manipulator_grasp'))

from manipulator_grasp.env.ur3e_grasp_env import UR3eGraspEnv

from vlm_process import segment_image
from grasp_process import run_grasp_inference, execute_grasp


# 全局变量
global color_img, depth_img, env, planner_type, target_name, grasp_model_type
color_img = None
depth_img = None
env = None
planner_type = 'rrtconnect'  # default planner
target_name = None  # 目标物体名称，用于保存 GraspNet  or GraspGen or GR-ConvNet预测可视化
grasp_model_type = 'graspnet'  # 抓取预测模型：'graspnet'、'graspgen' 或 'grconvnet'
manual_select = False  # 是否手动框选目标物体



def get_image(env):
    global color_img, depth_img

    imgs = env.render()

    color_img = imgs['img']
    depth_img = imgs['depth']

    # 将RGB图像转换为OpenCV常用的BGR格式
    color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('color_img_path.jpg', color_img)

    return color_img, depth_img

# 构造回调函数，不断调用
def callback(color_frame, depth_frame):
    global color_img, depth_img
    scaling_factor_x = 1
    scaling_factor_y = 1

    color_img = cv2.resize(
        color_frame, None,
        fx=scaling_factor_x,
        fy=scaling_factor_y,
        interpolation=cv2.INTER_AREA
    )
    depth_img = cv2.resize(
        depth_frame, None,
        fx=scaling_factor_x,
        fy=scaling_factor_y,
        interpolation=cv2.INTER_NEAREST
    )

    if color_img is not None and depth_img is not None:
        return test_grasp()
    return True


def test_grasp():
    global color_img, depth_img, env, planner_type, target_name, grasp_model_type, manual_select

    if color_img is None or depth_img is None:
        print("[WARNING] Waiting for image data...")
        return

    # 图像处理部分
    masks = segment_image(color_img, manual_select=manual_select) 

    if masks is None:
        print("[WARNING] Target selection failed or cancelled. Exiting.")
        return False

    # 完成后释放SAM和Whisper的内存
    torch.cuda.empty_cache()

    gg_list, cloud_o3d = run_grasp_inference(
        color_img, depth_img, masks,
        target_name=target_name,
        grasp_model=grasp_model_type,
    )

    execute_grasp(env, gg_list, cloud_o3d, planner_type=planner_type, target_name=target_name)

    # 释放抓取模型的内存
    del gg_list
    torch.cuda.empty_cache()
    
    return True



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VLM-based Grasp with Path Planning')
    parser.add_argument(
        '--planner', 
        type=str, 
        choices=['rl_ppo', 'rrtconnect'], 
        default='rrtconnect',
        help='Path planner to use: rl_ppo (RL PPO policy) or rrtconnect (RRT-Connect, default)'
    )
    parser.add_argument(
        '--target',
        type=str,
        default=None,
        help='Target object name (e.g. banana). When set, saves Grasp-model prediction visualizations to Img_grasping/{target}_gg/ folder'
    )
    parser.add_argument(
        '--grasp_model',
        type=str,
        choices=['graspnet', 'graspgen', 'grconvnet'],
        default='graspnet',
        help='Grasp prediction model: graspnet (GraspNet-baseline, default), graspgen (NVlabs GraspGen), or grconvnet (GR-ConvNet)'
    )
    parser.add_argument(
        '--manual_select',
        action='store_true',
        help='Skip VLM inference and use manual mouse click to select the target object'
    )
    args = parser.parse_args()
    
    planner_type = args.planner
    target_name = args.target
    grasp_model_type = args.grasp_model
    manual_select = args.manual_select
    print(f"[main_vlm] Using planner: {planner_type}")
    print(f"[main_vlm] Using grasp model: {grasp_model_type}")
    print(f"[main_vlm] Manual select mode: {manual_select}")
    if target_name:
        print(f"[main_vlm] Target object: {target_name} (will save grasp visualizations to Img_grasping/{target_name}_gg/)")
    
    env = UR3eGraspEnv()
    env.reset()


    try:
        while True:

            for i in range(100): 
                env.step()

            color_img, depth_img = get_image(env)

            ret = callback(color_img, depth_img)
            if ret is False:
                print("[INFO] Exiting main loop.")
                break
    except KeyboardInterrupt:
        print("\n[INFO] Stopping simulation...")
    finally:
        # 清理资源避免段错误 (Segmentation fault)
        if hasattr(env, 'close') and callable(getattr(env, 'close')):
            try:
                env.close()
            except Exception:
                pass
        elif hasattr(env, 'viewer') and hasattr(env.viewer, 'close'):
            try:
                env.viewer.close()
            except Exception:
                pass
        cv2.destroyAllWindows()
        print("[INFO] Exited.")