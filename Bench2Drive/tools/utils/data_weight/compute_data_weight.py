import os
import glob
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pickle
import cv2
import numpy as np
from PIL import Image
import json

import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset

from tools.visualization.bev_render import BEVRender
from tools.visualization.cam_render import CamRender
from tools.visualization.bevcam_render import BEVCamRender
import matplotlib.pyplot as plt


# cfg = Config.fromfile("projects/configs/id_71_motion_head_v5_baseline.py")
# dataset = build_dataset(cfg.data.train)
# tokens = []
# for info in dataset.data_infos:
#     token = info["token"]
#     tokens.append(token)
# save_file = 'tokens_list.pkl'
# with open(save_file, 'wb') as f:
#     pickle.dump(tokens, f)

# scene_cnt_dict = dict()
# scene_set = ['ParkingExit', 'NonSignalizedJunctionLeftTurnEnterFlow', 'YieldToEmergencyVehicle', 'HardBreakRoute', 'OppositeVehicleTakingPriority', 'MergerIntoSlowTrafficV2', 'ParkedObstacle', 'SignalizedJunctionLeftTurnEnterFlow', 'HighwayCutIn', 'VanillaNonSignalizedTurnEncounterStopsign', 'LaneChange', 'DynamicObjectCrossing', 'InterurbanActorFlow', 'VehicleOpensDoorTwoWays', 'TJunction', 'ParkedObstacleTwoWays', 'MergerIntoSlowTraffic', 'SignalizedJunctionRightTurn', 'VehicleTurningRoute', 'VehicleTurningRoutePedestrian', 'InvadingTurn', 'HighwayExit', 'ParkingCrossingPedestrian', 'ConstructionObstacleTwoWays', 'VanillaSignalizedTurnEncounterGreenLight', 'NonSignalizedJunctionLeftTurn', 'InterurbanAdvancedActorFlow', 'HazardAtSideLaneTwoWays', 'SignalizedJunctionLeftTurn', 'PedestrianCrossing', 'OppositeVehicleRunningRedLight', 'EnterActorFlow', 'StaticCutIn', 'ConstructionObstacle', 'HazardAtSideLane', 'NonSignalizedJunctionRightTurn', 'ParkingCutIn', 'Accident', 'VanillaSignalizedTurnEncounterRedLight', 'AccidentTwoWays', 'BlockedIntersection', 'CrossingBicycleFlow', 'ControlLoss']
# for scene in scene_set:
#     scene_cnt_dict[scene] = []

mode = "col_3s"

################# frame_freq #################
if mode == "frame_freq":
    save_file = 'tokens_list.pkl'
    with open(save_file, 'rb') as f:
        tokens = pickle.load(f)
    data_weights = np.zeros(len(tokens))
    cnt = {}
    for token in tokens:
        scene = token.split("/")[1].split("_")[0]
        if scene not in cnt:
            cnt[scene] = 1
        else:
            cnt[scene] += 1
    
    total = sum([v for v in cnt.values()])
    num_group = len(cnt)
    scene_weights = dict()
    for k, v in cnt.items():
        scene_weights[k] = total / v
    for i, token in enumerate(tokens):
        scene = token.split("/")[1].split("_")[0]
        data_weights[i] = scene_weights[scene]
    data_weights = data_weights * (len(tokens) / data_weights.sum())
    np.save(f"data/infos/data_weight_{mode}", data_weights)

################# scene_freq #################
if mode == "scene_freq":
    save_file = 'tokens_list.pkl'
    with open(save_file, 'rb') as f:
        tokens = pickle.load(f)
    data_weights = np.zeros(len(tokens))
    cnt = {}
    for token in tokens:
        scene = token.split("/")[1].split("_")[0]
        sub_scene = token[:-5]
        if scene not in cnt:
            cnt[scene] = [sub_scene]
        else:
            cnt[scene].append(sub_scene)
    cnt = {k: len(set(v)) for k, v in cnt.items()}
    total = sum([v for v in cnt.values()])
    num_group = len(cnt)
    scene_weights = dict()
    for k, v in cnt.items():
        scene_weights[k] = total / v
    for i, token in enumerate(tokens):
        scene = token.split("/")[1].split("_")[0]
        data_weights[i] = scene_weights[scene]
    scale = (len(tokens) / data_weights.sum())
    data_weights = data_weights * scale
    scene_weights_ad = {k: v * scale for k, v in scene_weights.items()}
    np.save(f"data/infos/data_weight_{mode}", data_weights)


################# col 3s #################
def get_corners(box):
    """
    box: (..., 7)  (x, y, z, w, l, h, yaw)
    return: (..., 4, 2)  四个角点
    """
    x, y, _, w, l, _, yaw = [box[..., i] for i in range(7)]
    cos, sin = np.cos(yaw), np.sin(yaw)

    # 局部坐标系下的 4 个角
    dx = np.stack([ w/2,  w/2, -w/2, -w/2], axis=-1)
    dy = np.stack([ l/2, -l/2, -l/2,  l/2], axis=-1)

    # 旋转到全局
    rx = dx * cos[..., None] - dy * sin[..., None]
    ry = dx * sin[..., None] + dy * cos[..., None]

    return np.stack([x[..., None] + rx,
                     y[..., None] + ry], axis=-1)     # (..., 4, 2)

def sat_2d_rotated_boxes(corners1: np.ndarray, corners2: np.ndarray, eps=1e-8):
    """
    利用分离轴定理判断两组旋转矩形是否相交。
    参数
    ----
    corners1 : ndarray, shape=(N, 4, 2)
    corners2 : ndarray, shape=(M, 4, 2)
    eps      : float, 浮点容差，防止数值误差
    返回
    ----
    collide  : ndarray, shape=(N, M), bool
               collide[i,j] == True  表示 corners1[i] 与 corners2[j] 相交
    """
    N = corners1.shape[0]
    M = corners2.shape[0]

    # 1. 构造 4 条边的方向向量（已经去重，每个 box 4 条边）
    #    边的方向向量 = 后一个角点 - 前一个角点
    def get_axes(c):
        # c: (K,4,2)
        edges = np.roll(c, -1, axis=1) - c          # (K,4,2)
        # 法向量（垂直向量）：(x,y)->(-y,x)
        normals = np.stack([-edges[..., 1], edges[..., 0]], axis=-1)  # (K,4,2)
        # 归一化（可选，这里只关心投影轴方向，可不归一化）
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = normals / (norms + eps)
        return normals                                # (K,4,2)
    axes1 = get_axes(corners1)   # (N,4,2)
    axes2 = get_axes(corners2)   # (M,4,2)

    # 2. 把所有待检测的轴拼到一起：N*4 + M*4 条轴
    #    需要广播到 (N,M,4+4,2)
    axes = np.concatenate([
        axes1[:, None, :, :].repeat(M, axis=1),       # (N,M,4,2)
        axes2[None, :, :, :].repeat(N, axis=0)       # (N,M,4,2)
    ], axis=2)                                        # (N,M,8,2)

    # 3. 把两个 box 的所有角点都投影到这些轴上
    #    project = dot(角点, 轴)
    #    corners1_proj: (N,1,4,1,2) * (1,M,1,8,2) -> (N,M,4,8)
    corners1_proj = np.einsum('npi,nmji->nmpj',
                              corners1,
                              axes)
    #    corners2_proj: (1,M,4,1,2) * (N,M,1,8,2) -> (N,M,4,8)
    corners2_proj = np.einsum('mpi,nmji->nmpj',
                              corners2,
                              axes)

    # 4. 计算每条轴上两个形状投影区间的最小/最大值
    min1 = corners1_proj.min(axis=2)   # (N,M,8)
    max1 = corners1_proj.max(axis=2)
    min2 = corners2_proj.min(axis=2)
    max2 = corners2_proj.max(axis=2)

    # 5. 判断区间是否重叠：若 max1 < min2 或 max2 < min1 则无交
    overlap = (min1 <= max2 + eps) & (min2 <= max1 + eps)  # (N,M,8)

    # 6. 只要有一条轴没有重叠，则这两个 box 分离
    collide = overlap.all(axis=2)   # (N,M)
    return collide

if mode == "col_3s":
    cfg = Config.fromfile("projects/configs/id_71_motion_head_v5_baseline.py")
    dataset = build_dataset(cfg.data.train)
    data_weights = np.ones(len(dataset))
    col_idxs = []
    for i, info in enumerate(dataset.data_infos):
        mask = (info['num_points'] != 0)
        gt_box = info["gt_boxes"][mask][:, :7]
        corners = get_corners(gt_box)
        ego_box = np.array([0,0,0, 4.89, 1.83, 0, 1.57]).reshape(1,7)
        ego_corners = get_corners(ego_box)
        col = sat_2d_rotated_boxes(ego_corners, corners)
        if col.any():
            col_idxs.append(i)
    for idx in col_idxs:
        cur_token = dataset.data_infos[idx]["token"][:-5]
        for j in range(idx, idx-30, -1):
            if dataset.data_infos[j]["token"][:-5] == cur_token:
                data_weights[j] = 0
            else:
                break
        for j in range(idx, idx+30, 1):
            if dataset.data_infos[j]["token"][:-5] == cur_token:
                data_weights[j] = 0
            else:
                break    
    np.save(f"data/infos/data_weight_{mode}", data_weights)

################# col scene #################
if mode == "col_scene":
    cfg = Config.fromfile("projects/configs/id_71_motion_head_v5_baseline.py")
    dataset = build_dataset(cfg.data.train)
    data_weights = np.ones(len(dataset))
    col_idxs = []
    for i, info in enumerate(dataset.data_infos):
        mask = (info['num_points'] != 0)
        gt_box = info["gt_boxes"][mask][:, :7]
        corners = get_corners(gt_box)
        ego_box = np.array([0,0,0, 4.89, 1.83, 0, 1.57]).reshape(1,7)
        ego_corners = get_corners(ego_box)
        col = sat_2d_rotated_boxes(ego_corners, corners)
        if col.any():
            col_idxs.append(i)
    for idx in col_idxs:
        cur_token = dataset.data_infos[idx]["token"][:-5]
        for j in range(idx, -1, -1):
            if dataset.data_infos[j]["token"][:-5] == cur_token:
                data_weights[j] = 0
            else:
                break
        for j in range(idx, len(dataset), 1):
            if dataset.data_infos[j]["token"][:-5] == cur_token:
                data_weights[j] = 0
            else:
                break    
    np.save(f"data/infos/data_weight_{mode}", data_weights)