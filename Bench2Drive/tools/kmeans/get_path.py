import os
import pickle
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import mmcv
from mmcv import Config, DictAction
from mmdet.datasets import build_dataset


K = 1024
cfg = Config.fromfile("projects/configs/sparsedrive_stage2.py")
path_interval = "1"
num_pts = 15

def compute_dis_matrix(points):
    dis = np.expand_dims(points, 0) - np.expand_dims(points, 1)
    dis = np.linalg.norm(dis, axis=-1)
    return dis

def farthest_point_sampling(points, sample_num):
    ''' 
        points: [len, 2]
    '''
    assert len(points) >= sample_num
    assert sample_num > 0
    
    sampled_indices = np.zeros(sample_num)
    distance = np.ones(len(points)) * 1e10
    cur_index = np.random.randint(len(points))

    for i in range(sample_num):
        sampled_indices[i] = cur_index
        cur_point = points[cur_index]
        dist = np.linalg.norm(points - cur_point, axis=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        cur_index = np.argmax(distance)
    return sampled_indices.astype(np.int64)

if hasattr(cfg, "plugin"):
    if cfg.plugin:
        import importlib

        if hasattr(cfg, "plugin_dir"):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_dir = _module_dir.split("/")
            _module_path = _module_dir[0]

            for m in _module_dir[1:]:
                _module_path = _module_path + "." + m
            print(_module_path)
            plg_lib = importlib.import_module(_module_path)
        else:
            # import dir is the dirpath for the config file
            _module_dir = os.path.dirname(args.config)
            _module_dir = _module_dir.split("/")
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + "." + m
            print(_module_path)
            plg_lib = importlib.import_module(_module_path)
dataset = build_dataset(cfg.data.train)

traj_path = f"data/kmeans/path_{path_interval}m_pts_{num_pts}_all_new_ego.pkl"
if os.path.exists(traj_path):
    with open(traj_path,'rb') as f:
        trajs = pickle.load(f)
else:
    trajs = []
    for idx in tqdm(range(len(dataset))):
        info = dataset.get_ann_info(idx)
        plan_traj = info[f'gt_lat']
        plan_mask = info[f'gt_lat_mask']
        if not plan_mask.sum() == num_pts:
            continue
        trajs.append(plan_traj)
    print(f"total {len(trajs)} trajs")
    with open(traj_path,'wb') as f:
        pickle.dump(trajs, f)


## cluster
trajs = np.concatenate(trajs, axis=0).reshape(-1, num_pts*2)
cluster = KMeans(n_clusters=K).fit(trajs).cluster_centers_
trajs = cluster.reshape(-1, num_pts, 2)

for j in range(K):
    plt.plot(trajs[j, :, 0], trajs[j, :,1])
plt.savefig(f'vis/kmeans/path_{path_interval}m_pts_{num_pts}_{K}_ego', bbox_inches='tight')
plt.close()
os.makedirs("data/kmeans", exist_ok=True)
os.makedirs("vis/kmeans", exist_ok=True)
np.save(f'data/kmeans/path_{path_interval}m_pts_{num_pts}_{K}_b2d_new_ego.npy', trajs)