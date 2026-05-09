import os
import pickle
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import mmcv
from mmcv import Config, DictAction
from mmdet.datasets import build_dataset


K = 256
cfg = Config.fromfile("projects/configs/sparsedrive_stage2.py")
time_points = [(0,0.5), (0.5,1.0), (1.0, 1.5), (1.5,2.0), (2.0,2.5), (2.5,3.0)]

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

vel_seqs = []
for i in tqdm(range(len(dataset))):
    vel_seq, mask = dataset.get_vel_seq(i, time_points, mode="vel")
    vel_seqs.append(vel_seq)
vel_seqs = np.array(vel_seqs)
cluster = KMeans(n_clusters=K).fit(vel_seqs).cluster_centers_
if len(time_points) == 1:
    cluster = np.sort(cluster,axis=0)
os.makedirs("data/kmeans", exist_ok=True)
os.makedirs("vis/kmeans", exist_ok=True)
np.save(f"data/kmeans/vel_seq_K{K}_t{int(time_points[-1][-1]*10)}", cluster)

vel_seqs = cluster
num_speed = len(time_points)
colors = plt.cm.Spectral(np.linspace(0, 1, num_speed))
x = np.arange(len(vel_seqs))
plt.figure(figsize=(10, 4))
plt.bar(x, vel_seqs[:, 0], color=colors[0], label='0-0.5 s')
bottom = vel_seqs[:, 0]
for i in range(1, num_speed):
    plt.bar(x, vel_seqs[:, i], bottom=bottom, color=colors[i], label=f'{i*0.5}-{(i+1)*0.5} s')
    bottom += vel_seqs[:, i]
plt.xlabel('sequence index')
plt.ylabel('speed  (m/s)')
plt.title('Stacked speed histogram')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f'vis/kmeans/vel_seq_K{K}_t{int(time_points[-1][-1]*10)}', bbox_inches='tight')
plt.close()