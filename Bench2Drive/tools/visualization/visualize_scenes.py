import os
import glob
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import cv2, os, glob
from joblib import Parallel, delayed
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image

import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset

from tools.visualization.bev_render import BEVRender
from tools.visualization.cam_render import CamRender
from tools.visualization.bevcam_render import BEVCamRender

plot_choices = dict(
    bev_pred = True,
    bev_gt = False,
    cam_pred = True,
    cam_gt = False,
    bevcam_pred = False,
    det = True,
    track = False,
    motion = False,
    map = True,
    planning = True,
    path = False,
    target_point = True,
    route = True,
    speed = True,
)

plot_choices = dict(
    bev_pred = False,
    bev_gt = True,
    cam_pred = False,
    cam_gt = True,
    bevcam_pred = False,
    det = True,
    track = False,
    motion = False,
    map = True,
    planning = True,
    path = False,
    target_point = False,
    route = True,
    speed = False,
)

class Visualizer:
    def __init__(
        self,
        plot_choices,
        out_dir,
        cfg,
        result_path=None,
        planning_key=None,
    ):
        self.out_dir = out_dir
        self.combine_dir = os.path.join(self.out_dir, 'combine')
        os.makedirs(self.combine_dir, exist_ok=True)
        
        self.dataset = build_dataset(cfg.data.val)
        if (plot_choices["bev_pred"] or plot_choices["bev_pred"]) and result_path is not None :
            self.results = mmcv.load(result_path)
        else:
            self.results = None
        
        self.bev_render = BEVRender(plot_choices)
        self.cam_render = CamRender(plot_choices)
        self.bevcam_render = BEVCamRender(plot_choices)
        if planning_key is None:
            self.planning_key = cfg.get("anchor_reference_group", None)
        else:
            self.planning_key = planning_key

    def add_vis(self, index, data=None, result=None):
        if data is None:
            data = self.dataset.get_data_info(index)
            data["index"] = index
        data["planning_key"] = self.planning_key
        if result is None:
            result = self.results[index]['img_bbox'] if self.results is not None else {}

        bev_gt, bev_pred = self.bev_render.render(data, result)
        cam_gt, cam_pred = self.cam_render.render(data, result)
        bevcam_pred = self.bevcam_render.render(data, result)
        self.combine(bev_gt, bev_pred, cam_gt, cam_pred, bevcam_pred, index)
    
    def combine(self, bev_gt, bev_pred, cam_gt, cam_pred, bevcam_pred, index):
        if (
            bev_gt is not None and
            bev_pred is not None and
            cam_gt is not None and
            cam_pred is not None and
            bevcam_pred is None
        ):
            pred = cv2.hconcat([cam_pred, bev_pred])
            gt = cv2.hconcat([cam_gt, bev_gt])
            merge_image = cv2.vconcat([pred, gt])

        # if bev_gt is None and cam_gt is None:
        #     merge_image = cv2.hconcat([cam_pred, bevcam_pred])
        
        if (
            bev_gt is not None and
            bev_pred is None and
            cam_gt is not None and
            cam_pred is None and
            bevcam_pred is None
        ):
            merge_image = cv2.hconcat([cam_gt, bev_gt])

        if (
            bev_gt is None and
            bev_pred is not None and
            cam_gt is None and
            cam_pred is not None and
            bevcam_pred is None
        ):
            merge_image = cv2.hconcat([cam_pred, bev_pred])

        if (
            bev_gt is None and
            bev_pred is not None and
            cam_gt is None and
            cam_pred is None and
            bevcam_pred is None
        ):
            merge_image = bev_pred

        if (
            bev_gt is None and
            bev_pred is None and
            cam_gt is None and
            cam_pred is not None and
            bevcam_pred is not None
        ):
            merge_image = cv2.hconcat([cam_pred, bevcam_pred])

        if (
            bev_gt is None and
            cam_gt is None and
            cam_pred is not None and
            bev_pred is not None and
            bevcam_pred is not None
        ):
            merge_image = cv2.hconcat([cam_pred, bevcam_pred, bev_pred])
            # merge_image = cv2.hconcat([bevcam_pred, bev_pred])
        
        save_path = os.path.join(self.combine_dir, str(index).zfill(4) + '.jpg')
        cv2.imwrite(save_path, merge_image)

    def image2video(self, video_name, fps=10, downsample=4):
        imgs_path = glob.glob(os.path.join(self.combine_dir, '*.jpg'))
        imgs_path = sorted(imgs_path)
        img_array = []
        for img_path in tqdm(imgs_path):
            img = cv2.imread(img_path)
            height, width, channel = img.shape
            img = cv2.resize(img, (width//downsample, height //
                             downsample), interpolation=cv2.INTER_AREA)
            height, width, channel = img.shape
            size = (width, height)
            img_array.append(img)
        out = cv2.VideoWriter(
            video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

    def image2video_fast(self, video_name, fps=10, downsample=4):
        imgs_path = sorted(glob.glob(os.path.join(self.combine_dir, '*.jpg')))
        if not imgs_path:
            return

        # 1. 并行读 + resize （joblib 默认用多进程，n_jobs=-1 吃满 CPU）
        img_list = Parallel(n_jobs=32, backend='threading')(
            delayed(_load_and_resize)(p, downsample) for p in tqdm(imgs_path, desc='load/resize')
        )

        # 2. 拿尺寸
        h, w = img_list[0].shape[:2]

        # 3. 一次性写视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_name, fourcc, fps, (w, h))
        for frame in tqdm(img_list, desc='write'):
            out.write(frame)
        out.release()

def _load_and_resize(path, down):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    return cv2.resize(img, (w//down, h//down), interpolation=cv2.INTER_AREA)



def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize groundtruth and results')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--result-path', 
        default=None,
        help='prediction result to visualize'
        'If submission file is not provided, only gt will be visualized')
    parser.add_argument(
        '--out-dir', 
        default='vis',
        help='directory where visualize results will be saved')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of processes to use')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=80)
    parser.add_argument('--interval', type=int, default=1)
    args = parser.parse_args()

    return args

def process_index(args, plot_choices, cfg, index):
    visualizer = Visualizer(plot_choices, args.out_dir, cfg, args.result_path)
    visualizer.add_vis(index)

def main():
    import time
    args = parse_args()
    cfg = Config.fromfile(args.config)
    import pickle
    save_file = 'tokens_dict.pkl'      # 同上路径
    with open(save_file, 'rb') as f:
        tokens = pickle.load(f)
    scenes = list(tokens.keys())                 # 场景名
    uniq_cnt = [len(set(v)) for v in tokens.values()]  # 每场景唯一 token 数
    scenes, uniq_cnt = zip(*sorted(zip(scenes, uniq_cnt), key=lambda x: x[1]))
    scenes   = list(scenes)
    uniq_cnt = list(uniq_cnt)
    cnt = [len(tokens[s]) for s in scenes]


    for i, scene in enumerate(scenes):
        s = time.time()
        cfg["data"]["val"]["ann_file"] = f"zz_filter/pkl/{scene}.pkl"
        visualizer = Visualizer(plot_choices, args.out_dir, cfg, args.result_path)
        # indices = list(range(min(2, len(visualizer.dataset))))
        indices = list(range(min(2000, len(visualizer.dataset))))
    
        num_workers = 32
        print(f"Using {num_workers} processes for parallel execution")
        
        if num_workers > 1:
            with Pool(processes=num_workers) as pool:
                from functools import partial
                worker_func = partial(process_index, args, plot_choices, cfg)
                list(tqdm(pool.imap(worker_func, indices), total=len(indices)))
        else:
            for idx in tqdm(indices):
                visualizer.add_vis(idx)
        
        video_name = f"videos/{str(uniq_cnt[i]).zfill(2)}_{str(cnt[i]).zfill(4)}_{scene}.mp4"
        # visualizer.image2video(video_name)
        visualizer.image2video_fast(video_name)
        os.system("rm vis/combine/*")

        e = time.time()
        print("visualization time:", e - s)


if __name__ == '__main__':
    main()