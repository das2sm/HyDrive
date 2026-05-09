import os
import glob
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
from PIL import Image

import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset

from tools.visualization.bev_render_road import BEVRender
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
    target_point = True,
    route = True,
)

# plot_choices = dict(
#     bev_pred = False,
#     bev_gt = True,
#     cam_pred = False,
#     cam_gt = True,
#     bevcam_pred = False,
#     det = True,
#     track = False,
#     motion = False,
#     map = True,
#     planning = True,
#     path = False,
#     target_point = True,
#     route = True,
# )

class Visualizer:
    def __init__(
        self,
        plot_choices,
        out_dir,
        config=None,
        result_path=None,
    ):
        self.out_dir = out_dir
        self.combine_dir = os.path.join(self.out_dir, 'combine')
        os.makedirs(self.combine_dir, exist_ok=True)
        
        if config is not None:
            cfg = Config.fromfile(config)
            # self.dataset = build_dataset(cfg.data.val)
            self.dataset = build_dataset(cfg.data.train)
        if (plot_choices["bev_pred"] or plot_choices["bev_pred"]) and result_path is not None :
            self.results = mmcv.load(result_path)
        else:
            self.results = None
        
        self.bev_render = BEVRender(plot_choices)
        self.cam_render = CamRender(plot_choices)
        self.bevcam_render = BEVCamRender(plot_choices)

    def add_vis(self, index, data=None, result=None):
        if data is None:
            data = self.dataset.get_data_info(index)
            data["index"] = index
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

    def image2video(self, fps=5, downsample=4):
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
        out_path = os.path.join(self.out_dir, 'video.mp4')
        out = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


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

def process_index(args, plot_choices, index):
    visualizer = Visualizer(plot_choices, args.out_dir, args.config, args.result_path)
    visualizer.add_vis(index)

def main():
    import time
    s = time.time()
    args = parse_args()
    visualizer = Visualizer(plot_choices, args.out_dir, args.config, args.result_path)
    
    indices = list(range(args.start, args.end, args.interval))
    
    num_workers = args.num_workers
    print(f"Using {num_workers} processes for parallel execution")
    
    if num_workers > 1:
        with Pool(processes=num_workers) as pool:
            from functools import partial
            worker_func = partial(process_index, args, plot_choices)
            list(tqdm(pool.imap(worker_func, indices), total=len(indices)))
    else:
        for idx in tqdm(indices):
            visualizer.add_vis(idx)
    
    visualizer.image2video()

    e = time.time()
    print("visualization time:", e - s)


if __name__ == '__main__':
    main()