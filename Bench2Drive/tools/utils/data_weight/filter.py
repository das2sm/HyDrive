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
        config=None,
        result_path=None,
        planning_key=None,
    ):
        self.out_dir = out_dir
        self.combine_dir = os.path.join(self.out_dir, 'combine')
        os.makedirs(self.combine_dir, exist_ok=True)
        
        if config is not None:
            cfg = Config.fromfile(config)
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
def get_infraction_status(record):
    for infraction,  value in record['infractions'].items():
        if infraction == "min_speed_infractions":
            continue
        elif len(value) > 0:
            return True
    return False
def main():
    # cfg = Config.fromfile("zz_filter/id_71_motion_head_v5_baseline.py")
    # dataset = build_dataset(cfg.data.train)
    # tokens = dict()
    # scenes = []
    # for info in dataset.data_infos:
    #     token = info["token"][:-5]
    #     scene = info["token"].split("/")[1].split("_")[0]
    #     scenes.append(scene)
    #     if scene not in tokens:
    #         tokens[scene] = [token]
    #     else:
    #         tokens[scene].append(token)
    # scene_set = set(scenes)
    # save_file = 'tokens_dict.pkl'
    # with open(save_file, 'wb') as f:
    #     pickle.dump(tokens, f)
    
    save_file = 'tokens_dict.pkl'      # 同上路径
    with open(save_file, 'rb') as f:
        tokens = pickle.load(f)
    scene_set = ['ParkingExit', 'NonSignalizedJunctionLeftTurnEnterFlow', 'YieldToEmergencyVehicle', 'HardBreakRoute', 'OppositeVehicleTakingPriority', 'MergerIntoSlowTrafficV2', 'ParkedObstacle', 'SignalizedJunctionLeftTurnEnterFlow', 'HighwayCutIn', 'VanillaNonSignalizedTurnEncounterStopsign', 'LaneChange', 'DynamicObjectCrossing', 'InterurbanActorFlow', 'VehicleOpensDoorTwoWays', 'TJunction', 'ParkedObstacleTwoWays', 'MergerIntoSlowTraffic', 'SignalizedJunctionRightTurn', 'VehicleTurningRoute', 'VehicleTurningRoutePedestrian', 'InvadingTurn', 'HighwayExit', 'ParkingCrossingPedestrian', 'ConstructionObstacleTwoWays', 'VanillaSignalizedTurnEncounterGreenLight', 'NonSignalizedJunctionLeftTurn', 'InterurbanAdvancedActorFlow', 'HazardAtSideLaneTwoWays', 'SignalizedJunctionLeftTurn', 'PedestrianCrossing', 'OppositeVehicleRunningRedLight', 'EnterActorFlow', 'StaticCutIn', 'ConstructionObstacle', 'HazardAtSideLane', 'NonSignalizedJunctionRightTurn', 'ParkingCutIn', 'Accident', 'VanillaSignalizedTurnEncounterRedLight', 'AccidentTwoWays', 'BlockedIntersection', 'CrossingBicycleFlow', 'ControlLoss']
    for scene, tok in tokens.items():
        print(scene, len(set(tok)), len(tok))

    ######## save infos by scene
    # cfg = Config.fromfile("zz_filter/id_71_motion_head_v5_baseline.py")
    # dataset = build_dataset(cfg.data.train)
    # infos=dict()
    # for info in dataset.data_infos:
    #     scene = info["token"].split("/")[1].split("_")[0]
    #     if scene not in infos:
    #         infos[scene] = [info]
    #     else:
    #         infos[scene].append(info)
    # for scene in scene_set:
    #     with open(f"zz_filter/pkl/{scene}.pkl", 'wb') as f:
    #         pickle.dump(infos[scene], f)

    ########################## scene cnt
    scenes = list(tokens.keys())                 # 场景名
    uniq_cnt = [len(set(v)) for v in tokens.values()]  # 每场景唯一 token 数
    scenes, uniq_cnt = zip(*sorted(zip(scenes, uniq_cnt), key=lambda x: x[1]))
    scenes   = list(scenes)
    uniq_cnt = list(uniq_cnt)
    cnt = [len(tokens[s]) for s in scenes] 
    import ipdb; ipdb.set_trace()

    '''
    # 画图
    plt.figure(figsize=(max(6, len(scenes)*0.5), 4))   # 场景多就拉宽
    bars = plt.bar(scenes, uniq_cnt, color='steelblue', edgecolor='black')

    # 让横轴标签可旋转，避免重叠
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('scene count')
    plt.title('scene count per scene')
    plt.tight_layout()
    # 保存
    save_path = 'scene_cnt.png'
    plt.savefig(save_path, dpi=300)
    print('figure saved ->', os.path.abspath(save_path))
    '''

    '''
    ########################### frame cnt
    cnt = [len(tokens[s]) for s in scenes] 

    # 画图
    plt.figure(figsize=(max(6, len(scenes)*0.5), 4))   # 场景多就拉宽
    bars = plt.bar(scenes, cnt, color='steelblue', edgecolor='black')

    # 让横轴标签可旋转，避免重叠
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('frame token count')
    plt.title('frame count per scene')
    plt.tight_layout()
    # 保存
    save_path = 'frame_cnt.png'
    plt.savefig(save_path, dpi=300)
    print('figure saved ->', os.path.abspath(save_path))


    ########### drivingscore
    with open("zz_filter/merged.json", 'r') as f:
        data = json.load(f)
    records = data["_checkpoint"]["records"]
    ds = dict()
    sn = []
    for record in records:
        record["scenario_name"] = record["scenario_name"].replace("T_Junction", "TJunction")
        sn.append(record["scenario_name"])
        scene = record["scenario_name"].split("_")[0]
        if scene not in ds:
            ds[scene] = [record['scores']["score_composed"]]
        else:
            ds[scene].append(record['scores']["score_composed"])
    ds["LaneChange"] = [0]
       # 画图
    cnt = [sum(ds[s])/len(ds[s]) for s in scenes if s in ds] 
    # import ipdb; ipdb.set_trace()
    plt.figure(figsize=(max(6, len(scenes)*0.5), 4))   # 场景多就拉宽
    bars = plt.bar(scenes, cnt, color='steelblue', edgecolor='black')

    # 让横轴标签可旋转，避免重叠
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('ds count')
    plt.title('ds count per scene')
    plt.tight_layout()
    # 保存
    save_path = 'ds_cnt.png'
    plt.savefig(save_path, dpi=300)
    print('figure saved ->', os.path.abspath(save_path))
    '''


    ########### sr
    with open("zz_filter/merged.json", 'r') as f:
        data = json.load(f)
    records = data["_checkpoint"]["records"]
    sr = dict()
    sn = []
    for record in records:
        record["scenario_name"] = record["scenario_name"].replace("T_Junction", "TJunction")
        sn.append(record["scenario_name"])
        scene = record["scenario_name"].split("_")[0]
        if (record["status"] == 'Completed' or record["status"] == "Perfect") and not get_infraction_status(record):
            status=1
        else:
            status = 0
        if scene not in sr:
            sr[scene] = [status]
        else:
            sr[scene].append(status)
    sr["LaneChange"] = [0]
       # 画图
    cnt = [sum(sr[s])/len(sr[s]) for s in scenes if s in sr] 
    # import ipdb; ipdb.set_trace()
    plt.figure(figsize=(max(6, len(scenes)*0.5), 4))   # 场景多就拉宽
    bars = plt.bar(scenes, cnt, color='steelblue', edgecolor='black')

    # 让横轴标签可旋转，避免重叠
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('sr count')
    plt.title('sr count per scene')
    plt.tight_layout()
    # 保存
    save_path = 'sr_cnt.png'
    plt.savefig(save_path, dpi=300)
    print('figure saved ->', os.path.abspath(save_path))


if __name__ == '__main__':
    main()