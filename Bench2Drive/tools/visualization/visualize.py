import os
import glob
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed


import cv2
import numpy as np
from PIL import Image

import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset

from tools.visualization.bev_render import BEVRender
from tools.visualization.cam_render import CamRender
from tools.visualization.bevcam_render import BEVCamRender

# plot_choices = dict(
#     bev_pred = True,
#     bev_gt = False,
#     cam_pred = True,
#     cam_gt = False,
#     bevcam_pred = False,
#     det = True,
#     track = False,
#     motion = False,
#     map = True,
#     planning = True,
#     path = False,
#     target_point = True,
#     route = True,
#     speed = True,
#     det_attn_weight = True,
#     map_attn_weight = True,
# )

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
    route = False,
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
            # self.cfg = cfg
            self.dataset = build_dataset(cfg.data.val)
            # self.dataset = build_dataset(cfg.data.train)
        if (plot_choices["bev_pred"] or plot_choices["bev_pred"]) and result_path is not None :
            self.results = mmcv.load(result_path)
        else:
            self.results = None
        
        self.bev_render = BEVRender(plot_choices)
        self.cam_render = CamRender(plot_choices)
        self.bevcam_render = BEVCamRender(plot_choices)
        if planning_key is None:
            self.planning_key = cfg.get("anchor_reference_group", "spatial")
        else:
            self.planning_key = planning_key

    def add_vis(self, index, data=None, result=None):
        if data is None:
            data = self.dataset.get_data_info(index)
            data["index"] = index
        data["planning_key"] = self.planning_key
        # data["cfg"] = self.cfg
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

    def image2video_fast(self, fps=10, downsample=4):
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
        out_path = os.path.join(self.out_dir, 'video.mp4')
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
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

def process_index(args, plot_choices, index):
    visualizer = Visualizer(plot_choices, args.out_dir, args.config, args.result_path)
    visualizer.add_vis(index)

def main():
    import time
    s = time.time()
    args = parse_args()
    visualizer = Visualizer(plot_choices, args.out_dir, args.config, args.result_path)

    indices = list(range(args.start, args.end, args.interval))
    # indices = list(range(len(visualizer.dataset)))
    # indices = [1734, 4673, 9256, 9257, 9268, 12727, 12728, 12799, 12800, 12801, 12802, 12803, 17706, 20039, 20040, 20098, 20099, 20100, 20159, 20160, 22536, 22561, 22563, 22567, 22568, 23467, 23468, 23471, 23472, 23473, 24037, 24038, 24039, 24040, 24041, 24212, 24213, 24214, 24215, 24216, 25495, 25496, 25918, 25919, 25920, 27201, 30343, 30347, 30348, 30349, 32169, 32170, 32171, 32172, 32173, 32174, 32175, 32176, 32177, 32178, 32382, 32383, 32384, 37566, 39487, 41996, 49186, 53308, 63220, 64314, 64315, 64316, 64317, 64318, 64319, 64320, 64321, 64322, 64323, 64324, 64325, 64326, 64327, 64328, 64329, 64330, 64331, 64332, 64333, 64334, 64335, 64336, 64337, 64338, 64339, 64340, 64341, 64342, 64343, 64344, 64345, 64346, 64347, 64348, 64349, 64350, 64351, 64352, 64353, 64354, 64355, 64356, 64357, 64367, 64368, 64371, 64372, 67230, 67305, 67306, 67462, 67463, 67464, 82629, 82630, 82631, 82632, 89645, 89646, 89647, 90237, 90238, 90239, 90240, 90241, 90242, 90243, 90244, 90245, 90246, 90247, 90248, 90249, 90250, 90251, 90252, 90253, 90254, 90255, 90256, 90257, 90258, 91533, 91534, 92543, 92544, 96282, 96283, 96284, 96285, 96286, 100911, 100912, 100915, 100916, 101565, 101667, 106681, 106682, 106683, 106684, 106685, 106686, 106687, 106688, 106689, 114637, 116867, 116868, 116869, 116870, 116871, 116872, 116873, 116874, 116875, 116876, 116877, 116878, 116879, 116880, 116881, 116882, 116883, 116884, 116885, 116886, 116887, 116888, 116889, 116892, 116893, 121791, 121890, 121892, 121893, 121894, 121949, 121950, 124068, 124069, 124405, 124406, 124649, 124700, 133375, 136425, 138866, 140327, 143287, 143288, 143289, 143301, 143302, 143303, 143306, 144665, 144666, 144724, 146645, 148340, 148341, 148342, 148343, 148344, 148345, 148346, 148347, 148348, 148349, 148350, 148351, 149963, 149967, 149969, 149970, 149971, 149972, 149973, 149974, 149975, 149976, 149977, 149978, 149979, 149980, 149981, 152968, 154557, 154558, 154559, 154560, 154561, 157546, 157547, 157548, 157549, 157550, 157551, 158383, 158389, 158390, 158391, 158392, 158393, 159135, 161510, 163572, 164269, 164334, 164388, 166075, 167718, 167719, 167720, 167721, 167722, 167723, 171401, 171402, 171440, 171441, 171442, 171443, 171444, 171445, 171446, 171447, 171448, 171449, 171450, 171451, 171452, 171453, 171454, 171455, 171456, 171457, 171458, 171459, 171460, 171461, 171462, 171463, 171464, 171465, 171466, 171467, 171468, 171469, 171470, 171471, 171472, 171473, 171474, 171475, 171476, 171477, 171478, 171479, 171480, 171481, 171482, 171483, 171484, 171485, 171486, 171487, 171488, 171489, 171490, 171491, 171492, 171493, 171494, 171495, 171496, 171497, 171498, 171499, 171500, 171501, 171502, 171503, 171504, 171505, 171506, 171507, 171508, 171509, 171510, 171511, 171512, 171513, 171514, 171515, 171516, 171517, 171518, 171519, 171520, 171521, 171522, 171523, 171524, 171525, 171526, 171527, 171528, 171529, 171530, 171531, 171532, 171533, 171534, 171535, 171536, 171537, 171538, 171539, 171540, 171541, 171542, 171543, 171544, 171545, 171546, 171547, 171548, 171549, 171550, 171551, 171552, 171553, 171554, 171555, 171556, 171557, 171558, 171559, 171560, 171561, 171562, 171563, 171564, 171565, 171566, 171567, 171568, 171569, 171570, 171571, 171572, 171573, 171574, 171575, 171576, 171577, 171578, 171579, 171580, 171581, 171582, 171583, 171584, 171585, 171586, 171587, 171588, 171589, 171590, 171591, 171592, 171593, 171594, 171595, 171596, 171597, 171598, 171599, 171600, 171601, 171602, 171603, 171604, 171605, 171606, 171607, 171608, 171609, 171610, 171611, 171612, 171613, 171614, 171615, 171616, 171617, 171618, 171619, 171620, 171621, 171622, 171623, 171624, 171625, 171626, 171627, 171628, 171629, 171630, 171631, 171632, 171633, 171634, 171635, 171636, 171637, 171638, 171639, 171640, 171641, 171642, 171651, 171652, 171653, 171654, 171655, 171656, 171657, 171658, 171659, 171660, 171661, 171662, 171663, 171664, 171665, 171666, 171667, 171668, 171669, 171670, 171671, 171672, 171673, 171674, 171675, 171676, 171677, 171678, 171679, 171680, 171681, 171682, 171683, 171684, 171685, 171686, 171687, 171688, 171689, 171690, 171691, 171692, 171693, 171694, 171695, 171696, 171697, 171698, 171699, 171700, 171701, 171702, 171703, 171704, 171705, 171706, 171707, 171708, 171709, 171710, 171711, 171712, 171713, 171714, 171715, 171716, 171717, 171718, 171719, 171720, 171721, 171722, 171723, 171724, 171725, 171726, 171727, 171728, 171729, 171730, 171731, 171732, 171733, 171734, 171735, 171736, 171737, 171738, 171739, 171740, 171741, 171742, 171743, 171744, 171745, 171746, 171747, 171748, 171749, 171750, 171751, 171752, 171753, 171754, 171755, 171756, 171757, 171758, 171759, 171760, 171761, 171762, 171763, 171764, 171765, 171766, 171767, 171768, 171769, 171770, 171771, 171772, 171773, 171774, 171775, 171776, 171777, 171778, 171779, 171780, 171781, 171782, 171783, 171784, 171785, 171786, 171787, 171788, 171789, 171790, 171791, 171792, 171793, 171794, 171795, 171796, 171797, 171798, 171799, 171800, 171801, 171802, 171803, 171804, 171805, 171806, 171807, 171808, 171809, 171810, 171811, 171812, 171813, 171814, 171815, 171816, 171817, 171818, 171819, 171820, 171821, 171822, 171823, 171824, 171825, 171826, 171827, 171828, 171829, 171830, 171831, 171832, 171835, 171836, 171837, 171838, 171839, 171840, 171841, 171842, 171843, 171859, 171860, 171861, 171862, 171863, 171864, 172037, 173002, 173005, 173006, 173007, 173008, 173009, 173010, 173011, 177396, 177397, 177398, 177399, 177400, 177401, 177402, 177403, 177404, 177405, 181810, 181811, 181813, 181814, 182933, 182992, 183048, 183460, 183467, 183468, 183469, 183470, 183471, 183472, 183473, 183474, 183475, 183476, 183477, 183478, 183479, 183480, 183481, 183482, 183483, 183484, 183485, 183486, 183487, 183488, 183489, 183490, 183491, 183492, 183493, 183494, 183495, 183496, 183497, 183498, 183499, 183500, 183501, 183502, 183503, 183504, 183505, 183506, 183507, 183508, 183509, 183510, 183511, 183512, 183513, 183514, 183515, 184002, 184003, 190124, 192989, 192993, 192994, 192995, 195646, 195647, 198527, 199677, 199678, 199679, 200541, 200544, 201550, 201553, 202184, 202188, 202189, 209701, 210623, 210624, 226451]
    # indices = [3592, 3593, 3594, 3595, 3596, 3679, 3680, 3681, 3682, 3683, 3684, 3685, 3686, 3687, 3688, 3689, 3690, 4067, 4068, 4069, 4070, 4071, 4072, 4073, 4074, 4075, 5885, 5886, 5887, 5888, 5889, 5890, 5891, 5892, 5893, 5894, 5991, 5992, 5993, 5994, 6319, 6320, 6321, 6322, 6323, 6324, 6325, 6326, 6327, 6892, 6893, 6894, 6895, 6896, 6897, 7262, 7263, 7264, 7265, 7266, 7267, 7268, 7466, 7467, 7468, 7469, 7470, 7471, 7472, 7473, 7474, 7498, 7499, 7500, 7501, 7502, 7503, 7504, 7505, 7506, 7507, 7508, 7509, 7510, 7511, 7512, 7513, 7706, 7707, 7708, 7709, 7710, 7711, 7712, 7713, 7714, 7715, 7716, 7717, 7925]
    
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
    
    visualizer.image2video_fast()

    e = time.time()
    print("visualization time:", e - s)


if __name__ == '__main__':
    main()