import os
from io import BytesIO

import numpy as np
import cv2

import matplotlib
import matplotlib.pyplot as plt

from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners

# nusc
# CMD_LIST = ['Turn Right', 'Turn Left', 'Go Straight']
# COLOR_VECTORS = ['cornflowerblue', 'royalblue', 'slategrey']
# b2d
CMD_LIST = ['Turn Left', 'Turn Right', 'Go Straight', 'Lane Follow', 'CHANGELANELEFT', 'CHANGELANERIGHT']
# COLOR_VECTORS = [
#     'cornflowerblue', 'royalblue', 'slategrey', 
#     'lightseagreen', 'darkseagreen', 'bisque',
# ]
COLOR_VECTORS = [
    '#FF0000',  # 纯红
    '#008000',  # 深绿
    '#FFA500',  # 橙色
    '#0000FF',  # 纯蓝
    '#800080',  # 紫色
    '#00FFFF'   # 青色
]
SCORE_THRESH = 0.3
MAP_SCORE_THRESH = 0.3
color_mapping = np.asarray([
    [0, 0, 0],
    [255, 179, 0],
    [128, 62, 117],
    [255, 104, 0],
    [166, 189, 215],
    [193, 0, 32],
    [206, 162, 98],
    [129, 112, 102],
    [0, 125, 52],
    [246, 118, 142],
    [0, 83, 138],
    [255, 122, 92],
    [83, 55, 122],
    [255, 142, 0],
    [179, 40, 81],
    [244, 200, 0],
    [127, 24, 13],
    [147, 170, 0],
    [89, 51, 21],
    [241, 58, 19],
    [35, 44, 22],
    [112, 224, 255],
    [70, 184, 160],
    [153, 0, 255],
    [71, 255, 0],
    [255, 0, 163],
    [255, 204, 0],
    [0, 255, 235],
    [255, 0, 235],
    [255, 0, 122],
    [255, 245, 0],
    [10, 190, 212],
    [214, 255, 0],
    [0, 204, 255],
    [20, 0, 255],
    [255, 255, 0],
    [0, 153, 255],
    [0, 255, 204],
    [41, 255, 0],
    [173, 0, 255],
    [0, 245, 255],
    [71, 0, 255],
    [0, 255, 184],
    [0, 92, 255],
    [184, 255, 0],
    [255, 214, 0],
    [25, 194, 194],
    [92, 0, 255],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [255, 184, 6],
    [10, 255, 71],
    [255, 41, 10],
    [7, 255, 255],
    [224, 255, 8],
    [102, 8, 255],
    [255, 61, 6],
    [255, 194, 7],
    [0, 255, 20],
    [255, 8, 41],
    [255, 5, 153],
    [6, 51, 255],
    [235, 12, 255],
    [160, 150, 20],
    [0, 163, 255],
    [140, 140, 140],
    [250, 10, 15],
    [20, 255, 0],
]) / 255


def plt_fig_to_cv2_image(fig=None, dpi=100):
    """
    将 Matplotlib 图形转换为 OpenCV 图像 (BGR格式)
    
    参数:
        fig: matplotlib figure 对象 (默认使用当前图形)
        dpi: 图像分辨率 (默认100)
    
    返回:
        cv2_image: OpenCV 格式的图像 (numpy数组, BGR格式)
    """
    # 获取或创建图形
    if fig is None:
        fig = plt.gcf()
    
    # 创建内存缓冲区
    buffer = BytesIO()
    
    # 保存图形到内存缓冲区 (RGB格式)
    fig.savefig(buffer)
    buffer.seek(0)
    
    # 将缓冲区数据转换为 numpy 数组
    img_array = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    buffer.close()
    
    # 使用 OpenCV 解码图像
    cv2_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    return cv2_image

class BEVRender:
    def __init__(
        self, 
        plot_choices,
        xlim = 40,
        ylim = 40,
    ):
        self.plot_choices = plot_choices
        self.xlim = xlim
        self.ylim = ylim

    def reset_canvas(self):
        plt.close()
        self.fig, self.axes = plt.subplots(1, 1, figsize=(20, 20))
        self.axes.set_xlim(- self.xlim, self.xlim)
        self.axes.set_ylim(- self.ylim, self.ylim)
        # self.axes.set_xlim(-15, 15)
        # self.axes.set_ylim(-5, 30)
        self.axes.axis('off')

    def render(
        self,
        data, 
        result,
    ):
        if self.plot_choices["bev_gt"]:
            self.reset_canvas()
            self.draw_detection_gt(data)
            self.draw_motion_gt(data)
            self.draw_map_gt(data)
            self.draw_planning_gt(data)
            self.draw_path_gt(data)
            self._render_sdc_car()
            self._render_command(data)
            self._render_target_point(data)
            self._render_route(data)
            self._render_legend()
            # import pickle
            # with open("on_road_test.pkl", "rb") as f:
            #     self.on_road = pickle.load(f)
            self.on_road = np.load("data/infos/on_road_1024.npy")
            self.draw_plan_anchor(data)
            bev_gt = self.get_fig()
        else:
            bev_gt = None

        if self.plot_choices["bev_pred"]:
            self.reset_canvas()
            self.draw_detection_pred(result)
            self.draw_track_pred(result)
            self.draw_motion_pred(result)
            self.draw_map_pred(result)
            self.draw_planning_pred(data, result)
            self.draw_path_pred(data, result)
            self._render_sdc_car()
            self._render_command(data)
            self._render_target_point(data)
            self._render_route(data)
            self._render_legend()
            bev_pred = self.get_fig()
        else:
            bev_pred = None

        return bev_gt, bev_pred

    def get_fig(self):
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        return plt_fig_to_cv2_image(self.fig)

    def draw_detection_gt(self, data):
        if not self.plot_choices['det']:
            return

        for i in range(data['gt_labels_3d'].shape[0]):
            label = data['gt_labels_3d'][i]
            if label == -1: 
                continue
            color = color_mapping[data['instance_inds'][i] % len(color_mapping)]

            # draw corners
            corners = box3d_to_corners(data['gt_bboxes_3d'])[i, [0, 3, 7, 4, 0]]
            x = corners[:, 0]
            y = corners[:, 1]
            self.axes.plot(x, y, color=color, linewidth=3, linestyle='-')

            # draw line to indicate forward direction
            forward_center = np.mean(corners[2:4], axis=0)
            center = np.mean(corners[0:4], axis=0)
            x = [forward_center[0], center[0]]
            y = [forward_center[1], center[1]]
            self.axes.plot(x, y, color=color, linewidth=3, linestyle='-')

    def draw_detection_pred(self, result):
        if not (self.plot_choices['det'] and "boxes_3d" in result):
            return

        bboxes = result['boxes_3d']
        for i in range(result['labels_3d'].shape[0]):
            score = result['scores_3d'][i]
            if score < SCORE_THRESH: 
                continue
            color = color_mapping[result['instance_ids'][i] % len(color_mapping)]

            # draw corners
            corners = box3d_to_corners(bboxes)[i, [0, 3, 7, 4, 0]]
            x = corners[:, 0]
            y = corners[:, 1]
            self.axes.plot(x, y, color=color, linewidth=3, linestyle='-')

            # draw line to indicate forward direction
            forward_center = np.mean(corners[2:4], axis=0)
            center = np.mean(corners[0:4], axis=0)
            x = [forward_center[0], center[0]]
            y = [forward_center[1], center[1]]
            self.axes.plot(x, y, color=color, linewidth=3, linestyle='-')

    def draw_track_pred(self, result):
        if not (self.plot_choices['track'] and "anchor_queue" in result):
            return
        
        temp_bboxes = result["anchor_queue"]
        period = result["period"]
        bboxes = result['boxes_3d']
        for i in range(result['labels_3d'].shape[0]):
            score = result['scores_3d'][i]
            if score < SCORE_THRESH: 
                continue
            color = color_mapping[result['instance_ids'][i] % len(color_mapping)]
            center = bboxes[i, :3]
            centers = [center]
            for j in range(period[i]):
                # draw corners
                corners = box3d_to_corners(temp_bboxes[:, -1-j])[i, [0, 3, 7, 4, 0]]
                x = corners[:, 0]
                y = corners[:, 1]
                self.axes.plot(x, y, color=color, linewidth=2, linestyle='-')

                # draw line to indicate forward direction
                forward_center = np.mean(corners[2:4], axis=0)
                center = np.mean(corners[0:4], axis=0)
                x = [forward_center[0], center[0]]
                y = [forward_center[1], center[1]]
                self.axes.plot(x, y, color=color, linewidth=2, linestyle='-')
                centers.append(center)

            centers = np.stack(centers)
            xs = centers[:, 0]
            ys = centers[:, 1]
            self.axes.plot(xs, ys, color=color, linewidth=2, linestyle='-')

    def draw_motion_gt(self, data):
        if not self.plot_choices['motion']:
            return

        for i in range(data['gt_labels_3d'].shape[0]):
            label = data['gt_labels_3d'][i]
            if label == -1: 
                continue
            color = color_mapping[i % len(color_mapping)]
            vehicle_id_list = [0, 1, 2, 3, 4, 6, 7]
            if label in vehicle_id_list:
                dot_size = 150
            else:
                dot_size = 25

            center = data['gt_bboxes_3d'][i, :2]
            masks = data['gt_agent_fut_masks'][i].astype(bool)
            if masks[0] == 0:
                continue
            trajs = data['gt_agent_fut_trajs'][i][masks]
            trajs = trajs.cumsum(axis=0) + center
            trajs = np.concatenate([center.reshape(1, 2), trajs], axis=0)
            
            self._render_traj(trajs, traj_score=1.0,
                            colormap='autumn', dot_size=dot_size)

    def draw_motion_pred(self, result, top_k=3):
        if not (self.plot_choices['motion'] and "trajs_3d" in result):
            return
        
        bboxes = result['boxes_3d']
        labels = result['labels_3d']
        for i in range(result['labels_3d'].shape[0]):
            score = result['scores_3d'][i]
            if score < SCORE_THRESH: 
                continue
            label = labels[i]
            vehicle_id_list = [0, 1, 2, 3, 4, 6, 7]
            if label in vehicle_id_list:
                dot_size = 150
            else:
                dot_size = 25

            traj_score = result['trajs_score'][i].numpy()
            traj = result['trajs_3d'][i].numpy()
            num_modes = len(traj_score)
            center = bboxes[i, :2][None, None].repeat(num_modes, 1, 1).numpy()
            traj = np.concatenate([center, traj], axis=1)

            sorted_ind = np.argsort(traj_score)[::-1]
            sorted_traj = traj[sorted_ind, :, :2]
            sorted_score = traj_score[sorted_ind]
            norm_score = np.exp(sorted_score[0])

            for j in range(top_k - 1, -1, -1):
                viz_traj = sorted_traj[j]
                traj_score = np.exp(sorted_score[j])/norm_score
                self._render_traj(viz_traj, traj_score=traj_score,
                                colormap='autumn', dot_size=dot_size)
    
    def draw_map_gt(self, data):
        if not self.plot_choices['map']:
            return
        
        vectors = data['map_infos']
        for label, vector_list in vectors.items():
            color = COLOR_VECTORS[label]
            for vector in vector_list:
                pts = vector[:, :2]
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])
                self.axes.plot(x, y, color=color, linewidth=3, marker='o', linestyle='-', markersize=7)

    def draw_map_pred(self, result):
        if not (self.plot_choices['map'] and "vectors" in result):
            return

        for i in range(result['scores'].shape[0]):
            score = result['scores'][i]
            if  score < MAP_SCORE_THRESH:
                continue
            color = COLOR_VECTORS[result['labels'][i]]
            pts = result['vectors'][i]
            x = pts[:, 0]
            y = pts[:, 1]
            plt.plot(x, y, color=color, linewidth=3, marker='o', linestyle='-', markersize=7)

    def draw_planning_gt(self, data):
        if not self.plot_choices['planning']:
            return
        # draw planning gt
        masks = data['gt_ego_fut_masks'].astype(bool)
        if masks[0] != 0:
            plan_traj = data['gt_ego_fut_trajs'][masks]
            plan_traj[abs(plan_traj) < 0.01] = 0.0
            plan_traj = plan_traj.cumsum(axis=0)
            plan_traj = np.concatenate((np.zeros((1, plan_traj.shape[1])), plan_traj), axis=0)
            # self.axes.plot(plan_traj[:,0], plan_traj[:,1], color="yellow", linewidth=40, marker='o', linestyle='-', markersize=50)
            self._render_traj(plan_traj, traj_score=1.0,
                colormap='winter', dot_size=70)

    def draw_planning_pred_ori(self, data, result, top_k=6):
        if not (self.plot_choices['planning'] and "planning" in result):
            return

        if self.plot_choices['track'] and "ego_anchor_queue" in result:
            ego_temp_bboxes = result["ego_anchor_queue"]
            ego_period = result["ego_period"]
            for j in range(ego_period[0]):
                # draw corners
                corners = box3d_to_corners(ego_temp_bboxes[:, -1-j])[0, [0, 3, 7, 4, 0]]
                x = corners[:, 0]
                y = corners[:, 1]
                self.axes.plot(x, y, color='mediumseagreen', linewidth=2, linestyle='-')

                # draw line to indicate forward direction
                forward_center = np.mean(corners[2:4], axis=0)
                center = np.mean(corners[0:4], axis=0)
                x = [forward_center[0], center[0]]
                y = [forward_center[1], center[1]]
                self.axes.plot(x, y, color='mediumseagreen', linewidth=2, linestyle='-')

        plan_trajs = result['planning'].cpu().numpy()
        num_cmd, num_mode = plan_trajs.shape[:2]
        plan_trajs = np.concatenate((np.zeros((num_cmd, num_mode, 1, 2)), plan_trajs), axis=2)
        plan_score = result['planning_score'].cpu().numpy()

        cmd = data['gt_ego_fut_cmd'].argmax()
        plan_trajs = plan_trajs[cmd]
        plan_score = plan_score[cmd]

        sorted_ind = np.argsort(plan_score)[::-1]
        sorted_traj = plan_trajs[sorted_ind, :, :2]
        sorted_score = plan_score[sorted_ind]
        norm_score = np.exp(sorted_score[0])

        for j in range(top_k - 1, -1, -1):
            viz_traj = sorted_traj[j]
            traj_score = np.exp(sorted_score[j]) / norm_score
            self._render_traj(viz_traj, traj_score=traj_score,
                            colormap='winter', dot_size=50)

    def draw_planning_pred_v1(self, data, result, top_k=6):
        if not (self.plot_choices['planning'] and "planning" in result):
            return

        if self.plot_choices['track'] and "ego_anchor_queue" in result:
            ego_temp_bboxes = result["ego_anchor_queue"]
            ego_period = result["ego_period"]
            for j in range(ego_period[0]):
                corners = box3d_to_corners(ego_temp_bboxes[:, -1-j])[0, [0, 3, 7, 4, 0]]
                x, y = corners[:, 0], corners[:, 1]
                self.axes.plot(x, y, color='mediumseagreen', linewidth=2, linestyle='-')
                forward_center = np.mean(corners[2:4], axis=0)
                center = np.mean(corners[0:4], axis=0)
                self.axes.plot([forward_center[0], center[0]],
                            [forward_center[1], center[1]],
                            color='mediumseagreen', linewidth=2, linestyle='-')

        plan_trajs = result['planning'].cpu().numpy()
        num_cmd, num_mode = plan_trajs.shape[:2]
        plan_trajs = np.concatenate((np.zeros((num_cmd, num_mode, 1, 2)), plan_trajs), axis=2)
        plan_score = result['planning_score'].cpu().numpy()

        cmd = data['gt_ego_fut_cmd'].argmax()
        plan_trajs = plan_trajs[cmd]
        plan_score = plan_score[cmd]

        sorted_ind = np.argsort(plan_score)[::-1]
        sorted_traj = plan_trajs[sorted_ind, :, :2]
        sorted_score = plan_score[sorted_ind]

        # 归一化概率
        norm_score = np.exp(sorted_score)
        norm_score = norm_score / norm_score.sum()

        # colormap 取前 top_k 种颜色
        cmap = plt.get_cmap('tab10')
        colors = cmap(np.linspace(0, 1, top_k))

        for j in range(top_k - 1, -1, -1):
            viz_traj = sorted_traj[j]
            traj_score = norm_score[j]
            self._render_traj(viz_traj, traj_score=traj_score,
                            colormap='winter', dot_size=50)

            # 画最后一个点并标注概率
            x_last, y_last = viz_traj[-1]
            self.axes.scatter(x_last, y_last, color=colors[j], s=100, zorder=5)
            self.axes.text(x_last, y_last + 0.5,
                        f'{sorted_score[j]:.2f}',
                        color=colors[j],
                        fontsize=9,
                        ha='center',
                        va='bottom',
                        zorder=6)

    def draw_planning_pred(self, data, result, top_k=6):
        if self.plot_choices['track'] and "ego_anchor_queue" in result:
            ego_temp_bboxes = result["ego_anchor_queue"]
            ego_period = result["ego_period"]
            for j in range(ego_period[0]):
                corners = box3d_to_corners(ego_temp_bboxes[:, -1 - j])[0, [0, 3, 7, 4, 0]]
                x, y = corners[:, 0], corners[:, 1]
                self.axes.plot(x, y, color='mediumseagreen', linewidth=2, linestyle='-')
                forward_center = np.mean(corners[2:4], axis=0)
                center = np.mean(corners[0:4], axis=0)
                self.axes.plot([forward_center[0], center[0]],
                            [forward_center[1], center[1]],
                            color='mediumseagreen', linewidth=2, linestyle='-')

        if not (self.plot_choices['planning'] and "planning" in result):
            return
        # 处理多模态轨迹
        plan_trajs = result['planning'].cpu().numpy()
        num_cmd, num_mode = plan_trajs.shape[:2]
        plan_trajs = np.concatenate(
            (np.zeros((num_cmd, num_mode, 1, 2)), plan_trajs), axis=2)  # (num_cmd, num_mode, T, 2)

        plan_score = result['planning_score'].cpu().numpy()  # (num_cmd, num_mode)

        cmd = data['gt_ego_fut_cmd'].argmax()
        if num_cmd == 1:
            cmd *= 0
        plan_trajs = plan_trajs[cmd]        # (num_mode, T, 2)
        plan_score = plan_score[cmd]        # (num_mode,)

        if num_mode < 20:
            colors = ['#e6194b', '#3cb44b', '#f032e6', '#4363d8', '#f58231', '#911eb4']
            idx = np.argsort(plan_score)[::-1][:top_k][::-1]
            top_trajs = plan_trajs[idx]         # (top_k, T, 2)
            top_score = plan_score[idx]         # (top_k,)
            top_prob = top_score
        else:
            top_k = 50
            idx = np.argsort(plan_score)[::-1][:top_k][::-1]
            top_trajs = plan_trajs[idx]         # (top_k, T, 2)
            top_score = plan_score[idx]         # (top_k,)
            top_prob = top_score
            # top_prob = np.exp(top_score)
            # top_prob /= top_prob.sum()
            color = ['#8B0000', '#A52A2A', '#CD5C5C', '#DC143C', '#FF6347', '#FFA07A']
            # cmap = plt.get_cmap('OrRd')
            colors = []
            for score in top_prob:
                if score >= 0.6:
                    idx = 0
                else:
                    idx = max(0, min(5, int((0.6 - score) / 0.1)))
                colors.append(color[idx])
        
        # 画轨迹
        for i, (traj, prob, color) in enumerate(zip(top_trajs, top_prob, colors)):

            # 最后一个点坐标
            x_last, y_last = traj[-1]

            # 概率最高的轨迹特殊处理
            if i == 0:          # idx[0] 对应概率最大
                self.axes.plot(traj[:, 0], traj[:, 1], color=color, linewidth=6)
                self.axes.scatter(x_last, y_last, color=color, s=300, zorder=5)
                if num_mode < 10:
                    self.axes.text(x_last + 2.8, y_last,
                                f'{prob:.2f}',
                                color=color,
                                fontsize=40,
                                ha='center',
                                va='bottom',
                                fontweight='bold',   # 加粗
                                zorder=6)
            else:
                self.axes.plot(traj[:, 0], traj[:, 1], color=color, linewidth=4)
                self.axes.scatter(x_last, y_last, color=color, s=200, zorder=5)
                if num_mode < 10:
                    self.axes.text(x_last + 2.8, y_last,
                                f'{prob:.2f}',
                                color=color,
                                fontsize=40,
                                ha='center',
                                va='bottom',
                                zorder=6)

    def draw_path_gt(self, data):
        if not self.plot_choices['path']:
            return

        # draw planning gt
        masks = data['path_mask'].astype(bool)
        if masks[0] != 0:
            path = data["path"][masks]
            x = np.array([pt[0] for pt in path])
            y = np.array([pt[1] for pt in path])
            self.axes.plot(x, y, color='k', linewidth=3, marker='o', linestyle='-', markersize=7)

    def draw_path_pred(self, data, result):
        if not (self.plot_choices['path'] and "path" in result):
            return

        # draw planning pred
        path = result["path"]
        x = np.array([pt[0] for pt in path])
        y = np.array([pt[1] for pt in path])
        self.axes.plot(x, y, color='k', linewidth=3, marker='o', linestyle='-', markersize=7)

    def _render_traj(
        self, 
        future_traj, 
        traj_score=1, 
        colormap='autumn', 
        points_per_step=20, 
        dot_size=25
    ):
        total_steps = (len(future_traj) - 1) * points_per_step + 1
        dot_colors = matplotlib.colormaps[colormap](
            np.linspace(0, 1, total_steps))[:, :3]
        dot_colors = dot_colors * traj_score + \
            (1 - traj_score) * np.ones_like(dot_colors)
        total_xy = np.zeros((total_steps, 2))
        for i in range(total_steps - 1):
            unit_vec = future_traj[i // points_per_step +
                                   1] - future_traj[i // points_per_step]
            total_xy[i] = (i / points_per_step - i // points_per_step) * \
                unit_vec + future_traj[i // points_per_step]
        total_xy[-1] = future_traj[-1]
        self.axes.scatter(
            total_xy[:, 0], total_xy[:, 1], c=dot_colors, s=dot_size)

    def _render_sdc_car(self):
        sdc_car_png = cv2.imread('resources/sdc_car.png')
        sdc_car_png = cv2.cvtColor(sdc_car_png, cv2.COLOR_BGR2RGB)
        im = self.axes.imshow(sdc_car_png, extent=(-1, 1, -2, 2))
        im.set_zorder(2)

    def _render_legend(self):
        legend = cv2.imread('resources/legend.png')
        legend = cv2.cvtColor(legend, cv2.COLOR_BGR2RGB)
        self.axes.imshow(legend, extent=(15, 40, -40, -30))

    def _render_command(self, data):
        cmd = data['gt_ego_fut_cmd'].argmax()
        # self.axes.text(-38, -38, CMD_LIST[cmd] + " " + str(data["index"]), fontsize=60)
        # self.axes.text(-17, -4.5, CMD_LIST[cmd] + " " + str(data["index"]), fontsize=60)
        self.axes.text(-38, -38, CMD_LIST[cmd] + " " + str(data["index"]) + " " + data["town_name"], fontsize=60)

    def _render_target_point(self, data):
        if not self.plot_choices['target_point']:
            return
        
        if "tp_near" in data:
            self.axes.scatter(
                data['tp_near'][0], data['tp_near'][1], c='k', s=300)
        if "tp_far" in data:
            self.axes.scatter(
                data['tp_far'][0], data['tp_far'][1], c='g', s=300)

    def _render_route(self, data):
        if not (self.plot_choices['route'] and "route" in data):
            return

        self.axes.scatter(
            data["route"][:, 0], data["route"][:, 1], c='k', s=200)

    def draw_plan_anchor(self, data):
        anchors = np.load("data/kmeans/kmeans_plan_1024_b2d_kmeans.npy")
        on_road_vecs = self.on_road[data["index"]]
        # for anchor, on_road in zip(anchors, on_road_vecs):
        #     if on_road:
        #          self.axes.plot(anchor[:,0], anchor[:, 1], c="blue")
        #     else:
        #          self.axes.plot(anchor[:,0], anchor[:, 1], c="red")
        for anchor in anchors[on_road_vecs==0]:
            self.axes.plot(anchor[:,0], anchor[:, 1], c="red")
        for anchor in anchors[on_road_vecs==1]:
            self.axes.plot(anchor[:,0], anchor[:, 1], c="blue")