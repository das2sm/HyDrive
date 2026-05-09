import os
import numpy as np
import cv2
from PIL import Image

from shapely.geometry import LineString
import matplotlib
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix

from tools.visualization.bev_render import (
    color_mapping, 
    COLOR_VECTORS,
    SCORE_THRESH, 
    MAP_SCORE_THRESH,
    plt_fig_to_cv2_image,
    CMD_LIST,
)

class BEVCamRender:
    def __init__(
        self, 
        plot_choices,
    ):
        self.plot_choices = plot_choices

    def reset_canvas(self):
        plt.close()
        plt.gca().set_axis_off()
        plt.axis('off')
        self.fig, self.axes = plt.subplots(1, 1, figsize=(20, 20))
        plt.tight_layout()

    def render(
        self,
        data, 
        result,
    ):
        if self.plot_choices["bevcam_pred"]:
            self.reset_canvas()
            self.render_image_data(data)
            self.draw_detection_pred(data, result)
            self.draw_map_pred(data, result)
            self.draw_motion_pred(data, result)
            self.draw_planning_pred_v1(data, result)
            self.render_control(result)
            self.render_target_point(data, result)
            self.render_route(data, result)
            # self.render_sdc_car(data)
            # self._render_command(data)
            cam_pred = self.get_fig()
        else:
            cam_pred = None

        return cam_pred
        
    def update_image(self, image):
        """Render image data for each camera."""
        ax = self.axes
        ax.imshow(image)
        plt.axis('off')
        ax.axis('off')
        ax.grid(False)

    def get_fig(self):
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        return plt_fig_to_cv2_image(self.fig)

    def render_image_data(self, data):
        """Load and annotate image based on the provided path."""
        image = data["bev_img"]
        self.update_image(image)
        self.img_size = (image.shape[1], image.shape[0])

    def draw_detection_pred(self, data, result):
        if not (self.plot_choices['det'] and "boxes_3d" in result):
            return

        bboxes = result['boxes_3d'].numpy()
        cam_intrinsic = data['bev_intrinsic']
        extrinsic = data['bev_extrinsic']
        trans = extrinsic[:3, 3]
        rot = Quaternion(matrix=extrinsic[:3, :3]).inverse
        imsize = self.img_size

        for i in range(result['labels_3d'].shape[0]):
            score = result['scores_3d'][i]
            if score < SCORE_THRESH: 
                continue
            color = color_mapping[result['instance_ids'][i] % len(color_mapping)]
            
            center = bboxes[i, 0 : 3]
            box_dims = bboxes[i, 3 : 6]
            nusc_dims = box_dims[..., [1, 0, 2]]
            quat = Quaternion(axis=[0, 0, 1], radians=bboxes[i, 6])
            box = NuScenesBox(
                center,
                nusc_dims,
                quat
            )
            box.rotate(rot)
            box.translate(trans)
            if box_in_image(box, cam_intrinsic, imsize):
                box.render(
                    self.axes, 
                    view=cam_intrinsic, 
                    normalize=True, 
                    colors=(color, color, color),
                    linewidth=4,
                )
        
        self.axes.set_xlim(0, imsize[0])
        self.axes.set_ylim(imsize[1], 0)

    def draw_motion_pred(self, data, result, points_per_step=10):
        if not (self.plot_choices['motion'] and "trajs_3d" in result):
            return

        bboxes = result['boxes_3d'].numpy()
        cam_intrinsic = data['bev_intrinsic']
        extrinsic = data['bev_extrinsic']
        trans = extrinsic[:3, 3]
        rot = Quaternion(matrix=extrinsic[:3, :3]).inverse
        imsize = self.img_size

        for i in range(result['labels_3d'].shape[0]):
            score = result['scores_3d'][i]
            if score < SCORE_THRESH: 
                continue
            color = color_mapping[result['instance_ids'][i] % len(color_mapping)]
            
            traj_score = result['trajs_score'][i].numpy()
            traj = result['trajs_3d'][i].numpy()
            
            mode_idx = traj_score.argmax()
            traj = traj[mode_idx]
            origin = bboxes[i, :2][None]
            traj = np.concatenate([origin, traj], axis=0)
            traj_expand = np.ones((traj.shape[0], 1)) 
            traj_expand[:] = bboxes[i, 2] - bboxes[i, 5] / 2
            traj = np.concatenate([traj, traj_expand], axis=1)

            center = bboxes[i, 0 : 3]
            box_dims = bboxes[i, 3 : 6]
            nusc_dims = box_dims[..., [1, 0, 2]]
            quat = Quaternion(axis=[0, 0, 1], radians=bboxes[i, 6])
            box = NuScenesBox(
                center,
                nusc_dims,
                quat
            )
            box.rotate(rot)
            box.translate(trans)
            if not box_in_image(box, cam_intrinsic, imsize):
                continue
            traj_points = traj @ extrinsic[:3, :3] + trans
            self._render_traj_bev(traj_points, cam_intrinsic, colormap="autumn")

    def draw_map_pred(self, data, result, ground_height=-1.8):
        if not (self.plot_choices['map'] and "vectors" in result):
            return
        
        cam_intrinsic = data['bev_intrinsic']
        extrinsic = data['bev_extrinsic']
        trans = extrinsic[:3, 3]
        rot = Quaternion(matrix=extrinsic[:3, :3]).inverse
        imsize = self.img_size

        for i in range(result['scores'].shape[0]):
            score = result['scores'][i]
            if  score < MAP_SCORE_THRESH:
                continue
            color = COLOR_VECTORS[result['labels'][i]]
            pts = result['vectors'][i]
            pts_expand = np.ones((pts.shape[0], 1)) * ground_height
            pts = np.concatenate([pts, pts_expand], axis=1)
            pts_cam = pts @ extrinsic[:3, :3] + trans

            pts_points = view_points(
                pts_cam.T, cam_intrinsic, normalize=True)[:2, :]

            visible = np.logical_and(pts_points[0, :] > 0, pts_points[0, :] < imsize[0]-1)
            visible = np.logical_and(visible, pts_points[1, :] < imsize[1]-1)
            visible = np.logical_and(visible, pts_points[1, :] > 0)
            visible = np.logical_and(visible, pts_cam[:, 2] > 0.)
            pts_points = pts_points[:2, visible]
            self.axes.plot(pts_points[0], pts_points[1], color=color, linewidth=3, linestyle='-')
        
    def draw_planning_pred_v1(self, data, result):
        # if not (self.plot_choices['planning'] and "planning" in result):
        #     return

        cam_intrinsic = data['bev_intrinsic']
        extrinsic = data['bev_extrinsic']
        trans = extrinsic[:3, 3]
        rot = Quaternion(matrix=extrinsic[:3, :3]).inverse
        imsize = self.img_size

        # plan_traj = result["lat_reg_final"]
        plan_traj = result["traj_final"]
        plan_traj = np.concatenate((np.zeros((1, 2)), plan_traj), axis=0)
        traj_expand = np.ones((plan_traj.shape[0], 1)) * -1.8
        plan_traj = np.concatenate([plan_traj, traj_expand], axis=1)
        traj_points = plan_traj @ extrinsic[:3, :3] + trans
        self._render_traj_bev(traj_points, cam_intrinsic)

    def draw_planning_pred_v2(self, data, result, top_k=6): ## old, with command
        if not (self.plot_choices['planning'] and "planning" in result):
            return

        cam_intrinsic = data['bev_intrinsic']
        extrinsic = data['bev_extrinsic']
        trans = extrinsic[:3, 3]
        rot = Quaternion(matrix=extrinsic[:3, :3]).inverse
        imsize = self.img_size

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

        if num_mode < 2000:
            colors = ['#e6194b', '#3cb44b', '#f032e6', '#4363d8', '#f58231', '#911eb4'][::-1]
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

        for i, (traj, prob, color) in enumerate(zip(top_trajs, top_prob, colors)):
            traj_expand = np.ones((traj.shape[0], 1)) * -1.8
            traj = np.concatenate([traj, traj_expand], axis=1)
            traj = traj @ extrinsic[:3, :3] + trans
            traj = view_points(
                traj.T, cam_intrinsic, normalize=True)[:2, :]
            traj = np.transpose(traj, (1, 0))

            # 最后一个点坐标
            x_last, y_last = traj[-1]

            # 概率最高的轨迹特殊处理
            if i == -1:          # idx[0] 对应概率最大
                self.axes.plot(traj[:, 0], traj[:, 1], color=color, linewidth=16)
                self.axes.scatter(x_last, y_last, color=color, s=800, zorder=5)
                if top_k < 10:
                    self.axes.text(x_last + 30, y_last,
                                f'{prob:.2f}',
                                color=color,
                                fontsize=60,
                                ha='center',
                                va='bottom',
                                fontweight='bold',   # 加粗
                                zorder=6)
            else:
                self.axes.plot(traj[:, 0], traj[:, 1], color=color, linewidth=12)
                self.axes.scatter(x_last, y_last, color=color, s=600, zorder=5)
                if top_k < 10:
                    self.axes.text(x_last + 30, y_last,
                                f'{prob:.2f}',
                                color=color,
                                fontsize=60,
                                ha='center',
                                va='bottom',
                                zorder=6)

    def draw_planning_pred(self, data, result, top_k=6):
        if not (self.plot_choices['planning'] and data["planning_key"]+"_reg" in result):
            return

        cam_intrinsic = data['bev_intrinsic']
        extrinsic = data['bev_extrinsic']
        trans = extrinsic[:3, 3]
        rot = Quaternion(matrix=extrinsic[:3, :3]).inverse
        imsize = self.img_size

        # 处理多模态轨迹
        reg_key = data["planning_key"]+"_reg"
        cls_key = data["planning_key"]+"_cls"
        plan_trajs = result[reg_key].cpu().numpy()
        num_mode = plan_trajs.shape[0]
        plan_trajs = np.concatenate(
            (np.zeros((num_mode, 1, 2)), plan_trajs), axis=1)  # (num_mode, T, 2)

        plan_score = result[cls_key].cpu().numpy()  # (num_cmd, num_mode)
        
        if num_mode < 2000:
            colors = ['#e6194b', '#3cb44b', '#f032e6', '#4363d8', '#f58231', '#911eb4'][::-1]
            idx = np.argsort(plan_score)[::-1][:top_k][::-1]
            top_trajs = plan_trajs[idx]         # (top_k, T, 2)
            top_score = plan_score[idx]         # (top_k,)
            top_prob = top_score
        else:
            top_k = 1
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

        for i, (traj, prob, color) in enumerate(zip(top_trajs, top_prob, colors)):
            traj_expand = np.ones((traj.shape[0], 1)) * -1.8
            traj = np.concatenate([traj, traj_expand], axis=1)
            traj = traj @ extrinsic[:3, :3] + trans
            traj = view_points(
                traj.T, cam_intrinsic, normalize=True)[:2, :]
            traj = np.transpose(traj, (1, 0))

            # 最后一个点坐标
            x_last, y_last = traj[-1]

            # 概率最高的轨迹特殊处理
            if i == -1:          # idx[0] 对应概率最大
                self.axes.plot(traj[:, 0], traj[:, 1], color=color, linewidth=16)
                self.axes.scatter(x_last, y_last, color=color, s=800, zorder=5)
                if top_k < 10:
                    self.axes.text(x_last + 30, y_last,
                                f'{prob:.2f}',
                                color=color,
                                fontsize=60,
                                ha='center',
                                va='bottom',
                                fontweight='bold',   # 加粗
                                zorder=6)
            else:
                self.axes.plot(traj[:, 0], traj[:, 1], color=color, linewidth=12)
                self.axes.scatter(x_last, y_last, color=color, s=600, zorder=5)
                if top_k < 10:
                    self.axes.text(x_last + 30, y_last,
                                f'{prob:.2f}',
                                color=color,
                                fontsize=60,
                                ha='center',
                                va='bottom',
                                zorder=6)

    def _render_traj(self, traj_points, cam_intrinsic, color=(1, 0.5, 0), s=300, points_per_step=20):
        total_steps = (len(traj_points)-1) * points_per_step + 1
        total_xy = np.zeros((total_steps, 3))
        for k in range(total_steps-1):
            unit_vec = traj_points[k//points_per_step +
                                    1] - traj_points[k//points_per_step]
            total_xy[k] = (k/points_per_step - k//points_per_step) * \
                unit_vec + traj_points[k//points_per_step]
        in_range_mask = total_xy[:, 2] > 0.1
        traj_points = view_points(
            total_xy.T, cam_intrinsic, normalize=True)[:2, :]
        traj_points = traj_points[:2, in_range_mask]
        self.axes.scatter(traj_points[0], traj_points[1], color=color, s=s)

    def _render_traj_bev(
        self, 
        future_traj,
        cam_intrinsic, 
        traj_score=1, 
        colormap='winter', 
        points_per_step=20, 
        dot_size=250,
    ):
        total_steps = (len(future_traj) - 1) * points_per_step + 1
        dot_colors = matplotlib.colormaps[colormap](
            np.linspace(0, 1, total_steps))[:, :3]
        dot_colors = dot_colors * traj_score + \
            (1 - traj_score) * np.ones_like(dot_colors)
        total_xy = np.zeros((total_steps, 3))
        for i in range(total_steps - 1):
            unit_vec = future_traj[i // points_per_step +
                                   1] - future_traj[i // points_per_step]
            total_xy[i] = (i / points_per_step - i // points_per_step) * \
                unit_vec + future_traj[i // points_per_step]
        total_xy[-1] = future_traj[-1]
        in_range_mask = total_xy[:, 2] > 0.1
        traj_points = view_points(
            total_xy.T, cam_intrinsic, normalize=True)[:2, :]
        self.axes.scatter(
            traj_points[0], traj_points[1], c=dot_colors, s=dot_size)

    def render_control(self, result):
        steer = np.round(result["control"].steer, 2)
        throttle = np.round(result["control"].throttle, 2)
        brake = np.round(result["control"].brake, 2)
        speed = np.round(result["pid_metadata"]["speed"], 2)
        desired_speed = np.round(result["pid_metadata"]["desired_speed"], 2)

        self.axes.text(10, 20, f"throttle: {throttle}", fontsize=60, color='white')
        self.axes.text(10, 40, f"steer: {steer}", fontsize=60, color='white')
        self.axes.text(10, 60, f"brake: {brake}", fontsize=60, color='white')
        self.axes.text(10, 80, f"speed: {speed}", fontsize=60, color='white')
        self.axes.text(10, 100, f"desired_speed: {desired_speed}", fontsize=60, color='white')

    def render_target_point(self, data, result):
        if not self.plot_choices['target_point']:
            return
        cam_intrinsic = data['bev_intrinsic']
        extrinsic = data['bev_extrinsic']
        trans = extrinsic[:3, 3]
        rot = Quaternion(matrix=extrinsic[:3, :3]).inverse

        tp = result["pid_metadata"]["local_command_xy"]        
        tp = np.concatenate([tp, np.array(0)[None]], axis=0).reshape(1, 3)
        tp = tp @ extrinsic[:3, :3] + trans
        tp = view_points(tp.T, cam_intrinsic, normalize=True)[:2]
        self.axes.scatter(tp[0], tp[1], c='k', s=1500)

    def render_route(self, data, result):
        if not (self.plot_choices['route'] and "route" in data):
            return
        cam_intrinsic = data['bev_intrinsic']
        extrinsic = data['bev_extrinsic']
        trans = extrinsic[:3, 3]
        rot = Quaternion(matrix=extrinsic[:3, :3]).inverse
        route = data["route"]

        for tp in route:
            tp = np.concatenate([tp, np.array(0)[None]], axis=0).reshape(1, 3)
            tp = tp @ extrinsic[:3, :3] + trans
            tp = view_points(tp.T, cam_intrinsic, normalize=True)[:2]
            self.axes.scatter(tp[0], tp[1], c='r', s=800)

    def render_sdc_car(self, data):
        cam_intrinsic = data['bev_intrinsic']
        extrinsic = data['bev_extrinsic']
        trans = extrinsic[:3, 3]
        rot = Quaternion(matrix=extrinsic[:3, :3]).inverse
        imsize = self.img_size

        color = 'r'
        center = np.array([0, 0, 0])
        box_dims = np.array([4, 2, 2])
        nusc_dims = box_dims[..., [1, 0, 2]]
        quat = Quaternion(axis=[0, 0, 1], radians=np.pi/2)
        box = NuScenesBox(
            center,
            nusc_dims,
            quat
        )
        box.rotate(rot)
        box.translate(trans)
        if box_in_image(box, cam_intrinsic, imsize):
            box.render(
                self.axes, 
                view=cam_intrinsic, 
                normalize=True, 
                colors=(color, color, color),
                linewidth=4,
            )
        
        self.axes.set_xlim(0, imsize[0])
        self.axes.set_ylim(imsize[1], 0)

    def _render_command(self, data):
        cmd = data['gt_ego_fut_cmd'].argmax()
        self.axes.text(10, 120, CMD_LIST[cmd] + " " + str(data["index"]), fontsize=60, color='white')


    # def draw_traj_bev(self, traj, raw_img,canvas_size=(512,512),thickness=3,is_ego=False,hue_start=120,hue_end=80):
    #     if is_ego:
    #         line = np.concatenate([np.zeros((1,2)),traj],axis=0)
    #     else:
    #         line = traj
    #     img = raw_img.copy()        
    #     pts_4d = np.stack([line[:,0],line[:,1],np.zeros((line.shape[0])),np.ones((line.shape[0]))])
    #     pts_2d = (self.coor2topdown @ pts_4d).T
    #     pts_2d[:, 0] /= pts_2d[:, 2]
    #     pts_2d[:, 1] /= pts_2d[:, 2]
    #     mask = (pts_2d[:, 0]>0) & (pts_2d[:, 0]<canvas_size[1]) & (pts_2d[:, 1]>0) & (pts_2d[:, 1]<canvas_size[0])
    #     if not mask.any():
    #         return img
    #     pts_2d = pts_2d[mask,0:2]
    #     try:
    #         tck, u = splprep([pts_2d[:, 0], pts_2d[:, 1]], s=0)
    #     except:
    #         return img
    #     unew = np.linspace(0, 1, 100)
    #     smoothed_pts = np.stack(splev(unew, tck)).astype(int).T

    #     num_points = len(smoothed_pts)
    #     for i in range(num_points-1):
    #         hue = hue_start + (hue_end - hue_start) * (i / num_points)
    #         hsv_color = np.array([hue, 255, 255], dtype=np.uint8)
    #         rgb_color = cv2.cvtColor(hsv_color[np.newaxis, np.newaxis, :], cv2.COLOR_HSV2RGB).reshape(-1)
    #         rgb_color_tuple = (float(rgb_color[0]),float(rgb_color[1]),float(rgb_color[2]))
    #         if smoothed_pts[i,0]>0 and smoothed_pts[i,0]<canvas_size[1] and smoothed_pts[i,1]>0 and smoothed_pts[i,1]<canvas_size[0]:
    #             cv2.line(img,(smoothed_pts[i,0],smoothed_pts[i,1]),(smoothed_pts[i+1,0],smoothed_pts[i+1,1]),color=rgb_color_tuple, thickness=thickness)   
    #         elif i==0:
    #             break
    #     return img