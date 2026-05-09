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
)


CAM_NAMES_NUSC = [
    'CAM_FRONT_LEFT',
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
]
CAM_NAMES_NUSC_converter = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]

class CamRender:
    def __init__(
        self, 
        plot_choices,
    ):
        self.plot_choices = plot_choices

    def reset_canvas(self):
        plt.close()
        plt.gca().set_axis_off()
        plt.axis('off')
        self.fig, self.axes = plt.subplots(2, 3, figsize=(160 /3 , 20))
        plt.tight_layout()

    def render(
        self,
        data, 
        result,
    ):
        if self.plot_choices["cam_gt"]:
            self.reset_canvas()
            self.render_image_data(data)
            self.draw_detection_gt(data)
            self.draw_map_gt(data, result)
            self.draw_motion_gt(data)
            self.draw_planning_gt(data)
            cam_gt = self.get_fig()
        else:
            cam_gt = None

        if self.plot_choices["cam_pred"]:
            self.reset_canvas()
            self.render_image_data(data)
            self.draw_detection_pred(data, result)
            self.draw_map_pred(data, result)
            self.draw_motion_pred(data, result)
            self.draw_planning_pred(data, result)
            # self.draw_speed_pred(data, result)
            cam_pred = self.get_fig()
        else:
            cam_pred = None

        return cam_gt, cam_pred
        

    def load_image(self, data_path, cam):
        """Update the axis of the plot with the provided image."""
        image = np.array(Image.open(data_path))
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 60)
        fontScale = 2
        color = (0, 0, 0)
        thickness = 4
        return cv2.putText(image, cam, org, font, fontScale, color, thickness, cv2.LINE_AA)

    def update_image(self, image, index, cam):
        """Render image data for each camera."""
        ax = self.get_axis(index)
        ax.imshow(image)
        plt.axis('off')
        ax.axis('off')
        ax.grid(False)

    def get_axis(self, index):
        """Retrieve the corresponding axis based on the index."""
        return self.axes[index//3, index % 3]

    def get_fig(self):
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        return plt_fig_to_cv2_image(self.fig)

    def render_image_data(self, data):
        """Load and annotate image based on the provided path."""
        for i, cam in enumerate(CAM_NAMES_NUSC):
            idx = CAM_NAMES_NUSC_converter.index(cam)
            if "img" in data:
                image = data["img"][idx]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                img_path = data['img_filename'][idx]
                image = self.load_image(img_path, cam)
            self.update_image(image, i, cam)
            self.img_size = (image.shape[1], image.shape[0])
    
    def draw_detection_gt(self, data):
        if not self.plot_choices['det']:
            return

        bboxes = data["gt_bboxes_3d"]
        for j, cam in enumerate(CAM_NAMES_NUSC):
            idx = CAM_NAMES_NUSC_converter.index(cam)
            cam_intrinsic = data['cam_intrinsic'][idx]
            lidar2cam = data['lidar2cam']
            extrinsic = lidar2cam[idx]
            trans = extrinsic[3, :3]
            rot = Quaternion(matrix=extrinsic[:3, :3]).inverse
            imsize = self.img_size

            for i in range(data['gt_labels_3d'].shape[0]):
                label = data['gt_labels_3d'][i]
                if label == -1: 
                    continue
                color = color_mapping[data['instance_inds'][i] % len(color_mapping)]
                
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
                        self.axes[j // 3, j % 3], 
                        view=cam_intrinsic, 
                        normalize=True, 
                        colors=(color, color, color),
                        linewidth=4,
                    )
            
            self.axes[j//3, j % 3].set_xlim(0, imsize[0])
            self.axes[j//3, j % 3].set_ylim(imsize[1], 0)

    def draw_detection_pred(self, data, result):
        if not (self.plot_choices['det'] and "boxes_3d" in result):
            return

        bboxes = result['boxes_3d'].numpy()
        for j, cam in enumerate(CAM_NAMES_NUSC):
            idx = CAM_NAMES_NUSC_converter.index(cam)
            cam_intrinsic = data['cam_intrinsic'][idx]
            lidar2cam = data['lidar2cam']
            extrinsic = lidar2cam[idx]
            trans = extrinsic[3, :3]
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
                        self.axes[j // 3, j % 3], 
                        view=cam_intrinsic, 
                        normalize=True, 
                        colors=(color, color, color),
                        linewidth=4,
                    )
            
            self.axes[j//3, j % 3].set_xlim(0, imsize[0])
            self.axes[j//3, j % 3].set_ylim(imsize[1], 0)

    def draw_motion_gt(self, data, points_per_step=10):
        if not self.plot_choices['motion']:
            return

        bboxes = data['gt_bboxes_3d']
        for j, cam in enumerate(CAM_NAMES_NUSC):
            idx = CAM_NAMES_NUSC_converter.index(cam)
            cam_intrinsic = data['cam_intrinsic'][idx]
            lidar2cam = data['lidar2cam']
            extrinsic = lidar2cam[idx]
            trans = extrinsic[3, :3]
            rot = Quaternion(matrix=extrinsic[:3, :3]).inverse
            imsize = self.img_size

            for i in range(data['gt_labels_3d'].shape[0]):
                label = data['gt_labels_3d'][i]
                if label == -1: 
                    continue
                color = color_mapping[data['instance_inds'][i] % len(color_mapping)]

                traj = data["gt_agent_fut_trajs"][i]
                masks = data['gt_agent_fut_masks'][i].astype(bool)
                if masks[0] == 0:
                    continue
                origin = bboxes[i, :2][None]
                traj = traj.cumsum(axis=0) + origin
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
                self._render_traj(traj_points, cam_intrinsic, j, color=color, s=15)

    def draw_motion_pred(self, data, result, points_per_step=10):
        if not (self.plot_choices['motion'] and "trajs_3d" in result):
            return

        bboxes = result['boxes_3d'].numpy()
        for j, cam in enumerate(CAM_NAMES_NUSC):
            idx = CAM_NAMES_NUSC_converter.index(cam)
            cam_intrinsic = data['cam_intrinsic'][idx]
            lidar2cam = data['lidar2cam']
            extrinsic = lidar2cam[idx]
            trans = extrinsic[3, :3]
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
                self._render_traj(traj_points, cam_intrinsic, j, color=color, s=15)

    def draw_map_gt(self, data, result, ground_height=-1.8):
        if not self.plot_choices['map']:
            return

        for j, cam in enumerate(CAM_NAMES_NUSC):
            idx = CAM_NAMES_NUSC_converter.index(cam)
            cam_intrinsic = data['cam_intrinsic'][idx]
            lidar2cam = data['lidar2cam']
            extrinsic = lidar2cam[idx]
            trans = extrinsic[3, :3]
            imsize = self.img_size

            vectors = data['map_infos']
            for label, vector_list in vectors.items():
                color = COLOR_VECTORS[label]
                for vector in vector_list:
                    pts = vector[:, :2]
                    line = LineString(pts)
                    distances = np.linspace(0, line.length, 20)
                    pts = np.array([list(line.interpolate(distance).coords) 
                        for distance in distances]).squeeze()

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
                    self.axes[j // 3, j % 3].plot(pts_points[0], pts_points[1], color=color, linewidth=3, linestyle='-')

    def draw_map_pred(self, data, result, ground_height=-1.8):
        if not (self.plot_choices['map'] and "vectors" in result):
            return
        
        for j, cam in enumerate(CAM_NAMES_NUSC):
            idx = CAM_NAMES_NUSC_converter.index(cam)
            cam_intrinsic = data['cam_intrinsic'][idx]
            lidar2cam = data['lidar2cam']
            extrinsic = lidar2cam[idx]
            trans = extrinsic[3, :3]
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
                self.axes[j // 3, j % 3].plot(pts_points[0], pts_points[1], color=color, linewidth=3, linestyle='-')

    def draw_planning_gt(self, data):
        if not self.plot_choices['planning']:
            return

        idx = 0 ## front camera
        cam_intrinsic = data['cam_intrinsic'][idx]
        lidar2cam = data['lidar2cam']
        extrinsic = lidar2cam[idx]
        trans = extrinsic[3, :3]
        rot = Quaternion(matrix=extrinsic[:3, :3]).inverse
        masks = data['gt_ego_fut_masks'].astype(bool)
        if masks[0] != 0:
            plan_traj = data['gt_ego_fut_trajs'][masks]
            plan_traj[abs(plan_traj) < 0.01] = 0.0
            plan_traj = plan_traj.cumsum(axis=0)
            plan_traj = np.concatenate((np.zeros((1, 2)), plan_traj), axis=0)
            traj_expand = np.ones((plan_traj.shape[0], 1)) * -1.8
            plan_traj = np.concatenate([plan_traj, traj_expand], axis=1)

            traj_points = plan_traj @ extrinsic[:3, :3] + trans
            self._render_traj(traj_points, cam_intrinsic, j=1)

    def draw_planning_anchor(self, data):
        if not self.plot_choices['planning']:
            return

        anchors = np.load("data/kmeans/kmeans_plan_1024_b2d_kmeans.npy")
        for anchor in anchors:
            for i, cam in enumerate(CAM_NAMES_NUSC):
                idx = CAM_NAMES_NUSC_converter.index(cam)
                cam_intrinsic = data['cam_intrinsic'][idx]
                lidar2cam = data['lidar2cam']
                extrinsic = lidar2cam[idx]
                trans = extrinsic[3, :3]
                rot = Quaternion(matrix=extrinsic[:3, :3]).inverse
            
                plan_traj = anchor
                plan_traj[abs(plan_traj) < 0.01] = 0.0
                plan_traj = np.concatenate((np.zeros((1, 2)), plan_traj), axis=0)
                traj_expand = np.ones((plan_traj.shape[0], 1)) * -1.8
                plan_traj = np.concatenate([plan_traj, traj_expand], axis=1)

                traj_points = plan_traj @ extrinsic[:3, :3] + trans
                self._render_traj(traj_points, cam_intrinsic, j=i)
        
    def draw_planning_pred(self, data, result):
        if not (self.plot_choices['planning'] and "planning" in result):
            return
        # for j, cam in enumerate(CAM_NAMES_NUSC[1]):
        #     idx = CAM_NAMES_NUSC_converter.index(cam)
        #     cam_intrinsic = data['cam_intrinsic'][idx]
        #     lidar2cam = data['lidar2cam']
        #     extrinsic = lidar2cam[idx]
        #     trans = extrinsic[3, :3]
        #     rot = Quaternion(matrix=extrinsic[:3, :3]).inverse
        #     imsize = (1600, 900)

        #     plan_trajs = result['planning'][0].cpu().numpy()
        #     plan_trajs = plan_trajs.reshape(3, -1, 6, 2)
        #     num_cmd = len(CMD_LIST)
        #     num_mode = plan_trajs.shape[1]
        #     plan_trajs = np.concatenate((np.zeros((num_cmd, num_mode, 1, 2)), plan_trajs), axis=2)
        #     plan_trajs = plan_trajs.cumsum(axis=-2)
        #     plan_score = result['planning_score'][0].cpu().numpy()
        #     plan_score = plan_score.reshape(3, -1)

        #     cmd = data['gt_ego_fut_cmd'].argmax()
        #     plan_trajs = plan_trajs[cmd]
        #     plan_score = plan_score[cmd]

        #     mode_idx = plan_score.argmax()
        #     plan_traj = plan_trajs[mode_idx]
        #     traj_expand = np.ones((plan_traj.shape[0], 1)) * -2
        #     # traj_expand[:] = bboxes[i, 2] - bboxes[i, 5] / 2
        #     plan_traj = np.concatenate([plan_traj, traj_expand], axis=1)

        #     traj_points = plan_traj @ extrinsic[:3, :3] + trans
        #     self._render_traj(traj_points, cam_intrinsic, j)

        idx = 0 ## front camera
        cam_intrinsic = data['cam_intrinsic'][idx]
        lidar2cam = data['lidar2cam']
        extrinsic = lidar2cam[idx]
        trans = extrinsic[3, :3]
        rot = Quaternion(matrix=extrinsic[:3, :3]).inverse
        # plan_trajs = result['planning'][0].cpu().numpy()
        # plan_trajs = plan_trajs.reshape(3, -1, 6, 2)
        # num_cmd = len(CMD_LIST)
        # num_mode = plan_trajs.shape[1]
        # plan_trajs = np.concatenate((np.zeros((num_cmd, num_mode, 1, 2)), plan_trajs), axis=2)
        # plan_trajs = plan_trajs.cumsum(axis=-2)
        # plan_score = result['planning_score'][0].cpu().numpy()
        # plan_score = plan_score.reshape(3, -1)

        # cmd = data['gt_ego_fut_cmd'].argmax()
        # plan_trajs = plan_trajs[cmd]
        # plan_score = plan_score[cmd]

        # mode_idx = plan_score.argmax()
        # plan_traj = plan_trajs[mode_idx]
        plan_traj = result["final_planning"]
        plan_traj = np.concatenate((np.zeros((1, 2)), plan_traj), axis=0)
        traj_expand = np.ones((plan_traj.shape[0], 1)) * -1.8
        plan_traj = np.concatenate([plan_traj, traj_expand], axis=1)

        traj_points = plan_traj @ extrinsic[:3, :3] + trans
        self._render_traj(traj_points, cam_intrinsic, j=1)

    def _render_traj(self, traj_points, cam_intrinsic, j, color=(1, 0.5, 0), s=150, points_per_step=10):
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
        self.axes[j // 3, j % 3].scatter(traj_points[0], traj_points[1], color=color, s=s)

    def draw_speed_pred(self, data, result):
        if not (self.plot_choices['speed'] and "target_speed_05s_score" in result):
            return
        speed_intervals = data["cfg"]["plan_config"]["target_speed_05s"]["speed_intervals"]
        score = result["target_speed_05s_score"].numpy()
        ax = self.fig.add_axes([0.015, 0.80, 0.3, 0.2])
        ax.bar(speed_intervals, score, width=0.15, color='skyblue', edgecolor='k')
        ax.set_ylim(bottom=0.008)
        # ax.set_xlabel('Speed (m/s)')
        ax.set_ylabel('Score')