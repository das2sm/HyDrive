import csv
import os
import json
import datetime
import pathlib
import time
import cv2
from matplotlib import pyplot as plt
import carla
from collections import deque
import math
from collections import OrderedDict
from scipy.optimize import fsolve
import torch
import carla
import numpy as np
from PIL import Image
from torchvision import transforms as T
from Bench2DriveZoo.team_code.pid_controller import PIDController
from Bench2DriveZoo.team_code.planner import RoutePlanner
from leaderboard.autoagents import autonomous_agent
from mmcv import Config
from mmcv.models import build_model
from mmcv.utils import (get_dist_info, init_dist, load_checkpoint,
                        wrap_fp16_model)
from mmcv.datasets.pipelines import Compose
from mmcv.parallel.collate import collate as  mm_collate_to_batch_form
from mmcv.core.bbox import get_box_type
from pyquaternion import Quaternion

SAVE_PATH = os.environ.get('SAVE_PATH', None)
IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)


def get_entry_point():
    return 'VadAgent'

class Guardian:
    """
    Safety Shield / Runtime Monitor for CARLA + Bench2Drive.
    Checks if the ego vehicle's future swept volume overlaps with any obstacles.
    """

    def __init__(self, world, log_dir='results/guardian_logs', debug=False):
        self.world = world
        self.debug = debug

        # Grid settings
        self.grid_size = 240                    # Larger grid for better side coverage
        self.resolution = 0.4                   # meters per pixel (finer than before)
        self.origin = self.grid_size // 2

        # === Configurable Safety Margins ===
        self.vehicle_length = 4.8               # meters
        self.vehicle_width = 2.1                # meters
        self.vehicle_height = 2.0               # meters
        self.longitudinal_margin = 1.2          # extra meters in front/back
        self.lateral_margin = 0.8               # extra meters on sides (important for future swerving)
        self.sweep_sample_spacing = 0.9         # max meters between swept-volume OBB samples
        self.max_debug_boxes = 80               # throttle CARLA debug draw load
        self.debug_draw_interval = 2
        self._draw_counter = 0
        self._static_level_bbs = None
        self.min_eval_traj_extent = 2.0
        self.path_block_area_threshold = 0.50
        self.path_block_near_distance = 2.75
        self.hold_frames_after_block = 6
        self.block_memory_frames_after_seen = 80
        self.min_new_block_speed = 0.50
        self.fast_block_speed = 2.00
        self.path_block_confirm_frames = 2
        self.emergency_decel_threshold = 3.5
        self.close_block_base_distance = 4.0
        self.close_block_time_headway = 0.8
        self._hold_frames = 0
        self._block_memory_frames = 0
        self._path_block_confirmations = 0
        self._last_valid_traj = None

        settings = self.world.get_settings() if self.world is not None else None
        self.fixed_delta_seconds = (
            settings.fixed_delta_seconds
            if settings is not None and settings.fixed_delta_seconds
            else 0.05
        )

        # VAD/Bench2Drive trajectories are in the model LiDAR frame:
        # [left, forward]. CARLA actor-local coordinates are [forward_x, right_y, up_z].
        self._ego_bbox_location = carla.Location(0.0, 0.0, self.vehicle_height / 2.0)
        self._ego_bbox_rotation = carla.Rotation()

        self.log_data = []

        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path = os.path.join(log_dir, f'guardian_{timestamp}.csv')

        # Initialize CSV
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'step', 'timestamp', 'ego_x', 'ego_y', 'ego_yaw', 'ego_speed',
                'overlap_ratio', 'min_occupied_distance', 'ttc_occupied',
                'req_decel', 'gc_score', 'gc_overlap_term', 'gc_potential_term',
                'gc_decel_term', 'gc_ttc_term', 'cri_score', 'risk_direction',
                'path_blocked', 'path_blockage_area', 'blocker_distance',
                'blocker_type', 'eval_traj_held',
                'occupancy_source',
                'intervention_flag', 'brake_command', 'throttle_command',
                'actor_count'
            ])

    def _refresh_ego_geometry(self, ego_actor):
        if ego_actor is None:
            return

        bbox = ego_actor.bounding_box
        self.vehicle_length = max(0.1, bbox.extent.x * 2.0)
        self.vehicle_width = max(0.1, bbox.extent.y * 2.0)
        self.vehicle_height = max(0.1, bbox.extent.z * 2.0)
        self._ego_bbox_location = bbox.location
        self._ego_bbox_rotation = bbox.rotation

    def _vad_traj_to_carla_local(self, traj):
        """Convert VAD [left, forward] waypoints to CARLA local [forward, right]."""
        traj = np.asarray(traj, dtype=np.float64)
        if traj.ndim != 2 or traj.shape[1] < 2 or len(traj) == 0:
            return np.zeros((0, 2), dtype=np.float64)
        return np.stack([traj[:, 1], -traj[:, 0]], axis=1)

    def _trajectory_extent(self, traj):
        carla_local = self._vad_traj_to_carla_local(traj)
        if len(carla_local) == 0:
            return 0.0
        return float(np.max(np.linalg.norm(carla_local, axis=1)))

    def _select_eval_trajectory(self, traj, speed):
        traj = np.asarray(traj, dtype=np.float64)
        extent = self._trajectory_extent(traj)
        if extent >= self.min_eval_traj_extent:
            self._last_valid_traj = traj.copy()
            return traj, False

        speed_value = float(np.asarray(speed).reshape(-1)[0])
        if (
            self._last_valid_traj is not None
            and speed_value < 1.0
            and (self._hold_frames > 0 or self._block_memory_frames > 0)
        ):
            return self._last_valid_traj.copy(), True

        return traj, False

    def _sample_carla_local_trajectory(self, traj):
        """Return dense CARLA-local poses: forward, right, yaw_rad."""
        carla_local = self._vad_traj_to_carla_local(traj)
        if len(carla_local) == 0:
            return []

        points = np.vstack([np.zeros((1, 2), dtype=np.float64), carla_local])
        sampled = [points[0]]
        for p0, p1 in zip(points[:-1], points[1:]):
            seg = p1 - p0
            dist = float(np.linalg.norm(seg))
            steps = max(1, int(np.ceil(dist / self.sweep_sample_spacing)))
            for j in range(1, steps + 1):
                sampled.append(p0 + seg * (j / steps))

        sampled = np.asarray(sampled, dtype=np.float64)
        poses = []
        last_yaw = 0.0
        for i, point in enumerate(sampled):
            if i < len(sampled) - 1:
                delta = sampled[i + 1] - point
            elif i > 0:
                delta = point - sampled[i - 1]
            else:
                delta = np.array([1.0, 0.0])

            if np.linalg.norm(delta) > 1e-4:
                last_yaw = math.atan2(delta[1], delta[0])

            poses.append((float(point[0]), float(point[1]), float(last_yaw)))

        return poses

    def _local_to_grid(self, points_local):
        points_local = np.asarray(points_local, dtype=np.float64)
        px = points_local[:, 0] / self.resolution + self.origin
        py = points_local[:, 1] / self.resolution + self.origin
        return np.stack([px, py], axis=1)

    def _grid_to_local(self, occupied_yx):
        occupied_yx = np.asarray(occupied_yx, dtype=np.float64)
        forward = (occupied_yx[:, 1] - self.origin) * self.resolution
        right = (occupied_yx[:, 0] - self.origin) * self.resolution
        return np.stack([forward, right], axis=1)

    def _world_to_ego_local_points(self, points_world, ego_transform):
        points_world = np.asarray(points_world, dtype=np.float64)
        if points_world.ndim == 2 and points_world.shape[1] == 2:
            points_world = np.column_stack([points_world, np.zeros(len(points_world))])

        try:
            inv = np.asarray(ego_transform.get_inverse_matrix(), dtype=np.float64)
            pts_h = np.column_stack([points_world[:, :3], np.ones(len(points_world))])
            return (inv @ pts_h.T).T[:, :3]
        except Exception:
            yaw = np.deg2rad(ego_transform.rotation.yaw)
            delta = points_world[:, :2] - np.array([ego_transform.location.x, ego_transform.location.y])
            c, s = np.cos(yaw), np.sin(yaw)
            forward = delta[:, 0] * c + delta[:, 1] * s
            right = -delta[:, 0] * s + delta[:, 1] * c
            z = points_world[:, 2] - ego_transform.location.z
            return np.column_stack([forward, right, z])

    def _local_to_world_points(self, points_local, ego_transform):
        points_local = np.asarray(points_local, dtype=np.float64)
        if points_local.ndim == 2 and points_local.shape[1] == 2:
            points_local = np.column_stack([points_local, np.zeros(len(points_local))])

        try:
            mat = np.asarray(ego_transform.get_matrix(), dtype=np.float64)
            pts_h = np.column_stack([points_local[:, :3], np.ones(len(points_local))])
            return (mat @ pts_h.T).T[:, :3]
        except Exception:
            yaw = np.deg2rad(ego_transform.rotation.yaw)
            c, s = np.cos(yaw), np.sin(yaw)
            x = ego_transform.location.x + points_local[:, 0] * c - points_local[:, 1] * s
            y = ego_transform.location.y + points_local[:, 0] * s + points_local[:, 1] * c
            z = ego_transform.location.z + points_local[:, 2]
            return np.column_stack([x, y, z])

    def _compose_rotation(self, base, offset):
        return carla.Rotation(
            pitch=base.pitch + offset.pitch,
            yaw=base.yaw + offset.yaw,
            roll=base.roll + offset.roll
        )

    def _transform_location(self, transform, location):
        loc = carla.Location(x=location.x, y=location.y, z=location.z)
        transformed = transform.transform(loc)
        return transformed if transformed is not None else loc

    def _future_transform_from_pose(self, ego_transform, forward, right, yaw_rad):
        world_xyz = self._local_to_world_points(
            np.array([[forward, right, 0.0]], dtype=np.float64),
            ego_transform
        )[0]
        return carla.Transform(
            carla.Location(x=float(world_xyz[0]), y=float(world_xyz[1]), z=float(world_xyz[2])),
            carla.Rotation(
                pitch=ego_transform.rotation.pitch,
                yaw=ego_transform.rotation.yaw + math.degrees(yaw_rad),
                roll=ego_transform.rotation.roll
            )
        )

    def build_ego_swept_mask(self, traj, ego_yaw=None):
        """Rasterize the ego swept volume as waypoint-oriented OBBs."""
        mask = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        half_len = (self.vehicle_length / 2 + self.longitudinal_margin) / self.resolution
        half_wid = (self.vehicle_width / 2 + self.lateral_margin) / self.resolution

        for forward, right, yaw in self._sample_carla_local_trajectory(traj):
            center_local = np.array([[
                forward + self._ego_bbox_location.x,
                right + self._ego_bbox_location.y
            ]], dtype=np.float64)
            gx, gy = self._local_to_grid(center_local)[0]

            if not (8 <= gx < self.grid_size-8 and 8 <= gy < self.grid_size-8):
                continue

            yaw += np.deg2rad(self._ego_bbox_rotation.yaw)

            # Oriented rectangle
            corners_local = np.array([
                [-half_len, -half_wid],
                [ half_len, -half_wid],
                [ half_len,  half_wid],
                [-half_len,  half_wid]
            ])

            rot = np.array([[np.cos(yaw), -np.sin(yaw)],
                            [np.sin(yaw),  np.cos(yaw)]])
            corners = (rot @ corners_local.T).T + np.array([gx, gy])

            pts = corners.astype(np.int32)
            cv2.fillConvexPoly(mask, pts, 1.0)

        # Safety dilation
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(np.float32)
        return mask

    def _bbox_world_vertices(self, bbox, actor_transform=None):
        try:
            if actor_transform is not None:
                return bbox.get_world_vertices(actor_transform)
            return bbox.get_world_vertices(carla.Transform())
        except Exception:
            center = np.array([bbox.location.x, bbox.location.y, bbox.location.z], dtype=np.float64)
            extent = bbox.extent
            corners = np.array([
                [ extent.x,  extent.y,  extent.z],
                [ extent.x, -extent.y,  extent.z],
                [-extent.x, -extent.y,  extent.z],
                [-extent.x,  extent.y,  extent.z],
                [ extent.x,  extent.y, -extent.z],
                [ extent.x, -extent.y, -extent.z],
                [-extent.x, -extent.y, -extent.z],
                [-extent.x,  extent.y, -extent.z],
            ], dtype=np.float64)
            yaw = np.deg2rad(bbox.rotation.yaw)
            c, s = np.cos(yaw), np.sin(yaw)
            rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            return [carla.Location(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                    for p in (rot @ corners.T).T + center]

    def _rasterize_world_vertices(self, occ_grid, vertices_world, ego_transform):
        if not vertices_world:
            return False

        points_world = np.array([[v.x, v.y, v.z] for v in vertices_world], dtype=np.float64)
        local = self._world_to_ego_local_points(points_world, ego_transform)

        # Ignore boxes that are fully outside the relevant vertical slab.
        if local[:, 2].max() < -2.5 or local[:, 2].min() > 5.0:
            return False

        grid = self._local_to_grid(local[:, :2])
        if (
            grid[:, 0].max() < -10 or grid[:, 0].min() > self.grid_size + 10
            or grid[:, 1].max() < -10 or grid[:, 1].min() > self.grid_size + 10
        ):
            return False

        hull = cv2.convexHull(np.clip(grid, -5, self.grid_size + 5).astype(np.float32))
        cv2.fillConvexPoly(occ_grid, hull.astype(np.int32), 1.0)
        return True

    def _get_static_level_bbs(self):
        if self._static_level_bbs is not None:
            return self._static_level_bbs

        labels = []
        for name in [
            'Static', 'Walls', 'Fences', 'GuardRail', 'TrafficLight',
            'TrafficSigns', 'Poles', 'Buildings'
        ]:
            if hasattr(carla.CityObjectLabel, name):
                labels.append(getattr(carla.CityObjectLabel, name))

        static_bbs = []
        for label in labels:
            try:
                static_bbs.extend(self.world.get_level_bbs(label))
            except Exception:
                continue

        self._static_level_bbs = static_bbs
        return self._static_level_bbs

    def build_carla_occupancy(self, ego_transform, max_dist=75.0):
        """Build occupancy grid from CARLA actors."""
        occ_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        ego_pos = np.array([ego_transform.location.x, ego_transform.location.y], dtype=np.float64)

        actor_count = 0
        static_count = 0

        for actor in self.world.get_actors():
            # Skip ego
            if actor.attributes.get('role_name') == 'hero':
                continue

            type_id = actor.type_id
            if not (type_id.startswith(('vehicle.', 'walker.')) or 'static.prop' in type_id):
                continue

            try:
                t = actor.get_transform()
                dist = np.linalg.norm([t.location.x - ego_pos[0], t.location.y - ego_pos[1]])
                if dist > max_dist:
                    continue

                if self._rasterize_world_vertices(
                    occ_grid,
                    self._bbox_world_vertices(actor.bounding_box, t),
                    ego_transform
                ):
                    actor_count += 1

            except Exception:
                continue

        for bbox in self._get_static_level_bbs():
            try:
                loc = bbox.location
                dx = loc.x - ego_pos[0]
                dy = loc.y - ego_pos[1]
                # Use the footprint radius so large static boxes are kept when nearby.
                radius = max(float(bbox.extent.x), float(bbox.extent.y))
                if np.linalg.norm([dx, dy]) - radius > max_dist:
                    continue
                if self._rasterize_world_vertices(
                    occ_grid,
                    self._bbox_world_vertices(bbox),
                    ego_transform
                ):
                    static_count += 1
            except Exception:
                continue

        # Slight dilation on obstacles
        occ_grid = cv2.dilate(occ_grid.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(np.float32)

        if self.debug:
            print(f"Guardian -> {actor_count} actors, {static_count} static boxes rasterized")

        return occ_grid, {'actor_count': actor_count + static_count, 'source': 'carla_oracle'}

    def _swept_footprint_polygons(self, traj):
        half_len = self.vehicle_length / 2 + self.longitudinal_margin
        half_wid = self.vehicle_width / 2 + self.lateral_margin
        corners = np.array([
            [-half_len, -half_wid],
            [ half_len, -half_wid],
            [ half_len,  half_wid],
            [-half_len,  half_wid]
        ], dtype=np.float32)

        polygons = []
        bbox_yaw = np.deg2rad(self._ego_bbox_rotation.yaw)
        for forward, right, yaw in self._sample_carla_local_trajectory(traj):
            center = np.array([
                forward + self._ego_bbox_location.x,
                right + self._ego_bbox_location.y
            ], dtype=np.float32)
            angle = yaw + bbox_yaw
            c, s = np.cos(angle), np.sin(angle)
            rot = np.array([[c, -s], [s, c]], dtype=np.float32)
            polygons.append((rot @ corners.T).T + center)

        return polygons

    def _footprint_from_world_vertices(self, vertices_world, ego_transform):
        if not vertices_world:
            return None, None

        points_world = np.array([[v.x, v.y, v.z] for v in vertices_world], dtype=np.float64)
        local = self._world_to_ego_local_points(points_world, ego_transform)
        if local[:, 2].max() < -2.5 or local[:, 2].min() > 5.0:
            return None, None

        hull = cv2.convexHull(local[:, :2].astype(np.float32)).reshape(-1, 2)
        center = np.mean(local[:, :2], axis=0)
        return hull, center

    def _convex_intersection_area(self, poly_a, poly_b):
        try:
            area, _ = cv2.intersectConvexConvex(
                poly_a.astype(np.float32),
                poly_b.astype(np.float32)
            )
            return float(area)
        except Exception:
            return 0.0

    def _compute_path_blockage(self, traj, ego_transform):
        swept_polys = self._swept_footprint_polygons(traj)
        if not swept_polys:
            return {
                'path_blocked': False,
                'path_blockage_area': 0.0,
                'blocker_type': 'none',
                'blocker_direction': 'none',
                'blocker_distance': 99.0
            }

        ego_xy = np.array([ego_transform.location.x, ego_transform.location.y], dtype=np.float64)
        max_area = 0.0
        min_distance = 99.0
        blocker_type = 'none'
        blocker_direction = 'none'

        def check_footprint(vertices_world, obj_type):
            nonlocal max_area, min_distance, blocker_type, blocker_direction
            footprint, center = self._footprint_from_world_vertices(vertices_world, ego_transform)
            if footprint is None:
                return

            distance = float(np.linalg.norm(center))

            # Broad phase: only compare obstacle footprints near the swept envelope.
            for sweep in swept_polys:
                if np.linalg.norm(np.mean(sweep, axis=0) - center) > 18.0:
                    continue
                area = self._convex_intersection_area(sweep, footprint)
                if area > max_area:
                    max_area = area
                    min_distance = distance
                    blocker_type = obj_type
                    blocker_direction = self._direction_bin(float(center[0]), float(center[1]))

        for actor in self.world.get_actors():
            if actor.attributes.get('role_name') == 'hero':
                continue

            type_id = actor.type_id
            if not (type_id.startswith(('vehicle.', 'walker.')) or 'static.prop' in type_id):
                continue

            try:
                transform = actor.get_transform()
                dist = transform.location.distance(ego_transform.location)
                if dist > 70.0:
                    continue
                check_footprint(self._bbox_world_vertices(actor.bounding_box, transform), type_id)
            except Exception:
                continue

        for bbox in self._get_static_level_bbs():
            try:
                loc = bbox.location
                radius = max(float(bbox.extent.x), float(bbox.extent.y))
                if np.linalg.norm([loc.x - ego_xy[0], loc.y - ego_xy[1]]) - radius > 70.0:
                    continue
                check_footprint(self._bbox_world_vertices(bbox), 'level_static')
            except Exception:
                continue

        path_blocked = (
            max_area >= self.path_block_area_threshold
            or (max_area > 0.0 and min_distance <= self.path_block_near_distance)
        )

        return {
            'path_blocked': path_blocked,
            'path_blockage_area': max_area,
            'blocker_type': blocker_type,
            'blocker_direction': blocker_direction,
            'blocker_distance': min_distance
        }

    def evaluate(self, traj, ego_transform, speed, ego_actor=None):
        """Main evaluation function."""
        self._draw_counter += 1
        self._refresh_ego_geometry(ego_actor)
        speed_value = float(np.asarray(speed).reshape(-1)[0])
        eval_traj, eval_traj_held = self._select_eval_trajectory(traj, speed_value)

        occ_grid, occ_meta = self.build_carla_occupancy(ego_transform)
        ego_swept = self.build_ego_swept_mask(eval_traj, ego_yaw=ego_transform.rotation.yaw)
        path_block = self._compute_path_blockage(eval_traj, ego_transform)

        # Overlap ratio
        overlap = (ego_swept * occ_grid).sum()
        total = ego_swept.sum() + 1e-8
        overlap_ratio = overlap / total

        # Minimum distance with forward bias
        occupied = np.argwhere(occ_grid > 0.5)
        if len(occupied) > 0:
            local_occ = self._grid_to_local(occupied)
            dists = np.linalg.norm(local_occ, axis=1)
            
            # Forward occupancy is more urgent than equally close rear/side occupancy.
            forward_bias = np.maximum(local_occ[:, 0], 0.0) * 0.25
            effective_dists = dists - forward_bias
            min_dist = max(float(effective_dists.min()), 0.5)
        else:
            min_dist = 99.0

        # Required deceleration is path-constrained. Generic nearest occupancy
        # pixels are logged, but they are too noisy to command braking.
        decel_dist = path_block['blocker_distance'] if path_block['path_blocked'] else 50.0
        req_decel = (speed_value ** 2) / (2 * max(decel_dist, 1.0)) if speed_value > 0.5 and decel_dist < 50 else 0.0
        req_decel = min(req_decel, 8.0)

        cri_score, cri_direction = self._compute_cri(ego_transform, speed_value)

        # Risk score
        Gc, gc_terms = self._compute_gc_score(overlap_ratio, min_dist, req_decel)
        Gc = max(Gc, cri_score)

        geometry_blocked = path_block['path_blocked']
        imminent_block = (
            path_block['path_blocked']
            and (
                path_block['blocker_distance'] <= (
                    self.close_block_base_distance
                    + self.close_block_time_headway * max(speed_value, 0.0)
                )
                or req_decel >= self.emergency_decel_threshold
            )
        )

        if geometry_blocked:
            self._path_block_confirmations += 1
        else:
            self._path_block_confirmations = 0

        # Do not arm a brake hold from a standstill-only prediction. VAD can emit
        # unstable full-length plans while the ego is stationary, which made a
        # single false blocker latch the Guardian forever.
        moving_confirmed_block = (
            imminent_block
            and speed_value > self.min_new_block_speed
            and (
                speed_value >= self.fast_block_speed
                or self._path_block_confirmations >= self.path_block_confirm_frames
            )
        )
        held_confirmed_block = (
            eval_traj_held
            and imminent_block
            and self._block_memory_frames > 0
        )

        if moving_confirmed_block or held_confirmed_block:
            self._hold_frames = self.hold_frames_after_block
            self._block_memory_frames = self.block_memory_frames_after_seen
            Gc = max(Gc, 1.0)
        else:
            self._hold_frames = max(0, self._hold_frames - 1)
            self._block_memory_frames = max(0, self._block_memory_frames - 1)
            if speed_value <= self.min_new_block_speed and not eval_traj_held:
                self._hold_frames = 0

        intervene = moving_confirmed_block or held_confirmed_block or (
            self._hold_frames > 0 and speed_value > self.min_new_block_speed
        )

        # Enhanced visualization
        # self.visualize_live(traj, ego_transform, occ_grid, ego_swept, speed, intervene)
        if self._draw_counter % self.debug_draw_interval == 0:
            self.visualize_in_carla(eval_traj, ego_transform, ego_actor=ego_actor, intervene=intervene)

        # Debug alignment check (every 20 frames)
        # self.debug_alignment(traj, ego_transform, occ_grid, ego_swept)

        # Logging
        ego_pos = np.array([ego_transform.location.x, ego_transform.location.y])
        ego_yaw_rad = np.deg2rad(ego_transform.rotation.yaw)

        self.log_data.append({
            'ego_x': ego_pos[0], 'ego_y': ego_pos[1], 'ego_yaw': ego_yaw_rad,
            'ego_speed': speed_value,
            'overlap_ratio': overlap_ratio,
            'min_occupied_distance': min_dist,
            'ttc_occupied': 99.0,           # TODO: improve if needed
            'req_decel': req_decel,
            'gc_score': Gc,
            **gc_terms,
            'cri_score': cri_score,
            'risk_direction': path_block['blocker_direction'] if path_block['path_blocked'] else cri_direction,
            'path_blocked': int(path_block['path_blocked']),
            'path_blockage_area': path_block['path_blockage_area'],
            'blocker_distance': path_block['blocker_distance'],
            'blocker_type': path_block['blocker_type'],
            'eval_traj_held': int(eval_traj_held),
            'occupancy_source': occ_meta['source'],
            'intervention_flag': int(intervene),
            'brake_command': 1.0 if intervene else 0.0,
            'actor_count': occ_meta['actor_count']
        })

        return intervene, (1.0 if intervene else 0.0)

    def _compute_gc_score(self, overlap_ratio, min_dist, req_decel):
        overlap_score = min(overlap_ratio * 1.1, 0.65)
        potential_score = np.exp(-min_dist / 4.5)
        decel_score = min(req_decel / 7.0, 1.0)

        Gc = max(overlap_score, potential_score * 0.45, decel_score)

        return Gc, {
            'gc_overlap_term': overlap_score,
            'gc_potential_term': potential_score,
            'gc_decel_term': decel_score,
            'gc_ttc_term': 0.0
        }

    def _direction_bin(self, forward, right):
        angle = math.degrees(math.atan2(right, forward))
        if -22.5 <= angle < 22.5:
            return 'front'
        if 22.5 <= angle < 67.5:
            return 'front-right'
        if 67.5 <= angle < 112.5:
            return 'right'
        if 112.5 <= angle < 157.5:
            return 'rear-right'
        if angle >= 157.5 or angle < -157.5:
            return 'rear'
        if -157.5 <= angle < -112.5:
            return 'rear-left'
        if -112.5 <= angle < -67.5:
            return 'left'
        return 'front-left'

    def _velocity_to_ego_local(self, velocity, ego_transform):
        yaw = np.deg2rad(ego_transform.rotation.yaw)
        c, s = np.cos(yaw), np.sin(yaw)
        forward = velocity.x * c + velocity.y * s
        right = -velocity.x * s + velocity.y * c
        return np.array([forward, right], dtype=np.float64)

    def _compute_cri(self, ego_transform, ego_speed, max_dist=45.0):
        """Lightweight directional risk index in the ego CARLA frame."""
        max_risk = 0.0
        max_direction = 'none'

        for actor in self.world.get_actors():
            if actor.attributes.get('role_name') == 'hero':
                continue

            type_id = actor.type_id
            if not (type_id.startswith(('vehicle.', 'walker.')) or 'static.prop' in type_id):
                continue

            try:
                transform = actor.get_transform()
                bbox_center = self._transform_location(transform, actor.bounding_box.location)
                local = self._world_to_ego_local_points(
                    np.array([[bbox_center.x, bbox_center.y, bbox_center.z]], dtype=np.float64),
                    ego_transform
                )[0]
                forward, right = float(local[0]), float(local[1])
                dist = math.hypot(forward, right)
                if dist < 1e-3 or dist > max_dist or local[2] < -2.5 or local[2] > 5.0:
                    continue

                if type_id.startswith(('vehicle.', 'walker.')):
                    actor_vel = self._velocity_to_ego_local(actor.get_velocity(), ego_transform)
                else:
                    actor_vel = np.zeros(2, dtype=np.float64)

                rel_vel = actor_vel - np.array([float(ego_speed), 0.0], dtype=np.float64)
                unit = np.array([forward, right], dtype=np.float64) / dist
                closing_speed = max(0.0, -float(np.dot(rel_vel, unit)))
                ttc = dist / max(closing_speed, 1e-3) if closing_speed > 0.1 else 99.0

                distance_term = math.exp(-dist / 12.0)
                ttc_term = math.exp(-ttc / 3.0) if ttc < 99.0 else 0.0
                lane_term = math.exp(-((abs(right) / max(self.vehicle_width + 1.5, 0.1)) ** 2))
                front_term = 1.0 if forward >= -1.0 else 0.35
                dynamic_term = 1.0 if type_id.startswith(('vehicle.', 'walker.')) else 0.8

                risk = dynamic_term * front_term * max(distance_term * 0.65, ttc_term) * max(lane_term, 0.25)
                if risk > max_risk:
                    max_risk = min(float(risk), 1.0)
                    max_direction = self._direction_bin(forward, right)
            except Exception:
                continue

        return max_risk, max_direction

    def save_log(self, step, timestamp, throttle_cmd):
        if not self.log_data:
            return
        entry = self.log_data[-1]
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                step, timestamp,
                round(entry['ego_x'], 2), round(entry['ego_y'], 2),
                round(entry['ego_yaw'], 4), round(entry['ego_speed'], 2),
                round(entry['overlap_ratio'], 4),
                round(entry['min_occupied_distance'], 2),
                round(entry['ttc_occupied'], 2),
                round(entry['req_decel'], 4),
                round(entry['gc_score'], 4),
                round(entry['gc_overlap_term'], 4),
                round(entry['gc_potential_term'], 4),
                round(entry['gc_decel_term'], 4),
                round(entry['gc_ttc_term'], 4),
                round(entry['cri_score'], 4),
                entry['risk_direction'],
                entry['path_blocked'],
                round(entry['path_blockage_area'], 4),
                round(entry['blocker_distance'], 2),
                entry['blocker_type'],
                entry['eval_traj_held'],
                entry['occupancy_source'],
                entry['intervention_flag'],
                round(entry['brake_command'], 2),
                round(throttle_cmd, 2),
                entry['actor_count']
            ])
    
    def visualize_live(self, traj, ego_transform, occ_grid, ego_swept, speed, intervene):
        """
        Enhanced debugging visualization with 4-panel layout.
        Panel 1: World Occupancy (what CARLA sees)
        Panel 2: Ego Swept Volume (what Guardian predicts)
        Panel 3: Overlap Detection (conflict zones)
        Panel 4: Composite View (everything together)
        """
        if not hasattr(self, 'vis_initialized'):
            import matplotlib.pyplot as plt
            plt.ion()
            # Create 2x2 subplot layout
            self.fig, self.axes = plt.subplots(2, 2, figsize=(18, 16))
            self.fig.tight_layout(pad=4.0)
            self.vis_initialized = True
            self.frame_count = 0
        
        self.frame_count += 1
        
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Get current Gc and overlap
        gc = self.log_data[-1]['gc_score'] if self.log_data else 0.0
        overlap_ratio = self.log_data[-1]['overlap_ratio'] if self.log_data else 0.0
        min_dist = self.log_data[-1]['min_occupied_distance'] if self.log_data else 99.0
        traj_carla = self._vad_traj_to_carla_local(traj)
        
        # ========== PANEL 1: World Occupancy ==========
        ax1 = self.axes[0, 0]
        ax1.imshow(occ_grid, origin='lower', cmap='Reds', alpha=0.8,
                extent=[0, self.grid_size, 0, self.grid_size])
        ax1.set_title(f'World Occupancy (CARLA Ground Truth)\n'
                    f'Actors: {self.log_data[-1]["actor_count"] if self.log_data else 0}',
                    fontsize=12, fontweight='bold')
        
        # Draw ego vehicle current position
        self._draw_ego_vehicle(ax1, ego_transform, color='lime', label='Ego')
        
        # Draw coordinate axes at ego position
        ego_gx, ego_gy = self.origin, self.origin
        ax1.arrow(ego_gx, ego_gy, 15, 0, head_width=3, head_length=2, 
                fc='cyan', ec='cyan', linewidth=2, alpha=0.7)
        ax1.text(ego_gx + 18, ego_gy, 'Forward', color='cyan', fontsize=10, fontweight='bold')
        ax1.arrow(ego_gx, ego_gy, 0, 15, head_width=3, head_length=2,
                fc='yellow', ec='yellow', linewidth=2, alpha=0.7)
        ax1.text(ego_gx + 2, ego_gy + 18, 'Right', color='yellow', fontsize=10, fontweight='bold')
        
        self._add_grid_formatting(ax1)
        
        # ========== PANEL 2: Ego Swept Volume ==========
        ax2 = self.axes[0, 1]
        ax2.imshow(ego_swept, origin='lower', cmap='Blues', alpha=0.8,
                extent=[0, self.grid_size, 0, self.grid_size])
        ax2.set_title(f'Ego Swept Volume (Predicted Path)\n'
                    f'Speed: {speed:.1f} m/s',
                    fontsize=12, fontweight='bold')
        
        # Draw ego vehicle current position
        self._draw_ego_vehicle(ax2, ego_transform, color='lime', label='Current Ego')
        
        # Draw trajectory waypoints with numbers
        traj_points = []
        for i, pt in enumerate(traj_carla[:12]):  # First 12 waypoints
            gx, gy = self._local_to_grid(np.array([pt], dtype=np.float64))[0]
            
            if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                traj_points.append((gx, gy))
                # Plot waypoint
                ax2.plot(gx, gy, 'o', color='cyan', markersize=8, alpha=0.9)
                # Add waypoint number
                if i % 2 == 0:  # Label every other waypoint to avoid clutter
                    ax2.text(gx + 2, gy + 2, f'{i}', color='white', 
                            fontsize=8, fontweight='bold',
                            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        # Connect trajectory
        if len(traj_points) > 1:
            tp = np.array(traj_points)
            ax2.plot(tp[:, 0], tp[:, 1], '--', color='cyan', linewidth=2.5, alpha=0.8)
        
        self._add_grid_formatting(ax2)
        
        # ========== PANEL 3: Overlap Detection ==========
        ax3 = self.axes[1, 0]
        
        # Show overlap region in RED
        overlap_mask = (ego_swept > 0.5) & (occ_grid > 0.5)
        
        # Create RGB image for better visualization
        rgb_image = np.zeros((self.grid_size, self.grid_size, 3))
        rgb_image[:, :, 0] = occ_grid * 0.6  # Red channel: obstacles
        rgb_image[:, :, 2] = ego_swept * 0.6  # Blue channel: ego swept
        rgb_image[overlap_mask, 0] = 1.0  # Overlap: bright red
        rgb_image[overlap_mask, 1] = 0.0
        rgb_image[overlap_mask, 2] = 0.0
        
        ax3.imshow(rgb_image, origin='lower', extent=[0, self.grid_size, 0, self.grid_size])
        ax3.set_title(f'Conflict Detection\n'
                    f'Overlap Ratio: {overlap_ratio:.3f} | Min Dist: {min_dist:.1f}m',
                    fontsize=12, fontweight='bold', 
                    color='red' if intervene else 'green')
        
        # Draw ego vehicle
        self._draw_ego_vehicle(ax3, ego_transform, color='yellow', label='Ego')
        
        # Highlight overlap pixels
        if overlap_mask.any():
            overlap_coords = np.argwhere(overlap_mask)
            ax3.scatter(overlap_coords[:, 1], overlap_coords[:, 0], 
                    c='red', s=5, alpha=0.6, marker='s', label='Conflict Zone')
            ax3.legend(loc='upper right')
        
        self._add_grid_formatting(ax3)
        
        # ========== PANEL 4: Composite View + Debug Info ==========
        ax4 = self.axes[1, 1]
        
        # Composite image
        composite = np.zeros((self.grid_size, self.grid_size, 3))
        composite[:, :, 0] = occ_grid * 0.7  # Red: obstacles
        composite[:, :, 2] = ego_swept * 0.5  # Blue: ego swept
        composite[overlap_mask, 0] = 1.0     # Bright red: conflicts
        composite[overlap_mask, 1] = 0.2
        
        ax4.imshow(composite, origin='lower', extent=[0, self.grid_size, 0, self.grid_size])
        
        # Title with intervention status
        status = "🚨 INTERVENING 🚨" if intervene else "✓ Safe Driving"
        title_color = 'red' if intervene else 'green'
        ax4.set_title(f'{status}\n'
                    f'Gc Score: {gc:.3f} (threshold: 0.72)',
                    fontsize=14, fontweight='bold', color=title_color)
        
        # Draw ego vehicle
        self._draw_ego_vehicle(ax4, ego_transform, color='lime', label='Ego Vehicle')
        
        # Draw trajectory
        for i, pt in enumerate(traj_carla[:12]):
            gx, gy = self._local_to_grid(np.array([pt], dtype=np.float64))[0]
            if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                ax4.plot(gx, gy, 'o', color='cyan', markersize=6, alpha=0.8)
        
        # Add debug text box
        if self.log_data:
            debug_text = (
                f"Frame: {self.frame_count}\n"
                f"Ego: ({ego_transform.location.x:.1f}, {ego_transform.location.y:.1f})\n"
                f"Yaw: {ego_transform.rotation.yaw:.1f}°\n"
                f"Speed: {speed:.2f} m/s\n"
                f"───────────────\n"
                f"Gc Terms:\n"
                f"  Overlap: {self.log_data[-1]['gc_overlap_term']:.3f}\n"
                f"  Potential: {self.log_data[-1]['gc_potential_term']:.3f}\n"
                f"  Decel: {self.log_data[-1]['gc_decel_term']:.3f}\n"
                f"───────────────\n"
                f"Req Decel: {self.log_data[-1]['req_decel']:.2f} m/s²\n"
                f"Actors: {self.log_data[-1]['actor_count']}"
            )
            ax4.text(0.02, 0.98, debug_text, transform=ax4.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                    color='white', family='monospace')
        
        self._add_grid_formatting(ax4)
        
        # Overall figure title
        self.fig.suptitle(f'Guardian Debug Visualization - Frame {self.frame_count}',
                        fontsize=16, fontweight='bold')
        
        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Optional: Save frames for post-analysis
        if intervene and self.frame_count % 5 == 0:
            save_path = f'debug_frames/frame_{self.frame_count:05d}.png'
            os.makedirs('debug_frames', exist_ok=True)
            self.fig.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"[DEBUG] Saved frame to {save_path}")

    def _draw_ego_vehicle(self, ax, ego_transform, color='lime', label='Ego'):
        """Draw ego in the ego-local grid frame."""
        ego_gx = self.origin
        ego_gy = self.origin
        yaw = np.deg2rad(self._ego_bbox_rotation.yaw)

        half_l = self.vehicle_length / 2 / self.resolution
        half_w = self.vehicle_width / 2 / self.resolution

        corners_local = np.array([
            [-half_l, -half_w],
            [ half_l, -half_w],
            [ half_l,  half_w],
            [-half_l,  half_w]
        ])

        rot = np.array([[np.cos(yaw), -np.sin(yaw)],
                        [np.sin(yaw),  np.cos(yaw)]])
        corners = (rot @ corners_local.T).T + [ego_gx, ego_gy]

        ax.fill(corners[:, 0], corners[:, 1], 
                color=color, alpha=0.45, linewidth=3, edgecolor=color, label=label)

        # Forward arrow
        arrow_len = half_l * 1.8
        ax.arrow(ego_gx, ego_gy, 
                 arrow_len * np.cos(yaw), 
                 arrow_len * np.sin(yaw),
                 head_width=4, head_length=4, 
                 fc=color, ec='white', linewidth=2)

    def _add_grid_formatting(self, ax):
        """Helper to add consistent grid formatting."""
        # Meter-based ticks
        ticks = np.linspace(0, self.grid_size, 9)
        labels = [f'{int((t - self.origin) * self.resolution)}' for t in ticks]
        
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Forward (meters)', fontsize=10)
        ax.set_ylabel('Right (meters)', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_aspect('equal')
    
    def debug_alignment(self, traj, ego_transform, occ_grid, ego_swept):
        """
        Print coordinate alignment debug info.
        Call this every 20 frames to verify everything is correct.
        """
        if self.frame_count % 20 != 0:
            return
        
        print("\n" + "="*60)
        print(f"[DEBUG ALIGNMENT CHECK - Frame {self.frame_count}]")
        print("="*60)
        
        # Ego position
        ego_world = np.array([ego_transform.location.x, ego_transform.location.y])
        ego_yaw = np.deg2rad(ego_transform.rotation.yaw)
        print(f"Ego World Position: ({ego_world[0]:.2f}, {ego_world[1]:.2f})")
        print(f"Ego Yaw: {np.rad2deg(ego_yaw):.1f}°")
        print(f"Ego Grid Position: ({self.origin}, {self.origin}) [should be center]")
        
        # First trajectory waypoint
        traj_carla = self._vad_traj_to_carla_local(traj)
        if len(traj_carla) > 0:
            first_wp = traj_carla[0]
            gx, gy = self._local_to_grid(np.array([first_wp], dtype=np.float64))[0]
            print(f"\nFirst Waypoint (CARLA local fwd/right): ({first_wp[0]:.2f}, {first_wp[1]:.2f})")
            print(f"First Waypoint (grid): ({gx:.1f}, {gy:.1f})")
            print(f"Distance from ego center: {np.linalg.norm(first_wp):.2f}m")
            
            # Check if first waypoint is in swept volume
            if 0 <= int(gx) < self.grid_size and 0 <= int(gy) < self.grid_size:
                is_in_swept = ego_swept[int(gy), int(gx)] > 0.5
                print(f"First waypoint in swept volume: {is_in_swept} ✓" if is_in_swept else "❌ ALIGNMENT ISSUE!")
        
        # Occupancy stats
        occupied_pixels = (occ_grid > 0.5).sum()
        occupied_area = occupied_pixels * (self.resolution ** 2)
        print(f"\nOccupied pixels: {occupied_pixels}")
        print(f"Occupied area: {occupied_area:.1f} m²")
        
        # Swept volume stats
        swept_pixels = (ego_swept > 0.5).sum()
        swept_area = swept_pixels * (self.resolution ** 2)
        print(f"\nSwept volume pixels: {swept_pixels}")
        print(f"Swept volume area: {swept_area:.1f} m²")
        print(f"Expected swept area: ~{self.vehicle_length * self.vehicle_width * len(traj):.1f} m²")
        
        # Overlap stats
        overlap = ((ego_swept > 0.5) & (occ_grid > 0.5)).sum()
        print(f"\nOverlap pixels: {overlap}")
        print(f"Overlap ratio: {overlap / max(swept_pixels, 1):.3f}")
        
        # Find nearest obstacle
        occupied = np.argwhere(occ_grid > 0.5)
        if len(occupied) > 0:
            local_occ = self._grid_to_local(occupied)
            dists = np.linalg.norm(local_occ, axis=1)
            min_idx = np.argmin(dists)
            nearest_px = occupied[min_idx]
            nearest_dist = dists[min_idx]
            
            print(f"\nNearest obstacle (grid): ({nearest_px[1]}, {nearest_px[0]})")
            print(f"Nearest obstacle distance: {nearest_dist:.2f}m")
        
        print("="*60 + "\n")

    def visualize_in_carla(self, traj, ego_transform, occ_grid=None, ego_actor=None, intervene=False):
        """
        Draw debug information directly in CARLA world view.
        Call this from evaluate() every few steps.
        """
        debug = self.world.debug

        # Color definitions
        COLOR_EGO = carla.Color(0, 255, 0)        # Green
        COLOR_SWEEP = carla.Color(0, 200, 255)    # Cyan
        COLOR_OBSTACLE = carla.Color(255, 50, 50) # Red
        COLOR_CONFLICT = carla.Color(255, 255, 0) # Yellow

        life_time = max(self.fixed_delta_seconds * (self.debug_draw_interval + 1), 0.05)

        # 1. Current Ego Vehicle Bounding Box
        ego_box_center = self._transform_location(ego_transform, self._ego_bbox_location)
        ego_bb = carla.BoundingBox(
            ego_box_center,
            carla.Vector3D(self.vehicle_length / 2, self.vehicle_width / 2, self.vehicle_height / 2)
        )
        debug.draw_box(
            ego_bb,
            self._compose_rotation(ego_transform.rotation, self._ego_bbox_rotation),
            thickness=0.08,
            color=COLOR_EGO,
            life_time=life_time
        )

        # 2. Draw Swept Volume along trajectory (as series of boxes)
        poses = self._sample_carla_local_trajectory(traj)
        stride = max(1, int(np.ceil(len(poses) / 18.0)))
        for i, (forward, right, yaw) in enumerate(poses[::stride][:18]):
            future_transform = self._future_transform_from_pose(ego_transform, forward, right, yaw)
            box_center = self._transform_location(future_transform, self._ego_bbox_location)
            box = carla.BoundingBox(
                box_center,
                carla.Vector3D(
                    self.vehicle_length / 2 + self.longitudinal_margin,
                    self.vehicle_width / 2 + self.lateral_margin,
                    self.vehicle_height / 2
                )
            )
            color = COLOR_SWEEP if not intervene else COLOR_CONFLICT

            debug.draw_box(
                box,
                self._compose_rotation(future_transform.rotation, self._ego_bbox_rotation),
                thickness=0.05,
                color=color,
                life_time=life_time
            )

        # 3. Draw detected obstacles (from CARLA actors)
        drawn = 0
        for actor in self.world.get_actors():
            if actor.attributes.get('role_name') == 'hero':
                continue
            if not (actor.type_id.startswith(('vehicle.', 'walker.')) or 'static.prop' in actor.type_id):
                continue
                
            try:
                dist = actor.get_transform().location.distance(ego_transform.location)
                if dist > 60.0:
                    continue
                    
                t = actor.get_transform()
                bbox = actor.bounding_box
                world_center = self._transform_location(t, bbox.location)
                world_box = carla.BoundingBox(world_center, bbox.extent)
                
                debug.draw_box(
                    world_box,
                    self._compose_rotation(t.rotation, bbox.rotation),
                    thickness=0.12,
                    color=COLOR_OBSTACLE,
                    life_time=life_time
                )
                drawn += 1
                if drawn >= self.max_debug_boxes:
                    break
            except:
                continue

        if drawn < self.max_debug_boxes:
            ego_xy = np.array([ego_transform.location.x, ego_transform.location.y], dtype=np.float64)
            for bbox in self._get_static_level_bbs():
                try:
                    loc = bbox.location
                    radius = max(float(bbox.extent.x), float(bbox.extent.y))
                    dist = np.linalg.norm([loc.x - ego_xy[0], loc.y - ego_xy[1]]) - radius
                    if dist > 60.0:
                        continue
                    debug.draw_box(
                        bbox,
                        bbox.rotation,
                        thickness=0.06,
                        color=COLOR_OBSTACLE,
                        life_time=life_time
                    )
                    drawn += 1
                    if drawn >= self.max_debug_boxes:
                        break
                except:
                    continue

        # 4. Optional: Big text when intervening
        '''
        if intervene:
            debug.draw_string(
                ego_transform.location + carla.Location(z=4.0),
                "🚨 GUARDIAN INTERVENING 🚨",
                color=carla.Color(255, 100, 0),
                life_time=0.1,
                size=0.8
            )
        '''
    
class VadAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self.steer_step = 0
        self.last_moving_status = 0
        self.last_moving_step = -1
        self.last_steer = 0
        self.pidcontroller = PIDController() 
        self.config_path = path_to_conf_file.split('+')[0]
        self.ckpt_path = path_to_conf_file.split('+')[1]
        if IS_BENCH2DRIVE:
            self.save_name = path_to_conf_file.split('+')[-1]
        else:
            self.save_name = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        cfg = Config.fromfile(self.config_path)
        if hasattr(cfg, 'plugin'):
            if cfg.plugin:
                import importlib
                if hasattr(cfg, 'plugin_dir'):
                    plugin_dir = cfg.plugin_dir
                    plugin_dir = os.path.join("Bench2DriveZoo", plugin_dir)
                    _module_dir = os.path.dirname(plugin_dir)
                    _module_dir = _module_dir.split('/')
                    _module_path = _module_dir[0]
                    for m in _module_dir[1:]:
                        _module_path = _module_path + '.' + m
                    print(_module_path)
                    plg_lib = importlib.import_module(_module_path)  
  
        self.model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        checkpoint = load_checkpoint(self.model, self.ckpt_path, map_location='cpu', strict=True)
        self.model.cuda()
        self.model.half()
        self.model.eval()
        print(next(self.model.parameters()).dtype)

        self.inference_only_pipeline = []
        for inference_only_pipeline in cfg.inference_only_pipeline:
            if inference_only_pipeline["type"] not in ['LoadMultiViewImageFromFilesInCeph','LoadMultiViewImageFromFiles']:
                self.inference_only_pipeline.append(inference_only_pipeline)
        self.inference_only_pipeline = Compose(self.inference_only_pipeline)

        self.takeover = False
        self.stop_time = 0
        self.takeover_time = 0
        self.save_path = None
        self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
        self.lat_ref, self.lon_ref = 42.0, 2.0

        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0	
        self.prev_control = control
        self.prev_control_cache = []

        try:
            self.world = carla.Client('localhost', 2000).get_world()
        except:
            client = carla.Client('localhost', 2000)
            client.set_timeout(10.0)
            self.world = client.get_world()
        
        # Initialize Guardian with CARLA world
        self.guardian = Guardian(world=self.world)
        
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ['ROUTES']).stem + '_'
            string += self.save_name
            self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
            self.save_path.mkdir(parents=True, exist_ok=True)
            (self.save_path / 'rgb_front').mkdir(parents=True, exist_ok=True)
            (self.save_path / 'rgb_front_right').mkdir(parents=True, exist_ok=True)
            (self.save_path / 'rgb_front_left').mkdir(parents=True, exist_ok=True)
            (self.save_path / 'rgb_back').mkdir(parents=True, exist_ok=True)
            (self.save_path / 'rgb_back_right').mkdir(parents=True, exist_ok=True)
            (self.save_path / 'rgb_back_left').mkdir(parents=True, exist_ok=True)
            (self.save_path / 'meta').mkdir(parents=True, exist_ok=True)
            (self.save_path / 'bev').mkdir(parents=True, exist_ok=True)
   
        self.lidar2img = {
        'CAM_FRONT':np.array([[ 1.14251841e+03,  8.00000000e+02,  0.00000000e+00, -9.52000000e+02],
                              [ 0.00000000e+00,  4.50000000e+02, -1.14251841e+03, -8.09704417e+02],
                              [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -1.19000000e+00],
                              [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_FRONT_LEFT':np.array([[ 6.03961325e-14,  1.39475744e+03,  0.00000000e+00, -9.20539908e+02],
                                   [-3.68618420e+02,  2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
                                   [-8.19152044e-01,  5.73576436e-01,  0.00000000e+00, -8.29094072e-01],
                                   [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_FRONT_RIGHT':np.array([[ 1.31064327e+03, -4.77035138e+02,  0.00000000e+00,-4.06010608e+02],
                                    [ 3.68618420e+02,  2.58109396e+02, -1.14251841e+03,-6.47296750e+02],
                                    [ 8.19152044e-01,  5.73576436e-01,  0.00000000e+00,-8.29094072e-01],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]),
        'CAM_BACK':np.array([[-5.60166031e+02, -8.00000000e+02,  0.00000000e+00, -1.28800000e+03],
                     [ 5.51091060e-14, -4.50000000e+02, -5.60166031e+02, -8.58939847e+02],
                     [ 1.22464680e-16, -1.00000000e+00,  0.00000000e+00, -1.61000000e+00],
                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_BACK_LEFT':np.array([[-1.14251841e+03,  8.00000000e+02,  0.00000000e+00, -6.84385123e+02],
                                  [-4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                  [-9.39692621e-01, -3.42020143e-01,  0.00000000e+00, -4.92889531e-01],
                                  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
  
        'CAM_BACK_RIGHT': np.array([[ 3.60989788e+02, -1.34723223e+03,  0.00000000e+00, -1.04238127e+02],
                                    [ 4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                    [ 9.39692621e-01, -3.42020143e-01,  0.00000000e+00, -4.92889531e-01],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        }
        self.lidar2cam = {
        'CAM_FRONT':np.array([[ 1.  ,  0.  ,  0.  ,  0.  ],
                              [ 0.  ,  0.  , -1.  , -0.24],
                              [ 0.  ,  1.  ,  0.  , -1.19],
                              [ 0.  ,  0.  ,  0.  ,  1.  ]]),
        'CAM_FRONT_LEFT':np.array([[ 0.57357644,  0.81915204,  0.  , -0.22517331],
                                   [ 0.        ,  0.        , -1.  , -0.24      ],
                                   [-0.81915204,  0.57357644,  0.  , -0.82909407],
                                   [ 0.        ,  0.        ,  0.  ,  1.        ]]),
        'CAM_FRONT_RIGHT':np.array([[ 0.57357644, -0.81915204, 0.  ,  0.22517331],
                                   [ 0.        ,  0.        , -1.  , -0.24      ],
                                   [ 0.81915204,  0.57357644,  0.  , -0.82909407],
                                   [ 0.        ,  0.        ,  0.  ,  1.        ]]),
        'CAM_BACK':np.array([[-1. ,  0.,  0.,  0.  ],
                             [ 0. ,  0., -1., -0.24],
                             [ 0. , -1.,  0., -1.61],
                             [ 0. ,  0.,  0.,  1.  ]]),
     
        'CAM_BACK_LEFT':np.array([[-0.34202014,  0.93969262,  0.  , -0.25388956],
                                  [ 0.        ,  0.        , -1.  , -0.24      ],
                                  [-0.93969262, -0.34202014,  0.  , -0.49288953],
                                  [ 0.        ,  0.        ,  0.  ,  1.        ]]),
  
        'CAM_BACK_RIGHT':np.array([[-0.34202014, -0.93969262,  0.  ,  0.25388956],
                                  [ 0.        ,  0.         , -1.  , -0.24      ],
                                  [ 0.93969262, -0.34202014 ,  0.  , -0.49288953],
                                  [ 0.        ,  0.         ,  0.  ,  1.        ]])
        }
        self.lidar2ego = np.array([[ 0. ,  1. ,  0. , -0.39],
                                   [-1. ,  0. ,  0. ,  0.  ],
                                   [ 0. ,  0. ,  1. ,  1.84],
                                   [ 0. ,  0. ,  0. ,  1.  ]])
        
        topdown_extrinsics =  np.array([[0.0, -0.0, -1.0, 50.0], [0.0, 1.0, -0.0, 0.0], [1.0, -0.0, 0.0, -0.0], [0.0, 0.0, 0.0, 1.0]])
        unreal2cam = np.array([[0,1,0,0], [0,0,-1,0], [1,0,0,0], [0,0,0,1]])
        self.coor2topdown = unreal2cam @ topdown_extrinsics
        topdown_intrinsics = np.array([[548.993771650447, 0.0, 256.0, 0], [0.0, 548.993771650447, 256.0, 0], [0.0, 0.0, 1.0, 0], [0, 0, 0, 1.0]])
        self.coor2topdown = topdown_intrinsics @ self.coor2topdown

    def _init(self):
        try:
            locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
            lon, lat = self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']
            EARTH_RADIUS_EQUA = 6378137.0
            def equations(vars):
                x, y = vars
                eq1 = lon * math.cos(x * math.pi / 180) - (locx * x * 180) / (math.pi * EARTH_RADIUS_EQUA) - math.cos(x * math.pi / 180) * y
                eq2 = math.log(math.tan((lat + 90) * math.pi / 360)) * EARTH_RADIUS_EQUA * math.cos(x * math.pi / 180) + locy - math.cos(x * math.pi / 180) * EARTH_RADIUS_EQUA * math.log(math.tan((90 + x) * math.pi / 360))
                return [eq1, eq2]
            initial_guess = [0, 0]
            solution = fsolve(equations, initial_guess)
            self.lat_ref, self.lon_ref = solution[0], solution[1]
        except Exception as e:
            print(e, flush=True)
            self.lat_ref, self.lon_ref = 0, 0      
        self._route_planner = RoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True
        self.metric_info = {}
  
  

    def sensors(self):
        sensors =[
                # camera rgb
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.80, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_LEFT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_RIGHT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -2.0, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                    'width': 1600, 'height': 900, 'fov': 110,
                    'id': 'CAM_BACK'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_LEFT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_RIGHT'
                },
                # imu
                {
                    'type': 'sensor.other.imu',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'IMU'
                },
                # gps
                {
                    'type': 'sensor.other.gnss',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'GPS'
                },
                # speed
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'SPEED'
                },
            ]
        
        '''
        if IS_BENCH2DRIVE:
            sensors += [
                    {	
                        'type': 'sensor.camera.rgb',
                        'x': 0.0, 'y': 0.0, 'z': 50.0,
                        'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                        'width': 512, 'height': 512, 'fov': 5 * 10.0,
                        'id': 'bev'
                    }]
        '''
        return sensors

    def tick(self, input_data):
        self.step += 1
        imgs = {}
        for cam in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']:
            # Direct conversion, no JPEG compression overhead
            # imgs[cam] = cv2.cvtColor(input_data[cam][1][:, :, :3], cv2.COLOR_BGR2RGB)
            imgs[cam] = input_data[cam][1][:, :, :3]

        # BEV is disabled in sensors(), so we skip it here
        bev = None 
        gps = input_data['GPS'][1][:2]
        speed = input_data['SPEED'][1]['speed']
        compass = input_data['IMU'][1][-1]
        acceleration = input_data['IMU'][1][:3]
        angular_velocity = input_data['IMU'][1][3:6]
  
        pos = self.gps_to_location(gps)
        near_node, near_command = self._route_planner.run_step(pos)
  
        if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
            compass = 0.0
            acceleration = np.zeros(3)
            angular_velocity = np.zeros(3)

        result = {
                'imgs': imgs,
                'gps': gps,
                'pos':pos,
                'speed': speed,
                'compass': compass,
                'bev': bev,
                'acceleration':acceleration,
                'angular_velocity':angular_velocity,
                'command_near':near_command,
                'command_near_xy':near_node
                }
        
        return result
    
    def get_metric_info(self):
        """
        Added for Fail2Drive compatibility. 
        Returns an empty dict or your specific research metrics.
        """
        return {}

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        tick_data = self.tick(input_data)
        results = {}
        results['lidar2img'] = []
        results['lidar2cam'] = []
        results['img'] = []
        results['folder'] = ' '
        results['scene_token'] = ' '  
        results['frame_idx'] = 0
        results['timestamp'] = self.step / 20
        results['box_type_3d'], _ = get_box_type('LiDAR')
  
        for cam in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']:
            results['lidar2img'].append(self.lidar2img[cam])
            results['lidar2cam'].append(self.lidar2cam[cam])
            results['img'].append(tick_data['imgs'][cam])
        results['lidar2img'] = np.stack(results['lidar2img'],axis=0)
        results['lidar2cam'] = np.stack(results['lidar2cam'],axis=0)
        raw_theta = tick_data['compass']   if not np.isnan(tick_data['compass']) else 0
        ego_theta = -raw_theta + np.pi/2
        rotation = list(Quaternion(axis=[0, 0, 1], radians=ego_theta))
        can_bus = np.zeros(18)
        can_bus[0] = tick_data['pos'][0]
        can_bus[1] = -tick_data['pos'][1]
        can_bus[3:7] = rotation
        can_bus[7] = tick_data['speed']
        can_bus[10:13] = tick_data['acceleration']
        can_bus[11] *= -1
        can_bus[13:16] = -tick_data['angular_velocity']
        can_bus[16] = ego_theta
        can_bus[17] = ego_theta / np.pi * 180 
        results['can_bus'] = can_bus
        ego_lcf_feat = np.zeros(9)
        ego_lcf_feat[0:2] = can_bus[0:2].copy()
        ego_lcf_feat[2:4] = can_bus[10:12].copy()
        ego_lcf_feat[4] = rotation[-1]
        ego_lcf_feat[5] = 4.89238167
        ego_lcf_feat[6] = 1.83671331
        ego_lcf_feat[7] = np.sqrt(can_bus[0]**2+can_bus[1]**2)

        if len(self.prev_control_cache)<10:
            ego_lcf_feat[8] = 0
        else:
            ego_lcf_feat[8] = self.prev_control_cache[0].steer

        command = tick_data['command_near']
        if command < 0:
            command = 4
        command -= 1
        results['command'] = command
        command_onehot = np.zeros(6)
        command_onehot[command] = 1
        results['ego_fut_cmd'] = command_onehot
        theta_to_lidar = raw_theta
        command_near_xy = np.array([tick_data['command_near_xy'][0]-can_bus[0],-tick_data['command_near_xy'][1]-can_bus[1]])
        rotation_matrix = np.array([[np.cos(theta_to_lidar),-np.sin(theta_to_lidar)],[np.sin(theta_to_lidar),np.cos(theta_to_lidar)]])
        local_command_xy = rotation_matrix @ command_near_xy
  
        ego2world = np.eye(4)
        ego2world[0:3,0:3] = Quaternion(axis=[0, 0, 1], radians=ego_theta).rotation_matrix
        ego2world[0:2,3] = can_bus[0:2]
        lidar2global = ego2world @ self.lidar2ego
        results['l2g_r_mat'] = lidar2global[0:3,0:3]
        results['l2g_t'] = lidar2global[0:3,3]
        stacked_imgs = np.stack(results['img'], axis=0) # Shape: (6, H, W, 3)
        results['img_shape'] = stacked_imgs.shape[1:] # (900, 1600, 3)
        results['ori_shape'] = results['img_shape']
        results['pad_shape'] = results['img_shape']
        results = self.inference_only_pipeline(results)
        self.device="cuda"
        input_data_batch = mm_collate_to_batch_form([results], samples_per_gpu=1)
        for key, data in input_data_batch.items():
            if key != 'img_metas':
                if isinstance(data, list):
                    # VAD often wraps images in a list [Tensor]
                    input_data_batch[key] = [d.to(self.device, non_blocking=True).half() for d in data]
                else:
                    input_data_batch[key] = data.to(self.device, non_blocking=True).half()

        with torch.cuda.amp.autocast():            
            output_data_batch = self.model(input_data_batch, return_loss=False, rescale=True)

        all_out_truck_d1 = output_data_batch[0]['pts_bbox']['ego_fut_preds'].cpu().numpy()
        all_out_truck =  np.cumsum(all_out_truck_d1,axis=1)
        out_truck = all_out_truck[command]

        steer_traj, throttle_traj, brake_traj, metadata_traj = self.pidcontroller.control_pid(out_truck, tick_data['speed'], local_command_xy)

        # Get ego transform from CARLA
        ego_vehicle = None
        for actor in self.world.get_actors().filter('vehicle.*'):
            if actor.attributes.get('role_name') == 'hero':
                ego_vehicle = actor
                break
        
        if ego_vehicle is not None:
            ego_transform = ego_vehicle.get_transform()
            
            # Guardian evaluation
            intervene, guardian_brake = self.guardian.evaluate(
                out_truck, ego_transform, tick_data['speed'], ego_actor=ego_vehicle
            )
            
            if intervene:
                print(f"🚨 GUARDIAN BRAKING (Gc={self.guardian.log_data[-1]['gc_score']:.3f})")
                brake_traj = max(brake_traj, guardian_brake)
                throttle_traj = 0.0
            
            # Log to CSV
            self.guardian.save_log(self.step, timestamp, throttle_traj)

        if brake_traj < 0.05: brake_traj = 0.0
        if throttle_traj > brake_traj: brake_traj = 0.0

        control = carla.VehicleControl()
        self.pid_metadata = metadata_traj
        self.pid_metadata['agent'] = 'only_traj'
        control.steer = np.clip(float(steer_traj), -1, 1)
        control.throttle = np.clip(float(throttle_traj), 0, 0.75)
        control.brake = np.clip(float(brake_traj), 0, 1)     
        self.pid_metadata['steer'] = control.steer
        self.pid_metadata['throttle'] = control.throttle
        self.pid_metadata['brake'] = control.brake
        self.pid_metadata['steer_traj'] = float(steer_traj)
        self.pid_metadata['throttle_traj'] = float(throttle_traj)
        self.pid_metadata['brake_traj'] = float(brake_traj)
        self.pid_metadata['plan'] = out_truck.tolist()
        self.pid_metadata['command'] = command
        self.pid_metadata['all_plan'] = all_out_truck.tolist()

        metric_info = self.get_metric_info()
        self.metric_info[self.step] = metric_info
        if SAVE_PATH is not None and self.step % 20 == 0:
            self.save(tick_data)
        self.prev_control = control
        
        if len(self.prev_control_cache)==10:
            self.prev_control_cache.pop(0)
        self.prev_control_cache.append(control)
        return control


    def save(self, tick_data):
        frame = self.step // 10

        '''
        Image.fromarray(tick_data['imgs']['CAM_FRONT']).save(self.save_path / 'rgb_front' / ('%04d.png' % frame))
        Image.fromarray(tick_data['imgs']['CAM_FRONT_LEFT']).save(self.save_path / 'rgb_front_left' / ('%04d.png' % frame))
        Image.fromarray(tick_data['imgs']['CAM_FRONT_RIGHT']).save(self.save_path / 'rgb_front_right' / ('%04d.png' % frame))
        Image.fromarray(tick_data['imgs']['CAM_BACK']).save(self.save_path / 'rgb_back' / ('%04d.png' % frame))
        Image.fromarray(tick_data['imgs']['CAM_BACK_LEFT']).save(self.save_path / 'rgb_back_left' / ('%04d.png' % frame))
        Image.fromarray(tick_data['imgs']['CAM_BACK_RIGHT']).save(self.save_path / 'rgb_back_right' / ('%04d.png' % frame))
        Image.fromarray(tick_data['bev']).save(self.save_path / 'bev' / ('%04d.png' % frame))
        '''

        outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
        json.dump(self.pid_metadata, outfile, indent=4)
        outfile.close()

        # metric info
        outfile = open(self.save_path / 'metric_info.json', 'w')
        json.dump(self.metric_info, outfile, indent=4)
        outfile.close()

    def destroy(self):
        del self.model
        torch.cuda.empty_cache()

    def gps_to_location(self, gps):
        EARTH_RADIUS_EQUA = 6378137.0
        # gps content: numpy array: [lat, lon, alt]
        lat, lon = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat+90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
        mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
        y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
        x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
        return np.array([x, y])
