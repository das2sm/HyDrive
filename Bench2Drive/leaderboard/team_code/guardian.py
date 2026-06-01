"""
Guardian Safety Monitor for CARLA Autonomous Driving
Runtime safety layer that checks planned trajectories against occupied space.
"""

import os
import csv
import datetime
import math
import numpy as np
import cv2
import carla


class Guardian:
    """
    Safety Shield / Runtime Monitor for CARLA + Bench2Drive.
    Checks if the ego vehicle's future swept volume overlaps with any obstacles.
    """
    def __init__(self, world, log_dir='results/guardian_logs', debug=True):
        self.world = world
        self.debug = debug
        
        # Grid settings
        self.grid_size = 120                    # Larger grid for better side coverage
        self.resolution = 0.5                  # meters per pixel (finer than before)
        self.origin = self.grid_size // 2
        
        # === Configurable Safety Margins ===
        self.vehicle_length = 4.8               # meters
        self.vehicle_width = 2.1                # meters
        self.vehicle_height = 2.0               # meters
        self.longitudinal_margin = 1.2          # extra meters in front/back
        self.lateral_margin = 0.0                # research logging — no safety margin (intervene=False)
        self.sweep_sample_spacing = 0.9         # max meters between swept-volume OBB samples
        self.max_debug_boxes = 80               # throttle CARLA debug draw load
        self.debug_draw_interval = 2
        self._draw_counter = 0
        self._static_level_bbs = None
        
        self.min_eval_traj_extent = 2.0

        # TTC parameters: forward cone half-angle (60° per AGENTS.md convention)
        self.ttc_half_angle_deg = 60
        self.ttc_half_angle_tan = math.tan(math.radians(self.ttc_half_angle_deg))

        self.expensive_step_interval = 1   # compute every step to minimise occupancy staleness
        self._step_counter = 0
        
        # Performance optimization: cache expensive actor iteration
        self._cached_occ_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self._cached_occ_meta = {'source': 'init', 'actor_count': 0}
        self._actor_list_cache = None
        self._actor_list_cache_age = 0
        self._actor_list_cache_max_age = 1  # Refresh actor list every frame to minimise staleness
        
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
        self.latest_occ_grid = self._cached_occ_grid
        self.latest_occ_meta = self._cached_occ_meta
        self.latest_min_dist = np.nan
        self.latest_min_dist_valid = False
        self.latest_ttc = np.nan
        self.latest_ttc_valid = False
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path = os.path.join(log_dir, f'guardian_{timestamp}.csv')
        
        # Initialize CSV
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'step', 'timestamp', 'ego_x', 'ego_y', 'ego_yaw', 'ego_speed',
                'overlap_ratio', 'min_occupied_distance', 'min_occupied_distance_valid',
                'ttc_occupied', 'ttc_occupied_valid',
                'eval_traj_held',
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
            return traj, False
        
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
        
        # Only consider vertices within the vehicle-height slab.
        # This excludes overhead parts (lamp arms, bridge undersides, canopies)
        # while keeping ground-level obstacles (poles, barriers, vehicles).
        max_z = self.vehicle_height + 0.5
        in_slab = (local[:, 2] >= -2.5) & (local[:, 2] <= max_z)
        if not in_slab.any():
            return False
        
        grid = self._local_to_grid(local[in_slab, :2])
        
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
            # 'Static' excluded — road surface meshes cause false positives
            'Walls', 'Fences', 'GuardRail', 'TrafficLight',
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
        """Build occupancy grid from CARLA actors (optimized with caching)."""
        occ_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        ego_pos = np.array([ego_transform.location.x, ego_transform.location.y], dtype=np.float64)
        
        actor_count = 0
        static_count = 0
        
        # OPTIMIZATION 1: Cache actor list to avoid fetching every frame
        self._actor_list_cache_age += 1
        if self._actor_list_cache is None or self._actor_list_cache_age >= self._actor_list_cache_max_age:
            try:
                self._actor_list_cache = list(self.world.get_actors())
                self._actor_list_cache_age = 0
            except Exception:
                self._actor_list_cache = []
        
        for actor in self._actor_list_cache:
            # Skip ego
            if hasattr(actor, 'attributes') and actor.attributes.get('role_name') == 'hero':
                continue
            
            type_id = actor.type_id
            if not type_id.startswith(('vehicle.', 'walker.')):
                continue
            
            try:
                # Skip actors with unrealistically large bounding boxes
                # Skip actors with unrealistically large bounding boxes
                # (e.g. road surface meshes with radius > 50m).
                extent = actor.bounding_box.extent
                if math.hypot(float(extent.x), float(extent.y)) > 10.0:
                    continue

                t = actor.get_transform()
                dx = t.location.x - ego_pos[0]
                dy = t.location.y - ego_pos[1]
                dist_sq = dx*dx + dy*dy
                
                # OPTIMIZATION 2: Use distance check with sqrt only when needed
                if dist_sq > max_dist * max_dist:
                    continue
                
                if self._rasterize_world_vertices(
                    occ_grid,
                    self._bbox_world_vertices(actor.bounding_box, t),
                    ego_transform
                ):
                    actor_count += 1
            except Exception:
                continue
        
        # Static obstacles (smaller subset)
        for bbox in self._get_static_level_bbs():
            try:
                loc = bbox.location
                dx = loc.x - ego_pos[0]
                dy = loc.y - ego_pos[1]
                dist_sq = dx*dx + dy*dy
                
                # Use the footprint radius for pruning
                radius = max(float(bbox.extent.x), float(bbox.extent.y))
                if np.sqrt(dist_sq) - radius > max_dist:
                    continue
                
                if self._rasterize_world_vertices(
                    occ_grid,
                    self._bbox_world_vertices(bbox),
                    ego_transform
                ):
                    static_count += 1
            except Exception:
                continue
        
        return occ_grid, {'actor_count': actor_count + static_count, 'source': 'carla_oracle'}
    
    def evaluate(self, traj, ego_transform, speed, ego_actor=None):
        """Lightweight evaluation for research logging."""
        self._draw_counter += 1
        self._refresh_ego_geometry(ego_actor)
        speed_value = float(np.asarray(speed).reshape(-1)[0])
        
        eval_traj, eval_traj_held = self._select_eval_trajectory(traj, speed_value)
        
        # ----- 1. Lazy occupancy (expensive, but cached) -----
        self._step_counter += 1
        if self._step_counter % self.expensive_step_interval == 0:
            occ_grid, occ_meta = self.build_carla_occupancy(ego_transform)
            self._cached_occ_grid = occ_grid
            self._cached_occ_meta = occ_meta
        else:
            occ_grid = getattr(self, '_cached_occ_grid', np.zeros((self.grid_size, self.grid_size)))
            occ_meta = getattr(
                self,
                '_cached_occ_meta',
                {
                    'source': 'cached',
                    'actor_count': 0
                }
            )
        
        # ----- 2. Swept volume mask (fast) -----
        ego_swept = self.build_ego_swept_mask(eval_traj, ego_yaw=ego_transform.rotation.yaw)
        overlap = (ego_swept * occ_grid).sum()
        total = ego_swept.sum() + 1e-8
        overlap_ratio = overlap / total
        
        # ----- 3. Path-filtered minimum distance -----
        conflict_mask = (ego_swept > 0.5) & (occ_grid > 0.5)

        if conflict_mask.any():
            occupied_conflict = np.argwhere(conflict_mask)
            local_conflict = self._grid_to_local(occupied_conflict)

            forward_vals, right_vals = local_conflict[:, 0], local_conflict[:, 1]
            front_mask = (forward_vals > 0) & (np.abs(right_vals) <= forward_vals * self.ttc_half_angle_tan)
            if front_mask.any():
                min_dist = float(
                    np.linalg.norm(local_conflict[front_mask], axis=1).min()
                )
            else:
                min_dist = 99.0
        else:
            min_dist = 99.0
        
        # ----- 4. TTC (closing-speed aware) -----
        # Per-actor closing-speed TTC for actors in the swept path + forward
        # cone.  Correctly handles car-following (same speed → closing near
        # zero → high TTC).  Falls back to path-based min_dist / ego_speed.
        ego_pos = np.array([ego_transform.location.x, ego_transform.location.y])
        ego_yaw_rad = math.radians(ego_transform.rotation.yaw)
        cos_yaw = math.cos(ego_yaw_rad)
        sin_yaw = math.sin(ego_yaw_rad)

        ttc_from_dist = 99.0
        if speed_value > 0.5 and min_dist < 50:
            ttc_from_dist = min_dist / max(speed_value, 0.1)

        closing_ttc = 99.0
        found_actor_on_path = False
        for actor in getattr(self, '_actor_list_cache', []):
            if actor.attributes.get('role_name') == 'hero':
                continue
            if not actor.type_id.startswith(('vehicle.', 'walker.')):
                continue
            try:
                extent = actor.bounding_box.extent
                if math.hypot(float(extent.x), float(extent.y)) > 10.0:
                    continue

                t = actor.get_transform()
                dx = t.location.x - ego_pos[0]
                dy = t.location.y - ego_pos[1]
                dist = math.hypot(dx, dy)
                if dist > 75.0:
                    continue

                forward = dx * cos_yaw + dy * sin_yaw
                right = -dx * sin_yaw + dy * cos_yaw
                if forward <= 0 or math.fabs(right) > forward * self.ttc_half_angle_tan:
                    continue

                col = forward / self.resolution + self.origin
                row = right / self.resolution + self.origin
                ri, ci = int(round(row)), int(round(col))
                if 0 <= ri < self.grid_size and 0 <= ci < self.grid_size:
                    if conflict_mask[ri, ci] <= 0.5:
                        continue
                else:
                    continue

                found_actor_on_path = True
                vel = actor.get_velocity()
                v_ego_x = speed_value * cos_yaw
                v_ego_y = speed_value * sin_yaw
                rel_x = vel.x - v_ego_x
                rel_y = vel.y - v_ego_y
                closing = max(0.0, -(rel_x * dx + rel_y * dy) / dist)

                if closing > 0.1:
                    ttc_a = dist / closing
                    if ttc_a < closing_ttc:
                        closing_ttc = ttc_a
            except Exception:
                continue

        if found_actor_on_path:
            ttc = closing_ttc
        else:
            ttc = ttc_from_dist

        proxy_source_valid = occ_meta.get('source') not in ('error', 'none', 'init')
        min_dist_valid = bool(proxy_source_valid and np.isfinite(min_dist) and min_dist >= 0.0)
        ttc_valid = bool(proxy_source_valid and np.isfinite(ttc) and ttc >= 0.0)

        # ----- 5. Intervention logic disabled -----
        intervene = False

        # ----- 6. Visualisation (reduce frequency) -----
        if self.debug and self._draw_counter % max(self.debug_draw_interval, 10) == 0:
            self.visualize_in_carla(eval_traj, ego_transform, ego_actor=ego_actor, intervene=intervene)

        # ----- 9. Logging (kept for divergence analysis) -----
        self.log_data.append({
            'ego_x': ego_pos[0], 'ego_y': ego_pos[1], 'ego_yaw': ego_yaw_rad,
            'ego_speed': speed_value,
            'overlap_ratio': overlap_ratio,
            'min_occupied_distance': min_dist,
            'min_occupied_distance_valid': min_dist_valid,
            'ttc_occupied': ttc,
            'ttc_occupied_valid': ttc_valid,
            'eval_traj_held': int(eval_traj_held),
            'occupancy_source': occ_meta['source'],
            'intervention_flag': 0,                  # disabled
            'brake_command': 0.0,
            'actor_count': occ_meta.get('actor_count', 0)
        })
        
        self.latest_occ_grid = occ_grid
        self.latest_occ_meta = occ_meta
        self.latest_min_dist = min_dist
        self.latest_min_dist_valid = min_dist_valid
        self.latest_ttc = ttc
        self.latest_ttc_valid = ttc_valid

        return False, 0.0   # never intervene

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
                int(entry['min_occupied_distance_valid']),
                round(entry['ttc_occupied'], 2),
                int(entry['ttc_occupied_valid']),
                int(entry['eval_traj_held']),
                entry['occupancy_source'],
                entry['intervention_flag'],
                round(entry['brake_command'], 2),
                round(throttle_cmd, 2),
                entry['actor_count']
            ])
    
    def visualize_in_carla(self, traj, ego_transform, occ_grid=None, ego_actor=None, intervene=False):
        """Draw debug information directly in CARLA world view."""
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
            except Exception:
                continue
