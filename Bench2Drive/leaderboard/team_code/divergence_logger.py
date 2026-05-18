"""
Temporal Divergence Logger
Logs planner distributions, occupancy evolution, and baseline signals
for offline divergence analysis.
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path


class DivergenceLogger:
    def __init__(self, log_dir, horizon_seconds=3.0, fps=20):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.horizon_seconds = horizon_seconds
        self.fps = fps
        self.horizon_frames = int(horizon_seconds * fps)
        
        # Log storage
        self.timesteps = []
        self.step = 0
        
        print(f"[DivergenceLogger] Initialized - horizon={self.horizon_seconds}s ({self.horizon_frames} frames)")

    def _future_offsets_for_waypoints(self, num_waypoints):
        """Map planner waypoint indices to future logger frame offsets."""
        if num_waypoints <= 0:
            return np.array([], dtype=np.int64)
        return np.rint(
            np.linspace(
                self.horizon_frames / num_waypoints,
                self.horizon_frames,
                num_waypoints,
            )
        ).astype(np.int64)

    def _attach_future_occupancy(self):
        """
        Attach occupancy_future[h, H, W] to each timestep using later logged
        occupancy grids. This is intentionally done at route finalization so it
        uses no lookahead at runtime.
        """
        if not self.timesteps:
            return

        first_trajs = next(
            (t.get('planner_trajs') for t in self.timesteps if t.get('planner_trajs') is not None),
            None,
        )
        if first_trajs is None:
            return

        num_waypoints = int(np.asarray(first_trajs).shape[1])
        offsets = self._future_offsets_for_waypoints(num_waypoints)
        occupancy_grids = [np.asarray(t['occupancy_grid'], dtype=np.float16) for t in self.timesteps]

        for i, timestep in enumerate(self.timesteps):
            future = []
            valid = []
            for offset in offsets:
                j = i + int(offset)
                if j < len(occupancy_grids):
                    future.append(occupancy_grids[j])
                    valid.append(True)
                else:
                    future.append(np.zeros_like(occupancy_grids[i], dtype=np.float16))
                    valid.append(False)

            timestep['occupancy_future'] = np.stack(future, axis=0).astype(np.float16)
            timestep['occupancy_future_valid'] = np.array(valid, dtype=bool)
            timestep['occupancy_future_offsets'] = offsets.copy()
    
    def log_timestep(self, 
                     planner_trajs,       # (K, T, 2) - K trajectory samples
                     planner_scores,      # (K,) - trajectory scores/probabilities
                     occupancy_grid,      # (H, W) - BEV occupancy at current time
                     ego_transform,       # CARLA transform
                     ego_speed,           # float
                     ttc=None,           # float or None
                     min_distance=None,  # float or None
                     ttc_valid=None,      # bool or None
                     min_distance_valid=None,  # bool or None
                     ttc_rel=None,        # actor-relative TTC, float or None
                     ttc_rel_valid=None,  # bool or None
                     ttc_rel_distance=None,      # float or None
                     ttc_rel_closing_speed=None, # float or None
                     ttc_rel_actor_type='none',  # str
                     collision=False,    # bool
                     near_miss=False,    # bool
                     metadata=None):     # dict
        """
        Log a single timestep of data.
        """
        def clean_float(value):
            if value is None:
                return np.nan
            try:
                return float(value)
            except (TypeError, ValueError):
                return np.nan

        def infer_valid(value):
            value = clean_float(value)
            return bool((np.isfinite(value) or np.isposinf(value)) and value >= 0.0)

        ttc_value = clean_float(ttc)
        min_distance_value = clean_float(min_distance)
        ttc_rel_value = clean_float(ttc_rel)
        
        entry = {
            'step': self.step,
            'timestamp': self.step / self.fps,
            
            # Planner distribution
            'planner_trajs': planner_trajs,  # Keep as numpy
            'planner_scores': planner_scores,
            
            # World state
            'occupancy_grid': occupancy_grid,
            'ego_x': ego_transform.location.x,
            'ego_y': ego_transform.location.y,
            'ego_yaw': np.deg2rad(ego_transform.rotation.yaw),
            'ego_speed': float(ego_speed),
            
            # Baseline signals
            'ttc': ttc_value,
            'min_distance': min_distance_value,
            'ttc_valid': infer_valid(ttc_value) if ttc_valid is None else bool(ttc_valid),
            'min_distance_valid': (
                infer_valid(min_distance_value)
                if min_distance_valid is None else bool(min_distance_valid)
            ),
            'ttc_rel': ttc_rel_value,
            'ttc_rel_valid': infer_valid(ttc_rel_value) if ttc_rel_valid is None else bool(ttc_rel_valid),
            'ttc_rel_distance': clean_float(ttc_rel_distance),
            'ttc_rel_closing_speed': clean_float(ttc_rel_closing_speed),
            'ttc_rel_actor_type': str(ttc_rel_actor_type),
            
            # Outcome labels
            'collision': collision,
            'near_miss': near_miss,
            
            # Optional metadata
            'metadata': metadata or {}
        }
        
        self.timesteps.append(entry)
        self.step += 1
    
    def save_route(self, route_name):
        """Save all logged data for this route."""
        output_file = self.log_dir / f"{route_name}.pkl"
        self._attach_future_occupancy()
        
        with open(output_file, 'wb') as f:
            pickle.dump({
                'timesteps': self.timesteps,
                'config': {
                    'horizon_seconds': self.horizon_seconds,
                    'fps': self.fps,
                    'horizon_frames': self.horizon_frames,
                    'occupancy_future': 'per-timestep occupancy_future[h,H,W] aligned to planner waypoint h',
                }
            }, f)
        
        print(f"[DivergenceLogger] Saved {len(self.timesteps)} timesteps to {output_file}")

        summary = {
            'route_name': route_name,
            'total_steps': len(self.timesteps),
            'collisions': sum(t['collision'] for t in self.timesteps),
            'near_misses': sum(t['near_miss'] for t in self.timesteps)
        }
        summary_file = self.log_dir / f"{route_name}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
