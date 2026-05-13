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
    
    def log_timestep(self, 
                     planner_trajs,       # (K, T, 2) - K trajectory samples
                     planner_scores,      # (K,) - trajectory scores/probabilities
                     occupancy_grid,      # (H, W) - BEV occupancy at current time
                     ego_transform,       # CARLA transform
                     ego_speed,           # float
                     ttc=None,           # float or None
                     min_distance=None,  # float or None
                     collision=False,    # bool
                     near_miss=False,    # bool
                     metadata=None):     # dict
        """
        Log a single timestep of data.
        """
        
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
            'ttc': float(ttc) if ttc is not None else 999.0,
            'min_distance': float(min_distance) if min_distance is not None else 999.0,
            
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
        
        with open(output_file, 'wb') as f:
            pickle.dump({
                'timesteps': self.timesteps,
                'config': {
                    'horizon_seconds': self.horizon_seconds,
                    'fps': self.fps,
                    'horizon_frames': self.horizon_frames
                }
            }, f)
        
        print(f"[DivergenceLogger] Saved {len(self.timesteps)} timesteps to {output_file}")
        
        # Also save JSON summary for quick inspection
        summary = {
            'route_name': route_name,
            'total_steps': len(self.timesteps),
            'collisions': sum(t['collision'] for t in self.timesteps),
            'near_misses': sum(t['near_miss'] for t in self.timesteps)
        }
        
        summary_file = self.log_dir / f"{route_name}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    # Not being used anywhere?
    def reset(self):
        """Clear logged data for next route."""
        self.timesteps = []
        self.step = 0