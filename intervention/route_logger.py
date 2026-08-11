"""Route logging for legacy diagnostics and compact campaign measurements."""

import json
import pickle
import numpy as np
from pathlib import Path


class RouteLogger:
    SCHEMA_VERSION = 7
    LOG_PROFILES = {'full', 'compact_campaign'}
    COMPACT_METADATA_FIELDS = {
        'pco_fallback_reason': 'selection_fallback',
        'pco_mode_changed': 'selection_changed',
        'pco_selected_mode_original': 'original_mode',
        'pco_selected_mode_rescored': 'executed_mode',
        'pco_cost_selected_original': 'original_occupancy_cost',
        'pco_cost_selected_rescored': 'executed_occupancy_cost',
        'pco_valid_candidate_count': 'valid_candidate_count',
        'pco_clear_candidate_count': 'clear_candidate_count',
        'pco_occupancy_source': 'occupancy_source',
        'pco_occupancy_valid': 'occupancy_valid',
    }

    def __init__(
        self,
        log_dir,
        horizon_seconds=3.0,
        fps=20,
        planner_waypoint_times_seconds=None,
        occupancy_cell_size_m=0.5,
        run_metadata=None,
        log_profile='full',
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.horizon_seconds = horizon_seconds
        self.fps = fps
        self.horizon_frames = int(horizon_seconds * fps)
        self.planner_waypoint_times_seconds = list(
            planner_waypoint_times_seconds
            if planner_waypoint_times_seconds is not None
            else [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        )
        self.occupancy_cell_size_m = float(occupancy_cell_size_m)
        self.run_metadata = dict(run_metadata or {})
        if log_profile not in self.LOG_PROFILES:
            raise ValueError(
                f"Unsupported RouteLogger profile {log_profile!r}; "
                f"expected one of {sorted(self.LOG_PROFILES)}"
            )
        self.log_profile = log_profile
        
        # Log storage
        self.timesteps = []
        self.step = 0
        self._first_simulator_elapsed_seconds = None
        self._last_simulator_frame = None
        
        print(f"[RouteLogger] Initialized - horizon={self.horizon_seconds}s ({self.horizon_frames} frames)")

    def log_timestep(self, 
                     planner_trajs,       # (K, T, 2) - K trajectory samples
                     planner_scores,      # (K,) - trajectory scores/probabilities
                     occupancy_grid,      # (H, W) - BEV occupancy at current time
                     ego_transform,       # CARLA transform
                     ego_speed,           # float
                     collision=False,    # bool
                     collision_event=None,  # collision callback payload
                     simulator_frame=None,  # CARLA world snapshot frame
                     simulator_elapsed_seconds=None,
                     agent_timestamp=None,
                     planner_logits=None, # (K,) - pre-rescore decoder logits
                     planner_logits_masked=None,  # (K,) - post-rescore logits
                     planner_scores_masked=None,  # (K,) - post-rescore softmax
                     planner_rescore_mask=None,  # (K,) - any downstream mask
                     planner_collision_rescore_mask=None,
                     planner_detection_rescore_mask=None,
                     planner_selected_mode_index=None,
                     traj_final=None,    # (T, 2) — decoder's selected trajectory (after rescore)
                     metadata=None):     # dict
        """
        Log a single timestep of data.
        """
        if simulator_frame is not None:
            simulator_frame = int(simulator_frame)
            if (
                self._last_simulator_frame is not None
                and simulator_frame <= self._last_simulator_frame
            ):
                raise ValueError(
                    "Simulator frames must be strictly increasing: "
                    f"{simulator_frame} after {self._last_simulator_frame}"
                )
            self._last_simulator_frame = simulator_frame

        relative_timestamp = self.step / self.fps
        if simulator_elapsed_seconds is not None:
            simulator_elapsed_seconds = float(simulator_elapsed_seconds)
            if self._first_simulator_elapsed_seconds is None:
                self._first_simulator_elapsed_seconds = simulator_elapsed_seconds
            relative_timestamp = (
                simulator_elapsed_seconds - self._first_simulator_elapsed_seconds
            )

        if self.log_profile == 'compact_campaign':
            source_metadata = metadata or {}
            entry = {
                'step': self.step,
                'timestamp': float(relative_timestamp),
            }
            for source, target in self.COMPACT_METADATA_FIELDS.items():
                entry[target] = source_metadata.get(source)
            self.timesteps.append(entry)
            self.step += 1
            return

        entry = {
            'step': self.step,
            'timestamp': float(relative_timestamp),
            'planner_selected_mode_index': (
                int(planner_selected_mode_index)
                if planner_selected_mode_index is not None else None
            ),
            'traj_final': np.asarray(traj_final, dtype=np.float32) if traj_final is not None else None,
            'ego_x': ego_transform.location.x,
            'ego_y': ego_transform.location.y,
            'ego_yaw': np.deg2rad(ego_transform.rotation.yaw),
            'ego_speed': float(ego_speed),
            'collision': bool(collision),
            'collision_event': dict(collision_event or {}),
            'collision_frame': (
                int(collision_event['frame'])
                if collision_event and collision_event.get('frame') is not None
                else None
            ),
            'simulator_frame': (
                int(simulator_frame) if simulator_frame is not None else None
            ),
            'simulator_elapsed_seconds': (
                float(simulator_elapsed_seconds)
                if simulator_elapsed_seconds is not None else None
            ),
            'agent_timestamp': (
                float(agent_timestamp) if agent_timestamp is not None else None
            ),
            'metadata': metadata or {}
        }

        entry.update({
                'planner_trajs': planner_trajs,
                'planner_scores': planner_scores,
                'planner_logits': (
                    np.asarray(planner_logits, dtype=np.float32)
                    if planner_logits is not None else None
                ),
                'planner_logits_pre_rescore': (
                    np.asarray(planner_logits, dtype=np.float32)
                    if planner_logits is not None else None
                ),
                'planner_logits_post_rescore': (
                    np.asarray(planner_logits_masked, dtype=np.float32)
                    if planner_logits_masked is not None else None
                ),
                'planner_scores_post_rescore': (
                    np.asarray(planner_scores_masked, dtype=np.float32)
                    if planner_scores_masked is not None else None
                ),
                'planner_rescore_mask': (
                    np.asarray(planner_rescore_mask, dtype=bool)
                    if planner_rescore_mask is not None else None
                ),
                'planner_collision_rescore_mask': (
                    np.asarray(planner_collision_rescore_mask, dtype=bool)
                    if planner_collision_rescore_mask is not None else None
                ),
                'planner_detection_rescore_mask': (
                    np.asarray(planner_detection_rescore_mask, dtype=bool)
                    if planner_detection_rescore_mask is not None else None
                ),
                'occupancy_grid': occupancy_grid,
        })
        
        self.timesteps.append(entry)
        self.step += 1
    
    def truncate_and_save(self, route_name, max_steps):
        """Save only up to max_steps, truncating post-collision garbage frames."""
        was = len(self.timesteps)
        self.timesteps = self.timesteps[:max_steps + 1]
        self.save_route(route_name)
        print(f"[RouteLogger] Truncated {was} → {len(self.timesteps)} timesteps")

    def save_route(self, route_name):
        """Save all logged data for this route."""
        output_file = self.log_dir / f"{route_name}.pkl"
        
        occupancy_shape = None
        if self.timesteps:
            occupancy = self.timesteps[0].get('occupancy_grid')
            if occupancy is not None:
                occupancy_shape = list(np.asarray(occupancy).shape)

        payload = {
            'schema_version': self.SCHEMA_VERSION,
            'timesteps': self.timesteps,
            'config': {
                'horizon_seconds': self.horizon_seconds,
                'fps': self.fps,
                'horizon_frames': self.horizon_frames,
                'planner_waypoint_times_seconds': self.planner_waypoint_times_seconds,
                'occupancy_cell_size_m': self.occupancy_cell_size_m,
                'occupancy_grid_shape': occupancy_shape,
                'log_profile': self.log_profile,
            },
            'provenance': self.run_metadata,
        }
        temp_output = output_file.with_suffix(output_file.suffix + '.tmp')
        with open(temp_output, 'wb') as f:
            pickle.dump({
                **payload,
            }, f)
        temp_output.replace(output_file)
        
        print(f"[RouteLogger] Saved {len(self.timesteps)} timesteps to {output_file}")

        summary = {
            'route_name': route_name,
            'total_steps': int(len(self.timesteps)),
            'schema_version': self.SCHEMA_VERSION,
            'log_profile': self.log_profile,
            'provenance': self.run_metadata,
        }
        if self.log_profile == 'full':
            summary['collisions'] = int(
                sum(t['collision'] for t in self.timesteps)
            )
        summary_file = self.log_dir / f"{route_name}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
