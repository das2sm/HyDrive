"""Default-off, outcome-inert capture for the paper's qualitative figure."""

import json
import os
import pathlib
import tempfile

import numpy as np


SCHEMA_VERSION = 3
CURRENT_HORIZONS = np.zeros(6, dtype=np.float32)
TEMPORAL_HORIZONS = np.arange(0.5, 3.01, 0.5, dtype=np.float32)
CAMERA_IDS = (
    'CAM_FRONT_LEFT',
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',
    'CAM_BACK',
    'CAM_BACK_RIGHT',
)


def validate_capture_configuration(raw_path, strict_campaign):
    """Return the normalized capture path or None when capture is disabled."""
    raw_path = (raw_path or '').strip()
    if not raw_path:
        return None
    if strict_campaign:
        raise RuntimeError(
            "HYDRIVE_QUAL_CAPTURE_PATH is forbidden when "
            "HYDRIVE_STRICT_CAMPAIGN=1"
        )
    path = pathlib.Path(raw_path).expanduser()
    if path.suffix.lower() != '.npz':
        path = path / 'qualitative_snapshot.npz'
    return path


class QualitativeCapture(object):
    """Keep the pre-specified best same-state frame and write it atomically."""

    def __init__(self, output_path):
        self.output_path = pathlib.Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists():
            raise FileExistsError(
                "Qualitative capture already exists; move or delete it before "
                "starting a new visualization run: {}".format(self.output_path)
            )
        self._best_has_original_disagreement = False
        self._best_mask_disagreement = -1
        self._best_frame = None
        self._payload = None
        self._front_target_frames = None

    @property
    def has_original_disagreement(self):
        return self._best_has_original_disagreement

    def consider(
        self,
        current_grids,
        temporal_grids,
        proposals,
        scores,
        planner_mask,
        original_mode,
        current_costs,
        temporal_costs,
        current_valid,
        temporal_valid,
        current_meta,
        temporal_meta,
        vehicle_length,
        vehicle_width,
        route,
        traffic_seed,
        simulator_frame,
        camera_images,
        fixed_delta_seconds,
    ):
        """Save this frame if it wins the fixed qualitative selection rule."""
        current_grids = np.asarray(current_grids)
        temporal_grids = np.asarray(temporal_grids)
        proposals = np.asarray(proposals)
        scores = np.asarray(scores).reshape(-1)
        planner_mask = np.asarray(planner_mask, dtype=bool).reshape(-1)
        current_costs = np.asarray(current_costs).reshape(-1)
        temporal_costs = np.asarray(temporal_costs).reshape(-1)
        current_valid = np.asarray(current_valid, dtype=bool).reshape(-1)
        temporal_valid = np.asarray(temporal_valid, dtype=bool).reshape(-1)
        camera_images = np.asarray(camera_images)

        if current_grids.shape != (6, 120, 120):
            raise ValueError("Current grids must have shape (6, 120, 120)")
        if temporal_grids.shape != (6, 120, 120):
            raise ValueError("Temporal grids must have shape (6, 120, 120)")
        if not np.array_equal(
            current_grids, np.repeat(current_grids[0:1], 6, axis=0)
        ):
            raise ValueError("Current-frame grids are not identical across horizons")
        if proposals.shape != (1024, 6, 2):
            raise ValueError("Proposals must have shape (1024, 6, 2)")
        if (
            camera_images.ndim != 4
            or camera_images.shape[0] != len(CAMERA_IDS)
            or camera_images.shape[-1] != 3
        ):
            raise ValueError(
                "Camera images must have shape (6, height, width, 3)"
            )
        if camera_images.dtype != np.uint8:
            raise ValueError("Camera images must use uint8 RGB values")
        if not all(array.shape == (1024,) for array in (
            scores, planner_mask, current_costs, temporal_costs,
            current_valid, temporal_valid,
        )):
            raise ValueError("Per-proposal arrays must have length 1024")
        original_mode = int(original_mode)
        if not 0 <= original_mode < 1024:
            raise ValueError("Original proposal index is out of range")

        current_clear = (
            current_valid & ~planner_mask & np.isfinite(current_costs)
            & (current_costs == 0.0)
        )
        temporal_clear = (
            temporal_valid & ~planner_mask & np.isfinite(temporal_costs)
            & (temporal_costs == 0.0)
        )
        current_original_clear = bool(current_clear[original_mode])
        temporal_original_clear = bool(temporal_clear[original_mode])
        original_disagreement = (
            current_original_clear != temporal_original_clear
        )
        mask_disagreement = int(np.count_nonzero(
            current_clear != temporal_clear
        ))
        frame = int(simulator_frame)
        fixed_delta_seconds = float(fixed_delta_seconds)
        if not np.isfinite(fixed_delta_seconds) or fixed_delta_seconds <= 0.0:
            raise ValueError("fixed_delta_seconds must be positive and finite")

        if self._best_has_original_disagreement:
            replace = original_disagreement and frame < self._best_frame
        elif original_disagreement:
            replace = True
        else:
            replace = (
                mask_disagreement > self._best_mask_disagreement
                or (
                    mask_disagreement == self._best_mask_disagreement
                    and (
                        self._best_frame is None
                        or frame < self._best_frame
                    )
                )
            )
        if not replace:
            return False

        criterion = (
            'earliest_original_classification_disagreement'
            if original_disagreement
            else 'largest_clear_mask_disagreement'
        )
        metadata = {
            'schema_version': SCHEMA_VERSION,
            'development_only': True,
            'outcome_eligible': False,
            'selection_criterion': criterion,
            'route': str(route),
            'traffic_seed': int(traffic_seed),
            'simulator_frame': frame,
            'fixed_delta_seconds': fixed_delta_seconds,
            'original_mode': original_mode,
            'current_original_clear': current_original_clear,
            'temporal_original_clear': temporal_original_clear,
            'clear_mask_disagreement_count': mask_disagreement,
            'current_actor_counts': dict(current_meta or {}),
            'temporal_actor_counts': dict(temporal_meta or {}),
        }
        payload = {
            'metadata_json': np.asarray(
                json.dumps(metadata, sort_keys=True), dtype=np.str_
            ),
            'current_grids': current_grids.astype(np.uint8),
            'temporal_grids': temporal_grids.astype(np.uint8),
            'proposals': proposals.astype(np.float32),
            'scores': scores.astype(np.float32),
            'planner_mask': planner_mask,
            'original_mode': np.asarray(original_mode, dtype=np.int64),
            'current_costs': current_costs.astype(np.float32),
            'temporal_costs': temporal_costs.astype(np.float32),
            'current_valid': current_valid,
            'temporal_valid': temporal_valid,
            'current_clear': current_clear,
            'temporal_clear': temporal_clear,
            'current_horizons': CURRENT_HORIZONS.copy(),
            'temporal_horizons': TEMPORAL_HORIZONS.copy(),
            'vehicle_length': np.asarray(vehicle_length, dtype=np.float32),
            'vehicle_width': np.asarray(vehicle_width, dtype=np.float32),
            'camera_ids': np.asarray(CAMERA_IDS, dtype=np.str_),
            'camera_images': camera_images,
        }
        if original_disagreement:
            frame_offsets = np.rint(
                TEMPORAL_HORIZONS / fixed_delta_seconds
            ).astype(np.int64)
            if not np.allclose(
                frame_offsets * fixed_delta_seconds,
                TEMPORAL_HORIZONS,
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError(
                    "Qualitative horizons are not aligned with the fixed step"
                )
            self._front_target_frames = frame + frame_offsets
            payload.update({
                'front_camera_horizons': TEMPORAL_HORIZONS.copy(),
                'front_camera_target_frames': self._front_target_frames.copy(),
                'front_camera_frames': np.full(6, -1, dtype=np.int64),
                'front_camera_images': np.zeros(
                    (6,) + tuple(camera_images.shape[1:]), dtype=np.uint8
                ),
            })
            metadata['front_camera_sequence_complete'] = False
            payload['metadata_json'] = np.asarray(
                json.dumps(metadata, sort_keys=True), dtype=np.str_
            )
        else:
            self._front_target_frames = None
        self._payload = payload
        self._atomic_save(payload)
        self._best_has_original_disagreement = original_disagreement
        self._best_mask_disagreement = mask_disagreement
        self._best_frame = frame
        return True

    def consider_future_front(self, simulator_frame, camera_image):
        """Record an exact future front-camera frame for the chosen snapshot."""
        if self._payload is None or self._front_target_frames is None:
            return False
        frame = int(simulator_frame)
        matches = np.flatnonzero(self._front_target_frames == frame)
        if matches.size == 0:
            return False

        camera_image = np.asarray(camera_image)
        expected_shape = self._payload['front_camera_images'].shape[1:]
        if camera_image.shape != expected_shape or camera_image.dtype != np.uint8:
            raise ValueError(
                "Future front-camera image must match the snapshot RGB shape"
            )

        index = int(matches[0])
        self._payload['front_camera_images'][index] = camera_image
        self._payload['front_camera_frames'][index] = frame
        complete = bool(np.all(self._payload['front_camera_frames'] >= 0))
        metadata = json.loads(str(self._payload['metadata_json']))
        metadata['front_camera_sequence_complete'] = complete
        self._payload['metadata_json'] = np.asarray(
            json.dumps(metadata, sort_keys=True), dtype=np.str_
        )
        self._atomic_save(self._payload)
        return True

    def _atomic_save(self, payload):
        descriptor, temp_name = tempfile.mkstemp(
            prefix=self.output_path.stem + '.',
            suffix='.tmp.npz',
            dir=str(self.output_path.parent),
        )
        os.close(descriptor)
        temp_path = pathlib.Path(temp_name)
        try:
            np.savez_compressed(str(temp_path), **payload)
            temp_path.replace(self.output_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
