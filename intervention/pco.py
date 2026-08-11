"""
Per-mode occupancy cost for trajectory veto.
"""

import numpy as np


GRID_H = 120
GRID_W = 120
GRID_RANGE = 30.0
CELL_SIZE = GRID_RANGE * 2 / GRID_H

VEHICLE_LENGTH_M = 4.8
VEHICLE_WIDTH_M = 2.1


def _traj_to_bev_coords(trajs):
    """Convert planner trajectories to BEV grid coords (row, col).

    Planner frame: dim0 = right, dim1 = forward (SparseDrive convention).
    BEV grid: row decreases with forward, col decreases with right (left-positive).
    The negation on dim0 (right→left) aligns with the planner BEV grid where
    col increases leftward.
    """
    col = (-trajs[..., 0] + GRID_RANGE) / CELL_SIZE
    row = (GRID_RANGE - trajs[..., 1]) / CELL_SIZE
    row_clipped = np.clip(row, 0, GRID_H - 1)
    col_clipped = np.clip(col, 0, GRID_W - 1)
    coords = np.stack([row_clipped, col_clipped], axis=-1)
    return coords


def _align_occupancy_to_planner_bev(occ):
    """Align Guardian occupancy (row=right, col=forward) to planner BEV (row=forward-inverted, col=left).

    Transform: occ.T swaps (right, forward) → (forward, right), then [::-1, ::-1]
    reverses both axes to get (forward-inverted, left-inverted).

    The 1px shift (result[1:, 1:] = aligned[:-1, :-1]) corrects an off-by-one
    from flipping an even-sized grid: the center at index 60 maps to 59 after
    [::-1], so we shift by 1 to restore alignment with _traj_to_bev_coords
    which places the origin at index 60.
    """
    aligned = occ.T[::-1, ::-1]
    result = np.zeros_like(aligned)
    result[1:, 1:] = aligned[:-1, :-1]
    return result


def _batch_footprint_hit(aligned, bev, h, L_px, W_px, overlap_thresh=0.0):
    """
    For each of K modes: does a vehicle-sized footprint at waypoint h overlap occupied cells?

    Steps:
      1. Compute yaw per mode from trajectory direction (next waypoint, or prev if last).
         At waypoint 0 the vehicle heading is the ego current heading (forward),
         not the planned initial movement direction.
      2. Build a dense sampling grid covering the L×W vehicle footprint in BEV-pixel coords.
      3. Rotate each sample point by the mode's yaw, then offset to the waypoint position.
      4. Sample aligned occupancy at those points.
      5. If any sample hits an occupied cell (value > 0.5), the footprint overlaps an obstacle.
         With overlap_thresh > 0, require that fraction of samples exceeds the threshold.

    Yaw convention: atan2(col_diff, -row_diff), where delta = (row_diff, col_diff).
    In BEV coords, row decreases with forward and col decreases with right.
    Straight-ahead forward (col_diff=0, row_diff<0) → yaw = 0;
    pure rightward (col_diff<0, row_diff=0) → yaw = -π/2.
    At waypoint 0 the heading is overridden to yaw = 0 (ego current heading).
    """
    K = bev.shape[0]

    # Yaw per mode from trajectory direction
    # At waypoint 0 the vehicle still has its current heading (forward in ego
    # frame, which maps to yaw=0 in the footprint rotation convention), not
    # the planned initial movement direction 0→1.
    if h == 0:
        yaws = np.zeros(K)
    elif h < bev.shape[1] - 1:
        delta = bev[:, h + 1, :] - bev[:, h, :]
        yaws = np.arctan2(delta[:, 1], -delta[:, 0])
        yaws[np.linalg.norm(delta, axis=-1) < 1e-6] = 0.0
    else:
        delta = bev[:, h, :] - bev[:, h - 1, :]
        yaws = np.arctan2(delta[:, 1], -delta[:, 0])
        yaws[np.linalg.norm(delta, axis=-1) < 1e-6] = 0.0

    # Sampling grid covering the footprint
    n_along = max(2, int(np.ceil(L_px * 2)) + 1)    # ~20 samples along vehicle length
    n_lat   = max(2, int(np.ceil(W_px * 2)) + 1)    # ~9  samples along vehicle width
    f_off = np.linspace(-L_px / 2, L_px / 2, n_along)
    l_off = np.linspace(-W_px / 2, W_px / 2, n_lat)
    FF, LL = np.meshgrid(f_off, l_off, indexing='ij')
    samples = np.stack([FF.ravel(), LL.ravel()], axis=-1)  # (N, 2) where N = n_along * n_lat

    # Rotate samples by yaw and offset to waypoint
    # col (right) offset: forward * sin + lateral * cos
    col_off = samples[:, 0][None, :] * np.sin(yaws)[:, None] \
            + samples[:, 1][None, :] * np.cos(yaws)[:, None]
    # row (forward) offset: -forward * cos + lateral * sin
    row_off = -samples[:, 0][None, :] * np.cos(yaws)[:, None] \
             + samples[:, 1][None, :] * np.sin(yaws)[:, None]

    rows = bev[:, h, 0]
    cols = bev[:, h, 1]
    sample_rows = np.round(rows[:, None] + row_off).astype(np.int32)
    sample_cols = np.round(cols[:, None] + col_off).astype(np.int32)

    # Sample occupancy, clamping OOB to 0
    H, W = aligned.shape[:2]
    in_bounds = (sample_rows >= 0) & (sample_rows < H) \
              & (sample_cols >= 0) & (sample_cols < W)
    sample_rows = np.clip(sample_rows, 0, H - 1)
    sample_cols = np.clip(sample_cols, 0, W - 1)
    occ_vals = np.where(in_bounds, aligned[sample_rows, sample_cols], 0.0)

    # Collapse N samples → one bool per mode
    if overlap_thresh > 0.0:
        return occ_vals.mean(axis=1) > overlap_thresh
    
    return occ_vals.max(axis=1) > 0.5


def per_mode_occupancy_cost(
    trajs,
    occ_grid,
    footprint_L=4.8,
    footprint_W=2.1,
    overlap_thresh=0.0,
    cost_mode='binary',
):
    """Online per-mode occupancy cost for trajectory rescoring.

    Unlike planner_conditioned_occupancy, this returns a per-mode cost vector
    (not an aggregated risk score) and does not require timestep dicts or
    mode weights. The agent calls this directly after Guardian evaluate().

    Args:
        trajs: (K, T, 2) float64 — trajectory waypoints [right, forward].
               K up to 1024 (all modes), T=6 (3.0s horizon at 2 Hz).
        occ_grid: (120, 120) float32 or (T, 120, 120) float32 — Guardian occupancy
                  (row=right, col=forward). If 3D, each waypoint h checks against
                  occ_grid[h] (temporal occupancy). If 2D, all waypoints check the
                  same grid (current-frame occupancy).
        footprint_L: vehicle length in meters. Default 4.8.
        footprint_W: vehicle width in meters. Default 2.1.
        overlap_thresh: if 0.0, any occupied cell in footprint → hit.
                        if >0.0, fraction of footprint cells must exceed threshold.
        cost_mode: 'binary' → c_k = 1[hit] (for binary veto/progress-preserving exclusion).
                   'urgency' → (deprecated) c_k = 1[hit] * (1 + (T - tau_k) / T).

    Returns:
        costs: (K,) float64 — per-mode occupancy cost. NaN if mode is invalid
               (any waypoint out of bounds).
        first_hits: (K,) int64 — first-hit waypoint index (T if no hit).
        valid_modes: (K,) bool — True for modes with all waypoints in-bounds.
        diagnostics: dict with 'valid_mode_fraction' (float), 'n_valid' (int).
    """
    trajs = np.asarray(trajs, dtype=np.float64)
    K, T, _ = trajs.shape

    bev = _traj_to_bev_coords(trajs)

    L_px = footprint_L / CELL_SIZE
    W_px = footprint_W / CELL_SIZE

    # Identify valid (in-bounds) modes
    col_raw = (-trajs[..., 0] + GRID_RANGE) / CELL_SIZE
    row_raw = (GRID_RANGE - trajs[..., 1]) / CELL_SIZE
    valid_modes = ~np.any(
        (row_raw < 0) | (row_raw >= GRID_H) | (col_raw < 0) | (col_raw >= GRID_W),
        axis=1,
    )
    valid_idx = np.where(valid_modes)[0]

    # Prepare per-waypoint occupancy grids
    occ_np = np.asarray(occ_grid, dtype=np.float64)
    if occ_np.ndim == 3:
        if occ_np.shape[0] != T:
            raise ValueError(
                f"3D occupancy grid has shape[0]={occ_np.shape[0]} but T={T}. "
                "When passing a 3D occupancy grid, the first dimension must "
                "match the number of trajectory waypoints."
            )
        # Temporal occupancy: one grid per waypoint
        occ_list = [_align_occupancy_to_planner_bev(occ_np[h]) for h in range(T)]
    else:
        # Current-frame occupancy: same grid for all waypoints
        aligned = _align_occupancy_to_planner_bev(occ_np)
        occ_list = [aligned] * T

    first_hits = np.full(K, T, dtype=np.int64)
    costs = np.full(K, np.nan, dtype=np.float64)

    for h in range(T):
        if valid_idx.size == 0:
            break
        if occ_list[h] is None:
            continue
        hit = _batch_footprint_hit(occ_list[h], bev[valid_idx], h, L_px, W_px, overlap_thresh)
        first_hit_this = np.where(hit & (first_hits[valid_idx] == T), h, first_hits[valid_idx])
        first_hits[valid_idx] = first_hit_this

    # Compute cost for valid modes
    if len(valid_idx) > 0:
        fh_valid = first_hits[valid_idx]
        hit_mask = fh_valid < T
        if cost_mode == 'binary':
            costs[valid_idx] = hit_mask.astype(np.float64)
        else:
            costs[valid_idx] = np.where(
                hit_mask,
                1.0 + (T - fh_valid.astype(np.float64)) / T,
                0.0,
            )

    n_valid = int(valid_modes.sum())
    diagnostics = {
        'valid_mode_fraction': float(n_valid) / K if K > 0 else 0.0,
        'n_valid': n_valid,
    }

    return costs, first_hits, valid_modes, diagnostics


def select_pco_candidate(
    scores,
    costs,
    candidate_idx,
    planner_trajs,
    original_mode_index=None,
    selection_policy='binary_veto',
    progress_tolerance_m=2.0,
):
    """Pure helper: select the best trajectory index among safe candidates.

    Args:
        scores: (K,) float64 — post-rescore sigmoid scores.
        costs: (K,) float64 — per-mode occupancy cost (0 = safe, 1 = unsafe, NaN = invalid).
        candidate_idx: (N,) int — indices of valid, unmasked candidate modes.
        planner_trajs: (K, T, 2) — trajectory waypoints [right, forward].
        original_mode_index: int — the planner's originally selected mode index.
        selection_policy: 'binary_veto' | 'progress_preserving_veto'.
            'binary_veto': argmax score among zero-cost candidates.
            'progress_preserving_veto': among zero-cost candidates that preserve
            forward progress (p_i >= p_b - delta), select the trajectory closest
            to the original (least-invasive), score as tie-breaker.
        progress_tolerance_m: float — max progress sacrifice vs original (default 2.0m).

    Returns:
        best_index: int or None (no valid/eligible candidate).
        fallback_reason: str — 'filtered_viable', 'filtered_safe_only',
            'all_unsafe', 'no_valid_candidate'.
        diagnostics: dict with per-frame selection metadata.
    """
    scores = np.asarray(scores, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    planner_trajs = np.asarray(planner_trajs, dtype=np.float64)
    candidate_idx = np.asarray(candidate_idx, dtype=np.int64)

    diag = {
        'progress_original': -1.0,
        'progress_selected': -1.0,
        'progress_max_safe': -1.0,
        'progress_threshold': -1.0,
        'progress_eligible_count': -1,
        'progress_filter_active': False,
        'traj_distance_selected': -1.0,
    }

    if original_mode_index is not None and 0 <= original_mode_index < len(planner_trajs):
        diag['progress_original'] = float(planner_trajs[original_mode_index, -1, 1])

    if len(candidate_idx) == 0:
        return None, 'no_valid_candidate', diag

    zero_cost = costs[candidate_idx] == 0.0
    zero_idx = candidate_idx[zero_cost]

    if len(zero_idx) == 0:
        return None, 'all_unsafe', diag

    if selection_policy == 'binary_veto':
        best = zero_idx[int(np.argmax(scores[zero_idx]))]
        diag['progress_selected'] = float(planner_trajs[best, -1, 1])
        return int(best), 'filtered', diag

    if selection_policy == 'progress_preserving_veto':
        progress = planner_trajs[zero_idx, -1, 1]
        max_safe_progress = float(np.max(progress))
        diag['progress_max_safe'] = max_safe_progress
        baseline_progress = diag['progress_original']
        threshold = baseline_progress - progress_tolerance_m
        diag['progress_threshold'] = threshold
        diag['progress_filter_active'] = True

        eligible_mask = progress >= threshold
        eligible_idx = zero_idx[eligible_mask]
        diag['progress_eligible_count'] = int(eligible_mask.sum())

        if len(eligible_idx) > 0:
            # Tier 1: safe + viable -> least-invasive (closest to original)
            if original_mode_index is not None and 0 <= original_mode_index < len(planner_trajs):
                traj_dists = np.mean(np.linalg.norm(
                    planner_trajs[eligible_idx] - planner_trajs[original_mode_index],
                    axis=-1,
                ), axis=-1)
                best_local = int(np.lexsort((-scores[eligible_idx], traj_dists))[0])
                diag['traj_distance_selected'] = float(traj_dists[best_local])
            else:
                best_local = int(np.argmax(scores[eligible_idx]))
            best = eligible_idx[best_local]
            diag['progress_selected'] = float(progress[eligible_mask][best_local])
            return int(best), 'filtered_viable', diag
        else:
            # Tier 2: safe exists but none progress-preserving -> binary veto (V1 fallback)
            best = zero_idx[int(np.argmax(scores[zero_idx]))]
            diag['progress_selected'] = float(planner_trajs[best, -1, 1])
            return int(best), 'filtered_safe_only', diag

    best = zero_idx[int(np.argmax(scores[zero_idx]))]
    diag['progress_selected'] = float(planner_trajs[best, -1, 1])
    return int(best), 'filtered', diag
