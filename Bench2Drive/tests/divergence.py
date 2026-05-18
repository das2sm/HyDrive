"""
Temporal planner-world conflict measurement.

The publication-facing predictive signal is temporal_conflict_score:
weighted planner occupancy over the future horizon compared against matched
future world occupancy slices. Raw Jensen-Shannon divergence is also returned
for auditability, but collision risk is expected to rise when planner futures
overlap world occupancy futures, i.e. when JS divergence is low.
"""

import numpy as np


# BEV grid parameters (must match agent: 120x120, 60m range → 0.5m/cell)
GRID_H = 120
GRID_W = 120
GRID_RANGE = 30.0  # meters, symmetric around ego
CELL_SIZE = GRID_RANGE * 2 / GRID_H  # 0.5 m/cell

# Trajectory
TRAJ_WAYPOINTS = 6   # T waypoints per trajectory
JS_MAX = np.log(2.0)


def _traj_to_bev_coords(trajs):
    """
    Convert ego-frame trajectories (K, T, 2) to BEV pixel coords.
    Ego frame: x=left, y=forward. BEV: row=forward (inverted), col=left.
    Returns (K, T, 2) pixel coords, clipped to [0, GRID_H) x [0, GRID_W).
    """
    # Compute continuous coordinates
    col = (trajs[..., 0] + GRID_RANGE) / CELL_SIZE
    row = (GRID_RANGE - trajs[..., 1]) / CELL_SIZE
    
    # Clip to grid bounds [0, GRID_H) x [0, GRID_W)
    row_clipped = np.clip(row, 0, GRID_H - 1)
    col_clipped = np.clip(col, 0, GRID_W - 1)
        
    coords = np.stack([row_clipped, col_clipped], axis=-1)
    return coords


def _normalise_scores(planner_scores):
    scores = np.asarray(planner_scores, dtype=np.float64).reshape(-1)
    total = scores.sum()
    if total > 1e-12:
        return scores / total
    return np.ones_like(scores, dtype=np.float64) / max(len(scores), 1)


def _add_gaussian(probability_grid, row_center, col_center, weight, sigma):
    kernel_radius = int(np.ceil(3 * sigma))
    kernel_offsets_y, kernel_offsets_x = np.mgrid[
        -kernel_radius : kernel_radius + 1,
        -kernel_radius : kernel_radius + 1,
    ]
    gaussian_kernel = np.exp(
        -(kernel_offsets_y**2 + kernel_offsets_x**2) / (2 * sigma**2)
    )

    row_pixel = int(round(row_center))
    col_pixel = int(round(col_center))

    row_start_full = row_pixel - kernel_radius
    row_end_full = row_pixel + kernel_radius + 1
    col_start_full = col_pixel - kernel_radius
    col_end_full = col_pixel + kernel_radius + 1

    grid_row_start = max(row_start_full, 0)
    grid_row_end = min(row_end_full, GRID_H)
    grid_col_start = max(col_start_full, 0)
    grid_col_end = min(col_end_full, GRID_W)

    kernel_row_start = max(0, -row_start_full)
    kernel_row_end = kernel_row_start + (grid_row_end - grid_row_start)
    kernel_col_start = max(0, -col_start_full)
    kernel_col_end = kernel_col_start + (grid_col_end - grid_col_start)

    if grid_row_end > grid_row_start and grid_col_end > grid_col_start:
        probability_grid[
            grid_row_start:grid_row_end,
            grid_col_start:grid_col_end,
        ] += weight * gaussian_kernel[
            kernel_row_start:kernel_row_end,
            kernel_col_start:kernel_col_end,
        ]


def _normalise_grid(grid):
    grid = np.asarray(grid, dtype=np.float64)
    total = grid.sum()
    if total > 1e-12:
        return grid / total
    return grid


def rasterize_planner(planner_trajs, planner_scores, sigma=3.0):
    """
    Rasterize weighted planner trajectories onto a BEV probability grid.

    Args:
        planner_trajs: (K, T, 2) ego-frame waypoints [left, forward]
        planner_scores: (K,) probability weights (sum to 1)
        sigma: Gaussian spread in pixels

    Returns:
        probability_grid: (H, W) normalised probability grid, sums to 1.
    """
    planner_trajs = np.asarray(planner_trajs, dtype=np.float64)
    if planner_trajs.ndim == 2:
        planner_trajs = planner_trajs[None, ...]
    planner_scores = _normalise_scores(planner_scores)

    probability_grid = np.zeros((GRID_H, GRID_W), dtype=np.float64)
    waypoint_coords = _traj_to_bev_coords(planner_trajs)  # (K, T, 2)
    num_waypoints = waypoint_coords.shape[1]

    for traj_idx, trajectory_weight in enumerate(planner_scores):
        for waypoint_idx in range(num_waypoints):
            row_center, col_center = waypoint_coords[traj_idx, waypoint_idx]
            _add_gaussian(probability_grid, row_center, col_center, trajectory_weight, sigma)

    # Normalise to a valid probability distribution
    return _normalise_grid(probability_grid)


def rasterize_planner_temporal(planner_trajs, planner_scores, sigma=3.0):
    """
    Rasterize each planner waypoint timestep separately.

    Returns:
        probability_grids: (T, H, W), one normalized BEV distribution per
        future waypoint.
    """
    planner_trajs = np.asarray(planner_trajs, dtype=np.float64)
    if planner_trajs.ndim == 2:
        planner_trajs = planner_trajs[None, ...]
    planner_scores = _normalise_scores(planner_scores)

    waypoint_coords = _traj_to_bev_coords(planner_trajs)
    num_waypoints = waypoint_coords.shape[1]
    probability_grids = np.zeros((num_waypoints, GRID_H, GRID_W), dtype=np.float64)

    for waypoint_idx in range(num_waypoints):
        grid = probability_grids[waypoint_idx]
        for traj_idx, trajectory_weight in enumerate(planner_scores):
            row_center, col_center = waypoint_coords[traj_idx, waypoint_idx]
            _add_gaussian(grid, row_center, col_center, trajectory_weight, sigma)
        probability_grids[waypoint_idx] = _normalise_grid(grid)

    return probability_grids


def js_divergence(p, q, eps=1e-10):
    """
    Jensen-Shannon divergence between two probability distributions.
    p, q: flat arrays, will be normalized.
    Returns scalar in [0, log(2)].
    """
    p = p.ravel().astype(np.float64)
    q = q.ravel().astype(np.float64)

    p_sum = p.sum()
    q_sum = q.sum()

    if p_sum < eps or q_sum < eps:
        # If one is empty, they are maximally different (disjoint)
        return np.log(2.0)
    
    p = p / p_sum
    q = q / q_sum
    m = 0.5 * (p + q)

    mask_p = p > eps
    mask_q = q > eps
    
    kl_pm = np.sum(p[mask_p] * np.log(p[mask_p] / m[mask_p]))
    kl_qm = np.sum(q[mask_q] * np.log(q[mask_q] / m[mask_q]))
    
    jsd = 0.5 * (kl_pm + kl_qm)
    return np.clip(jsd, 0.0, JS_MAX)


def js_conflict_score(js_value):
    """
    Convert JS divergence into an overlap/conflict score in [0, 1].

    High values mean planner probability mass and world occupancy mass are
    spatially aligned at the matched future slice.
    """
    if not np.isfinite(js_value):
        return np.nan
    return float(np.clip(1.0 - js_value / JS_MAX, 0.0, 1.0))


def planner_spread_entropy(planner_trajs, planner_scores):
    """
    Fallback divergence signal when occupancy is unavailable.
    Measures weighted variance of trajectory endpoints.
    """
    if np.sum(planner_scores) < 1e-12:
        return 0.0
    endpoints = planner_trajs[:, -1, :]  # (K, 2)
    mean = np.average(endpoints, weights=planner_scores, axis=0)
    diffs = endpoints - mean  # (K, 2)
    weighted_var = np.average(np.sum(diffs**2, axis=1), weights=planner_scores)
    return float(weighted_var)


def build_future_occupancy_windows(timesteps, num_waypoints=None, horizon_frames=None):
    """
    Offline helper for logs that do not already contain occupancy_future.

    Returns an array with shape (N, T, H, W), where T matches planner waypoint
    count. The last frames whose future window extends past the route end are
    zero-filled and marked invalid in the returned validity mask.
    """
    if not timesteps:
        return (
            np.zeros((0, 0, GRID_H, GRID_W), dtype=np.float16),
            np.zeros((0, 0), dtype=bool),
            np.array([], dtype=np.int64),
        )

    if num_waypoints is None:
        first_trajs = next((t.get('planner_trajs') for t in timesteps if 'planner_trajs' in t), None)
        num_waypoints = int(np.asarray(first_trajs).shape[1]) if first_trajs is not None else TRAJ_WAYPOINTS
    if horizon_frames is None:
        horizon_frames = num_waypoints

    offsets = np.rint(
        np.linspace(horizon_frames / num_waypoints, horizon_frames, num_waypoints)
    ).astype(np.int64)
    occupancy_grids = [np.asarray(t['occupancy_grid'], dtype=np.float16) for t in timesteps]
    windows = np.zeros((len(timesteps), num_waypoints, GRID_H, GRID_W), dtype=np.float16)
    valid = np.zeros((len(timesteps), num_waypoints), dtype=bool)

    for i in range(len(timesteps)):
        for h, offset in enumerate(offsets):
            j = i + int(offset)
            if j < len(timesteps):
                windows[i, h] = occupancy_grids[j]
                valid[i, h] = True

    return windows, valid, offsets


def _align_occupancy_to_planner_bev(occ):
    """
    Align Guardian occupancy grid (row=right, col=forward) 
    to planner BEV (row=ahead_inverted, col=left).
    Transformation: Transpose then flip both axes.
    """
    return occ.T[::-1, ::-1]


def check_coordinate_alignment(planner_trajs, planner_scores, ego_forward_m=10.0):
    """
    Sanity check: place a synthetic obstacle directly ahead and verify that
    the planner grid and aligned occupancy grid both peak in the same region.
    """
    res = GRID_RANGE * 2 / GRID_H 
    origin = GRID_H // 2
    
    # Guardian Frame: row=right, col=forward
    ahead_col = int(ego_forward_m / res + origin)
    ahead_row = origin

    synthetic_occ = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    synthetic_occ[ahead_row, ahead_col] = 1.0

    # Align to planner BEV
    occ_aligned = _align_occupancy_to_planner_bev(synthetic_occ)
    occ_peak = np.unravel_index(occ_aligned.argmax(), occ_aligned.shape)

    # Planner grid peak
    p_grid = rasterize_planner(planner_trajs, planner_scores)
    planner_peak = np.unravel_index(p_grid.argmax(), p_grid.shape)

    expected_row = int((GRID_RANGE - ego_forward_m) / res)  
    expected_col = origin  

    row_diff = abs(int(occ_peak[0]) - expected_row)
    col_diff = abs(int(occ_peak[1]) - expected_col)

    return {
        'occ_peak_aligned': occ_peak,
        'expected': (expected_row, expected_col),
        'row_diff': row_diff,
        'col_diff': col_diff,
        'aligned': row_diff <= 2 and col_diff <= 2,
    }


def compute_divergence_series(timesteps, use_occupancy=True, temporal_window=5):
    """
    Compute temporal planner-world conflict and raw JS divergence for all steps.

    The predictive score is temporal_conflict_raw/smooth. Raw JS divergence is
    returned separately as temporal_js_raw/smooth.
    """
    N = len(timesteps)
    temporal_conflict_raw = np.full(N, np.nan)
    temporal_js_raw = np.full(N, np.nan)
    temporal_valid_count = np.zeros(N, dtype=np.int64)
    planner_spread = np.zeros(N)
    has_occupancy = np.zeros(N, dtype=bool)

    for i, t in enumerate(timesteps):
        trajs = t['planner_trajs']    
        scores = t['planner_scores']  
        occ_current = t['occupancy_grid']

        spread = planner_spread_entropy(trajs, scores)
        planner_spread[i] = spread

        if 'occupancy_future' in t:
            occupancy_future = np.asarray(t['occupancy_future'])
            occupancy_valid = np.asarray(
                t.get('occupancy_future_valid', np.ones(len(occupancy_future), dtype=bool)),
                dtype=bool,
            )
        else:
            occupancy_future = np.asarray(occ_current)[None, ...]
            occupancy_valid = np.array([True], dtype=bool)

        occ_has_signal = np.array([occ.max() > 1e-6 for occ in occupancy_future], dtype=bool)
        valid_slices = occupancy_valid & occ_has_signal
        has_occupancy[i] = valid_slices.any()

        if use_occupancy and valid_slices.any():
            planner_future = rasterize_planner_temporal(trajs, scores)
            num_slices = min(len(planner_future), len(occupancy_future))
            js_values = []
            conflict_values = []
            for h in range(num_slices):
                if not valid_slices[h]:
                    continue
                p_grid = planner_future[h]
                q_grid = _align_occupancy_to_planner_bev(occupancy_future[h]).astype(np.float64)
                q_grid = _normalise_grid(q_grid)

                js = js_divergence(p_grid, q_grid)
                js_values.append(js)
                conflict_values.append(js_conflict_score(js))

            temporal_valid_count[i] = len(js_values)
            if js_values:
                temporal_js_raw[i] = float(np.mean(js_values))
                temporal_conflict_raw[i] = float(np.mean(conflict_values))

    # Causal rolling mean
    temporal_conflict_smooth = np.full(N, np.nan)
    temporal_js_smooth = np.full(N, np.nan)
    for i in range(N):
        if not np.isfinite(temporal_conflict_raw[i]):
            continue
        start = max(0, i - temporal_window + 1)
        conflict_window = temporal_conflict_raw[start:i+1]
        conflict_valid = np.isfinite(conflict_window)
        if conflict_valid.any():
            temporal_conflict_smooth[i] = conflict_window[conflict_valid].mean()
        js_window = temporal_js_raw[start:i+1]
        js_valid = np.isfinite(js_window)
        if js_valid.any():
            temporal_js_smooth[i] = js_window[js_valid].mean()

    return {
        'temporal_conflict_raw': temporal_conflict_raw,
        'temporal_conflict_smooth': temporal_conflict_smooth,
        'temporal_js_raw': temporal_js_raw,
        'temporal_js_smooth': temporal_js_smooth,
        'temporal_valid_count': temporal_valid_count,
        # Backward-compatible aliases used by existing analysis code.
        'divergence_raw': temporal_conflict_raw,
        'divergence_smooth': temporal_conflict_smooth,
        'planner_spread': planner_spread,
        'has_occupancy': has_occupancy,
    }
