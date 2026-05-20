"""
Temporal planner-world conflict measurement.

The publication-facing predictive signal is temporal_conflict_score:
weighted planner occupancy over the future horizon compared against matched
future world occupancy slices. Raw Jensen-Shannon divergence is also returned
for auditability, but collision risk is expected to rise when planner futures
overlap world occupancy futures, i.e. when JS divergence is low.
"""

import cv2
import numpy as np


# BEV grid parameters (must match agent: 120x120, 60m range → 0.5m/cell)
GRID_H = 120
GRID_W = 120
GRID_RANGE = 30.0  # meters, symmetric around ego
CELL_SIZE = GRID_RANGE * 2 / GRID_H  # 0.5 m/cell

# Ego vehicle footprint used for planner footprint rasterization.
VEHICLE_LENGTH_M = 4.8
VEHICLE_WIDTH_M = 2.1
VEHICLE_LONG_MARGIN_M = 1.2
VEHICLE_LAT_MARGIN_M = 0.35

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


def _normalise_occupancy_grid(grid, blur_sigma=1.5):
    grid = np.asarray(grid, dtype=np.float64)
    grid = np.clip(grid, 0.0, None)
    total = grid.sum()
    if total > 1e-12:
        grid = grid / total
    if blur_sigma > 0.1 and grid.max() > 1e-12:
        grid = cv2.GaussianBlur(grid, (0, 0), blur_sigma)
        total = grid.sum()
        if total > 1e-12:
            grid = grid / total
    return grid


def _traj_pixel_yaw(traj_coords, idx):
    if idx < len(traj_coords) - 1:
        delta = traj_coords[idx + 1] - traj_coords[idx]
    elif idx > 0:
        delta = traj_coords[idx] - traj_coords[idx - 1]
    else:
        # Default to forward orientation [delta_row=-1, delta_col=0]
        return 0.0

    if np.linalg.norm(delta) < 1e-6:
        return 0.0

    # BEV Pixel Coordinates:
    # Forward motion -> delta_row < 0, delta_col = 0
    # Left motion -> delta_row = 0, delta_col > 0
    # We want yaw=0 for forward. arctan2(y, x) with x=forward, y=left.
    # So x = -delta_row, y = delta_col.
    return float(np.arctan2(delta[1], -delta[0]))


def _add_footprint(grid, row, col, yaw, weight, sigma):
    """
    Rasterize an oriented vehicle footprint blurred by Gaussian uncertainty.
    This creates a proper probability distribution P_pi(x_t).
    """
    # Dimensions in pixels
    half_l = VEHICLE_LENGTH_M / 2.0 / CELL_SIZE
    half_w = VEHICLE_WIDTH_M / 2.0 / CELL_SIZE

    # Corners in [forward, left] relative to center
    corners_local = np.array([
        [ half_l,  half_w],
        [ half_l, -half_w],
        [-half_l, -half_w],
        [-half_l,  half_w]
    ], dtype=np.float32)

    # Rotation matrix (yaw 0 is forward)
    c, s = np.cos(yaw), np.sin(yaw)
    
    # BEV Coordinate Mapping:
    # Row decreases with forward motion. Col increases with left motion.
    # OpenCV fillConvexPoly expects (x, y) coordinates which map to (col, row).
    pts = np.zeros((4, 2), dtype=np.float32)
    for i in range(4):
        f, l = corners_local[i]
        f_rot = f * c - l * s
        l_rot = f * s + l * c
        pts[i] = [float(col + l_rot), float(row - f_rot)]

    mask = np.zeros_like(grid, dtype=np.float32)
    cv2.fillConvexPoly(mask, pts.astype(np.int32), 1.0)

    if sigma > 0.1:
        # Uncertainty is modeled as spatial blurring of the footprint
        ksize = int(2 * np.ceil(2 * sigma) + 1)
        if ksize % 2 == 0: ksize += 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), sigma)

    grid += weight * mask.astype(np.float64)


def _normalise_grid(grid):
    grid = np.asarray(grid, dtype=np.float64)
    total = grid.sum()
    if total > 1e-12:
        return grid / total
    return grid


def rasterize_planner(planner_trajs, planner_scores, sigma_base=1.5, expansion_rate=0.4):
    """
    Rasterize weighted planner trajectories onto a BEV probability grid.
    Uses OBB footprints and growing temporal uncertainty.
    """
    planner_trajs = np.asarray(planner_trajs, dtype=np.float64)
    if planner_trajs.ndim == 2:
        planner_trajs = planner_trajs[None, ...]
    planner_scores = _normalise_scores(planner_scores)

    probability_grid = np.zeros((GRID_H, GRID_W), dtype=np.float64)
    waypoint_coords = _traj_to_bev_coords(planner_trajs)  # (K, T, 2)
    num_waypoints = waypoint_coords.shape[1]

    for traj_idx, weight in enumerate(planner_scores):
        traj_coords = waypoint_coords[traj_idx]
        for t in range(num_waypoints):
            row_center, col_center = traj_coords[t]
            yaw = _traj_pixel_yaw(traj_coords, t)
            sigma_t = sigma_base + (t * expansion_rate)
            _add_footprint(probability_grid, row_center, col_center, yaw, weight, sigma_t)

    return _normalise_grid(probability_grid)


def rasterize_planner_temporal(planner_trajs, planner_scores, sigma_base=1.5, expansion_rate=0.4):
    """
    Rasterize each planner waypoint timestep separately.
    Implements linearly growing uncertainty: sigma_t = sigma_base + t * expansion_rate.
    """
    planner_trajs = np.asarray(planner_trajs, dtype=np.float64)
    if planner_trajs.ndim == 2:
        planner_trajs = planner_trajs[None, ...]
    planner_scores = _normalise_scores(planner_scores)

    waypoint_coords = _traj_to_bev_coords(planner_trajs)
    num_waypoints = waypoint_coords.shape[1]
    probability_grids = np.zeros((num_waypoints, GRID_H, GRID_W), dtype=np.float64)

    for t in range(num_waypoints):
        sigma_t = sigma_base + (t * expansion_rate)
        grid = probability_grids[t]
        for traj_idx, weight in enumerate(planner_scores):
            traj_coords = waypoint_coords[traj_idx]
            row_center, col_center = traj_coords[t]
            yaw = _traj_pixel_yaw(traj_coords, t)
            _add_footprint(grid, row_center, col_center, yaw, weight, sigma_t)
        probability_grids[t] = _normalise_grid(grid)

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
    Transformation: Transpose, flip both axes, then shift 1px
    down-right to align the even-sized grid centers.
    Uses zero-fill instead of wrap to avoid ghost occupancy
    at the opposite edge.
    """
    aligned = occ.T[::-1, ::-1]
    result = np.zeros_like(aligned)
    result[1:, 1:] = aligned[:-1, :-1]
    return result


def planner_conditioned_occupancy(timesteps, blur_sigma=0.0):
    """
    Planner-conditioned occupancy risk score.
    
    Instead of computing full-grid JS divergence, directly query the
    aligned occupancy grid at each planner waypoint cell.  This isolates
    "is there something where the planner expects to go?" from the
    scene-wide occupancy statistics that dominate the full-grid JS.
    
    For each timestep i:
      1. Convert K trajectory modes to BEV pixel coordinates.
      2. For each waypoint h, align occupancy_future[h] to planner BEV.
      3. Sample aligned occupancy at the (row, col) of each mode.
      4. Weight by mode probability and average over waypoints.
    
    Parameters
    ----------
    timesteps : list of dict
        Logger timesteps, each containing planner_trajs, planner_scores,
        occupancy_grid, and optionally occupancy_future / occupancy_future_valid.
    blur_sigma : float, default 0.0
        Optional Gaussian blur applied to the aligned occupancy before
        sampling.  0.0 = no blur (raw binary).  0.5 = light spatial spread.
    
    Returns
    -------
    risk : np.ndarray, shape (N,)
        Per-timestep planner-conditioned occupancy score: the expected
        occupancy probability at the planner's predicted trajectory cells.
        NaN for steps with missing planner data or no valid occupancy.
    has_valid : np.ndarray, shape (N,), bool
        True for steps where the computation was valid.
    """
    N = len(timesteps)
    risk = np.full(N, np.nan, dtype=np.float64)
    has_valid = np.zeros(N, dtype=bool)

    for i, t in enumerate(timesteps):
        trajs = t['planner_trajs']
        scores = t['planner_scores']
        occ_current = t['occupancy_grid']

        if trajs is None or scores is None or len(trajs) == 0:
            continue

        # Normalise scores to a probability distribution over modes
        scores = np.asarray(scores, dtype=np.float64).ravel()
        s_total = scores.sum()
        if s_total < 1e-12:
            scores = np.ones_like(scores) / max(len(scores), 1)
        else:
            scores = scores / s_total

        K, T, _ = trajs.shape
        bev = _traj_to_bev_coords(trajs)          # (K, T, 2) — (row, col)

        # Occupancy future slices
        if 'occupancy_future' in t:
            occ_future = np.asarray(t['occupancy_future'])
            occ_valid = np.asarray(
                t.get('occupancy_future_valid',
                      np.ones(len(occ_future), dtype=bool)),
                dtype=bool,
            )
        else:
            occ_future = np.asarray(occ_current)[None, ...]
            occ_valid = np.array([True], dtype=bool)

        h_max = min(T, len(occ_future))
        total_risk = 0.0
        n_valid = 0

        for h in range(h_max):
            if not occ_valid[h]:
                continue

            aligned = _align_occupancy_to_planner_bev(occ_future[h]).astype(np.float64)

            if blur_sigma > 0.0 and aligned.max() > 1e-12:
                aligned = cv2.GaussianBlur(aligned, (0, 0), blur_sigma)

            # Sample occupancy at each mode's (row, col) for this waypoint
            mode_vals = np.zeros(K, dtype=np.float64)
            for k in range(K):
                r = int(round(bev[k, h, 0]))
                c = int(round(bev[k, h, 1]))
                r = int(np.clip(r, 0, GRID_H - 1))
                c = int(np.clip(c, 0, GRID_W - 1))
                mode_vals[k] = aligned[r, c]

            total_risk += float(np.sum(scores * mode_vals))
            n_valid += 1

        if n_valid > 0:
            risk[i] = float(total_risk / n_valid)
            has_valid[i] = True

    return risk, has_valid


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
                q_grid = _normalise_occupancy_grid(q_grid)

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
        'temporal_agreement_raw': 1.0 - temporal_conflict_raw,
        'temporal_agreement_smooth': 1.0 - temporal_conflict_smooth,
        'temporal_js_raw': temporal_js_raw,
        'temporal_js_smooth': temporal_js_smooth,
        'temporal_valid_count': temporal_valid_count,
        # Backward-compatible aliases used by existing analysis code.
        'divergence_raw': temporal_conflict_raw,
        'divergence_smooth': temporal_conflict_smooth,
        'planner_spread': planner_spread,
        'has_occupancy': has_occupancy,
    }
