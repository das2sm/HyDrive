"""
Divergence and Planner Conditioned Occupancy (PCO) metrics
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


def _normalise_grid(grid):
    grid = np.asarray(grid, dtype=np.float64)
    total = grid.sum()
    if total > 1e-12:
        return grid / total
    return grid


def _smooth_normalise(grid, blur_sigma=1.5):
    grid = np.asarray(grid, dtype=np.float64)
    grid = np.clip(grid, 0.0, None)
    
    if blur_sigma > 0.1:
        grid = cv2.GaussianBlur(grid, (0, 0), blur_sigma)
        
    return _normalise_grid(grid)


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
    half_l = (VEHICLE_LENGTH_M / 2.0 + VEHICLE_LONG_MARGIN_M) / CELL_SIZE
    half_w = (VEHICLE_WIDTH_M / 2.0 + VEHICLE_LAT_MARGIN_M) / CELL_SIZE

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
        mask = cv2.GaussianBlur(mask, (0, 0), sigma)
        ''' Debug visualization of the footprint mask
        import matplotlib.pyplot as plt
        plt.imshow(mask, cmap='hot', interpolation='nearest')
        plt.title(f"Footprint mask (weight={weight:.3f}, sigma={sigma:.2f})")
        plt.colorbar()
        plt.show()
        '''

    grid += weight * mask.astype(np.float64)


def rasterize_planner(planner_trajs, planner_scores, sigma_base=1.5, expansion_rate=0.4):
    """
    Rasterize weighted planner trajectories onto a BEV probability grid.
    Uses Oriented Bounding Box footprints and growing temporal uncertainty.
    """
    planner_trajs = np.asarray(planner_trajs, dtype=np.float64)
    if planner_trajs.ndim == 2:
        planner_trajs = planner_trajs[None, ...]

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


def _align_occupancy_to_planner_bev(occ):
    """
    Align Guardian occupancy grid (row=right, col=forward) 
    to planner BEV (row=ahead_inverted, col=left).
    Transformation: Transpose, flip both axes, then shift 1px
    down-right to align the even-sized grid centers.
    Uses zero-fill instead of wrap to avoid ghost occupancy
    at the opposite edge.

    Note: visualizing this reveals that the result mirrors the real world horizontally (left-right)
    """
    aligned = occ.T[::-1, ::-1]
    result = np.zeros_like(aligned)
    result[1:, 1:] = aligned[:-1, :-1]
    return result


def planner_conditioned_occupancy(timesteps, blur_sigma=0.0, causal=True):
    """
    Planner-conditioned occupancy risk score.
    
    Measures whether the ground-truth world currently has objects where the
    planner intends to drive.  The occupancy grid (occupancy_grid) is
    CARLA ground truth — every vehicle, walker, and static prop within 75m
    rasterized onto a 120×120 grid in Guardian frame.  It is NOT the
    planner's own obstacle prediction.
    
    PCO samples this ground-truth occupancy at the planner's trajectory
    waypoints and averages over modes weighted by mode probability.
    High PCO = the planner intends to go where objects currently are.
    
    Unlike full-grid JS divergence, this isolates the specific question
    "is there something where the planner expects to go?" from scene-wide
    occupancy statistics.
    
    For each timestep i:
      1. Convert K trajectory modes to BEV pixel coordinates.
      2. For each waypoint h, align occupancy to planner BEV.
      3. For each mode k, sample aligned occupancy at the (row, col) of the corresponding BEV coordinate.
      4. Weight by mode probability and average over waypoints.
    
    Parameters
    ----------
    timesteps : list of dict
        Logger timesteps, each containing planner_trajs, planner_scores,
        occupancy_grid (CARLA ground-truth world occupancy), and optionally
        occupancy_future / occupancy_future_valid.
    blur_sigma : float, default 0.0
        Optional Gaussian blur applied to the aligned occupancy before
        sampling.  0.0 = no blur (raw binary).  0.5 = light spatial spread.
    causal : bool, default True
        If True: repeat current-frame ground-truth occupancy across all
        waypoints.  This is the deployable predictor — it knows where
        obstacles are right now but not where they will be.
        If False: use ground-truth future occupancy (occupancy_future)
        when available.  This is an oracle upper bound.
    
    Returns
    -------
    risk : np.ndarray, shape (N,)
        Per-timestep PCO score in [0, 1]: the expected ground-truth
        occupancy intensity at the planner's predicted trajectory cells.
        
        Note: This is a fraction of planner belief mass intersecting occupancy,
        not a calibrated collision probability. Typical collision-window
        values are 0.2–0.5 (only later waypoints overlap the obstacle,
        and not all modes fire simultaneously). Safe values are <0.05.
        AUROC is invariant to scale, so the low dynamic range does not
        affect discriminative power.
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

        scores = np.asarray(scores, dtype=np.float64)
        s_total = scores.sum()
        if s_total < 1e-12:
            continue

        K, T, _ = trajs.shape
        bev = _traj_to_bev_coords(trajs)          # (K, T, 2) —> (row, col)

        if causal:
            # Limitation: tiles current occupancy across all waypoints, so PCO
            # only catches obstacles already overlapping the planner's cells.
            # Dynamic obstacles not yet in collision path are invisible until
            # they enter the occupied region — this is the oracle gap.
            occ_future = np.tile(np.asarray(occ_current)[None, ...], (T, 1, 1))
            occ_valid = np.ones(T, dtype=bool)
        else:
            # Oracle version with future occupancy
            occ_future = np.asarray(t['occupancy_future'])
            occ_valid = np.asarray(t['occupancy_future_valid'])
           
        h_max = min(T, len(occ_future))
        total_risk = 0.0
        n_valid = 0

        # For each WAYPOINT -> 0 to T-1
        for h in range(h_max):
            if not occ_valid[h]:
                continue

            aligned = _align_occupancy_to_planner_bev(occ_future[h]).astype(np.float64)

            if blur_sigma > 0.0 and aligned.max() > 1e-12:
                aligned = cv2.GaussianBlur(aligned, (0, 0), blur_sigma)

            mode_vals = np.zeros(K, dtype=np.float64)

            # For each MODE -> 0 to K-1
            for k in range(K):
                r = int(round(bev[k, h, 0]))
                c = int(round(bev[k, h, 1]))

                # Taking r, c from the planner's bev coordinates
                # And sampling the aligned occupancy at that location
                # from the ground-truth occupancy grid.
                mode_vals[k] = aligned[r, c]

            total_risk += float(np.sum(scores * mode_vals))
            n_valid += 1

        if n_valid > 0:
            risk[i] = float(total_risk / n_valid)
            has_valid[i] = True

    return risk, has_valid


def compute_divergence_series(timesteps, use_occupancy=True, temporal_window=5):
    """
    Legacy full-grid JS divergence between planner rasterization and world occupancy.

    This is the confounded global JS metric — NOT the primary PCO metric.
    Retained only for ablation comparisons.

    For each timestep, rasters the K planner modes into per-waypoint occupancy
    distributions (planner_future) and computes Jensen-Shannon divergence against
    the aligned world occupancy grid.  The per-waypoint scores are averaged,
    then smoothed with a causal rolling mean over temporal_window frames.

    Uses occupancy_future (ground-truth future slices) — still a confounded
    metric even with oracle information (AUROC 0.610, scene-density limited).

    Parameters
    ----------
    timesteps : list of dict
        Logger timesteps, each containing planner_trajs, planner_scores,
        occupancy_future / occupancy_future_valid.
    use_occupancy : bool, default True
        If False, skip all occupancy-based computation (all-NaN output).
    temporal_window : int, default 5
        Causal rolling mean window for smoothing.

    Returns
    -------
    dict with keys:
        temporal_conflict_raw     — per-step raw conflict score (mean JS per waypoint)
        temporal_conflict_smooth  — causal rolling mean of conflict_raw
        temporal_valid_count      — number of valid waypoint slices averaged
        divergence_raw            — alias for temporal_conflict_raw
        divergence_smooth         — alias for temporal_conflict_smooth
        has_occupancy             — bool array, True where occupancy was available
    """
    N = len(timesteps)
    temporal_conflict_raw = np.full(N, np.nan)
    temporal_valid_count = np.zeros(N, dtype=np.int64)
    has_occupancy = np.zeros(N, dtype=bool)

    for i, t in enumerate(timesteps):
        trajs = t['planner_trajs']    
        scores = t['planner_scores']  
        occ_future = np.asarray(t['occupancy_future'])
        occ_valid = np.asarray(t['occupancy_future_valid'], dtype=bool)

        occ_has_signal = np.array([occ.max() > 1e-6 for occ in occ_future], dtype=bool)
        valid_slices = occ_valid & occ_has_signal
        has_occupancy[i] = valid_slices.any()

        if use_occupancy and valid_slices.any():
            planner_future = rasterize_planner_temporal(trajs, scores)
            num_slices = min(len(planner_future), len(occ_future))
            conflict_values = []
            for h in range(num_slices):
                if not valid_slices[h]:
                    continue
                p_grid = planner_future[h]
                q_grid = _align_occupancy_to_planner_bev(occ_future[h]).astype(np.float64)
                q_grid = _smooth_normalise(q_grid)

                js = js_divergence(p_grid, q_grid)
                conflict_values.append(js_conflict_score(js))

            temporal_valid_count[i] = len(conflict_values)
            if conflict_values:
                temporal_conflict_raw[i] = float(np.mean(conflict_values))

    # Causal rolling mean
    temporal_conflict_smooth = np.full(N, np.nan)
    for i in range(N):
        start = max(0, i - temporal_window + 1)

        if not np.isfinite(temporal_conflict_raw[i]):
            continue

        conflict_window = temporal_conflict_raw[start:i+1]
        conflict_valid = np.isfinite(conflict_window)
        if conflict_valid.any():
            temporal_conflict_smooth[i] = conflict_window[conflict_valid].mean()

    return {
        'temporal_conflict_raw': temporal_conflict_raw,
        'temporal_conflict_smooth': temporal_conflict_smooth,
        'temporal_valid_count': temporal_valid_count,
        'divergence_raw': temporal_conflict_raw,
        'divergence_smooth': temporal_conflict_smooth,
        'has_occupancy': has_occupancy,
    }
