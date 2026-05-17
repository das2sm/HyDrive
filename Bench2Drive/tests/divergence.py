"""
Divergence computation for planner-world temporal mismatch analysis.

D(t) = JS divergence between:
  - Pπ: planner trajectory distribution rasterized to BEV grid
  - Pw: world occupancy grid (normalized to probability)

For routes with no occupancy (all-zero grid), falls back to
planner_spread_entropy (weighted variance of trajectory endpoints).
"""

import numpy as np


# BEV grid parameters (must match agent: 120x120, 60m range → 0.5m/cell)
GRID_H = 120
GRID_W = 120
GRID_RANGE = 30.0  # meters, symmetric around ego
CELL_SIZE = GRID_RANGE * 2 / GRID_H  # 0.5 m/cell

# Trajectory
TRAJ_WAYPOINTS = 6   # T waypoints per trajectory


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
    probability_grid = np.zeros((GRID_H, GRID_W), dtype=np.float64)
    waypoint_coords = _traj_to_bev_coords(planner_trajs)  # (K, T, 2)

    # Radius of the Gaussian kernel (cover 3σ in each direction)
    kernel_radius = int(np.ceil(3 * sigma))

    # Precompute a 2D Gaussian kernel centred at (0,0)
    kernel_offsets_y, kernel_offsets_x = np.mgrid[
        -kernel_radius : kernel_radius + 1,
        -kernel_radius : kernel_radius + 1,
    ]
    gaussian_kernel = np.exp(
        -(kernel_offsets_y**2 + kernel_offsets_x**2) / (2 * sigma**2)
    )

    for traj_idx, trajectory_weight in enumerate(planner_scores):
        for waypoint_idx in range(TRAJ_WAYPOINTS):
            # Continuous coordinates of the waypoint on the BEV grid
            row_center, col_center = waypoint_coords[traj_idx, waypoint_idx]

            # Nearest pixel for placing the kernel centre
            row_pixel = int(round(row_center))
            col_pixel = int(round(col_center))

            # Full (unbounded) row and column spans that the kernel would cover
            row_start_full = row_pixel - kernel_radius
            row_end_full   = row_pixel + kernel_radius + 1
            col_start_full = col_pixel - kernel_radius
            col_end_full   = col_pixel + kernel_radius + 1

            # --- Clipping: compute valid slices in the grid ---
            # Grid row slice (exclusive upper bound)
            grid_row_start = max(row_start_full, 0)
            grid_row_end   = min(row_end_full, GRID_H)

            # Grid column slice
            grid_col_start = max(col_start_full, 0)
            grid_col_end   = min(col_end_full, GRID_W)

            # --- Corresponding slices inside the kernel array ---
            # Row start in the kernel: skip rows that fall above the grid
            kernel_row_start = max(0, -row_start_full)
            # Row end in the kernel: start + number of valid rows
            kernel_row_end = kernel_row_start + (grid_row_end - grid_row_start)

            kernel_col_start = max(0, -col_start_full)
            kernel_col_end   = kernel_col_start + (grid_col_end - grid_col_start)

            # Only add if there is an actual overlap (non-empty slice)
            if grid_row_end > grid_row_start and grid_col_end > grid_col_start:
                probability_grid[
                    grid_row_start:grid_row_end,
                    grid_col_start:grid_col_end,
                ] += (
                    trajectory_weight
                    * gaussian_kernel[
                        kernel_row_start:kernel_row_end,
                        kernel_col_start:kernel_col_end,
                    ]
                )

    # Normalise to a valid probability distribution
    total_mass = probability_grid.sum()
    if total_mass > 1e-12:
        probability_grid /= total_mass
    return probability_grid


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
        # One or both distributions are essentially empty
        return np.nan
    
    p = p / p_sum
    q = q / q_sum
    m = 0.5 * (p + q)

    mask_p = p > eps
    mask_q = q > eps
    
    # KL(p||m): only sum where p > 0
    # m[i] >= 0.5*p[i] > 0 wherever p[i] > 0, so m[mask_p] > 0 guaranteed
    kl_pm = np.sum(p[mask_p] * np.log(p[mask_p] / m[mask_p]))
    
    # KL(q||m): only sum where q > 0
    kl_qm = np.sum(q[mask_q] * np.log(q[mask_q] / m[mask_q]))
    
    jsd = 0.5 * (kl_pm + kl_qm)
    return np.clip(jsd, 0.0, np.log(2.0))


def planner_spread_entropy(planner_trajs, planner_scores):
    """
    Fallback divergence signal when occupancy is unavailable.
    Measures weighted variance of trajectory endpoints (spread of planner distribution).
    Higher = more uncertain/spread planner = higher divergence signal.

    Returns scalar >= 0.
    """
    endpoints = planner_trajs[:, -1, :]  # (K, 2)
    mean = np.average(endpoints, weights=planner_scores, axis=0)
    diffs = endpoints - mean  # (K, 2)
    weighted_var = np.average(np.sum(diffs**2, axis=1), weights=planner_scores)
    return float(weighted_var)


def _align_occupancy_to_planner_bev(occ):
    """
    Align Guardian occupancy grid to planner BEV convention.

    Guardian grid (from guardian._local_to_grid):
        axis-0 (row) = right  (row increases with right distance)
        axis-1 (col) = forward (col increases with forward distance)

    Planner BEV (from _traj_to_bev_coords):
        axis-0 (row) = forward inverted  (row=0 is far forward, row=H is behind ego)
        axis-1 (col) = left  (col=0 is far right, col=W is far left) - confusing! - higher axis-1 values are more left

    Transformation:
        PlannerRow = (GRID_RANGE - forward) / resolution
        PlannerCol = (left + GRID_RANGE) / resolution

        Since Guardian:
        OldCol = forward / resolution + origin
        OldRow = right / resolution + origin
        And left = -right

        We have:
        PlannerRow = 120 - OldCol
        PlannerCol = 120 - OldRow

        Implementation: Transpose then flip both axes (equivalent to 180 deg rot of transpose).
    """
    return occ.T[::-1, ::-1]



def check_coordinate_alignment(planner_trajs, planner_scores, ego_forward_m=10.0):
    """
    Sanity check: place a synthetic obstacle directly ahead and verify that
    the planner grid and aligned occupancy grid both peak in the same region.

    Returns a dict with:
        planner_peak_row, planner_peak_col  - where planner mass concentrates
        occ_peak_row, occ_peak_col          - where occupancy mass concentrates
        row_diff, col_diff                  - pixel offset between peaks
        aligned                             - True if peaks are within 10px of each other
    """
    # Synthetic occupancy: blob directly ahead at ego_forward_m in Guardian convention
    # Guardian: row = forward/res + origin → row for ego_forward_m ahead
    res = GRID_RANGE * 2 / GRID_H  # same as CELL_SIZE
    origin = GRID_H // 2
    ahead_col = int(ego_forward_m / res + origin)
    ahead_row = origin  # right=0 → center column

    synthetic_occ = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    r0, r1 = max(0, ahead_row - 3), min(GRID_H, ahead_row + 4)
    c0, c1 = max(0, ahead_col - 3), min(GRID_W, ahead_col + 4)
    synthetic_occ[r0:r1, c0:c1] = 1.0

    # Align to planner BEV
    occ_aligned = _align_occupancy_to_planner_bev(synthetic_occ)
    occ_peak = np.unravel_index(occ_aligned.argmax(), occ_aligned.shape)

    # Planner grid peak
    p_grid = rasterize_planner(planner_trajs, planner_scores)
    planner_peak = np.unravel_index(p_grid.argmax(), p_grid.shape)

    row_diff = abs(int(occ_peak[0]) - int(planner_peak[0]))
    col_diff = abs(int(occ_peak[1]) - int(planner_peak[1]))

    # For a vehicle going straight, both peaks should be near top-center of BEV
    # (low row index = far forward, center col = straight ahead)
    expected_row = int((GRID_RANGE - ego_forward_m) / res)  # planner convention
    expected_col = origin  # left=0 → center

    return {
        'planner_peak': planner_peak,
        'occ_peak_aligned': occ_peak,
        'expected_row': expected_row,
        'expected_col': expected_col,
        'row_diff': row_diff,
        'col_diff': col_diff,
        'aligned': row_diff < 10 and col_diff < 10,
    }


def compute_divergence_series(timesteps, use_occupancy=True, temporal_window=5):
    """
    Compute D(t) for all timesteps.

    use_occupancy: if True and occupancy is non-zero, use JS divergence.
                   if False or occupancy is all-zero, use planner_spread_entropy.
    temporal_window: smooth D(t) with a causal rolling mean of this width.

    Returns dict with arrays of length N:
        divergence_raw, divergence_smooth, planner_spread,
        has_occupancy (bool per step)
    """
    N = len(timesteps)
    divergence_raw = np.zeros(N)
    planner_spread = np.zeros(N)
    has_occupancy = np.zeros(N, dtype=bool)

    for i, t in enumerate(timesteps):
        trajs = t['planner_trajs']    # (K, T, 2)
        scores = t['planner_scores']  # (K,)
        occ = t['occupancy_grid']     # (H, W) in Guardian convention

        spread = planner_spread_entropy(trajs, scores)
        planner_spread[i] = spread

        occ_has_signal = occ.max() > 1e-6
        has_occupancy[i] = occ_has_signal

        if use_occupancy and occ_has_signal:
            p_grid = rasterize_planner(trajs, scores)
            # Align occupancy to planner BEV convention before computing divergence
            q_grid = _align_occupancy_to_planner_bev(occ).astype(np.float64)
            q_grid = q_grid / (q_grid.sum() + 1e-10)
            
            # JS divergence is log(2) (~0.693) when disjoint (safe) and 0 when identical (overlap).
            # We want a RISK signal that RISES before collision (high overlap).
            js = js_divergence(p_grid, q_grid)
            divergence_raw[i] = 0.7 - js
        else:
            # No occupancy: use planner spread as proxy
            divergence_raw[i] = spread

    # Causal rolling mean (no lookahead)
    divergence_smooth = np.zeros(N)
    for i in range(N):
        start = max(0, i - temporal_window + 1)
        divergence_smooth[i] = divergence_raw[start:i+1].mean()

    return {
        'divergence_raw': divergence_raw,
        'divergence_smooth': divergence_smooth,
        'planner_spread': planner_spread,
        'has_occupancy': has_occupancy,
    }
