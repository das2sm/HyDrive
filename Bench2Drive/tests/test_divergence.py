"""
Unit tests for divergence computation

Run with:
    conda run -n sparsedrive python tests/test_divergence.py
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from divergence import (
    _traj_to_bev_coords,
    _normalise_grid,
    _smooth_normalise,
    _traj_pixel_yaw,
    rasterize_planner,
    rasterize_planner_temporal,
    js_divergence,
    js_conflict_score,
    compute_divergence_series,
    _align_occupancy_to_planner_bev,
    planner_conditioned_occupancy,
    GRID_H, GRID_W, CELL_SIZE, JS_MAX,
)


def _planner_bev_to_guardian(r_p, c_p):
    """
    Inverse of _align_occupancy_to_planner_bev.

    Guardian(r_g, c_g) -> Planner(GRID_W - c_g, GRID_H - r_g),
    so Planner(r_p, c_p) <- Guardian(GRID_H - c_p, GRID_W - r_p).
    """
    return GRID_H - c_p, GRID_W - r_p


# ===================================================================
# _traj_to_bev_coords
# ===================================================================

def test_bev_coord_transform():
    """BEV coordinate transforms: origin, forward, left, clipping."""
    print("\n[TEST] BEV coordinate transforms")

    # Ego at origin -> grid center
    coords = _traj_to_bev_coords(np.array([[[0.0, 0.0]]]))
    assert np.array_equal(coords[0, 0], [GRID_H // 2, GRID_W // 2]), f"origin: {coords[0,0]}"
    print(f"  origin -> [{coords[0,0,0]}, {coords[0,0,1]}]")

    # 10m forward -> row decreases
    coords = _traj_to_bev_coords(np.array([[[0.0, 10.0]]]))
    expected_r = GRID_H // 2 - 10.0 / CELL_SIZE
    assert coords[0, 0, 0] == expected_r, f"forward row: {coords[0,0,0]} != {expected_r}"
    print(f"  10m forward -> row={coords[0,0,0]}")

    # 10m left -> col increases
    coords = _traj_to_bev_coords(np.array([[[10.0, 0.0]]]))
    expected_c = GRID_W // 2 + 10.0 / CELL_SIZE
    assert coords[0, 0, 1] == expected_c, f"left col: {coords[0,0,1]} != {expected_c}"
    print(f"  10m left -> col={coords[0,0,1]}")

    # Out-of-range -> clipped to bounds
    coords = _traj_to_bev_coords(np.array([[[100.0, 100.0]]]))
    r, c = coords[0, 0]
    assert r == 0.0 and c == GRID_W - 1, f"clip: got ({r},{c})"
    print(f"  [100,100] clipped to (0, {GRID_W-1})  \u2713 PASS")


# ===================================================================
# _normalise_grid
# ===================================================================

def test_normalise_grid():
    """Normalize: basic, all-zero."""
    print("\n[TEST] _normalise_grid")
    assert np.allclose(_normalise_grid(np.array([1.0, 2.0, 3.0])),
                       [1/6, 2/6, 3/6]), "basic"
    print(f"  [1,2,3] -> normalized")
    assert _normalise_grid(np.zeros((10, 10))).sum() == 0.0, "all-zero"
    print(f"  zeros sum = 0.0  \u2713 PASS")


def test_smooth_normalise():
    """Smooth normalize: blur spreads, negative clipping, all-zero."""
    print("\n[TEST] _smooth_normalise")
    # No blur: just normalize
    r = _smooth_normalise(np.array([[0, 0], [0, 10.0]]), blur_sigma=0.0)
    assert np.isclose(r[1, 1], 1.0), f"no-blur peak: {r[1,1]}"
    print(f"  no-blur peak = {r[1,1]:.4f}")

    # Blur spreads: same grid with sigma > 0 has lower peak
    r_blur = _smooth_normalise(np.array([[0, 0], [0, 10.0]]), blur_sigma=2.0)
    assert r_blur[1, 1] < r[1, 1], f"blur peak {r_blur[1,1]:.4f} >= no-blur {r[1,1]:.4f}"
    assert np.isclose(r_blur.sum(), 1.0), f"blur sum: {r_blur.sum()}"
    print(f"  blur reduces peak ({r[1,1]:.4f} -> {r_blur[1,1]:.4f}), sum={r_blur.sum():.4f}")

    # Negative values clipped
    r = _smooth_normalise(np.array([-5.0, 5.0]), blur_sigma=0.0)
    assert r.min() >= 0.0, f"negative remains: {r}"
    assert np.isclose(r.sum(), 1.0), f"neg sum: {r.sum()}"
    print(f"  neg clipped, min={r.min():.4f}")

    # All-zero -> zero
    r = _smooth_normalise(np.zeros((GRID_H, GRID_W)), blur_sigma=1.5)
    assert r.sum() == 0.0, f"all-zero sum: {r.sum()}"
    print(f"  all-zero sum = {r.sum()}  \u2713 PASS")


# ===================================================================
# _traj_pixel_yaw
# ===================================================================

def test_yaw():
    """Yaw: forward=0, left=pi/2, right=-pi/2, edge cases."""
    print("\n[TEST] Yaw computation")
    assert np.isclose(_traj_pixel_yaw(np.array([[60, 60], [40, 60]]), 0), 0.0,
                      atol=1e-6), "forward"
    print(f"  forward -> 0")
    assert np.isclose(_traj_pixel_yaw(np.array([[60, 60], [60, 80]]), 0),
                      np.pi / 2, atol=1e-6), "left"
    print(f"  left -> pi/2")
    assert np.isclose(_traj_pixel_yaw(np.array([[60, 60], [60, 40]]), 0),
                      -np.pi / 2, atol=1e-6), "right"
    print(f"  right -> -pi/2")

    # Edge cases: single waypoint, zero delta, last waypoint
    assert np.isclose(_traj_pixel_yaw(np.array([[60, 60]]), 0), 0.0,
                      atol=1e-6), "single wp"
    print(f"  single wp -> 0")
    assert np.isclose(_traj_pixel_yaw(np.array([[60, 60], [60, 60]]), 0), 0.0,
                      atol=1e-6), "zero delta"
    print(f"  zero delta -> 0")
    assert np.isclose(_traj_pixel_yaw(np.array([[60, 60], [60, 80]]), 1),
                      np.pi / 2, atol=1e-6), "last wp"
    print(f"  last waypoint uses backward delta -> pi/2  \u2713 PASS")


# ===================================================================
# js_divergence
# ===================================================================

def test_js_divergence():
    """JS divergence: symmetry, identity, disjoint, empty inputs."""
    print("\n[TEST] JS divergence")
    np.random.seed(42)
    p = np.random.rand(100); p = p / p.sum()
    q = np.random.rand(100); q = q / q.sum()
    assert np.isclose(js_divergence(p, q), js_divergence(q, p), atol=1e-8), "symmetry"
    print(f"  symmetric: D(p,q)={js_divergence(p,q):.6f}")
    assert js_divergence(p, p) < 1e-6, "identity"
    print(f"  identity: D(p,p)={js_divergence(p,p):.8f}")

    p_dj = np.zeros(100); p_dj[:50] = 1; p_dj /= p_dj.sum()
    q_dj = np.zeros(100); q_dj[50:] = 1; q_dj /= q_dj.sum()
    assert np.isclose(js_divergence(p_dj, q_dj), JS_MAX, atol=1e-6), "disjoint"
    print(f"  disjoint -> {js_divergence(p_dj,q_dj):.6f}")

    # Empty inputs all return JS_MAX
    q_full = np.ones(100); q_full /= q_full.sum()
    assert np.isclose(js_divergence(np.zeros(100), q_full), JS_MAX, atol=1e-6), "empty p"
    assert np.isclose(js_divergence(q_full, np.zeros(100)), JS_MAX, atol=1e-6), "empty q"
    assert np.isclose(js_divergence(np.zeros(100), np.zeros(100)), JS_MAX, atol=1e-6), "both empty"
    print(f"  empty inputs -> {JS_MAX:.6f}  \u2713 PASS")


# ===================================================================
# js_conflict_score
# ===================================================================

def test_conflict_score():
    """Conflict score: 0->1.0, log2->0.0, NaN/Inf->NaN."""
    print("\n[TEST] Conflict score")
    assert js_conflict_score(0.0) == 1.0, "perfect overlap"
    print(f"  conflict(0) = {js_conflict_score(0.0)}")
    assert js_conflict_score(JS_MAX) == 0.0, "disjoint"
    print(f"  conflict(log2) = {js_conflict_score(JS_MAX)}")
    assert np.isnan(js_conflict_score(np.nan)), "NaN"
    assert np.isnan(js_conflict_score(np.inf)), "Inf"
    print(f"  conflict(NaN/Inf) = NaN  \u2713 PASS")


# ===================================================================
# rasterize_planner
# ===================================================================

def test_rasterize_planner():
    """Rasterization: mass placement, blur, weighted modes, zero modes, bounds."""
    print("\n[TEST] Rasterization")

    # Center of mass matches trajectory position
    grid = rasterize_planner(np.array([[[0.0, 10.0]]]), np.array([1.0]),
                             sigma_base=0.0, expansion_rate=0.0)
    rows, cols = np.arange(GRID_H), np.arange(GRID_W)
    com_r = (grid * rows[:, None]).sum() / grid.sum()
    com_c = (grid * cols[None, :]).sum() / grid.sum()
    assert np.allclose([com_r, com_c], [40.0, 60.0], atol=0.6), f"COM=({com_r:.2f},{com_c:.2f})"
    print(f"  single mode COM = ({com_r:.2f}, {com_c:.2f})")

    # Blur reduces peak (spreads mass)
    grid_blur = rasterize_planner(np.array([[[0.0, 10.0]]]), np.array([1.0]),
                                  sigma_base=1.5, expansion_rate=0.0)
    peak_noblur = grid[round(com_r), round(com_c)]
    peak_blur = grid_blur[round(com_r), round(com_c)]
    assert peak_blur < peak_noblur, f"blur peak {peak_blur:.6f} >= noblur peak {peak_noblur:.6f}"
    # Also verify blur preserves COM location
    com_rb = (grid_blur * rows[:, None]).sum() / grid_blur.sum()
    com_cb = (grid_blur * cols[None, :]).sum() / grid_blur.sum()
    assert np.allclose([com_rb, com_cb], [40.0, 60.0], atol=0.6), f"blur COM=({com_rb:.2f},{com_cb:.2f})"
    print(f"  blur reduces peak ({peak_noblur:.6f} -> {peak_blur:.6f}), COM preserved")

    # Weighted modes
    trajs = np.array([[[0.0, 10.0]], [[10.0, 0.0]]])
    grid = rasterize_planner(trajs, np.array([0.3, 0.7]),
                             sigma_base=0.0, expansion_rate=0.0)
    com_r = (grid * rows[:, None]).sum() / grid.sum()
    com_c = (grid * cols[None, :]).sum() / grid.sum()
    assert np.allclose([com_r, com_c], [54.0, 74.0], atol=0.6), f"weighted COM=({com_r:.2f},{com_c:.2f})"
    print(f"  weighted modes COM = ({com_r:.2f}, {com_c:.2f})")

    # Zero modes -> zero grid (not a crash)
    grid = rasterize_planner(np.zeros((0, 6, 2)), np.zeros(0))
    assert grid.shape == (GRID_H, GRID_W)
    assert grid.sum() == 0.0, f"zero modes sum = {grid.sum()}"
    print(f"  zero modes -> sum = {grid.sum()}")

    # Out-of-bounds
    grid = rasterize_planner(np.ones((1, 6, 2)) * 100.0, np.ones(1))
    assert grid.shape == (GRID_H, GRID_W) and not np.any(np.isnan(grid))
    assert np.isclose(grid.sum(), 1.0, rtol=1e-5)
    print(f"  out-of-bounds: sum={grid.sum():.4f}  \u2713 PASS")


# ===================================================================
# _align_occupancy_to_planner_bev
# ===================================================================

def test_coordinate_alignment():
    """
    Guardian (row=right, col=forward) -> Planner (row=ahead_inverted, col=left).
    Independent formula: Guardian(r,c) -> Planner(GRID_W-c, GRID_H-r).
    """
    print("\n[TEST] Coordinate alignment")
    occ = np.zeros((GRID_H, GRID_W))
    r_ahead = int(10.0 / CELL_SIZE + GRID_H // 2)
    occ[r_ahead, GRID_W // 2] = 1.0  # 10m ahead, centered

    peak = np.unravel_index(_align_occupancy_to_planner_bev(occ).argmax(),
                            (GRID_H, GRID_W))
    expected = (GRID_W - GRID_W // 2, GRID_H - r_ahead)
    assert peak == expected, f"center: got {peak}, expected {expected}"
    print(f"  10m ahead center -> {peak}")

    # 5m left, 10m ahead
    occ = np.zeros((GRID_H, GRID_W))
    c_left = int(-5.0 / CELL_SIZE + GRID_W // 2)
    occ[r_ahead, c_left] = 1.0
    peak = np.unravel_index(_align_occupancy_to_planner_bev(occ).argmax(),
                            (GRID_H, GRID_W))
    expected = (GRID_W - c_left, GRID_H - r_ahead)
    assert peak == expected, f"left: got {peak}, expected {expected}"
    print(f"  5m left, 10m ahead -> {peak}")

    # Self-inverse: two applications = identity for interior points
    occ = np.zeros((GRID_H, GRID_W))
    occ[80, 60] = 1.0  # interior point
    twice = _align_occupancy_to_planner_bev(_align_occupancy_to_planner_bev(occ))
    peak = np.unravel_index(twice.argmax(), twice.shape)
    assert peak == (80, 60), f"self-inverse: got {peak}, expected (80, 60)"
    print(f"  self-inverse: interior point preserved ({peak})  \u2713 PASS")


# ===================================================================
# Integration: rasterize -> JS -> conflict score
# ===================================================================

def test_integration_js_conflict():
    """
    Full pipeline: planner rasterization, aligned occupancy, JS, conflict score.
    Occupancy on planner path -> high conflict; no occupancy -> low conflict.
    """
    print("\n[TEST] Integration: rasterize -> JS -> conflict")
    trajs = np.array([[[0.0, 10.0]]])  # Straight ahead
    scores = np.array([1.0])
    planner_future = rasterize_planner_temporal(trajs, scores,
                                                sigma_base=0.0, expansion_rate=0.0)

    # Occupancy aligned to planner waypoint (BEV 40,60) -> Guardian (60, 80)
    occ_on_path = np.zeros((GRID_H, GRID_W))
    occ_on_path[60, 80] = 1.0
    occ_empty = np.zeros((GRID_H, GRID_W))

    for label, occ_grid in [("on path", occ_on_path), ("empty", occ_empty)]:
        aligned = _align_occupancy_to_planner_bev(occ_grid).astype(np.float64)
        q_grid = _smooth_normalise(aligned)
        p_grid = planner_future[0]
        js = js_divergence(p_grid, q_grid)
        cf = js_conflict_score(js)
        print(f"  {label}: JS={js:.6f}, conflict={cf:.6f}")

    # On-path: high conflict (overlap). Empty: low conflict (disjoint).
    cf_on = js_conflict_score(js_divergence(
        planner_future[0],
        _smooth_normalise(_align_occupancy_to_planner_bev(occ_on_path).astype(np.float64))))
    cf_empty = js_conflict_score(js_divergence(
        planner_future[0],
        _smooth_normalise(_align_occupancy_to_planner_bev(occ_empty).astype(np.float64))))
    assert cf_on > cf_empty, f"on-path conflict {cf_on:.4f} should be > empty {cf_empty:.4f}"
    assert cf_empty < 0.05, f"empty conflict {cf_empty:.4f} should be ~0"
    print(f"  on-path={cf_on:.4f} > empty={cf_empty:.4f}  \u2713 PASS")


# ===================================================================
# planner_conditioned_occupancy  (PRIMARY METRIC)
# ===================================================================

def _make_ts(trajs, scores, occ_grid, occ_future=None):
    ts = {'planner_trajs': trajs, 'planner_scores': scores,
          'occupancy_grid': occ_grid}
    if occ_future is not None:
        ts['occupancy_future'] = occ_future
        ts['occupancy_future_valid'] = np.ones(len(occ_future), dtype=bool)
    return ts


def test_pco():
    """PCO: hit, clear, weighted, full, invalid input, causal/oracle, blur."""
    print("\n[TEST] PCO")

    # Hit
    trajs = np.array([[[5.0, 10.0]]])
    occ = np.zeros((GRID_H, GRID_W))
    r_g, c_g = _planner_bev_to_guardian(40, 70)
    occ[r_g, c_g] = 1.0
    risk, valid = planner_conditioned_occupancy(
        [_make_ts(trajs, np.array([1.0]), occ)], blur_sigma=0.0, causal=True)
    assert valid[0] and risk[0] > 0.0, f"hit: {risk[0]}"
    print(f"  hit -> {risk[0]:.4f}")

    # Clear
    risk, valid = planner_conditioned_occupancy(
        [_make_ts(trajs, np.array([1.0]), np.zeros((GRID_H, GRID_W)))],
        blur_sigma=0.0, causal=True)
    assert valid[0] and risk[0] < 1e-10, f"clear: {risk[0]}"
    print(f"  clear -> {risk[0]:.4f}")

    # Multi-mode weighted
    trajs2 = np.array([[[5.0, 10.0]], [[0.0, 5.0]]])
    occ = np.zeros((GRID_H, GRID_W))
    occ[r_g, c_g] = 1.0
    risk, _ = planner_conditioned_occupancy(
        [_make_ts(trajs2, np.array([0.5, 0.5]), occ)],
        blur_sigma=0.0, causal=True)
    assert np.isclose(risk[0], 0.5, atol=1e-6), f"weighted: {risk[0]}"
    print(f"  multi-mode weighted -> {risk[0]:.4f}")

    # Full occupancy
    risk, _ = planner_conditioned_occupancy(
        [_make_ts(np.array([[[0.0, 10.0]]]), np.array([1.0]),
                  np.ones((GRID_H, GRID_W)))],
        blur_sigma=0.0, causal=True)
    assert np.isclose(risk[0], 1.0, atol=1e-6), f"full: {risk[0]}"
    print(f"  full occupancy -> {risk[0]:.4f}")

    # Invalid inputs -> NaN
    for case in [
        _make_ts(np.array([]), np.array([]), np.zeros((GRID_H, GRID_W))),
        _make_ts(None, None, np.zeros((GRID_H, GRID_W))),
        _make_ts(np.array([[[5.0, 10.0]]]), np.array([0.0]),
                 np.ones((GRID_H, GRID_W))),
    ]:
        risk, valid = planner_conditioned_occupancy(
            [case], blur_sigma=0.0, causal=True)
        assert np.isnan(risk[0]) and not valid[0], f"invalid: {risk[0]}, {valid[0]}"
    print(f"  empty/None/zero-scores -> NaN, invalid")

    # Causal vs oracle
    trajs = np.zeros((1, 6, 2))
    trajs[0, :, 0] = 5.0
    trajs[0, :, 1] = np.arange(1.0, 7.0)
    occ_future = np.zeros((6, GRID_H, GRID_W))
    r_g, c_g = _planner_bev_to_guardian(52, 70)
    occ_future[3, r_g, c_g] = 1.0
    ts = [_make_ts(trajs, np.array([1.0]), np.zeros((GRID_H, GRID_W)), occ_future)]
    risk_c, _ = planner_conditioned_occupancy(ts, blur_sigma=0.0, causal=True)
    risk_o, _ = planner_conditioned_occupancy(ts, blur_sigma=0.0, causal=False)
    assert risk_c[0] < 1e-10 and risk_o[0] > 0.0, f"causal={risk_c[0]:.4f}, oracle={risk_o[0]:.4f}"
    print(f"  causal={risk_c[0]:.4f}, oracle={risk_o[0]:.4f}")

    # Zero modes -> NaN
    risk, valid = planner_conditioned_occupancy(
        [_make_ts(np.zeros((0, 6, 2)), np.zeros(0), np.zeros((GRID_H, GRID_W)))],
        blur_sigma=0.0, causal=True)
    assert np.isnan(risk[0]) and not valid[0], f"zero modes: {risk[0]}"
    print(f"  zero modes -> NaN")

    # Blur spreads
    trajs = np.array([[[0.0, 10.0]]])
    occ = np.zeros((GRID_H, GRID_W))
    r_g, c_g = _planner_bev_to_guardian(39, 60)
    occ[r_g, c_g] = 1.0
    ts = [_make_ts(trajs, np.array([1.0]), occ)]
    risk_nb, _ = planner_conditioned_occupancy(ts, blur_sigma=0.0, causal=True)
    risk_b, _ = planner_conditioned_occupancy(ts, blur_sigma=1.5, causal=True)
    assert risk_nb[0] < 1e-10 and risk_b[0] > risk_nb[0], f"noblur={risk_nb[0]:.6f}, blur={risk_b[0]:.6f}"
    print(f"  no-blur={risk_nb[0]:.4f}, blur={risk_b[0]:.4f}  \u2713 PASS")


# ===================================================================
# compute_divergence_series
# ===================================================================

def test_compute_divergence_series():
    """Full divergence pipeline: presence detection, NaN handling, meaningful values."""
    print("\n[TEST] Divergence series")

    # --- Controlled single-step: occupancy on planner path vs far away ---
    trajs_ctrl = np.array([[[0.0, 10.0]]])  # Straight, 10m; BEV row=40, col=60
    scores_ctrl = np.array([1.0])

    occ_on = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    occ_on[60, 80] = 1.0  # Guardian aligned to BEV (40, 60)
    occ_future_on = np.tile(occ_on[None, ...], (6, 1, 1))

    occ_off = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    occ_off[0, 0] = 1.0  # Far corner, far from planner path
    occ_future_off = np.tile(occ_off[None, ...], (6, 1, 1))

    def _compute_one(occ_grid, occ_future):
        ts = [{
            'planner_trajs': trajs_ctrl,
            'planner_scores': scores_ctrl,
            'occupancy_grid': occ_grid,
            'occupancy_future': occ_future,
            'occupancy_future_valid': np.ones(6, dtype=bool),
        }]
        result = compute_divergence_series(ts, use_occupancy=True)
        return result['divergence_raw'][0]

    val_on = _compute_one(occ_on, occ_future_on)
    val_off = _compute_one(occ_off, occ_future_off)
    assert np.isfinite(val_on), f"on-path should be finite, got {val_on}"
    assert np.isfinite(val_off), f"off-path should be finite, got {val_off}"
    # On-path: occupancy overlaps planner -> higher conflict than off-path
    assert val_on > val_off, f"on-path conflict {val_on:.4f} should be > off-path {val_off:.4f}"
    print(f"  controlled: on-path={val_on:.4f}, off-path={val_off:.4f} (on > off)")

    # --- Multi-step with random data ---
    N = 100
    timesteps = []
    for i in range(N):
        np.random.seed(i)
        occ_future = np.zeros((6, GRID_H, GRID_W), dtype=np.float32)
        occ_grid = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        if i > 20:
            occ_future[:, 50:70, 55:65] = 0.5
            occ_grid[50:70, 55:65] = 0.5
        timesteps.append({
            'planner_trajs': np.random.randn(6, 6, 2) * (i / N),
            'planner_scores': np.ones(6) / 6,
            'occupancy_grid': occ_grid,
            'occupancy_future': occ_future,
            'occupancy_future_valid': np.ones(6, dtype=bool),
        })

    result = compute_divergence_series(timesteps, use_occupancy=True)
    assert len(result['divergence_raw']) == N
    assert len(result['divergence_smooth']) == N
    occ_count = result['has_occupancy'].sum()
    assert occ_count > 0, f"has_occupancy sum = {occ_count}"
    assert result['has_occupancy'][50], "no occ at step 50"
    assert np.isnan(result['divergence_raw'][:20]).all(), "early (no occ) should be NaN"
    occ_mask = result['has_occupancy']
    assert np.isnan(result['divergence_raw'][occ_mask]).sum() == 0, "NaN where occ exists"
    print(f"  {occ_count}/{N} with occupancy  \u2713 PASS")

    # No-occupancy path
    timesteps = [{
        'planner_trajs': np.random.randn(3, 6, 2),
        'planner_scores': np.ones(3) / 3,
        'occupancy_grid': np.zeros((GRID_H, GRID_W), dtype=np.float32),
        'occupancy_future': np.zeros((6, GRID_H, GRID_W), dtype=np.float32),
        'occupancy_future_valid': np.ones(6, dtype=bool),
    } for _ in range(10)]
    result = compute_divergence_series(timesteps, use_occupancy=False)
    assert np.all(np.isnan(result['divergence_raw'])), "no occ should be NaN"
    print(f"  use_occupancy=False -> all NaN  \u2713 PASS")


# ===================================================================
# Test runner
# ===================================================================

def main():
    print("=" * 70)
    print("DIVERGENCE COMPUTATION UNIT TESTS")
    print("=" * 70)

    tests = [
        test_bev_coord_transform,
        test_normalise_grid,
        test_smooth_normalise,
        test_yaw,
        test_js_divergence,
        test_conflict_score,
        test_rasterize_planner,
        test_coordinate_alignment,
        test_integration_js_conflict,
        test_pco,
        test_compute_divergence_series,
    ]

    failed = []
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"  \u2717 FAIL: {e}")
            failed.append(test.__name__)
        except Exception as e:
            print(f"  \u2717 ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed.append(test.__name__)

    print("\n" + "=" * 70)
    passed = len(tests) - len(failed)
    if failed:
        print(f"PASSED: {passed}/{len(tests)}, FAILED: {len(failed)}")
        for name in failed:
            print(f"  \u2717 {name}")
        return 1
    else:
        print(f"\u2713 ALL {len(tests)} TESTS PASSED")
        return 0


if __name__ == "__main__":
    exit(main())
