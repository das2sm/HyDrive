"""
Unit tests for divergence computation

Run with:
    python tests/test_divergence.py
"""

import numpy as np
import sys
from pathlib import Path

# Add tests directory and project root to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from divergence import (
    _traj_to_bev_coords,
    rasterize_planner,
    rasterize_planner_temporal,
    js_divergence,
    js_conflict_score,
    planner_spread_entropy,
    build_future_occupancy_windows,
    compute_divergence_series,
    _align_occupancy_to_planner_bev,
    check_coordinate_alignment,
    GRID_H, GRID_W, GRID_RANGE, CELL_SIZE,
)


def test_ego_at_origin():
    """Ego at center of BEV grid should map to grid center."""
    print("\n[TEST] Ego at origin → grid center")
    trajs = np.array([[[0.0, 0.0]]])  # Ego at origin
    coords = _traj_to_bev_coords(trajs)
    print(f"  Ego [0,0] → grid coords {coords[0, 0]}")
    assert np.allclose(coords[0, 0], [60, 60]), f"Expected [60,60], got {coords[0,0]}"
    print("  ✓ PASS")


def test_forward_motion():
    print("\n[TEST] Forward motion → reduced row")
    trajs = np.array([[[0.0, 10.0]]])
    coords = _traj_to_bev_coords(trajs)
    origin = GRID_H // 2
    expected_row = origin - 10.0 / CELL_SIZE
    actual_row = coords[0, 0, 0]
    print(f"  10m forward: row = {actual_row:.1f} (expected ≈{expected_row:.1f})")
    assert abs(actual_row - expected_row) < 1.0, f"Expected row ≈{expected_row:.0f}, got {actual_row}"
    print("  ✓ PASS")


def test_left_motion():
    print("\n[TEST] Left motion → increased col")
    trajs = np.array([[[10.0, 0.0]]])
    coords = _traj_to_bev_coords(trajs)
    origin = GRID_W // 2
    expected_col = origin + 10.0 / CELL_SIZE
    actual_col = coords[0, 0, 1]
    print(f"  10m left: col = {actual_col:.1f} (expected ≈{expected_col:.1f})")
    assert abs(actual_col - expected_col) < 1.0, f"Expected col ≈{expected_col:.0f}, got {actual_col}"
    print("  ✓ PASS")


def test_js_divergence_symmetry():
    """JS-divergence should be symmetric: D(p,q) == D(q,p)."""
    print("\n[TEST] JS-divergence symmetry")
    p = np.random.rand(100)
    q = np.random.rand(100)
    d_pq = js_divergence(p, q)
    d_qp = js_divergence(q, p)
    print(f"  D(p,q) = {d_pq:.6f}, D(q,p) = {d_qp:.6f}")
    assert np.isclose(d_pq, d_qp, atol=1e-8), f"Asymmetric: {d_pq:.6f} vs {d_qp:.6f}"
    print("  ✓ PASS")


def test_js_divergence_identical():
    """JS-divergence of identical distributions should be near zero."""
    print("\n[TEST] JS-divergence identity → zero")
    p = np.random.rand(100)
    p = p / p.sum()
    d = js_divergence(p, p)
    print(f"  D(p,p) = {d:.8f}")
    assert d < 1e-6, f"Expected ~0, got {d}"
    print("  ✓ PASS")


def test_rasterize_normalization():
    """Rasterized planner grid should sum to 1.0."""
    print("\n[TEST] Rasterization → normalized")
    np.random.seed(42)
    trajs = np.random.rand(10, 6, 2) * 20 - 10  # Random in [-10, 10]
    scores = np.ones(10) / 10
    grid = rasterize_planner(trajs, scores)
    total = grid.sum()
    print(f"  Grid sum = {total:.8f}")
    assert np.isclose(total, 1.0, rtol=1e-5), f"Grid sum={total}, expected 1.0"
    print("  ✓ PASS")


def test_planner_spread_entropy():
    """Spreading trajectory endpoints should increase entropy."""
    print("\n[TEST] Planner spread → entropy")
    
    # Concentrated endpoints
    trajs_conc = np.array([[[0, 0]], [[0.1, 0.1]], [[0.05, 0.05]]])  # (K=3, T=1, 2)
    scores = np.ones(3) / 3
    spread_conc = planner_spread_entropy(trajs_conc, scores)
    
    # Spread endpoints
    trajs_spread = np.array([[[5, 5]], [[-5, -5]], [[0, 10]]])  # (K=3, T=1, 2)
    spread_spread = planner_spread_entropy(trajs_spread, scores)
    
    print(f"  Concentrated: spread = {spread_conc:.6f}")
    print(f"  Spread:       spread = {spread_spread:.6f}")
    assert spread_spread > spread_conc, f"Spread should be higher: {spread_spread} vs {spread_conc}"
    print("  ✓ PASS")


def test_compute_divergence_series():
    """Full divergence pipeline on synthetic data."""
    print("\n[TEST] Full divergence series computation")
    
    # Create synthetic episode data
    N = 100  # timesteps
    timesteps = []
    for i in range(N):
        # Planner: simple Gaussian modes
        trajs = np.random.randn(6, 6, 2) * (i / N)  # Uncertainty increases over time
        scores = np.ones(6) / 6
        
        # Occupancy: simple random grid
        occ = np.zeros((120, 120), dtype=np.float32)
        if i > 20:  # Obstacles appear after step 20
            occ[50:70, 55:65] = 0.5  # Static obstacle
        
        timesteps.append({
            'planner_trajs': trajs,
            'planner_scores': scores,
            'occupancy_grid': occ,
        })
    
    result = compute_divergence_series(timesteps, use_occupancy=True)
    
    print(f"  Processed {len(timesteps)} timesteps")
    print(f"  Divergence shape: {result['divergence_raw'].shape}")
    print(f"  Min divergence: {np.nanmin(result['divergence_raw']):.6f}")
    print(f"  Max divergence: {np.nanmax(result['divergence_raw']):.6f}")
    print(f"  Steps with occupancy: {result['has_occupancy'].sum()}/{N}")
    
    assert len(result['divergence_raw']) == N
    assert len(result['divergence_smooth']) == N
    assert not np.any(np.isnan(result['divergence_raw'][result['has_occupancy']]))
    assert np.any(np.isnan(result['divergence_raw'][~result['has_occupancy']]))
    assert 'temporal_conflict_raw' in result
    assert 'temporal_js_raw' in result
    print("  ✓ PASS")


def test_future_occupancy_windows():
    print("\n[TEST] Future occupancy windows → aligned offsets")
    timesteps = []
    for i in range(8):
        occ = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        occ[GRID_H // 2, min(GRID_W - 1, GRID_W // 2 + i)] = 1.0
        timesteps.append({
            'planner_trajs': np.zeros((1, 2, 2), dtype=np.float32),
            'planner_scores': np.ones(1, dtype=np.float32),
            'occupancy_grid': occ,
        })

    windows, valid, offsets = build_future_occupancy_windows(
        timesteps,
        num_waypoints=2,
        horizon_frames=4,
    )

    print(f"  windows shape: {windows.shape}, offsets={offsets.tolist()}")
    assert windows.shape == (8, 2, GRID_H, GRID_W)
    assert offsets.tolist() == [2, 4]
    assert valid[0].tolist() == [True, True]
    assert valid[-1].tolist() == [False, False]
    assert windows[0, 0, GRID_H // 2, GRID_W // 2 + 2] == 1.0
    print("  ✓ PASS")


def test_temporal_conflict_uses_future_slice():
    print("\n[TEST] Temporal conflict uses matched future occupancy")
    planner_trajs = np.array([[[0.0, 5.0], [0.0, 10.0]]], dtype=np.float32)
    planner_scores = np.ones(1, dtype=np.float32)
    planner_future = rasterize_planner_temporal(planner_trajs, planner_scores)

    occ_future = np.zeros((2, GRID_H, GRID_W), dtype=np.float32)
    for h in range(2):
        # Convert planner-aligned future distribution back into Guardian frame
        # for storage. The alignment transform is its own inverse.
        occ_future[h] = _align_occupancy_to_planner_bev(planner_future[h]).astype(np.float32)

    result = compute_divergence_series([{
        'planner_trajs': planner_trajs,
        'planner_scores': planner_scores,
        'occupancy_grid': np.zeros((GRID_H, GRID_W), dtype=np.float32),
        'occupancy_future': occ_future,
        'occupancy_future_valid': np.array([True, True]),
    }])

    print(f"  temporal conflict: {result['temporal_conflict_raw'][0]:.4f}")
    assert result['temporal_valid_count'][0] == 2
    assert result['temporal_conflict_raw'][0] > 0.95
    assert np.isfinite(result['temporal_js_raw'][0])
    print("  ✓ PASS")


def test_grid_bounds():
    """Rasterization handles out-of-bounds trajectories gracefully."""
    print("\n[TEST] Out-of-bounds trajectories → handled")
    
    # Far outside grid range (K=1, T=6 waypoints)
    trajs = np.ones((1, 6, 2)) * 100.0  # Well outside ±50m range
    scores = np.ones(1)
    grid = rasterize_planner(trajs, scores)
    
    print(f"  [100, 100] trajectory → grid sum = {grid.sum():.4f}")
    # Grid should still be valid even with out-of-bounds contributions
    assert grid.shape == (120, 120)
    assert not np.any(np.isnan(grid))
    print("  ✓ PASS (out-of-bounds handled)")


def test_coordinate_alignment():
    print("\n[TEST] Coordinate alignment: Guardian occ → planner BEV")
    origin = GRID_H // 2
    res = CELL_SIZE

    # Place blob 10m ahead in Guardian
    ahead_row = int(10.0 / res + origin)
    occ = np.zeros((GRID_H, GRID_W))
    occ[ahead_row - 2:ahead_row + 3, origin - 2:origin + 3] = 1.0

    # Apply alignment
    occ_aligned = _align_occupancy_to_planner_bev(occ)
    peak = np.unravel_index(occ_aligned.argmax(), occ_aligned.shape)

    # Expected: manually apply the same transformation to a single point
    single = np.zeros((GRID_H, GRID_W))
    single[ahead_row, origin] = 1.0
    expected_aligned = _align_occupancy_to_planner_bev(single)
    expected_peak = np.unravel_index(expected_aligned.argmax(), expected_aligned.shape)

    print(f"  Original blob at row={ahead_row}, col={origin}")
    print(f"  After alignment: peak at row={peak[0]}, col={peak[1]}")
    print(f"  Expected peak (from single point): row={expected_peak[0]}, col={expected_peak[1]}")

    assert np.allclose(peak, expected_peak, atol=2), f"Peak mismatch: got {peak}, expected {expected_peak}"
    print("  ✓ PASS")


def main():
    print("=" * 70)
    print("DIVERGENCE COMPUTATION UNIT TESTS")
    print("=" * 70)
    
    tests = [
        test_ego_at_origin,
        test_forward_motion,
        test_left_motion,
        test_js_divergence_symmetry,
        test_js_divergence_identical,
        test_rasterize_normalization,
        test_planner_spread_entropy,
        test_compute_divergence_series,
        test_future_occupancy_windows,
        test_temporal_conflict_uses_future_slice,
        test_grid_bounds,
        test_coordinate_alignment,
    ]
    
    failed = []
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}")
            failed.append(test.__name__)
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed.append(test.__name__)
    
    print("\n" + "=" * 70)
    if failed:
        print(f"FAILED: {len(failed)}/{len(tests)} tests")
        for name in failed:
            print(f"  ✗ {name}")
        return 1
    else:
        print(f"✓ PASSED: All {len(tests)} tests")
        return 0


if __name__ == "__main__":
    exit(main())
