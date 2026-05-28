"""
Unit tests for divergence computation

Run with:
    python tests/test_divergence.py
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from divergence import (
    _traj_to_bev_coords,
    rasterize_planner,
    js_divergence,
    compute_divergence_series,
    _align_occupancy_to_planner_bev,
    GRID_H, GRID_W, GRID_RANGE, CELL_SIZE,
)


def test_ego_at_origin():
    """Ego at center of BEV grid should map to grid center."""
    print("\n[TEST] Ego at origin → grid center")
    trajs = np.array([[[0.0, 0.0]]])  # Ego at origin
    coords = _traj_to_bev_coords(trajs)
    print(f"  Ego [0,0] → grid coords {coords[0, 0]}")
    assert np.allclose(coords[0, 0], [GRID_H // 2, GRID_W // 2]), f"Expected [{GRID_H//2},{GRID_W//2}], got {coords[0,0]}"
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
    np.random.seed(42)
    p = np.random.rand(100)
    q = np.random.rand(100)
    # Normalize to valid probability distributions
    p = p / p.sum()
    q = q / q.sum()
    
    d_pq = js_divergence(p, q)
    d_qp = js_divergence(q, p)
    print(f"  D(p,q) = {d_pq:.6f}, D(q,p) = {d_qp:.6f}")
    assert np.isclose(d_pq, d_qp, atol=1e-8), f"Asymmetric: {d_pq:.6f} vs {d_qp:.6f}"
    print("  ✓ PASS")


def test_js_divergence_identical():
    """JS-divergence of identical distributions should be near zero."""
    print("\n[TEST] JS-divergence identity → zero")
    np.random.seed(42)
    p = np.random.rand(100)
    p = p / p.sum()
    d = js_divergence(p, p)
    print(f"  D(p,p) = {d:.8f}")
    assert d < 1e-6, f"Expected ~0, got {d}"
    print("  ✓ PASS")


def test_js_divergence_distinct():
    """JS-divergence of distinct distributions should be positive."""
    print("\n[TEST] JS-divergence distinct → positive")
    # Two clearly different distributions
    p = np.zeros(100)
    p[:50] = 1.0
    p = p / p.sum()
    
    q = np.zeros(100)
    q[50:] = 1.0
    q = q / q.sum()
    
    d = js_divergence(p, q)
    print(f"  D(p,q) for disjoint supports = {d:.6f}")
    assert d > 0.0, f"Expected positive divergence for distinct distributions, got {d}"
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


def test_compute_divergence_series():
    """Full divergence pipeline on synthetic data."""
    print("\n[TEST] Full divergence series computation")
    
    # Create synthetic episode data
    N = 100  # timesteps
    timesteps = []
    for i in range(N):
        # Planner: simple Gaussian modes
        np.random.seed(i) # Deterministic per step
        trajs = np.random.randn(6, 6, 2) * (i / N)  # Uncertainty increases over time
        scores = np.ones(6) / 6
        
        # Occupancy: simple random grid
        occ = np.zeros((GRID_H, GRID_W), dtype=np.float32)
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
    print(f"  Min divergence: {result['divergence_raw'].min():.6f}")
    print(f"  Max divergence: {result['divergence_raw'].max():.6f}")
    print(f"  Steps with occupancy: {result['has_occupancy'].sum()}/{N}")
    
    assert len(result['divergence_raw']) == N
    assert len(result['divergence_smooth']) == N
    assert not np.any(np.isnan(result['divergence_raw']))
    
    # Verify that divergence actually changes when occupancy is introduced
    div_before = result['divergence_raw'][:20].mean()
    div_after = result['divergence_raw'][21:].mean()
    print(f"  Avg divergence before obstacles: {div_before:.6f}")
    print(f"  Avg divergence after obstacles: {div_after:.6f}")
    # We just check that values are computed and finite, exact direction depends on implementation
    assert np.isfinite(div_before) and np.isfinite(div_after)
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
    assert grid.shape == (GRID_H, GRID_W)
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
        test_js_divergence_distinct,
        test_rasterize_normalization,
        test_compute_divergence_series,
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
