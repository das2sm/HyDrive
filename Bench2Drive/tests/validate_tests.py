"""
Validate test suite: inject each bug, confirm expected test(s) fail.
"""
import subprocess, sys, re, os

WORKDIR = "/home/ace428/Soham/HyDrive/Bench2Drive/tests"
SCRIPT = "test_divergence.py"
ORIG = "divergence.py.orig"
TARGET = "divergence.py"


def read_source():
    with open(os.path.join(WORKDIR, TARGET)) as f:
        return f.read()


def write_source(text):
    with open(os.path.join(WORKDIR, TARGET), 'w') as f:
        f.write(text)


def restore():
    subprocess.run(["cp", os.path.join(WORKDIR, ORIG),
                    os.path.join(WORKDIR, TARGET)])


def run_tests():
    r = subprocess.run(
        ["conda", "run", "-n", "sparsedrive", "python", SCRIPT],
        capture_output=True, text=True, timeout=120, cwd=WORKDIR)
    return r.stdout + r.stderr


BUGS = [
    # (label, expected_failing_test_substring, (old, new) replacements)
    # BEV coords: swap row/col formula
    ("BEV: swapped row/col",
     "test_bev_coord_transform",
     [
         ("col = (trajs[..., 0] + GRID_RANGE) / CELL_SIZE",
          "col = (trajs[..., 1] + GRID_RANGE) / CELL_SIZE"),
         ("row = (GRID_RANGE - trajs[..., 1]) / CELL_SIZE",
          "row = (GRID_RANGE - trajs[..., 0]) / CELL_SIZE"),
     ]),

    # BEV coords: remove clipping
    ("BEV: removed clipping",
     "test_bev_coord_transform",
     [
         ("row_clipped = np.clip(row, 0, GRID_H - 1)", "row_clipped = row"),
         ("col_clipped = np.clip(col, 0, GRID_W - 1)", "col_clipped = col"),
     ]),

    # Yaw: swap arctan2 args
    ("Yaw: swapped atan2 args",
     "test_yaw",
     [("np.arctan2(delta[1], -delta[0])", "np.arctan2(delta[0], delta[1])")]),

    # JS: wrong KL combination (not symmetric)
    ("JS: asymmetric KL",
     "test_js_divergence",
     [("jsd = 0.5 * (kl_pm + kl_qm)", "jsd = kl_pm")]),

    # JS: broken empty-input guard (divides by zero)
    ("JS: broken empty guard",
     "test_js_divergence",
     [("if p_sum < eps or q_sum < eps:\n        # If one is empty, they are maximally different (disjoint)\n        return np.log(2.0)", "if False: pass")]),

    # Conflict: remove NaN guard
    ("Conflict: removed NaN guard",
     "test_conflict_score",
     [("if not np.isfinite(js_value):\n        return np.nan", "")]),

    # Rasterize: skip footprint placement (grid stays all-zero)
    ("Rasterize: skip footprint",
     "test_rasterize_planner",
     [("_add_footprint(probability_grid, row_center, col_center, yaw, weight, sigma_t)",
       "pass  # bug")]),

    # Rasterize: skip normalization
    ("Rasterize: skip normalize",
     "test_rasterize_planner",
     [("return _normalise_grid(probability_grid)", "return probability_grid")]),

    # Alignment: remove 1px shift
    ("Align: removed shift",
     "test_coordinate_alignment",
     [("result[1:, 1:] = aligned[:-1, :-1]", "result = aligned")]),

    # Alignment: remove transpose+flip
    ("Align: removed transform",
     "test_coordinate_alignment",
     [("aligned = occ.T[::-1, ::-1]", "aligned = occ")]),

    # PCO: wrong cell (off-by-one)
    ("PCO: off-by-one cell",
     "test_pco",
     [("mode_vals[k] = aligned[r, c]",
       "mode_vals[k] = aligned[max(0, r-1), max(0, c-1)]")]),

    # PCO: skip alignment
    ("PCO: skip align",
     "test_pco",
     [("aligned = _align_occupancy_to_planner_bev(occ_future[h]).astype(np.float64)",
       "aligned = np.asarray(occ_future[h], dtype=np.float64)")]),

    # PCO: always causal (oracle branch dead)
    ("PCO: always causal",
     "test_pco",
     [("if causal:", "if True:  # bug")]),

    # PCO: h_max = 0 (no waypoints processed)
    ("PCO: h_max=0",
     "test_pco",
     [("h_max = min(T, len(occ_future))", "h_max = 0")]),

    # DivSeries: skip alignment in loop (uses raw occ)
    ("DivSeries: skip align in loop",
     "test_compute_divergence_series",
     [("q_grid = _align_occupancy_to_planner_bev(occ_future[h]).astype(np.float64)",
       "q_grid = occ_future[h].astype(np.float64)")]),

    # DivSeries: broken occ_has_signal (always False)
    ("DivSeries: occ signal always False",
     "test_compute_divergence_series",
     [("occ_has_signal = np.array([occ.max() > 1e-6 for occ in occ_future], dtype=bool)",
       "occ_has_signal = np.zeros(len(occ_future), dtype=bool)")]),
]

print("=" * 70)
print("TEST VALIDATION: Injecting bugs, verifying tests fail")
print("=" * 70)

passed = 0
failed = 0
skip_detail = []

for label, expected_test, replacements in BUGS:
    restore()
    src = read_source()

    for old, new in replacements:
        count = src.count(old)
        if count == 0:
            print(f"  SKIP {label}: pattern not found in source")
            skip_detail.append(label)
            break
        src = src.replace(old, new, 1)

    # Write buggy source
    write_source(src)
    output = run_tests()
    restore()

    # Check if the expected test failed (listed in failure summary)
    fail_pattern = f"✗ {re.escape(expected_test)}"
    if re.search(fail_pattern, output):
        print(f"  \u2713 {label:40s} -> {expected_test} FAILS")
        passed += 1
    else:
        print(f"  \u2717 {label:40s} -> {expected_test} DID NOT FAIL")
        for line in output.split("\n"):
            if "FAIL" in line or "ERROR" in line:
                print(f"      {line}")
        # Show what tests DID fail
        failed_lines = [l for l in output.split("\n") if "FAIL" in l or "ERROR" in l or "✗" in l]
        if failed_lines:
            print(f"      Actual failures: {failed_lines}")
        failed += 1

print(f"\n{'='*70}")
print(f"Results: {passed}/{passed+failed} bugs caught")
if failed > 0:
    print(f"Uncaught: {failed}")
    sys.exit(1)
else:
    print(f"All {passed} bugs caught \u2713")
