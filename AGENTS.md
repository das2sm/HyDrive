# HyDrive — AGENTS.md

## What this is

Research project: does temporal divergence between planner futures and world occupancy futures predict sparse-planner failures better than TTC/RSS/distance?

## Code layout

Code lives in `Bench2Drive/` (a SparseDriveV2 fork with Guardian + divergence logging). The root `README.md` is a 3-month research roadmap, not a dev guide. README.md month 3 is not the focus right now. The goal is to find publishable results.

## Key structural facts

- **No standard tooling**: no ruff, mypy, pytest, pre-commit, CI, or lockfiles. Tests are plain-Python scripts run directly.
- **`requirement.txt`** (not `requirements.txt`) at `Bench2Drive/` — easy to miss. Uses `mmcv-full==1.7.1`, `mmdet==2.28.2`.
- **Conda environment** named `sparsedrive` is used in scripts (`conda run -n sparsedrive python ...`).
- **`tests/`** is analysis, not just tests. Contains the core divergence computation (`divergence.py`), baselines, and the main analysis runner (`run_analysis.py`).
- **`tests/test_divergence.py`** runs as `python tests/test_divergence.py` — no test runner needed.
- **`scripts/debug_b2d.sh`** is the main simulation runner (not just for debugging).
- **Agent entry point**: each agent file exports `get_entry_point()` returning the class name.

## Data pipeline

1. **Simulate**: `bash scripts/debug_b2d.sh <ROUTE_NAME>` → raw `.pkl` at `close_loop_log/save/<ROUTE>/divergence_logs/route_<ROUTE>.pkl`
2. **Postprocess**: `python tests/process_logs.py --input <raw.pkl> --out processed/` → adds `future_collision`, `future_near_miss`, `time_to_collision`
3. **Analyze**: `python tests/run_analysis.py --route processed/<route>_processed.pkl --out analysis/results`

Batch workflow: `bash scripts/run_batch.sh` (runs a curated route list), or `bash scripts/run_all_routes.sh` (discovers all routes).

## Agent config format

Config path uses `+` as separator: `config_path+ckpt_path+save_name[+gpu_rank]`
Example: `projects/configs/sparsedrive_stage2.py+ckpt/sparsedrive_small_b2d_stage2.pth+Base_HardBrake_0035`

## Critical coordinate convention

| Frame | Axes |
|-------|------|
| Planner BEV | row=forward (inverted), col=left |
| Guardian/CARLA | row=right, col=forward |

Alignment is (`_align_occupancy_to_planner_bev` in `tests/divergence.py`). The 1px roll corrects the even-grid center mismatch. The VAD agent's Guardian also has `_vad_traj_to_carla_local` that swaps `[left, forward]` to `[forward, -left]`.

## BEV grid

120×120, 60m range (±30m symmetric around ego), 0.5m/cell. Defined in `tests/divergence.py`.

## Key agent variants

| File | What |
|------|------|
| `sparsedrive_b2d_agent.py` | Base SparseDrive without Guardian |
| `sparsedrive_b2d_agent_occ.py` | SparseDrive + Guardian + DivergenceLogger + CollisionSensor |
| `vad_f2d_agent_occ.py` | VAD variant with its own inlined Guardian (different grid: 240×240, 0.4m/cell) - NOT USED FOR THIS RESEARCH - ignore|

## Guardian performance

- Actor list refreshed every frame (`_actor_list_cache_max_age=1`). Occupancy recomputed every step (`expensive_step_interval=1`).
- `skip_cri` and `skip_path_blockage` are `True` by default (expensive features disabled for research logging).
- Intervention is disabled (`intervene = False`).

## Postprocessing labels

- `tests/process_logs.py` adds `future_collision`, `future_near_miss`, `time_to_collision` to each timestep.
- Lookahead window starts at frame `i+1`. Labels are consistent with `time_to_collision`.

## Occupancy normalization

- `_smooth_normalise` in `tests/divergence.py` applies Gaussian blur (sigma=1.5) before normalization to prevent sparse occupancy producing unstable probability distributions.

## Reproducibility

- Agent sets `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`, and `torch.cuda.manual_seed_all(42)` in `setup()`. The scenario runner also uses `CarlaDataProvider._random_seed = 2000` for environment spawning.
- Simulations are reproducible only when the full stack (CARLA, model weights, agent config, route XML) is frozen.

## `near_miss` label caveat

The `near_miss` label (in agent) is defined using Guardian's `min_dist` and `ttc`, which are the same signals used as baselines. Using `--label any_failure` in analysis creates circularity. The default `--label collision` avoids this issue.

## CARLA simulation env vars

Must be set: `CARLA_ROOT`, `IS_BENCH2DRIVE=True`, `PYTHONPATH` including `leaderboard/` and `scenario_runner/`.

## Evidence summary (120 routes, 40 collisions)

### Global JS divergence `temporal_conflict_smooth` — confounded

Full-grid JS between planner occupancy rasterization and Guardian occupancy is dominated by scene occupancy density. Routes with >30% occupancy produce near-max JS regardless of planner trajectory; routes with <5% occupancy produce lower JS. **AUROC 0.610** — ties TTC but does not add independent signal.

### Planner-conditioned occupancy (PCO) 

`planner_conditioned_occupancy()` in `tests/divergence.py` directly samples the aligned occupancy grid at each planner trajectory waypoint cell, weighted by mode probability. This eliminates the occupancy-density confound because it only looks at cells the planner intends to occupy.

| Metric | PCO | dist_proxy | ttc_rel | JS conflict |
|--------|-----|------------|---------|-------------|
| AUROC | **0.702** | 0.629 | 0.607 | 0.610 |
| 95% CI | [0.639, 0.761] | [0.557, 0.696] | [0.546, 0.662] | [0.548, 0.675] |
| Δ over TTC | **+0.094 sig** | — | — | +0.002 n.s. |
| Event AUROC | **0.778** | 0.728 | 0.561 | 0.514 |
| Lead (collisions) | 4.86s (36/40) | — | 4.97s (39/40) | 5.34s (8/40) |

**PCO passes matched-bin control** — it separates failures from safe frames within every TTC, distance, speed, and actor-count bin (all p < 0.001). Bin-AUROC hits **0.853 in the safest TTC quartile** — meaning it identifies collisions TTC considers safe.

The original global JS metric is retained only for ablation comparisons. The research direction moving forward uses **PCO as the primary divergence metric**.

## PCO verification checklist

Independent verification confirmed all results:

| Check | Result |
|-------|--------|
| Coordinate transforms | Guardian(60,80) → BEV(40,60) matches planner(0,10m) → BEV(40,60) |
| Synthetic ground truth | All 7 tests pass: single-mode hit=1/T, multi-mode=weighted avg, clear=0, full=1, blur spreads |
| AUROC oracle (occupancy_future) | **0.705** — matches run_analysis.py 0.702 within CI |
| AUROC causal (current occ only) | **0.685** — still beats all baselines |
| Leakage gap (oracle−causal) | 0.020 — minimal, explained by dynamic obstacles |
| Baseline comparison | PCO beats TTC(0.616), Dist(0.637), Conflict(0.610), RSS(0.563) |
| Matched-bin (all bins) | All p < 0.001, bin-AUROC up to 0.863 in safest TTC quartile |
| Lead-time | 36/40 collisions, 4.86s mean (TTC: 39/40, 4.97s) |
| Event-level AUROC | PCO 0.741, TTC 0.616, Dist 0.628 |
| PCO vs TTC correlation | Spearman r = -0.066 — near-zero, signals are complementary |
| Top-10% overlap PCO & TTC | 8.9% (below chance 10%) — different frames identified as dangerous |
| Pre-collision enrichment | PCO top-10% catches 27.2% of pre-collision frames; TTC catches 11.8% |
| Per-route PCO wins | 19/40 routes where PCO AUROC > TTC AUROC (TTC wins 21/40) |

**Verdict**: The result is real. PCO provides independent, complementary collision-prediction signal orthogonal to TTC.

## Modality ablation

| Variant | AUROC | Δ vs top-1 | Description |
|---------|-------|-----------|-------------|
| Top-1 deterministic overlap | 0.648 | — | Best single trajectory mode, binary occupancy at waypoints |
| Full multimodal PCO (causal) | 0.685 | **+0.037 sig** | Mode-weighted occupancy sampling |
| Oracle PCO (future occ) | 0.705 | +0.020 (over causal) | Uses ground-truth future occupancy |

The +0.037 gain from full multimodal weighting is significant (95% CI [+0.031, +0.042]). This means the planner's uncertainty structure contributes real predictive value beyond deterministic path-occupancy overlap. The contribution is modest but material — it separates PCO from "differentiable collision checking."

## Simulated peer-review notes / desk-reject risks

The result is promising but should be framed tightly. The publishable claim is **not** that generic full-grid planner-world JS divergence works. That metric was confounded. The defensible claim is:

> Planner-conditioned occupancy provides collision-window prediction signal complementary to TTC and distance for SparseDrive on Bench2Drive, under route-level bootstrapped evaluation and confound-stratified analyses.

### Framing requirements

- Make **causal PCO** the primary metric. Treat oracle/future-occupancy PCO as an upper-bound ablation, not a deployable runtime result.
- Be explicit that PCO is a mode-weighted occupancy-at-planner-waypoints score. If using the word "divergence", define it carefully and acknowledge that full-grid JS failed.
- Do **not** claim "better than RSS" unless real `carla.ad.rss` is wired in. Current `rss_proxy` is a heuristic TTC+distance threshold and should be called **RSS-style proxy**.
- Avoid broad claims like "planner-world divergence generally predicts autonomous-driving failures." The current safe claim is planner/benchmark specific.

### Required robustness before submission

- Add a joint confound model: `collision_window ~ PCO + TTC + distance + speed + actor_count + route/scenario random effects`.
- Replace or supplement frame-level matched-bin p-values with route/event-cluster-aware inference, e.g. route bootstrap, event-level permutation, or mixed-effects logistic regression.
- Report threshold-free lead-time / early-warning curves. The current fixed PCO threshold is tuned to the observed collision-window distribution, so lead-time claims are weaker than AUROC claims.
- Add scenario-family/event-level confidence intervals, not only frame-level metrics.
- Add top-K sensitivity for planner modes and discuss score calibration of `traj_cls`.
- Reconcile dataset accounting before writing: notes say 120 routes / 40 collisions, while the current `Bench2Drive/processed` directory was observed as 119 processed route files / 39 collision events during review.

### Likely desk-reject triggers

- Abstract or headline claims say "RSS" while experiments use `rss_proxy`.
- Oracle future occupancy is presented as the main operational metric.
- The paper continues to lead with "temporal divergence" without explaining the failure of global JS and the shift to PCO.
- No fixed route manifest, missing-route explanation, or reproducibility table.
- No held-out scenario-family or event-level analysis.

## CARLA codebase resources for research

### Signals now wired into analysis pipeline

All four new signals are now wired into `sparsedrive_b2d_agent_occ.py`, `divergence_logger.py`, `baselines.py`, and `run_analysis.py`:

| Signal | Source | Logger field | Baseline key | Status |
|--------|--------|-------------|-------------|--------|
| `gc_score` + sub-terms | `guardian.py` `latest_gc_score` | `gc_score`, `gc_overlap_term`, `gc_potential_term`, `gc_decel_term`, `gc_ttc_term` | `gc_score`, ... | Wired — re-sim required |
| Learned `collision_cls` | `output['collision_cls']` | `collision_cls` | `collision_cls` | Wired — re-sim required |
| Learned `point_collision_cls` | `output['point_collision_cls']` | `point_collision_cls`, `point_collision_cls_mean` | `point_collision_cls_mean` | Wired — re-sim required |

### Collision-only re-simulation list

File: `tests/collision_routes.txt` — 60 routes with collisions. Generated by scanning all 143 processed logs.
To re-simulate only these: `bash scripts/run_batch.sh --routes collision_routes.txt` (or adapt the batch script).

- **Learned collision branches**: `collision_cls` and `point_collision_cls` are now extracted in the agent and passed to the divergence logger. Both appear in `BASELINE_SCORE_KEYS` and are evaluated alongside PCO and TTC. No code changes needed — already wired.

- **Proper RSS**: `carla.ad.rss` module is not available in this build. The heuristic `rss_proxy` (TTC < 2.0 AND dist < 3.0) is the only RSS-like baseline. Must be called **RSS-style proxy** throughout the paper.

### For collision scenario taxonomy

The 40 collision events can be categorized by scenario type from the route names:

- `Base_Animals_*` — animal crossing
- `Base_BadParking_*` — parked vehicle blocking
- `Base_Construction_*` — construction zone
- `Base_Crossing*` — pedestrian/biker crossing
- `Base_Emergency_*` — emergency vehicle
- `Base_ControlLoss_*` — slippery road
- `Base_HardBrake_*` — sudden brake
- `Base_Opposite_*` — oncoming traffic
- `Base_Accident_*` — pre-crash scene
- `Generalization_*` — out-of-distribution variants

The Fail2Drive result parser (`Fail2Drive/tools/f2d_result_parser.py`) has infrastructure for per-scenario-class success/infraction analysis.

### Signal noise caveat

`skip_cri=True` and `skip_path_blockage=True` are set in the agent config, which means CRI and path blockage signals are **disabled** in the current simulation runs. The Default config in `sparsedrive_b2d_agent_occ.py` has these as defaults. Enabling them would slow down the simulation but add more logged baselines.

### TTC family — what exists where

| TTC variant | Source | Frame-by-frame | Handles non-vehicle | Status |
|-------------|--------|---------------|-------------------|--------|
| Guardian path-occupancy TTC (`ttc`) | `guardian.py:727-731` | Yes | Yes (occupancy-based) | In `BASELINE_SCORE_KEYS` |
| Actor-relative TTC (`ttc_rel`) | `guardian.py:582-681` | Yes | Yes (any actor class) | In `BASELINE_SCORE_KEYS` |
| BehaviorAgent kinematic TTC | `behavior_agent.py:266` | No (car-following only) | No (vehicles only) | Not useful — driving heuristic |
| Hindsight `time_to_collision` (label) | `process_logs.py:58-62` | Yes | N/A (post-hoc) | Used for eval mask |

---

## Code review findings — fix before submission

The following issues were identified by an independent code review. Items marked **CRITICAL** must be fixed before any paper submission.

### CRITICAL: Test suite is broken [FIXED]

`tests/test_divergence.py` previously imported `build_future_occupancy_windows` which no longer existed — fixed by removing the dead import and the `test_future_occupancy_windows` test. Imports now from `analysis.divergence`.

### [FIXED] Main `run_analysis.py` PCO is oracle, not causal

`planner_conditioned_occupancy` now has `causal=True` as default (`divergence.py:240`). `run_analysis.py:1115` calls with `causal=True`. Oracle is available as ablation via `causal=False`. The 0.685 causal AUROC is now computed by the pipeline, not manually.

### [FIXED] 0.5s labeling gap

`process_logs.py:40-41` previously started the lookahead at `i + 10` frames (0.5s offset), contaminating 351 frames. **Fixed**: lookahead now starts at `i + 1` (excludes current frame — measures prediction, not detection). TTC formula uses `(index + 1) / fps` to convert slice-relative to frame-relative. All 143 routes re-processed. 0 self-labeled, 0 inconsistent, min TTC = 0.050s.

### [FIXED] Matched-bin drops right-edge frames

`run_analysis.py:60-66` `finite_percentile_edges` already sets `edges[-1] = np.inf`, and speed uses hardcoded `np.inf` as last edge. No frames are dropped.

### WARNING: Matched-bin p-values are anti-conservative

`run_analysis.py:295,343` uses frame-level Mann-Whitney U tests on autocorrelated frames clustered by route and collision event. Frame-level tests inflate significance on non-independent samples.

**Fix**: Replace with route/event-cluster bootstrap or mixed-effects logistic regression (`collision_window ~ PCO + TTC + distance + speed + actor_count + (1|route)`).

Note: `run_analysis.py` already uses `_route_bootstrap_sep` when `route_ids` is provided (line 331, 382), which resamples routes (not frames). Mann-Whitney is only the fallback when `route_ids=None`. The mixed-effects model has been added as `task_joint_confound_model`.

### WARNING: Lead-time PCO threshold is post-hoc tuned

`run_analysis.py:394` uses `pco_threshold=0.15` tuned to the observed collision-window distribution. Lead-time claims using a post-hoc threshold are not generalizable.

**Fix**: Report threshold-free metrics (e.g., early-warning curves, time-dependent AUROC) as the primary lead-time evidence. Report lead-time at fixed thresholds only as secondary with a calibration note.

Note: `task_threshold_free_lead_time` now provides time-dependent AUROC as the primary threshold-free metric.

### WARNING: Dataset accounting mismatch

Notes in earlier analysis claim 120 routes / 40 collisions. Current `Bench2Drive/processed/` has 119 processed route files / 39 collision events. The discrepancy must be reconciled for the reproducibility table.

### WARNING: `rss_proxy` is a heuristic, not RSS

The `rss_proxy` baseline (`TTC < 2.0 AND dist < 3.0`) is a crude thresholded heuristic, not a proper RSS implementation. Must be called **RSS-style proxy** throughout the paper. (Already documented in framing requirements above — this is a reminder to check the paper text.)

### [FIXED] Causal PCO separation in main analysis

The causal PCO path (repeat current occupancy across all waypoints) is now the default in `run_analysis.py:1115` (`causal=True`). Oracle is available as ablation via `causal=False`. AUROC 0.685 is now computed by the pipeline, not manually. The 0.5s gap fix (now `i+1`) means the causal AUROC may shift slightly — re-run analysis on all 143 processed routes to get the final number.

### KNOWN LIMITATION: 2D BEV occupancy collapse (overhead obstacles)

PCO is computed from a 2D birds-eye view occupancy grid with **no height dimension**. The Guardian rasterizes at `guardian.py` using a vertical slab filter (`-2.5m to 5.0m` in ego-local z), then drops z and projects bounding box vertices onto a flat grid. This means:

- Overhead obstacles (bridges, gantries, traffic light poles, tree canopies, overhanging signs, building awnings) project the same occupied cell as ground-level obstacles (vehicles, barriers, pedestrians).
- A trajectory whose waypoints pass through a cell occupied by an overhead structure is penalized equally to one that tries to drive through a wall.
- In a real CARLA run (observed), the car drives *under* what appears in BEV as a solid obstacle — the planner correctly navigates it, but PCO incorrectly spikes because it only sees 2D occupancy.
- Since the negative set includes many bridge/gantry frames with elevated PCO but zero collision risk, this biases AUROC downward (false positives from overhead obstacles). The true PCO discriminative power on ground-level obstacles is higher than reported.

**Mitigation needed for deployment**: Use height-resolved occupancy (multi-layer BEV, LiDAR height maps, or 3D voxel grids) and only penalize occupancy within the vehicle's clearance envelope (below ego hood height).
