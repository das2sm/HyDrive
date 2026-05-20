# HyDrive — AGENTS.md

## What this is

Research project: does temporal divergence between planner futures and world occupancy futures predict sparse-planner failures better than TTC/RSS/distance?

## Code layout

Code lives in `Bench2Drive/` (a SparseDriveV2 fork with Guardian + divergence logging). The root `README.md` is a 3-month research roadmap, not a dev guide.

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

Alignment is `np.roll(occ.T[::-1, ::-1], shift=1, axis=(0,1))` (`_align_occupancy_to_planner_bev` in `tests/divergence.py:289`). The 1px roll corrects the even-grid center mismatch. The VAD agent's Guardian also has `_vad_traj_to_carla_local` that swaps `[left, forward]` to `[forward, -left]`.

## BEV grid

120×120, 60m range (±30m symmetric around ego), 0.5m/cell. Defined in `tests/divergence.py:16-19`.

## Key agent variants

| File | What |
|------|------|
| `sparsedrive_b2d_agent.py` | Base SparseDrive without Guardian |
| `sparsedrive_b2d_agent_occ.py` | SparseDrive + Guardian + DivergenceLogger + CollisionSensor |
| `vad_f2d_agent_occ.py` | VAD variant with its own inlined Guardian (different grid: 240×240, 0.4m/cell) |

## Guardian performance

- Actor list refreshed every frame (`_actor_list_cache_max_age=1`). Occupancy recomputed every step (`expensive_step_interval=1`).
- `skip_cri` and `skip_path_blockage` are `True` by default (expensive features disabled for research logging).
- Intervention is disabled (`intervene = False`).

## Postprocessing labels

- `tests/process_logs.py` (formerly `tools/postprocess_labels.py`) adds `future_collision`, `future_near_miss`, `time_to_collision` to each timestep.
- Uses `i+10` lookahead start (0.5s offset) to align with the planner's first waypoint. `time_to_collision` is measured from frame `i`, not from window start.

## Occupancy normalization

- `_normalise_occupancy_grid` in `tests/divergence.py` applies Gaussian blur (sigma=1.5) before normalization to prevent sparse occupancy producing unstable probability distributions.

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

### Planner-conditioned occupancy (PCO) — breakthrough

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
