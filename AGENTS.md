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
2. **Postprocess**: `python tools/postprocess_labels.py --input <raw.pkl> --output processed/<route>_processed.pkl` → adds `future_collision`, `time_to_collision`
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

- Actor list cached for 3 frames, occupancy recomputed every `expensive_step_interval=2` steps.
- `skip_cri` and `skip_path_blockage` are `True` by default (expensive features disabled for research logging).
- Intervention is disabled (`intervene = False`).

## Postprocessing labels

- Only `tools/postprocess_labels.py` is used (not `tests/process_logs.py`). Both now use `i+1` lookahead start (future-only, excludes current frame).

## Occupancy normalization

- `_normalise_occupancy_grid` in `tests/divergence.py` applies Gaussian blur (sigma=1.5) before normalization to prevent sparse occupancy producing unstable probability distributions.

## CARLA simulation env vars

Must be set: `CARLA_ROOT`, `IS_BENCH2DRIVE=True`, `PYTHONPATH` including `leaderboard/` and `scenario_runner/`.
