# HyDrive — AGENTS.md

## Goal
Prove that planner-conditioned occupancy (PCO) predicts SparseDrive collisions better than TTC/distance baselines on the same occupancy grid.

## Code layout
- `leaderboard/team_code/guardian.py` — occupancy grid, TTC (closing-speed aware), conflict mask, min_dist
- `leaderboard/team_code/sparsedrive_b2d_agent_occ.py` — agent + Guardian + DivergenceLogger
- `leaderboard/team_code/divergence_logger.py` — per-timestep signal logging
- `tests/divergence.py` — PCO, JS divergence, trajectory dispersion
- `tests/baselines.py` — TTC risk, dist risk, collision_cls, point_collision_cls_mean
- `tests/run_analysis.py` — AUROC, matched-bin, lead-time, joint model, event-level
- `tests/process_logs.py` — hindsight labels (future_collision, time_to_collision)
- `scripts/debug_b2d.sh` — simulation entry point
- Agent config: `config_path+ckpt_path+save_name[+gpu_rank]`

## Data pipeline
1. **Simulate**: `bash scripts/debug_b2d.sh <ROUTE_NAME>` → raw `.pkl` at `close_loop_log/save/<ROUTE>/divergence_logs/route_<ROUTE>.pkl`
2. **Postprocess**: `python tests/process_logs.py --input <raw.pkl> --out processed/`
3. **Analyze**: `python tests/run_analysis.py --route processed/<route>_processed.pkl --out analysis/results`

Batch: `bash scripts/run_all_routes.sh`. Collision-only routes in `tests/collision_routes.txt`.

## Critical conventions
- **Planner BEV**: row=forward (inverted), col=left
- **Guardian/CARLA**: row=right, col=forward
- Alignment: `_align_occupancy_to_planner_bev` in `divergence.py`
- **BEV grid**: 120×120, 60m range (±30m), 0.5m/cell
- **Conda env**: `sparsedrive`
- **Conda run**: `conda run -n sparsedrive python ...`
- **Requirement**: `Bench2Drive/requirement.txt` (not requirements.txt)

## Baseline set
| Baseline | Source | What it measures |
|----------|--------|-----------------|
| `temporal_conflict_smooth` | `divergence.py` | Full-grid JS divergence (planner vs Guardian occupancy) |
| `planner_cond_occ` (PCO) | `divergence.py` | Occupancy fraction at planner waypoints (13×13 footprint) |
| `traj_dispersion` | `divergence.py` | Mean per-waypoint variance across K trajectory modes |
| `ttc_risk` | `guardian.py` | Closing-speed-aware TTC: per-actor min of `dist/closing` in forward cone (60° half-angle) + swept path; falls back to `min_dist/ego_speed` when no dynamic actor overlaps path |
| `dist_risk` | `guardian.py` | 1/(1+min_dist) from forward-cone filtered conflict mask |
| `collision_cls` | SparseDriveV2 head | Learned binary collision classifier (⚠ all NaN — needs investigation) |
| `point_collision_cls_mean` | SparseDriveV2 head | Mean learned per-waypoint collision score (⚠ all NaN) |

### TTC details
- **Closing-speed** = `max(0, -rel_vel · unit_toward)` for each actor in the forward cone that overlaps the swept path (`conflict_mask`).
- **`found_actor_on_path` flag**: if any dynamic actor is on the path, use closing TTC (safe car-following → 99.0). Only fall back to `min_dist/ego_speed` when no dynamic actor exists (pure static obstacles).
- **Cone**: 60° half-angle, defined by `self.ttc_half_angle_deg`. Applied identically to both `min_dist` and per-actor TTC.

## Current state
- Closing-speed TTC implemented and verified on `Generalization_FullyBlocked_1033`: TTC AUROC=0.914, PCO AUROC=0.882
- PCO lead time 6.0s vs TTC 1.3s on that route
- Joint model PCO coef +5.48 with TTC present, full model AUROC 0.944
- PCO passes matched-bin control within TTC/dist/speed/actor-count bins (even splits)
- Hyperparameters: `ttc_half_angle_deg=60`, `ttc_horizon=5`, `smoothing_window=3`, `dist_horizon=30`, `dist_tau=10`, `ttc_tau=1.5`
- Static BBS from CARLA: `Walls, Fences, GuardRail, TrafficLight, TrafficSigns, Poles, Buildings` (all included — phantom not yet resolved)

## Prioritized TODO

### Need-to-know before batch re-sim
1. **Investigate collision_cls NaN** — Check if SparseDriveV2 checkpoint has a `collision_cls` head. If not, remove from baselines.
2. **Investigate phantom obstacle** — Re-sim a route with visualization to identify which obstacles fill the grid. Don't blindly remove labels.

### Then batch re-sim 60 collision routes
3. Batch re-sim with all fixes → run analysis → get real PCO vs TTC numbers
4. Route bootstrap for CIs

### Paper-blocking gaps
5. Event-level CIs (need route/event random effects, need ≥2 events)
6. Scenario taxonomy (route-name-based by class: Animals, HardBrake, Opposite, etc.)
7. Threshold-free lead-time curves

### Tier-bump (pick one)
8. VAD integration — show PCO generalizes beyond SparseDrive
9. Intervention — show PCO prevents, not just predicts (month 3 work)

## Known issues
- **collision_cls still all NaN** — the learned head either doesn't exist in the checkpoint or isn't extracted correctly
- **Phantom obstacle** — some static BBS (possibly Buildings/Walls) saturate the grid; happens only on certain routes
- **2D BEV collapse**: PCO penalizes overhead obstacles (bridges, gantries, tree canopies) equally to ground-level — biases AUROC downward. Need height-resolved occupancy for deployment.
- **near_miss label circularity**: uses `min_dist`/`ttc` from Guardian — don't use `--label any_failure` in analysis

## Future work (not started)
- **VAD**: untested. Agent file exists (`vad_f2d_agent_occ.py`) with inlined Guardian (240×240, 0.4m/cell). Needs coordinate alignment port, divergence logger wiring, and re-sim.
- **Intervention**: `intervene=False` currently. Month 3 work.

## Env vars
`CARLA_ROOT`, `IS_BENCH2DRIVE=True`, `PYTHONPATH` including `leaderboard/` and `scenario_runner/`.
