# Running the closed-loop campaign

Complete `SETUP.md` before using this guide. If this is a new shell, restore
the same workspace paths chosen during setup and activate the `sparsedrive`
environment:

```bash
export HYDRIVE_WORKSPACE="$HOME/hydrive-reproduction"
export HYDRIVE="$HYDRIVE_WORKSPACE/HyDrive"
export RUNTIME="$HYDRIVE_WORKSPACE/hydrive-runtime"
export OUTPUT="$HYDRIVE_WORKSPACE/campaign-output"

if [ -n "${VIRTUAL_ENV:-}" ]; then deactivate; fi
conda activate sparsedrive
cd "$RUNTIME"
```

Do not run the campaign from Conda's `base` environment or from a Python
virtual environment. If you selected a different workspace in `SETUP.md`, use
that same absolute `HYDRIVE_WORKSPACE` path here.

## Required patched runtime

This guide starts after the complete runtime installation in `SETUP.md`. In
particular, both repository patches must already be applied and committed:

- `patches/sparsedrive_planner_outputs.patch` exposes the proposal fields used
  by the intervention; and
- `patches/fail2drive_campaign_controls.patch` installs the strict evaluator
  and scenario-initialization controls.

Verify both installations before creating a schedule:

```bash
grep -q -- 'traj_selected_mode_index' \
  projects/mmdet3d_plugin/models/motion/decoder.py &&
grep -q -- 'traj_cls_logits_post_rescore' \
  projects/mmdet3d_plugin/models/motion/decoder.py &&
grep -q -- '--no-resume' \
  Fail2Drive/fail2drive_leaderboard/leaderboard/leaderboard_evaluator.py &&
grep -q -- 'HYDRIVE_SCENARIO_INIT_OK' \
  Fail2Drive/fail2drive_leaderboard/leaderboard/scenarios/scenario_manager.py &&
echo "Required runtime patches verified"
```

If any `grep` command fails, return to the SparseDriveV2 and Fail2Drive steps
in `SETUP.md`. Apply and commit the missing patch before generating a campaign
lock. Preflight intentionally rejects an unpatched runtime.

## What can be reproduced

The exact paper estimates and confidence intervals are reproduced from the
released route-level outcomes:

```bash
python "$HYDRIVE/analysis/verify_artifact.py" --full-bootstrap
```

The commands below reconstruct the experimental protocol and collect fresh
CARLA executions. CARLA is not bitwise deterministic, so a new campaign is an
independent replication rather than a promise of identical route outcomes.

## 1. Set the campaign paths

```bash
cd "$RUNTIME"
PROTOCOL="$RUNTIME/campaigns/multiseed_v1/protocol.json"
PYTHON="$(python -c 'import sys; print(sys.executable)')"
```

`OUTPUT` must equal the `artifact_root` value in the protocol. `PYTHON` must
equal `python_executable`. Confirm that `repo_root` points to the current
`hydrive-runtime/` directory.

The runtime and nested Fail2Drive worktrees must be clean before a schedule is
created:

```bash
git status --porcelain
git -C Fail2Drive status --porcelain
```

Both commands must produce no output.

## 2. Generate the pilot schedule

The infrastructure pilot contains one development route, three evaluation
seeds, and all three arms: nine jobs total. Pilot outcomes are not included in
the paper analysis.

```bash
"$PYTHON" scripts/multiseed/prepare_campaign.py \
  --protocol "$PROTOCOL" \
  --output-dir "$OUTPUT/pilot/spec" \
  --pilot
```

This creates `campaign.lock.json` and `schedule.csv`. Do not edit either file.

## 3. Preflight, run, and finalize the pilot

Run these commands one at a time. Do not start the campaign unless preflight
prints `Preflight passed`.

```bash
"$PYTHON" scripts/multiseed/preflight.py \
  --protocol "$PROTOCOL" \
  --lock "$OUTPUT/pilot/spec/campaign.lock.json" \
  --schedule "$OUTPUT/pilot/spec/schedule.csv"
```

After preflight passes, run the pilot:

```bash
"$PYTHON" scripts/multiseed/run_campaign.py \
  --protocol "$PROTOCOL" \
  --lock "$OUTPUT/pilot/spec/campaign.lock.json" \
  --schedule "$OUTPUT/pilot/spec/schedule.csv" \
  --pilot
```

After all nine jobs are accepted, finalize it:

```bash
"$PYTHON" scripts/multiseed/finalize_campaign.py \
  --protocol "$PROTOCOL" \
  --lock "$OUTPUT/pilot/spec/campaign.lock.json" \
  --schedule "$OUTPUT/pilot/spec/schedule.csv"
```

`preflight.py` checks the lock, GPU/CARLA imports, output storage, CARLA RPC,
streaming and Traffic Manager ports, and the CARLA launcher. Finalization
requires all nine pilot jobs and writes `$OUTPUT/pilot/pilot_report.json`.
Proceed only when that report has `"passed": true`.

The pilot gates technical retries and projects full-campaign runtime and
storage. It does not select a method based on pilot driving outcomes.

## 4. Generate the full schedule

```bash
"$PYTHON" scripts/multiseed/prepare_campaign.py \
  --protocol "$PROTOCOL" \
  --output-dir "$OUTPUT/spec"

"$PYTHON" scripts/multiseed/preflight.py \
  --protocol "$PROTOCOL" \
  --lock "$OUTPUT/spec/campaign.lock.json" \
  --schedule "$OUTPUT/spec/schedule.csv"
```

The deterministic schedule contains 1,665 jobs:

- 185 Fail2Drive routes;
- evaluation seeds 1, 2, and 3; and
- baseline, current-frame filtering, and temporal filtering.

Within each route-seed block, all six arm orders are balanced across the
schedule. The same route-specific Traffic Manager seed is used for all three
arms in a block.

## 5. Run the full campaign

```bash
"$PYTHON" scripts/multiseed/run_campaign.py \
  --protocol "$PROTOCOL" \
  --lock "$OUTPUT/spec/campaign.lock.json" \
  --schedule "$OUTPUT/spec/schedule.csv"
```

Each job launches a new Fail2Drive CARLA server and evaluator, writes to an
immutable attempt directory, and terminates the process group before the next
job. The runner accepts official driving outcomes such as collision, timeout,
blockage, and deviation. Only technical failures are eligible for a later
retry, up to the protocol's three-attempt limit.

### Resume after interruption

Run the same command again with the same protocol, lock, and schedule. The
runner skips jobs with accepted final receipts. If an interruption left an
attempt directory without a receipt, the isolated-job runner preserves it
under an `.orphaned-<timestamp>` name before starting the next attempt.

Do not delete attempts, regenerate the schedule, or create a new lock while
resuming the same campaign.

### Cleanup failure

If the runner reports `process_cleanup_failed`, it stops before launching the
next job. Confirm that the CARLA and evaluator processes from that job are no
longer running, then invoke the same campaign command again. The failed
technical attempt remains recorded and the job is retried according to the
protocol.

## 6. Finalize the collected data

```bash
"$PYTHON" scripts/multiseed/finalize_campaign.py \
  --protocol "$PROTOCOL" \
  --lock "$OUTPUT/spec/campaign.lock.json" \
  --schedule "$OUTPUT/spec/schedule.csv"
```

Finalization validates accepted receipts and compact-log provenance, records
missing or quarantined mechanism logs, identifies complete paired route
cohorts, and writes `$OUTPUT/data.lock.json`. The analysis refuses to run
without this file.

## 7. Extract outcomes and estimate effects

```bash
"$PYTHON" analysis/multiseed_campaign.py \
  --protocol "$PROTOCOL" \
  --schedule "$OUTPUT/spec/schedule.csv" \
  --output-dir "$OUTPUT/analysis"
```

The analysis writes:

- `job_outcomes.csv`: one row per accepted route-arm-seed job;
- `results.json`: arm summaries, paired contrasts, seed-level estimates, and
  the primary interpretation;
- `contrast_table.csv`: effect estimates and 95% intervals; and
- `results_macros.tex`: paper-ready primary-result values.

It averages the three seeds within each route, weights routes equally, and
uses 20,000 route-family cluster-bootstrap resamples for uncertainty.

## Non-negotiable campaign rules

- Never rerun a job because of its driving outcome or score.
- Never replace the first accepted official outcome.
- Keep interrupted and invalid technical attempts.
- Never edit a schedule or lock after execution begins.
- Keep the matched route-specific Traffic Manager seed across arms.
- Do not run another CARLA workload on the same ports during the campaign.
- Do not inspect partial aggregate results to alter the running protocol.
