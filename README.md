# Goal

Show that temporal divergence between planner futures and world occupancy futures predicts sparse-planner failures better than TTC/RSS/distance baselines under controlled confounds.

No system-building beyond what is needed to validate the signal.

---

# 3-Month Roadmap: Distributional Planner–World Divergence

---

# Month 1 — Build the Measurement System (No claims, no interventions)

## Goal

Create a reliable, bias-controlled dataset of:

* planner future distributions
* world occupancy future distributions
* failure outcomes

This month is purely about instrumentation correctness.

---

## 1. Freeze experimental stack

Lock:

* one sparse planner (e.g., SparseDrive / VAD variant)
* one simulator setup (CARLA / Bench2Drive)
* one occupancy source (oracle or learned, but fixed for entire study)
* one evaluation split (nominal + OOD + occlusion)

No changes later.

---

## 2. Implement planner distribution extraction

Convert deterministic outputs into:

* K trajectory samples OR
* multi-modal prediction heads OR
* ensemble rollouts

Store:

Pπ(x_{t:t+H})

Minimum requirement:

* at least 5–10 plausible futures per timestep

---

## 3. Implement world occupancy evolution

Create time-indexed occupancy:

* BEV grid over horizon H
* probabilistic occupancy (not binary if possible)

Store:

Pw(x_{t:t+H})

---

## 4. Synchronize time horizon

Define:

* H = 1–3 seconds
* fixed timestep resolution

Ensure both distributions evolve on the same temporal grid.

---

## 5. Define outcome labels

At each timestep:

* safe
* near-miss
* failure (collision or unavoidable intervention)

Important:

* labels are future-conditioned, not reactive

---

## Month 1 Deliverable

A dataset containing:

* planner trajectory distributions
* occupancy distributions over time
* labeled outcomes
* synchronized temporal alignment

No metrics yet.

---

# Month 2 — Prove Predictive Value (Core scientific result)

## Goal

Show divergence adds information beyond TTC/RSS/distance under strict controls.

This is the paper-defining month.

---

## 1. Compute baseline signals

For every timestep:

* TTC (Responsibility-Sensitive Safety)
* minimum distance to occupancy
* RSS violation indicator (Responsibility-Sensitive Safety)
* divergence signal D(t)

---

## 2. Define divergence functional

Use a consistent metric (fixed for all experiments):

* Wasserstein distance OR
* JS divergence OR
* set-based overlap distance

Apply over:

D(t) = D(Pπ(x_{t:t+H}), Pw(x_{t:t+H}))

---

## 3. Main task: early failure prediction

Predict failure within 1–3 seconds.

Report:

* AUROC
* AUPRC
* calibration curve (optional but strong)

---

## 4. Critical experiment: matched-bin separation

Bin by:

* TTC range
* distance range
* speed range
* occlusion level

Then test:

Does divergence still separate failure vs non-failure within each bin?

This is the anti-TTC-collapse test.

---

## 5. Lead-time analysis

Show:

* divergence rises earlier than TTC/RSS triggers

Measure:

* time-to-signal vs time-to-failure

Plot:

* divergence lead-time distribution

---

## 6. Ablation study (mandatory)

Remove:

* multi-modal planner sampling
* temporal aggregation
* uncertainty in occupancy (if used)

Show:

* predictive performance degrades

This demonstrates structural necessity.

---

## Month 2 Deliverable

A validated result:

divergence provides statistically significant predictive information beyond TTC/RSS under controlled confounds and matched conditions.

---

# Month 3 — Minimal Intervention Validation (Secondary evidence)

## Goal

Show the signal is usable, not just predictive.

This is not the core contribution.

---

## 1. Three-system comparison only

* baseline planner (no safety layer)
* TTC / AEB braking (Responsibility-Sensitive Safety baseline)
* divergence-triggered braking

No steering. No replanning.

---

## 2. Intervention logic

Trigger braking based on:

* threshold on D(t)
* optionally smoothed over time

No additional heuristics.

---

## 3. Metrics

### Safety

* collision rate
* near-miss rate

### Utility

* route completion
* average speed

### Comfort

* jerk
* braking frequency

---

## 4. Key result

Show:

divergence-based intervention improves safety without collapsing utility compared to TTC/AEB.

Not:

* best performance
* optimal driving

Only Pareto improvement or tradeoff shift.

---

## 5. Required figure

Safety–utility tradeoff curve:

Compare:

* TTC / AEB (Responsibility-Sensitive Safety)
* RSS baseline (Responsibility-Sensitive Safety)
* divergence-based monitor

---

## Month 3 Deliverable

A validated runtime use case:

divergence can act as a lightweight early-warning monitor without excessive conservatism.

---

# Final Output of Entire Project

If executed correctly, you produce:

## 1. A predictive signal

Time-resolved planner–world divergence.

## 2. A scientific claim

Temporal divergence between planner and environment futures provides predictive information about failures beyond classical geometric safety metrics.

## 3. A controlled dataset

* distributions over time
* labeled outcomes
* confound-stratified splits

## 4. A minimal runtime monitor (optional)

Braking-only safety wrapper.

---

# Success Conditions (Strict)

Must succeed:

* divergence beats TTC in at least one of:

  * AUROC OR
  * lead time OR
  * stratified separation
* survives speed/density/occlusion stratification
* does not collapse into distance metric

Failure modes:

* no improvement over TTC
* signal disappears after binning
* strong dependence on scene complexity
* no temporal advantage

---

# One-line summary

You are testing whether temporal mismatch between predicted planner futures and predicted world futures is an early statistical indicator of sparse planner failure.
