# Geometric Divergence for Sparse Planners — 3-Month Research Roadmap

## Core Objective

Validate whether **trajectory–occupancy geometric inconsistency** is a **predictive early-warning signal of failure** in sparse transformer planners, beyond TTC/RSS/distance baselines.

---

# MONTH 1 — Signal + Dataset Construction (NO MODELING YET)

## Goal

Build a **clean, bias-minimized measurement system**.

You are NOT building a safety system.
You are building a **failure measurement instrument**.

---

## 1. Freeze system interfaces

Lock:

* planner (SparseDrive / VAD-style model)
* BEV occupancy (oracle or learned, but fixed)
* ego state + control logs

No architecture changes after this point.

---

## 2. Define only 3 signals

You are strictly limited to:

### (A) Geometric overlap

* trajectory tube vs occupancy intersection

### (B) TTC

* time to first collision with occupancy

### (C) Confidence proxy (pick ONE)

* trajectory score margin
* entropy over multi-modal trajectories
* top-1 vs top-2 gap

---

## 3. Build failure labeling (critical)

Define outcome using **future horizon (1–3s)**:

* safe
* near-miss
* failure (collision or unavoidable intervention)

Must be forward-looking, not reactive.

---

## 4. Dataset generation

Run controlled scenarios:

* nominal driving
* occlusion-heavy scenes
* OOD objects (rare geometry, animals, cones, construction)
* dense traffic vs sparse traffic

Log per timestep:

* (G, TTC, C, outcome)

---

## Month 1 Deliverable

A clean, temporally aligned dataset of geometry–confidence–failure tuples

No claims yet. Just measurement.

---

# MONTH 2 — Predictive Validity + Confound Control

## Goal

Prove divergence adds **real predictive information beyond baselines**.

This is the scientific core.

---

## 1. Define candidate signals

Evaluate:

* TTC
* RSS violation (Responsibility-Sensitive Safety)
* distance-to-occupancy
* divergence metric (combinations of G, TTC, C)

---

## 2. Main experiment: Early failure prediction

Task:

Predict failure within 1–3 seconds.

Metric:

* AUROC
* AUPRC
* calibration (optional but strong)

---

## 3. Key requirement: lead-time analysis

Show:

> divergence rises earlier than TTC/RSS triggers

---

## 4. Confound control (critical)

Stratify results by:

* occlusion level
* traffic density
* ego speed bins

Then show:

> divergence still separates failure vs non-failure within each bin

---

## 5. Ablation study (mandatory)

* w/o confidence term
* w/o temporal smoothing
* G-only
* TTC-only
* combined vs individual

---

## Month 2 Deliverable

A validated predictive signal that survives confound controls and beats TTC/RSS in at least one meaningful metric (lead time or AUROC)

---

# MONTH 3 — Minimal Intervention Validation

## Goal

Show your signal is usable, not just predictive.

BUT: this is secondary evidence.

---

## 1. Intervention modes (ONLY 3)

* no safety system (baseline planner)
* TTC / AEB baseline
* divergence-triggered braking

No steering
No trajectory rewriting

---

## 2. Evaluation metrics

### Safety

* collision rate
* near-miss rate

### Utility

* average speed
* route completion

### Comfort

* jerk
* braking frequency

---

## 3. Key result

Show:

> divergence-based intervention improves safety without collapsing utility compared to TTC/AEB

---

## 4. Required plot

Safety–Utility tradeoff curve:

Compare:

* TTC / AEB
* RSS baseline
* divergence system

---

## Month 3 Deliverable

Evidence that the signal is practically usable as a lightweight runtime monitor

---

# FINAL OUTPUT

You end with:

## 1. Predictive failure signal

* geometric divergence metric

## 2. Validated claim

> divergence improves early failure prediction over TTC/RSS under controlled confounds

## 3. Failure analysis dataset

* labeled + stratified

## 4. Minimal runtime monitor

* braking-only safety layer

---

# WHAT THIS REALLY IS

Not:

* safety system
* planner improvement
* controller

But:

> a statistical study of whether geometric inconsistency is an early-warning signal of learned planner failure

---

# SUCCESS CRITERIA

If ANY fail, paper is weak:

* no improvement over TTC
* confound collapse
* behaves like distance proxy
* no lead-time gain
