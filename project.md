# VAD Safety Guardian Roadmap

## Decision

Use **Option 1: Learned Occupancy** together with **Option 3: Stronger Baselines**.

This is the highest-value path for the paper because it addresses the two reviewer objections most likely to weaken the contribution:

1. **"The Guardian only works because it uses perfect CARLA perception."**  
   The main runtime path must use a learned occupancy estimate, not CARLA ground-truth actors. CARLA ground truth should remain only for label generation, diagnostics, and oracle upper-bound analysis.

2. **"Fuzzy arbitration is just another threshold braking heuristic."**  
   The evaluation must compare against real runtime safety monitors and naive threshold policies, not only against raw VAD.

The revised thesis is:

> A lightweight learned-occupancy safety layer can recover class-agnostic occupied-space awareness for vectorized autonomous driving planners, and fuzzy arbitration can reduce OOD collisions with fewer false positives and smoother interventions than hard-threshold and rule-based safety monitors.

## Core Claim

VAD-style vectorized planners can lose class-agnostic occupied-space awareness in occluded or out-of-distribution scenes. A runtime Guardian can restore this safety prior by:

1. estimating occupied BEV space with a lightweight learned occupancy head such as FlashOcc;
2. checking whether VAD's planned ego swept volume conflicts with occupied space;
3. converting geometric risk into smooth control intervention through fuzzy arbitration;
4. outperforming naive hard thresholds and established safety-monitor baselines on the safety/comfort tradeoff.

The paper should be framed as a **safety-grounding layer for vectorized planners**, not as a new occupancy network and not as a full planner replacement.

## Revised Timeline

| Dates | Focus | Main Deliverable |
| --- | --- | --- |
| May 16 - June 15 | Integrate FlashOcc learned occupancy | Guardian runs from learned occupancy instead of CARLA ground truth |
| June 16 - July 15 | Implement stronger baselines | RSS, SFF, hard TTC braking, and distance-based braking are evaluated under a shared protocol |
| July 16 - August 15 | Full experiments and paper writing | Main result tables, ablations, figures, and draft paper |

## Pre-May-16 Freeze

Before the revised plan starts, freeze the current CARLA-based geometric logger as a diagnostic/oracle tool only.

Keep:

```text
planned_trajectory
ego_pose
ego_speed
occupancy_source
min_occupied_distance
ttc_occupied
req_decel
gc_score
overlap_ratio
intervention_flag
collision_gt
```

Add or preserve:

```text
occ_source = learned | carla_oracle | debug
occ_confidence
monitor_name = vad | guardian | rss | sff | hard_ttc | distance
intervention_reason
intervention_start_time
intervention_duration
longitudinal_accel
lateral_accel
jerk
```

The CARLA occupancy path should be retained for:

```text
1. generating occupancy labels for FlashOcc training/evaluation
2. debugging learned occupancy failures
3. oracle upper-bound experiments
4. separating safety-logic failures from perception failures
```

It should not be the main perception source for the final Guardian results.

## Phase 1: Learned Occupancy Integration

**Timeline:** May 16 - June 15

### Objective

Replace CARLA ground-truth occupancy in the main Guardian loop with a lightweight learned occupancy head, preferably FlashOcc or a FlashOcc-style BEV occupancy module.

This phase is critical. Without it, reviewers can dismiss the method as a safety controller that depends on perfect simulator perception.

### Runtime Perception Contract

The Guardian should consume learned occupancy in ego-frame BEV:

```text
occ_prob_grid: H x W probability grid
grid_resolution_m
grid_origin_ego
time_offset
valid_mask or unknown_mask
occupancy_confidence
latency_ms
```

Use CARLA ground truth only to generate labels and to evaluate the learned occupancy error. At runtime, the main Guardian condition should be:

```text
camera/sensor input -> FlashOcc -> learned BEV occupancy -> swept-volume risk -> fuzzy arbitration
```

not:

```text
CARLA actors -> oracle occupancy -> swept-volume risk -> fuzzy arbitration
```

### FlashOcc Integration Tasks

#### 1. Data and Label Pipeline

Generate BEV occupancy labels from CARLA/Bench2Drive for training and validation:

```text
dynamic actors
static vehicles
pedestrians and cyclists
traffic barriers
cones and road obstacles
scenario props
OOD objects used in Fail2Drive-style tests
```

Labels must be class-agnostic for the Guardian:

```text
occupied = 1
free = 0
unknown/ignored = mask
```

Semantic labels may be stored for debugging, but the Guardian risk score should not depend on semantic class.

#### 2. Coordinate and Timing Alignment

Verify that learned occupancy and VAD trajectories share:

```text
ego-frame origin
BEV axis convention
grid resolution
future timestep indexing
sensor timestamp
planner timestamp
```

Acceptance check:

```text
Projected ego swept footprints align with occupied BEV regions in visualization
for straight driving, turning, braking, and route-intersection cases.
```

#### 3. Occupancy Postprocessing

Convert FlashOcc output into a conservative safety grid:

```python
occ_binary = occ_prob_grid >= occ_threshold
occ_binary = occ_binary & valid_mask
occ_binary = dilate(occ_binary, safety_margin_m)
```

Log both probability and binary versions:

```text
occ_prob_grid for risk scoring and calibration
occ_binary for hard-baseline comparability
```

Tune thresholds on a validation split only. Freeze them before final test runs.

#### 4. Probabilistic Swept-Volume Risk

Replace purely binary overlap with probability-aware overlap:

```python
expected_overlap_t = sum(ego_mask_t * occ_prob_grid) / max(sum(ego_mask_t), eps)
binary_overlap_t = sum(ego_mask_t & occ_binary) / max(sum(ego_mask_t), eps)
```

Use learned occupancy confidence to avoid over-trusting uncertain free space:

```python
unknown_pressure_t = sum(ego_mask_t * unknown_mask) / max(sum(ego_mask_t), eps)
```

The Guardian should treat unknown space conservatively only when it is near the planned swept volume. It should not globally brake for all uncertain perception.

#### 5. Geometric Conflict Score

Keep the score physically grounded and class-agnostic:

```python
overlap_score = max_t(exp(-t / tau_decay) * expected_overlap_t)
binary_score = max_t(exp(-t / tau_decay) * binary_overlap_t)
req_decel_score = clip(req_decel / decel_limit, 0.0, 1.0)
ttc_score = exp(-ttc_occupied / tau_ttc) if isfinite(ttc_occupied) else 0.0
unknown_score = max_t(exp(-t / tau_decay) * unknown_pressure_t)

Gc = max(overlap_score, binary_score, req_decel_score, ttc_score, unknown_weight * unknown_score)
```

Recommended starting values:

```text
decel_limit = 8.0 m/s^2
tau_decay = 1.5 s
tau_ttc = 1.5 s
unknown_weight = 0.3
```

Report all final constants and tune them only on validation routes.

#### 6. Learned-vs-Oracle Split

Every final result should distinguish:

```text
Learned occupancy + Guardian
Oracle occupancy + Guardian
Pure VAD
```

Interpretation:

```text
Oracle succeeds, learned fails:
  occupancy estimation is the bottleneck

Oracle fails:
  risk logic or intervention policy is insufficient

Learned succeeds close to oracle:
  the Guardian works under realistic perception noise
```

### Phase 1 Metrics

Evaluate learned occupancy independently:

```text
occupied-cell IoU
free-space IoU
precision/recall for occupied cells
calibration curve or reliability bins
false-free rate near ego planned trajectory
latency/FPS
memory overhead
```

Evaluate downstream safety during integration:

```text
collision rate on smoke-test routes
false positives/km
intervention duration
jerk
route completion
```

### Phase 1 Acceptance Criteria

By June 15:

```text
1. FlashOcc or FlashOcc-style learned occupancy runs in the VAD/Bench2Drive loop.
2. Guardian uses learned occupancy as its main runtime input.
3. CARLA occupancy is retained only as label/oracle/debug source.
4. Learned and oracle occupancy overlays can be visualized for the same frame.
5. Initial safety smoke tests run on nominal, occlusion, and OOD scenarios.
6. Runtime overhead is measured.
```

## Phase 2: Stronger Baselines

**Timeline:** June 16 - July 15

### Objective

Implement baselines that make the evaluation credible against both naive threshold policies and established runtime safety monitors.

The Guardian must not only beat raw VAD. It must show a better safety/comfort tradeoff than:

```text
RSS
SFF
hard TTC braking
distance-based braking
```

### Shared Baseline Interface

Implement all monitors behind the same interface:

```python
class SafetyMonitor:
    name: str

    def update(self, ego_state, vad_plan, occupancy, tracked_objects=None):
        return SafetyDecision(
            intervene: bool,
            target_accel: float,
            target_steer: float | None,
            safety_weight: float,
            reason: str,
            debug: dict,
        )
```

All baselines should use the same:

```text
route split
sensor inputs
learned occupancy source when applicable
controller limits
maximum braking bound
logging schema
metric scripts
validation tuning protocol
```

This prevents the comparison from becoming a perception or controller mismatch.

### Baseline 1: Pure VAD

No Guardian and no safety override.

Purpose:

```text
measure the base failure rate and route-completion behavior
```

### Baseline 2: Distance-Based Braking

Naive occupancy threshold baseline:

```python
if min_occupied_distance < d_threshold:
    brake()
```

Expected behavior:

```text
low complexity
many false positives in cluttered or adjacent-lane scenes
poor handling of ego speed and temporal urgency
```

What it tests:

```text
whether fuzzy arbitration reduces unnecessary interventions versus simple distance thresholds
```

### Baseline 3: Hard TTC Braking

Simple threshold on occupied-space TTC:

```python
if ttc_occupied < ttc_threshold:
    brake()
```

Expected behavior:

```text
stronger than distance-only braking
abrupt intervention
sensitive to noisy occupancy and trajectory jitter
```

What it tests:

```text
whether fuzzy arbitration is smoother than hard temporal thresholding
```

### Baseline 4: RSS

Implement a Responsibility-Sensitive Safety style monitor with longitudinal and lateral safe-distance rules.

Inputs:

```text
ego speed and acceleration
relative distance
relative velocity
lane relationship or BEV object relation
response time
maximum acceleration/deceleration assumptions
```

Decision:

```text
if RSS safe distance is violated, restrict acceleration or brake
```

Expected behavior:

```text
strong on structured car-following and lane-interaction cases
potentially conservative
less direct coverage for weird OOD occupied-space conflicts that are not clean tracked-object interactions
```

What to show:

```text
Guardian has lower jerk or fewer abrupt interventions than RSS
Guardian catches OOD occupied-space conflicts RSS misses
RSS remains a strong baseline on standard traffic interactions
```

### Baseline 5: SFF

Implement a Safety Force Field style baseline that computes repulsive safety pressure from nearby occupied regions or objects.

Inputs:

```text
learned occupancy grid
ego footprint
relative distance field
ego velocity
planned path
```

Decision:

```text
convert obstacle proximity and closing speed into braking pressure or a control envelope
```

Expected behavior:

```text
stronger than raw distance thresholds because pressure is continuous
may be conservative in dense occupancy
may not match the planned swept-volume conflict as directly as the Guardian
```

What to show:

```text
Guardian provides better targeted intervention on actual plan/occupancy conflicts
Guardian reduces false positives in nearby-but-non-conflicting occupancy
```

### Baseline 6: Fuzzy Guardian

Main method:

```text
learned occupancy
swept ego-footprint conflict
required deceleration
occupied-space TTC
fuzzy arbitration
smooth control blending
```

The first paper version should prioritize braking-only safety intervention. Evasive steering should be included only if free-space validation is already robust.

### Baseline 7: Oracle Guardian

Use CARLA ground-truth occupancy with the same Guardian logic.

Purpose:

```text
upper bound for the safety logic
diagnostic split between perception failures and arbitration failures
```

Do not present this as the main deployable method.

### Tuning Protocol

Use a validation split and freeze all thresholds before test evaluation:

```text
distance threshold
TTC threshold
RSS response-time and deceleration parameters
SFF distance/force scaling
Guardian fuzzy membership constants
occupancy probability threshold
unknown-space weight
```

Report the chosen values. Avoid tuning on the final test set.

### Phase 2 Acceptance Criteria

By July 15:

```text
1. RSS, SFF, hard TTC, and distance braking run through the shared monitor interface.
2. All baselines log intervention reasons, jerk, false positives, and collision outcomes.
3. Baseline parameters are tuned on validation routes and frozen.
4. A first comparison table exists on nominal and OOD route subsets.
5. Failure cases are tagged where RSS/SFF/thresholds miss or over-intervene.
```

## Phase 3: Full Experiments and Paper

**Timeline:** July 16 - August 15

### Objective

Run the final evaluation and write the paper around the learned-occupancy Guardian and stronger baseline comparison.

### Main Experimental Groups

Run each method on the same routes and random seeds where possible:

```text
1. Pure VAD
2. VAD + distance-based braking
3. VAD + hard TTC braking
4. VAD + RSS
5. VAD + SFF
6. VAD + learned occupancy + Fuzzy Guardian
7. VAD + oracle occupancy + Fuzzy Guardian
```

### Scenario Sets

Use three scenario categories:

```text
Bench2Drive nominal routes:
  standard driving, route completion, comfort, traffic-rule behavior

Fail2Drive/OOD routes:
  unusual objects, perception shifts, nonstandard obstacles, degraded semantic assumptions

Targeted occlusion cases:
  zebra emerging from behind vending machine
  static wall or prop in ego path
  pedestrian/animal crossing from occlusion
  unusual lane obstacle not represented as a normal tracked object
```

### Primary Claims to Test

#### Claim 1: Learned Occupancy Makes the Method Realistic

Show:

```text
Learned occupancy + Guardian reduces OOD collisions versus pure VAD.
Learned occupancy + Guardian keeps a meaningful fraction of the oracle Guardian benefit.
Failures are analyzed as either occupancy misses or arbitration misses.
```

This directly answers the perfect-perception critique.

#### Claim 2: Fuzzy Guardian Has Fewer False Positives Than Hard Thresholds

Compare against:

```text
distance-based braking
hard TTC braking
hard binary occupancy braking if retained as an ablation
```

Show:

```text
false positives/km
false positives/min
unnecessary intervention rate
route-completion impact
```

Expected result:

```text
hard thresholds brake for nearby occupancy even when the VAD swept path is not actually in conflict;
the Guardian intervenes when geometric conflict, temporal urgency, and required deceleration jointly indicate risk.
```

#### Claim 3: Guardian Is Smoother Than RSS and Hard Braking

Compare:

```text
jerk
hard-brake count
peak deceleration
intervention duration
comfort score
```

Expected result:

```text
fuzzy safety_weight ramps intervention intensity, producing lower jerk than hard monitor switches.
```

#### Claim 4: Guardian Catches OOD Failures RSS Misses

Analyze cases where:

```text
RSS does not identify a clean responsible actor interaction
semantic object tracking is wrong or absent
occupied space still intersects the planned ego swept volume
```

Show:

```text
RSS miss examples
Guardian risk time series
BEV occupancy and swept-footprint overlap
intervention timing before collision
```

### Metrics

#### Safety

```text
collision rate
OOD collision rate
near-miss rate
minimum occupied distance
minimum TTC_occupied
warning lead time
collision-within-K-seconds AUROC/AUPRC
```

#### False Positives and Intervention Quality

```text
false positives/km
false positives/min
unnecessary interventions/km
intervention count
intervention duration
hard-brake count
intervention reason distribution
```

#### Driving Quality

```text
route completion
average speed
traffic infractions
longitudinal acceleration
lateral acceleration
jerk
comfort score
```

#### Runtime

```text
FlashOcc latency
Guardian risk computation latency
total monitor overhead
FPS
memory overhead
```

#### Occupancy Quality

```text
occupied-cell IoU
occupied-cell precision/recall
false-free rate near the planned ego path
false-occupied rate near the planned ego path
oracle-vs-learned downstream safety gap
```

### Result Tables

#### Table 1: Main Safety Results

Columns:

```text
method
occupancy source
nominal collision rate
OOD collision rate
near-miss rate
route completion
traffic infractions
FPS
```

Rows:

```text
Pure VAD
Distance braking
Hard TTC braking
RSS
SFF
Fuzzy Guardian learned occupancy
Fuzzy Guardian oracle occupancy
```

#### Table 2: Intervention Quality

Columns:

```text
method
false positives/km
false positives/min
hard-brake count
mean peak decel
mean jerk
intervention duration
comfort score
```

Main expected story:

```text
Guardian has fewer false positives than hard thresholds and smoother interventions than hard switching/RSS.
```

#### Table 3: OOD Failure Coverage

Columns:

```text
scenario type
Pure VAD collision
RSS result
SFF result
Guardian result
RSS miss reason
Guardian trigger lead time
```

Main expected story:

```text
Guardian catches occupied-space conflicts that are not cleanly represented as standard tracked-object safety cases.
```

#### Table 4: Learned vs Oracle Occupancy

Columns:

```text
method
occupancy source
occupancy IoU
false-free near path
OOD collision rate
false positives/km
latency
```

Main expected story:

```text
learned occupancy is imperfect but good enough for meaningful downstream safety gains;
oracle occupancy explains the remaining perception gap.
```

### Required Figures

#### Figure 1: System Overview

```text
VAD sensors/features -> VAD plan
camera/sensors -> FlashOcc learned BEV occupancy
VAD plan + learned occupancy -> swept-volume conflict
Gc, TTC, req_decel -> fuzzy Guardian
control blending -> final control
```

#### Figure 2: Hero Failure Case

```text
camera frame: occluded OOD obstacle emerging
BEV: learned occupancy grid
BEV: VAD planned swept ego footprints
red overlay: predicted footprint/occupancy conflict
plot: Gc and safety_weight rise before impact
outcome: VAD collides, Guardian brakes
```

#### Figure 3: Safety/Comfort Tradeoff

Plot:

```text
x-axis: false positives/km or mean jerk
y-axis: OOD collision reduction
points: Distance, Hard TTC, RSS, SFF, Guardian
```

Target story:

```text
Guardian is on a better Pareto frontier than naive thresholds and hard monitors.
```

#### Figure 4: Learned vs Oracle Gap

Plot:

```text
oracle Guardian
learned Guardian
pure VAD
```

Show:

```text
how much of the oracle safety benefit survives realistic learned perception noise.
```

### Paper Contributions

1. **Problem diagnosis:** Vectorized planners can generate trajectories that geometrically conflict with occupied space in occluded and OOD scenes.
2. **Learned occupancy grounding:** A lightweight FlashOcc-style occupancy head supplies class-agnostic occupied-space evidence without CARLA ground-truth perception at runtime.
3. **Fuzzy safety arbitration:** A runtime Guardian maps swept-volume conflict, TTC, required deceleration, and occupancy confidence into smooth intervention.
4. **Stronger evaluation:** The method is compared against RSS, SFF, hard TTC braking, distance-based braking, pure VAD, and oracle occupancy.
5. **Perception-planning split:** Learned-vs-oracle experiments identify whether remaining failures come from occupancy estimation or arbitration logic.

### Paper Structure

```text
1. Introduction
   - vectorized planners are efficient but can lose occupied-space grounding
   - perfect-perception safety monitors are not enough
   - learned occupancy + fuzzy arbitration as the proposed remedy

2. Related Work
   - vectorized autonomous driving
   - learned occupancy prediction
   - occupancy-aware planning
   - runtime safety monitors: RSS and SFF
   - fuzzy and neuro-symbolic control

3. Method
   - FlashOcc-style learned occupancy interface
   - class-agnostic BEV occupancy construction
   - swept ego-footprint conflict checking
   - probabilistic Gc score
   - fuzzy arbitration and control blending

4. Baselines
   - pure VAD
   - distance threshold
   - hard TTC threshold
   - RSS
   - SFF
   - oracle Guardian

5. Experiments
   - Bench2Drive nominal routes
   - Fail2Drive/OOD routes
   - targeted occlusion cases
   - tuning and held-out test protocol

6. Results
   - safety gains
   - false-positive comparison
   - smoothness and jerk comparison
   - RSS/SFF miss cases
   - learned-vs-oracle perception gap
   - runtime overhead

7. Limitations
   - learned occupancy can miss hazards
   - braking-only intervention cannot solve every failure
   - simulation-to-real validation remains future work

8. Conclusion
```

## Final Success Criteria

### Must Have

```text
FlashOcc or FlashOcc-style learned occupancy replaces CARLA ground truth in the main Guardian loop.
Guardian reduces OOD collisions versus pure VAD using learned occupancy.
Guardian has fewer false positives than distance and hard TTC thresholds.
Guardian has lower jerk or fewer abrupt interventions than hard switching and RSS.
Guardian catches at least several OOD occupied-space failures that RSS misses.
Runtime remains practical for Bench2Drive evaluation.
```

### Should Have

```text
oracle Guardian upper-bound results
learned-vs-oracle downstream gap analysis
occupancy false-free analysis near planned path
SFF comparison on the same learned occupancy grid
warning lead-time distributions
AUROC/AUPRC for collision-within-K-seconds prediction
```

### Nice to Have

```text
full 220-route Bench2Drive evaluation
additional occupancy heads beyond FlashOcc
ablation of unknown-space conservatism
ablation of fuzzy rules and membership functions
limited steering intervention with validated free-space checks
```

## Positioning

Use this framing:

```text
Learned occupancy makes the Guardian realistic.
Stronger baselines make the fuzzy arbitration claim credible.
The contribution is the safety/comfort tradeoff: fewer OOD collisions than VAD, fewer false positives than hard thresholds, and smoother intervention than hard runtime monitors.
```

Avoid this framing:

```text
The method assumes perfect simulator perception.
The method is only a TTC threshold.
The method replaces occupancy networks or end-to-end planners.
```

The clean final message is:

> Do not drive through occupied space, even when the vectorized planner fails to represent the hazard cleanly. Learned occupancy supplies the occupied-space signal, and fuzzy arbitration turns it into smoother, less overreactive safety intervention than naive or hard-rule monitors.
