# Multi-Seed Closed-Loop Evaluation Protocol

## Research question

Does privileged temporal occupancy filtering change closed-loop collision
outcomes relative to unmodified SparseDriveV2? An otherwise identical
current-frame filter is a controlled ablation of occupancy time indexing.

## Design

The evaluation uses 185 Fail2Drive evaluation routes and three evaluation
seeds. Each route-seed block contains baseline, current-frame, and temporal
conditions. Every job starts a fresh CARLA server and evaluator.

Condition order was generated once with scheduler seed 20260711. All six arm
permutations occur 92 or 93 times, every condition occurs exactly 185 times in
each ordinal position, ordered precedence differs by at most one, and
permutation counts differ by at most one within evaluation-seed and
base/generalization strata. Conditions within a route-seed block share the
Traffic Manager seed
`(route_id mod 1000) + 10000 * evaluation_seed`. The agent seed is 42.

The baseline executes SparseDriveV2 unchanged. Current-frame and temporal arms
apply the same binary proposal filter and differ only in occupancy time
indexing. Both call the same grid builder with waypoint horizons set to zero
for current-frame occupancy and to `{0.5, 1.0, ..., 3.0}` seconds for temporal
occupancy. The filter never selects a proposal suppressed by SparseDriveV2 and
retains the original proposal when no occupancy-clear candidate exists.

## Runtime controls

CARLA and Traffic Manager run synchronously with a fixed 0.05-second step.
Every route-arm-seed job launches a fresh process. Runtime assertions check the
arm, occupancy source, synchronous mode, fixed step, and locked identities.
CARLA readiness requires two consecutive client/world health checks.

Each job uses offscreen Epic rendering, no sound, a disabled primary server
port, and the benchmark evaluator's 300-second client timeout. A four-hour
outer wall cap bounds a hung simulator but does not terminate scoreable
driving. Process cleanup owns and reaps the complete CARLA/evaluator process
groups and verifies release of CARLA's client-command (RPC),
sensor/world-streaming, and Traffic Manager ports.

## Intervention

SparseDriveV2 exposes 1,024 scored trajectories with six waypoints at
0.5-second intervals. Privileged CARLA actors and selected static geometry are
rasterized into six ego-centered 120-by-120 grids at 0.5 m/cell. The temporal
arm propagates vehicles and walkers at constant velocity to each waypoint
time; current-frame occupancy uses zero propagation. Actor yaw and extent are
fixed. The ego footprint is densely sampled against the time-aligned grids.

A proposal is clear only when every footprint sample at every waypoint is
unoccupied. Out-of-grid proposals are invalid. Among valid, planner-unmasked,
clear proposals, the filter executes the highest SparseDriveV2 post-rescore
score. If no such proposal exists, it retains the original. There is no
emergency brake, trajectory optimization, progress rule, or hysteresis.

## Outcomes

The primary endpoint is the paired difference in official collision-route
proportion, temporal minus baseline. A collision is present when the official
record contains any vehicle, pedestrian, or layout collision.

The secondary contrast is temporal minus current-frame. Supporting outcomes
are route completion, Driving Score, Fail2Drive Success Rate, collision type,
route and scenario timeout, blockage, and route deviation. Success follows the
released Fail2Drive parser: every official infraction list must be empty except
`min_speed_infractions` and `outside_route_lanes`. A favorable primary result
also requires the lower confidence bound for temporal-minus-baseline route
completion to exceed -5 points.

For each route, outcomes are averaged over the three seeds before routes are
weighted equally. Percentile confidence intervals use 20,000 route-family
cluster-bootstrap resamples with seed 20260711, carrying every route, seed, and
condition in a sampled family.

## Acceptance, retries, and missing data

A job is accepted only if its official record passes schema and route-identity
checks, the evaluator exits zero, the strict initialization marker appears,
and no scenario-skip warning appears. Collision, timeout, blockage, deviation,
and incomplete completion are retained and never authorize a rerun.

Only technical failures are retryable, for at most three attempts. Retries
occur in deferred passes. Interrupted attempts are preserved as orphaned
directories, no artifact is overwritten, and the first valid official outcome
is immutable. Missing optional mechanism logs do not invalidate official
outcomes.

If technical failures remain, each contrast uses routes with all three seeds
for both arms in that contrast. Analysis requires at least 95% complete routes
for the primary contrast and reports missingness plus worst-case binary bounds.

## Pre-specification and result isolation

The schedule, arms, outcomes, estimator, bootstrap settings, acceptance rules,
and interpretation rule were specified before execution. Earlier exploratory
runs contribute no estimate. The strict benchmark revision is common to every
arm. The campaign produced 1,665 accepted first attempts, complete data for all
185 routes, no route exclusions, and no missing mechanism logs.
