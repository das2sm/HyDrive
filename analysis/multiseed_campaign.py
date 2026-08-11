#!/usr/bin/env python3
"""Frozen analysis for HyDrive's multi-seed closed-loop campaign."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


COLLISION_KEYS = (
    "collisions_pedestrian", "collisions_vehicle", "collisions_layout"
)
IGNORED_INFRACTIONS_FOR_SUCCESS = {
    "min_speed_infractions", "outside_route_lanes"
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as infile:
        return json.load(infile)


def official_record(path: Path) -> dict:
    records = load_json(path).get("_checkpoint", {}).get("records", [])
    if len(records) != 1:
        raise ValueError(f"Expected one official record in {path}")
    return records[0]


def has_items(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, dict, str)):
        return len(value) > 0
    return bool(value)


def extract_outcomes(record: dict) -> dict[str, float]:
    infractions = record.get("infractions") or {}
    scores = record.get("scores") or {}
    status = str(record.get("status") or "")
    collision_types = {
        "collision_pedestrian": float(has_items(infractions.get("collisions_pedestrian"))),
        "collision_vehicle": float(has_items(infractions.get("collisions_vehicle"))),
        "collision_layout": float(has_items(infractions.get("collisions_layout"))),
    }
    collision = any(collision_types.values())
    route_timeout = (
        "timed out" in status.lower()
        or has_items(infractions.get("route_timeout"))
    )
    scenario_timeout = has_items(infractions.get("scenario_timeouts"))
    timeout = route_timeout or scenario_timeout
    success = float(not any(
        has_items(entries)
        for name, entries in infractions.items()
        if name not in IGNORED_INFRACTIONS_FOR_SUCCESS
    ))
    return {
        "collision": float(collision),
        **collision_types,
        "route_completion": float(scores["score_route"]),
        "driving_score": float(scores["score_composed"]),
        "success": success,
        "timeout": float(timeout),
        "route_timeout": float(route_timeout),
        "scenario_timeout": float(scenario_timeout),
        "blocked": float(has_items(infractions.get("vehicle_blocked"))),
        "route_deviation": float(has_items(infractions.get("route_dev"))),
    }


def mechanism_summary(path: Path) -> dict[str, float]:
    with path.open("rb") as infile:
        payload = pickle.load(infile)
    if payload.get("config", {}).get("log_profile") != "compact_campaign":
        raise ValueError(f"Not a compact campaign log: {path}")
    steps = payload.get("timesteps") or []
    denominator = float(len(steps))
    if denominator == 0:
        return {"frame_count": 0, "switch_rate": np.nan,
                "original_flagged_rate": np.nan, "all_unsafe_rate": np.nan}
    return {
        "frame_count": int(denominator),
        "switch_rate": sum(bool(step["selection_changed"]) for step in steps) / denominator,
        "original_flagged_rate": sum(
            step["original_occupancy_cost"] == 1 for step in steps
        ) / denominator,
        "all_unsafe_rate": sum(
            step["selection_fallback"] == "all_unsafe" for step in steps
        ) / denominator,
    }


def paired_route_deltas(
    rows: list[dict], treatment: str, control: str, metric: str
) -> tuple[dict[str, float], dict[str, str]]:
    by_route_arm: dict[tuple[str, str], list[float]] = defaultdict(list)
    route_family = {}
    for row in rows:
        by_route_arm[(row["route"], row["arm"])].append(float(row[metric]))
        route_family[row["route"]] = row["family"]
    route_deltas = {}
    for route in sorted(route_family):
        treatment_values = by_route_arm.get((route, treatment), [])
        control_values = by_route_arm.get((route, control), [])
        if not treatment_values or not control_values:
            continue
        route_deltas[route] = float(np.mean(treatment_values) - np.mean(control_values))
    return route_deltas, route_family


def cluster_bootstrap(rows: list[dict], treatment: str, control: str,
                      metric: str, replicates: int, seed: int) -> tuple[float, float, float]:
    route_deltas, route_family = paired_route_deltas(
        rows, treatment, control, metric
    )
    if not route_deltas:
        raise ValueError(f"No paired routes for {treatment}-{control} {metric}")
    families: dict[str, list[str]] = defaultdict(list)
    for route in route_deltas:
        families[route_family[route]].append(route)
    family_ids = sorted(families)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.choice(family_ids, size=len(family_ids), replace=True)
        values = [route_deltas[route] for family in sampled for route in families[family]]
        draws[index] = np.mean(values)
    point = float(np.mean(list(route_deltas.values())))
    lower, upper = np.percentile(draws, [2.5, 97.5])
    return point, float(lower), float(upper)


def collect_rows(root: Path, schedule: list[dict]) -> list[dict]:
    rows = []
    for scheduled in schedule:
        job_root = root / "jobs" / scheduled["job_id"]
        final = load_json(job_root / "final_receipt.json")
        attempt = job_root / "attempts" / f"attempt-{int(final['accepted_attempt']):02d}"
        row = dict(scheduled)
        row.update(extract_outcomes(official_record(attempt / "official.json")))
        logs = list((attempt / "route_logs").glob("*.pkl"))
        if len(logs) > 1:
            raise ValueError(f"Expected one compact log for {scheduled['job_id']}")
        if logs:
            row.update(mechanism_summary(logs[0]))
            row["mechanism_available"] = 1
        else:
            row.update({
                "frame_count": 0, "switch_rate": np.nan,
                "original_flagged_rate": np.nan, "all_unsafe_rate": np.nan,
            })
            row["mechanism_available"] = 0
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_json(args.protocol)
    data_lock_path = Path(protocol["artifact_root"]) / "data.lock.json"
    if not data_lock_path.is_file():
        raise RuntimeError("Finalize the campaign before analysis")
    data_lock = load_json(data_lock_path)
    import hashlib
    for relative, expected_hash in data_lock["files"].items():
        digest = hashlib.sha256()
        with (Path(protocol["artifact_root"]) / relative).open("rb") as infile:
            for chunk in iter(lambda: infile.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            raise RuntimeError(f"Data-lock hash mismatch: {relative}")
    with args.schedule.open(newline="", encoding="utf-8") as infile:
        schedule = [
            row for row in csv.DictReader(infile)
            if row["job_id"] in set(data_lock["accepted_job_ids"])
        ]
    rows = collect_rows(Path(protocol["artifact_root"]), schedule)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "job_outcomes.csv").open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metrics = (
        "collision", "collision_vehicle", "collision_pedestrian",
        "collision_layout", "route_completion", "driving_score", "success",
        "timeout", "route_timeout", "scenario_timeout", "blocked",
        "route_deviation",
    )
    contrasts = {
        "temporal_vs_baseline": ("temporal", "baseline"),
        "temporal_vs_current_frame": ("temporal", "current_frame"),
        "current_frame_vs_baseline": ("current_frame", "baseline"),
    }
    estimates = {}
    for contrast_name, (treatment, control) in contrasts.items():
        contrast_rows = [
            row for row in rows
            if row["route"] in set(data_lock["contrast_routes"][contrast_name])
        ]
        estimates[contrast_name] = {}
        for metric in metrics:
            point, lower, upper = cluster_bootstrap(
                contrast_rows, treatment, control, metric,
                int(protocol["bootstrap_replicates"]), int(protocol["bootstrap_seed"]),
            )
            estimates[contrast_name][metric] = {
                "estimate": point, "ci95_lower": lower, "ci95_upper": upper
            }
    arm_summaries = {}
    for arm in ("baseline", "current_frame", "temporal"):
        arm_rows = [row for row in rows if row["arm"] == arm]
        arm_summaries[arm] = {
            "jobs": len(arm_rows),
            "collision_jobs": int(sum(row["collision"] for row in arm_rows)),
            "successful_jobs": int(sum(row["success"] for row in arm_rows)),
            "mean_route_completion": float(np.mean([
                row["route_completion"] for row in arm_rows
            ])),
            "mean_driving_score": float(np.mean([
                row["driving_score"] for row in arm_rows
            ])),
        }
        if arm != "baseline":
            frames = sum(row["frame_count"] for row in arm_rows)
            mechanism = {
                "mechanism_jobs": int(sum(row["mechanism_available"] for row in arm_rows)),
                "logged_frames": int(frames),
            }
            if frames > 0:
                mechanism.update({
                    "switch_rate": float(sum(
                    row["switch_rate"] * row["frame_count"] for row in arm_rows
                    ) / frames),
                    "original_flagged_rate": float(sum(
                    row["original_flagged_rate"] * row["frame_count"]
                    for row in arm_rows
                    ) / frames),
                    "all_unsafe_rate": float(sum(
                    row["all_unsafe_rate"] * row["frame_count"] for row in arm_rows
                    ) / frames),
                })
            arm_summaries[arm].update(mechanism)
    seed_estimates = {}
    for eval_seed in protocol["evaluation_seeds"]:
        seed_rows = [
            row for row in rows if int(row["evaluation_seed"]) == int(eval_seed)
        ]
        seed_estimates[str(eval_seed)] = {}
        for name, treatment, control in (
            ("temporal_vs_baseline", "temporal", "baseline"),
            ("temporal_vs_current_frame", "temporal", "current_frame"),
        ):
            contrast_routes_for_seed = set(data_lock["contrast_routes"][name])
            contrast_seed_rows = [
                row for row in seed_rows if row["route"] in contrast_routes_for_seed
            ]
            seed_estimates[str(eval_seed)][name] = {}
            for metric in ("collision", "route_completion"):
                point, lower, upper = cluster_bootstrap(
                    contrast_seed_rows, treatment, control, metric,
                    int(protocol["bootstrap_replicates"]),
                    int(protocol["bootstrap_seed"]),
                )
                seed_estimates[str(eval_seed)][name][metric] = {
                    "estimate": point, "ci95_lower": lower, "ci95_upper": upper
                }
    primary = estimates["temporal_vs_baseline"]
    collision = primary["collision"]
    completion = primary["route_completion"]
    guardrail = completion["ci95_lower"] > float(protocol["completion_margin_points"])
    if collision["ci95_upper"] < 0 and guardrail:
        interpretation = "positive"
    elif collision["ci95_upper"] < 0:
        interpretation = "collision_reduction_with_mobility_tradeoff"
    elif collision["ci95_lower"] > 0:
        interpretation = "negative"
    else:
        interpretation = "null"
    primary_rows = [
        row for row in rows
        if row["route"] in set(data_lock["primary_routes"])
    ]
    primary_deltas, _ = paired_route_deltas(
        primary_rows, "temporal", "baseline", "collision"
    )
    expected_route_count = int(protocol["expected_route_count"])
    missing_primary_routes = expected_route_count - len(primary_deltas)
    observed_sum = float(sum(primary_deltas.values()))
    worst_case_bounds = {
        "lower": (observed_sum - missing_primary_routes) / expected_route_count,
        "upper": (observed_sum + missing_primary_routes) / expected_route_count,
    }
    result = {
        "campaign_id": protocol["campaign_id"], "job_count": len(rows),
        "primary_route_count": len(data_lock["primary_routes"]),
        "contrast_route_counts": {
            name: len(routes) for name, routes in data_lock["contrast_routes"].items()
        },
        "excluded_routes": data_lock["excluded_routes"],
        "retry_summary": data_lock["retry_summary"],
        "arm_summaries": arm_summaries,
        "estimates": estimates,
        "seed_estimates": seed_estimates,
        "completion_guardrail_pass": guardrail,
        "primary_collision_worst_case_missing_bounds": worst_case_bounds,
        "primary_interpretation": interpretation,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "contrast_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as outfile:
        writer = csv.DictWriter(outfile, fieldnames=(
            "contrast", "metric", "estimate", "ci95_lower", "ci95_upper"
        ))
        writer.writeheader()
        for contrast_name, contrast_metrics in estimates.items():
            for metric, values in contrast_metrics.items():
                writer.writerow({"contrast": contrast_name, "metric": metric, **values})
    macros = {
        "PrimaryCollisionDelta": 100.0 * collision["estimate"],
        "PrimaryCollisionCILower": 100.0 * collision["ci95_lower"],
        "PrimaryCollisionCIUpper": 100.0 * collision["ci95_upper"],
        "PrimaryCompletionDelta": completion["estimate"],
        "PrimaryCompletionCILower": completion["ci95_lower"],
        "PrimaryCompletionCIUpper": completion["ci95_upper"],
        "AnalyzedRoutes": len(data_lock["primary_routes"]),
        "ExcludedRoutes": len(data_lock["excluded_routes"]),
    }
    with (args.output_dir / "results_macros.tex").open("w", encoding="ascii") as outfile:
        for name, value in macros.items():
            formatted = str(value) if isinstance(value, int) else f"{value:.1f}"
            outfile.write(f"\\newcommand{{\\{name}}}{{{formatted}}}\n")
    print(json.dumps({"primary_interpretation": interpretation,
                      "completion_guardrail_pass": guardrail}, sort_keys=True))


if __name__ == "__main__":
    main()
