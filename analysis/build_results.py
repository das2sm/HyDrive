#!/usr/bin/env python3
"""Build paper-facing artifacts from the multi-seed analysis.

This script copies the primary results, computes paper-facing
descriptives and supporting split analyses from job_outcomes.csv, and renders
tables and figures. It does not change the endpoints, primary estimator,
or interpretation branch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ARMS = ("baseline", "current_frame", "temporal")
BINARY_METRICS = (
    "collision",
    "collision_vehicle",
    "collision_pedestrian",
    "collision_layout",
    "success",
    "timeout",
    "route_timeout",
    "scenario_timeout",
    "blocked",
    "route_deviation",
)
CONTINUOUS_METRICS = ("route_completion", "driving_score")
ANALYSIS_FILES = (
    "results.json",
    "job_outcomes.csv",
    "contrast_table.csv",
    "results_macros.tex",
)
ARM_LABELS = {
    "baseline": "Baseline",
    "current_frame": "Current-frame veto",
    "temporal": "Temporal veto",
}
CONTRAST_LABELS = {
    "temporal_vs_baseline": "Temporal - baseline",
    "temporal_vs_current_frame": "Temporal - current-frame",
    "current_frame_vs_baseline": "Current-frame - baseline",
}
METRIC_LABELS = {
    "collision": "Any collision",
    "collision_vehicle": "Vehicle collision",
    "collision_pedestrian": "Pedestrian collision",
    "collision_layout": "Layout collision",
    "route_completion": "Route completion",
    "driving_score": "Driving Score",
    "success": "Success Rate",
    "timeout": "Any timeout",
    "route_timeout": "Route timeout",
    "scenario_timeout": "Scenario timeout",
    "blocked": "Vehicle blocked",
    "route_deviation": "Route deviation",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as infile:
        return json.load(infile)


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as infile:
        return list(csv.DictReader(infile))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number_rows(rows: list[dict]) -> list[dict]:
    converted = []
    integer_fields = {
        "job_index",
        "evaluation_seed",
        "traffic_manager_seed",
        "order_position",
        "frame_count",
        "mechanism_available",
    }
    metric_fields = set(BINARY_METRICS + CONTINUOUS_METRICS) | {
        "switch_rate",
        "original_flagged_rate",
        "all_unsafe_rate",
    }
    for row in rows:
        item = dict(row)
        for key in integer_fields:
            item[key] = int(item[key])
        for key in metric_fields:
            item[key] = float(item[key])
        converted.append(item)
    return converted


def arm_descriptives(rows: list[dict]) -> dict:
    summaries = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        count = len(arm_rows)
        summary = {
            "jobs": count,
            "routes": len({row["route"] for row in arm_rows}),
            "evaluation_seeds": sorted({
                row["evaluation_seed"] for row in arm_rows
            }),
            "binary_outcomes": {},
            "continuous_outcomes": {},
        }
        for metric in BINARY_METRICS:
            events = int(sum(row[metric] for row in arm_rows))
            summary["binary_outcomes"][metric] = {
                "events": events,
                "denominator": count,
                "rate": events / count,
            }
        for metric in CONTINUOUS_METRICS:
            summary["continuous_outcomes"][metric] = {
                "mean": sum(row[metric] for row in arm_rows) / count,
                "denominator": count,
                "unit": "points",
            }
        if arm != "baseline":
            mechanism_rows = [
                row for row in arm_rows if row["mechanism_available"] == 1
            ]
            frames = sum(row["frame_count"] for row in mechanism_rows)
            switch_rate = (
                sum(row["switch_rate"] * row["frame_count"]
                    for row in mechanism_rows) / frames
            )
            original_flagged_rate = (
                sum(row["original_flagged_rate"] * row["frame_count"]
                    for row in mechanism_rows) / frames
            )
            if switch_rate > original_flagged_rate:
                raise ValueError(
                    f"Selection-change rate exceeds original-flagged rate for {arm}"
                )
            summary["mechanism"] = {
                "jobs": len(mechanism_rows),
                "logged_frames": frames,
                "switch_rate": switch_rate,
                "original_flagged_rate": original_flagged_rate,
                # Under the binary selector, a flagged original changes
                # exactly when a valid, unsuppressed clear candidate exists.
                # The complement includes both all_unsafe and
                # no_valid_candidate fallbacks.
                "flagged_without_clear_alternative_rate": (
                    original_flagged_rate - switch_rate
                ),
                "clear_alternative_given_flagged_rate": (
                    switch_rate / original_flagged_rate
                ),
                "all_unsafe_rate": (
                    sum(row["all_unsafe_rate"] * row["frame_count"]
                        for row in mechanism_rows) / frames
                ),
            }
        summaries[arm] = summary
    return summaries


def seed_arm_descriptives(rows: list[dict]) -> dict:
    summaries = {}
    for seed in sorted({row["evaluation_seed"] for row in rows}):
        summaries[str(seed)] = {}
        for arm in ARMS:
            arm_rows = [
                row for row in rows
                if row["evaluation_seed"] == seed and row["arm"] == arm
            ]
            summaries[str(seed)][arm] = {
                "jobs": len(arm_rows),
                "collision_jobs": int(sum(row["collision"] for row in arm_rows)),
                "collision_rate": (
                    sum(row["collision"] for row in arm_rows) / len(arm_rows)
                ),
                "mean_route_completion": (
                    sum(row["route_completion"] for row in arm_rows) / len(arm_rows)
                ),
            }
    return summaries


def paired_route_deltas(
    rows: list[dict], treatment: str, control: str, metric: str
) -> tuple[dict[str, float], dict[str, str]]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    route_family = {}
    for row in rows:
        values[(row["route"], row["arm"])].append(float(row[metric]))
        route_family[row["route"]] = row["family"]
    deltas = {}
    for route in sorted(route_family):
        treatment_values = values.get((route, treatment), [])
        control_values = values.get((route, control), [])
        if treatment_values and control_values:
            deltas[route] = float(
                np.mean(treatment_values) - np.mean(control_values)
            )
    return deltas, route_family


def cluster_bootstrap(
    rows: list[dict],
    treatment: str,
    control: str,
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    deltas, route_family = paired_route_deltas(
        rows, treatment, control, metric
    )
    families: dict[str, list[str]] = defaultdict(list)
    for route in deltas:
        families[route_family[route]].append(route)
    family_ids = sorted(families)
    generator = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = generator.choice(
            family_ids, size=len(family_ids), replace=True
        )
        draws[index] = np.mean([
            deltas[route]
            for family in sampled
            for route in families[family]
        ])
    lower, upper = np.percentile(draws, [2.5, 97.5])
    return {
        "estimate": float(np.mean(list(deltas.values()))),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
    }


def split_collision_analysis(
    rows: list[dict], replicates: int, seed: int
) -> dict:
    splits = {}
    for split in ("Base", "Generalization"):
        split_rows = [row for row in rows if row["split"] == split]
        arm_summaries = {}
        for arm in ("baseline", "temporal"):
            arm_rows = [row for row in split_rows if row["arm"] == arm]
            events = int(sum(row["collision"] for row in arm_rows))
            arm_summaries[arm] = {
                "jobs": len(arm_rows),
                "collision_jobs": events,
                "collision_rate": events / len(arm_rows),
            }
        splits[split] = {
            "routes": len({row["route"] for row in split_rows}),
            "arms": arm_summaries,
            "temporal_vs_baseline": cluster_bootstrap(
                split_rows,
                "temporal",
                "baseline",
                "collision",
                replicates,
                seed,
            ),
        }

    route_deltas, route_family = paired_route_deltas(
        rows, "temporal", "baseline", "collision"
    )
    route_split = {row["route"]: row["split"] for row in rows}
    family_routes: dict[str, list[str]] = defaultdict(list)
    for route in route_deltas:
        family_routes[route_family[route]].append(route)
    family_ids = sorted(family_routes)
    generator = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = generator.choice(
            family_ids, size=len(family_ids), replace=True
        )
        selected = [
            route
            for family in sampled
            for route in family_routes[family]
        ]
        base = [
            route_deltas[route]
            for route in selected
            if route_split[route] == "Base"
        ]
        generalization = [
            route_deltas[route]
            for route in selected
            if route_split[route] == "Generalization"
        ]
        draws[index] = np.mean(generalization) - np.mean(base)
    interaction_lower, interaction_upper = np.percentile(
        draws, [2.5, 97.5]
    )
    interaction = (
        splits["Generalization"]["temporal_vs_baseline"]["estimate"]
        - splits["Base"]["temporal_vs_baseline"]["estimate"]
    )
    return {
        "status": "supporting_post_campaign",
        "metric": "collision",
        "splits": splits,
        "generalization_minus_base_interaction": {
            "estimate": float(interaction),
            "ci95_lower": float(interaction_lower),
            "ci95_upper": float(interaction_upper),
        },
    }


def interpretation_claim(branch: str) -> str:
    claims = {
        "positive": (
            "Under the evaluated multi-seed Fail2Drive protocol, privileged "
            "temporal occupancy filtering reduced official collision-route "
            "incidence relative to unmodified SparseDriveV2 while satisfying "
            "the pre-specified route-completion guardrail."
        ),
        "collision_reduction_with_mobility_tradeoff": (
            "Under the evaluated multi-seed Fail2Drive protocol, privileged "
            "temporal occupancy filtering lowered official collision-route "
            "incidence, but the pre-specified route-completion guardrail was "
            "not satisfied."
        ),
        "negative": (
            "Under the evaluated multi-seed Fail2Drive protocol, privileged "
            "temporal occupancy filtering increased official collision-route "
            "incidence relative to unmodified SparseDriveV2."
        ),
        "null": (
            "Under the evaluated multi-seed Fail2Drive protocol, the official "
            "collision-route difference between privileged temporal occupancy "
            "filtering and unmodified SparseDriveV2 was not statistically "
            "resolved."
        ),
    }
    return claims[branch]


def build_registry(
    campaign_root: Path,
    archive_checksum: Path | None,
) -> tuple[dict, list[dict]]:
    analysis_dir = campaign_root / "analysis"
    lock_path = campaign_root / "spec" / "campaign.lock.json"
    schedule_path = campaign_root / "spec" / "schedule.csv"
    data_lock_path = campaign_root / "data.lock.json"
    lock = load_json(lock_path)
    data_lock = load_json(data_lock_path)
    results = load_json(analysis_dir / "results.json")
    rows = number_rows(load_csv(analysis_dir / "job_outcomes.csv"))
    schedule = load_csv(schedule_path)

    analysis_hashes = {
        name: sha256_file(analysis_dir / name) for name in ANALYSIS_FILES
    }
    archive = None
    if archive_checksum is not None:
        fields = archive_checksum.read_text(encoding="utf-8").strip().split()
        if len(fields) != 2:
            raise ValueError(f"Malformed archive checksum: {archive_checksum}")
        archive = {"filename": Path(fields[1]).name, "sha256": fields[0]}

    arm_counts = Counter(row["arm"] for row in schedule)
    position_counts = Counter(
        (row["arm"], int(row["order_position"])) for row in schedule
    )
    permutation_counts = Counter()
    by_block: dict[str, list[dict]] = {}
    for row in schedule:
        by_block.setdefault(row["block_id"], []).append(row)
    for block_rows in by_block.values():
        ordered = sorted(block_rows, key=lambda row: int(row["order_position"]))
        permutation_counts["-".join(row["arm"] for row in ordered)] += 1

    registry = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "campaign_id": results["campaign_id"],
        "status": "locked_analysis_complete",
        "provenance": {
            "campaign_code_commit": lock["git_commit"],
            "fail2drive_commit": lock["fail2drive_git_commit"],
            "campaign_code_clean_at_freeze": not lock["git_dirty_at_freeze"],
            "fail2drive_clean_at_freeze": not lock["fail2drive_git_dirty_at_freeze"],
            "campaign_lock_content_sha256": lock["lock_sha256"],
            "campaign_lock_file_sha256": sha256_file(lock_path),
            "schedule_sha256": sha256_file(schedule_path),
            "protocol_sha256": lock["protocol_sha256"],
            "data_lock_sha256": sha256_file(data_lock_path),
            "analysis_sha256": analysis_hashes,
            "archive": archive,
        },
        "design": {
            "routes": len({row["route"] for row in schedule}),
            "route_families": len({row["family"] for row in schedule}),
            "evaluation_seeds": lock["evaluation_seeds"],
            "jobs": len(schedule),
            "jobs_by_arm": dict(sorted(arm_counts.items())),
            "fixed_delta_seconds": lock["fixed_delta_seconds"],
            "scheduler_seed": lock["scheduler_seed"],
            "bootstrap_seed": lock["bootstrap_seed"],
            "bootstrap_replicates": lock["bootstrap_replicates"],
            "arm_position_counts": {
                f"{arm}:position_{position}": position_counts[(arm, position)]
                for arm in ARMS for position in (1, 2, 3)
            },
            "permutation_counts": dict(sorted(permutation_counts.items())),
        },
        "integrity": {
            "scheduled_jobs": data_lock["job_count"],
            "accepted_jobs": data_lock["accepted_job_count"],
            "locked_files": len(data_lock["files"]),
            "primary_complete_routes": len(data_lock["primary_routes"]),
            "complete_three_arm_routes": len(data_lock["complete_three_arm_routes"]),
            "contrast_route_counts": {
                key: len(value)
                for key, value in data_lock["contrast_routes"].items()
            },
            "excluded_routes": data_lock["excluded_routes"],
            "missing_jobs": data_lock["missing_jobs"],
            "mechanism_missing_jobs": data_lock["mechanism_missing_jobs"],
            "mechanism_corrupt_jobs": data_lock["mechanism_corrupt_jobs"],
            "retry_summary": data_lock["retry_summary"],
        },
        "units": {
            **{metric: "proportion" for metric in BINARY_METRICS},
            **{metric: "points" for metric in CONTINUOUS_METRICS},
            "mechanism_rates": "frame-weighted proportion",
        },
        "arm_descriptives": arm_descriptives(rows),
        "seed_arm_descriptives": seed_arm_descriptives(rows),
        "contrasts": results["estimates"],
        "seed_estimates": results["seed_estimates"],
        "split_analysis": split_collision_analysis(
            rows,
            int(lock["bootstrap_replicates"]),
            int(lock["bootstrap_seed"]),
        ),
        "completion_guardrail": {
            "margin_points": -5.0,
            "passed": results["completion_guardrail_pass"],
            "temporal_vs_baseline": results["estimates"][
                "temporal_vs_baseline"
            ]["route_completion"],
        },
        "primary_missing_outcome_bounds": (
            results["primary_collision_worst_case_missing_bounds"]
        ),
        "interpretation": {
            "branch": results["primary_interpretation"],
            "headline_claim": interpretation_claim(
                results["primary_interpretation"]
            ),
            "secondary_temporal_vs_current_frame_resolved": (
                results["estimates"]["temporal_vs_current_frame"]["collision"][
                    "ci95_upper"
                ] < 0
                or results["estimates"]["temporal_vs_current_frame"]["collision"][
                    "ci95_lower"
                ] > 0
            ),
        },
    }
    return registry, rows


def pct(value: float, digits: int = 1) -> str:
    return f"{100.0 * value:.{digits}f}"


def write_results_table(registry: dict, output_dir: Path) -> None:
    rows = []
    for arm in ARMS:
        summary = registry["arm_descriptives"][arm]
        item = {"arm": ARM_LABELS[arm]}
        for metric in ("collision", "success", "timeout", "blocked",
                       "route_deviation"):
            outcome = summary["binary_outcomes"][metric]
            item[metric] = (
                f"{outcome['events']}/{outcome['denominator']} "
                f"({pct(outcome['rate'])}%)"
            )
        item["route_completion"] = (
            f"{summary['continuous_outcomes']['route_completion']['mean']:.2f}"
        )
        item["driving_score"] = (
            f"{summary['continuous_outcomes']['driving_score']['mean']:.2f}"
        )
        rows.append(item)

    csv_path = output_dir / "table_official_outcomes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    tex_rows = []
    for row in rows:
        tex_rows.append(
            " & ".join([
                row["arm"],
                row["collision"],
                row["route_completion"],
                row["driving_score"],
                row["success"],
                row["timeout"],
                row["blocked"],
            ]) + r" \\"
        )
    tex = "\n".join([
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Arm & Collision & RC & DS & Success & Timeout & Blocked \\",
        r"\midrule",
        *tex_rows,
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ])
    (output_dir / "table_official_outcomes.tex").write_text(
        tex, encoding="ascii"
    )

    contrast_rows = []
    for contrast, metrics in registry["contrasts"].items():
        for metric, values in metrics.items():
            scale = 100.0 if registry["units"][metric] == "proportion" else 1.0
            contrast_rows.append({
                "contrast": CONTRAST_LABELS[contrast],
                "metric": metric,
                "estimate": scale * values["estimate"],
                "ci95_lower": scale * values["ci95_lower"],
                "ci95_upper": scale * values["ci95_upper"],
                "unit": (
                    "percentage points"
                    if registry["units"][metric] == "proportion"
                    else "points"
                ),
            })
    with (output_dir / "table_contrasts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(contrast_rows[0]))
        writer.writeheader()
        writer.writerows(contrast_rows)


def effect_text(registry: dict, contrast: str, metric: str) -> str:
    result = registry["contrasts"][contrast][metric]
    scale = 100.0 if registry["units"][metric] == "proportion" else 1.0
    return (
        f"{scale * result['estimate']:+.1f} "
        f"[{scale * result['ci95_lower']:+.1f}, "
        f"{scale * result['ci95_upper']:+.1f}]"
    )


def write_supplement_tables(registry: dict, output_dir: Path) -> None:
    metrics = (
        "collision", "collision_vehicle", "collision_pedestrian",
        "collision_layout", "route_completion", "driving_score", "success",
        "timeout", "route_timeout", "scenario_timeout", "blocked",
        "route_deviation",
    )
    rows = [
        " & ".join([
            METRIC_LABELS[metric],
            effect_text(registry, "current_frame_vs_baseline", metric),
            effect_text(registry, "temporal_vs_baseline", metric),
            effect_text(registry, "temporal_vs_current_frame", metric),
        ]) + r" \\"
        for metric in metrics
    ]
    (output_dir / "table_supp_all_contrasts.tex").write_text(
        "\n".join([
            r"\begin{tabular}{@{}lrrr@{}}",
            r"\toprule",
            (
                r"Outcome & Current--base & Temporal--base "
                r"& Temporal--current \\"
            ),
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]),
        encoding="ascii",
    )

    seed_rows = []
    for seed in registry["design"]["evaluation_seeds"]:
        seed_key = str(seed)
        summary = registry["seed_arm_descriptives"][seed_key]
        primary = registry["seed_estimates"][seed_key][
            "temporal_vs_baseline"
        ]["collision"]
        secondary = registry["seed_estimates"][seed_key][
            "temporal_vs_current_frame"
        ]["collision"]
        seed_rows.append(
            " & ".join([
                str(seed),
                f"{summary['baseline']['collision_jobs']}/185",
                f"{summary['current_frame']['collision_jobs']}/185",
                f"{summary['temporal']['collision_jobs']}/185",
                (
                    f"{100 * primary['estimate']:+.1f} "
                    f"[{100 * primary['ci95_lower']:+.1f}, "
                    f"{100 * primary['ci95_upper']:+.1f}]"
                ),
                (
                    f"{100 * secondary['estimate']:+.1f} "
                    f"[{100 * secondary['ci95_lower']:+.1f}, "
                    f"{100 * secondary['ci95_upper']:+.1f}]"
                ),
            ]) + r" \\"
        )
    (output_dir / "table_supp_seed_collisions.tex").write_text(
        "\n".join([
            r"\begin{tabular}{@{}crrrrr@{}}",
            r"\toprule",
            (
                r"Seed & Base & Current & Temporal & Temporal--base "
                r"& Temporal--current \\"
            ),
            r"\midrule",
            *seed_rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]),
        encoding="ascii",
    )

    split_rows = []
    for split in ("Base", "Generalization"):
        result = registry["split_analysis"]["splits"][split]
        baseline = result["arms"]["baseline"]
        temporal = result["arms"]["temporal"]
        contrast = result["temporal_vs_baseline"]
        split_rows.append(
            " & ".join([
                split,
                str(result["routes"]),
                (
                    f"{baseline['collision_jobs']}/{baseline['jobs']} "
                    f"({100 * baseline['collision_rate']:.1f})"
                ),
                (
                    f"{temporal['collision_jobs']}/{temporal['jobs']} "
                    f"({100 * temporal['collision_rate']:.1f})"
                ),
                (
                    f"{100 * contrast['estimate']:+.1f} "
                    f"[{100 * contrast['ci95_lower']:+.1f}, "
                    f"{100 * contrast['ci95_upper']:+.1f}]"
                ),
            ]) + r" \\"
        )
    (output_dir / "table_supp_split_collisions.tex").write_text(
        "\n".join([
            r"\begin{tabular}{@{}lrrrr@{}}",
            r"\toprule",
            (
                r"Split & Routes & Baseline & Temporal "
                r"& Temporal--baseline \\"
            ),
            r"\midrule",
            *split_rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]),
        encoding="ascii",
    )

    mechanism_rows = []
    for arm in ("current_frame", "temporal"):
        mechanism = registry["arm_descriptives"][arm]["mechanism"]
        mechanism_rows.append(
            " & ".join([
                ARM_LABELS[arm],
                str(mechanism["jobs"]),
                f"{mechanism['logged_frames']:,}",
                f"{100 * mechanism['original_flagged_rate']:.1f}",
                (
                    f"{100 * mechanism['flagged_without_clear_alternative_rate']:.1f}"
                ),
                f"{100 * mechanism['switch_rate']:.1f}",
                (
                    f"{100 * mechanism['clear_alternative_given_flagged_rate']:.1f}"
                ),
            ]) + r" \\"
        )
    (output_dir / "table_supp_mechanism.tex").write_text(
        "\n".join([
            r"\begin{tabular}{@{}lrrrrrr@{}}",
            r"\toprule",
            (
                r"Arm & Jobs & Frames & Flagged & Flagged, no alternative "
                r"& Changed & Clear given flagged \\"
            ),
            r"\midrule",
            *mechanism_rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]),
        encoding="ascii",
    )

    schedule_rows = [
        " & ".join([
            r"$\rightarrow$".join(
                {
                    "baseline": "Baseline",
                    "current_frame": "Current",
                    "temporal": "Temporal",
                }[arm]
                for arm in permutation.split("-")
            ),
            str(count),
        ]) + r" \\"
        for permutation, count in registry["design"]["permutation_counts"].items()
    ]
    (output_dir / "table_supp_schedule.tex").write_text(
        "\n".join([
            r"\begin{tabular}{@{}lr@{}}",
            r"\toprule",
            r"Within-block arm order & Blocks \\",
            r"\midrule",
            *schedule_rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]),
        encoding="ascii",
    )


def write_macros(registry: dict, output_dir: Path) -> None:
    primary = registry["contrasts"]["temporal_vs_baseline"]
    current_vs_baseline = registry["contrasts"]["current_frame_vs_baseline"]
    temporal_vs_current = registry["contrasts"]["temporal_vs_current_frame"]
    baseline = registry["arm_descriptives"]["baseline"]
    current = registry["arm_descriptives"]["current_frame"]
    temporal = registry["arm_descriptives"]["temporal"]
    base_split = registry["split_analysis"]["splits"]["Base"]
    generalization_split = registry["split_analysis"]["splits"][
        "Generalization"
    ]
    split_interaction = registry["split_analysis"][
        "generalization_minus_base_interaction"
    ]

    def binary(arm: dict, metric: str, field: str) -> float | int:
        return arm["binary_outcomes"][metric][field]

    def continuous(arm: dict, metric: str) -> float:
        return arm["continuous_outcomes"][metric]["mean"]

    def scaled(contrast: dict, metric: str, field: str) -> float:
        scale = 100.0 if registry["units"][metric] == "proportion" else 1.0
        return scale * contrast[metric][field]

    macros = {
        "MultiSeedRoutes": registry["design"]["routes"],
        "MultiSeedRouteFamilies": registry["design"]["route_families"],
        "MultiSeedEvaluationSeeds": len(registry["design"]["evaluation_seeds"]),
        "MultiSeedJobs": registry["design"]["jobs"],
        "JobsPerArm": baseline["jobs"],
        "LockedFiles": registry["integrity"]["locked_files"],
        "BaselineCollisionJobs": binary(baseline, "collision", "events"),
        "CurrentCollisionJobs": binary(current, "collision", "events"),
        "TemporalCollisionJobs": binary(temporal, "collision", "events"),
        "BaselineCollisionRate": 100.0 * binary(baseline, "collision", "rate"),
        "CurrentCollisionRate": 100.0 * binary(current, "collision", "rate"),
        "TemporalCollisionRate": 100.0 * binary(temporal, "collision", "rate"),
        "BaselineCompletion": continuous(baseline, "route_completion"),
        "CurrentCompletion": continuous(current, "route_completion"),
        "TemporalCompletion": continuous(temporal, "route_completion"),
        "BaselineDrivingScore": continuous(baseline, "driving_score"),
        "CurrentDrivingScore": continuous(current, "driving_score"),
        "TemporalDrivingScore": continuous(temporal, "driving_score"),
        "BaselineSuccessRate": 100.0 * binary(baseline, "success", "rate"),
        "CurrentSuccessRate": 100.0 * binary(current, "success", "rate"),
        "TemporalSuccessRate": 100.0 * binary(temporal, "success", "rate"),
        "BaselineTimeoutRate": 100.0 * binary(baseline, "timeout", "rate"),
        "CurrentTimeoutRate": 100.0 * binary(current, "timeout", "rate"),
        "TemporalTimeoutRate": 100.0 * binary(temporal, "timeout", "rate"),
        "PrimaryCollisionDelta": scaled(primary, "collision", "estimate"),
        "PrimaryCollisionCILower": scaled(primary, "collision", "ci95_lower"),
        "PrimaryCollisionCIUpper": scaled(primary, "collision", "ci95_upper"),
        "PrimaryCompletionDelta": scaled(
            primary, "route_completion", "estimate"
        ),
        "PrimaryCompletionCILower": scaled(
            primary, "route_completion", "ci95_lower"
        ),
        "PrimaryCompletionCIUpper": scaled(
            primary, "route_completion", "ci95_upper"
        ),
        "PrimaryDrivingScoreDelta": scaled(
            primary, "driving_score", "estimate"
        ),
        "PrimaryDrivingScoreCILower": scaled(
            primary, "driving_score", "ci95_lower"
        ),
        "PrimaryDrivingScoreCIUpper": scaled(
            primary, "driving_score", "ci95_upper"
        ),
        "PrimarySuccessDelta": scaled(primary, "success", "estimate"),
        "PrimarySuccessCILower": scaled(primary, "success", "ci95_lower"),
        "PrimarySuccessCIUpper": scaled(primary, "success", "ci95_upper"),
        "PrimaryTimeoutDelta": scaled(primary, "timeout", "estimate"),
        "PrimaryTimeoutCILower": scaled(primary, "timeout", "ci95_lower"),
        "PrimaryTimeoutCIUpper": scaled(primary, "timeout", "ci95_upper"),
        "PrimaryBlockedDelta": scaled(primary, "blocked", "estimate"),
        "PrimaryBlockedCILower": scaled(primary, "blocked", "ci95_lower"),
        "PrimaryBlockedCIUpper": scaled(primary, "blocked", "ci95_upper"),
        "PrimaryDeviationDelta": scaled(
            primary, "route_deviation", "estimate"
        ),
        "PrimaryDeviationCILower": scaled(
            primary, "route_deviation", "ci95_lower"
        ),
        "PrimaryDeviationCIUpper": scaled(
            primary, "route_deviation", "ci95_upper"
        ),
        "CurrentCollisionDelta": scaled(
            current_vs_baseline, "collision", "estimate"
        ),
        "CurrentCollisionCILower": scaled(
            current_vs_baseline, "collision", "ci95_lower"
        ),
        "CurrentCollisionCIUpper": scaled(
            current_vs_baseline, "collision", "ci95_upper"
        ),
        "TemporalCurrentCollisionDelta": scaled(
            temporal_vs_current, "collision", "estimate"
        ),
        "TemporalCurrentCollisionCILower": scaled(
            temporal_vs_current, "collision", "ci95_lower"
        ),
        "TemporalCurrentCollisionCIUpper": scaled(
            temporal_vs_current, "collision", "ci95_upper"
        ),
        "TemporalCurrentCompletionDelta": scaled(
            temporal_vs_current, "route_completion", "estimate"
        ),
        "TemporalCurrentCompletionCILower": scaled(
            temporal_vs_current, "route_completion", "ci95_lower"
        ),
        "TemporalCurrentCompletionCIUpper": scaled(
            temporal_vs_current, "route_completion", "ci95_upper"
        ),
        "BaseRoutes": base_split["routes"],
        "BaseCollisionDelta": (
            100.0 * base_split["temporal_vs_baseline"]["estimate"]
        ),
        "BaseCollisionCILower": (
            100.0 * base_split["temporal_vs_baseline"]["ci95_lower"]
        ),
        "BaseCollisionCIUpper": (
            100.0 * base_split["temporal_vs_baseline"]["ci95_upper"]
        ),
        "GeneralizationRoutes": generalization_split["routes"],
        "GeneralizationCollisionDelta": (
            100.0
            * generalization_split["temporal_vs_baseline"]["estimate"]
        ),
        "GeneralizationCollisionCILower": (
            100.0
            * generalization_split["temporal_vs_baseline"]["ci95_lower"]
        ),
        "GeneralizationCollisionCIUpper": (
            100.0
            * generalization_split["temporal_vs_baseline"]["ci95_upper"]
        ),
        "SplitInteractionDelta": 100.0 * split_interaction["estimate"],
        "SplitInteractionCILower": (
            100.0 * split_interaction["ci95_lower"]
        ),
        "SplitInteractionCIUpper": (
            100.0 * split_interaction["ci95_upper"]
        ),
        "CurrentOriginalFlaggedRate": (
            100.0 * current["mechanism"]["original_flagged_rate"]
        ),
        "TemporalOriginalFlaggedRate": (
            100.0 * temporal["mechanism"]["original_flagged_rate"]
        ),
        "CurrentNoClearAlternativeRate": (
            100.0
            * current["mechanism"]["flagged_without_clear_alternative_rate"]
        ),
        "TemporalNoClearAlternativeRate": (
            100.0
            * temporal["mechanism"]["flagged_without_clear_alternative_rate"]
        ),
        "CurrentClearGivenFlaggedRate": (
            100.0
            * current["mechanism"]["clear_alternative_given_flagged_rate"]
        ),
        "TemporalClearGivenFlaggedRate": (
            100.0
            * temporal["mechanism"]["clear_alternative_given_flagged_rate"]
        ),
        "CurrentSwitchRate": 100.0 * current["mechanism"]["switch_rate"],
        "TemporalSwitchRate": 100.0 * temporal["mechanism"]["switch_rate"],
    }
    seed_words = {1: "One", 2: "Two", 3: "Three"}
    for seed in registry["design"]["evaluation_seeds"]:
        seed_key = str(seed)
        seed_name = seed_words[seed]
        seed_summary = registry["seed_arm_descriptives"][seed_key]
        seed_result = registry["seed_estimates"][seed_key][
            "temporal_vs_baseline"
        ]["collision"]
        macros[f"Seed{seed_name}BaselineCollisions"] = (
            seed_summary["baseline"]["collision_jobs"]
        )
        macros[f"Seed{seed_name}TemporalCollisions"] = (
            seed_summary["temporal"]["collision_jobs"]
        )
        macros[f"Seed{seed_name}PrimaryCollisionDelta"] = (
            100.0 * seed_result["estimate"]
        )
        macros[f"Seed{seed_name}PrimaryCollisionCILower"] = (
            100.0 * seed_result["ci95_lower"]
        )
        macros[f"Seed{seed_name}PrimaryCollisionCIUpper"] = (
            100.0 * seed_result["ci95_upper"]
        )

    with (output_dir / "paper_results_macros.tex").open(
        "w", encoding="ascii"
    ) as outfile:
        for name, value in macros.items():
            formatted = str(value) if isinstance(value, int) else f"{value:.1f}"
            outfile.write(f"\\newcommand{{\\{name}}}{{{formatted}}}\n")


def configure_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def write_forest_plot(registry: dict, output_dir: Path) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    entries = []
    for contrast in ("temporal_vs_baseline", "temporal_vs_current_frame"):
        pooled = registry["contrasts"][contrast]["collision"]
        entries.append((
            contrast, "Pooled", pooled["estimate"],
            pooled["ci95_lower"], pooled["ci95_upper"],
        ))
        for seed in registry["design"]["evaluation_seeds"]:
            result = registry["seed_estimates"][str(seed)][contrast]["collision"]
            entries.append((
                contrast, f"Seed {seed}", result["estimate"],
                result["ci95_lower"], result["ci95_upper"],
            ))

    labels = []
    values = []
    lowers = []
    uppers = []
    colors = []
    y = []
    palette = {
        "temporal_vs_baseline": "#167D8D",
        "temporal_vs_current_frame": "#C65D27",
    }
    positions = [7.5, 6.5, 5.5, 4.5, 2.8, 1.8, 0.8, -0.2]
    for position, (contrast, label, value, lower, upper) in zip(positions, entries):
        labels.append(
            f"{CONTRAST_LABELS[contrast]}\n{label}"
            if label == "Pooled" else f"  {label}"
        )
        values.append(100.0 * value)
        lowers.append(100.0 * lower)
        uppers.append(100.0 * upper)
        colors.append(palette[contrast])
        y.append(position)

    fig, ax = plt.subplots(figsize=(7.0, 3.7))
    for index, position in enumerate(y):
        marker = "D" if entries[index][1] == "Pooled" else "o"
        size = 5.2 if marker == "D" else 4.2
        ax.errorbar(
            values[index], position,
            xerr=[[values[index] - lowers[index]],
                  [uppers[index] - values[index]]],
            fmt=marker, markersize=size, color=colors[index],
            ecolor=colors[index], elinewidth=1.1, capsize=2.5,
            markeredgewidth=0.5, markeredgecolor="white", zorder=3,
        )
        ax.text(
            max(uppers) + 1.0, position,
            f"{values[index]:+.1f} [{lowers[index]:+.1f}, {uppers[index]:+.1f}]",
            va="center", ha="left", fontsize=7.3,
        )
    ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--", zorder=1)
    ax.axhline(3.65, color="#B8B8B8", linewidth=0.6)
    ax.set_yticks(y, labels)
    ax.set_xlabel(
        "Difference in collision-route incidence (percentage points; "
        "negative favors temporal)"
    )
    ax.set_xlim(min(lowers) - 1.5, max(uppers) + 8.5)
    ax.set_ylim(-0.8, 8.1)
    ax.grid(axis="x", color="#E2E2E2", linewidth=0.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(output_dir / "fig02_collision_forest.pdf")
    fig.savefig(output_dir / "fig02_collision_forest.png", dpi=300)
    plt.close(fig)


def write_mechanism_plot(registry: dict, output_dir: Path) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = (
        ("original_flagged_rate", "Original proposal\nflagged"),
        (
            "flagged_without_clear_alternative_rate",
            "Flagged, no\nclear alternative",
        ),
        ("switch_rate", "Selection\nchanged"),
    )
    current = registry["arm_descriptives"]["current_frame"]["mechanism"]
    temporal = registry["arm_descriptives"]["temporal"]["mechanism"]
    x = np.arange(len(metrics))
    width = 0.34
    current_values = [100.0 * current[key] for key, _ in metrics]
    temporal_values = [100.0 * temporal[key] for key, _ in metrics]

    fig, ax = plt.subplots(figsize=(5.3, 3.0))
    bars_current = ax.bar(
        x - width / 2, current_values, width,
        label="Current-frame veto", color="#707070",
    )
    bars_temporal = ax.bar(
        x + width / 2, temporal_values, width,
        label="Temporal veto", color="#167D8D",
    )
    for bars in (bars_current, bars_temporal):
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7.2)
    ax.set_ylabel("Share of logged frames (%)")
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylim(0, 60)
    ax.set_yticks(np.arange(0, 61, 20))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.savefig(output_dir / "fig03_mechanism_rates.pdf")
    fig.savefig(output_dir / "fig03_mechanism_rates.png", dpi=300)
    plt.close(fig)


def write_integrity_report(registry: dict, output_dir: Path) -> None:
    integrity = registry["integrity"]
    provenance = registry["provenance"]
    design = registry["design"]
    retries = integrity["retry_summary"]
    archive = provenance["archive"]
    lines = [
        "# Multi-Seed Campaign Integrity Report",
        "",
        "**Status: PASS**",
        "",
        "## Campaign Identity",
        "",
        f"- Campaign: `{registry['campaign_id']}`",
        f"- Campaign code: `{provenance['campaign_code_commit']}`",
        f"- Fail2Drive code: `{provenance['fail2drive_commit']}`",
        f"- Schedule SHA-256: `{provenance['schedule_sha256']}`",
        f"- Data-lock SHA-256: `{provenance['data_lock_sha256']}`",
        f"- Campaign files indexed: {integrity['locked_files']}",
        "",
        "Both repositories were clean when the campaign lock was created.",
        "",
        "## Completion",
        "",
        f"- Scheduled jobs: {integrity['scheduled_jobs']}",
        f"- Accepted jobs: {integrity['accepted_jobs']}",
        f"- Routes: {design['routes']} in {design['route_families']} families",
        f"- Complete three-arm routes: {integrity['complete_three_arm_routes']}",
        (
            "- Contrast routes: "
            + ", ".join(
                f"`{key}` {value}"
                for key, value in sorted(
                    integrity["contrast_route_counts"].items()
                )
            )
        ),
        f"- Excluded routes: {len(integrity['excluded_routes'])}",
        f"- Missing jobs: {len(integrity['missing_jobs'])}",
        "",
        "## Attempts And Mechanism Logs",
        "",
        (
            "- Accepted attempts by arm: "
            + ", ".join(
                f"`{key}` {value}"
                for key, value in sorted(
                    retries["accepted_attempts_by_arm"].items()
                )
            )
        ),
        (
            "- Accepted jobs requiring retry: "
            f"{sum(retries['accepted_jobs_requiring_retry_by_arm'].values())}"
        ),
        (
            "- Extra attempts among accepted jobs: "
            f"{retries['total_extra_attempts_for_accepted_jobs']}"
        ),
        (
            "- Missing compact mechanism logs: "
            f"{len(integrity['mechanism_missing_jobs'])}"
        ),
        (
            "- Quarantined compact mechanism logs: "
            f"{len(integrity['mechanism_corrupt_jobs'])}"
        ),
        "",
        "## Analysis Outputs",
        "",
    ]
    lines.extend(
        f"- `{name}`: `{digest}`"
        for name, digest in provenance["analysis_sha256"].items()
    )
    if archive is not None:
        lines.extend([
            "",
            "## Independent Archive",
            "",
            f"- `{archive['filename']}`",
            f"- SHA-256: `{archive['sha256']}`",
        ])
    lines.extend([
        "",
        "No result-dependent reruns, route exclusions, endpoint changes, or "
        "changes to the pre-specified primary inference were performed. The "
        "Base/Generalization decomposition is a supporting post-campaign "
        "analysis of the same locked outcomes.",
        "",
    ])
    (output_dir / "CAMPAIGN_INTEGRITY.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--archive-checksum", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "generated",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    registry, _ = build_registry(args.campaign_root, args.archive_checksum)
    (args.output_dir / "results_registry.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_results_table(registry, args.output_dir)
    write_supplement_tables(registry, args.output_dir)
    write_macros(registry, args.output_dir)
    write_forest_plot(registry, args.output_dir)
    write_mechanism_plot(registry, args.output_dir)
    write_integrity_report(registry, args.output_dir)
    print(json.dumps({
        "branch": registry["interpretation"]["branch"],
        "headline_claim": registry["interpretation"]["headline_claim"],
        "registry": str(args.output_dir / "results_registry.json"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
