#!/usr/bin/env python3
"""Independently verify the multi-seed paper artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ARMS = ("baseline", "current_frame", "temporal")
CONTRASTS = {
    "current_frame_vs_baseline": ("current_frame", "baseline"),
    "temporal_vs_baseline": ("temporal", "baseline"),
    "temporal_vs_current_frame": ("temporal", "current_frame"),
}
METRICS = (
    "collision",
    "collision_vehicle",
    "collision_pedestrian",
    "collision_layout",
    "route_completion",
    "driving_score",
    "success",
    "timeout",
    "route_timeout",
    "scenario_timeout",
    "blocked",
    "route_deviation",
)

MUTABLE_REPOSITORY_PATHS = {
    "README.md",
    "CITATION.cff",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "TOOL_USE_DISCLOSURE.md",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as infile:
        rows = list(csv.DictReader(infile))
    for row in rows:
        row["evaluation_seed"] = int(row["evaluation_seed"])
        for metric in METRICS:
            row[metric] = float(row[metric])
    return rows


def route_deltas(
    rows: list[dict], treatment: str, control: str, metric: str
) -> tuple[dict[str, float], dict[str, str]]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    families = {}
    for row in rows:
        values[(row["route"], row["arm"])].append(row[metric])
        families[row["route"]] = row["family"]
    deltas = {}
    for route in sorted(families):
        treated = values[(route, treatment)]
        controlled = values[(route, control)]
        deltas[route] = float(np.average(treated) - np.average(controlled))
    return deltas, families


def bootstrap(
    rows: list[dict],
    treatment: str,
    control: str,
    metric: str,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    deltas, route_families = route_deltas(
        rows, treatment, control, metric
    )
    family_routes: dict[str, list[str]] = defaultdict(list)
    for route in deltas:
        family_routes[route_families[route]].append(route)
    family_ids = sorted(family_routes)
    generator = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        selected = generator.choice(
            family_ids, size=len(family_ids), replace=True
        )
        values = [
            deltas[route]
            for family in selected
            for route in family_routes[family]
        ]
        samples.append(float(np.average(values)))
    lower, upper = np.percentile(np.asarray(samples), (2.5, 97.5))
    return float(np.average(list(deltas.values()))), float(lower), float(upper)


def verify_manifest(root: Path) -> None:
    manifest = root / "MANIFEST.sha256"
    entries = []
    for line in manifest.read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        entries.append(relative)
        path = root / relative
        assert path.is_file(), f"Missing artifact file: {relative}"
        assert sha256_file(path) == expected, f"Hash mismatch: {relative}"
    mutable = MUTABLE_REPOSITORY_PATHS.intersection(entries)
    assert not mutable, (
        "Mutable repository files must not be listed in MANIFEST.sha256: "
        + ", ".join(sorted(mutable))
    )


def verify_schedule(root: Path) -> None:
    with (root / "data/schedule.csv").open(
        newline="", encoding="utf-8"
    ) as infile:
        rows = list(csv.DictReader(infile))
    assert len(rows) == 1665
    assert Counter(row["arm"] for row in rows) == Counter({
        "baseline": 555, "current_frame": 555, "temporal": 555
    })
    positions = Counter(
        (row["arm"], int(row["order_position"])) for row in rows
    )
    assert set(positions.values()) == {185}
    blocks: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        blocks[row["block_id"]].append(row)
    permutations = Counter()
    for block in blocks.values():
        order = tuple(
            row["arm"]
            for row in sorted(
                block, key=lambda item: int(item["order_position"])
            )
        )
        permutations[order] += 1
    assert len(blocks) == 555
    assert len(permutations) == 6
    assert sorted(permutations.values()) == [92, 92, 92, 93, 93, 93]


def verify_results(root: Path, full_bootstrap: bool) -> None:
    registry = json.loads(
        (root / "data/results_registry.json").read_text(encoding="utf-8")
    )
    rows = load_rows(root / "data/job_outcomes.csv")
    assert len(rows) == 1665
    assert len({row["route"] for row in rows}) == 185
    assert {row["evaluation_seed"] for row in rows} == {1, 2, 3}

    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        summary = registry["arm_descriptives"][arm]
        assert len(arm_rows) == summary["jobs"] == 555
        for metric, outcome in summary["binary_outcomes"].items():
            assert int(sum(row[metric] for row in arm_rows)) == outcome["events"]
        for metric, outcome in summary["continuous_outcomes"].items():
            assert np.isclose(
                np.average([row[metric] for row in arm_rows]),
                outcome["mean"],
                atol=1e-14,
            )

    settings = registry["design"]
    for contrast, (treatment, control) in CONTRASTS.items():
        for metric in METRICS:
            result = registry["contrasts"][contrast][metric]
            deltas, _ = route_deltas(rows, treatment, control, metric)
            assert len(deltas) == 185
            assert np.isclose(
                np.average(list(deltas.values())),
                result["estimate"],
                atol=1e-14,
            )
            if full_bootstrap:
                point, lower, upper = bootstrap(
                    rows,
                    treatment,
                    control,
                    metric,
                    settings["bootstrap_replicates"],
                    settings["bootstrap_seed"],
                )
                assert np.isclose(point, result["estimate"], atol=1e-14)
                assert np.isclose(lower, result["ci95_lower"], atol=1e-14)
                assert np.isclose(upper, result["ci95_upper"], atol=1e-14)

    if full_bootstrap:
        for seed in settings["evaluation_seeds"]:
            seed_rows = [
                row for row in rows if row["evaluation_seed"] == seed
            ]
            for contrast in (
                "temporal_vs_baseline", "temporal_vs_current_frame"
            ):
                treatment, control = CONTRASTS[contrast]
                for metric, result in registry["seed_estimates"][str(seed)][
                    contrast
                ].items():
                    point, lower, upper = bootstrap(
                        seed_rows,
                        treatment,
                        control,
                        metric,
                        settings["bootstrap_replicates"],
                        settings["bootstrap_seed"],
                    )
                    assert np.isclose(point, result["estimate"], atol=1e-14)
                    assert np.isclose(lower, result["ci95_lower"], atol=1e-14)
                    assert np.isclose(upper, result["ci95_upper"], atol=1e-14)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-bootstrap",
        action="store_true",
        help="also reproduce every reported 20,000-resample interval",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    verify_manifest(root)
    verify_schedule(root)
    verify_results(root, args.full_bootstrap)
    mode = "full bootstrap" if args.full_bootstrap else "quick"
    print(f"PASS: {mode} artifact verification")


if __name__ == "__main__":
    main()
