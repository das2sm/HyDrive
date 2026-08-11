#!/usr/bin/env python3
"""Freeze a deterministic, counterbalanced multi-seed campaign schedule."""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
from collections import Counter
from itertools import combinations
from pathlib import Path

from campaign_common import (
    ARMS,
    artifact_root,
    atomic_write_json,
    canonical_json_sha256,
    ensure_artifact_identity,
    route_parts,
    sha256_file,
)


PERMUTATIONS = (
    ("baseline", "current_frame", "temporal"),
    ("baseline", "temporal", "current_frame"),
    ("current_frame", "baseline", "temporal"),
    ("current_frame", "temporal", "baseline"),
    ("temporal", "baseline", "current_frame"),
    ("temporal", "current_frame", "baseline"),
)
# With 555 route-seed blocks, these three permutations receive 93 blocks and
# the others receive 92. This choice balances every arm-position exactly,
# ordered precedence to 277/278, and adjacency to the minimum 184/186 spread.
HIGH_PERMUTATION_INDICES = (0, 3, 4)


def read_manifest(path: Path) -> list[str]:
    routes = [
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(routes) != len(set(routes)):
        raise ValueError("Route manifest contains duplicates")
    return routes


def _allocate_stratum_extras(
    remainders: list[int], quotas: list[int], rng: random.Random
) -> list[set[int]]:
    """Assign at most one extra copy of a permutation within each stratum."""
    assignments: list[set[int]] = [set() for _ in remainders]

    def search(stratum_index: int, remaining: tuple[int, ...]) -> bool:
        if stratum_index == len(remainders):
            return not any(remaining)
        width = remainders[stratum_index]
        candidates = [
            choice for choice in combinations(
                [index for index, quota in enumerate(remaining) if quota > 0],
                width,
            )
        ]
        rng.shuffle(candidates)
        candidates.sort(
            key=lambda choice: sum(remaining[index] for index in choice),
            reverse=True,
        )
        for choice in candidates:
            updated = list(remaining)
            for index in choice:
                updated[index] -= 1
            future_strata = len(remainders) - stratum_index - 1
            if any(value < 0 or value > future_strata for value in updated):
                continue
            assignments[stratum_index] = set(choice)
            if search(stratum_index + 1, tuple(updated)):
                return True
        return False

    if not search(0, tuple(quotas)):
        raise ValueError("Unable to balance permutations across schedule strata")
    return assignments


def build_schedule(routes: list[str], seeds: list[int], scheduler_seed: int) -> list[dict]:
    rng = random.Random(scheduler_seed)
    strata: dict[tuple[int, str], list[tuple[str, int]]] = {}
    for route in routes:
        split, _scenario, _route_id, _family = route_parts(route)
        for seed in seeds:
            strata.setdefault((seed, split), []).append((route, seed))
    stratum_keys = sorted(strata)
    for blocks in strata.values():
        rng.shuffle(blocks)

    total_blocks = len(routes) * len(seeds)
    base_target, target_remainder = divmod(total_blocks, len(PERMUTATIONS))
    if target_remainder != len(HIGH_PERMUTATION_INDICES):
        raise ValueError(
            "This exact counterbalance requires total route-seed blocks = 3 mod 6"
        )
    targets = [base_target] * len(PERMUTATIONS)
    for index in HIGH_PERMUTATION_INDICES:
        targets[index] += 1
    stratum_bases = [len(strata[key]) // len(PERMUTATIONS) for key in stratum_keys]
    baseline_total = sum(stratum_bases)
    quotas = [target - baseline_total for target in targets]
    remainders = [len(strata[key]) % len(PERMUTATIONS) for key in stratum_keys]
    extras = _allocate_stratum_extras(remainders, quotas, rng)

    assigned_blocks = []
    for stratum_index, key in enumerate(stratum_keys):
        pool = []
        base_count = stratum_bases[stratum_index]
        for permutation_index, permutation in enumerate(PERMUTATIONS):
            count = base_count + int(permutation_index in extras[stratum_index])
            pool.extend([permutation] * count)
        rng.shuffle(pool)
        assigned_blocks.extend(zip(strata[key], pool))
    rng.shuffle(assigned_blocks)

    rows = []
    job_index = 0
    for block_index, ((route, eval_seed), order) in enumerate(assigned_blocks):
        split, _scenario, route_id, family = route_parts(route)
        traffic_seed = route_id % 1000 + 10000 * eval_seed
        for position, arm in enumerate(order, start=1):
            rows.append({
                "job_index": job_index,
                "block_id": f"block-{block_index:04d}",
                "route": route,
                "family": family,
                "split": split,
                "evaluation_seed": eval_seed,
                "traffic_manager_seed": traffic_seed,
                "arm": arm,
                "order_position": position,
                "job_id": f"{route}__s{eval_seed}__{arm}",
            })
            job_index += 1
    return rows


def validate_schedule(rows: list[dict], route_count: int, seed_count: int) -> None:
    expected = route_count * seed_count
    if len(rows) != expected * len(ARMS):
        raise ValueError("Unexpected schedule size")
    jobs = Counter((row["route"], row["evaluation_seed"], row["arm"]) for row in rows)
    if len(jobs) != expected * len(ARMS) or set(jobs.values()) != {1}:
        raise ValueError("Each route/seed/arm must occur exactly once")
    positions = Counter((row["arm"], row["order_position"]) for row in rows)
    for arm in ARMS:
        counts = [positions[(arm, position)] for position in (1, 2, 3)]
        if max(counts) - min(counts) > 1:
            raise ValueError(f"Unbalanced order positions for {arm}: {counts}")
    blocks = {}
    for row in rows:
        blocks.setdefault(row["block_id"], []).append(row)
    orders = []
    stratum_orders: dict[tuple[int, str], list[tuple[str, ...]]] = {}
    for block_rows in blocks.values():
        block_rows.sort(key=lambda row: int(row["order_position"]))
        order = tuple(row["arm"] for row in block_rows)
        if order not in PERMUTATIONS:
            raise ValueError(f"Invalid arm order: {order}")
        orders.append(order)
        first = block_rows[0]
        key = (int(first["evaluation_seed"]), first["split"])
        stratum_orders.setdefault(key, []).append(order)
    permutation_counts = Counter(orders)
    permutation_values = [
        permutation_counts[permutation] for permutation in PERMUTATIONS
    ]
    if expected >= len(PERMUTATIONS):
        if set(permutation_counts) != set(PERMUTATIONS):
            raise ValueError("Every arm permutation must appear in the schedule")
        if max(permutation_values) - min(permutation_values) > 1:
            raise ValueError(f"Unbalanced permutation counts: {permutation_counts}")
    elif max(permutation_values) > 1:
        raise ValueError(
            f"Small pilot repeats an arm permutation: {permutation_counts}"
        )
    adjacency = Counter(
        pair for order in orders for pair in zip(order, order[1:])
    )
    expected_pairs = {(left, right) for left in ARMS for right in ARMS if left != right}
    # Odd permutation counts make exact adjacency equality impossible while
    # retaining exact arm-position balance; 184/186 is the minimum spread.
    if expected >= len(PERMUTATIONS):
        if (
            set(adjacency) != expected_pairs
            or max(adjacency.values()) - min(adjacency.values()) > 2
        ):
            raise ValueError(f"Unbalanced ordered adjacency: {adjacency}")
    precedence = Counter(
        (order[left], order[right])
        for order in orders
        for left in range(len(order))
        for right in range(left + 1, len(order))
    )
    if expected >= len(PERMUTATIONS):
        if (
            set(precedence) != expected_pairs
            or max(precedence.values()) - min(precedence.values()) > 1
        ):
            raise ValueError(f"Unbalanced ordered precedence: {precedence}")
    for key, stratum in stratum_orders.items():
        counts = Counter(stratum)
        values = [counts[permutation] for permutation in PERMUTATIONS]
        if max(values) - min(values) > 1:
            raise ValueError(f"Permutation spread exceeds one in stratum {key}: {values}")


def git_state(repo: Path) -> tuple[str, bool]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    ).strip())
    return commit, dirty


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    configured_repo = Path(protocol["repo_root"])
    repo = (
        configured_repo.resolve()
        if configured_repo.is_absolute()
        else (args.protocol.resolve().parent / configured_repo).resolve()
    )
    route_dir = (repo / protocol["route_dir"]).resolve()
    if args.pilot:
        routes = list(protocol["pilot_routes"])
        manifest = (repo / "tests/manifests/fail2drive_dev.txt").resolve()
        expected_routes = len(routes)
    else:
        manifest = (repo / protocol["route_manifest"]).resolve()
        routes = read_manifest(manifest)
        expected_routes = int(protocol["expected_route_count"])
    if len(routes) != expected_routes:
        raise ValueError(f"Expected {expected_routes} routes, found {len(routes)}")
    missing = [route for route in routes if not (route_dir / f"{route}.xml").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing route XML files: {missing[:5]}")

    commit, dirty = git_state(repo)
    f2d_commit, f2d_dirty = git_state(repo / "Fail2Drive")
    if dirty or f2d_dirty:
        raise RuntimeError("Refusing to freeze a campaign from a dirty worktree")

    rows = build_schedule(
        routes, [int(seed) for seed in protocol["evaluation_seeds"]],
        int(protocol["scheduler_seed"]),
    )
    validate_schedule(rows, len(routes), len(protocol["evaluation_seeds"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schedule_path = args.output_dir / "schedule.csv"
    fieldnames = list(rows[0])
    with schedule_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    paths = {
        "route_manifest": manifest,
        "config": repo / protocol["config_path"],
        "checkpoint": repo / protocol["checkpoint_path"],
        "agent": repo / protocol["agent_path"],
        "protocol": args.protocol.resolve(),
        "python_executable": Path(protocol["python_executable"]).resolve(),
        "carla_launcher": repo / protocol["carla_root"] / "CarlaUE4.sh",
        "carla_binary": (
            repo / protocol["carla_root"]
            / "CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
        ),
    }
    for relative, expected_sha256 in protocol["model_assets"].items():
        asset = repo / relative
        actual_sha256 = sha256_file(asset)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Model asset hash mismatch: {relative} "
                f"({actual_sha256} != {expected_sha256})"
            )
        paths[f"model_asset::{relative}"] = asset
    runtime_roots = (
        repo / "leaderboard/team_code",
        repo / "projects",
        repo / "Fail2Drive/fail2drive_leaderboard",
        repo / "Fail2Drive/fail2drive_scenario_runner",
        repo / "Fail2Drive/tools",
        repo / "scripts/multiseed",
        repo / "tests/analysis",
    )
    runtime_files = []
    for root in runtime_roots:
        runtime_files.extend(root.rglob("*.py"))
        runtime_files.extend(root.rglob("*.sh"))
    runtime_files.append(repo / "leaderboard/scripts/run_evaluation.sh")
    # Name runtime entries by their repo-relative (unresolved) location so the
    # lock is stable when runtime roots are symlinks into a shared checkout;
    # store and hash the resolved target so verification reads the real file.
    runtime_by_target: dict[Path, Path] = {}
    for path in runtime_files:
        if "__pycache__" in path.parts:
            continue
        runtime_by_target.setdefault(path.resolve(), path)
    for resolved, original in sorted(
        runtime_by_target.items(), key=lambda item: str(item[1])
    ):
        relative = original.relative_to(repo)
        paths[f"runtime::{relative}"] = resolved
    lock = {
        "campaign_id": protocol["campaign_id"] + ("_pilot" if args.pilot else ""),
        "pilot": args.pilot,
        "protocol_sha256": sha256_file(args.protocol),
        "schedule_sha256": sha256_file(schedule_path),
        "git_commit": commit,
        "git_dirty_at_freeze": dirty,
        "fail2drive_git_commit": f2d_commit,
        "fail2drive_git_dirty_at_freeze": f2d_dirty,
        "repo_root": str(repo),
        "agent_seed": int(protocol["agent_seed"]),
        "evaluation_seeds": protocol["evaluation_seeds"],
        "scheduler_seed": int(protocol["scheduler_seed"]),
        "bootstrap_seed": int(protocol["bootstrap_seed"]),
        "bootstrap_replicates": int(protocol["bootstrap_replicates"]),
        "fixed_delta_seconds": float(protocol["fixed_delta_seconds"]),
        "artifact_root": protocol["artifact_root"],
        "paths": {name: str(path.resolve()) for name, path in paths.items()},
        "file_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "file_stat": {
            name: {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for name, path in paths.items()
        },
        "route_xml_sha256": {
            route: sha256_file(route_dir / f"{route}.xml") for route in routes
        },
    }
    lock["lock_sha256"] = canonical_json_sha256(lock)
    atomic_write_json(args.output_dir / "campaign.lock.json", lock)
    ensure_artifact_identity(artifact_root(protocol, lock), lock, create=True)
    print(f"Frozen {len(rows)} jobs in {args.output_dir}")


if __name__ == "__main__":
    main()
