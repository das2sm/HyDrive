#!/usr/bin/env python3
"""Verify campaign completeness and create an immutable data index."""

from __future__ import annotations

import argparse
import pickle
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from campaign_common import (
    artifact_root, atomic_write_json, ensure_artifact_identity, load_json,
    load_schedule, sha256_file,
)
from run_isolated_job import utc_now, verify_lock


def quarantine_mechanism_log(
    log_path: Path, row: dict, reason: str, detail: str
) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    quarantine = log_path.with_name(f"{log_path.name}.corrupt-{stamp}")
    suffix = 0
    while quarantine.exists():
        suffix += 1
        quarantine = log_path.with_name(
            f"{log_path.name}.corrupt-{stamp}-{suffix:02d}"
        )
    log_path.rename(quarantine)
    atomic_write_json(
        quarantine.with_name(quarantine.name + ".json"),
        {
            "job_id": row["job_id"],
            "quarantined_at": utc_now(),
            "original_name": log_path.name,
            "reason": reason,
            "detail": detail,
        },
    )


def reject_or_quarantine(
    log_path: Path, row: dict, pilot: bool, reason: str, detail: str
) -> str:
    if pilot:
        raise ValueError(
            f"Pilot compact log invalid for {row['job_id']}: {reason}: {detail}"
        )
    quarantine_mechanism_log(log_path, row, reason, detail)
    return "corrupt"


def validate_mechanism_log(attempt_dir: Path, row: dict, lock: dict) -> str:
    """Classify the accepted attempt's compact mechanism log.

    Returns "ok", "missing", or "corrupt". Unreadable bytes (disk rot after a
    valid run) are tolerable outside pilot mode: the file is quarantined with
    a marker so re-running finalize is idempotent, and the job is reported as
    mechanism-missing. Wrong profile, empty content, and provenance mismatch
    receive the same treatment because the official evaluator record is the
    accepted outcome; the compact log is an optional mechanism diagnostic.
    Every such failure remains fatal in pilot mode.
    """
    pilot = bool(lock.get("pilot", False))
    route_logs = list((attempt_dir / "route_logs").glob("*.pkl"))
    if len(route_logs) > 1:
        raise ValueError(f"Expected one compact route log for {row['job_id']}")
    if not route_logs:
        if pilot:
            raise ValueError(f"Pilot compact log missing for {row['job_id']}")
        quarantined = list(
            (attempt_dir / "route_logs").glob("*.pkl.corrupt-*")
        )
        if any(path.suffix != ".json" for path in quarantined):
            return "corrupt"
        return "missing"
    log_path = route_logs[0]
    try:
        with log_path.open("rb") as infile:
            payload = pickle.load(infile)
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError) as exc:
        return reject_or_quarantine(
            log_path, row, pilot,
            f"unreadable_compact_log:{type(exc).__name__}", repr(exc),
        )
    if not isinstance(payload, dict):
        return reject_or_quarantine(
            log_path, row, pilot, "invalid_payload_type", type(payload).__name__
        )
    profile = payload.get("config", {}).get("log_profile")
    if profile != "compact_campaign":
        return reject_or_quarantine(
            log_path, row, pilot, "wrong_log_profile", repr(profile)
        )
    if not payload.get("timesteps"):
        return reject_or_quarantine(
            log_path, row, pilot, "empty_compact_log", "timesteps is empty"
        )
    provenance = payload.get("provenance", {})
    expected_provenance = {
        "campaign_id": lock["campaign_id"],
        "campaign_lock_sha256": lock["lock_sha256"],
        "schedule_sha256": lock["schedule_sha256"],
        "job_id": row["job_id"],
        "arm": row["arm"],
        "evaluation_seed": int(row["evaluation_seed"]),
        "order_position": int(row["order_position"]),
        "traffic_manager_seed": int(row["traffic_manager_seed"]),
        "config_sha256": lock["file_sha256"]["config"],
        "checkpoint_sha256": lock["file_sha256"]["checkpoint"],
        "synchronous_mode": True,
        "fixed_delta_seconds": lock["fixed_delta_seconds"],
    }
    mismatches = {
        key: (provenance.get(key), expected)
        for key, expected in expected_provenance.items()
        if provenance.get(key) != expected
    }
    if mismatches:
        return reject_or_quarantine(
            log_path, row, pilot, "provenance_mismatch", repr(mismatches)
        )
    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_json(args.protocol)
    lock = load_json(args.lock)
    verify_lock(lock, args.lock, args.schedule)
    root = artifact_root(protocol, lock)
    ensure_artifact_identity(root, lock)
    rows = load_schedule(args.schedule)
    files = {}
    missing = []
    mechanism_missing_jobs = []
    mechanism_corrupt_jobs = []
    accepted_jobs = set()
    accepted_attempts_by_arm = Counter()
    retry_jobs_by_arm = Counter()
    for row in rows:
        final_path = root / "jobs" / row["job_id"] / "final_receipt.json"
        if not final_path.is_file():
            missing.append(row["job_id"])
            continue
        accepted_jobs.add(row["job_id"])
        final = load_json(final_path)
        accepted_attempt = int(final["accepted_attempt"])
        accepted_attempts_by_arm[row["arm"]] += accepted_attempt
        if accepted_attempt > 1:
            retry_jobs_by_arm[row["arm"]] += 1
        receipt_path = Path(final["receipt"])
        receipt = load_json(receipt_path)
        if not receipt["classification"]["valid"]:
            raise ValueError(f"Accepted receipt is invalid: {receipt_path}")
        result_path = receipt_path.parent / "official.json"
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        files[str(result_path.relative_to(root))] = sha256_file(result_path)
        files[str(receipt_path.relative_to(root))] = sha256_file(receipt_path)
        log_state = validate_mechanism_log(receipt_path.parent, row, lock)
        if log_state == "missing":
            mechanism_missing_jobs.append(row["job_id"])
        elif log_state == "corrupt":
            mechanism_corrupt_jobs.append(row["job_id"])
    jobs_by_route = {}
    for row in rows:
        jobs_by_route.setdefault(row["route"], {}).setdefault(row["arm"], []).append(
            row["job_id"]
        )

    def complete_routes(required_arms: tuple[str, ...]) -> list[str]:
        return sorted(
            route for route, arm_jobs in jobs_by_route.items()
            if all(
                len(arm_jobs.get(arm, [])) == len(lock["evaluation_seeds"])
                and all(job_id in accepted_jobs for job_id in arm_jobs[arm])
                for arm in required_arms
            )
        )

    contrast_routes = {
        "temporal_vs_baseline": complete_routes(("temporal", "baseline")),
        "temporal_vs_current_frame": complete_routes(("temporal", "current_frame")),
        "current_frame_vs_baseline": complete_routes(("current_frame", "baseline")),
    }
    complete_three_arm_routes = complete_routes(
        ("baseline", "current_frame", "temporal")
    )
    if lock.get("pilot", False) and len(accepted_jobs) != len(rows):
        raise RuntimeError(
            f"Pilot requires all {len(rows)} jobs; accepted {len(accepted_jobs)}"
        )
    if lock.get("pilot", False):
        receipt_paths = sorted((root / "jobs").glob("*/attempts/attempt-*/receipt.json"))
        retry_count = len(receipt_paths) - len(accepted_jobs)
        total_attempt_seconds = 0.0
        for receipt_path in receipt_paths:
            receipt = load_json(receipt_path)
            started = datetime.fromisoformat(receipt["started_at"])
            finished = datetime.fromisoformat(receipt["finished_at"])
            total_attempt_seconds += max(0.0, (finished - started).total_seconds())
        buffer = float(protocol["pilot_projection_buffer"])
        full_job_count = int(protocol["expected_route_count"]) * len(
            protocol["evaluation_seeds"]
        ) * 3
        projected_runtime_days = (
            total_attempt_seconds / len(accepted_jobs) * full_job_count * buffer / 86400.0
        )
        pilot_bytes = sum(
            path.stat().st_size for path in (root / "jobs").rglob("*") if path.is_file()
        )
        projected_storage_gib = (
            pilot_bytes / len(accepted_jobs) * full_job_count * buffer / 1024**3
        )
        pilot_report = {
            "accepted_jobs": len(accepted_jobs),
            "technical_retries": retry_count,
            "projected_runtime_days_with_buffer": projected_runtime_days,
            "projected_storage_gib_with_buffer": projected_storage_gib,
            "passed": (
                retry_count <= int(protocol["pilot_maximum_technical_retries"])
                and projected_runtime_days
                <= float(protocol["pilot_projected_runtime_limit_days"])
                and projected_storage_gib
                <= float(protocol["pilot_projected_storage_limit_gib"])
            ),
        }
        atomic_write_json(root / "pilot_report.json", pilot_report)
        if not pilot_report["passed"]:
            raise RuntimeError(f"Infrastructure pilot gates failed: {pilot_report}")
    primary_routes = contrast_routes["temporal_vs_baseline"]
    excluded_routes = sorted(set(jobs_by_route) - set(primary_routes))
    minimum_routes = int(len(jobs_by_route) * 0.95 + 0.999999)
    if len(primary_routes) < minimum_routes:
        raise RuntimeError(
            f"Only {len(primary_routes)}/{len(jobs_by_route)} primary-complete routes; "
            f"minimum is {minimum_routes}"
        )
    for path in sorted((root / "jobs").rglob("*")):
        if path.is_file():
            files[str(path.relative_to(root))] = sha256_file(path)
    data_lock = {
        "campaign_id": lock["campaign_id"],
        "campaign_lock_sha256": lock["lock_sha256"],
        "schedule_sha256": lock["schedule_sha256"],
        "job_count": len(rows),
        "accepted_job_count": len(accepted_jobs),
        "accepted_job_ids": sorted(accepted_jobs),
        "primary_routes": primary_routes,
        "contrast_routes": contrast_routes,
        "complete_three_arm_routes": complete_three_arm_routes,
        "excluded_routes": excluded_routes,
        "missing_jobs": missing,
        "mechanism_missing_jobs": mechanism_missing_jobs,
        "mechanism_corrupt_jobs": mechanism_corrupt_jobs,
        "retry_summary": {
            "accepted_attempts_by_arm": dict(accepted_attempts_by_arm),
            "accepted_jobs_requiring_retry_by_arm": dict(retry_jobs_by_arm),
            "total_extra_attempts_for_accepted_jobs": int(
                sum(accepted_attempts_by_arm.values()) - len(accepted_jobs)
            ),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    atomic_write_json(root / "data.lock.json", data_lock)
    print(
        f"Locked {len(accepted_jobs)} accepted jobs and {len(files)} files; "
        f"primary analysis includes {len(primary_routes)}/{len(jobs_by_route)} routes"
    )


if __name__ == "__main__":
    main()
