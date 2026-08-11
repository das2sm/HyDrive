#!/usr/bin/env python3
"""Execute the frozen schedule sequentially without outcome inspection."""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
from pathlib import Path

from campaign_common import (
    artifact_root, atomic_write_json, ensure_artifact_identity, load_json,
    load_schedule,
)
from run_isolated_job import verify_lock


def run_isolated(command: list[str], cleanup_timeout: float) -> None:
    """Let the isolated runner clean CARLA before honoring Ctrl+C."""
    process = subprocess.Popen(command, start_new_session=True)
    try:
        process.wait()
    except KeyboardInterrupt:
        print(
            "Interrupt requested; waiting for isolated-job cleanup...",
            flush=True,
        )
        try:
            process.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=cleanup_timeout + 60.0)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "Isolated-job cleanup did not finish after interrupt; "
                "inspect CARLA processes before resuming"
            ) from exc
        raise SystemExit(130)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    protocol = load_json(args.protocol)
    lock = load_json(args.lock)
    if bool(args.pilot) != bool(lock.get("pilot", False)):
        raise ValueError("--pilot must match the frozen campaign lock")
    verify_lock(lock, args.lock, args.schedule)
    output_root = artifact_root(protocol, lock)
    if not output_root.is_dir() or not os_access_write(output_root):
        raise RuntimeError(f"Artifact root is not writable: {output_root}")
    ensure_artifact_identity(output_root, lock)

    rows = load_schedule(args.schedule)
    if args.pilot:
        pilot_routes = set(protocol.get("pilot_routes", []))
        rows = [row for row in rows if row["route"] in pilot_routes]
        if not rows:
            raise ValueError("No pilot routes configured")
    maximum_attempts = int(protocol["maximum_attempts"])
    runner = Path(__file__).with_name("run_isolated_job.py")
    accepted_jobs = {
        row["job_id"] for row in rows
        if (output_root / "jobs" / row["job_id"] / "final_receipt.json").is_file()
    }
    terminal_jobs = set()
    completed_steps = 0
    for attempt in range(1, maximum_attempts + 1):
        for row in rows:
            if row["job_id"] in accepted_jobs or row["job_id"] in terminal_jobs:
                continue
            job_root = output_root / "jobs" / row["job_id"]
            final_path = job_root / "final_receipt.json"
            if attempt > 1:
                previous_path = (
                    job_root / "attempts" / f"attempt-{attempt - 1:02d}" / "receipt.json"
                )
                if not previous_path.is_file():
                    raise RuntimeError(
                        f"Missing prior receipt before retrying {row['job_id']}"
                    )
                previous = load_json(previous_path)["classification"]
                if previous["valid"]:
                    raise RuntimeError(
                        f"Valid prior attempt lacks final receipt: {row['job_id']}"
                    )
                if not previous["retryable"]:
                    terminal_jobs.add(row["job_id"])
                    continue
            receipt_path = job_root / "attempts" / f"attempt-{attempt:02d}" / "receipt.json"
            if not receipt_path.is_file():
                command = [
                    sys.executable, str(runner), "--protocol", str(args.protocol),
                    "--lock", str(args.lock), "--schedule", str(args.schedule),
                    "--job-id", row["job_id"], "--attempt", str(attempt),
                ]
                run_isolated(
                    command,
                    float(protocol.get("cleanup_timeout_seconds", 90)),
                )
            if not receipt_path.is_file():
                raise RuntimeError(f"Runner produced no receipt for {row['job_id']}")
            receipt = load_json(receipt_path)
            classification = receipt["classification"]
            if not receipt.get("cleanup_ok", False):
                raise RuntimeError(f"Cleanup failed for {row['job_id']}; campaign stopped")
            if classification["valid"]:
                final = {
                    "job_id": row["job_id"], "accepted_attempt": attempt,
                    "receipt": str(receipt_path), "classification": classification,
                }
                atomic_write_json(final_path, final)
                accepted_jobs.add(row["job_id"])
            elif not classification["retryable"]:
                terminal_jobs.add(row["job_id"])
            completed_steps += 1
            print(
                f"[pass {attempt}/{maximum_attempts}; step {completed_steps}] "
                f"accepted={len(accepted_jobs)} pending={len(rows) - len(accepted_jobs) - len(terminal_jobs)}",
                flush=True,
            )
    exhausted = len(rows) - len(accepted_jobs)
    atomic_write_json(output_root / "campaign_run_summary.json", {
        "campaign_id": lock["campaign_id"], "scheduled": len(rows),
        "accepted": len(accepted_jobs), "exhausted": exhausted,
        "pilot": args.pilot, "retry_mode": "deferred_passes",
    })
    raise SystemExit(0 if exhausted == 0 else 3)


def os_access_write(path: Path) -> bool:
    import os
    return os.access(path, os.W_OK)


if __name__ == "__main__":
    main()
