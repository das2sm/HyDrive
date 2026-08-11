#!/usr/bin/env python3
"""Fail-fast host checks before a pilot or full campaign."""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
from pathlib import Path

from campaign_common import (
    artifact_root, ensure_artifact_identity, load_json, load_schedule,
)
from run_isolated_job import verify_lock


def verify_fail2drive_campaign_controls(repo: Path) -> None:
    """Fail before CARLA launch when the required Fail2Drive patch is absent."""
    required_markers = {
        repo / "Fail2Drive/fail2drive_leaderboard/leaderboard/leaderboard_evaluator.py": (
            "--no-resume",
            "parser.set_defaults(resume=False)",
        ),
        repo / "Fail2Drive/fail2drive_leaderboard/leaderboard/scenarios/scenario_manager.py": (
            "HYDRIVE_SCENARIO_INIT_OK",
            "HYDRIVE_STRICT_CAMPAIGN",
        ),
        repo / "Fail2Drive/fail2drive_leaderboard/leaderboard/scenarios/route_scenario.py": (
            "HYDRIVE_STRICT_CAMPAIGN",
        ),
    }
    missing = []
    for path, markers in required_markers.items():
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            missing.append(str(path.relative_to(repo)))
            continue
        absent = [marker for marker in markers if marker not in text]
        if absent:
            missing.append(
                f"{path.relative_to(repo)} ({', '.join(absent)})"
            )
    if missing:
        details = "; ".join(missing)
        raise RuntimeError(
            "Fail2Drive campaign-control patch is missing or incomplete: "
            f"{details}. Apply patches/fail2drive_campaign_controls.patch "
            "as described in docs/SETUP.md, commit the nested Fail2Drive "
            "worktree, and generate a new campaign lock."
        )


def verify_sparsedrive_campaign_outputs(repo: Path) -> None:
    """Require the decoder fields consumed by the occupancy intervention."""
    decoder = repo / "projects/mmdet3d_plugin/models/motion/decoder.py"
    required = (
        'output[b]["traj_reg"]',
        'output[b]["traj_cls_logits_post_rescore"]',
        'output[b]["traj_rescore_mask"]',
        'output[b]["traj_selected_mode_index"]',
    )
    try:
        source = decoder.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Cannot read SparseDriveV2 decoder: {decoder}") from exc
    missing = [marker for marker in required if marker not in source]
    if missing:
        raise RuntimeError(
            "SparseDriveV2 planner-output patch is missing: "
            f"{', '.join(missing)}. Apply "
            "patches/sparsedrive_planner_outputs.patch as described in "
            "docs/SETUP.md, commit the runtime worktree, and generate a new "
            "campaign lock."
        )


def verify_runtime_imports(repo: Path, python_executable: str) -> None:
    """Import the campaign agent before spending time launching CARLA."""
    env = os.environ.copy()
    python_paths = [
        repo / "leaderboard/team_code",
        repo / "Fail2Drive/fail2drive_leaderboard",
        repo / "Fail2Drive/fail2drive_scenario_runner",
        repo / "leaderboard",
        repo,
        repo / "f2d_carla/PythonAPI/carla",
    ]
    if env.get("PYTHONPATH"):
        python_paths.append(Path(env["PYTHONPATH"]))
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in python_paths)
    probe = subprocess.run(
        [
            python_executable,
            "-c",
            "import carla, einops, torch; import sparsedrive_b2d_agent_occ",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo,
        env=env,
    )
    if probe.returncode != 0:
        details = (probe.stderr or probe.stdout).strip()
        raise RuntimeError(f"Campaign Python dependency check failed: {details}")


def port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_json(args.protocol)
    lock = load_json(args.lock)
    verify_lock(lock, args.lock, args.schedule)
    repo = Path(lock["repo_root"])
    verify_fail2drive_campaign_controls(repo)
    verify_sparsedrive_campaign_outputs(repo)
    verify_runtime_imports(repo, str(protocol["python_executable"]))
    rows = load_schedule(args.schedule)
    root = artifact_root(protocol, lock)
    root.mkdir(parents=True, exist_ok=True)
    ensure_artifact_identity(root, lock)
    probe = root / ".write_probe"
    with probe.open("w", encoding="ascii") as outfile:
        outfile.write("ok\n")
        outfile.flush()
        os.fsync(outfile.fileno())
    probe.unlink()
    directory_fd = os.open(root, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    free = shutil.disk_usage(root).free
    minimum_free_gib = int(protocol["minimum_free_gib"])
    if free < minimum_free_gib * 1024**3:
        raise RuntimeError(f"Less than {minimum_free_gib} GiB free at {root}")
    ports = {
        "CARLA RPC": int(protocol["rpc_port"]),
        "CARLA streaming": int(protocol["streaming_port"]),
        "Traffic Manager": int(protocol["traffic_manager_port"]),
    }
    for name, port in ports.items():
        if not port_available(port):
            raise RuntimeError(f"{name} port is already in use: {port}")
    python_executable = Path(protocol["python_executable"])
    if not os.access(python_executable, os.X_OK):
        raise RuntimeError(f"Campaign Python is not executable: {python_executable}")
    subprocess.run(
        [
            str(python_executable), "-c",
            "import carla,torch; assert torch.cuda.is_available(); "
            "assert torch.cuda.device_count() >= 1",
        ], check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(["nvidia-smi"], check=True, stdout=subprocess.DEVNULL)
    carla = Path(lock["repo_root"]) / protocol["carla_root"] / "CarlaUE4.sh"
    if not os.access(carla, os.X_OK):
        raise RuntimeError(f"CARLA launcher is not executable: {carla}")
    print(f"Preflight passed: {len(rows)} jobs, {free / 1024**3:.1f} GiB free")


if __name__ == "__main__":
    main()
