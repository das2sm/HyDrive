#!/usr/bin/env python3
"""Run one route/seed/arm in fresh, owned process groups."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import signal
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from campaign_common import (
    atomic_write_json,
    artifact_root,
    canonical_json_sha256,
    classify_official_result,
    load_json,
    load_schedule,
    sha256_file,
    strict_job_environment,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def prepare_attempt_directory(job_root: Path, attempt: int) -> Path:
    attempts_root = job_root / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    attempt_dir = attempts_root / f"attempt-{attempt:02d}"
    if attempt_dir.exists():
        receipt = attempt_dir / "receipt.json"
        if receipt.is_file():
            raise FileExistsError(f"Attempt already has a receipt: {attempt_dir}")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        orphan = attempts_root / f"attempt-{attempt:02d}.orphaned-{stamp}"
        suffix = 0
        while orphan.exists():
            suffix += 1
            orphan = attempts_root / (
                f"attempt-{attempt:02d}.orphaned-{stamp}-{suffix:02d}"
            )
        attempt_dir.rename(orphan)
        atomic_write_json(orphan / "orphaned.json", {
            "attempt": attempt,
            "orphaned_at": utc_now(),
            "reason": "attempt_directory_existed_without_receipt",
        })
    attempt_dir.mkdir()
    return attempt_dir


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def port_available(port: int) -> bool:
    """True when the port can be bound.

    SO_REUSEADDR makes lingering TIME_WAIT connections from a torn-down
    server acceptable while a live listener still fails the bind; without
    it, CARLA's own closed connections block the port for up to a minute
    after every clean teardown.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def wait_for_cleanup(
    server: subprocess.Popen | None,
    evaluator: subprocess.Popen | None,
    ports: tuple[int, ...],
    timeout: float,
) -> dict:
    """Poll teardown conditions until they hold or the timeout expires.

    UE4 runs a crash/shutdown handler that can keep the process group and
    its sockets alive for several seconds after SIGTERM/SIGKILL; a one-shot
    check misclassifies that routine lag as a cleanup failure and aborts
    the campaign. Cleanup is a state to be reached, not an instant.
    """
    deadline = time.monotonic() + timeout
    start = time.monotonic()
    while True:
        state = {
            "processes_reaped": (
                (server is None or server.poll() is not None)
                and (evaluator is None or evaluator.poll() is not None)
            ),
            "process_groups_gone": (
                process_group_gone(server) and process_group_gone(evaluator)
            ),
            "ports_released": all(port_available(port) for port in ports),
        }
        state["cleanup_ok"] = all(state.values())
        state["cleanup_wait_seconds"] = round(time.monotonic() - start, 2)
        if state["cleanup_ok"] or time.monotonic() >= deadline:
            return state
        time.sleep(1.0)


def wait_for_carla_health(
    python_executable: str, port: int, timeout: float, env: dict[str, str]
) -> bool:
    code = (
        "import carla,sys; "
        "client=carla.Client('127.0.0.1',int(sys.argv[1])); "
        "client.set_timeout(5.0); world=client.get_world(); "
        "assert world.get_map().name; assert world.get_snapshot().frame >= 0"
    )
    deadline = time.monotonic() + timeout
    consecutive = 0
    while time.monotonic() < deadline:
        try:
            result = subprocess.run(
                [python_executable, "-c", code, str(port)],
                env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=10, check=False,
            )
            healthy = result.returncode == 0
        except subprocess.TimeoutExpired:
            healthy = False
        if healthy:
            consecutive += 1
            if consecutive == 2:
                return True
        else:
            consecutive = 0
        time.sleep(1.0)
    return False


def stop_process_group(process: subprocess.Popen | None, grace: float = 20.0) -> None:
    """Terminate the whole process group, escalating on the group's state.

    The escalation decision must key on the *group*, not the direct child:
    CarlaUE4.sh exits on SIGTERM within milliseconds while a wedged
    CarlaUE4-Linux-Shipping grandchild in the same group can ignore SIGTERM
    indefinitely. Waiting on the child and returning early leaves that
    grandchild alive and listening (observed in campaign job
    Base_BadParking_0009__s3__current_frame).
    """
    if process is None:
        return
    process.poll()
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass
        if process_group_gone(process):
            return
        time.sleep(0.5)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def process_group_gone(process: subprocess.Popen | None) -> bool:
    if process is None:
        return True
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def verify_lock(
    lock: dict, lock_path: Path, schedule_path: Path, verify_contents: bool = True
) -> None:
    expected = dict(lock)
    recorded = expected.pop("lock_sha256")
    if canonical_json_sha256(expected) != recorded:
        raise ValueError(f"Campaign lock is internally inconsistent: {lock_path}")
    if sha256_file(schedule_path) != lock["schedule_sha256"]:
        raise ValueError("Schedule hash differs from frozen lock")
    for name, path in lock["paths"].items():
        path = Path(path)
        stat = path.stat()
        expected_stat = lock["file_stat"][name]
        if stat.st_size != expected_stat["size"] or stat.st_mtime_ns != expected_stat["mtime_ns"]:
            raise ValueError(f"Frozen {name} identity changed: {path}")
        if verify_contents and sha256_file(path) != lock["file_sha256"][name]:
            raise ValueError(f"Frozen {name} changed: {path}")
    repositories = (
        (Path(lock["repo_root"]), lock["git_commit"]),
        (Path(lock["repo_root"]) / "Fail2Drive", lock["fail2drive_git_commit"]),
    )
    for repo, expected_commit in repositories:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
        if commit != expected_commit:
            raise ValueError(f"Repository commit changed after freeze: {repo}")
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True
        ).strip()
        if dirty:
            raise ValueError(f"Repository became dirty after freeze: {repo}")


def classify_compact_log(attempt_dir: Path, job: dict, lock: dict) -> dict:
    logs = list((attempt_dir / "route_logs").glob("*.pkl"))
    if len(logs) != 1:
        return {
            "valid": False,
            "reason": f"expected_one_compact_log_found_{len(logs)}",
        }
    try:
        with logs[0].open("rb") as infile:
            payload = pickle.load(infile)
    except (OSError, EOFError, pickle.UnpicklingError) as exc:
        return {"valid": False, "reason": f"invalid_log:{exc}"}
    if payload.get("config", {}).get("log_profile") != "compact_campaign":
        return {"valid": False, "reason": "wrong_log_profile"}
    if not payload.get("timesteps"):
        return {"valid": False, "reason": "empty_compact_log"}
    provenance = payload.get("provenance", {})
    expected = {
        "campaign_id": lock["campaign_id"],
        "campaign_lock_sha256": lock["lock_sha256"],
        "schedule_sha256": lock["schedule_sha256"],
        "job_id": job["job_id"],
        "arm": job["arm"],
        "attempt": int(job["attempt"]),
        "evaluation_seed": int(job["evaluation_seed"]),
        "order_position": int(job["order_position"]),
        "traffic_manager_seed": int(job["traffic_manager_seed"]),
        "synchronous_mode": True,
        "fixed_delta_seconds": lock["fixed_delta_seconds"],
    }
    mismatches = [key for key, value in expected.items() if provenance.get(key) != value]
    if mismatches:
        return {
            "valid": False,
            "reason": f"log_provenance_mismatch:{','.join(mismatches)}",
        }
    return {"valid": True, "reason": "compact_log_complete"}


def run_job(protocol: dict, lock: dict, row: dict, attempt: int) -> dict:
    output_root = artifact_root(protocol, lock)
    job_root = output_root / "jobs" / row["job_id"]
    attempt_dir = prepare_attempt_directory(job_root, attempt)
    result_path = attempt_dir / "official.json"
    repo = Path(lock["repo_root"])
    carla_root = repo / protocol["carla_root"]
    route_xml = repo / protocol["route_dir"] / f"{row['route']}.xml"
    if sha256_file(route_xml) != lock["route_xml_sha256"][row["route"]]:
        raise ValueError(f"Route XML changed after freeze: {route_xml}")
    rpc_port = int(protocol["rpc_port"])
    streaming_port = int(protocol["streaming_port"])
    tm_port = int(protocol["traffic_manager_port"])

    required_ports = (rpc_port, streaming_port, tm_port)
    if not all(port_available(port) for port in required_ports):
        raise RuntimeError(f"Required ports are already in use: {required_ports}")

    job = dict(row)
    job["attempt"] = attempt
    job["evaluation_seed"] = int(job["evaluation_seed"])
    job["traffic_manager_seed"] = int(job["traffic_manager_seed"])
    job["order_position"] = int(job["order_position"])
    job["job_dir"] = str(attempt_dir)
    env = strict_job_environment(os.environ.copy(), job, lock)
    env["HYDRIVE_JOB_DIR"] = str(attempt_dir)
    env["OVERWRITE_EXISTING"] = "0"
    env["RESUME"] = "False"
    env["F2D_DIR"] = str(repo / "Fail2Drive")
    env["CARLA_ROOT"] = str(carla_root)
    python_executable = str(protocol["python_executable"])
    env["HYDRIVE_PYTHON_EXECUTABLE"] = python_executable
    env["PATH"] = str(Path(python_executable).parent) + os.pathsep + env["PATH"]

    # Flag parity with Fail2Drive's released runner (slurm_evaluate.py, rgb
    # path): default Epic rendering quality — the released benchmark never
    # lowers it, and quality changes the camera imagery the agent sees —
    # plus a disabled primary server port.
    server_argv = [
        str(carla_root / "CarlaUE4.sh"),
        "-RenderOffScreen",
        "-nosound",
        f"-carla-rpc-port={rpc_port}",
        "-carla-primary-port=0",
        f"-carla-streaming-port={streaming_port}",
        "-graphicsadapter=0",
    ]
    team_config = "+".join((
        lock["paths"]["config"], lock["paths"]["checkpoint"], row["job_id"], "0"
    ))
    evaluator_argv = [
        "bash", str(repo / "leaderboard/scripts/run_evaluation.sh"),
        str(rpc_port), str(tm_port), "True", str(route_xml),
        lock["paths"]["agent"], team_config, str(result_path),
        str(attempt_dir / "recordings"), "only_traj", "0",
    ]
    receipt = {
        "campaign_id": lock["campaign_id"],
        "campaign_lock_sha256": lock["lock_sha256"],
        "schedule_sha256": lock["schedule_sha256"],
        "job": job,
        "attempt": attempt,
        "started_at": utc_now(),
        "server_argv": server_argv,
        "evaluator_argv": evaluator_argv,
        "ports": {
            "rpc": rpc_port, "streaming": streaming_port,
            "traffic_manager": tm_port,
        },
        "environment_sha256": canonical_json_sha256({
            key: env[key] for key in sorted(env)
            if key.startswith("PCO_") or key.startswith("HYDRIVE_")
            or key in {"TRAFFIC_MANAGER_SEED", "AGENT_SEED", "PYTHONHASHSEED"}
        }),
    }
    server = None
    evaluator = None
    try:
        with (attempt_dir / "carla.stdout.log").open("wb") as server_out, \
             (attempt_dir / "evaluator.stdout.log").open("wb") as evaluator_out:
            server = subprocess.Popen(
                server_argv, cwd=repo, env=env, stdout=server_out,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            receipt["server_pid"] = server.pid
            receipt["server_process_group"] = server.pid
            if not wait_for_carla_health(
                python_executable, rpc_port,
                int(protocol["server_start_timeout_seconds"]), env,
            ):
                receipt["classification"] = {
                    "valid": False, "retryable": True, "reason": "carla_start_timeout"
                }
            else:
                evaluator = subprocess.Popen(
                    evaluator_argv, cwd=repo, env=env, stdout=evaluator_out,
                    stderr=subprocess.STDOUT, start_new_session=True,
                )
                receipt["evaluator_pid"] = evaluator.pid
                receipt["evaluator_process_group"] = evaluator.pid
                try:
                    receipt["evaluator_exit_code"] = evaluator.wait(
                        timeout=int(protocol["job_timeout_seconds"])
                    )
                except subprocess.TimeoutExpired:
                    receipt["evaluator_exit_code"] = None
                    receipt["classification"] = {
                        "valid": False, "retryable": True, "reason": "job_timeout"
                    }
    except KeyboardInterrupt:
        receipt["classification"] = {
            "valid": False,
            "retryable": True,
            "reason": "operator_interrupt",
        }
        receipt["runner_exception"] = "KeyboardInterrupt"
    except Exception as exc:
        receipt["classification"] = {
            "valid": False,
            "retryable": True,
            "reason": f"runner_exception:{type(exc).__name__}",
        }
        receipt["runner_exception"] = repr(exc)
    finally:
        stop_process_group(evaluator)
        stop_process_group(server)

    cleanup_state = wait_for_cleanup(
        server, evaluator, required_ports,
        float(protocol.get("cleanup_timeout_seconds", 90)),
    )
    cleanup_ok = cleanup_state["cleanup_ok"]
    receipt.update(cleanup_state)
    if not cleanup_ok:
        survivors = []
        for owner, process in (("server", server), ("evaluator", evaluator)):
            if process is None:
                continue
            try:
                listing = subprocess.check_output(
                    ["ps", "-o", "pid=,stat=,comm=", "--pgid", str(process.pid)],
                    text=True,
                ).strip()
            except subprocess.CalledProcessError:
                listing = ""
            if listing:
                survivors.append({"group_of": owner, "pgid": process.pid,
                                  "processes": listing.splitlines()})
        receipt["cleanup_survivors"] = survivors
        # run_campaign aborts on cleanup_ok=False regardless; retryable=True
        # lets a *resumed* campaign re-attempt this job after the operator
        # (or time) has cleared the host, instead of excluding its route.
        receipt["classification"] = {
            "valid": False, "retryable": True, "reason": "process_cleanup_failed"
        }
    elif "classification" not in receipt:
        receipt["classification"] = classify_official_result(
            result_path,
            expected_route=row["route"],
            evaluator_exit_code=receipt.get("evaluator_exit_code"),
            evaluator_log_path=attempt_dir / "evaluator.stdout.log",
        )
        if receipt["classification"]["valid"]:
            receipt["compact_log_classification"] = classify_compact_log(
                attempt_dir, job, lock
            )
    receipt["finished_at"] = utc_now()
    if result_path.is_file():
        receipt["official_result_sha256"] = sha256_file(result_path)
    atomic_write_json(attempt_dir / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    args = parser.parse_args()
    protocol = load_json(args.protocol)
    lock = load_json(args.lock)
    verify_lock(lock, args.lock, args.schedule, verify_contents=False)
    matches = [row for row in load_schedule(args.schedule) if row["job_id"] == args.job_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one scheduled job {args.job_id!r}")
    receipt = run_job(protocol, lock, matches[0], args.attempt)
    print(json.dumps(receipt["classification"], sort_keys=True))
    raise SystemExit(0 if receipt["classification"]["valid"] else 2)


if __name__ == "__main__":
    main()
