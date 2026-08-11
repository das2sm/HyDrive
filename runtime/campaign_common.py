"""Shared contracts for the isolated HyDrive multi-seed campaign."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path


ARMS = ("baseline", "current_frame", "temporal")
ARM_ENV = {
    "baseline": {
        "PCO_RESCORE_ENABLED": "0",
        "PCO_OCCUPANCY_SOURCE": "current_frame",
    },
    "current_frame": {
        "PCO_RESCORE_ENABLED": "1",
        "PCO_OCCUPANCY_SOURCE": "current_frame",
    },
    "temporal": {
        "PCO_RESCORE_ENABLED": "1",
        "PCO_OCCUPANCY_SOURCE": "temporal_cv",
    },
}
SHARED_PCO_ENV = {
    "PCO_RESCORE_COST_MODE": "binary",
    "PCO_SELECTION_POLICY": "binary_veto",
    "PCO_RESCORE_TOPK": "0",
    "PCO_RESCORE_SCORE_SOURCE": "post_rescore",
    "PCO_RESCORE_FALLBACK": "keep_original",
    "PCO_HYSTERESIS_MARGIN": "0",
    "PCO_RESCORE_LOG_FULL_COSTS": "0",
}
VALID_DRIVING_STATUSES = {
    "Perfect",
    "Completed",
    "Failed",
    "Failed - Agent deviated from the route",
    "Failed - Agent got blocked",
    "Failed - Agent timed out",
}
RETRYABLE_SOFTWARE_STATUSES = {
    "Failed - Agent couldn't be set up",
    "Failed - Agent crashed",
    "Failed - Simulation crashed",
}
FATAL_CONFIGURATION_STATUSES = {
    "Failed - Agent's sensors were invalid",
}
OFFICIAL_INFRACTION_KEYS = {
    "collisions_layout",
    "collisions_pedestrian",
    "collisions_vehicle",
    "red_light",
    "stop_infraction",
    "outside_route_lanes",
    "min_speed_infractions",
    "yield_emergency_vehicle_infractions",
    "scenario_timeouts",
    "route_dev",
    "vehicle_blocked",
    "route_timeout",
}
INHERITED_RUNTIME_SELECTORS = {
    "CARLA_ROOT",
    "F2D_DIR",
    "LEADERBOARD_ROOT",
    "SCENARIO_RUNNER_ROOT",
    "PYTHONPATH",
    "RECORD_PATH",
    "RESUME",
    "SAVE_PATH",
    "VIZ_PATH",
    "OVERWRITE_EXISTING",
    "TEAM_AGENT",
    "TEAM_CONFIG",
    "ROUTES",
    "PORT",
    "TM_PORT",
    "GPU_RANK",
    "REPETITIONS",
}
ROUTE_RE = re.compile(r"^(Base|Generalization)_(.+)_(\d{4})$")


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: os.PathLike[str] | str, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, delete=False
    ) as outfile:
        json.dump(value, outfile, indent=2, sort_keys=True)
        outfile.write("\n")
        temp_path = Path(outfile.name)
    temp_path.replace(target)


def load_json(path: os.PathLike[str] | str) -> dict:
    with open(path, encoding="utf-8") as infile:
        value = json.load(infile)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def load_schedule(path: os.PathLike[str] | str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as infile:
        rows = list(csv.DictReader(infile))
    if not rows:
        raise ValueError(f"Empty campaign schedule: {path}")
    job_ids = [row["job_id"] for row in rows]
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("Campaign schedule contains duplicate job IDs")
    return rows


def artifact_root(protocol: dict, lock: dict) -> Path:
    root = Path(protocol["artifact_root"])
    return root / "pilot" if lock.get("pilot", False) else root


def ensure_artifact_identity(root: Path, lock: dict, create: bool = False) -> None:
    """Bind an artifact directory to exactly one frozen campaign lock."""
    marker = root / ".campaign_identity.json"
    expected = {
        "campaign_id": lock["campaign_id"],
        "lock_sha256": lock["lock_sha256"],
        "schedule_sha256": lock["schedule_sha256"],
    }
    if marker.is_file():
        if load_json(marker) != expected:
            raise RuntimeError(f"Artifact root belongs to another campaign: {root}")
        return
    if not create:
        raise RuntimeError(f"Artifact root has no frozen identity marker: {root}")
    jobs_root = root / "jobs"
    if jobs_root.exists() and any(jobs_root.iterdir()):
        raise RuntimeError(f"Refusing to adopt a nonempty jobs directory: {jobs_root}")
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(marker, expected)


def route_parts(route_name: str) -> tuple[str, str, int, str]:
    match = ROUTE_RE.match(route_name)
    if not match:
        raise ValueError(f"Invalid Fail2Drive route name: {route_name}")
    split, scenario, route_id_raw = match.groups()
    route_id = int(route_id_raw)
    family = f"{scenario}_{route_id % 1000:03d}"
    return split, scenario, route_id, family


def official_record(result_path: os.PathLike[str] | str) -> dict:
    payload = load_json(result_path)
    records = payload.get("_checkpoint", {}).get("records") or []
    if len(records) != 1 or not isinstance(records[0], dict):
        raise ValueError("Official result must contain exactly one route record")
    return records[0]


def classify_official_result(
    result_path: os.PathLike[str] | str,
    expected_route: str | None = None,
    evaluator_exit_code: int | None = 0,
    evaluator_log_path: os.PathLike[str] | str | None = None,
) -> dict:
    if evaluator_exit_code not in (None, 0):
        return {
            "valid": False,
            "retryable": True,
            "reason": f"nonzero_evaluator_exit:{evaluator_exit_code}",
        }
    try:
        payload = load_json(result_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {"valid": False, "retryable": True, "reason": str(exc)}
    checkpoint = payload.get("_checkpoint")
    if not isinstance(checkpoint, dict):
        return {"valid": False, "retryable": True, "reason": "missing_checkpoint"}
    if checkpoint.get("progress") != [1, 1]:
        return {
            "valid": False,
            "retryable": True,
            "reason": f"incomplete_checkpoint_progress:{checkpoint.get('progress')!r}",
        }
    records = checkpoint.get("records")
    if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict):
        return {
            "valid": False,
            "retryable": True,
            "reason": "official_result_must_have_one_record",
        }
    if not isinstance(checkpoint.get("global_record"), dict) or not checkpoint["global_record"]:
        return {"valid": False, "retryable": True, "reason": "missing_global_record"}
    if payload.get("entry_status") != "Finished" or payload.get("eligible") is not True:
        return {
            "valid": False,
            "retryable": True,
            "reason": f"incomplete_entry_status:{payload.get('entry_status')!r}",
        }

    record = records[0]
    status = str(record.get("status") or "")
    if status in RETRYABLE_SOFTWARE_STATUSES:
        return {
            "valid": False,
            "retryable": True,
            "reason": f"software_status:{status}",
            "status": status,
        }
    if status in FATAL_CONFIGURATION_STATUSES:
        return {
            "valid": False,
            "retryable": False,
            "reason": f"configuration_status:{status}",
            "status": status,
        }
    if status not in VALID_DRIVING_STATUSES:
        return {
            "valid": False,
            "retryable": True,
            "reason": f"unknown_status:{status}",
            "status": status,
        }

    if not isinstance(payload.get("sensors"), list) or not payload["sensors"]:
        return {"valid": False, "retryable": True, "reason": "missing_sensor_manifest"}
    if not isinstance(payload.get("values"), list) or not payload["values"]:
        return {"valid": False, "retryable": True, "reason": "missing_global_values"}
    if not isinstance(payload.get("labels"), list) or not payload["labels"]:
        return {"valid": False, "retryable": True, "reason": "missing_global_labels"}

    if expected_route is not None:
        _split, _scenario, route_id, _family = route_parts(expected_route)
        expected_id = f"RouteScenario_{route_id}_rep0"
        if record.get("route_id") != expected_id:
            return {
                "valid": False,
                "retryable": True,
                "reason": (
                    f"wrong_route_record:{record.get('route_id')!r}!={expected_id!r}"
                ),
                "status": status,
            }
    infractions = record.get("infractions")
    if not isinstance(infractions, dict):
        return {
            "valid": False, "retryable": True,
            "reason": "missing_official_infractions", "status": status,
        }
    missing_infractions = sorted(OFFICIAL_INFRACTION_KEYS - set(infractions))
    invalid_infractions = sorted(
        key for key in OFFICIAL_INFRACTION_KEYS
        if key in infractions and not isinstance(infractions[key], list)
    )
    if missing_infractions or invalid_infractions:
        return {
            "valid": False,
            "retryable": True,
            "reason": (
                f"invalid_infraction_schema:missing={missing_infractions},"
                f"nonlists={invalid_infractions}"
            ),
            "status": status,
        }
    scores = record.get("scores") or {}
    score_ranges = {
        "score_route": (0.0, 100.0),
        "score_penalty": (0.0, 1.0),
        "score_composed": (0.0, 100.0),
    }
    invalid_scores = []
    for name, (lower, upper) in score_ranges.items():
        value = scores.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not lower <= float(value) <= upper
        ):
            invalid_scores.append(name)
    if invalid_scores:
        return {
            "valid": False,
            "retryable": True,
            "reason": f"invalid_official_scores:{','.join(invalid_scores)}",
            "status": status,
        }
    if evaluator_log_path is not None:
        try:
            log_text = Path(evaluator_log_path).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError as exc:
            return {
                "valid": False, "retryable": True,
                "reason": f"unreadable_evaluator_log:{exc}", "status": status,
            }
        invalid_markers = (
            "Skipping scenario '",
            "WARNING: Ignoring scenario '",
        )
        found = [marker for marker in invalid_markers if marker in log_text]
        if found:
            return {
                "valid": False, "retryable": True,
                "reason": f"scenario_initialization_warning:{found}",
                "status": status,
            }
        if "HYDRIVE_SCENARIO_INIT_OK" not in log_text:
            return {
                "valid": False, "retryable": True,
                "reason": "missing_strict_scenario_initialization_marker",
                "status": status,
            }
    return {
        "valid": True,
        "retryable": False,
        "reason": "official_record_complete",
        "status": status,
    }


def strict_job_environment(base: dict[str, str], job: dict, lock: dict) -> dict[str, str]:
    env = {
        key: value
        for key, value in base.items()
        if (
            not key.startswith("PCO_")
            and not key.startswith("HYDRIVE_")
            and key not in INHERITED_RUNTIME_SELECTORS
        )
    }
    arm = job["arm"]
    if arm not in ARM_ENV:
        raise ValueError(f"Unknown campaign arm: {arm}")
    env.update(SHARED_PCO_ENV)
    env.update(ARM_ENV[arm])
    env.update({
        "HYDRIVE_STRICT_CAMPAIGN": "1",
        "HYDRIVE_CAMPAIGN_ID": lock["campaign_id"],
        "HYDRIVE_CAMPAIGN_LOCK_SHA256": lock["lock_sha256"],
        "HYDRIVE_SCHEDULE_SHA256": lock["schedule_sha256"],
        "HYDRIVE_JOB_ID": job["job_id"],
        "HYDRIVE_ARM": arm,
        "HYDRIVE_ATTEMPT": str(job["attempt"]),
        "HYDRIVE_EVAL_SEED": str(job["evaluation_seed"]),
        "HYDRIVE_ORDER_POSITION": str(job["order_position"]),
        "HYDRIVE_LOG_PROFILE": "compact_campaign",
        "HYDRIVE_DISABLE_VISUALIZER": "1",
        "HYDRIVE_EARLY_TRUNCATION": "0",
        "HYDRIVE_DIAGNOSTIC_COLLISION_SENSOR": "0",
        "TRAFFIC_MANAGER_SEED": str(job["traffic_manager_seed"]),
        "AGENT_SEED": str(lock["agent_seed"]),
        "PYTHONHASHSEED": str(lock["agent_seed"]),
        "HYDRIVE_CONFIG_SHA256": lock["file_sha256"]["config"],
        "HYDRIVE_CHECKPOINT_SHA256": lock["file_sha256"]["checkpoint"],
        "HYDRIVE_FIXED_DELTA_SECONDS": str(lock["fixed_delta_seconds"]),
    })
    return env
