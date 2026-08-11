"""Lightweight tests for the public HyDrive release."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VERIFY = load_module("verify_artifact", ROOT / "analysis/verify_artifact.py")
PCO = load_module("pco", ROOT / "intervention/pco.py")
CAMPAIGN = load_module(
    "campaign_common", ROOT / "runtime/campaign_common.py"
)


def test_manifest_schedule_and_point_estimates():
    VERIFY.verify_manifest(ROOT)
    VERIFY.verify_schedule(ROOT)
    VERIFY.verify_results(ROOT, full_bootstrap=False)


def test_manifest_excludes_mutable_repository_documentation():
    entries = {
        line.split("  ", 1)[1]
        for line in (ROOT / "MANIFEST.sha256").read_text(
            encoding="ascii"
        ).splitlines()
    }
    assert "README.md" not in entries
    assert not any(path.startswith("docs/") for path in entries)
    assert not any(path.startswith("licenses/") for path in entries)


def test_primary_collision_result_recomputes_from_public_rows():
    rows = VERIFY.load_rows(ROOT / "data/job_outcomes.csv")
    deltas, families = VERIFY.route_deltas(
        rows, "temporal", "baseline", "collision"
    )
    assert len(deltas) == 185
    assert len(set(families.values())) == 97
    assert np.isclose(np.average(list(deltas.values())), -0.039639639639639644)


def test_route_family_structure_is_documented_correctly():
    with (ROOT / "data/schedule.csv").open(
        newline="", encoding="utf-8"
    ) as infile:
        rows = list(csv.DictReader(infile))
    family_routes: dict[str, set[str]] = {}
    for row in rows:
        family_routes.setdefault(row["family"], set()).add(row["route"])
    sizes = sorted(len(routes) for routes in family_routes.values())
    assert sizes.count(1) == 9
    assert sizes.count(2) == 88


def test_binary_selector_chooses_highest_scored_clear_candidate():
    scores = np.asarray([0.95, 0.80, 0.70])
    costs = np.asarray([1.0, 0.0, 0.0])
    trajectories = np.zeros((3, 6, 2), dtype=np.float64)
    selected, reason, _ = PCO.select_pco_candidate(
        scores,
        costs,
        np.asarray([0, 1, 2]),
        trajectories,
        original_mode_index=0,
        selection_policy="binary_veto",
    )
    assert selected == 1
    assert reason == "filtered"


def test_binary_selector_retains_original_when_all_candidates_are_unsafe():
    selected, reason, _ = PCO.select_pco_candidate(
        np.asarray([0.9, 0.8]),
        np.asarray([1.0, 1.0]),
        np.asarray([0, 1]),
        np.zeros((2, 6, 2), dtype=np.float64),
        original_mode_index=0,
        selection_policy="binary_veto",
    )
    assert selected is None
    assert reason == "all_unsafe"


def test_public_protocol_contains_no_machine_specific_paths():
    protocol = json.loads(
        (ROOT / "configs/protocol.example.json").read_text(encoding="utf-8")
    )
    serialized = json.dumps(protocol)
    assert "/home/" not in serialized
    assert "/media/" not in serialized
    assert protocol["evaluation_seeds"] == [1, 2, 3]
    assert protocol["bootstrap_replicates"] == 20000
    assert protocol["minimum_free_gib"] == 10
    assert protocol["model_assets"][
        "data/kmeans/trajectory_1024_256.npz"
    ] == "5bc170d6ec03627ae568161d4b071ecc1bac593cb2e25304060580dcfff7f321"


def test_runtime_preflight_requires_fail2drive_campaign_controls():
    preflight = (ROOT / "runtime/preflight.py").read_text(encoding="utf-8")
    assert "verify_fail2drive_campaign_controls" in preflight
    assert "verify_sparsedrive_campaign_outputs" in preflight
    assert "verify_runtime_imports" in preflight
    assert "import carla, einops, torch; import sparsedrive_b2d_agent_occ" in preflight
    assert "--no-resume" in preflight
    assert "HYDRIVE_SCENARIO_INIT_OK" in preflight


def test_evaluator_uses_protocol_python_executable():
    runner = (ROOT / "runtime/run_isolated_job.py").read_text(encoding="utf-8")
    wrapper = (ROOT / "runtime/run_evaluation.sh").read_text(encoding="utf-8")
    assert 'env["HYDRIVE_PYTHON_EXECUTABLE"] = python_executable' in runner
    assert 'CAMPAIGN_PYTHON="${HYDRIVE_PYTHON_EXECUTABLE:-python}"' in wrapper


def test_campaign_interrupt_waits_for_isolated_cleanup():
    campaign = (ROOT / "runtime/run_campaign.py").read_text(encoding="utf-8")
    isolated = (ROOT / "runtime/run_isolated_job.py").read_text(encoding="utf-8")
    assert "start_new_session=True" in campaign
    assert "waiting for isolated-job cleanup" in campaign
    assert '"reason": "operator_interrupt"' in isolated


def test_nonzero_evaluator_exit_is_reported_before_missing_output(tmp_path):
    result = CAMPAIGN.classify_official_result(
        tmp_path / "missing.json", evaluator_exit_code=2
    )
    assert result == {
        "valid": False,
        "retryable": True,
        "reason": "nonzero_evaluator_exit:2",
    }


def test_agent_setup_failure_is_reported_before_missing_sensor_manifest(tmp_path):
    result_path = tmp_path / "official.json"
    result_path.write_text(json.dumps({
        "_checkpoint": {
            "progress": [1, 1],
            "global_record": {"status": "Failed"},
            "records": [{
                "route_id": "RouteScenario_75_rep0",
                "status": "Failed - Agent couldn't be set up",
            }],
        },
        "entry_status": "Finished",
        "eligible": True,
        "sensors": [],
    }), encoding="utf-8")
    result = CAMPAIGN.classify_official_result(result_path)
    assert result["reason"] == "software_status:Failed - Agent couldn't be set up"


def test_release_metadata_matches_the_final_report():
    title = (
        "Evaluating Privileged Occupancy Filtering for "
        "Fixed-Vocabulary End-to-End Driving"
    )
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert title in readme.replace("\n", " ")
    assert title in citation
    assert "das2sm/HyDrive" in readme


def test_public_release_contains_no_local_machine_paths():
    text_extensions = {
        ".cff", ".csv", ".json", ".md", ".py", ".tex", ".txt", ".yml"
    }
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or path == Path(__file__)
            or path.suffix.lower() not in text_extensions
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert "/home/ace428/" not in text, path
        assert "/media/ace428/" not in text, path


def test_upstream_license_texts_and_modification_notice_are_present():
    required = {
        "SPARSEDRIVEV2_LICENSE": "Apache License",
        "FAIL2DRIVE_LICENSE": "MIT License",
        "BENCH2DRIVE_LICENSE": "MIT License",
        "CARLA_LEADERBOARD_LICENSE": "MIT License",
        "CARLA_SCENARIO_RUNNER_LICENSE": "MIT License",
    }
    for filename, marker in required.items():
        text = (ROOT / "licenses" / filename).read_text(encoding="utf-8")
        assert marker in text

    agent = (
        ROOT / "intervention" / "sparsedrive_b2d_agent_occ.py"
    ).read_text(encoding="utf-8")
    assert "Derived from the SparseDriveV2/Bench2Drive evaluation agent" in agent
    assert "Apache-2.0" in agent
