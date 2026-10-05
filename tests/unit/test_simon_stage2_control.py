from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONTROL_PATH = ROOT / "src/neural_sat_speck/followups/stage2_control.py"


def _control() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_simon_stage2_control_test", CONTROL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _dataset(control: Any) -> dict[str, Any]:
    instances = []
    for key in range(8):
        parent = 1000 + key * 10
        child = 50000 + key
        instances.append(
            {
                "instance_index": key,
                "instance_id": f"simon32_64_demo_key_{key}",
                "master_key": [0xABCD, key, 0x1234, 0x9876],
                "attack_plaintexts": [
                    [[[i, 1], [i, 2]][0], [[i, 3], [i, 4]][0]] for i in range(64)
                ],
                "verification_plaintexts": [[1, 2], [3, 4]],
                "rounds_data": {
                    "10": {
                        "round_keys": [0x1000 + i for i in range(8)] + [child, parent],
                        "attack_ciphertexts": [
                            [[[i, 1], [i, 2]][0], [[i, 3], [i, 4]][0]]
                            for i in range(64)
                        ],
                        "verification_ciphertexts": [[5, 6], [7, 8]],
                    }
                },
            }
        )
    return {
        "seed": 999999,
        "split_role": "demo",
        "metadata": {
            "key_count_per_cipher": 8,
            "max_differential_pairs": 64,
            "verification_blocks_count": 2,
        },
        "ciphers": {"simon32_64": {"instances": instances}},
    }


def _ordered_stage1(key: int) -> list[int]:
    parents = [1000 + key * 10 + i for i in range(8)]
    tail = list(range(3000 + key * 512, 3000 + key * 512 + 504))
    return parents + tail


def _child_order(
    key: int, ordinal: int, truth_child: int, include_truth: bool
) -> list[int]:
    words = list(
        range(
            10000 + key * 4096 + ordinal * 512, 10000 + key * 4096 + ordinal * 512 + 512
        )
    )
    if include_truth:
        words[0] = truth_child
    return words


def _complete_row(control: Any, job: dict[str, Any], job_number: int) -> dict[str, Any]:
    invocation = f"{job_number + 1:032x}"
    key = job["key_index"]
    job_start = 10000.0 + job_number * 100.0
    event_complete = job_start + 1.0
    events: list[dict[str, Any]] = [
        {
            "schema": "stage2_rank_worker_event_v1",
            "event": "loaded",
            "event_index": 0,
            "invocation": invocation,
            "job_id": job["job_id"],
            "scorer": job["scorer"],
            "completed_monotonic": event_complete,
            "load_wall_seconds": 0.8,
            "event_path": "events/0000.json",
            "event_sha256": "a" * 64,
        }
    ]
    calls: list[dict[str, Any]] = []
    count = 1 if job["operation"] == "stage1" else 8
    stage1 = _ordered_stage1(key)
    for ordinal in range(count):
        started = event_complete + 0.1
        completed = event_complete + 0.5
        next_event = event_complete + 1.0
        if job["operation"] == "stage1":
            parent, words = None, stage1
        else:
            parent = stage1[ordinal]
            truth_child = 50000 + key
            include_truth = not (key == 0 and job["scorer"] == "ddt7")
            words = _child_order(key, ordinal, truth_child, include_truth)
        call = {
            "invocation": invocation,
            "job_id": job["job_id"],
            "ordinal": ordinal,
            "scorer": job["scorer"],
            "parent": parent,
            "ordered_words": words,
            "component_times_seconds": {"rank": 0.2},
            "call_started_monotonic": started,
            "call_completed_monotonic": completed,
            "phase_wall_seconds": next_event - event_complete,
        }
        calls.append(call)
        events.append(
            {
                "schema": "stage2_rank_worker_event_v1",
                "event": "call_complete",
                "event_index": ordinal + 1,
                "ordinal": ordinal,
                "invocation": invocation,
                "job_id": job["job_id"],
                "scorer": job["scorer"],
                "parent": parent,
                "call_started_monotonic": started,
                "call_completed_monotonic": completed,
                "completed_monotonic": next_event,
                "event_path": f"events/{ordinal + 1:04d}.json",
                "event_sha256": "b" * 64,
                "start_path": f"calls/{ordinal:04d}_start.json",
                "start_sha256": "c" * 64,
                "result_path": f"calls/{ordinal:04d}_result.json",
                "result_sha256": "d" * 64,
            }
        )
        event_complete = next_event

    measurements = {}
    if job["operation"] == "stage2":
        parents = stage1[:8]
        for width in (1, 8):
            queue = control.assemble_queue(
                parents,
                {i: calls[i]["ordered_words"] for i in range(width)},
                width,
            )
            measurements[str(width)] = {
                "seconds": 0.01,
                "queue_sha256": control.ordered_hash(queue),
            }
    receipt = {
        "schema": "simon_stage2_guardian_receipt_v1",
        "invocation": invocation,
        "job_id": job["job_id"],
        "accepted_events": events,
        "process_group_absent": True,
        "worker_pid": 1234,
        "process_group": 1234,
        "status": "worker_exit",
        "worker_returncode": 0,
        "cleanup_completed_monotonic": event_complete + 0.5,
    }
    return {
        **job,
        "outcome": "completed",
        "dispatched": True,
        "directory": f"/tmp/stage2/{job['job_id']}/{invocation}",
        "invocation": invocation,
        "calls": calls,
        "load_wall_seconds": 1.0,
        "worker_wall_seconds": (count + 1.5),
        "queue_measurements": measurements,
        "guardian_receipt": receipt,
        "accepted_events": events,
        "completed_utc": "2026-10-05T00:00:00Z",
    }


def _closed_grid(control: Any) -> list[dict[str, Any]]:
    return [
        _complete_row(control, job, i) for i, job in enumerate(control.fixed_jobs())
    ]


def test_fixed_job_grid_and_ordered_ciphertext_only_projection() -> None:
    control = _control()
    jobs = control.fixed_jobs()
    assert len(jobs) == 24
    assert jobs[:3] == [
        {
            "job_id": "simon_r10_k00_stage1_vv9",
            "key_index": 0,
            "operation": "stage1",
            "scorer": "vv9",
        },
        {
            "job_id": "simon_r10_k00_stage2_vv8",
            "key_index": 0,
            "operation": "stage2",
            "scorer": "vv8",
        },
        {
            "job_id": "simon_r10_k00_stage2_ddt7",
            "key_index": 0,
            "operation": "stage2",
            "scorer": "ddt7",
        },
    ]
    assert jobs[3:6] == [
        {
            "job_id": "simon_r10_k01_stage1_vv9",
            "key_index": 1,
            "operation": "stage1",
            "scorer": "vv9",
        },
        {
            "job_id": "simon_r10_k01_stage2_ddt7",
            "key_index": 1,
            "operation": "stage2",
            "scorer": "ddt7",
        },
        {
            "job_id": "simon_r10_k01_stage2_vv8",
            "key_index": 1,
            "operation": "stage2",
            "scorer": "vv8",
        },
    ]
    case = control.public_case(_dataset(control), jobs[0])
    assert set(case) == {
        "cipher",
        "rounds",
        "key_index",
        "instance_id",
        "ciphertext_pairs",
    }
    assert len(case["ciphertext_pairs"]) == 64
    assert "master_key" not in case and "round_keys" not in case
    assert "plaintext" not in repr(case) and "verification" not in repr(case)


def test_public_payload_and_words_reject_extra_truth_and_non_integer_values() -> None:
    control = _control()
    case = control.public_case(_dataset(control), control.fixed_jobs()[0])
    control.validate_public_case(case)
    with pytest.raises(ValueError):
        control.validate_public_case({**case, "master_key": [1, 2, 3, 4]})
    bad_case = {
        **case,
        "ciphertext_pairs": [[[[True, 1], [2, 3]], [[4, 5], [6, 7]]]] * 64,
    }
    with pytest.raises(ValueError):
        control.validate_public_case(bad_case)
    with pytest.raises(ValueError):
        control.validate_order([False, *range(1, 512)])
    with pytest.raises(ValueError):
        control.validate_order(list(range(511)) + [0])


def test_payload_contract_and_rank_sum_queue_order() -> None:
    control = _control()
    job = control.fixed_jobs()[1]
    payload = {
        "schema": "stage2_rank_payload_v1",
        "job": job,
        "public_case": control.public_case(_dataset(control), job),
        "parents": list(range(100, 108)),
        "invocation": "a" * 32,
        "directory": "/tmp/invocation",
        "source_root": str(ROOT),
        "source_commit": "b" * 40,
        "source_files": {
            "scripts/run_simon_matched_stage2_worker.py": {
                "path": str(ROOT / "scripts/run_simon_matched_stage2_worker.py"),
                "sha256": "c" * 64,
            },
            "src/neural_sat_speck/followups/stage2_control.py": {
                "path": str(CONTROL_PATH),
                "sha256": "d" * 64,
            },
            "src/neural_sat_speck/protocol/worker_telemetry.py": {
                "path": str(ROOT / "src/neural_sat_speck/protocol/worker_telemetry.py"),
                "sha256": "e" * 64,
            },
            "src/neural_sat_speck/simon/cipher.py": {
                "path": str(ROOT / "src/neural_sat_speck/simon/cipher.py"),
                "sha256": "f" * 64,
            },
            "src/neural_sat_speck/simon/primitives.py": {
                "path": str(ROOT / "src/neural_sat_speck/simon/primitives.py"),
                "sha256": "0" * 64,
            },
            "src/neural_sat_speck/simon/ranking_features.py": {
                "path": str(ROOT / "src/neural_sat_speck/simon/ranking_features.py"),
                "sha256": "1" * 64,
            },
            "src/neural_sat_speck/simon/published_vv.py": {
                "path": str(ROOT / "src/neural_sat_speck/simon/published_vv.py"),
                "sha256": "2" * 64,
            },
            "src/neural_sat_speck/simon/published_vv_ranking.py": {
                "path": str(
                    ROOT / "src/neural_sat_speck/simon/published_vv_ranking.py"
                ),
                "sha256": "3" * 64,
            },
            "src/neural_sat_speck/simon/classical_ranking.py": {
                "path": str(ROOT / "src/neural_sat_speck/simon/classical_ranking.py"),
                "sha256": "4" * 64,
            },
        },
        "assets": {
            name: {
                "path": f"/tmp/{name}",
                "sha256": digest,
                "identity": {"device": 1, "inode": i + 1, "size": 1, "mtime_ns": 1},
            }
            for i, (name, digest) in enumerate(control._EXPECTED_ASSET_HASHES.items())
        },
        "runtime": {},
        "profile": control.PROFILE,
    }
    # Payload contract includes complete runtime telemetry. Reuse synthetic
    # smoke pins and preserve genuine dynamic Python version strings.
    package_names = ["numpy", "tensorflow", "keras", "h5py"]
    modules = sorted(
        [
            "neural_sat_speck.simon.cipher",
            "neural_sat_speck.simon.ranking_features",
            "neural_sat_speck.simon.published_vv",
            "neural_sat_speck.simon.published_vv_ranking",
            "neural_sat_speck.simon.classical_ranking",
            "neural_sat_speck.simon.key_schedule",
            "neural_sat_speck.simon.primitives",
        ]
    )
    boundary = {
        "minimal_namespace_bootstrap": True,
        "simon_initializer_loaded": False,
        "protocol_package_loaded": False,
        "forbidden_modules_loaded": [],
        "forbidden_import_attempts": [],
        "ranking_modules_loaded": modules,
    }
    base_path = str(ROOT / ".venv/bin/python")
    tf_path = str(ROOT / "runtime/simon_tf_venv/bin/python3.12")
    common = {
        "executable": tf_path,
        "python_version": "3.12.13",
        "packages": dict(zip(package_names, ["1.2.3", "2.16.2", "3.15.1", "3.16.0"])),
        "origins": dict(
            zip(package_names, ["/pkg/numpy", "/pkg/tf", "/pkg/keras", "/pkg/h5py"])
        ),
        "devices": {
            "machine": "arm64",
            "processor": "test",
            "tensorflow_physical": [{"name": "CPU:0", "type": "CPU"}],
        },
        "threads": {
            "numeric_environment": control.PROFILE["numeric_thread_env"],
            "tensorflow": {"intra_op": 8, "inter_op": 1},
            "feature_batch_size": 1024,
            "inference_batch_size": 16384,
            "candidate_batch_size": 1024,
        },
        "import_boundary": boundary,
    }
    base = {
        **common,
        "executable": base_path,
        "python_version": "3.14.7",
        "packages": {"numpy": "2.5.2", "tensorflow": None, "keras": None, "h5py": None},
        "origins": {
            "numpy": "/pkg/base-numpy",
            "tensorflow": None,
            "keras": None,
            "h5py": None,
        },
        "devices": {"machine": "arm64", "processor": "test", "tensorflow_physical": []},
        "threads": {**common["threads"], "tensorflow": None},
    }
    payload["runtime"] = {"python": base, "tensorflow": common}
    control.validate_payload(payload)
    environment_drift = deepcopy(payload)
    environment_drift["runtime"]["tensorflow"]["threads"]["numeric_environment"][
        "TF_NUM_INTRAOP_THREADS"
    ] = "1"
    with pytest.raises(ValueError, match="thread environment"):
        control.validate_payload(environment_drift)
    api_drift = deepcopy(payload)
    api_drift["runtime"]["tensorflow"]["threads"]["tensorflow"]["intra_op"] = 1
    with pytest.raises(ValueError, match="TensorFlow"):
        control.validate_payload(api_drift)
    with pytest.raises(ValueError):
        control.validate_payload(
            {
                **payload,
                "public_case": {
                    **payload["public_case"],
                    "verification_plaintexts": [],
                },
            }
        )

    parents = list(range(100, 108))
    orders = {
        index: list(range(1000 + index * 512, 1000 + index * 512 + 512))
        for index in range(8)
    }
    width1 = control.assemble_queue(parents, {0: orders[0]}, 1)
    width8 = control.assemble_queue(parents, orders, 8)
    assert width1 == tuple((child, parents[0]) for child in orders[0][:128])
    assert len(width8) == 1024
    assert width8 == tuple(
        sorted(
            width8,
            key=lambda pair: (
                parents.index(pair[1]) + orders[parents.index(pair[1])].index(pair[0]),
                parents.index(pair[1]),
                orders[parents.index(pair[1])].index(pair[0]),
                pair[1],
                pair[0],
            ),
        )
    )
    with pytest.raises(ValueError):
        control.assemble_queue(parents, {0: orders[0]}, 8)
    with pytest.raises(ValueError):
        control.assemble_queue(parents, {False: orders[0]}, 1)


def test_full_binding_accepts_d052_and_rejects_old_d051_pilot(tmp_path: Path) -> None:
    control = _control()
    commit = "a" * 40
    common = {
        "source_commit": commit,
        "source_root": str(ROOT),
        "profile": control.PROFILE,
        "files": {},
        "limits": [],
        "assets": {},
        "runtime": {},
    }
    pilot_manifest = {
        **common,
        "mode": "pilot",
        "status": "APPROVED",
        "execution_authorized": True,
        "authorization_decision": "D-052",
        "jobs": control.fixed_jobs()[:3],
    }
    manifest_path = tmp_path / "manifest.json"
    report_path = tmp_path / "report.json"

    def checked_write(path: Path, value: Any) -> dict[str, str]:
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        digest = control.streaming_sha256(path)
        path.with_name(path.name + ".sha256").write_text(
            f"{digest}  {path.name}\n", encoding="ascii"
        )
        return {"path": str(path), "sha256": digest}

    manifest_pin = checked_write(manifest_path, pilot_manifest)
    report = {
        "schema": "stage2_control_pilot_report_v1",
        "status": "PASS",
        "mechanical_pass": True,
        "source_commit": commit,
        "profile": control.PROFILE,
        "profile_sha256": control.ordered_hash(control.PROFILE),
        "manifest_sha256": manifest_pin["sha256"],
        "ledger_sha256": "b" * 64,
        "jobs": [job["job_id"] for job in control.fixed_jobs()[:3]],
        "completed_calls": 17,
        "completed_workers": 3,
        "all_groups_absent": True,
        "truth_join_performed": False,
        "raw_sidecars_checked": 1,
        "physical_worker_seconds": 42.0,
    }
    pilot_pin = {
        "manifest": manifest_pin,
        "report": checked_write(report_path, report),
    }
    control._validate_pilot_binding(pilot_pin, ROOT, common, commit)

    # Consistent hashes cannot make a superseded authorization eligible.
    pilot_manifest["authorization_decision"] = "D-051"
    pilot_pin["manifest"] = checked_write(manifest_path, pilot_manifest)
    report["manifest_sha256"] = pilot_pin["manifest"]["sha256"]
    pilot_pin["report"] = checked_write(report_path, report)
    with pytest.raises(ValueError, match="Pilot manifest does not match"):
        control._validate_pilot_binding(pilot_pin, ROOT, common, commit)


def test_full_analyzer_keeps_8_key_denominators_and_uses_phase_costs() -> None:
    control = _control()
    report = control.analyze_cells(_dataset(control), _closed_grid(control))
    assert report["status"] == "descriptive_opened_demo"
    assert report["cell_count"] == 32
    assert report["assigned_keys"] == 8
    assert report["significance_tests"] == "none"
    cells = report["cells"]
    ddt_key0_width1 = next(
        cell
        for cell in cells
        if cell["key_index"] == 0
        and cell["scorer"] == "ddt7"
        and cell["parent_width"] == 1
    )
    assert ddt_key0_width1["true_parent_in_width"] is True
    assert ddt_key0_width1["true_child_rank_within_512"] is None
    assert ddt_key0_width1["true_child_in_top128_given_parent"] is False
    assert ddt_key0_width1["true_tuple_contained"] is False
    vv8_key0 = next(
        cell
        for cell in cells
        if cell["key_index"] == 0
        and cell["scorer"] == "vv8"
        and cell["parent_width"] == 1
    )
    assert vv8_key0["true_child_rank_within_512"] == 1
    assert vv8_key0["true_tuple_contained"] is True
    assert vv8_key0["projected_ranking_cost_seconds"] == pytest.approx(4.51)
    assert vv8_key0["selected_stage2_call_phase_seconds"] == 1.0
    for aggregate in report["aggregates"]:
        assert aggregate["assigned_key_denominator"] == 8
        assert aggregate["usable_true_tuple_queue_yield_denominator"] == 8


def test_analyzer_separates_structural_parent_miss_from_partial_stage2_unavailability() -> (
    None
):
    control = _control()
    dataset = _dataset(control)
    # This synthetic truth lies outside the shared top-eight parent list.
    dataset["ciphers"]["simon32_64"]["instances"][1]["rounds_data"]["10"]["round_keys"][
        -1
    ] = 0xFFFF
    grid = _closed_grid(control)
    partial = next(row for row in grid if row["job_id"] == "simon_r10_k00_stage2_ddt7")
    partial["outcome"] = "rank_timeout"
    partial["calls"] = partial["calls"][:2]
    partial["accepted_events"] = partial["accepted_events"][:3]
    partial["guardian_receipt"]["accepted_events"] = partial["accepted_events"]
    partial["guardian_receipt"]["status"] = "rank_timeout"
    partial["guardian_receipt"]["worker_returncode"] = -9
    partial["guardian_receipt"]["cleanup_completed_monotonic"] = (
        partial["accepted_events"][-1]["completed_monotonic"] + 0.5
    )
    partial["worker_wall_seconds"] = 3.5
    partial["queue_measurements"].pop("8")

    report = control.analyze_cells(dataset, grid)
    structural = next(
        cell
        for cell in report["cells"]
        if cell["key_index"] == 1
        and cell["scorer"] == "vv8"
        and cell["parent_width"] == 8
    )
    assert structural["stage1_true_parent_status"] == "not_in_fixed_shortlist"
    assert structural["true_parent_in_width"] is False
    assert structural["true_child_status"] == "parent_not_selected"
    partial_width1 = next(
        cell
        for cell in report["cells"]
        if cell["key_index"] == 0
        and cell["scorer"] == "ddt7"
        and cell["parent_width"] == 1
    )
    partial_width8 = next(
        cell
        for cell in report["cells"]
        if cell["key_index"] == 0
        and cell["scorer"] == "ddt7"
        and cell["parent_width"] == 8
    )
    assert partial_width1["queue_complete"] is True
    assert partial_width1["true_child_status"] == "not_in_fixed_shortlist"
    assert partial_width8["queue_complete"] is False
    assert partial_width8["true_child_status"] == "not_in_fixed_shortlist"
    ddt_width8 = next(
        group
        for group in report["aggregates"]
        if group["scorer"] == "ddt7" and group["parent_width"] == 8
    )
    assert ddt_width8["assigned_key_denominator"] == 8
    assert ddt_width8["complete_queue_count"] == 7
    assert ddt_width8["unavailable_or_incomplete_queue_count"] == 1


def test_tombstones_keep_unknown_dispatch_and_proven_no_worker_distinct() -> None:
    control = _control()
    grid = _closed_grid(control)
    stage1 = next(row for row in grid if row["job_id"] == "simon_r10_k07_stage1_vv9")
    stage1.update(
        outcome="interrupted",
        dispatched=True,
        directory="/tmp/stage2/interrupted/" + "7" * 32,
        invocation="7" * 32,
        calls=[],
        load_wall_seconds=None,
        worker_wall_seconds=None,
        queue_measurements={},
    )
    stage1.pop("accepted_events")
    stage1.pop("guardian_receipt")
    grid = [
        row
        for row in grid
        if not (row["key_index"] == 7 and row["operation"] == "stage2")
    ]
    for job in control.fixed_jobs():
        if job["key_index"] == 7 and job["operation"] == "stage2":
            grid.append(
                {
                    **job,
                    "outcome": "unavailable_after_interrupted",
                    "dispatched": False,
                    "calls": [],
                    "load_wall_seconds": None,
                    "worker_wall_seconds": None,
                    "queue_measurements": {},
                    "completed_utc": "2026-10-05T00:00:00Z",
                }
            )

    # The guardian started but proves it did not launch a worker process.
    no_worker = next(row for row in grid if row["job_id"] == "simon_r10_k00_stage2_vv8")
    invocation = "8" * 32
    no_worker.update(
        outcome="load_timeout",
        dispatched=False,
        directory=f"/tmp/stage2/no-worker/{invocation}",
        invocation=invocation,
        calls=[],
        load_wall_seconds=None,
        worker_wall_seconds=None,
        queue_measurements={},
        accepted_events=[],
        guardian_receipt={
            "schema": "simon_stage2_guardian_receipt_v1",
            "invocation": invocation,
            "job_id": no_worker["job_id"],
            "worker_not_started": True,
            "worker_pid": None,
            "process_group": None,
            "process_group_absent": True,
            "accepted_events": [],
            "cleanup_completed_monotonic": 99999.0,
        },
    )

    report = control.analyze_cells(_dataset(control), grid)
    interrupted = next(
        cell
        for cell in report["cells"]
        if cell["key_index"] == 7
        and cell["scorer"] == "vv8"
        and cell["parent_width"] == 8
    )
    assert interrupted["stage1_dispatch_state"] == "dispatch_unknown"
    assert interrupted["stage2_dispatch_state"] == "not_dispatched"
    assert interrupted["projected_ranking_cost_seconds"] is None
    proven_absent = next(
        cell
        for cell in report["cells"]
        if cell["key_index"] == 0
        and cell["scorer"] == "vv8"
        and cell["parent_width"] == 1
    )
    assert proven_absent["stage2_dispatch_state"] == "not_started_proven"
    assert proven_absent["projected_ranking_cost_seconds"] is None
    assert all(item["assigned_key_denominator"] == 8 for item in report["aggregates"])


def test_worker_cleanup_overshoot_is_preserved_without_violating_phase_caps() -> None:
    control = _control()
    grid = _closed_grid(control)
    row = next(
        item
        for item in grid
        if item["job_id"] == "simon_r10_k00_stage2_vv8"
    )
    phase_budget = (
        control.PROFILE["startup_limit_seconds"]
        + 8 * control.PROFILE["call_limit_seconds"]
    )
    row["worker_wall_seconds"] = phase_budget + 0.25

    report = control.analyze_cells(_dataset(control), grid)
    cell = next(
        item
        for item in report["cells"]
        if item["key_index"] == 0
        and item["scorer"] == "vv8"
        and item["parent_width"] == 8
    )
    assert cell["stage2_worker_physical_wall_seconds"] == phase_budget + 0.25
    assert cell["stage2_physical_worker_seconds_over_phase_budget"] == pytest.approx(
        0.25
    )
    assert cell["stage2_worker_tail_seconds_after_accepted_phases"] == pytest.approx(
        phase_budget + 0.25 - 9.0
    )


def test_interrupted_success_receipt_does_not_invent_queue_persistence() -> None:
    control = _control()
    grid = _closed_grid(control)
    row = next(
        item
        for item in grid
        if item["job_id"] == "simon_r10_k00_stage2_vv8"
    )
    assert row["guardian_receipt"]["status"] == "worker_exit"
    assert row["guardian_receipt"]["worker_returncode"] == 0
    row["outcome"] = "interrupted"
    # The coordinator stopped after the worker exited and before queue records
    # were committed. Accepted rank calls remain usable; no queue is fabricated.
    row["queue_measurements"] = {}

    report = control.analyze_cells(_dataset(control), grid)
    cells = [
        item
        for item in report["cells"]
        if item["key_index"] == 0 and item["scorer"] == "vv8"
    ]
    assert len(cells) == 2
    assert all(item["stage2_dispatch_state"] == "worker_started" for item in cells)
    assert all(item["queue_complete"] is False for item in cells)
    assert all(item["queue_sha256"] is None for item in cells)
    assert all(item["projected_ranking_cost_seconds"] is None for item in cells)


def test_only_exact_sealed_orphan_tombstone_is_analyzable() -> None:
    control = _control()
    grid = _closed_grid(control)
    job = control.fixed_jobs()[-1]
    assert job["job_id"] == "simon_r10_k07_stage2_vv8"
    invocation = "f" * 32
    orphan = {
        **job,
        "outcome": "unavailable_after_coordinator_interrupted",
        "dispatched": False,
        "directory": f"/tmp/stage2/orphan/{invocation}",
        "invocation": invocation,
        "calls": [],
        "load_wall_seconds": None,
        "worker_wall_seconds": None,
        "queue_measurements": {},
        "completed_utc": "2026-10-05T00:00:00Z",
        "error": "Checked start was written before its ledger snapshot; dispatch was not reached",
    }
    grid[-1] = orphan

    report = control.analyze_cells(_dataset(control), grid)
    orphan_cells = [
        item
        for item in report["cells"]
        if item["key_index"] == 7 and item["scorer"] == "vv8"
    ]
    assert len(orphan_cells) == 2
    assert all(
        item["stage2_dispatch_state"] == "not_started_orphan"
        for item in orphan_cells
    )
    assert all(item["queue_complete"] is False for item in orphan_cells)

    unbound = dict(orphan)
    unbound.pop("directory")
    grid[-1] = unbound
    with pytest.raises(ValueError, match="exact checked-start tombstone"):
        control.analyze_cells(_dataset(control), grid)


def test_pilot_returns_non_scientific_without_touching_truth_and_closure_is_strict() -> (
    None
):
    control = _control()
    pilot = [
        {**job, "outcome": "completed", "calls": []} for job in control.fixed_jobs()[:3]
    ]
    result = control.analyze_cells(object(), {"mode": "pilot", "jobs": pilot})
    assert result["status"] == "pilot_not_scientific"
    assert result["cells"] == []

    grid = _closed_grid(control)
    # An altered Stage 2 parent can otherwise look like an ordinary rank row;
    # even with internally consistent events it must fail before the truth join.
    row = next(item for item in grid if item["job_id"] == "simon_r10_k00_stage2_vv8")
    call = row["calls"][1]
    call["parent"] += 1
    event = row["accepted_events"][2]
    event["parent"] += 1
    with pytest.raises(ValueError, match="common Stage 1 order"):
        control.analyze_cells(object(), grid)

    with pytest.raises(ValueError, match="all 24 worker jobs"):
        control.analyze_cells(_dataset(control), {"mode": "full", "jobs": grid[:-1]})


def test_interpreter_file_pin_preserves_lexical_symlink(tmp_path: Path) -> None:
    control = _control()
    target = tmp_path / "python-real"
    target.write_bytes(b"interpreter")
    link = tmp_path / "python"
    link.symlink_to(target)
    pinned = control._file_path(
        {"path": str(link), "sha256": control.streaming_sha256(link)},
        tmp_path,
        "python",
    )
    assert pinned == link
    assert pinned.is_symlink()
    assert pinned.samefile(target)
