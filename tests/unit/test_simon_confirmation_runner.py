"""Integration and entry-point tests for SIMON confirmation runner with worker spies.

Enforces:
- Zero worker calls on preflight, provenance, dataset, certificate, or resume rejections.
- Complete 9-arm counterbalanced schedule and resource path propagation.
- Partial resume skipping completed cells and validating sidecar.
- Checksum drift detection and atomic ledger persistence.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from neural_sat_speck.simon.attack_instance import (
    SimonAttackInstanceConfig,
    generate_simon_attack_instances,
)
from neural_sat_speck.simon.confirmation_guard import (
    ALL_CONFIRMATION_METHODS,
    FROZEN_QUEUE_CONTROL_SEED,
    REQUIRED_COMPARATOR_METHODS,
    DisjointFamilyCertificateError,
    _training_key_overlaps,
    build_disjoint_family_certificate,
    compute_execution_identity,
    file_sha256,
)
from neural_sat_speck.simon.method_worker import MethodWorkerOutcome
from scripts import evaluate_simon_confirmation as runner
from tests.fixtures.simon_reference_oracle import oracle_simon32_64_key_schedule


@pytest.fixture
def synthetic_runner_environment(tmp_path: Path):
    """Create a fully valid synthetic fixture environment for runner integration tests."""
    # 1. Create dummy resource files
    ddt_path = tmp_path / "simon_predicted_ddt_7r.bin"
    ddt_path.write_bytes(b"DDT_7R_DUMMY_DATA_BYTES")
    ddt_sha = hashlib.sha256(ddt_path.read_bytes()).hexdigest()

    cpp_path = tmp_path / "simon_queried_row_8r.cpp"
    cpp_path.write_bytes(b"// C++ source fixture")
    cpp_sha = hashlib.sha256(cpp_path.read_bytes()).hexdigest()

    bin_path = tmp_path / "simon_queried_row_8r"
    bin_path.write_bytes(b"\x7fELF_DUMMY_BINARY")
    bin_sha = hashlib.sha256(bin_path.read_bytes()).hexdigest()

    b1_path = tmp_path / "simon_b1_mean_speckhead.pt"
    b1_path.write_bytes(b"PYTORCH_CHECKPOINT_DUMMY")
    b1_sha = hashlib.sha256(b1_path.read_bytes()).hexdigest()

    bao_dir = tmp_path / "bao_vv"
    bao_dir.mkdir()
    vv9_path = bao_dir / "ND_VV_Simon32_9R.h5"
    vv9_path.write_bytes(b"HDF5_VV9_DUMMY")
    vv9_sha = hashlib.sha256(vv9_path.read_bytes()).hexdigest()

    vv8_path = bao_dir / "ND_VV_Simon32_8R.h5"
    vv8_path.write_bytes(b"HDF5_VV8_DUMMY")
    vv8_sha = hashlib.sha256(vv8_path.read_bytes()).hexdigest()

    tf_py = tmp_path / "python3.12"
    tf_py.write_bytes(b"#!/bin/sh\n")

    # 2. Manifest
    manifest = {
        "manifest_version": "1.1.0",
        "protocol_id": "simon32_64_r10_unseen_confirmation_v1",
        "status": "APPROVED",
        "evidence_class": "CONFIRMATION",
        "confirmation_execution_authorized": True,
        "seed_governance": {
            "seed": 777001,
            "seed_status": "APPROVED_FRESH_CONFIRMATION",
        },
        "target_cipher": {
            "cipher_name": "SIMON32/64",
            "rounds": 10,
            "input_difference": [0, 64],
        },
        "sample_size_specification": {
            "selected_instances": 32,
            "pair_count": 64,
            "known_observations": 128,
            "verification_observations": 2,
        },
        "candidate_sat_specification": {
            "observations_used": 4,
            "formula_variables": 1856,
            "cnf_clauses": 2176,
            "native_xor_clauses": 1024,
        },
        "sat_only_controls_specification": {
            "sat_only_n4": {
                "observations_used": 4,
                "formula_variables": 1856,
                "cnf_clauses": 2176,
                "native_xor_clauses": 1024,
            },
            "sat_only_n128": {
                "observations_used": 128,
                "formula_variables": 45504,
                "cnf_clauses": 69632,
                "native_xor_clauses": 20864,
            },
        },
        "resource_limits": {
            "global_deadline_seconds": 300.0,
            "candidate_cutoff_soft_seconds": 45.0,
            "ranking_time_limit_seconds": 120.0,
        },
        "comparator_arms": [{"method_id": m} for m in REQUIRED_COMPARATOR_METHODS],
        "diagnostic_control": {
            "method_id": "oracle_2key_n4",
        },
        "random_control_stream": {
            "control_seed": FROZEN_QUEUE_CONTROL_SEED,
        },
        "provenance_hashes": {
            "simon_predicted_ddt_7r_bin_sha256": ddt_sha,
            "simon_queried_row_8r_cpp_sha256": cpp_sha,
            "simon_queried_row_8r_bin_sha256": bin_sha,
            "b1_checkpoint_sha256": b1_sha,
            "vv9_checkpoint_sha256": vv9_sha,
            "vv8_checkpoint_sha256": vv8_sha,
        },
        "development_resource_locations": {
            "ddt_7r": str(ddt_path),
            "queried_row_8r_binary": str(bin_path),
            "bao_vv_directory": str(bao_dir),
            "tensorflow_python": str(tf_py),
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 3. Confirmation Dataset (32 instances with fresh seed 777001)
    cfg = SimonAttackInstanceConfig(
        instance_count=32,
        pair_count=64,
        rounds=10,
        seed=777001,
        split_role="confirmation",
    )
    instances = generate_simon_attack_instances(cfg)
    dataset_path = tmp_path / "dataset.json"
    ds_raw = json.dumps([inst.to_dict() for inst in instances], indent=2)
    dataset_path.write_text(ds_raw, encoding="utf-8")
    (tmp_path / "dataset.json.sha256").write_text(
        f"{hashlib.sha256(ds_raw.encode()).hexdigest()}  dataset.json\n", encoding="utf-8"
    )

    # 4. Disjoint-Family Certificate
    cert_path = tmp_path / "certificate.json"
    build_disjoint_family_certificate(manifest, instances, output_path=cert_path)

    # Patch repo tools path for tools/simon_queried_row_8r.cpp and models/
    resource_paths = {
        "ddt_7r": ddt_path,
        "queried_row_8r_binary": bin_path,
        "queried_row_8r_cpp": cpp_path,
        "b1_checkpoint": b1_path,
        "vv9_checkpoint": vv9_path,
        "vv8_checkpoint": vv8_path,
        "bao_vv_directory": bao_dir,
        "tensorflow_python": tf_py,
    }

    return {
        "manifest_path": manifest_path,
        "manifest": manifest,
        "dataset_path": dataset_path,
        "instances": instances,
        "cert_path": cert_path,
        "resource_paths": resource_paths,
        "tmp_path": tmp_path,
    }


def test_runner_rejection_manifest_unapproved_zero_worker_calls(synthetic_runner_environment, monkeypatch):
    env = synthetic_runner_environment
    # Tamper manifest status
    m_path = env["tmp_path"] / "unapproved_manifest.json"
    m_data = copy.deepcopy(env["manifest"])
    m_data["status"] = "PROPOSED_SEEDLESS_DRAFT"
    m_path.write_text(json.dumps(m_data), encoding="utf-8")

    worker_spy = MagicMock()
    monkeypatch.setattr(runner, "run_method_worker", worker_spy)
    monkeypatch.setattr(runner.sys, "argv", [
        "evaluate_simon_confirmation.py",
        "--manifest", str(m_path),
        "--dataset", str(env["dataset_path"]),
        "--certificate", str(env["cert_path"]),
        "--output", str(env["tmp_path"] / "ledger.json"),
    ])

    ret = runner.main()
    assert ret == 1
    assert worker_spy.call_count == 0


def test_runner_rejection_unauthorized_execution_zero_worker_calls(synthetic_runner_environment, monkeypatch):
    env = synthetic_runner_environment
    m_path = env["tmp_path"] / "unauthorized_manifest.json"
    m_data = copy.deepcopy(env["manifest"])
    m_data["confirmation_execution_authorized"] = False
    m_path.write_text(json.dumps(m_data), encoding="utf-8")

    worker_spy = MagicMock()
    monkeypatch.setattr(runner, "run_method_worker", worker_spy)
    monkeypatch.setattr(runner.sys, "argv", [
        "evaluate_simon_confirmation.py",
        "--manifest", str(m_path),
        "--dataset", str(env["dataset_path"]),
        "--certificate", str(env["cert_path"]),
        "--output", str(env["tmp_path"] / "ledger.json"),
    ])

    ret = runner.main()
    assert ret == 1
    assert worker_spy.call_count == 0


def test_runner_rejection_missing_dataset_zero_worker_calls(synthetic_runner_environment, monkeypatch):
    env = synthetic_runner_environment
    worker_spy = MagicMock()
    monkeypatch.setattr(runner, "run_method_worker", worker_spy)
    monkeypatch.setattr(runner, "verify_runtime_provenance", lambda *_: env["resource_paths"])
    monkeypatch.setattr(runner.sys, "argv", [
        "evaluate_simon_confirmation.py",
        "--manifest", str(env["manifest_path"]),
        "--dataset", str(env["tmp_path"] / "nonexistent_dataset.json"),
        "--certificate", str(env["cert_path"]),
        "--output", str(env["tmp_path"] / "ledger.json"),
    ])

    ret = runner.main()
    assert ret == 3
    assert worker_spy.call_count == 0


def test_runner_rejection_non_confirmation_dataset_zero_worker_calls(synthetic_runner_environment, monkeypatch):
    env = synthetic_runner_environment
    # Generate instances with split_role="development"
    dev_cfg = SimonAttackInstanceConfig(
        instance_count=32, pair_count=64, rounds=10, seed=777001, split_role="development",
    )
    dev_instances = generate_simon_attack_instances(dev_cfg)
    dev_ds_path = env["tmp_path"] / "dev_dataset.json"
    dev_ds_path.write_text(json.dumps([inst.to_dict() for inst in dev_instances]), encoding="utf-8")

    worker_spy = MagicMock()
    monkeypatch.setattr(runner, "run_method_worker", worker_spy)
    monkeypatch.setattr(runner, "verify_runtime_provenance", lambda *_: env["resource_paths"])
    monkeypatch.setattr(runner.sys, "argv", [
        "evaluate_simon_confirmation.py",
        "--manifest", str(env["manifest_path"]),
        "--dataset", str(dev_ds_path),
        "--certificate", str(env["cert_path"]),
        "--output", str(env["tmp_path"] / "ledger.json"),
    ])

    ret = runner.main()
    assert ret == 4
    assert worker_spy.call_count == 0


def test_runner_rejection_disjoint_family_overlap_zero_worker_calls(synthetic_runner_environment, monkeypatch):
    env = synthetic_runner_environment
    # Create dataset of 32 instances where first instance reuses key from demo family (seed 999999)
    demo_inst = generate_simon_attack_instances(SimonAttackInstanceConfig(
        instance_count=1, pair_count=4, rounds=10, seed=999999, split_role="demo",
    ))[0]
    clash_instances = [replace(demo_inst, split_role="confirmation")] + list(env["instances"][1:])
    clash_path = env["tmp_path"] / "clash_dataset.json"
    clash_path.write_text(json.dumps([inst.to_dict() for inst in clash_instances]), encoding="utf-8")

    worker_spy = MagicMock()
    monkeypatch.setattr(runner, "run_method_worker", worker_spy)
    monkeypatch.setattr(runner, "verify_runtime_provenance", lambda *_: env["resource_paths"])
    monkeypatch.setattr(runner.sys, "argv", [
        "evaluate_simon_confirmation.py",
        "--manifest", str(env["manifest_path"]),
        "--dataset", str(clash_path),
        "--certificate", str(env["cert_path"]),
        "--output", str(env["tmp_path"] / "ledger.json"),
    ])

    ret = runner.main()
    assert ret == 5
    assert worker_spy.call_count == 0


def test_runner_rejection_unflagged_resume_zero_worker_calls(synthetic_runner_environment, monkeypatch):
    env = synthetic_runner_environment
    ledger_path = env["tmp_path"] / "existing_ledger.json"
    ledger_path.write_text(json.dumps({"execution_identity": {}, "records": []}), encoding="utf-8")
    sha = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    (env["tmp_path"] / "existing_ledger.json.sha256").write_text(f"{sha}  existing_ledger.json\n", encoding="utf-8")

    worker_spy = MagicMock()
    monkeypatch.setattr(runner, "run_method_worker", worker_spy)
    monkeypatch.setattr(runner, "verify_runtime_provenance", lambda *_: env["resource_paths"])
    monkeypatch.setattr(runner.sys, "argv", [
        "evaluate_simon_confirmation.py",
        "--manifest", str(env["manifest_path"]),
        "--dataset", str(env["dataset_path"]),
        "--certificate", str(env["cert_path"]),
        "--output", str(ledger_path),
        # Notice: --resume is absent!
    ])

    ret = runner.main()
    assert ret == 6
    assert worker_spy.call_count == 0


def test_runner_rejection_corrupted_sidecar_on_resume_zero_worker_calls(synthetic_runner_environment, monkeypatch):
    env = synthetic_runner_environment
    ledger_path = env["tmp_path"] / "corrupt_ledger.json"
    ledger_path.write_text(json.dumps({"execution_identity": {}, "records": []}), encoding="utf-8")
    # Wrong sha in sidecar
    (env["tmp_path"] / "corrupt_ledger.json.sha256").write_text("0" * 64 + "  corrupt_ledger.json\n", encoding="utf-8")

    worker_spy = MagicMock()
    monkeypatch.setattr(runner, "run_method_worker", worker_spy)
    monkeypatch.setattr(runner, "verify_runtime_provenance", lambda *_: env["resource_paths"])
    monkeypatch.setattr(runner.sys, "argv", [
        "evaluate_simon_confirmation.py",
        "--manifest", str(env["manifest_path"]),
        "--dataset", str(env["dataset_path"]),
        "--certificate", str(env["cert_path"]),
        "--output", str(ledger_path),
        "--resume",
    ])

    ret = runner.main()
    assert ret == 6
    assert worker_spy.call_count == 0


def test_runner_full_nine_arm_schedule_and_completion(synthetic_runner_environment, monkeypatch):
    env = synthetic_runner_environment
    ledger_path = env["tmp_path"] / "out_ledger.json"
    worker_calls = []
    inst_by_id = {inst.instance_id: inst for inst in env["instances"]}

    def mock_worker(**kwargs):
        worker_calls.append(kwargs)
        method = kwargs["method"]
        inst_id = kwargs["comparison_input"].instance_id
        inst = inst_by_id[inst_id]
        return MethodWorkerOutcome(
            method=method,
            outcome="queue_exhausted" if not method.startswith("oracle_") else "verified_recovery",
            complete_online_wall_seconds=0.01,
            ranking_wall_seconds=0.005,
            solving_wall_seconds=0.005,
            verification_wall_seconds=0.0,
            candidates_tested_count=0,
            candidates_remaining_count=0,
            recovered_master_key=inst.master_key if method.startswith("oracle_") else None,
            verified_on_fresh_pairs=bool(method.startswith("oracle_")),
            candidate_attempts=(),
            peak_rss_gib=0.001,
        )

    monkeypatch.setattr(runner, "run_method_worker", mock_worker)
    monkeypatch.setattr(runner, "verify_runtime_provenance", lambda *_: env["resource_paths"])
    monkeypatch.setattr(runner.sys, "argv", [
        "evaluate_simon_confirmation.py",
        "--manifest", str(env["manifest_path"]),
        "--dataset", str(env["dataset_path"]),
        "--certificate", str(env["cert_path"]),
        "--output", str(ledger_path),
    ])

    ret = runner.main()
    assert ret == 0

    # 1. Exactly 32 * 9 = 288 calls (32 instances x 9 methods)
    assert len(worker_calls) == 32 * 9
    methods_called = {c["method"] for c in worker_calls}
    assert methods_called == set(ALL_CONFIRMATION_METHODS)

    # 2. Path propagation to worker
    for c in worker_calls:
        assert c["ddt_7r_path"] == env["resource_paths"]["ddt_7r"]
        assert c["tool_8r_binary_path"] == env["resource_paths"]["queried_row_8r_binary"]
        assert c["bao_checkpoint_dir"] == env["resource_paths"]["bao_vv_directory"]
        assert c["tf_python_path"] == env["resource_paths"]["tensorflow_python"]
        assert c["queue_control_seed"] == FROZEN_QUEUE_CONTROL_SEED
        inst_id = c["comparison_input"].instance_id
        inst = inst_by_id[inst_id]
        if c["method"] == "oracle_2key_n4":
            exp_trailing = oracle_simon32_64_key_schedule(inst.master_key, rounds=10)[8:10]
            assert c["oracle_trailing_keys"] == exp_trailing
        else:
            assert c["oracle_trailing_keys"] is None

    # 3. Output ledger and sidecar existence and contents
    assert ledger_path.is_file()
    sidecar_path = env["tmp_path"] / "out_ledger.json.sha256"
    assert sidecar_path.is_file()
    actual_sha = file_sha256(ledger_path)
    assert sidecar_path.read_text(encoding="utf-8").startswith(actual_sha)

    with open(ledger_path, encoding="utf-8") as f:
        ledger = json.load(f)
    assert len(ledger["records"]) == 288
    assert ledger["completed_count"] == 288


def test_runner_partial_resume_skips_completed_cells(synthetic_runner_environment, monkeypatch):
    env = synthetic_runner_environment
    ledger_path = env["tmp_path"] / "partial_ledger.json"

    # Pre-create ledger with all completed records except 4 pending on the last instance
    ds_sha = file_sha256(env["dataset_path"])
    exec_id = compute_execution_identity(env["manifest"], ds_sha)
    records = []
    # Complete all cells for instances 0..30
    for idx in range(31):
        inst = env["instances"][idx]
        for m in ALL_CONFIRMATION_METHODS:
            records.append({
                "instance_id": inst.instance_id,
                "instance_index": idx,
                "method": m,
                "outcome": "solver_timeout",
                "complete_online_wall_seconds": 0.01,
                "ranking_wall_seconds": 0.005,
                "solving_wall_seconds": 0.005,
                "verification_wall_seconds": 0.0,
                "candidates_tested_count": 0,
                "candidates_remaining_count": 0,
                "recovered_master_key": None,
                "verified_on_fresh_pairs": False,
                "peak_rss_gib": 0.001,
                "full_pairs_used": 64,
                "error_message": None,
                "candidate_attempts": [],
                "recorded_at": "2026-09-30T12:00:00Z",
            })
    # For instance 31, complete the first 5 methods (leaving 4 pending)
    last_inst = env["instances"][31]
    completed_5 = list(REQUIRED_COMPARATOR_METHODS[:5])
    for m in completed_5:
        records.append({
            "instance_id": last_inst.instance_id,
            "instance_index": 31,
            "method": m,
            "outcome": "solver_timeout",
            "complete_online_wall_seconds": 0.01,
            "ranking_wall_seconds": 0.005,
            "solving_wall_seconds": 0.005,
            "verification_wall_seconds": 0.0,
            "candidates_tested_count": 0,
            "candidates_remaining_count": 0,
            "recovered_master_key": None,
            "verified_on_fresh_pairs": False,
            "peak_rss_gib": 0.001,
            "full_pairs_used": 64,
            "error_message": None,
            "candidate_attempts": [],
            "recorded_at": "2026-09-30T12:00:00Z",
        })

    runner._save_ledger_atomic(ledger_path, records, exec_id, "CONFIRMATION")

    worker_calls = []

    def mock_worker(**kwargs):
        worker_calls.append(kwargs)
        method = kwargs["method"]
        return MethodWorkerOutcome(
            method=method,
            outcome="queue_exhausted" if not method.startswith("oracle_") else "verified_recovery",
            complete_online_wall_seconds=0.01,
            ranking_wall_seconds=0.005,
            solving_wall_seconds=0.005,
            verification_wall_seconds=0.0,
            candidates_tested_count=0,
            candidates_remaining_count=0,
            recovered_master_key=last_inst.master_key if method.startswith("oracle_") else None,
            verified_on_fresh_pairs=bool(method.startswith("oracle_")),
            candidate_attempts=(),
            peak_rss_gib=0.001,
        )

    monkeypatch.setattr(runner, "run_method_worker", mock_worker)
    monkeypatch.setattr(runner, "verify_runtime_provenance", lambda *_: env["resource_paths"])
    monkeypatch.setattr(runner.sys, "argv", [
        "evaluate_simon_confirmation.py",
        "--manifest", str(env["manifest_path"]),
        "--dataset", str(env["dataset_path"]),
        "--certificate", str(env["cert_path"]),
        "--output", str(ledger_path),
        "--resume",
    ])

    ret = runner.main()
    assert ret == 0

    # Exactly 4 pending calls (288 total - 284 completed = 4 pending)
    assert len(worker_calls) == 4
    pending_called = [c["method"] for c in worker_calls]
    assert set(pending_called) == set(ALL_CONFIRMATION_METHODS) - set(completed_5)

    # Final ledger has all 288 records
    with open(ledger_path, encoding="utf-8") as f:
        final_ledger = json.load(f)
    assert len(final_ledger["records"]) == 288


@pytest.mark.parametrize("seed,count", [(880001, 1_000_000), (880002, 99_968)])
def test_training_replay_detects_first_and_last_actual_initial_draws(seed, count):
    drawn = np.random.default_rng(seed).integers(0, 65536, size=(count, 4), dtype=np.uint16)
    requested = [tuple(int(word) for word in drawn[index]) for index in (0, count - 1)]
    assert _training_key_overlaps(requested, seed=seed, key_count=count) == set(requested)


def test_certificate_rejects_actual_numpy_training_key():
    public = generate_simon_attack_instances(SimonAttackInstanceConfig(
        instance_count=2, pair_count=64, rounds=10, seed=999999, split_role="demo",
    ))
    training_key = tuple(int(word) for word in np.random.default_rng(880001).integers(
        0, 65536, size=(1, 4), dtype=np.uint16,
    )[0])
    fixtures = [replace(inst, split_role="confirmation", master_key=training_key if i == 0 else inst.master_key)
                for i, inst in enumerate(public)]
    manifest = {"sample_size_specification": {"selected_instances": 2}}
    with pytest.raises(DisjointFamilyCertificateError, match="synthetic_training_seed_880001"):
        build_disjoint_family_certificate(manifest, fixtures)
