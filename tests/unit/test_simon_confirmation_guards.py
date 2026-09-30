"""Adversarial fixture tests for SIMON confirmation execution guards and resume."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from neural_sat_speck.simon.attack_instance import (
    SimonAttackInstance,
    SimonAttackInstanceConfig,
    generate_simon_attack_instances,
)
from neural_sat_speck.simon.confirmation_guard import (
    FROZEN_QUEUE_CONTROL_SEED,
    REQUIRED_COMPARATOR_METHODS,
    ConfirmationPreflightError,
    DisjointFamilyCertificateError,
    ProvenanceVerificationError,
    ResumeIntegrityError,
    build_disjoint_family_certificate,
    compute_execution_identity,
    derive_confirmation_queue_seed,
    validate_and_filter_resume,
    validate_confirmation_manifest,
    verify_disjoint_family_certificate,
    verify_runtime_provenance,
)


@pytest.fixture
def mock_valid_manifest() -> dict:
    """Synthetic approved manifest fixture for unit tests."""
    return {
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
            "test_dummy": "a" * 64,
        },
    }


def test_unapproved_manifest_rejected(mock_valid_manifest):
    m = copy.deepcopy(mock_valid_manifest)
    m["status"] = "PROPOSED_SEEDLESS_DRAFT_EXECUTION_AND_ANALYSIS_GATES_INCOMPLETE"
    with pytest.raises(ConfirmationPreflightError, match="must be 'APPROVED'"):
        validate_confirmation_manifest(m)


def test_unauthorized_execution_rejected(mock_valid_manifest):
    m = copy.deepcopy(mock_valid_manifest)
    m["confirmation_execution_authorized"] = False
    with pytest.raises(ConfirmationPreflightError, match="confirmation_execution_authorized must be explicitly True"):
        validate_confirmation_manifest(m)


def test_null_seed_rejected(mock_valid_manifest):
    m = copy.deepcopy(mock_valid_manifest)
    m["seed_governance"]["seed"] = None
    with pytest.raises(ConfirmationPreflightError, match="must be an integer"):
        validate_confirmation_manifest(m)


@pytest.mark.parametrize("ineligible_seed", [260926, 406737, 98001, 880001, 880002, 999999, 669981, 101001])
def test_ineligible_seeds_rejected(mock_valid_manifest, ineligible_seed):
    m = copy.deepcopy(mock_valid_manifest)
    m["seed_governance"]["seed"] = ineligible_seed
    with pytest.raises(ConfirmationPreflightError, match="INELIGIBLE_CONFIRMATION_SEEDS"):
        validate_confirmation_manifest(m)


@pytest.mark.parametrize("invalid_n", [None, 16, 48, 128, "32"])
def test_invalid_selected_instances_rejected(mock_valid_manifest, invalid_n):
    m = copy.deepcopy(mock_valid_manifest)
    m["sample_size_specification"]["selected_instances"] = invalid_n
    with pytest.raises(ConfirmationPreflightError, match="selected_instances must be fixed to 32 or 64"):
        validate_confirmation_manifest(m)


def test_missing_or_modified_methods_rejected(mock_valid_manifest):
    m = copy.deepcopy(mock_valid_manifest)
    m["comparator_arms"] = [{"method_id": "sat_only_n4"}]
    with pytest.raises(ConfirmationPreflightError, match="comparator_arms methods mismatch"):
        validate_confirmation_manifest(m)


def test_modified_queue_control_seed_rejected(mock_valid_manifest):
    m = copy.deepcopy(mock_valid_manifest)
    m["random_control_stream"]["control_seed"] = 12345
    with pytest.raises(ConfirmationPreflightError, match="control_seed must be 20261001"):
        validate_confirmation_manifest(m)


def test_spy_verifies_zero_worker_calls_on_preflight_rejection(mock_valid_manifest):
    m = copy.deepcopy(mock_valid_manifest)
    m["status"] = "REJECTED_DRAFT"
    worker_spy = MagicMock()
    with pytest.raises(ConfirmationPreflightError):
        validate_confirmation_manifest(m)
        worker_spy()
    assert worker_spy.call_count == 0


def test_provenance_verification_fails_closed_on_hash_drift(tmp_path):
    dummy_file = tmp_path / "dummy.bin"
    dummy_file.write_bytes(b"content_a")
    manifest = {
        "provenance_hashes": {
            "simon_predicted_ddt_7r_bin_sha256": "0" * 64,
        },
    }
    locs = {"ddt_7r": dummy_file}
    with pytest.raises(ProvenanceVerificationError, match="SHA-256 mismatch"):
        verify_runtime_provenance(manifest, locs)


def test_disjoint_family_certificate_rejects_duplicate_keys(mock_valid_manifest):
    conf_cfg = SimonAttackInstanceConfig(instance_count=1, pair_count=64, rounds=10, seed=777001, split_role="confirmation")
    inst = generate_simon_attack_instances(conf_cfg)[0]
    with pytest.raises(DisjointFamilyCertificateError, match="duplicate master keys"):
        build_disjoint_family_certificate(mock_valid_manifest, [inst] * 32)


def test_disjoint_family_certificate_detects_development_overlap(mock_valid_manifest):
    conf_cfg = SimonAttackInstanceConfig(instance_count=32, pair_count=64, rounds=10, seed=777001, split_role="confirmation")
    conf_instances = list(generate_simon_attack_instances(conf_cfg))

    # Instance sharing key with development seed 260926
    dev_cfg = SimonAttackInstanceConfig(instance_count=1, pair_count=64, rounds=10, seed=260926, split_role="development")
    dev_inst = generate_simon_attack_instances(dev_cfg)[0]

    # Replace first instance with one reusing dev key
    clash_inst = SimonAttackInstance(
        instance_id=conf_instances[0].instance_id,
        split_role="confirmation",
        master_key=dev_inst.master_key,
        rounds=10,
        pairs=dev_inst.pairs,
        verification_pairs=dev_inst.verification_pairs,
    )
    conf_instances[0] = clash_inst

    with pytest.raises(DisjointFamilyCertificateError, match="Disjoint family violation"):
        build_disjoint_family_certificate(mock_valid_manifest, conf_instances)


def test_disjoint_family_certificate_rejects_non_confirmation_role(mock_valid_manifest):
    dev_cfg = SimonAttackInstanceConfig(instance_count=32, pair_count=64, rounds=10, seed=555001, split_role="development")
    dev_instances = generate_simon_attack_instances(dev_cfg)
    with pytest.raises(DisjointFamilyCertificateError, match="must be 'confirmation'"):
        build_disjoint_family_certificate(mock_valid_manifest, dev_instances)


def test_disjoint_family_certificate_build_and_verify_passes_on_fresh_keys(mock_valid_manifest, tmp_path):
    conf_cfg = SimonAttackInstanceConfig(
        instance_count=32, pair_count=64, rounds=10, seed=777001, split_role="confirmation",
    )
    conf_instances = generate_simon_attack_instances(conf_cfg)
    cert_path = tmp_path / "cert.json"
    cert = build_disjoint_family_certificate(mock_valid_manifest, conf_instances, output_path=cert_path)
    assert cert["disjoint_family_verified"] is True
    assert cert["intersection_count"] == 0

    # Verify certificate passes
    verify_disjoint_family_certificate(mock_valid_manifest, conf_instances, cert_path)

    # Tampered certificate fails
    cert_tampered = copy.deepcopy(cert)
    cert_tampered["confirmation_seed"] = 999
    with pytest.raises(DisjointFamilyCertificateError, match="confirmation_seed does not match"):
        verify_disjoint_family_certificate(mock_valid_manifest, conf_instances, cert_tampered)


def _write_ledger_with_sidecar(path: Path, data: dict) -> None:
    raw = json.dumps(data)
    path.write_text(raw, encoding="utf-8")
    import hashlib
    sha = hashlib.sha256(raw.encode()).hexdigest()
    (path.parent / f"{path.name}.sha256").write_text(f"{sha}  {path.name}\n", encoding="utf-8")


def test_resume_guards_reject_unflagged_rerun_on_existing_output(mock_valid_manifest, tmp_path):
    conf_cfg = SimonAttackInstanceConfig(instance_count=2, pair_count=64, rounds=10, seed=777001, split_role="confirmation")
    instances = generate_simon_attack_instances(conf_cfg)
    identity = compute_execution_identity(mock_valid_manifest, "0" * 64)

    ledger_path = tmp_path / "ledger.json"
    _write_ledger_with_sidecar(ledger_path, {"execution_identity": identity, "records": []})

    with pytest.raises(ResumeIntegrityError, match="--resume flag was not provided"):
        validate_and_filter_resume(ledger_path, identity, instances, ["sat_only_n4"], resume_flag=False)


def test_resume_guards_reject_missing_checksum_sidecar(mock_valid_manifest, tmp_path):
    conf_cfg = SimonAttackInstanceConfig(instance_count=2, pair_count=64, rounds=10, seed=777001, split_role="confirmation")
    instances = generate_simon_attack_instances(conf_cfg)
    identity = compute_execution_identity(mock_valid_manifest, "0" * 64)

    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps({"execution_identity": identity, "records": []}), encoding="utf-8")

    with pytest.raises(ResumeIntegrityError, match="Checksum sidecar missing"):
        validate_and_filter_resume(ledger_path, identity, instances, ["sat_only_n4"], resume_flag=True)


def test_resume_guards_reject_corrupted_checksum_sidecar(mock_valid_manifest, tmp_path):
    conf_cfg = SimonAttackInstanceConfig(instance_count=2, pair_count=64, rounds=10, seed=777001, split_role="confirmation")
    instances = generate_simon_attack_instances(conf_cfg)
    identity = compute_execution_identity(mock_valid_manifest, "0" * 64)

    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps({"execution_identity": identity, "records": []}), encoding="utf-8")
    (tmp_path / "ledger.json.sha256").write_text("0" * 64 + "  ledger.json\n", encoding="utf-8")

    with pytest.raises(ResumeIntegrityError, match="Checksum sidecar mismatch"):
        validate_and_filter_resume(ledger_path, identity, instances, ["sat_only_n4"], resume_flag=True)


def test_resume_guards_reject_tampered_execution_identity(mock_valid_manifest, tmp_path):
    conf_cfg = SimonAttackInstanceConfig(instance_count=2, pair_count=64, rounds=10, seed=777001, split_role="confirmation")
    instances = generate_simon_attack_instances(conf_cfg)
    identity = compute_execution_identity(mock_valid_manifest, "0" * 64)

    ledger_path = tmp_path / "ledger.json"
    ledger_data = {
        "execution_identity": copy.deepcopy(identity),
        "records": [],
    }
    # Tamper identity in ledger
    ledger_data["execution_identity"]["global_deadline_seconds"] = 100.0
    _write_ledger_with_sidecar(ledger_path, ledger_data)

    with pytest.raises(ResumeIntegrityError, match="execution_identity does not match"):
        validate_and_filter_resume(ledger_path, identity, instances, ["sat_only_n4"], resume_flag=True)


def test_resume_guards_reject_duplicate_cell(mock_valid_manifest, tmp_path):
    conf_cfg = SimonAttackInstanceConfig(instance_count=2, pair_count=64, rounds=10, seed=777001, split_role="confirmation")
    instances = generate_simon_attack_instances(conf_cfg)
    identity = compute_execution_identity(mock_valid_manifest, "0" * 64)

    rec = {
        "instance_id": instances[0].instance_id, "instance_index": 0, "method": "sat_only_n4",
        "outcome": "solver_timeout", "complete_online_wall_seconds": 1.0, "ranking_wall_seconds": 0.0,
        "solving_wall_seconds": 1.0, "verification_wall_seconds": 0.0,
    }
    ledger_path = tmp_path / "ledger.json"
    ledger_data = {
        "execution_identity": identity,
        "records": [rec, copy.deepcopy(rec)],
    }
    _write_ledger_with_sidecar(ledger_path, ledger_data)

    with pytest.raises(ResumeIntegrityError, match="Duplicate record found"):
        validate_and_filter_resume(ledger_path, identity, instances, ["sat_only_n4"], resume_flag=True)


def test_resume_guards_reject_unverified_claimed_recovery(mock_valid_manifest, tmp_path):
    conf_cfg = SimonAttackInstanceConfig(instance_count=1, pair_count=64, rounds=10, seed=777001, split_role="confirmation")
    instances = generate_simon_attack_instances(conf_cfg)
    identity = compute_execution_identity(mock_valid_manifest, "0" * 64)

    ledger_path = tmp_path / "ledger.json"
    ledger_data = {
        "execution_identity": identity,
        "records": [
            {
                "instance_id": instances[0].instance_id,
                "instance_index": 0,
                "method": "sat_only_n4",
                "outcome": "verified_recovery",
                "recovered_master_key": [0, 0, 0, 0],  # Wrong key
                "complete_online_wall_seconds": 1.0,
                "ranking_wall_seconds": 0.0,
                "solving_wall_seconds": 1.0,
                "verification_wall_seconds": 0.0,
            },
        ],
    }
    _write_ledger_with_sidecar(ledger_path, ledger_data)

    with pytest.raises(ResumeIntegrityError, match="recovered wrong key|failed independent verification"):
        validate_and_filter_resume(ledger_path, identity, instances, ["sat_only_n4"], resume_flag=True)


def test_resume_correctly_filters_completed_cells(mock_valid_manifest, tmp_path):
    conf_cfg = SimonAttackInstanceConfig(instance_count=2, pair_count=64, rounds=10, seed=777001, split_role="confirmation")
    instances = generate_simon_attack_instances(conf_cfg)
    identity = compute_execution_identity(mock_valid_manifest, "0" * 64)

    ledger_path = tmp_path / "ledger.json"
    ledger_data = {
        "execution_identity": identity,
        "records": [
            {
                "instance_id": instances[0].instance_id,
                "instance_index": 0,
                "method": "sat_only_n4",
                "outcome": "solver_timeout",
                "complete_online_wall_seconds": 1.0,
                "ranking_wall_seconds": 0.0,
                "solving_wall_seconds": 1.0,
                "verification_wall_seconds": 0.0,
            },
        ],
    }
    _write_ledger_with_sidecar(ledger_path, ledger_data)

    completed, pending = validate_and_filter_resume(ledger_path, identity, instances, ["sat_only_n4", "sat_only_n128"], resume_flag=True)
    assert len(completed) == 1
    assert (instances[0].instance_id, "sat_only_n4") in completed
    # 4 total cells (2 inst x 2 methods) - 1 completed = 3 pending
    assert len(pending) == 3


def test_forged_zero_certificate_rejected_by_verify_disjoint_family_certificate(mock_valid_manifest, tmp_path):
    # Create instances that reuse a key from demo family (seed 999999)
    demo_inst = generate_simon_attack_instances(SimonAttackInstanceConfig(
        instance_count=1, pair_count=4, rounds=10, seed=999999, split_role="demo",
    ))[0]
    conf_inst = SimonAttackInstance(
        instance_id="simon32_64_inst_0001",
        split_role="confirmation",
        master_key=demo_inst.master_key,
        rounds=10,
        pairs=demo_inst.pairs,
        verification_pairs=demo_inst.verification_pairs,
    )
    instances = [conf_inst] * 32

    # A forged certificate claiming zero intersection
    dataset_repr = json.dumps([inst.to_dict() for inst in instances], sort_keys=True)
    dataset_sha = hashlib.sha256(dataset_repr.encode()).hexdigest()
    manifest_repr = json.dumps(mock_valid_manifest, sort_keys=True)
    manifest_sha = hashlib.sha256(manifest_repr.encode()).hexdigest()

    forged_cert = {
        "certificate_version": "1.0.0",
        "protocol_id": mock_valid_manifest["protocol_id"],
        "confirmation_seed": mock_valid_manifest["seed_governance"]["seed"],
        "selected_instances": 32,
        "instance_count": 32,
        "dataset_sha256": dataset_sha,
        "manifest_sha256": manifest_sha,
        "prior_roles_checked": [],
        "intersection_count": 0,
        "disjoint_family_verified": True,
    }
    cert_path = tmp_path / "forged_cert.json"
    cert_path.write_text(json.dumps(forged_cert), encoding="utf-8")

    # verify_disjoint_family_certificate must reject the forged certificate
    with pytest.raises(DisjointFamilyCertificateError, match="Disjoint family violation"):
        verify_disjoint_family_certificate(mock_valid_manifest, instances, cert_path)


def test_generate_confirmation_dataset_fails_closed_without_authorization(mock_valid_manifest, tmp_path):
    from neural_sat_speck.simon.confirmation_guard import generate_confirmation_dataset

    # Rejection 1: missing authorization_token kwarg
    with pytest.raises(ConfirmationPreflightError, match="requires explicit authorization token"):
        generate_confirmation_dataset(
            manifest_raw=mock_valid_manifest,
            output_dataset_path=tmp_path / "ds.json",
            output_cert_path=tmp_path / "cert.json",
            authorization_token=False,
        )

    # Rejection 2: authorization_token True but confirmation_dataset_generation_authorized False in manifest
    m = copy.deepcopy(mock_valid_manifest)
    m["confirmation_dataset_generation_authorized"] = False
    with pytest.raises(ConfirmationPreflightError, match="confirmation_dataset_generation_authorized must be explicitly True"):
        generate_confirmation_dataset(
            manifest_raw=m,
            output_dataset_path=tmp_path / "ds.json",
            output_cert_path=tmp_path / "cert.json",
            authorization_token=True,
        )


def test_generate_confirmation_dataset_succeeds_when_authorized(mock_valid_manifest, tmp_path):
    from neural_sat_speck.simon.confirmation_guard import generate_confirmation_dataset

    m = copy.deepcopy(mock_valid_manifest)
    m["confirmation_dataset_generation_authorized"] = True
    m["sample_size_specification"]["selected_instances"] = 32

    ds_path = tmp_path / "confirmation_dataset.json"
    cert_path = tmp_path / "confirmation_cert.json"

    out_ds, out_cert = generate_confirmation_dataset(
        manifest_raw=m,
        output_dataset_path=ds_path,
        output_cert_path=cert_path,
        authorization_token=True,
    )

    assert out_ds.is_file()
    assert out_cert.is_file()
    assert (tmp_path / "confirmation_dataset.json.sha256").is_file()

    with open(out_ds, encoding="utf-8") as f:
        ds = json.load(f)
    assert len(ds) == 32
    assert all(inst["split_role"] == "confirmation" for inst in ds)

    with open(out_cert, encoding="utf-8") as f:
        cert = json.load(f)
    assert cert["disjoint_family_verified"] is True
    assert cert["intersection_count"] == 0


def test_derive_confirmation_queue_seed_properties():
    # Determinism
    s1 = derive_confirmation_queue_seed(20261001, "inst_0001", "random_control_128")
    s2 = derive_confirmation_queue_seed(20261001, "inst_0001", "random_control_128")
    assert s1 == s2

    # Distinctness across methods
    s3 = derive_confirmation_queue_seed(20261001, "inst_0001", "random_control_1024")
    assert s1 != s3

    # Distinctness across instances
    s4 = derive_confirmation_queue_seed(20261001, "inst_0002", "random_control_128")
    assert s1 != s4
