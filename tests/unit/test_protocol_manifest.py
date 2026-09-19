"""Unit tests for protocol manifest validation, integrity enforcement, and seed governance."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from neural_sat_speck.protocol.manifest_validator import (
    ManifestValidationError,
    save_protocol_manifest,
    validate_protocol_manifest,
    validate_protocol_manifest_file,
)

MANIFEST_PATH = Path("docs/seven_round_protocol_manifest.json")


def _load_manifest_copy() -> dict:
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_authoritative_manifest_passes_validation() -> None:
    """The authoritative seven-round protocol manifest must pass all validation checks."""
    assert MANIFEST_PATH.exists()
    data = validate_protocol_manifest_file(MANIFEST_PATH)
    assert data["manifest_version"] == "1.0.0"
    assert data["protocol_status"] == "PROPOSED_NOT_AUTHORIZED"
    assert data["seed_governance"]["protected_dataset_seed"] is None
    assert data["search_and_resource_envelope"]["candidate_budget_max_tuples"] == 1024
    assert len(data["comparator_methods"]) == 6
    assert len(data["failure_taxonomy"]) == 10
    assert "primary_paired_contrast" in data["statistical_plan"]


def test_seed_governance_rejection() -> None:
    """Assigning a protected seed before authorization must fail validation."""
    data = _load_manifest_copy()
    data["seed_governance"]["protected_dataset_seed"] = 400001
    with pytest.raises(ManifestValidationError, match="Protected seed must be null"):
        validate_protocol_manifest(data, verify_file_hashes=False)


def test_hash_tamper_rejection() -> None:
    """Altering any artifact or code hash must fail validation."""
    data = _load_manifest_copy()
    data["artifacts"]["stage1_model"]["sha256"] = "0" * 64
    with pytest.raises(ManifestValidationError, match="Hash mismatch for artifact stage1_model"):
        validate_protocol_manifest(data, verify_file_hashes=True)


def test_parameter_tamper_rejection() -> None:
    """Altering any core execution or search parameter must fail validation."""
    data = _load_manifest_copy()

    # Tamper with global cutoff
    bad_cutoff_data = copy.deepcopy(data)
    bad_cutoff_data["search_and_resource_envelope"]["global_method_cutoff_seconds"] = 120.0
    with pytest.raises(ManifestValidationError, match="Expected global cutoff = 300.0s"):
        validate_protocol_manifest(bad_cutoff_data, verify_file_hashes=False)

    # Tamper with rounds
    bad_rounds_data = copy.deepcopy(data)
    bad_rounds_data["cryptographic_parameters"]["rounds"] = 6
    with pytest.raises(ManifestValidationError, match="Expected rounds = 7"):
        validate_protocol_manifest(bad_rounds_data, verify_file_hashes=False)


def test_comparator_methods_completeness() -> None:
    """Omitting any of the six required comparator methods must fail validation."""
    data = _load_manifest_copy()
    data["comparator_methods"] = [
        m for m in data["comparator_methods"] if m["method_id"] != "sat_only_all_observations"
    ]
    with pytest.raises(ManifestValidationError, match="Comparator methods mismatch"):
        validate_protocol_manifest(data, verify_file_hashes=False)


def test_duplicate_method_ids_rejection() -> None:
    """Duplicate comparator method IDs must be rejected."""
    data = _load_manifest_copy()
    data["comparator_methods"].append(data["comparator_methods"][0])
    with pytest.raises(ManifestValidationError, match="Duplicate method IDs detected"):
        validate_protocol_manifest(data, verify_file_hashes=False)


def test_method_parameter_bounds_rejection() -> None:
    """Invalid observation counts or candidate budgets in comparator entries must fail."""
    data = _load_manifest_copy()
    bad_data = copy.deepcopy(data)
    bad_data["comparator_methods"][0]["sat_observations"] = 8  # Not in (4, 128)
    with pytest.raises(ManifestValidationError, match="sat_observations must be 4 or 128"):
        validate_protocol_manifest(bad_data, verify_file_hashes=False)

    bad_data2 = copy.deepcopy(data)
    bad_data2["comparator_methods"][0]["backend"] = "glucose"  # Not pycryptosat
    with pytest.raises(ManifestValidationError, match="backend must be 'pycryptosat'"):
        validate_protocol_manifest(bad_data2, verify_file_hashes=False)


def test_missing_required_section_rejection() -> None:
    """Omitting any required top-level section must fail validation."""
    data = _load_manifest_copy()
    del data["failure_taxonomy"]
    with pytest.raises(ManifestValidationError, match="Manifest missing required top-level sections.*failure_taxonomy"):
        validate_protocol_manifest(data, verify_file_hashes=False)


def test_failure_taxonomy_completeness_and_distinction() -> None:
    """All 10 failure categories must be present, and normal candidate UNSAT must not be classified as backend failure."""
    data = _load_manifest_copy()

    # Missing a category
    bad_tax_data = copy.deepcopy(data)
    bad_tax_data["failure_taxonomy"] = [
        c for c in bad_tax_data["failure_taxonomy"] if c["category"] != "expected_candidate_unsat"
    ]
    with pytest.raises(ManifestValidationError, match="missing required categories.*expected_candidate_unsat"):
        validate_protocol_manifest(bad_tax_data, verify_file_hashes=False)

    # Classifying candidate UNSAT as a backend failure
    bad_tax_data2 = copy.deepcopy(data)
    for c in bad_tax_data2["failure_taxonomy"]:
        if c["category"] == "expected_candidate_unsat":
            c["is_backend_failure"] = True
    with pytest.raises(ManifestValidationError, match="must NOT be classified as a backend failure"):
        validate_protocol_manifest(bad_tax_data2, verify_file_hashes=False)


def test_statistical_plan_and_multiplicity_rejection() -> None:
    """Statistical plan must contain valid primary contrast and Holm multiplicity control."""
    data = _load_manifest_copy()

    bad_plan = copy.deepcopy(data)
    bad_plan["statistical_plan"]["primary_paired_contrast"]["alpha"] = 0.01  # Must be 0.05
    with pytest.raises(ManifestValidationError, match="primary_paired_contrast alpha must be 0.05"):
        validate_protocol_manifest(bad_plan, verify_file_hashes=False)

    bad_plan2 = copy.deepcopy(data)
    bad_plan2["statistical_plan"]["multiplicity_control"]["adjusted_secondary_contrasts"] = []
    with pytest.raises(ManifestValidationError, match="must specify 'adjusted_secondary_contrasts'"):
        validate_protocol_manifest(bad_plan2, verify_file_hashes=False)


def test_formula_metrics_and_environment_rejection() -> None:
    """Missing formula provenance or environment provenance fields must fail."""
    data = _load_manifest_copy()

    bad_metrics = copy.deepcopy(data)
    bad_metrics["formula_metrics"]["sat_4obs"]["provenance"] = ""
    with pytest.raises(ManifestValidationError, match="missing 'provenance'"):
        validate_protocol_manifest(bad_metrics, verify_file_hashes=False)

    bad_env = copy.deepcopy(data)
    del bad_env["environment_provenance"]["pycryptosat_version"]
    with pytest.raises(ManifestValidationError, match="missing 'pycryptosat_version'"):
        validate_protocol_manifest(bad_env, verify_file_hashes=False)


def test_overwrite_protection(tmp_path: Path) -> None:
    """Manifest save must refuse to overwrite existing manifest without explicit flag."""
    test_manifest_path = tmp_path / "manifest.json"
    test_data = {"test": True}
    test_manifest_path.write_text(json.dumps(test_data), encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite immutable protocol manifest"):
        save_protocol_manifest(test_data, test_manifest_path, allow_overwrite=False)

    save_protocol_manifest({"test": "updated"}, test_manifest_path, allow_overwrite=True)
    assert json.loads(test_manifest_path.read_text())["test"] == "updated"


def test_prospective_execution_refused_on_proposed_status() -> None:
    """Prospective evaluation must be refused when protocol_status is PROPOSED_NOT_AUTHORIZED."""
    from neural_sat_speck.protocol.manifest_validator import (
        validate_manifest_for_prospective_execution,
    )

    data = _load_manifest_copy()
    assert data["protocol_status"] == "PROPOSED_NOT_AUTHORIZED"
    with pytest.raises(ManifestValidationError, match="Prospective execution refused.*PROPOSED_NOT_AUTHORIZED"):
        validate_manifest_for_prospective_execution(data)


def test_prospective_execution_refused_on_null_seed() -> None:
    """Prospective evaluation must be refused when protected_dataset_seed is null."""
    from neural_sat_speck.protocol.manifest_validator import (
        validate_manifest_for_prospective_execution,
    )

    data = _load_manifest_copy()
    data["protocol_status"] = "FROZEN_AUTHORIZED"
    data["seed_governance"]["protected_dataset_seed"] = None
    with pytest.raises(ManifestValidationError, match="protected_dataset_seed is null"):
        validate_manifest_for_prospective_execution(data)


def test_prospective_execution_accepted_on_frozen_authorized() -> None:
    """Prospective validation passes when status is FROZEN_AUTHORIZED with full authorization."""
    from neural_sat_speck.protocol.manifest_validator import (
        validate_manifest_for_prospective_execution,
    )

    data = _load_manifest_copy()
    data["protocol_status"] = "FROZEN_AUTHORIZED"
    data["seed_governance"]["protected_dataset_seed"] = 400001
    data["seed_governance"]["authorization"] = {
        "authorized_by": "COMP6002 Project Lead",
        "authorization_decision": "D-030",
        "authorized_at": "2026-09-19T14:00:00Z",
    }
    data["protected_dataset"] = {
        "dataset_path": "datasets/phase11_prospective/prospective_evaluation.json",
        "dataset_id": "speck32_64_attack_phase11_prospective_r7",
        "sha256": "a" * 64,
        "seed": 400001,
        "sample_size_n": 128,
        "role": "PROSPECTIVE_EVALUATION",
        "rounds": 7,
        "ranking_differential_pairs": 64,
        "generation_config_hash": "b" * 64,
        "schema": "neural_sat_speck.attack_shard.v1",
    }
    validated = validate_manifest_for_prospective_execution(data)
    assert validated["protocol_status"] == "FROZEN_AUTHORIZED"
    assert validated["seed_governance"]["protected_dataset_seed"] == 400001


def test_retired_seed_rejection_in_prospective() -> None:
    """Assigning an opened or retired seed (e.g. 300001) in FROZEN_AUTHORIZED must be rejected."""
    from neural_sat_speck.protocol.manifest_validator import (
        validate_manifest_for_prospective_execution,
    )

    for retired_seed in (98001, 97001, 99001, 101001, 201001, 300001):
        data = _load_manifest_copy()
        data["protocol_status"] = "FROZEN_AUTHORIZED"
        data["seed_governance"]["protected_dataset_seed"] = retired_seed
        data["seed_governance"]["authorization"] = {
            "authorized_by": "COMP6002 Project Lead",
            "authorization_decision": "D-030",
            "authorized_at": "2026-09-19T14:00:00Z",
        }
        data["protected_dataset"] = {
            "dataset_path": "datasets/phase11_prospective/prospective_evaluation.json",
            "dataset_id": "speck32_64_attack_phase11_prospective_r7",
            "sha256": "a" * 64,
            "seed": retired_seed,
            "sample_size_n": 128,
            "role": "PROSPECTIVE_EVALUATION",
            "rounds": 7,
            "ranking_differential_pairs": 64,
            "generation_config_hash": "b" * 64,
            "schema": "neural_sat_speck.attack_shard.v1",
        }
        with pytest.raises(ManifestValidationError, match="RETIRED_OR_OPENED_SEEDS"):
            validate_manifest_for_prospective_execution(data)


def test_failure_taxonomy_orthogonal_fields_validation() -> None:
    """Corrupting orthogonal taxonomy fields must fail validation."""
    data = _load_manifest_copy()

    # Missing field
    bad_data = copy.deepcopy(data)
    del bad_data["failure_taxonomy"][0]["failure_domain"]
    with pytest.raises(ManifestValidationError, match="missing required fields.*failure_domain"):
        validate_protocol_manifest(bad_data, verify_file_hashes=False)

    # Invalid failure domain
    bad_data2 = copy.deepcopy(data)
    bad_data2["failure_taxonomy"][0]["failure_domain"] = "invalid_domain"
    with pytest.raises(ManifestValidationError, match="failure_domain 'invalid_domain' not in"):
        validate_protocol_manifest(bad_data2, verify_file_hashes=False)

    # Contradictory fields (both recovery and failure True)
    bad_data3 = copy.deepcopy(data)
    bad_data3["failure_taxonomy"][0]["counts_as_unconditional_failure"] = True
    with pytest.raises(ManifestValidationError, match="cannot both count as recovery and failure"):
        validate_protocol_manifest(bad_data3, verify_file_hashes=False)


def test_classify_attack_outcome_mapping() -> None:
    """classify_attack_outcome must correctly map status and success to 4 orthogonal fields."""
    from neural_sat_speck.protocol.manifest_validator import classify_attack_outcome

    # Success
    res_success = classify_attack_outcome("sat_and_verified", success=True)
    assert res_success["category"] == "successful_verified_recovery"
    assert res_success["counts_as_verified_recovery"] is True
    assert res_success["counts_as_unconditional_failure"] is False
    assert res_success["is_resolved_sat_state"] is True
    assert res_success["failure_domain"] == "none"

    # Global timeout
    res_gt = classify_attack_outcome("global_timeout", success=False)
    assert res_gt["category"] == "global_timeout"
    assert res_gt["counts_as_verified_recovery"] is False
    assert res_gt["counts_as_unconditional_failure"] is True
    assert res_gt["is_resolved_sat_state"] is False
    assert res_gt["failure_domain"] == "envelope"

    # Exhausted with no unresolved timeouts
    res_exh_clean = classify_attack_outcome("candidates_exhausted", success=False, unresolved_timeouts=0)
    assert res_exh_clean["category"] == "candidates_exhausted_without_unresolved"
    assert res_exh_clean["is_resolved_sat_state"] is True
    assert res_exh_clean["failure_domain"] == "ranking"

    # Exhausted with unresolved timeouts
    res_exh_unres = classify_attack_outcome("candidates_exhausted", success=False, unresolved_timeouts=3)
    assert res_exh_unres["category"] == "unsuccessful_with_unresolved_timeouts"
    assert res_exh_unres["is_resolved_sat_state"] is False
    assert res_exh_unres["failure_domain"] == "solver"
