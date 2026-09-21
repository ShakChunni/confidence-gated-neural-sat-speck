"""Comprehensive validation and integrity enforcement for protocol manifests."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy
import pycryptosat
import torch

from neural_sat_speck.data.seed_policy import (
    HISTORICAL_SEEDS,
    PROTECTED_RESEARCH_SEEDS,
    SeedRole,
    validate_seed_for_role,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

REQUIRED_TOP_LEVEL_SECTIONS = {
    "manifest_version",
    "protocol_id",
    "protocol_status",
    "created_at",
    "governing_decisions",
    "target_publication",
    "seed_governance",
    "protected_dataset",
    "cryptographic_parameters",
    "search_and_resource_envelope",
    "backend_specification",
    "artifacts",
    "code_manifest",
    "formula_metrics",
    "comparator_methods",
    "failure_taxonomy",
    "statistical_plan",
    "environment_provenance",
}

RETIRED_OR_OPENED_SEEDS: frozenset[int] = frozenset(
    HISTORICAL_SEEDS
    | PROTECTED_RESEARCH_SEEDS
    | {
        98001,
        97001,
        99001,
        100001,
        101001,
        200001,
        201001,
        300001,
        400001,
        400002,
        406737,
    }
)

EXPECTED_METHODS = {
    "hierarchical_neural_screened",
    "hierarchical_neural_full",
    "hierarchical_neural_classical",
    "random_sat",
    "sat_only_4obs",
    "sat_only_all_observations",
}

REQUIRED_FAILURE_CATEGORIES = {
    "successful_verified_recovery",
    "expected_candidate_unsat",
    "candidates_exhausted_without_unresolved",
    "unsuccessful_with_unresolved_timeouts",
    "solver_timeout",
    "global_timeout",
    "solver_unknown",
    "solver_exception",
    "decoding_failure",
    "verification_failure",
}

REQUIRED_TAXONOMY_FIELDS = {
    "category",
    "description",
    "counts_as_verified_recovery",
    "counts_as_unconditional_failure",
    "is_resolved_sat_state",
    "failure_domain",
}

ALLOWED_FAILURE_DOMAINS = {
    "none",
    "search_progress",
    "ranking",
    "solver",
    "envelope",
    "system",
    "verification",
}

TAXONOMY_LOOKUP: dict[str, dict[str, Any]] = {
    "successful_verified_recovery": {
        "counts_as_verified_recovery": True,
        "counts_as_unconditional_failure": False,
        "is_resolved_sat_state": True,
        "failure_domain": "none",
    },
    "expected_candidate_unsat": {
        "counts_as_verified_recovery": False,
        "counts_as_unconditional_failure": False,
        "is_resolved_sat_state": True,
        "failure_domain": "search_progress",
    },
    "candidates_exhausted_without_unresolved": {
        "counts_as_verified_recovery": False,
        "counts_as_unconditional_failure": True,
        "is_resolved_sat_state": True,
        "failure_domain": "ranking",
    },
    "unsuccessful_with_unresolved_timeouts": {
        "counts_as_verified_recovery": False,
        "counts_as_unconditional_failure": True,
        "is_resolved_sat_state": False,
        "failure_domain": "solver",
    },
    "solver_timeout": {
        "counts_as_verified_recovery": False,
        "counts_as_unconditional_failure": True,
        "is_resolved_sat_state": False,
        "failure_domain": "solver",
    },
    "global_timeout": {
        "counts_as_verified_recovery": False,
        "counts_as_unconditional_failure": True,
        "is_resolved_sat_state": False,
        "failure_domain": "envelope",
    },
    "solver_unknown": {
        "counts_as_verified_recovery": False,
        "counts_as_unconditional_failure": True,
        "is_resolved_sat_state": False,
        "failure_domain": "solver",
    },
    "solver_exception": {
        "counts_as_verified_recovery": False,
        "counts_as_unconditional_failure": True,
        "is_resolved_sat_state": False,
        "failure_domain": "system",
    },
    "decoding_failure": {
        "counts_as_verified_recovery": False,
        "counts_as_unconditional_failure": True,
        "is_resolved_sat_state": True,
        "failure_domain": "system",
    },
    "verification_failure": {
        "counts_as_verified_recovery": False,
        "counts_as_unconditional_failure": True,
        "is_resolved_sat_state": True,
        "failure_domain": "verification",
    },
}


class ManifestValidationError(ValueError):
    """Raised when a protocol manifest fails integrity, environment, or governance checks."""


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    if not path.exists():
        raise FileNotFoundError(f"Artifact file not found: {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


EXECUTION_CRITICAL_CODE_PATHS: dict[str, str] = {
    "runner_script": "scripts/evaluate_seven_round_comparators.py",
    "analyzer_script": "scripts/analyze_seven_round_results.py",
    "hierarchical_multi_round": "src/neural_sat_speck/attacks/hierarchical_multi_round.py",
    "multi_round_recovery": "src/neural_sat_speck/attacks/multi_round_recovery.py",
    "screened_group_ranking": "src/neural_sat_speck/attacks/screened_group_ranking.py",
    "group_ranking": "src/neural_sat_speck/attacks/group_ranking.py",
    "classical_ranking": "src/neural_sat_speck/attacks/classical_ranking.py",
    "multi_round_instance": "src/neural_sat_speck/attacks/multi_round_instance.py",
    "recovery": "src/neural_sat_speck/attacks/recovery.py",
    "solver_isolation": "src/neural_sat_speck/sat/isolation.py",
    "sat_model": "src/neural_sat_speck/sat/model.py",
    "sat_encoding": "src/neural_sat_speck/sat/encoding.py",
    "sat_multi_round": "src/neural_sat_speck/sat/multi_round.py",
    "dimacs": "src/neural_sat_speck/sat/dimacs.py",
    "group_model": "src/neural_sat_speck/neural/group_model.py",
    "numpy_inference": "src/neural_sat_speck/neural/numpy_inference.py",
    "manifest_validator": "src/neural_sat_speck/protocol/manifest_validator.py",
    "evaluation_metrics": "src/neural_sat_speck/evaluation/metrics.py",
    "speck_cipher": "src/neural_sat_speck/speck/cipher.py",
    "speck_key_schedule": "src/neural_sat_speck/speck/key_schedule.py",
    "speck_primitives": "src/neural_sat_speck/speck/primitives.py",
    "data_attack": "src/neural_sat_speck/data/attack.py",
    "seed_policy": "src/neural_sat_speck/data/seed_policy.py",
}


def compute_execution_critical_code_hashes(base_dir: Path | None = None) -> dict[str, str]:
    """Compute current SHA-256 hashes of all execution-critical code files."""
    if base_dir is None:
        base_dir = REPOSITORY_ROOT
    hashes: dict[str, str] = {}
    for key, rel_path in EXECUTION_CRITICAL_CODE_PATHS.items():
        p = base_dir / rel_path
        if not p.exists():
            raise FileNotFoundError(f"Execution-critical code file not found: {p}")
        hashes[key] = compute_sha256(p)
    return hashes



def classify_attack_outcome(
    status: str,
    success: bool,
    unresolved_timeouts: int = 0,
) -> dict[str, Any]:
    """Map raw attack status and success to the 4 orthogonal failure taxonomy fields.

    Parameters
    ----------
    status : str
        Raw execution status from the attack runner.
    success : bool
        Whether fresh-pair verification succeeded.
    unresolved_timeouts : int
        Count of candidate SAT queries that timed out without SAT/UNSAT.

    Returns
    -------
    dict[str, Any]
        Dictionary containing category, counts_as_verified_recovery,
        counts_as_unconditional_failure, is_resolved_sat_state, and failure_domain.
    """
    if success:
        cat = "successful_verified_recovery"
    elif status == "global_timeout":
        cat = "global_timeout"
    elif status == "solver_timeout":
        cat = "solver_timeout"
    elif status in ("sat_verification_failed", "verification_failure"):
        cat = "verification_failure"
    elif status in ("parse_error", "decoding_failure"):
        cat = "decoding_failure"
    elif status in ("process_error", "solver_exception"):
        cat = "solver_exception"
    elif status in ("indeterminate", "solver_unknown"):
        cat = "solver_unknown"
    elif status == "unsuccessful_with_unresolved_timeouts":
        cat = "unsuccessful_with_unresolved_timeouts"
    elif status in (
        "unsat",
        "proven_unsat",
        "candidates_exhausted",
        "candidates_exhausted_without_unresolved",
        "candidates_exhausted_without_unresolved_attempts",
    ):
        if unresolved_timeouts > 0:
            cat = "unsuccessful_with_unresolved_timeouts"
        else:
            cat = "candidates_exhausted_without_unresolved"
    else:
        # Fallback to general unresolved timeout if unresolved > 0, else exhausted
        if unresolved_timeouts > 0:
            cat = "unsuccessful_with_unresolved_timeouts"
        else:
            cat = "candidates_exhausted_without_unresolved"

    meta = TAXONOMY_LOOKUP.get(cat, TAXONOMY_LOOKUP["candidates_exhausted_without_unresolved"])
    return {
        "category": cat,
        "counts_as_verified_recovery": meta["counts_as_verified_recovery"],
        "counts_as_unconditional_failure": meta["counts_as_unconditional_failure"],
        "is_resolved_sat_state": meta["is_resolved_sat_state"],
        "failure_domain": meta["failure_domain"],
    }


def validate_environment_provenance(
    manifest_data: dict[str, Any],
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Verify that current runtime environment matches declared provenance.

    Parameters
    ----------
    manifest_data : dict[str, Any]
        Parsed JSON dictionary of the protocol manifest.
    strict : bool
        If True, raises ManifestValidationError on any mismatch.
        If False, logs or includes mismatches in the returned dict.

    Returns
    -------
    dict[str, Any]
        Summary of observed environment and match status.
    """
    env = manifest_data.get("environment_provenance")
    if not isinstance(env, dict):
        raise ManifestValidationError("environment_provenance must be a dictionary")

    current_python = sys.version.split()[0]
    expected_python = env.get("python_version")
    if expected_python and current_python != expected_python:
        msg = f"Python version mismatch: current={current_python!r}, manifest={expected_python!r}"
        if strict:
            raise ManifestValidationError(msg)

    current_torch = torch.__version__
    expected_torch = env.get("torch_version")
    if expected_torch and current_torch != expected_torch:
        msg = f"PyTorch version mismatch: current={current_torch!r}, manifest={expected_torch!r}"
        if strict:
            raise ManifestValidationError(msg)

    current_numpy = numpy.__version__
    expected_numpy = env.get("numpy_version")
    if expected_numpy and current_numpy != expected_numpy:
        msg = f"NumPy version mismatch: current={current_numpy!r}, manifest={expected_numpy!r}"
        if strict:
            raise ManifestValidationError(msg)

    current_pycryptosat = pycryptosat.__version__
    expected_pycryptosat = env.get("pycryptosat_version")
    if expected_pycryptosat and current_pycryptosat != expected_pycryptosat:
        msg = f"pycryptosat version mismatch: current={current_pycryptosat!r}, manifest={expected_pycryptosat!r}"
        if strict:
            raise ManifestValidationError(msg)

    current_machine = platform.machine()
    expected_machine = env.get("machine")
    if expected_machine and current_machine != expected_machine:
        msg = f"Platform machine mismatch: current={current_machine!r}, manifest={expected_machine!r}"
        if strict:
            raise ManifestValidationError(msg)

    return {
        "python": current_python,
        "torch": current_torch,
        "numpy": current_numpy,
        "pycryptosat": current_pycryptosat,
        "machine": current_machine,
        "all_matched": True,
    }


def validate_protocol_manifest(
    manifest_data: dict[str, Any],
    base_dir: Path | None = None,
    *,
    verify_file_hashes: bool = True,
    allow_frozen_authorized: bool = True,
) -> None:
    """Validate completeness, integrity, taxonomy, and seed governance of a protocol manifest.

    Parameters
    ----------
    manifest_data : dict[str, Any]
        Parsed JSON dictionary of the protocol manifest.
    base_dir : Path | None
        Repository root directory against which artifact paths are resolved.
    verify_file_hashes : bool
        If True, reads on-disk artifact files and confirms SHA-256 digests match.
    allow_frozen_authorized : bool
        If True, permits protocol_status == 'FROZEN_AUTHORIZED' with authorized seed.
        If False (pre-freeze default), requires 'PROPOSED_NOT_AUTHORIZED' and null seed.

    Raises
    ------
    ManifestValidationError
        If any specification, hash, seed governance, or method check fails.
    """
    if base_dir is None:
        base_dir = REPOSITORY_ROOT

    if not isinstance(manifest_data, dict):
        raise ManifestValidationError("Manifest top level must be a JSON object")

    # 1. Top-Level Sections Check
    missing_sections = REQUIRED_TOP_LEVEL_SECTIONS - set(manifest_data.keys())
    if missing_sections:
        raise ManifestValidationError(
            f"Manifest missing required top-level sections: {sorted(missing_sections)}"
        )

    # 2. Version and Status
    version = manifest_data.get("manifest_version")
    if version != "1.0.0":
        raise ManifestValidationError(
            f"Unsupported manifest_version: {version!r}; expected '1.0.0'"
        )

    status = manifest_data.get("protocol_status")
    if status == "PROPOSED_NOT_AUTHORIZED":
        pass  # Pre-freeze development status
    elif status == "FROZEN_AUTHORIZED":
        if not allow_frozen_authorized:
            raise ManifestValidationError(
                f"Pre-freeze protocol_status must be 'PROPOSED_NOT_AUTHORIZED'; found {status!r}. "
                "allow_frozen_authorized=True is required to accept FROZEN_AUTHORIZED status."
            )
    else:
        raise ManifestValidationError(
            f"Invalid protocol_status: {status!r}; expected 'PROPOSED_NOT_AUTHORIZED' or 'FROZEN_AUTHORIZED'"
        )

    # 3. Seed Governance
    seed_gov = manifest_data.get("seed_governance")
    if not isinstance(seed_gov, dict):
        raise ManifestValidationError("seed_governance must be a dictionary")
    protected_seed = seed_gov.get("protected_dataset_seed")

    if status == "PROPOSED_NOT_AUTHORIZED":
        if protected_seed is not None:
            raise ManifestValidationError(
                f"Protected seed must be null before authorization; found {protected_seed!r}"
            )
    elif status == "FROZEN_AUTHORIZED":
        if protected_seed is None or not isinstance(protected_seed, int):
            raise ManifestValidationError(
                f"FROZEN_AUTHORIZED status requires a non-null integer protected_dataset_seed; found {protected_seed!r}"
            )
        if protected_seed in RETIRED_OR_OPENED_SEEDS:
            raise ManifestValidationError(
                f"Protected seed {protected_seed} is in RETIRED_OR_OPENED_SEEDS "
                f"({sorted(RETIRED_OR_OPENED_SEEDS)}) and cannot be used for prospective evaluation"
            )
        auth = seed_gov.get("authorization")
        if not isinstance(auth, dict) or not auth.get("authorized_by") or not auth.get("authorization_decision"):
            raise ManifestValidationError(
                "FROZEN_AUTHORIZED status requires seed_governance.authorization with 'authorized_by' and 'authorization_decision'"
            )
        decision = auth.get("authorization_decision")
        if decision in ("D-025", "D-027", "D-028", "D-029", "D-030"):
            raise ManifestValidationError(
                f"Authorization decision {decision!r} is a development-only or prior decision and cannot satisfy "
                "prospective evaluation authorization. Prospective evaluation requires a new explicit authorization "
                "decision (expected D-031)."
            )
        if decision != "D-031":
            raise ManifestValidationError(
                f"Prospective evaluation requires explicit authorization decision 'D-031'; found {decision!r}"
            )
        gov_decisions = manifest_data.get("governing_decisions", [])
        if decision not in gov_decisions:
            raise ManifestValidationError(
                f"Authorization decision {decision!r} must be listed in manifest governing_decisions {gov_decisions}"
            )
        for req_auth_field in (
            "authorized_by",
            "authorization_decision",
            "authorized_at",
            "authorization_scope",
            "protocol_id",
            "sample_size_n",
        ):
            if not auth.get(req_auth_field):
                raise ManifestValidationError(
                    f"seed_governance.authorization missing required field {req_auth_field!r}"
                )
        if auth.get("sample_size_n") != 128:
            raise ManifestValidationError(
                f"seed_governance.authorization.sample_size_n must be 128; found {auth.get('sample_size_n')!r}"
            )
        if auth.get("protocol_id") != manifest_data.get("protocol_id"):
            raise ManifestValidationError(
                f"seed_governance.authorization.protocol_id {auth.get('protocol_id')!r} does not match manifest {manifest_data.get('protocol_id')!r}"
            )

    sample_size = seed_gov.get("sample_size_n")
    if sample_size != 128:
        raise ManifestValidationError(
            f"Expected sample_size_n = 128; found {sample_size!r}"
        )

    # 3b. Protected Dataset Specification Check
    prot_ds = manifest_data.get("protected_dataset")
    if not isinstance(prot_ds, dict):
        raise ManifestValidationError("protected_dataset must be a dictionary")

    if status == "PROPOSED_NOT_AUTHORIZED":
        for null_field in ("dataset_path", "dataset_id", "sha256", "seed", "generation_config_hash", "schema"):
            if prot_ds.get(null_field) is not None:
                raise ManifestValidationError(
                    f"protected_dataset.{null_field} must be null before authorization; found {prot_ds.get(null_field)!r}"
                )
    elif status == "FROZEN_AUTHORIZED":
        for req_field in (
            "dataset_path",
            "dataset_id",
            "sha256",
            "seed",
            "sample_size_n",
            "role",
            "rounds",
            "ranking_differential_pairs",
            "generation_config_hash",
            "schema",
        ):
            if prot_ds.get(req_field) is None:
                raise ManifestValidationError(
                    f"FROZEN_AUTHORIZED status requires non-null protected_dataset.{req_field}"
                )
        if prot_ds["seed"] != protected_seed:
            raise ManifestValidationError(
                f"protected_dataset.seed ({prot_ds['seed']}) does not match seed_governance.protected_dataset_seed ({protected_seed})"
            )
        if prot_ds["seed"] in RETIRED_OR_OPENED_SEEDS:
            raise ManifestValidationError(
                f"protected_dataset.seed {prot_ds['seed']} is in RETIRED_OR_OPENED_SEEDS and cannot be used for prospective evaluation"
            )
        if prot_ds["sample_size_n"] != 128:
            raise ManifestValidationError(
                f"Expected protected_dataset.sample_size_n = 128; found {prot_ds['sample_size_n']!r}"
            )
        if prot_ds["rounds"] != 7:
            raise ManifestValidationError(
                f"Expected protected_dataset.rounds = 7; found {prot_ds['rounds']!r}"
            )
        if prot_ds["ranking_differential_pairs"] != 64:
            raise ManifestValidationError(
                f"Expected protected_dataset.ranking_differential_pairs = 64; found {prot_ds['ranking_differential_pairs']!r}"
            )

    # 4. Cryptographic Parameters
    crypto = manifest_data.get("cryptographic_parameters")
    if not isinstance(crypto, dict):
        raise ManifestValidationError("cryptographic_parameters must be a dictionary")
    if crypto.get("cipher") != "SPECK32/64":
        raise ManifestValidationError(f"Expected cipher SPECK32/64; found {crypto.get('cipher')!r}")
    if crypto.get("rounds") != 7:
        raise ManifestValidationError(f"Expected rounds = 7; found {crypto.get('rounds')!r}")
    if crypto.get("ranking_differential_pairs") != 64:
        raise ManifestValidationError(
            f"Expected 64 ranking pairs; found {crypto.get('ranking_differential_pairs')!r}"
        )
    if crypto.get("ranking_total_observations") != 128:
        raise ManifestValidationError(
            f"Expected 128 ranking observations; found {crypto.get('ranking_total_observations')!r}"
        )
    if crypto.get("verification_pairs") != 2:
        raise ManifestValidationError(
            f"Expected 2 verification pairs; found {crypto.get('verification_pairs')!r}"
        )
    if crypto.get("verification_rounds") != 7:
        raise ManifestValidationError(
            f"Expected 7 verification rounds; found {crypto.get('verification_rounds')!r}"
        )

    # 5. Search and Resource Envelope
    search = manifest_data.get("search_and_resource_envelope")
    if not isinstance(search, dict):
        raise ManifestValidationError("search_and_resource_envelope must be a dictionary")
    if search.get("m1_stage1_beam_width") != 32:
        raise ManifestValidationError(f"Expected m1 = 32; found {search.get('m1_stage1_beam_width')!r}")
    if search.get("m2_stage2_beam_width") != 32:
        raise ManifestValidationError(f"Expected m2 = 32; found {search.get('m2_stage2_beam_width')!r}")
    if search.get("screen_pairs") != 16:
        raise ManifestValidationError(f"Expected screen_pairs = 16; found {search.get('screen_pairs')!r}")
    if search.get("screen_top_k") != 512:
        raise ManifestValidationError(f"Expected screen_top_k = 512; found {search.get('screen_top_k')!r}")
    if search.get("candidate_budget_max_tuples") != 1024:
        raise ManifestValidationError(
            f"Expected candidate budget = 1024; found {search.get('candidate_budget_max_tuples')!r}"
        )
    if search.get("per_candidate_solver_timeout_seconds") != 10.0:
        raise ManifestValidationError(
            f"Expected solver timeout = 10.0s; found {search.get('per_candidate_solver_timeout_seconds')!r}"
        )
    if search.get("global_method_cutoff_seconds") != 300.0:
        raise ManifestValidationError(
            f"Expected global cutoff = 300.0s; found {search.get('global_method_cutoff_seconds')!r}"
        )
    if search.get("threads") != 1:
        raise ManifestValidationError(f"Expected threads = 1; found {search.get('threads')!r}")

    # 6. Backend Specification
    backend = manifest_data.get("backend_specification")
    if not isinstance(backend, dict):
        raise ManifestValidationError("backend_specification must be a dictionary")
    if backend.get("primary_backend") != "pycryptosat":
        raise ManifestValidationError(
            f"Primary backend must be 'pycryptosat'; found {backend.get('primary_backend')!r}"
        )
    if backend.get("primary_backend_version") != "5.14.7":
        raise ManifestValidationError(
            f"Primary backend version must be '5.14.7'; found {backend.get('primary_backend_version')!r}"
        )
    if backend.get("primary_backend_threads") != 1:
        raise ManifestValidationError(
            f"Primary backend threads must be 1; found {backend.get('primary_backend_threads')!r}"
        )
    if backend.get("rebuild_on_timeout") is not True:
        raise ManifestValidationError("rebuild_on_timeout must be True")

    # 7. Comparator Methods (exact 6, unique, no duplicates)
    methods_list = manifest_data.get("comparator_methods")
    if not isinstance(methods_list, list):
        raise ManifestValidationError("comparator_methods must be a list")
    found_method_ids: list[str] = []
    for m in methods_list:
        if not isinstance(m, dict):
            raise ManifestValidationError("Each comparator method must be a dictionary")
        mid = m.get("method_id")
        if not mid or not isinstance(mid, str):
            raise ManifestValidationError("Comparator method missing valid 'method_id'")
        found_method_ids.append(mid)

        # Method-level parameter validation
        if m.get("backend") != "pycryptosat":
            raise ManifestValidationError(
                f"Primary comparator {mid} backend must be 'pycryptosat'; found {m.get('backend')!r}"
            )
        if m.get("sat_observations") not in (4, 128):
            raise ManifestValidationError(
                f"Method {mid} sat_observations must be 4 or 128; found {m.get('sat_observations')!r}"
            )
        if m.get("candidate_budget_max_tuples") not in (1, 1024):
            raise ManifestValidationError(
                f"Method {mid} candidate_budget_max_tuples must be 1 or 1024; found {m.get('candidate_budget_max_tuples')!r}"
            )

    if len(found_method_ids) != len(set(found_method_ids)):
        raise ManifestValidationError(f"Duplicate method IDs detected in comparator_methods: {found_method_ids}")

    if set(found_method_ids) != EXPECTED_METHODS:
        diff = EXPECTED_METHODS.symmetric_difference(set(found_method_ids))
        raise ManifestValidationError(
            f"Comparator methods mismatch. Expected {EXPECTED_METHODS}; difference: {diff}"
        )

    # 8. Failure Taxonomy Check (including the 4 orthogonal fields)
    tax_list = manifest_data.get("failure_taxonomy")
    if not isinstance(tax_list, list):
        raise ManifestValidationError("failure_taxonomy must be a list of category definitions")
    found_categories = set()
    for cat_entry in tax_list:
        if not isinstance(cat_entry, dict):
            raise ManifestValidationError("Failure taxonomy entry must be a dictionary")
        cat_name = cat_entry.get("category")
        if not cat_name:
            raise ManifestValidationError("Failure taxonomy entry missing 'category'")
        found_categories.add(cat_name)

        missing_fields = REQUIRED_TAXONOMY_FIELDS - set(cat_entry.keys())
        if missing_fields:
            raise ManifestValidationError(
                f"Failure taxonomy entry '{cat_name}' missing required fields: {sorted(missing_fields)}"
            )

        if not isinstance(cat_entry["counts_as_verified_recovery"], bool):
            raise ManifestValidationError(
                f"Category '{cat_name}': counts_as_verified_recovery must be bool"
            )
        if not isinstance(cat_entry["counts_as_unconditional_failure"], bool):
            raise ManifestValidationError(
                f"Category '{cat_name}': counts_as_unconditional_failure must be bool"
            )
        if not isinstance(cat_entry["is_resolved_sat_state"], bool):
            raise ManifestValidationError(
                f"Category '{cat_name}': is_resolved_sat_state must be bool"
            )
        if cat_entry["failure_domain"] not in ALLOWED_FAILURE_DOMAINS:
            raise ManifestValidationError(
                f"Category '{cat_name}': failure_domain {cat_entry['failure_domain']!r} not in {ALLOWED_FAILURE_DOMAINS}"
            )

        # Invariant checks:
        if cat_entry["counts_as_verified_recovery"]:
            if cat_entry["counts_as_unconditional_failure"]:
                raise ManifestValidationError(
                    f"Category '{cat_name}' cannot both count as recovery and failure"
                )
            if not cat_entry["is_resolved_sat_state"]:
                raise ManifestValidationError(
                    f"Category '{cat_name}' must be a resolved SAT state"
                )

        # Do not classify normal candidate UNSAT as a backend failure
        if cat_name in ("expected_candidate_unsat", "candidates_exhausted_without_unresolved"):
            if cat_entry.get("is_backend_failure") is True:
                raise ManifestValidationError(
                    f"Category '{cat_name}' must NOT be classified as a backend failure (is_backend_failure must be False)"
                )

    missing_cats = REQUIRED_FAILURE_CATEGORIES - found_categories
    if missing_cats:
        raise ManifestValidationError(f"failure_taxonomy missing required categories: {sorted(missing_cats)}")

    # 9. Statistical Plan & Multiplicity Handling
    stat_plan = manifest_data.get("statistical_plan")
    if not isinstance(stat_plan, dict):
        raise ManifestValidationError("statistical_plan must be a dictionary")
    if not stat_plan.get("primary_estimand"):
        raise ManifestValidationError("statistical_plan missing 'primary_estimand'")
    primary_contrast = stat_plan.get("primary_paired_contrast")
    if not isinstance(primary_contrast, dict):
        raise ManifestValidationError("statistical_plan must define a single 'primary_paired_contrast'")
    if primary_contrast.get("method_a") not in EXPECTED_METHODS or primary_contrast.get("method_b") not in EXPECTED_METHODS:
        raise ManifestValidationError("primary_paired_contrast references unknown method IDs")
    if primary_contrast.get("alpha") != 0.05:
        raise ManifestValidationError("primary_paired_contrast alpha must be 0.05")

    multiplicity = stat_plan.get("multiplicity_control")
    if not isinstance(multiplicity, dict):
        raise ManifestValidationError("statistical_plan must define 'multiplicity_control'")
    if not multiplicity.get("adjusted_secondary_contrasts"):
        raise ManifestValidationError("multiplicity_control must specify 'adjusted_secondary_contrasts'")

    # 10. Formula Metrics & Provenance
    f_metrics = manifest_data.get("formula_metrics")
    if not isinstance(f_metrics, dict):
        raise ManifestValidationError("formula_metrics must be a dictionary")
    for f_key in ("sat_4obs", "sat_128obs"):
        if f_key not in f_metrics:
            raise ManifestValidationError(f"formula_metrics missing required key '{f_key}'")
        fm = f_metrics[f_key]
        if not fm.get("provenance"):
            raise ManifestValidationError(f"formula_metrics['{f_key}'] missing 'provenance'")
        if not fm.get("variables") or not fm.get("cnf_clauses_unpinned") or not fm.get("xor_clauses"):
            raise ManifestValidationError(f"formula_metrics['{f_key}'] missing counts")

    # 11. Environment Provenance
    env_prov = manifest_data.get("environment_provenance")
    if not isinstance(env_prov, dict):
        raise ManifestValidationError("environment_provenance must be a dictionary")
    for field in ("python_version", "torch_version", "pycryptosat_version"):
        if not env_prov.get(field):
            raise ManifestValidationError(f"environment_provenance missing '{field}'")

    # 12. File Hashes Verification
    if verify_file_hashes:
        artifacts = manifest_data.get("artifacts", {})
        for name, art in artifacts.items():
            rel_path = art.get("path")
            expected_hash = art.get("sha256")
            if not rel_path or not expected_hash:
                raise ManifestValidationError(f"Artifact {name} missing path or sha256")
            full_path = base_dir / rel_path
            actual_hash = compute_sha256(full_path)
            if actual_hash != expected_hash:
                raise ManifestValidationError(
                    f"Hash mismatch for artifact {name} ({rel_path}): "
                    f"expected {expected_hash}, computed {actual_hash}"
                )

        code_manifest = manifest_data.get("code_manifest", {})
        missing_code_keys = set(EXECUTION_CRITICAL_CODE_PATHS.keys()) - set(code_manifest.keys())
        if missing_code_keys:
            raise ManifestValidationError(
                f"code_manifest missing required execution-critical entries: {sorted(missing_code_keys)}"
            )
        for name, code in code_manifest.items():
            rel_path = code.get("path")
            expected_hash = code.get("sha256")
            if not rel_path or not expected_hash:
                raise ManifestValidationError(f"Code entry {name} missing path or sha256")
            full_path = base_dir / rel_path
            actual_hash = compute_sha256(full_path)
            if actual_hash != expected_hash:
                raise ManifestValidationError(
                    f"Hash mismatch for code entry {name} ({rel_path}): "
                    f"expected {expected_hash}, computed {actual_hash}"
                )



def verify_attack_dataset_integrity(
    dataset_path: Path,
    manifest_data: dict[str, Any] | None = None,
    *,
    is_prospective: bool = True,
) -> dict[str, Any]:
    """Verify integrity, hash, seed governance, and schema of an attack dataset.

    Parameters
    ----------
    dataset_path : Path
        Path to the attack dataset JSON file on disk.
    manifest_data : dict[str, Any] | None
        Parsed JSON dictionary of the protocol manifest.
    is_prospective : bool
        If True, validates against protected_dataset section and rejects retired/opened seeds.
        If False, validates development/opened-diagnostic structure.

    Returns
    -------
    dict[str, Any]
        Validation summary containing instance count, seed, sha256, and verified status.

    Raises
    ------
    ManifestValidationError
        If dataset fails hash, instance count, rounds, pairs, seed, or duplicate-ID checks.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Attack dataset file not found: {dataset_path}")

    actual_sha256 = compute_sha256(dataset_path)

    try:
        with dataset_path.open("r", encoding="utf-8") as handle:
            doc = json.load(handle)
    except Exception as err:
        raise ManifestValidationError(f"Failed to parse attack dataset JSON: {err}") from err

    if not isinstance(doc, dict):
        raise ManifestValidationError("Attack dataset top level must be a JSON object")

    cipher = doc.get("cipher")
    if cipher != "SPECK32/64":
        raise ManifestValidationError(f"Expected cipher SPECK32/64; found {cipher!r}")

    dataset_id = doc.get("dataset_id")
    if not dataset_id or not isinstance(dataset_id, str):
        raise ManifestValidationError("Attack dataset missing valid 'dataset_id'")

    metadata = doc.get("metadata")
    if not isinstance(metadata, dict):
        raise ManifestValidationError("Attack dataset missing 'metadata' dictionary")

    rounds = metadata.get("rounds")
    if rounds != 7:
        raise ManifestValidationError(f"Expected 7 rounds in dataset metadata; found {rounds!r}")

    pair_count = metadata.get("pair_count")
    if pair_count != 64:
        raise ManifestValidationError(f"Expected pair_count = 64; found {pair_count!r}")

    instances = doc.get("instances")
    if not isinstance(instances, list):
        raise ManifestValidationError("Attack dataset missing 'instances' list")

    instance_ids: list[str] = []
    for idx, inst in enumerate(instances):
        if not isinstance(inst, dict):
            raise ManifestValidationError(f"Instance {idx} is not a dictionary")
        iid = inst.get("instance_id")
        if not iid or not isinstance(iid, str):
            raise ManifestValidationError(f"Instance {idx} missing valid 'instance_id'")
        instance_ids.append(iid)

    if len(instance_ids) != len(set(instance_ids)):
        raise ManifestValidationError(
            f"Duplicate instance IDs detected in dataset: {len(instance_ids)} instances, "
            f"{len(set(instance_ids))} unique IDs"
        )

    declared_seed = metadata.get("seed")

    if is_prospective:
        if declared_seed in RETIRED_OR_OPENED_SEEDS:
            raise ManifestValidationError(
                f"Prospective dataset cannot use retired/opened seed {declared_seed} "
                f"(retired seeds: {sorted(RETIRED_OR_OPENED_SEEDS)})"
            )
        expected_n = 128
        if manifest_data is not None:
            expected_n = manifest_data.get("seed_governance", {}).get("sample_size_n", 128)
        if len(instances) != expected_n:
            raise ManifestValidationError(
                f"Expected exactly {expected_n} instances for prospective evaluation; found {len(instances)}"
            )

        if manifest_data is not None:
            prot_ds = manifest_data.get("protected_dataset", {})
            if prot_ds.get("sha256") and actual_sha256 != prot_ds["sha256"]:
                raise ManifestValidationError(
                    f"Dataset SHA-256 {actual_sha256} does not match manifest protected_dataset.sha256 {prot_ds['sha256']}"
                )
            if prot_ds.get("dataset_id") and dataset_id != prot_ds["dataset_id"]:
                raise ManifestValidationError(
                    f"Dataset ID {dataset_id!r} does not match manifest protected_dataset.dataset_id {prot_ds['dataset_id']!r}"
                )
            if prot_ds.get("seed") is not None and declared_seed != prot_ds["seed"]:
                raise ManifestValidationError(
                    f"Dataset declared seed {declared_seed} does not match manifest protected_dataset.seed {prot_ds['seed']}"
                )
    else:
        # Development mode
        if len(instances) == 0:
            raise ManifestValidationError("Development dataset contains 0 instances")

    return {
        "dataset_path": str(dataset_path),
        "dataset_id": dataset_id,
        "seed": declared_seed,
        "instance_count": len(instances),
        "sha256": actual_sha256,
        "is_prospective": is_prospective,
    }


def validate_manifest_for_prospective_execution(
    manifest_data_or_path: dict[str, Any] | Path,
    cli_args: Any | None = None,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Strict pre-execution gate for prospective evaluation on protected data.

    Enforces that:
    1. protocol_status is strictly FROZEN_AUTHORIZED (refuses execution if PROPOSED_NOT_AUTHORIZED).
    2. protected_dataset_seed is non-null integer and not in RETIRED_OR_OPENED_SEEDS.
    3. Active environment matches declared provenance.
    4. All artifact and code hashes match on disk.
    5. CLI execution arguments match manifest search parameters.
    6. Protected dataset integrity and hash match if dataset path is provided.

    Raises
    ------
    ManifestValidationError
        If any prospective precondition is not satisfied.
    """
    if base_dir is None:
        base_dir = REPOSITORY_ROOT

    if isinstance(manifest_data_or_path, (str, Path)):
        p = Path(manifest_data_or_path)
        if not p.exists():
            raise FileNotFoundError(f"Protocol manifest not found: {p}")
        with p.open("r", encoding="utf-8") as handle:
            manifest_data = json.load(handle)
    else:
        manifest_data = manifest_data_or_path

    # Check status first - fail closed
    status = manifest_data.get("protocol_status")
    if status != "FROZEN_AUTHORIZED":
        raise ManifestValidationError(
            f"Prospective execution refused: protocol status is {status!r}. "
            "Execution on protected data is strictly forbidden until formal supervisor "
            "authorization transitions status to 'FROZEN_AUTHORIZED'."
        )

    seed = manifest_data.get("seed_governance", {}).get("protected_dataset_seed")
    if seed is None:
        raise ManifestValidationError(
            "Prospective execution refused: protected_dataset_seed is null. "
            "Seed generation and assignment require prior authorization."
        )
    if seed in RETIRED_OR_OPENED_SEEDS:
        raise ManifestValidationError(
            f"Prospective execution refused: protected seed {seed} is in RETIRED_OR_OPENED_SEEDS "
            f"({sorted(RETIRED_OR_OPENED_SEEDS)}). Historical or quarantined seeds cannot be used for prospective evaluation."
        )
    try:
        validate_seed_for_role(seed, SeedRole.PROSPECTIVE_VALIDATION)
    except ValueError as err:
        raise ManifestValidationError(
            f"Prospective execution refused by seed policy: {err}"
        ) from err

    # Perform full structural and hash check
    validate_protocol_manifest(
        manifest_data,
        base_dir=base_dir,
        verify_file_hashes=True,
        allow_frozen_authorized=True,
    )

    # Enforce strict environment match
    validate_environment_provenance(manifest_data, strict=True)

    # If CLI args provided, ensure parameters match manifest
    if cli_args is not None:
        envelope = manifest_data["search_and_resource_envelope"]
        if getattr(cli_args, "m1", None) != envelope["m1_stage1_beam_width"]:
            raise ManifestValidationError(
                f"CLI argument m1={getattr(cli_args, 'm1')} does not match manifest {envelope['m1_stage1_beam_width']}"
            )
        if getattr(cli_args, "m2", None) != envelope["m2_stage2_beam_width"]:
            raise ManifestValidationError(
                f"CLI argument m2={getattr(cli_args, 'm2')} does not match manifest {envelope['m2_stage2_beam_width']}"
            )
        cand_budget = getattr(cli_args, "candidate_budget", None)
        if cand_budget is not None and cand_budget != envelope["candidate_budget_max_tuples"]:
            raise ManifestValidationError(
                f"CLI argument candidate_budget={cand_budget} does not match manifest {envelope['candidate_budget_max_tuples']}"
            )
        batch_size = getattr(cli_args, "batch_size", None)
        if batch_size is not None and batch_size != envelope.get("batch_size", 1024):
            raise ManifestValidationError(
                f"CLI argument batch_size={batch_size} does not match manifest {envelope.get('batch_size')}"
            )
        threads = getattr(cli_args, "threads", None)
        if threads is not None and threads != envelope.get("threads", 1):
            raise ManifestValidationError(
                f"CLI argument threads={threads} does not match manifest {envelope.get('threads')}"
            )
        if getattr(cli_args, "timeout", None) != envelope["per_candidate_solver_timeout_seconds"]:
            raise ManifestValidationError(
                f"CLI timeout={getattr(cli_args, 'timeout')} does not match manifest {envelope['per_candidate_solver_timeout_seconds']}"
            )
        if getattr(cli_args, "global_timeout", None) != envelope["global_method_cutoff_seconds"]:
            raise ManifestValidationError(
                f"CLI global_timeout={getattr(cli_args, 'global_timeout')} does not match manifest {envelope['global_method_cutoff_seconds']}"
            )
        if getattr(cli_args, "start", None) not in (None, 0):
            raise ManifestValidationError(
                f"Prospective execution requires start=0; got {getattr(cli_args, 'start')}"
            )
        if getattr(cli_args, "instances", None) not in (None, 128):
            raise ManifestValidationError(
                f"Prospective execution requires instances=128; got {getattr(cli_args, 'instances')}"
            )
        device_arg = getattr(cli_args, "device", None)
        if device_arg not in (None, "mps"):
            raise ManifestValidationError(
                f"Prospective execution requires device='mps'; got {device_arg!r}"
            )
        expected_manifest_methods = {
            m["method_id"] for m in manifest_data.get("comparator_methods", [])
        }
        cli_methods = getattr(cli_args, "methods", None)
        is_resume = bool(getattr(cli_args, "resume", False))
        if cli_methods is not None:
            cli_methods_set = set(cli_methods)
            if not is_resume:
                if cli_methods_set != expected_manifest_methods:
                    raise ManifestValidationError(
                        f"Fresh prospective launch requires all {len(expected_manifest_methods)} methods; "
                        f"got {cli_methods_set}. Resume mode (--resume) is required to run a subset."
                    )
            else:
                if not cli_methods_set.issubset(expected_manifest_methods):
                    raise ManifestValidationError(
                        f"Resumed prospective run methods {cli_methods_set} must be a subset of {expected_manifest_methods}"
                    )
        dataset_arg = getattr(cli_args, "dataset", None)
        if dataset_arg is not None:
            verify_attack_dataset_integrity(
                Path(dataset_arg),
                manifest_data=manifest_data,
                is_prospective=True,
            )

    return manifest_data


def verify_record_integrity(r: dict[str, Any]) -> bool:
    """Explicitly cross-check fresh-pair recovery consistency.

    Returns True if the record represents an independently verified fresh-pair recovery.
    Raises ValueError if there are contradictory status, classification, key, or cutoff flags.
    """
    success = bool(r.get("success", False))
    counts_recovery = bool(r.get("counts_as_verified_recovery", False))
    counts_fail = bool(r.get("counts_as_unconditional_failure", False))
    status = r.get("status")
    failure_domain = r.get("failure_domain")
    key = r.get("recovered_master_key")
    late_verified = bool(r.get("late_verified_recovery", False))
    cutoff = float(r.get("cutoff_seconds", 300.0))
    wall_sec = float(r.get("method_wall_seconds", r.get("wall_seconds", 0.0)))

    if success:
        if wall_sec >= cutoff:
            raise ValueError(
                f"Contradictory record: success=True but method_wall_seconds ({wall_sec}) >= cutoff ({cutoff}) "
                f"for instance {r.get('instance_id')}"
            )
        if late_verified:
            raise ValueError(
                f"Contradictory record: success=True but late_verified_recovery=True "
                f"for instance {r.get('instance_id')}"
            )
        if not counts_recovery:
            raise ValueError(
                f"Contradictory record: success=True but counts_as_verified_recovery=False "
                f"for instance {r.get('instance_id')}"
            )
        if counts_fail:
            raise ValueError(
                f"Contradictory record: success=True but counts_as_unconditional_failure=True "
                f"for instance {r.get('instance_id')}"
            )
        if status not in ("sat_and_verified", "verified_recovery"):
            raise ValueError(
                f"Contradictory record: success=True but status={status!r} "
                f"for instance {r.get('instance_id')}"
            )
        if failure_domain != "none":
            raise ValueError(
                f"Contradictory record: success=True but failure_domain={failure_domain!r} "
                f"for instance {r.get('instance_id')}"
            )
        if not key or len(key) != 4:
            raise ValueError(
                f"Contradictory record: success=True but invalid recovered_master_key {key!r} "
                f"for instance {r.get('instance_id')}"
            )
        return True
    else:
        if counts_recovery:
            raise ValueError(
                f"Contradictory record: success=False but counts_as_verified_recovery=True "
                f"for instance {r.get('instance_id')}"
            )
        if late_verified:
            if not counts_fail:
                raise ValueError(
                    f"Contradictory record: late_verified_recovery=True but counts_as_unconditional_failure=False "
                    f"for instance {r.get('instance_id')}"
                )
            if status != "global_timeout":
                raise ValueError(
                    f"Contradictory record: late_verified_recovery=True but status={status!r} != 'global_timeout' "
                    f"for instance {r.get('instance_id')}"
                )
            if failure_domain != "envelope":
                raise ValueError(
                    f"Contradictory record: late_verified_recovery=True but failure_domain={failure_domain!r} != 'envelope' "
                    f"for instance {r.get('instance_id')}"
                )
        if not counts_fail:
            raise ValueError(
                f"Contradictory record: success=False but counts_as_unconditional_failure=False "
                f"for instance {r.get('instance_id')}"
            )
        if key is not None:
            raise ValueError(
                f"Contradictory record: success=False but recovered_master_key is not None ({key!r}) "
                f"for instance {r.get('instance_id')}"
            )
        return False



def validate_manifest_for_development_execution(
    manifest_data_or_path: dict[str, Any] | Path,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Validation path for development / opened-seed diagnostic runs."""
    if base_dir is None:
        base_dir = REPOSITORY_ROOT

    if isinstance(manifest_data_or_path, (str, Path)):
        p = Path(manifest_data_or_path)
        if not p.exists():
            raise FileNotFoundError(f"Protocol manifest not found: {p}")
        with p.open("r", encoding="utf-8") as handle:
            manifest_data = json.load(handle)
    else:
        manifest_data = manifest_data_or_path

    validate_protocol_manifest(
        manifest_data,
        base_dir=base_dir,
        verify_file_hashes=True,
        allow_frozen_authorized=True,
    )
    validate_environment_provenance(manifest_data, strict=False)
    return manifest_data


def validate_protocol_manifest_file(
    manifest_path: Path,
    base_dir: Path | None = None,
    *,
    allow_frozen_authorized: bool = True,
) -> dict[str, Any]:
    """Load and validate a manifest JSON file from disk."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as err:
        raise ManifestValidationError(f"Failed to parse manifest JSON: {err}") from err

    if not isinstance(data, dict):
        raise ManifestValidationError("Manifest top level must be a JSON object")

    validate_protocol_manifest(
        data,
        base_dir=base_dir,
        verify_file_hashes=True,
        allow_frozen_authorized=allow_frozen_authorized,
    )
    return data


def save_protocol_manifest(
    manifest_data: dict[str, Any],
    target_path: Path,
    *,
    allow_overwrite: bool = False,
) -> None:
    """Save manifest with overwrite protection."""
    if target_path.exists() and not allow_overwrite:
        raise FileExistsError(
            f"Refusing to overwrite immutable protocol manifest at {target_path}. "
            "Set allow_overwrite=True explicitly if authorized."
        )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest_data, handle, indent=2)
        handle.write("\n")
