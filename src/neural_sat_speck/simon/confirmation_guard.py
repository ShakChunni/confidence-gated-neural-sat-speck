"""Manifest-enforced execution profile and resume guards for SIMON confirmation.

Enforces strict preflight validation, seed governance, disjoint-family certificates,
exact provenance hashing, and execution-identity-bound resume for the prospective
SIMON32/64 R=10 confirmation evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from neural_sat_speck.simon.attack_instance import (
    DEFAULT_SIMON_INPUT_DIFFERENCE,
    SimonAttackInstance,
    SimonAttackInstanceConfig,
    generate_simon_attack_instances,
)
from neural_sat_speck.simon.comparison_input import (
    SimonComparisonInput,
)
from neural_sat_speck.simon.comparison_input import (
    derive_confirmation_queue_seed as derive_confirmation_queue_seed,
)
from neural_sat_speck.simon.confirmation_profile import (
    FROZEN_SCIENTIFIC_SHA256,
    scientific_digest,
    verify_execution_pins,
)
from neural_sat_speck.simon.verification import verify_simon_master_key

# Union of all ineligible confirmation seeds (SPECK + SIMON prior roles)
INELIGIBLE_CONFIRMATION_SEEDS: Final[frozenset[int]] = frozenset(
    {
        1001,
        2001,
        3001,
        4001,
        5001,
        6001,
        51001,
        52001,
        53001,
        61001,
        62001,
        63001,
        64001,
        65001,
        72001,
        73001,
        81001,
        82001,
        83001,
        91001,
        92001,
        93001,
        96001,
        97001,
        98001,
        99001,
        100001,
        101001,
        200001,
        201001,
        260926,
        300001,
        400001,
        400002,
        406737,
        669981,
        880001,
        880002,
        999999,
        8398726052707596859,  # Retired SIMON confirmation Attempt 1 seed (Decision D-041)
    }
)

REQUIRED_COMPARATOR_METHODS: Final[tuple[str, ...]] = (
    "fully_classical_8parent",
    "neural_classical_hybrid_8parent",
    "fully_neural_1parent",
    "faithful_b1_hybrid",
    "random_control_128",
    "random_control_1024",
    "sat_only_n4",
    "sat_only_n128",
)

DIAGNOSTIC_ORACLE_METHOD: Final[str] = "oracle_2key_n4"
ALL_CONFIRMATION_METHODS: Final[tuple[str, ...]] = REQUIRED_COMPARATOR_METHODS + (DIAGNOSTIC_ORACLE_METHOD,)

FROZEN_QUEUE_CONTROL_SEED: Final[int] = 20261001


class ConfirmationPreflightError(ValueError):
    """Raised when confirmation manifest preflight validation fails."""


class ProvenanceVerificationError(ValueError):
    """Raised when code, tool, or artifact hashes fail verification."""


class DisjointFamilyCertificateError(ValueError):
    """Raised when disjoint-family key isolation check fails."""


class ResumeIntegrityError(ValueError):
    """Raised when ledger resume validation fails."""


def file_sha256(path: Path | str) -> str:
    """Compute lowercase 64-character SHA-256 digest of a file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found for hashing: {p}")
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def validate_confirmation_manifest(manifest_raw: dict[str, Any] | str | Path) -> dict[str, Any]:
    """Validate confirmation manifest schema, approval status, seed, methods, and limits."""
    if isinstance(manifest_raw, (str, Path)):
        p = Path(manifest_raw)
        if not p.is_file():
            raise ConfirmationPreflightError(f"Manifest file does not exist: {p}")
        try:
            with open(p, encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as exc:
            raise ConfirmationPreflightError(f"Failed to parse manifest JSON: {exc}") from exc
    elif isinstance(manifest_raw, dict):
        manifest = manifest_raw
    else:
        raise TypeError("manifest_raw must be a dict, str, or Path")

    # 1. Header and Approval Status
    if manifest.get("manifest_version") not in ("1.2.0", "1.3.0", "2.0.0"):
        raise ConfirmationPreflightError(
            f"Unsupported manifest_version: {manifest.get('manifest_version')!r}; expected '1.2.0', '1.3.0', or '2.0.0'"
        )
    if manifest.get("protocol_id") not in (
        "simon32_64_r10_unseen_confirmation_v1",
        "simon32_64_r10_unseen_confirmation_replacement_v1",
    ):
        raise ConfirmationPreflightError(f"Invalid protocol_id: {manifest.get('protocol_id')!r}")
    if manifest.get("evidence_class") != "CONFIRMATION":
        raise ConfirmationPreflightError(f"Invalid evidence_class: {manifest.get('evidence_class')!r}")
    if manifest.get("status") != "APPROVED":
        raise ConfirmationPreflightError(
            f"Manifest status is {manifest.get('status')!r}; must be 'APPROVED' for confirmation"
        )
    if manifest.get("confirmation_execution_authorized") is not True:
        raise ConfirmationPreflightError("confirmation_execution_authorized must be explicitly True")

    readiness = manifest.get("execution_readiness", {})
    if readiness.get("open_approval_gates") != [] or readiness.get("preparation_blockers") != []:
        raise ConfirmationPreflightError("Confirmation approval gates or preparation blockers remain open")
    if readiness.get("status") != "AUTHORIZED_FROZEN_CONFIRMATION":
        raise ConfirmationPreflightError("Confirmation execution readiness is not authorized and frozen")

    # 2. Seed Governance
    seed_gov = manifest.get("seed_governance", {})
    seed = seed_gov.get("seed")
    if seed_gov.get("seed_status") != "APPROVED_FRESH_CONFIRMATION":
        raise ConfirmationPreflightError("Confirmation seed is not registered and approved")
    if seed is None or isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfirmationPreflightError(f"seed_governance.seed must be an integer, got {seed!r}")
    if seed in INELIGIBLE_CONFIRMATION_SEEDS:
        raise ConfirmationPreflightError(
            f"Seed {seed} belongs to INELIGIBLE_CONFIRMATION_SEEDS and cannot be used for confirmation"
        )

    # 3. Target Cipher Specification
    cipher = manifest.get("target_cipher", {})
    if cipher.get("cipher_name") != "SIMON32/64":
        raise ConfirmationPreflightError(f"target_cipher.cipher_name must be 'SIMON32/64', got {cipher.get('cipher_name')!r}")
    if cipher.get("rounds") != 10:
        raise ConfirmationPreflightError(f"target_cipher.rounds must be 10, got {cipher.get('rounds')!r}")
    if tuple(cipher.get("input_difference", ())) != DEFAULT_SIMON_INPUT_DIFFERENCE:
        raise ConfirmationPreflightError(
            f"target_cipher.input_difference must be {DEFAULT_SIMON_INPUT_DIFFERENCE}, got {cipher.get('input_difference')!r}"
        )

    # 4. Sample Size Specification
    sample_spec = manifest.get("sample_size_specification", {})
    selected_n = sample_spec.get("selected_instances")
    if type(selected_n) is not int or selected_n not in (32, 64):
        raise ConfirmationPreflightError(
            f"selected_instances must be fixed to 32 or 64, got {selected_n!r}"
        )
    if sample_spec.get("pair_count") != 64:
        raise ConfirmationPreflightError(f"pair_count must be 64, got {sample_spec.get('pair_count')!r}")
    if sample_spec.get("known_observations") != 128:
        raise ConfirmationPreflightError(f"known_observations must be 128, got {sample_spec.get('known_observations')!r}")
    if sample_spec.get("verification_observations") != 2:
        raise ConfirmationPreflightError(f"verification_observations must be 2, got {sample_spec.get('verification_observations')!r}")

    # 5. Candidate SAT Specification
    cand_sat = manifest.get("candidate_sat_specification", {})
    if cand_sat.get("observations_used") != 4:
        raise ConfirmationPreflightError(f"candidate_sat_specification.observations_used must be 4, got {cand_sat.get('observations_used')!r}")
    if cand_sat.get("formula_variables") != 1856:
        raise ConfirmationPreflightError(f"candidate formula_variables must be 1856, got {cand_sat.get('formula_variables')!r}")
    if cand_sat.get("cnf_clauses") != 2176:
        raise ConfirmationPreflightError(f"candidate cnf_clauses must be 2176, got {cand_sat.get('cnf_clauses')!r}")
    if cand_sat.get("native_xor_clauses") != 1024:
        raise ConfirmationPreflightError(f"candidate native_xor_clauses must be 1024, got {cand_sat.get('native_xor_clauses')!r}")

    # 6. Resource Limits
    limits = manifest.get("resource_limits", {})
    if float(limits.get("global_deadline_seconds", 0.0)) != 300.0:
        raise ConfirmationPreflightError("global_deadline_seconds must be 300.0")
    if float(limits.get("candidate_cutoff_soft_seconds", 0.0)) != 45.0:
        raise ConfirmationPreflightError("candidate_cutoff_soft_seconds must be 45.0")
    if float(limits.get("ranking_time_limit_seconds", 0.0)) != 120.0:
        raise ConfirmationPreflightError("ranking_time_limit_seconds must be 120.0")

    # 7. Comparator Arms & Matrix
    arms = manifest.get("comparator_arms", [])
    arm_ids = tuple(a.get("method_id") for a in arms)
    if arm_ids != REQUIRED_COMPARATOR_METHODS:
        raise ConfirmationPreflightError(
            f"comparator_arms methods mismatch: expected {REQUIRED_COMPARATOR_METHODS}, got {arm_ids}"
        )
    diag = manifest.get("diagnostic_control", {})
    if diag.get("method_id") != DIAGNOSTIC_ORACLE_METHOD:
        raise ConfirmationPreflightError(
            f"diagnostic_control.method_id must be {DIAGNOSTIC_ORACLE_METHOD!r}, got {diag.get('method_id')!r}"
        )

    # 8. Queue Control Stream
    rcs = manifest.get("random_control_stream", {})
    ctrl_seed = rcs.get("control_seed")
    if ctrl_seed != FROZEN_QUEUE_CONTROL_SEED:
        raise ConfirmationPreflightError(
            f"random_control_stream.control_seed must be {FROZEN_QUEUE_CONTROL_SEED}, got {ctrl_seed!r}"
        )

    validate_scientific_profile(manifest)
    declared_ineligible = set(seed_gov.get("ineligible_confirmation_seeds", INELIGIBLE_CONFIRMATION_SEEDS))
    if not INELIGIBLE_CONFIRMATION_SEEDS.issubset(declared_ineligible):
        raise ConfirmationPreflightError("Manifest omits ineligible seed roles")
    if seed in declared_ineligible:
        raise ConfirmationPreflightError("Seed belongs to the manifest's ineligible confirmation registry")

    return manifest


def validate_scientific_profile(manifest: dict[str, Any]) -> None:
    """Reject every nested change to the reviewed complete scientific profile."""
    try:
        actual = scientific_digest(manifest)
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfirmationPreflightError(f"Incomplete scientific profile: {exc}") from exc
    if actual != FROZEN_SCIENTIFIC_SHA256:
        raise ConfirmationPreflightError("Scientific profile differs from the reviewed frozen specification")


def verify_runtime_provenance(
    manifest: dict[str, Any],
    resource_locations: Mapping[str, str | Path] | None = None,
) -> dict[str, Path]:
    """Verify runtime resource presence and full SHA-256 digests against manifest."""
    provenance = manifest.get("provenance_hashes", {})
    locs = resource_locations or manifest.get("development_resource_locations", {})

    resolved_paths: dict[str, Path] = {}

    repo_root = Path(__file__).resolve().parents[3]
    default_runtime = repo_root / "runtime"

    def _resolve_default(filename: str, fallback: str) -> str:
        candidate = default_runtime / filename
        if candidate.exists():
            return str(candidate)
        candidate_res = default_runtime / "resources" / filename
        if candidate_res.exists():
            return str(candidate_res)
        return fallback

    default_bao = (
        str(default_runtime / "bao-simon-audit" / "simonNDvsDD" / "SENet")
        if (default_runtime / "bao-simon-audit" / "simonNDvsDD" / "SENet").is_dir()
        else "/private/tmp/bao-simon-audit/simonNDvsDD/SENet"
    )
    default_tf = (
        str(default_runtime / "simon_tf_venv" / "bin" / "python3.12")
        if (default_runtime / "simon_tf_venv" / "bin" / "python3.12").is_file()
        else "/private/tmp/simon_tf_venv/bin/python3.12"
    )

    expected_hashes = {
        "ddt_7r": (
            locs.get("ddt_7r", _resolve_default("simon_predicted_ddt_7r.bin", "/private/tmp/simon_predicted_ddt_7r.bin")),
            provenance.get("simon_predicted_ddt_7r_bin_sha256"),
        ),
        "queried_row_8r_binary": (
            locs.get("queried_row_8r_binary", _resolve_default("simon_queried_row_8r", "/private/tmp/simon_queried_row_8r")),
            provenance.get("simon_queried_row_8r_bin_sha256"),
        ),
        "queried_row_8r_cpp": (
            locs.get("queried_row_8r_cpp", repo_root / "tools" / "simon_queried_row_8r.cpp"),
            provenance.get("simon_queried_row_8r_cpp_sha256"),
        ),
        "b1_checkpoint": (
            locs.get("b1_checkpoint", repo_root / "models" / "simon" / "simon_b1_mean_speckhead.pt"),
            provenance.get("b1_checkpoint_sha256"),
        ),
    }

    # Unconditionally require Bao checkpoints
    bao_dir = Path(locs.get("bao_vv_directory", default_bao))
    expected_hashes["vv9_checkpoint"] = (
        bao_dir / "ND_VV_Simon32_9R.h5",
        provenance.get("vv9_checkpoint_sha256"),
    )
    expected_hashes["vv8_checkpoint"] = (
        bao_dir / "ND_VV_Simon32_8R.h5",
        provenance.get("vv8_checkpoint_sha256"),
    )

    for key, (path_val, exp_hash) in expected_hashes.items():
        p = Path(path_val).resolve()
        if not p.is_file():
            raise ProvenanceVerificationError(f"Required runtime resource missing: {p}")
        if exp_hash is None or len(exp_hash) != 64:
            raise ProvenanceVerificationError(f"Missing or invalid SHA-256 for {key} in manifest")
        actual_hash = file_sha256(p)
        if actual_hash.lower() != exp_hash.lower():
            raise ProvenanceVerificationError(
                f"SHA-256 mismatch for {key} ({p}): expected {exp_hash}, got {actual_hash}"
            )
        resolved_paths[key] = p

    resolved_paths["bao_vv_directory"] = bao_dir.resolve()

    tf_python = Path(locs.get("tensorflow_python", default_tf))
    if not tf_python.is_file():
        raise ProvenanceVerificationError(f"Required TensorFlow Python binary missing: {tf_python}")
    # Preserve the virtualenv invocation path: resolving a venv interpreter
    # symlink before invocation would select the base Python environment.
    resolved_paths["tensorflow_python"] = tf_python.absolute()

    # Verify python and pycryptosat
    import pycryptosat
    if pycryptosat.__version__ != "5.14.7":
        raise ProvenanceVerificationError(
            f"pycryptosat version mismatch: expected '5.14.7', got {pycryptosat.__version__!r}"
        )

    try:
        verify_execution_pins(manifest, tensorflow_python=resolved_paths["tensorflow_python"])
    except (ValueError, OSError) as exc:
        raise ProvenanceVerificationError(str(exc)) from exc

    return resolved_paths


ALLOWED_FINAL_OUTCOMES: Final[frozenset[str]] = frozenset(
    {
        "verified_recovery",
        "proven_unsat",
        "candidate_unresolved",
        "queue_exhausted",
        "global_deadline",
        "ranking_timeout",
        "solver_timeout",
        "solver_unsat_or_timeout",
        "verification_failure",
        "sat_unverified",
        "exception",
    }
)


def check_key_overlaps_with_prior_families(
    keys: Sequence[tuple[int, int, int, int]],
) -> dict[str, list[tuple[int, int, int, int]]]:
    """Check key sequence against all recorded prior SIMON demo, development, and training families."""
    prior_draws: dict[str, set[tuple[int, int, int, int]]] = {}

    # 1. Opened demo key draws: seed 999999, 8 keys at 4, 8, 32, 64 pairs
    for p_cnt in (4, 8, 32, 64):
        cfg = SimonAttackInstanceConfig(instance_count=8, pair_count=p_cnt, rounds=10, seed=999999, split_role="demo")
        prior_draws[f"demo_seed_999999_p{p_cnt}"] = {inst.master_key for inst in generate_simon_attack_instances(cfg)}

    # 2. Development cohorts: seed 260926 (R=10, 32 keys across 32, 64 pairs)
    for p_cnt in (32, 64):
        cfg = SimonAttackInstanceConfig(instance_count=32, pair_count=p_cnt, rounds=10, seed=260926, split_role="development")
        prior_draws[f"development_seed_260926_p{p_cnt}"] = {inst.master_key for inst in generate_simon_attack_instances(cfg)}

    # 3. Actual initial NumPy uint16 key draws used by B0/B1 training/validation:
    for seed, key_count in ((880001, 1_000_000), (880002, 99_968)):
        prior_draws[f"synthetic_training_seed_{seed}"] = _training_key_overlaps(
            keys, seed=seed, key_count=key_count,
        )

    # 4. Confirmation Attempt 1: seed 8398726052707596859 (R=10, 64 keys, 64 pairs)
    cfg_att1 = SimonAttackInstanceConfig(instance_count=64, pair_count=64, rounds=10, seed=8398726052707596859, split_role="confirmation")
    prior_draws["confirmation_attempt1_seed_8398726052707596859"] = {inst.master_key for inst in generate_simon_attack_instances(cfg_att1)}

    key_set = set(keys)
    intersections: dict[str, list[tuple[int, int, int, int]]] = {}
    for role_name, prior_keys in prior_draws.items():
        overlap = key_set.intersection(prior_keys)
        if overlap:
            intersections[role_name] = sorted(list(overlap))

    return intersections


def build_disjoint_family_certificate(
    manifest: dict[str, Any],
    instances: Sequence[SimonAttackInstance],
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Reconstruct key draws across prior datasets and verify zero overlap with confirmation."""
    selected_n = manifest["sample_size_specification"]["selected_instances"]
    if len(instances) != selected_n:
        raise DisjointFamilyCertificateError(
            f"Instance count {len(instances)} does not match selected_instances {selected_n}"
        )

    # Check instance uniqueness and role
    confirmation_keys = [inst.master_key for inst in instances]
    if len(set(confirmation_keys)) != len(confirmation_keys):
        raise DisjointFamilyCertificateError("Confirmation cohort contains duplicate master keys")

    for inst in instances:
        if inst.split_role != "confirmation":
            raise DisjointFamilyCertificateError(
                f"Instance {inst.instance_id} has split_role {inst.split_role!r}; must be 'confirmation'"
            )
        if inst.rounds != 10:
            raise DisjointFamilyCertificateError(f"Instance {inst.instance_id} has rounds={inst.rounds}; expected 10")
        if len(inst.pairs) != 64:
            raise DisjointFamilyCertificateError(f"Instance {inst.instance_id} has {len(inst.pairs)} pairs; expected 64")
        if len(inst.verification_pairs) != 2:
            raise DisjointFamilyCertificateError(f"Instance {inst.instance_id} has {len(inst.verification_pairs)} verif pairs; expected 2")

    intersections = check_key_overlaps_with_prior_families(confirmation_keys)
    if intersections:
        raise DisjointFamilyCertificateError(
            f"Disjoint family violation! Overlapping keys found: {intersections}"
        )

    expected_roles = [
        "demo_seed_999999_p4",
        "demo_seed_999999_p8",
        "demo_seed_999999_p32",
        "demo_seed_999999_p64",
        "development_seed_260926_p32",
        "development_seed_260926_p64",
        "synthetic_training_seed_880001",
        "synthetic_training_seed_880002",
        "confirmation_attempt1_seed_8398726052707596859",
    ]

    # Compute dataset hash
    dataset_repr = json.dumps([inst.to_dict() for inst in instances], sort_keys=True)
    dataset_sha256 = hashlib.sha256(dataset_repr.encode("utf-8")).hexdigest()
    manifest_sha256 = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()

    cert = {
        "certificate_version": "1.0.0",
        "protocol_id": manifest["protocol_id"],
        "confirmation_seed": manifest["seed_governance"]["seed"],
        "selected_instances": selected_n,
        "instance_count": len(instances),
        "dataset_sha256": dataset_sha256,
        "manifest_sha256": manifest_sha256,
        "prior_roles_checked": expected_roles,
        "intersection_count": 0,
        "disjoint_family_verified": True,
    }

    if output_path is not None:
        p = Path(output_path)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cert, f, indent=2)

    return cert


def _training_key_overlaps(
    keys: Sequence[tuple[int, int, int, int]], *, seed: int, key_count: int,
) -> set[tuple[int, int, int, int]]:
    """Check all initial draws from b1_data.py against an already opened cohort."""
    drawn = np.random.default_rng(seed).integers(
        0, 65536, size=(key_count, 4), dtype=np.uint16,
    )
    packed = drawn[:, 0].astype(np.uint64)
    for column in range(1, 4):
        packed = (packed << 16) | drawn[:, column]
    requested = np.array([
        (key[0] << 48) | (key[1] << 32) | (key[2] << 16) | key[3]
        for key in keys
    ], dtype=np.uint64)
    matches = drawn[np.isin(packed, requested)]
    return {tuple(int(word) for word in row) for row in matches}


def verify_disjoint_family_certificate(
    manifest: dict[str, Any],
    instances: Sequence[SimonAttackInstance],
    certificate_raw: dict[str, Any] | str | Path,
) -> None:
    """Verify that the disjoint-family certificate is valid, complete, and matches data."""
    if isinstance(certificate_raw, (str, Path)):
        p = Path(certificate_raw)
        if not p.is_file():
            raise DisjointFamilyCertificateError(f"Certificate file missing: {p}")
        with open(p, encoding="utf-8") as f:
            cert = json.load(f)
    elif isinstance(certificate_raw, dict):
        cert = certificate_raw
    else:
        raise TypeError("certificate_raw must be a dict, str, or Path")

    if cert.get("disjoint_family_verified") is not True or cert.get("intersection_count") != 0:
        raise DisjointFamilyCertificateError("Certificate does not certify disjoint family verification")

    if cert.get("confirmation_seed") != manifest["seed_governance"]["seed"]:
        raise DisjointFamilyCertificateError("Certificate confirmation_seed does not match manifest")

    if cert.get("selected_instances") != manifest["sample_size_specification"]["selected_instances"]:
        raise DisjointFamilyCertificateError("Certificate selected_instances does not match manifest")

    if cert.get("instance_count") != len(instances):
        raise DisjointFamilyCertificateError("Certificate instance_count does not match instances count")

    # Check collisions before assignment replay for an explicit leakage diagnosis.
    expected_certificate = build_disjoint_family_certificate(manifest, instances)
    validate_confirmation_assignment(manifest, instances)
    if cert != expected_certificate:
        raise DisjointFamilyCertificateError("Certificate contents differ from recomputed complete certificate")

    dataset_repr = json.dumps([inst.to_dict() for inst in instances], sort_keys=True)
    actual_dataset_sha256 = hashlib.sha256(dataset_repr.encode("utf-8")).hexdigest()
    if cert.get("dataset_sha256") != actual_dataset_sha256:
        raise DisjointFamilyCertificateError("Certificate dataset_sha256 does not match instances data")

    manifest_repr = json.dumps(manifest, sort_keys=True)
    actual_manifest_sha256 = hashlib.sha256(manifest_repr.encode("utf-8")).hexdigest()
    if cert.get("manifest_sha256") != actual_manifest_sha256:
        raise DisjointFamilyCertificateError("Certificate manifest_sha256 does not match manifest data")

    # Repeat independent verification of key isolation against all prior families
    conf_keys = [inst.master_key for inst in instances]
    intersections = check_key_overlaps_with_prior_families(conf_keys)
    if intersections:
        raise DisjointFamilyCertificateError(
            f"Disjoint family violation! Overlapping keys found during certificate verification: {intersections}"
        )


def compute_execution_identity(manifest: dict[str, Any], dataset_sha256: str) -> dict[str, Any]:
    """Compute binding execution identity tuple for confirmation and resume."""
    manifest_repr = json.dumps(manifest, sort_keys=True)
    manifest_sha256 = hashlib.sha256(manifest_repr.encode("utf-8")).hexdigest()

    return {
        "manifest_sha256": manifest_sha256,
        "dataset_sha256": dataset_sha256,
        "seed": manifest["seed_governance"]["seed"],
        "selected_instances": manifest["sample_size_specification"]["selected_instances"],
        "comparator_methods": list(REQUIRED_COMPARATOR_METHODS),
        "diagnostic_oracle": DIAGNOSTIC_ORACLE_METHOD,
        "control_seed": manifest["random_control_stream"]["control_seed"],
        "global_deadline_seconds": manifest["resource_limits"]["global_deadline_seconds"],
        "candidate_cutoff_soft_seconds": manifest["resource_limits"]["candidate_cutoff_soft_seconds"],
        "ranking_time_limit_seconds": manifest["resource_limits"]["ranking_time_limit_seconds"],
        "provenance_hashes": manifest.get("provenance_hashes", {}),
        "execution_pins": manifest.get("execution_pins", {}),
    }


def validate_and_filter_resume(
    ledger_path: Path | str,
    expected_identity: dict[str, Any],
    instances: Sequence[SimonAttackInstance],
    methods: Sequence[str],
    *,
    resume_flag: bool = True,
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[tuple[SimonAttackInstance, str]]]:
    """Validate existing ledger against expected identity and return pending cells to evaluate."""
    p = Path(ledger_path)
    if not p.is_file():
        # Clean start: all cells pending
        pending: list[tuple[SimonAttackInstance, str]] = []
        for inst in instances:
            for m in methods:
                pending.append((inst, m))
        return {}, pending

    # 1. Output exists: require explicit resume_flag
    if not resume_flag:
        raise ResumeIntegrityError(
            f"Output file already exists at {p}, but --resume flag was not provided. Aborting to prevent accidental overwrite."
        )

    # 2. Checksum sidecar validation
    sha_path = p.parent / f"{p.name}.sha256"
    if not sha_path.is_file():
        raise ResumeIntegrityError(f"Checksum sidecar missing for resume ledger: {sha_path}")
    sidecar_text = sha_path.read_text(encoding="utf-8").strip()
    expected_sha = sidecar_text.split()[0] if sidecar_text else ""
    actual_sha = file_sha256(p)
    if actual_sha.lower() != expected_sha.lower():
        raise ResumeIntegrityError(
            f"Checksum sidecar mismatch for {p}: expected {expected_sha}, got {actual_sha}"
        )

    # 3. Read and parse ledger JSON
    try:
        with open(p, encoding="utf-8") as f:
            ledger = json.load(f)
    except Exception as exc:
        raise ResumeIntegrityError(f"Corrupted or invalid JSON in resume ledger: {exc}") from exc

    ledger_identity = ledger.get("execution_identity")
    if ledger_identity != expected_identity:
        raise ResumeIntegrityError(
            "Ledger execution_identity does not match expected execution identity for resume. "
            f"Expected: {expected_identity}; Found: {ledger_identity}"
        )

    records = ledger.get("records", [])
    if ledger.get("evidence_class") != "CONFIRMATION" or ledger.get("completed_count") != len(records):
        raise ResumeIntegrityError("Ledger evidence class or completed_count is inconsistent")
    completed_cells: dict[tuple[str, str], dict[str, Any]] = {}
    instance_map = {inst.instance_id: (idx, inst) for idx, inst in enumerate(instances)}

    for rec in records:
        inst_id = rec.get("instance_id")
        method = rec.get("method")
        if not inst_id or not method or not isinstance(inst_id, str) or not isinstance(method, str):
            raise ResumeIntegrityError(f"Malformed record missing instance_id or method: {rec}")
        cell = (inst_id, method)
        if cell in completed_cells:
            raise ResumeIntegrityError(f"Duplicate record found in resume ledger for cell {cell}")
        if inst_id not in instance_map or method not in methods:
            raise ResumeIntegrityError(f"Record for cell {cell} does not belong to expected confirmation grid")

        inst_idx, inst = instance_map[inst_id]
        if rec.get("instance_index") != inst_idx:
            raise ResumeIntegrityError(
                f"Record {cell} has mismatched instance_index: expected {inst_idx}, got {rec.get('instance_index')}"
            )

        outcome = rec.get("outcome")
        if outcome not in ALLOWED_FINAL_OUTCOMES:
            raise ResumeIntegrityError(f"Record {cell} has unknown or invalid outcome: {outcome!r}")

        # Check finite nonnegative durations
        for time_key in (
            "complete_online_wall_seconds",
            "ranking_wall_seconds",
            "solving_wall_seconds",
            "verification_wall_seconds",
        ):
            val = rec.get(time_key)
            if val is None and time_key != "complete_online_wall_seconds" and outcome == "exception" and rec.get("telemetry_status") == "UNAVAILABLE_OUTER_WORKER_EXCEPTION":
                continue
            if type(val) not in (int, float) or not math.isfinite(val) or val < 0.0:
                raise ResumeIntegrityError(f"Record {cell} has invalid {time_key}: {val!r}")

        verified = rec.get("verified_on_fresh_pairs")
        if type(verified) is not bool or verified != (outcome == "verified_recovery"):
            raise ResumeIntegrityError(f"Record {cell} has missing or contradictory fresh-verification flag")
        if "counts_as_verified_recovery" in rec and rec["counts_as_verified_recovery"] is not verified:
            raise ResumeIntegrityError(f"Record {cell} has contradictory recovery-count flag")
        if outcome != "verified_recovery" and rec.get("recovered_master_key") is not None:
            raise ResumeIntegrityError(f"Failure record {cell} contains a claimed recovered key")
        for field in ("candidates_tested_count", "candidates_remaining_count"):
            if type(rec.get(field)) is not int or rec[field] < 0:
                raise ResumeIntegrityError(f"Record {cell} has invalid {field}")

        # Independent fresh verification check on claimed successes
        if outcome == "verified_recovery":
            key_raw = rec.get("recovered_master_key")
            if not isinstance(key_raw, (list, tuple)) or len(key_raw) != 4:
                raise ResumeIntegrityError(f"Claimed success missing recovered_master_key: {cell}")
            if any(type(word) is not int or not 0 <= word <= 65535 for word in key_raw):
                raise ResumeIntegrityError(f"Claimed success has malformed recovered key: {cell}")
            key = tuple(key_raw)
            if key != inst.master_key:
                raise ResumeIntegrityError(
                    f"Claimed success for {cell} recovered wrong key: expected {inst.master_key}, got {key}"
                )
            v_check = verify_simon_master_key(
                master_key=key,
                known_pairs=inst.known_pairs,
                verification_pairs=inst.verification_pairs[:2],
                rounds=inst.rounds,
            )
            if not v_check.verified or not verify_independent_fresh(key, inst):
                raise ResumeIntegrityError(f"Claimed success for {cell} failed independent verification on resume")

            if rec["complete_online_wall_seconds"] <= 0.0 or rec["complete_online_wall_seconds"] > 300.0:
                raise ResumeIntegrityError(f"Claimed success for {cell} lies outside the common deadline")

        completed_cells[cell] = rec

    # Determine pending cells
    pending_cells: list[tuple[SimonAttackInstance, str]] = []
    for inst in instances:
        for m in methods:
            if (inst.instance_id, m) not in completed_cells:
                pending_cells.append((inst, m))

    return completed_cells, pending_cells


def generate_confirmation_dataset(
    manifest_raw: dict[str, Any] | str | Path,
    output_dataset_path: Path | str,
    output_cert_path: Path | str,
    *,
    authorization_token: bool = False,
) -> tuple[Path, Path]:
    """Generate confirmation attack instances and certificate from approved manifest.

    Fails closed unless authorization_token is True AND confirmation_dataset_generation_authorized is True.
    """
    if not authorization_token:
        raise ConfirmationPreflightError(
            "Confirmation dataset generation requires explicit authorization token (authorization_token=True)"
        )

    if isinstance(manifest_raw, (str, Path)):
        p = Path(manifest_raw)
        if not p.is_file():
            raise ConfirmationPreflightError(f"Manifest file does not exist: {p}")
        with open(p, encoding="utf-8") as f:
            manifest = json.load(f)
    elif isinstance(manifest_raw, dict):
        manifest = manifest_raw
    else:
        raise TypeError("manifest_raw must be a dict, str, or Path")

    if manifest.get("confirmation_dataset_generation_authorized") is not True:
        raise ConfirmationPreflightError(
            "confirmation_dataset_generation_authorized must be explicitly True in manifest"
        )

    validate_confirmation_manifest(manifest)
    verify_runtime_provenance(manifest)
    for target in (Path(output_dataset_path), Path(output_cert_path)):
        if target.exists() or target.with_name(target.name + ".sha256").exists():
            raise ConfirmationPreflightError(f"Refusing to overwrite an existing confirmation artifact: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)

    # Validate seed governance
    seed = manifest.get("seed_governance", {}).get("seed")
    if seed is None or isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfirmationPreflightError(f"seed_governance.seed must be an integer, got {seed!r}")
    if seed in INELIGIBLE_CONFIRMATION_SEEDS:
        raise ConfirmationPreflightError(
            f"Seed {seed} belongs to INELIGIBLE_CONFIRMATION_SEEDS and cannot be used for confirmation"
        )

    sample_spec = manifest.get("sample_size_specification", {})
    selected_n = sample_spec.get("selected_instances")
    if selected_n not in (32, 64):
        raise ConfirmationPreflightError(f"selected_instances must be 32 or 64, got {selected_n!r}")

    pair_count = sample_spec.get("pair_count", 64)
    if pair_count != 64:
        raise ConfirmationPreflightError(f"pair_count must be 64, got {pair_count!r}")

    rounds = manifest.get("target_cipher", {}).get("rounds", 10)
    if rounds != 10:
        raise ConfirmationPreflightError(f"rounds must be 10, got {rounds!r}")

    input_diff = tuple(manifest.get("target_cipher", {}).get("input_difference", ()))
    if input_diff != DEFAULT_SIMON_INPUT_DIFFERENCE:
        raise ConfirmationPreflightError(f"input_difference must be {DEFAULT_SIMON_INPUT_DIFFERENCE}")

    cfg = SimonAttackInstanceConfig(
        instance_count=selected_n,
        pair_count=pair_count,
        rounds=rounds,
        seed=seed,
        split_role="confirmation",
    )
    instances = generate_simon_attack_instances(cfg)

    # Validate generated instances
    if len(instances) != selected_n:
        raise ConfirmationPreflightError(f"Generated {len(instances)} instances, expected {selected_n}")

    unique_keys = {inst.master_key for inst in instances}
    if len(unique_keys) != selected_n:
        raise ConfirmationPreflightError("Generated instances contain duplicate master keys")

    unique_ids = {inst.instance_id for inst in instances}
    if len(unique_ids) != selected_n:
        raise ConfirmationPreflightError("Generated instances contain duplicate instance IDs")

    for inst in instances:
        if inst.split_role != "confirmation":
            raise ConfirmationPreflightError(f"Instance {inst.instance_id} has invalid role {inst.split_role!r}")
        if len(inst.pairs) != 64:
            raise ConfirmationPreflightError(f"Instance {inst.instance_id} has {len(inst.pairs)} pairs; expected 64")
        if len(inst.verification_pairs) != 2:
            raise ConfirmationPreflightError(f"Instance {inst.instance_id} has {len(inst.verification_pairs)} verification pairs; expected 2")
        for p in inst.pairs:
            diff = (p.plaintext0[0] ^ p.plaintext1[0], p.plaintext0[1] ^ p.plaintext1[1])
            if diff != DEFAULT_SIMON_INPUT_DIFFERENCE:
                raise ConfirmationPreflightError(f"Instance {inst.instance_id} pair has wrong difference: {diff}")
        attack_pts = {p.plaintext0 for p in inst.pairs} | {p.plaintext1 for p in inst.pairs}
        verif_pts = {pt for pt, _ct in inst.verification_pairs}
        if attack_pts.intersection(verif_pts):
            raise ConfirmationPreflightError(f"Instance {inst.instance_id} has overlap between attack and verification plaintexts")

    # Build disjoint-family certificate (checks all prior families)
    out_cert = Path(output_cert_path)
    build_disjoint_family_certificate(manifest, instances, output_path=out_cert)

    # Write dataset atomically with .sha256 sidecar
    out_ds = Path(output_dataset_path)
    tmp_ds = out_ds.with_suffix(".tmp")
    ds_data = [inst.to_dict() for inst in instances]
    raw_ds = json.dumps(ds_data, indent=2)
    with open(tmp_ds, "w", encoding="utf-8") as f:
        f.write(raw_ds)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_ds, out_ds)

    ds_sha = hashlib.sha256(raw_ds.encode()).hexdigest()
    sha_file = out_ds.parent / f"{out_ds.name}.sha256"
    tmp_sha = out_ds.parent / f"{out_ds.name}.sha256.tmp"
    with open(tmp_sha, "w", encoding="utf-8") as f:
        f.write(f"{ds_sha}  {out_ds.name}\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_sha, sha_file)

    return out_ds, out_cert


def validate_confirmation_assignment(manifest: dict, instances: Sequence[SimonAttackInstance]) -> None:
    """Bind a unique complete cohort to the registered deterministic assignment."""
    n = manifest["sample_size_specification"]["selected_instances"]
    if len(instances) != n:
        raise DisjointFamilyCertificateError("Actual cohort size differs from the frozen N")
    if len({inst.instance_id for inst in instances}) != n or len({inst.master_key for inst in instances}) != n:
        raise DisjointFamilyCertificateError("Confirmation assignment contains duplicate IDs or keys")
    expected = generate_simon_attack_instances(SimonAttackInstanceConfig(
        instance_count=n, pair_count=64, rounds=10,
        seed=manifest["seed_governance"]["seed"], split_role="confirmation",
    ))
    if [inst.to_dict() for inst in instances] != [inst.to_dict() for inst in expected]:
        raise DisjointFamilyCertificateError("Dataset differs from the registered seed/configuration/order")
    for inst in instances:
        SimonComparisonInput.from_attack_instance(inst, manifest_seed=manifest["seed_governance"]["seed"], pair_count=64)


def verify_independent_fresh(key: tuple[int, int, int, int], instance: SimonAttackInstance) -> bool:
    """Clean-room cipher check, separate from the SAT and production cipher paths."""
    import importlib.util
    reference = Path(__file__).resolve().parents[3] / "tests/fixtures/simon_reference_oracle.py"
    spec = importlib.util.spec_from_file_location("simon_independent_reference", reference)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return all(
        module.oracle_simon32_64_encrypt(pt, key, rounds=instance.rounds) == ct
        for pt, ct in instance.verification_pairs
    )
