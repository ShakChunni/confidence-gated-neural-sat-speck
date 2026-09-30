"""Manifest-enforced execution profile and resume guards for SIMON confirmation.

Enforces strict preflight validation, seed governance, disjoint-family certificates,
exact provenance hashing, and execution-identity-bound resume for the prospective
SIMON32/64 R=10 confirmation evaluation.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from neural_sat_speck.simon.attack_instance import (
    DEFAULT_SIMON_INPUT_DIFFERENCE,
    SimonAttackInstance,
    SimonAttackInstanceConfig,
    generate_simon_attack_instances,
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


def derive_confirmation_queue_seed(control_seed: int, instance_id: str, method: str) -> int:
    """Deterministically derive a 64-bit random control seed from frozen queue control seed."""
    key = f"simon_confirmation_control:{control_seed}:{instance_id}:{method}".encode()
    h = hashlib.sha256(key).digest()
    return int.from_bytes(h[:8], "big")


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
    if manifest.get("manifest_version") != "1.1.0":
        raise ConfirmationPreflightError(
            f"Unsupported manifest_version: {manifest.get('manifest_version')!r}; expected '1.1.0'"
        )
    if manifest.get("protocol_id") != "simon32_64_r10_unseen_confirmation_v1":
        raise ConfirmationPreflightError(f"Invalid protocol_id: {manifest.get('protocol_id')!r}")
    if manifest.get("evidence_class") != "CONFIRMATION":
        raise ConfirmationPreflightError(f"Invalid evidence_class: {manifest.get('evidence_class')!r}")
    if manifest.get("status") != "APPROVED":
        raise ConfirmationPreflightError(
            f"Manifest status is {manifest.get('status')!r}; must be 'APPROVED' for confirmation"
        )
    if manifest.get("confirmation_execution_authorized") is not True:
        raise ConfirmationPreflightError("confirmation_execution_authorized must be explicitly True")

    # 2. Seed Governance
    seed_gov = manifest.get("seed_governance", {})
    seed = seed_gov.get("seed")
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
    if selected_n not in (32, 64):
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

    return manifest


def verify_runtime_provenance(
    manifest: dict[str, Any],
    resource_locations: Mapping[str, str | Path] | None = None,
) -> dict[str, Path]:
    """Verify runtime resource presence and full SHA-256 digests against manifest."""
    provenance = manifest.get("provenance_hashes", {})
    locs = resource_locations or manifest.get("development_resource_locations", {})

    resolved_paths: dict[str, Path] = {}

    expected_hashes = {
        "ddt_7r": (
            locs.get("ddt_7r", "/private/tmp/simon_predicted_ddt_7r.bin"),
            provenance.get("simon_predicted_ddt_7r_bin_sha256"),
        ),
        "queried_row_8r_binary": (
            locs.get("queried_row_8r_binary", "/private/tmp/simon_queried_row_8r"),
            provenance.get("simon_queried_row_8r_bin_sha256"),
        ),
        "queried_row_8r_cpp": (
            Path(__file__).resolve().parent.parent.parent.parent / "tools" / "simon_queried_row_8r.cpp",
            provenance.get("simon_queried_row_8r_cpp_sha256"),
        ),
        "b1_checkpoint": (
            Path(__file__).resolve().parent.parent.parent.parent / "models" / "simon" / "simon_b1_mean_speckhead.pt",
            provenance.get("b1_checkpoint_sha256"),
        ),
    }

    # Also Bao checkpoints if directory provided
    bao_dir = Path(locs.get("bao_vv_directory", "/private/tmp/bao-simon-audit/simonNDvsDD/SENet"))
    if bao_dir.is_dir():
        expected_hashes["vv9_checkpoint"] = (
            bao_dir / "ND_VV_Simon32_9R.h5",
            provenance.get("vv9_checkpoint_sha256"),
        )
        expected_hashes["vv8_checkpoint"] = (
            bao_dir / "ND_VV_Simon32_8R.h5",
            provenance.get("vv8_checkpoint_sha256"),
        )

    for key, (path_val, exp_hash) in expected_hashes.items():
        p = Path(path_val)
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

    # Verify python and pycryptosat
    import pycryptosat
    if pycryptosat.__version__ != "5.14.7":
        raise ProvenanceVerificationError(
            f"pycryptosat version mismatch: expected '5.14.7', got {pycryptosat.__version__!r}"
        )

    return resolved_paths


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

    # Reconstruct initial key draws from prior datasets
    prior_draws: dict[str, set[tuple[int, int, int, int]]] = {}

    # 1. Demo cohort: seed 999999 (R=10, 8 keys)
    demo_cfg = SimonAttackInstanceConfig(instance_count=8, pair_count=32, rounds=10, seed=999999, split_role="demo")
    demo_instances = generate_simon_attack_instances(demo_cfg)
    prior_draws["demo_seed_999999"] = {inst.master_key for inst in demo_instances}

    # 2. Development cohort: seed 260926 (R=10, 32 keys)
    dev_cfg = SimonAttackInstanceConfig(instance_count=32, pair_count=64, rounds=10, seed=260926, split_role="development")
    dev_instances = generate_simon_attack_instances(dev_cfg)
    prior_draws["development_seed_260926"] = {inst.master_key for inst in dev_instances}

    # 3. B0 / B1 training seeds: 880001, 880002
    for s in (880001, 880002):
        rng = random.Random(s)
        # sample 50,000 keys drawn by synthetic generator
        keys = {(rng.randrange(1 << 16), rng.randrange(1 << 16), rng.randrange(1 << 16), rng.randrange(1 << 16)) for _ in range(50000)}
        prior_draws[f"synthetic_training_seed_{s}"] = keys

    conf_key_set = set(confirmation_keys)
    intersections: dict[str, list[tuple[int, int, int, int]]] = {}
    for role_name, prior_keys in prior_draws.items():
        overlap = conf_key_set.intersection(prior_keys)
        if overlap:
            intersections[role_name] = sorted(list(overlap))

    if intersections:
        raise DisjointFamilyCertificateError(
            f"Disjoint family violation! Overlapping keys found: {intersections}"
        )

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
        "prior_roles_checked": list(prior_draws.keys()),
        "intersection_count": 0,
        "disjoint_family_verified": True,
    }

    if output_path is not None:
        p = Path(output_path)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cert, f, indent=2)

    return cert


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

    dataset_repr = json.dumps([inst.to_dict() for inst in instances], sort_keys=True)
    actual_dataset_sha256 = hashlib.sha256(dataset_repr.encode("utf-8")).hexdigest()
    if cert.get("dataset_sha256") != actual_dataset_sha256:
        raise DisjointFamilyCertificateError("Certificate dataset_sha256 does not match instances data")


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
    }


def validate_and_filter_resume(
    ledger_path: Path | str,
    expected_identity: dict[str, Any],
    instances: Sequence[SimonAttackInstance],
    methods: Sequence[str],
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

    with open(p, encoding="utf-8") as f:
        ledger = json.load(f)

    ledger_identity = ledger.get("execution_identity")
    if ledger_identity != expected_identity:
        raise ResumeIntegrityError(
            "Ledger execution_identity does not match expected execution identity for resume. "
            f"Expected: {expected_identity}; Found: {ledger_identity}"
        )

    records = ledger.get("records", [])
    completed_cells: dict[tuple[str, str], dict[str, Any]] = {}
    instance_map = {inst.instance_id: inst for inst in instances}

    for rec in records:
        inst_id = rec.get("instance_id")
        method = rec.get("method")
        if not inst_id or not method:
            raise ResumeIntegrityError(f"Malformed record missing instance_id or method: {rec}")
        cell = (inst_id, method)
        if cell in completed_cells:
            raise ResumeIntegrityError(f"Duplicate record found in resume ledger for cell {cell}")
        if inst_id not in instance_map or method not in methods:
            raise ResumeIntegrityError(f"Record for cell {cell} does not belong to expected confirmation grid")

        # Independent fresh verification check on claimed successes
        if rec.get("outcome") == "verified_recovery":
            key_raw = rec.get("recovered_master_key")
            if not key_raw:
                raise ResumeIntegrityError(f"Claimed success missing recovered_master_key: {cell}")
            key = (int(key_raw[0]), int(key_raw[1]), int(key_raw[2]), int(key_raw[3]))
            inst = instance_map[inst_id]
            v_check = verify_simon_master_key(
                master_key=key,
                known_pairs=inst.known_pairs,
                verification_pairs=inst.verification_pairs[:2],
                rounds=inst.rounds,
            )
            if not v_check.verified:
                raise ResumeIntegrityError(f"Claimed success for {cell} failed independent verification on resume")

        completed_cells[cell] = rec

    # Determine pending cells
    pending_cells: list[tuple[SimonAttackInstance, str]] = []
    for inst in instances:
        for m in methods:
            if (inst.instance_id, m) not in completed_cells:
                pending_cells.append((inst, m))

    return completed_cells, pending_cells
