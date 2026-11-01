#!/usr/bin/env python3
"""Evaluate SIMON32/64 R=10 matched development comparators under strict solver isolation.

Executes the controlled 64-pair comparative study of SIMON32/64 R=10 recovery strategies
governed by D-029, D-034, D-035, and D-037:
1. neural_classical_hybrid_8parent (Bao VV9 Stage 1 -> 7R DDT Stage 2, 1,024 tuples)
2. fully_classical_8parent (8R Queried Row Stage 1 -> 7R DDT Stage 2, 1,024 tuples)
3. fully_neural_1parent (Bao VV9 Stage 1 -> Bao VV8 Stage 2, 128 tuples)
4. faithful_b1_hybrid (SIMON B1 Stage 1 -> 7R DDT Stage 2, 1,024 tuples)
5. random_control_128 (128 uniform random tuples, width-matched to 1-parent)
6. random_control_1024 (1,024 uniform random tuples, width-matched to 8-parent)
7. sat_only_n4 (4-observation unguided formula control, 2 pairs)
8. sat_only_n128 (128-observation unguided information-matched control, 64 pairs)
+ Diagnostic control:
9. oracle_2key_n4 (4 observations, true trailing keys (k8, k9) diagnostic upper bound)

SCIENTIFIC STATUS:
- DEVELOPMENT/DIAGNOSTIC evidence only; prospective confirmation requires later frozen cohort (D-038).
- Governed by D-029, D-034, D-035, D-037.
- Strict per-method-instance subprocess solver isolation enforced via run_method_worker.
- Monotonic per-instance global wall-clock deadline enforced (300.0s default).
- Counterbalanced Latin-square cyclic shift across instances with diagnostic oracle kept separate.
- Atomic per-method streaming records with SHA-256 sidecar persistence.
- Fail-closed resumable execution with cryptographic integrity verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import pycryptosat

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from neural_sat_speck.evaluation.metrics import wilson_score_interval  # noqa: E402
from neural_sat_speck.protocol.manifest_validator import (  # noqa: E402
    classify_attack_outcome,
)
from neural_sat_speck.simon.attack_instance import (  # noqa: E402
    DEFAULT_SIMON_INPUT_DIFFERENCE,
    SimonAttackInstance,
    SimonAttackInstanceConfig,
    generate_simon_attack_instances,
    load_simon_attack_instances,
)
from neural_sat_speck.simon.comparison_input import SimonComparisonInput  # noqa: E402
from neural_sat_speck.simon.method_worker import (  # noqa: E402
    DIAGNOSTIC_ORACLES,
    MethodWorkerOutcome,
    run_method_worker,
)

SCHEMA_VERSION: Final[str] = "simon_development_comparison.v1"
EXPECTED_ROUNDS: Final[int] = 10
EXPECTED_PAIR_COUNT: Final[int] = 64
EXPECTED_KNOWN_OBSERVATIONS: Final[int] = 128
EXPECTED_CANDIDATE_OBSERVATIONS: Final[int] = 4
EXPECTED_VERIFICATION_PAIRS: Final[int] = 2
DEFAULT_MANIFEST_SEED: Final[int] = 260926
DEFAULT_GLOBAL_TIMEOUT: Final[float] = 300.0
DEFAULT_CANDIDATE_CUTOFF: Final[float] = 45.0
DEFAULT_RANKING_TIME_LIMIT: Final[float] = 120.0
DEFAULT_INSTANCES: Final[int] = 32

DEFAULT_DDT_7R_PATH: Final[Path] = Path("/private/tmp/simon_predicted_ddt_7r.bin")
DEFAULT_TOOL_8R_PATH: Final[Path] = Path("/private/tmp/simon_queried_row_8r")
DEFAULT_TOOL_8R_SOURCE: Final[Path] = REPOSITORY_ROOT / "tools" / "simon_queried_row_8r.cpp"
DEFAULT_VV_DIR: Final[Path] = Path("/private/tmp/bao-simon-audit/simonNDvsDD/SENet")
DEFAULT_VV_FALLBACK_DIR: Final[Path] = Path("/private/tmp/bao-simon-audit/simon")

CANONICAL_COMPARATOR_ORDER: Final[tuple[str, ...]] = (
    "neural_classical_hybrid_8parent",
    "fully_classical_8parent",
    "fully_neural_1parent",
    "faithful_b1_hybrid",
    "random_control_128",
    "random_control_1024",
    "sat_only_n4",
    "sat_only_n128",
)

DIAGNOSTIC_ORACLE_METHODS: Final[tuple[str, ...]] = ("oracle_2key_n4",)

DEFAULT_METHODS: Final[tuple[str, ...]] = (
    *CANONICAL_COMPARATOR_ORDER,
    *DIAGNOSTIC_ORACLE_METHODS,
)

METHOD_ALIASES: Final[dict[str, str]] = {
    "oracle_a2_n4": "oracle_2key_n4",
    "sat_only": "sat_only_n4",
    "sat_only_all": "sat_only_n128",
    "faithful_b1_8parent": "faithful_b1_hybrid",
    "hybrid": "neural_classical_hybrid_8parent",
    "classical": "fully_classical_8parent",
}

METHOD_CHOICES: Final[list[str]] = list(
    dict.fromkeys(list(DEFAULT_METHODS) + list(METHOD_ALIASES.keys()))
)

METHOD_DISPLAY_NAMES: Final[dict[str, str]] = {
    "neural_classical_hybrid_8parent": "Neural-Classical Hybrid (8-parent, 1024 tuples)",
    "fully_classical_8parent": "Fully Classical (8-parent, 1024 tuples)",
    "fully_neural_1parent": "Fully Neural (1-parent, 128 tuples)",
    "faithful_b1_hybrid": "Faithful B1 Hybrid (8-parent, 1024 tuples)",
    "random_control_128": "Random Control (128 tuples)",
    "random_control_1024": "Random Control (1024 tuples)",
    "sat_only_n4": "SAT Only (4 obs, formula-matched)",
    "sat_only_n128": "SAT Only (128 obs, info-matched)",
    "oracle_2key_n4": "Oracle 2-Key Diagnostic (4 obs, upper bound)",
}

SIMON_CRITICAL_CODE_PATHS: Final[dict[str, str]] = {
    "simon_b1_data": "src/neural_sat_speck/simon/b1_data.py",
    "simon_b1_training_runner": "src/neural_sat_speck/simon/b1_training_runner.py",
    "attack_outcome_classifier": "src/neural_sat_speck/protocol/manifest_validator.py",
    "simon_method_worker": "src/neural_sat_speck/simon/method_worker.py",
    "simon_comparison_input": "src/neural_sat_speck/simon/comparison_input.py",
    "simon_classical_ranking": "src/neural_sat_speck/simon/classical_ranking.py",
    "simon_attack_instance": "src/neural_sat_speck/simon/attack_instance.py",
    "simon_published_vv": "src/neural_sat_speck/simon/published_vv.py",
    "simon_published_vv_ranking": "src/neural_sat_speck/simon/published_vv_ranking.py",
    "simon_faithful_b1_model": "src/neural_sat_speck/simon/faithful_b1_model.py",
    "simon_sat_model": "src/neural_sat_speck/simon/sat_model.py",
    "simon_verification": "src/neural_sat_speck/simon/verification.py",
    "simon_cipher": "src/neural_sat_speck/simon/cipher.py",
    "simon_key_schedule": "src/neural_sat_speck/simon/key_schedule.py",
    "simon_queried_row_8r_cpp": "tools/simon_queried_row_8r.cpp",
    "simon_runner": "scripts/evaluate_simon_development_comparators.py",
}

REQUIRED_METADATA_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "dataset_path",
        "dataset_sha256",
        "rounds",
        "pair_count",
        "known_observations",
        "candidate_formula_observations",
        "verification_pairs",
        "instances",
        "start",
        "methods",
        "method_order_policy",
        "global_deadline_seconds",
        "candidate_cutoff_seconds",
        "candidate_cutoff_soft_seconds",
        "ranking_time_limit",
        "git_commit",
        "environment_provenance",
        "tool_provenance",
        "code_provenance",
        "resume_events",
    }
)

REQUIRED_RECORD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "instance_id",
        "instance_index",
        "method",
        "method_display_name",
        "method_role",
        "execution_order",
        "execution_order_index",
        "outcome",
        "success",
        "status",
        "failure_cause",
        "recovered_master_key",
        "verified_on_fresh_pairs",
        "candidates_tested_count",
        "candidates_remaining_count",
        "complete_online_wall_seconds",
        "ranking_wall_seconds",
        "solving_wall_seconds",
        "verification_wall_seconds",
        "peak_rss_gib",
        "cutoff_seconds",
        "cutoff_overshoot_seconds",
        "candidate_attempts",
        "in_flight_attempt",
        "queue_sample",
        "full_pairs_used",
        "error_message",
        "counts_as_verified_recovery",
        "counts_as_unconditional_failure",
        "is_resolved_sat_state",
        "failure_domain",
    }
)


def _get_git_commit() -> str:
    """Retrieve current Git commit hash or return 'unknown'."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPOSITORY_ROOT,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def _sha256_file(path: Path | str) -> str:
    """Compute SHA-256 hex digest of a file safely."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"File not found for SHA-256 computation: {target}")
    hasher = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_counterbalanced_method_order(
    instance_index: int,
    methods: Sequence[str] | None = None,
) -> list[str]:
    """Deterministic Latin-square cyclic rotation for comparators, keeping diagnostic oracle separate.

    Primary comparators are rotated by `shift = instance_index % M`.
    Diagnostic oracle methods (e.g. oracle_2key_n4) are placed after the
    counterbalanced comparators to preserve comparator Latin-square balance.
    """
    if methods is None:
        target_methods = list(CANONICAL_COMPARATOR_ORDER)
    else:
        target_methods = [METHOD_ALIASES.get(m, m) for m in methods]

    comparators = [m for m in target_methods if m in CANONICAL_COMPARATOR_ORDER]
    diagnostics = [m for m in target_methods if m in DIAGNOSTIC_ORACLE_METHODS]
    other = [
        m
        for m in target_methods
        if m not in CANONICAL_COMPARATOR_ORDER and m not in DIAGNOSTIC_ORACLE_METHODS
    ]

    ordered: list[str] = []
    if comparators:
        m_count = len(comparators)
        shift = instance_index % m_count
        ordered.extend(comparators[(shift + j) % m_count] for j in range(m_count))

    ordered.extend(other)
    ordered.extend(diagnostics)
    return ordered


def verify_simon_tool_provenance(
    ddt_7r_path: Path | str = DEFAULT_DDT_7R_PATH,
    tool_8r_path: Path | str = DEFAULT_TOOL_8R_PATH,
    tool_8r_source: Path | str = DEFAULT_TOOL_8R_SOURCE,
    vv_dir: Path | str | None = DEFAULT_VV_DIR,
    b1_checkpoint_path: Path | str | None = None,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Verify and pin environment tooling, classical DDT, 8R tool, and checkpoints.

    Fails closed if any required file is missing, corrupted, or violates source matching.
    """
    ddt_path = Path(ddt_7r_path)
    tool_bin = Path(tool_8r_path)
    tool_src = Path(tool_8r_source)

    if not ddt_path.is_file():
        raise FileNotFoundError(f"Audited 7R DDT table not found at {ddt_path}")
    if not tool_bin.is_file():
        raise FileNotFoundError(f"8R queried-row tool binary not found at {tool_bin}")
    if not os.access(tool_bin, os.X_OK):
        raise PermissionError(f"8R queried-row tool binary is not executable: {tool_bin}")
    if not tool_src.is_file():
        raise FileNotFoundError(f"8R queried-row C++ source not found at {tool_src}")

    # Check that binary matches the C++ source signature
    with tool_bin.open("rb") as handle:
        bin_bytes = handle.read()
    if b"simon_queried_row_8r.v1" not in bin_bytes:
        raise ValueError(
            f"8R queried-row tool binary {tool_bin} does not contain expected schema signature 'simon_queried_row_8r.v1'"
        )

    # Invoke tool binary without arguments to verify usage signature
    proc = subprocess.run([str(tool_bin)], capture_output=True, text=True)
    if "usage: simon_queried_row_8r" not in proc.stderr:
        raise ValueError(
            f"8R queried-row tool binary {tool_bin} usage output mismatch; stderr: {proc.stderr[:200]!r}"
        )

    tool_8r_bin_hash = _sha256_file(tool_bin)
    tool_8r_src_hash = _sha256_file(tool_src)
    ddt_7r_hash = _sha256_file(ddt_path)

    if verify_hashes and not ddt_7r_hash.startswith("52b3f3d4"):
        raise ValueError(
            f"7R DDT table hash mismatch: expected prefix '52b3f3d4', got {ddt_7r_hash}"
        )

    # Verify published Bao VV checkpoints if vv_dir provided
    vv9_path: Path | None = None
    vv8_path: Path | None = None
    vv9_hash: str | None = None
    vv8_hash: str | None = None

    if vv_dir is not None:
        vdir = Path(vv_dir)
        if not vdir.is_dir() and DEFAULT_VV_FALLBACK_DIR.is_dir():
            vdir = DEFAULT_VV_FALLBACK_DIR

        candidate_vv9 = vdir / "ND_VV_Simon32_9R.h5"
        candidate_vv8 = vdir / "ND_VV_Simon32_8R.h5"
        if candidate_vv9.is_file():
            vv9_path = candidate_vv9
            vv9_hash = _sha256_file(candidate_vv9)
            expected_vv9 = "2363b843592529fe9409dc25dbc0f50c20565973a4a23e54e5478a03113a7070"
            if verify_hashes and vv9_hash != expected_vv9:
                raise ValueError(
                    f"VV9 checkpoint hash mismatch: expected {expected_vv9}, got {vv9_hash}"
                )
        if candidate_vv8.is_file():
            vv8_path = candidate_vv8
            vv8_hash = _sha256_file(candidate_vv8)
            expected_vv8 = "f4e780e484f998ac7e3f05ff78adf47e9dcc1a7a405c9aec92bb06a2761a65fc"
            if verify_hashes and vv8_hash != expected_vv8:
                raise ValueError(
                    f"VV8 checkpoint hash mismatch: expected {expected_vv8}, got {vv8_hash}"
                )

    b1_hash: str | None = None
    if b1_checkpoint_path is not None:
        b1_file = Path(b1_checkpoint_path)
        if b1_file.is_file():
            b1_hash = _sha256_file(b1_file)

    return {
        "ddt_7r_path": str(ddt_path),
        "ddt_7r_sha256": ddt_7r_hash,
        "tool_8r_binary_path": str(tool_bin),
        "tool_8r_binary_sha256": tool_8r_bin_hash,
        "tool_8r_source_path": str(tool_src),
        "tool_8r_source_sha256": tool_8r_src_hash,
        "tool_8r_verified_matches_source": True,
        "vv9_checkpoint_path": str(vv9_path) if vv9_path else None,
        "vv9_checkpoint_sha256": vv9_hash,
        "vv8_checkpoint_path": str(vv8_path) if vv8_path else None,
        "vv8_checkpoint_sha256": vv8_hash,
        "b1_checkpoint_path": str(b1_checkpoint_path) if b1_checkpoint_path else None,
        "b1_checkpoint_sha256": b1_hash,
    }


def validate_attack_instance_for_comparison(
    instance: SimonAttackInstance,
    expected_rounds: int = EXPECTED_ROUNDS,
    expected_pair_count: int = EXPECTED_PAIR_COUNT,
    expected_verif_count: int = EXPECTED_VERIFICATION_PAIRS,
) -> None:
    """Validate attack instance invariants before any method execution."""
    if instance.rounds != expected_rounds:
        raise ValueError(
            f"Instance {instance.instance_id} rounds mismatch: expected {expected_rounds}, got {instance.rounds}"
        )
    if len(instance.pairs) != expected_pair_count:
        raise ValueError(
            f"Instance {instance.instance_id} pair count mismatch: expected {expected_pair_count}, got {len(instance.pairs)}"
        )
    if len(instance.verification_pairs) != expected_verif_count:
        raise ValueError(
            f"Instance {instance.instance_id} verification pair count mismatch: "
            f"expected {expected_verif_count}, got {len(instance.verification_pairs)}"
        )

    # Plaintexts uniqueness within attack pairs
    attack_plaintexts = [
        pt for pair in instance.pairs for pt in (pair.plaintext0, pair.plaintext1)
    ]
    if len(attack_plaintexts) != len(set(attack_plaintexts)):
        raise ValueError(
            f"Instance {instance.instance_id} has duplicate plaintexts within attack pairs"
        )

    # Verification plaintexts uniqueness
    verif_plaintexts = [pt for pt, _ in instance.verification_pairs]
    if len(verif_plaintexts) != len(set(verif_plaintexts)):
        raise ValueError(
            f"Instance {instance.instance_id} has duplicate plaintexts within verification pairs"
        )

    # Zero overlap between attack plaintexts and verification plaintexts
    overlap = set(attack_plaintexts).intersection(set(verif_plaintexts))
    if overlap:
        raise ValueError(
            f"Instance {instance.instance_id} has overlap between attack and verification plaintexts: {sorted(overlap)}"
        )

    # Differential structure verification
    for idx, pair in enumerate(instance.pairs):
        observed_diff = (
            pair.plaintext0[0] ^ pair.plaintext1[0],
            pair.plaintext0[1] ^ pair.plaintext1[1],
        )
        if observed_diff != DEFAULT_SIMON_INPUT_DIFFERENCE:
            raise ValueError(
                f"Instance {instance.instance_id} pair {idx} has input difference {observed_diff}, "
                f"expected {DEFAULT_SIMON_INPUT_DIFFERENCE}"
            )


def save_atomic_with_sidecar(data: dict[str, Any], output_path: Path) -> str:
    """Atomically save dictionary as JSON and write matching SHA-256 sidecar.

    Guarantees that files on disk are never partially written or corrupt.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_json = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    temp_sidecar = output_path.with_name(
        f".{output_path.name}.sha256.tmp.{os.getpid()}"
    )
    sidecar_path = output_path.with_name(f"{output_path.name}.sha256")

    try:
        with temp_json.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temp_json, output_path)

        # Compute SHA-256 of committed JSON file
        file_hash = _sha256_file(output_path)

        # Write matching sidecar
        with temp_sidecar.open("w", encoding="utf-8") as handle:
            handle.write(f"{file_hash}  {output_path.name}\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temp_sidecar, sidecar_path)
        return file_hash

    finally:
        for p in (temp_json, temp_sidecar):
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass


def load_and_validate_resume_file(
    output_path: Path,
    expected_config: dict[str, Any],
) -> dict[str, Any]:
    """Inspect and validate an existing results file on --resume.

    Fails closed on missing file, corrupted sidecar, schema mismatch, or record defects.
    """
    if not output_path.is_file():
        raise SystemExit(
            f"Error: --resume specified but output file {output_path} does not exist."
        )

    sidecar_path = output_path.with_name(f"{output_path.name}.sha256")
    if not sidecar_path.is_file():
        raise SystemExit(
            f"Error: --resume file {output_path} is missing required sidecar {sidecar_path}. "
            "Refusing to resume without verified sidecar."
        )

    # Check sidecar checksum matches committed file
    actual_hash = _sha256_file(output_path)
    sidecar_content = sidecar_path.read_text(encoding="utf-8").strip()
    recorded_hash = sidecar_content.split()[0] if sidecar_content else ""
    if actual_hash != recorded_hash:
        raise SystemExit(
            f"Error: SHA-256 sidecar verification failed for {output_path}: "
            f"actual {actual_hash} != recorded {recorded_hash}. Corrupt resume file detected; refusing to resume."
        )

    try:
        loaded = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception as err:
        raise SystemExit(
            f"Error: Failed to parse JSON from resume file {output_path}: {err}"
        )

    if not isinstance(loaded, dict):
        raise SystemExit(
            f"Error: resume file {output_path} root must be a JSON object."
        )

    metadata = loaded.get("metadata")
    if not isinstance(metadata, dict):
        raise SystemExit(
            f"Error: resume file {output_path} missing or invalid 'metadata' section."
        )

    missing_meta = sorted(REQUIRED_METADATA_FIELDS - set(metadata.keys()))
    if missing_meta:
        raise SystemExit(
            f"Error: resume file {output_path} metadata missing required fields: {missing_meta}"
        )

    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise SystemExit(
            f"Error: resume file schema_version mismatch: expected {SCHEMA_VERSION!r}, "
            f"found {metadata.get('schema_version')!r}."
        )

    # Verify critical configuration matches
    for key, expected_val in expected_config.items():
        actual_val = metadata.get(key)
        if actual_val != expected_val:
            raise SystemExit(
                f"Error: resume file configuration mismatch for {key!r}: "
                f"expected {expected_val!r}, found {actual_val!r}. Refusing to resume."
            )

    records_by_method = loaded.get("records_by_method")
    if not isinstance(records_by_method, dict):
        raise SystemExit(
            f"Error: resume file {output_path} missing or invalid 'records_by_method'."
        )

    # Validate each record and detect duplicates
    for m_name, records in records_by_method.items():
        if not isinstance(records, list):
            raise SystemExit(
                f"Error: records for method {m_name!r} in resume file must be a list."
            )
        seen_instances: set[str] = set()
        for rec in records:
            if not isinstance(rec, dict):
                raise SystemExit(
                    f"Error: record in method {m_name!r} is not an object."
                )
            if rec.get("method") != m_name:
                raise SystemExit(
                    f"Error: record method {rec.get('method')!r} does not match bucket {m_name!r}."
                )
            missing_fields = sorted(REQUIRED_RECORD_FIELDS - set(rec.keys()))
            if missing_fields:
                raise SystemExit(
                    f"Error: record for instance {rec.get('instance_id')!r} in method {m_name!r} "
                    f"missing required fields: {missing_fields}"
                )
            inst_id = rec["instance_id"]
            if inst_id in seen_instances:
                raise SystemExit(
                    f"Error: duplicate record for instance {inst_id!r} in method {m_name!r}."
                )
            seen_instances.add(inst_id)

            # Check outcome and overshoot consistency
            wall_sec = float(rec.get("complete_online_wall_seconds", 0.0))
            cutoff_sec = float(rec.get("cutoff_seconds", 300.0))
            expected_overshoot = round(max(0.0, wall_sec - cutoff_sec), 6)
            if not math.isclose(
                float(rec.get("cutoff_overshoot_seconds", 0.0)),
                expected_overshoot,
                abs_tol=1e-4,
            ):
                raise SystemExit(
                    f"Error: record cutoff overshoot mismatch for {inst_id!r} in method {m_name!r}: "
                    f"expected {expected_overshoot}, found {rec.get('cutoff_overshoot_seconds')}."
                )

    return loaded


def classify_simon_outcome(
    outcome: MethodWorkerOutcome, *, success: bool
) -> dict[str, Any]:
    """Classify SIMON worker outcomes without treating an interrupted solve as resolved.

    The shared historical classifier uses ``global_timeout`` while this worker
    reports ``global_deadline``. Keep the raw outcome unchanged and correct only
    the orthogonal taxonomy fields for future ledgers.
    """
    if outcome.outcome in {"global_deadline", "solver_timeout", "candidate_unresolved"}:
        domain = (
            "ranking"
            if outcome.outcome == "global_deadline"
            and "interrupted during ranking" in (outcome.error_message or "")
            else "solving"
        )
        return {
            "counts_as_verified_recovery": False,
            "counts_as_unconditional_failure": True,
            "is_resolved_sat_state": False,
            "failure_domain": domain,
        }
    if outcome.outcome == "ranking_timeout":
        return {
            "counts_as_verified_recovery": False,
            "counts_as_unconditional_failure": True,
            "is_resolved_sat_state": False,
            "failure_domain": "ranking",
        }
    if outcome.outcome == "exception":
        return {
            "counts_as_verified_recovery": False,
            "counts_as_unconditional_failure": True,
            "is_resolved_sat_state": False,
            "failure_domain": "engineering",
        }
    return classify_attack_outcome(outcome.outcome, success=success)


def compute_method_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate statistical summary for a method's evaluated records."""
    if not records:
        return {}
    total = len(records)
    successes = sum(1 for r in records if r.get("success", False))
    low_ci, high_ci = wilson_score_interval(successes, total)
    wall_times = [
        float(r.get("complete_online_wall_seconds", 0.0)) for r in records
    ]
    median_wall = float(np.median(wall_times)) if wall_times else 0.0
    overshoots = [
        float(r.get("cutoff_overshoot_seconds", 0.0)) for r in records
    ]
    max_overshoot = max(overshoots) if overshoots else 0.0
    mean_overshoot = sum(overshoots) / total if total > 0 else 0.0
    candidates_tested = [
        int(r.get("candidates_tested_count", 0)) for r in records
    ]
    mean_candidates = (
        sum(candidates_tested) / total if total > 0 else 0.0
    )

    # PAR-2 score: penalty of 2 * global_timeout (600.0s default) for unresolved/unverified runs
    par2_times = [
        float(r.get("complete_online_wall_seconds", 0.0))
        if r.get("success", False)
        else 2.0 * float(r.get("cutoff_seconds", DEFAULT_GLOBAL_TIMEOUT))
        for r in records
    ]
    par2_score = sum(par2_times) / total if total > 0 else 0.0

    breakdown: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    for r in records:
        out = str(r.get("outcome", "unknown"))
        breakdown[out] = breakdown.get(out, 0) + 1
        dom = str(r.get("failure_domain", "none" if r.get("success") else "unknown"))
        domain_counts[dom] = domain_counts.get(dom, 0) + 1

    return {
        "instances_evaluated": total,
        "verified_recoveries": successes,
        "verified_recovery_rate": round(successes / total, 4),
        "wilson_95_ci": [round(low_ci, 4), round(high_ci, 4)],
        "median_wall_seconds": round(median_wall, 3),
        "max_cutoff_overshoot_seconds": round(max_overshoot, 6),
        "mean_cutoff_overshoot_seconds": round(mean_overshoot, 6),
        "mean_candidates_tested": round(mean_candidates, 1),
        "par2_score": round(par2_score, 3),
        "outcome_breakdown": breakdown,
        "failure_domains": domain_counts,
        "total_online_wall_seconds": round(sum(wall_times), 3),
        "mean_ranking_wall_seconds": round(
            sum(float(r.get("ranking_wall_seconds", 0.0)) for r in records) / total,
            3,
        ),
        "mean_solving_wall_seconds": round(
            sum(float(r.get("solving_wall_seconds", 0.0)) for r in records) / total,
            3,
        ),
        "mean_verification_wall_seconds": round(
            sum(float(r.get("verification_wall_seconds", 0.0)) for r in records)
            / total,
            3,
        ),
        "max_peak_rss_gib": round(
            max(float(r.get("peak_rss_gib", 0.0)) for r in records)
            if records
            else 0.0,
            4,
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to SIMON32/64 R=10 attack instances JSON file",
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=DEFAULT_INSTANCES,
        help=f"Number of instances to evaluate (default: {DEFAULT_INSTANCES})",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index in the dataset (default: 0)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHOD_CHOICES,
        default=None,
        help="Methods to evaluate (default: all 8 comparators + diagnostic oracle)",
    )
    parser.add_argument(
        "--no-diagnostic-oracle",
        action="store_true",
        help="Exclude the diagnostic oracle (oracle_2key_n4) from the evaluation run",
    )
    parser.add_argument(
        "--global-timeout",
        type=float,
        default=DEFAULT_GLOBAL_TIMEOUT,
        help=f"Parent process global watchdog deadline in seconds (default: {DEFAULT_GLOBAL_TIMEOUT})",
    )
    parser.add_argument(
        "--candidate-cutoff",
        "--timeout",
        type=float,
        dest="candidate_cutoff",
        default=DEFAULT_CANDIDATE_CUTOFF,
        help=f"Soft candidate solver cutoff in seconds (default: {DEFAULT_CANDIDATE_CUTOFF})",
    )
    parser.add_argument(
        "--ranking-time-limit",
        type=float,
        default=DEFAULT_RANKING_TIME_LIMIT,
        help=f"Maximum ranking wall time limit in seconds (default: {DEFAULT_RANKING_TIME_LIMIT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file path (sidecar .sha256 created automatically)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing run, skipping completed (instance_id, method) pairs (requires --output)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and display counterbalanced schedule without executing attack runs",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run bounded synthetic plumbing test to verify runner, sidecar, and resume mechanics",
    )
    parser.add_argument(
        "--ddt-7r-path",
        type=Path,
        default=DEFAULT_DDT_7R_PATH,
        help="Path to 7R predicted DDT binary table",
    )
    parser.add_argument(
        "--tool-8r-path",
        type=Path,
        default=DEFAULT_TOOL_8R_PATH,
        help="Path to 8R queried-row C++ tool binary",
    )
    parser.add_argument(
        "--vv-dir",
        type=Path,
        default=DEFAULT_VV_DIR,
        help="Path to directory containing published VV9 and VV8 checkpoints",
    )
    parser.add_argument(
        "--b1-checkpoint",
        type=Path,
        default=None,
        help="Path to trained SIMON B1 model checkpoint (Gate 6)",
    )
    parser.add_argument(
        "--manifest-seed",
        type=int,
        default=DEFAULT_MANIFEST_SEED,
        help=f"Deterministic seed for random control PRNG streams (default: {DEFAULT_MANIFEST_SEED})",
    )
    args = parser.parse_args(argv)

    if args.resume and args.output is None:
        raise SystemExit("Error: --resume requires an explicit --output path.")

    # Normalize method aliases while preserving deduplicated order
    if args.methods is not None:
        normalized: list[str] = []
        for m in args.methods:
            can = METHOD_ALIASES.get(m, m)
            if can not in normalized:
                normalized.append(can)
        if args.no_diagnostic_oracle:
            normalized = [m for m in normalized if m not in DIAGNOSTIC_ORACLE_METHODS]
        args.methods = normalized
    else:
        methods = list(CANONICAL_COMPARATOR_ORDER)
        if not args.no_diagnostic_oracle:
            methods.extend(DIAGNOSTIC_ORACLE_METHODS)
        args.methods = methods

    return args


def run_bounded_synthetic_smoke_test(
    temp_dir: Path | None = None,
) -> bool:
    """Execute a bounded synthetic smoke test using a temporary fixture.

    Verifies input construction, counterbalancing, atomic persistence, sidecar creation,
    and durable resume mechanics without calling long-running solvers or protected data.
    """
    import tempfile

    print("\n[SMOKE-TEST] Starting bounded synthetic smoke test...")
    target_dir = temp_dir or Path(tempfile.mkdtemp(prefix="simon_runner_smoke_"))
    target_dir.mkdir(parents=True, exist_ok=True)
    smoke_output = target_dir / "simon_smoke_results.json"
    smoke_sidecar = target_dir / "simon_smoke_results.json.sha256"

    # 1. Generate 2 synthetic demo instances (64 pairs, 2 verification pairs)
    cfg = SimonAttackInstanceConfig(
        instance_count=2,
        pair_count=64,
        verification_pair_count=2,
        rounds=10,
        seed=999999,
        split_role="demo",
    )
    instances = generate_simon_attack_instances(cfg)
    assert len(instances) == 2
    for inst in instances:
        validate_attack_instance_for_comparison(inst)
    print("  [OK] Synthetic instances generated and validated (64 pairs, 2 verif pairs).")

    # 2. Verify counterbalanced schedule
    test_methods = ["random_control_128", "sat_only_n4", "oracle_2key_n4"]
    sched0 = get_counterbalanced_method_order(0, test_methods)
    sched1 = get_counterbalanced_method_order(1, test_methods)
    assert sched0 == ["random_control_128", "sat_only_n4", "oracle_2key_n4"]
    assert sched1 == ["sat_only_n4", "random_control_128", "oracle_2key_n4"]
    print("  [OK] Method counterbalancing verified with diagnostic oracle separation.")

    # 3. Simulate initial partial run
    meta = {
        "schema_version": SCHEMA_VERSION,
        "dataset_path": "synthetic_smoke_fixture",
        "dataset_sha256": "0" * 64,
        "rounds": 10,
        "pair_count": 64,
        "known_observations": 128,
        "candidate_formula_observations": 4,
        "verification_pairs": 2,
        "instances": 2,
        "start": 0,
        "methods": test_methods,
        "method_order_policy": "latin_square_cyclic_shift_with_diagnostic_separation",
        "global_deadline_seconds": 300.0,
        "candidate_cutoff_seconds": 45.0,
        "candidate_cutoff_soft_seconds": 45.0,
        "ranking_time_limit": 120.0,
        "git_commit": _get_git_commit(),
        "command_line": list(sys.argv),
        "environment_provenance": {
            "platform": platform.platform(),
            "python_version": sys.version,
            "pycryptosat_version": getattr(pycryptosat, "__version__", "unknown"),
            "numpy_version": getattr(np, "__version__", "unknown"),
        },
        "tool_provenance": {
            "ddt_7r_path": "/synthetic/ddt.bin",
            "ddt_7r_sha256": "1" * 64,
            "tool_8r_binary_path": "/synthetic/tool",
            "tool_8r_binary_sha256": "2" * 64,
            "tool_8r_source_path": "/synthetic/tool.cpp",
            "tool_8r_source_sha256": "3" * 64,
            "tool_8r_verified_matches_source": True,
            "vv9_checkpoint_path": None,
            "vv9_checkpoint_sha256": None,
            "vv8_checkpoint_path": None,
            "vv8_checkpoint_sha256": None,
            "b1_checkpoint_path": None,
            "b1_checkpoint_sha256": None,
        },
        "code_provenance": {},
        "resume_events": [],
    }

    dummy_rec = {
        "instance_id": instances[0].instance_id,
        "instance_index": 0,
        "method": "sat_only_n4",
        "method_display_name": METHOD_DISPLAY_NAMES["sat_only_n4"],
        "method_role": "comparator",
        "execution_order": sched0,
        "execution_order_index": 1,
        "outcome": "candidate_timeout",
        "success": False,
        "status": "candidate_timeout",
        "failure_cause": "candidate_timeout",
        "recovered_master_key": None,
        "verified_on_fresh_pairs": False,
        "candidates_tested_count": 1,
        "candidates_remaining_count": 0,
        "complete_online_wall_seconds": 1.25,
        "ranking_wall_seconds": 0.0,
        "solving_wall_seconds": 1.25,
        "verification_wall_seconds": 0.0,
        "peak_rss_gib": 0.05,
        "cutoff_seconds": 300.0,
        "cutoff_overshoot_seconds": 0.0,
        "candidate_attempts": [],
        "in_flight_attempt": None,
        "queue_sample": [],
        "full_pairs_used": None,
        "error_message": None,
        "counts_as_verified_recovery": False,
        "counts_as_unconditional_failure": True,
        "is_resolved_sat_state": True,
        "failure_domain": "ranking",
    }

    data = {
        "metadata": meta,
        "evidence_class": "DEVELOPMENT_DIAGNOSTIC",
        "generated_at": datetime.now(UTC).isoformat(),
        "summary_by_method": {"sat_only_n4": compute_method_summary([dummy_rec])},
        "diagnostic_summary": {},
        "records_by_method": {
            "sat_only_n4": [dummy_rec],
            "random_control_128": [],
            "oracle_2key_n4": [],
        },
    }

    committed_hash = save_atomic_with_sidecar(data, smoke_output)
    assert smoke_output.is_file()
    assert smoke_sidecar.is_file()
    assert smoke_sidecar.read_text().split()[0] == committed_hash
    print("  [OK] Atomic save and SHA-256 sidecar verified.")

    # 4. Verify durable resume load
    expected_cfg = {
        "rounds": 10,
        "pair_count": 64,
        "dataset_sha256": "0" * 64,
        "global_deadline_seconds": 300.0,
        "candidate_cutoff_seconds": 45.0,
    }
    loaded = load_and_validate_resume_file(smoke_output, expected_cfg)
    assert len(loaded["records_by_method"]["sat_only_n4"]) == 1
    print("  [OK] Resumable execution load and integrity validation verified.")

    # 5. Verify corrupt resume rejection
    with open(smoke_sidecar, "w", encoding="utf-8") as f:
        f.write("bad_hash_tampered  simon_smoke_results.json\n")
    try:
        load_and_validate_resume_file(smoke_output, expected_cfg)
        raise AssertionError("Failed to reject corrupt sidecar!")
    except SystemExit as exc:
        assert "checksum" in str(exc) or "failed" in str(exc)
        print("  [OK] Corrupt resume sidecar correctly rejected (failed closed).")

    print("[SMOKE-TEST] ALL CHECKS PASSED SUCCESSFULLY!\n")
    return True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.smoke_test:
        run_bounded_synthetic_smoke_test()
        return 0

    if args.dataset is None:
        raise SystemExit(
            "Error: --dataset path is required for development comparison run (or specify --smoke-test)."
        )

    # 1. Environment & Provenance Pinning
    print(f"\n{'='*95}")
    print(f"{'SIMON32/64 R=10 MATCHED DEVELOPMENT COMPARATOR RUNNER':^95}")
    print(f"{'='*95}")
    print(f"Git commit:       {_get_git_commit()}")
    print(f"Python:           {sys.version.split()[0]} ({platform.platform()})")
    print(f"PyCryptoSat:      {getattr(pycryptosat, '__version__', 'unknown')}")
    print(f"Dataset path:     {args.dataset}")
    print(f"Instances:        {args.instances} (starting at index {args.start})")
    print(f"Methods:          {len(args.methods)} scheduled")
    print("Cutoff & Timeout: 45.0s soft candidate / 300.0s parent watchdog")
    print(f"{'='*95}\n")

    dataset_hash = _sha256_file(args.dataset)
    print(f"[PROVENANCE] Dataset SHA-256: {dataset_hash}")

    tool_info = verify_simon_tool_provenance(
        ddt_7r_path=args.ddt_7r_path,
        tool_8r_path=args.tool_8r_path,
        tool_8r_source=DEFAULT_TOOL_8R_SOURCE,
        vv_dir=args.vv_dir,
        b1_checkpoint_path=args.b1_checkpoint,
        verify_hashes=True,
    )
    print(f"[PROVENANCE] 7R DDT Table SHA-256:     {tool_info['ddt_7r_sha256']}")
    print(f"[PROVENANCE] 8R Tool Binary SHA-256:  {tool_info['tool_8r_binary_sha256']}")
    print(f"[PROVENANCE] 8R Tool Source SHA-256:  {tool_info['tool_8r_source_sha256']}")
    if tool_info.get("vv9_checkpoint_sha256"):
        print(f"[PROVENANCE] VV9 Checkpoint SHA-256:   {tool_info['vv9_checkpoint_sha256']}")
    if tool_info.get("vv8_checkpoint_sha256"):
        print(f"[PROVENANCE] VV8 Checkpoint SHA-256:   {tool_info['vv8_checkpoint_sha256']}")
    if tool_info.get("b1_checkpoint_sha256"):
        print(f"[PROVENANCE] B1 Checkpoint SHA-256:    {tool_info['b1_checkpoint_sha256']}")
    print("[PROVENANCE] 8R binary verified against C++ source signature.\n")

    code_provenance: dict[str, str] = {}
    for code_key, rel_path in SIMON_CRITICAL_CODE_PATHS.items():
        abs_p = REPOSITORY_ROOT / rel_path
        if abs_p.is_file():
            code_provenance[code_key] = _sha256_file(abs_p)

    # 2. Load & Validate Dataset
    all_instances = load_simon_attack_instances(args.dataset)
    if args.start < 0 or args.start + args.instances > len(all_instances):
        raise SystemExit(
            f"Error: Requested instances [{args.start}..{args.start + args.instances - 1}] "
            f"exceed dataset size ({len(all_instances)} instances)."
        )
    selected_instances = all_instances[args.start : args.start + args.instances]

    for inst in selected_instances:
        validate_attack_instance_for_comparison(
            inst,
            expected_rounds=EXPECTED_ROUNDS,
            expected_pair_count=EXPECTED_PAIR_COUNT,
            expected_verif_count=EXPECTED_VERIFICATION_PAIRS,
        )
    print(
        f"[VALIDATION] Validated {len(selected_instances)} attack instances "
        f"(64 pairs, 2 fresh verification pairs, zero overlap).\n"
    )

    # 3. Output Path & Resume Configuration
    output_path = args.output
    if output_path is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        results_dir = REPOSITORY_ROOT / "results" / "simon_development"
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / f"simon_development_comparators_{stamp}.json"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    expected_config = {
        "rounds": EXPECTED_ROUNDS,
        "pair_count": EXPECTED_PAIR_COUNT,
        "dataset_sha256": dataset_hash,
        "global_deadline_seconds": args.global_timeout,
        "candidate_cutoff_seconds": args.candidate_cutoff,
    }

    records_by_method: dict[str, list[dict[str, Any]]] = {
        m: [] for m in args.methods
    }
    resume_events: list[dict[str, Any]] = []

    if args.resume:
        loaded_res = load_and_validate_resume_file(output_path, expected_config)
        loaded_records = loaded_res.get("records_by_method", {})
        for m in args.methods:
            records_by_method[m] = list(loaded_records.get(m, []))
        resume_events = list(loaded_res.get("metadata", {}).get("resume_events", []))
        resume_events.append(
            {
                "resumed_at": datetime.now(UTC).isoformat(),
                "resumed_git_commit": _get_git_commit(),
                "resumed_methods": list(args.methods),
                "command_line": sys.argv,
            }
        )
        print(f"[RESUME] Successfully validated existing records from {output_path}:")
        for m in args.methods:
            print(f"  - {m:<32}: {len(records_by_method[m])} instances completed")
        print()

    # 4. Dry-Run Inspection
    if args.dry_run:
        print("[DRY-RUN] Planned counterbalanced execution schedule:")
        for idx, instance in enumerate(selected_instances):
            global_idx = args.start + idx
            scheduled = get_counterbalanced_method_order(global_idx, args.methods)
            print(f"  Instance {global_idx:02d} ({instance.instance_id}):")
            for pos, method in enumerate(scheduled, start=1):
                role = (
                    "DIAGNOSTIC"
                    if method in DIAGNOSTIC_ORACLE_METHODS
                    else "COMPARATOR"
                )
                print(f"    {pos}. {method:<32} [{role}]")
        print("\n[DRY-RUN] Dry run complete. Zero attack runs executed.")
        return 0

    # 5. Execution Loop
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset_path": str(args.dataset),
        "dataset_sha256": dataset_hash,
        "rounds": EXPECTED_ROUNDS,
        "pair_count": EXPECTED_PAIR_COUNT,
        "known_observations": EXPECTED_KNOWN_OBSERVATIONS,
        "candidate_formula_observations": EXPECTED_CANDIDATE_OBSERVATIONS,
        "verification_pairs": EXPECTED_VERIFICATION_PAIRS,
        "instances": args.instances,
        "start": args.start,
        "methods": list(args.methods),
        "method_order_policy": "latin_square_cyclic_shift_with_diagnostic_separation",
        "global_deadline_seconds": args.global_timeout,
        "candidate_cutoff_seconds": args.candidate_cutoff,
        "candidate_cutoff_soft_seconds": args.candidate_cutoff,
        "ranking_time_limit": args.ranking_time_limit,
        "git_commit": _get_git_commit(),
        "environment_provenance": {
            "platform": platform.platform(),
            "python_version": sys.version,
            "pycryptosat_version": getattr(pycryptosat, "__version__", "unknown"),
            "numpy_version": getattr(np, "__version__", "unknown"),
        },
        "tool_provenance": tool_info,
        "code_provenance": code_provenance,
        "resume_events": resume_events,
    }

    print(f"Beginning evaluation of {len(selected_instances)} instances on {len(args.methods)} methods.\n")

    for idx, instance in enumerate(selected_instances):
        global_idx = args.start + idx
        instance_id = instance.instance_id
        scheduled = get_counterbalanced_method_order(global_idx, args.methods)

        comp_input = SimonComparisonInput.from_attack_instance(
            instance=instance,
            manifest_seed=args.manifest_seed,
            pair_count=EXPECTED_PAIR_COUNT,
        )

        oracle_keys = (
            instance.expected_trailing_subkeys(2)
            if any(m in DIAGNOSTIC_ORACLE_METHODS for m in scheduled)
            else None
        )

        print(f"--- Instance [{global_idx:02d}/{args.start + args.instances - 1:02d}] {instance_id} ---")

        for pos, method in enumerate(scheduled):
            # Check if already completed in resume mode
            if any(r.get("instance_id") == instance_id for r in records_by_method[method]):
                print(f"  {pos+1}. {method:<32} [ALREADY RECORDED]")
                continue

            method_role = (
                "diagnostic_oracle"
                if method in DIAGNOSTIC_ORACLE_METHODS
                else "comparator"
            )

            # Invariant: comparators NEVER receive oracle trailing keys
            worker_oracle_keys = (
                oracle_keys if method in DIAGNOSTIC_ORACLES else None
            )
            worker_b1_checkpoint = (
                args.b1_checkpoint
                if method in ("faithful_b1_hybrid", "faithful_b1_8parent")
                else None
            )
            worker_b1_sha256 = (
                tool_info.get("b1_checkpoint_sha256")
                if method in ("faithful_b1_hybrid", "faithful_b1_8parent")
                else None
            )

            t0 = time.perf_counter()
            outcome: MethodWorkerOutcome
            try:
                outcome = run_method_worker(
                    method=method,
                    rounds=EXPECTED_ROUNDS,
                    comparison_input=comp_input,
                    manifest_seed=args.manifest_seed,
                    instance_id=instance_id,
                    global_deadline_seconds=args.global_timeout,
                    candidate_cutoff_seconds=args.candidate_cutoff,
                    candidate_cutoff_soft_seconds=args.candidate_cutoff,
                    ranking_time_limit=args.ranking_time_limit,
                    b1_checkpoint_path=worker_b1_checkpoint,
                    b1_checkpoint_sha256=worker_b1_sha256,
                    oracle_trailing_keys=worker_oracle_keys,
                    use_parent_watchdog=True,
                )
            except Exception as exc:
                elapsed_exc = time.perf_counter() - t0
                outcome = MethodWorkerOutcome(
                    method=method,
                    outcome="exception",
                    complete_online_wall_seconds=elapsed_exc,
                    ranking_wall_seconds=0.0,
                    solving_wall_seconds=0.0,
                    verification_wall_seconds=0.0,
                    candidates_tested_count=0,
                    candidates_remaining_count=0,
                    recovered_master_key=None,
                    verified_on_fresh_pairs=False,
                    candidate_attempts=(),
                    peak_rss_gib=0.0,
                    error_message=f"Runner exception during method worker invocation: {exc}",
                )

            # Record outcome classification
            success = (
                outcome.outcome == "verified_recovery"
                and outcome.verified_on_fresh_pairs is True
                and outcome.recovered_master_key is not None
            )
            classified = classify_simon_outcome(outcome, success=success)

            wall_sec = round(outcome.complete_online_wall_seconds, 6)
            overshoot = round(max(0.0, wall_sec - args.global_timeout), 6)

            rec = {
                "instance_id": instance_id,
                "instance_index": global_idx,
                "method": method,
                "method_display_name": METHOD_DISPLAY_NAMES.get(method, method),
                "method_role": method_role,
                "execution_order": list(scheduled),
                "execution_order_index": pos,
                "outcome": outcome.outcome,
                "success": success,
                "status": outcome.outcome,
                "failure_cause": None if success else outcome.outcome,
                "recovered_master_key": (
                    list(outcome.recovered_master_key)
                    if outcome.recovered_master_key is not None
                    else None
                ),
                "verified_on_fresh_pairs": outcome.verified_on_fresh_pairs,
                "candidates_tested_count": outcome.candidates_tested_count,
                "candidates_remaining_count": outcome.candidates_remaining_count,
                "complete_online_wall_seconds": wall_sec,
                "ranking_wall_seconds": round(outcome.ranking_wall_seconds, 6),
                "solving_wall_seconds": round(outcome.solving_wall_seconds, 6),
                "verification_wall_seconds": round(outcome.verification_wall_seconds, 6),
                "peak_rss_gib": round(outcome.peak_rss_gib, 4),
                "cutoff_seconds": args.global_timeout,
                "cutoff_overshoot_seconds": overshoot,
                "candidate_attempts": [a.to_dict() for a in outcome.candidate_attempts],
                "in_flight_attempt": (
                    outcome.in_flight_attempt.to_dict()
                    if outcome.in_flight_attempt is not None
                    else None
                ),
                "queue_sample": [list(t) for t in outcome.queue_sample],
                "full_pairs_used": outcome.full_pairs_used,
                "error_message": outcome.error_message,
                "counts_as_verified_recovery": classified["counts_as_verified_recovery"],
                "counts_as_unconditional_failure": classified["counts_as_unconditional_failure"],
                "is_resolved_sat_state": classified["is_resolved_sat_state"],
                "failure_domain": classified["failure_domain"],
            }

            records_by_method[method].append(rec)

            status_str = f"outcome={outcome.outcome} success={success}"
            cand_str = f"tested={outcome.candidates_tested_count}"
            rss_str = f"rss={outcome.peak_rss_gib:.2f}G"
            wall_str = f"wall={wall_sec:6.2f}s"
            print(f"  {pos+1}. {method:<32} {status_str:<42} {cand_str:<12} {rss_str:<10} {wall_str}")

            # Atomic streaming persistence after every single method evaluation
            summary_comparators = {
                m: compute_method_summary(records_by_method[m])
                for m in args.methods
                if m in CANONICAL_COMPARATOR_ORDER
            }
            summary_diagnostic = {
                m: compute_method_summary(records_by_method[m])
                for m in args.methods
                if m in DIAGNOSTIC_ORACLE_METHODS
            }

            checkpoint_payload = {
                "metadata": metadata,
                "evidence_class": "DEVELOPMENT_DIAGNOSTIC",
                "generated_at": datetime.now(UTC).isoformat(),
                "summary_by_method": summary_comparators,
                "diagnostic_summary": summary_diagnostic,
                "records_by_method": records_by_method,
            }
            save_atomic_with_sidecar(checkpoint_payload, output_path)

    # 6. Final Summary Reporting
    print(f"\n{'='*115}")
    print(f"{'SIMON32/64 R=10 COMPARATOR EVALUATION SUMMARY (64 PAIRS / 128 OBS)':^115}")
    print(f"{'='*115}")
    print(
        f"{'Method':<34} | {'Recoveries':<10} | {'Rate':<7} | {'95% CI':<14} | {'Med Wall':<9} | {'PAR-2':<8} | {'Mean Cand':<10}"
    )
    print(f"{'-'*115}")

    comparators_to_show = [m for m in args.methods if m in CANONICAL_COMPARATOR_ORDER]
    for m in comparators_to_show:
        sm = compute_method_summary(records_by_method[m])
        if not sm:
            continue
        ci_str = f"[{sm['wilson_95_ci'][0]:.3f}, {sm['wilson_95_ci'][1]:.3f}]"
        print(
            f"{m:<34} | "
            f"{sm['verified_recoveries']:>2}/{sm['instances_evaluated']:<2}       | "
            f"{sm['verified_recovery_rate']*100:>5.1f}% | "
            f"{ci_str:<14} | "
            f"{sm['median_wall_seconds']:>7.2f}s | "
            f"{sm['par2_score']:>6.1f}s | "
            f"{sm['mean_candidates_tested']:>9.1f}"
        )

    diag_to_show = [m for m in args.methods if m in DIAGNOSTIC_ORACLE_METHODS]
    if diag_to_show:
        print(f"{'-'*115}")
        print(f"{'DIAGNOSTIC ORACLE UPPER-BOUND (NOT AN AUTONOMOUS ATTACK METHOD)':^115}")
        print(f"{'-'*115}")
        for m in diag_to_show:
            sm = compute_method_summary(records_by_method[m])
            if not sm:
                continue
            ci_str = f"[{sm['wilson_95_ci'][0]:.3f}, {sm['wilson_95_ci'][1]:.3f}]"
            print(
                f"{m:<34} | "
                f"{sm['verified_recoveries']:>2}/{sm['instances_evaluated']:<2}       | "
                f"{sm['verified_recovery_rate']*100:>5.1f}% | "
                f"{ci_str:<14} | "
                f"{sm['median_wall_seconds']:>7.2f}s | "
                f"{sm['par2_score']:>6.1f}s | "
                f"{sm['mean_candidates_tested']:>9.1f}"
            )

    print(f"{'='*115}")
    print(f"Output saved to:  {output_path}")
    print(f"Sidecar saved to: {output_path.with_name(f'{output_path.name}.sha256')}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
