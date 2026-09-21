#!/usr/bin/env python3
"""Deterministic Generation for the 7-Round SPECK32/64 Prospective Evaluation Dataset (N=128).

Evidence Class: PROSPECTIVE_EVALUATION.
Supervisor Authorization: Decision D-031 (2026-09-21).

Enforces:
- Seed drawn exactly once from OS CSPRNG (secrets.SystemRandom).
- Pre-generation rejection of all historical, development, quarantined, or retired seeds.
- Pre-generation recording of the drawn seed in the prospective dataset manifest.
- Exactly 128 independent attack instances on 7-round SPECK32/64.
- Exactly 64 differential pairs per instance (128 observations) with input difference (0x0040, 0x0000).
- Established independent fresh-pair verification construction.
- Verification of deterministic replay before saving.
- Atomic write with overwrite protection.
- Labelled strictly as PROSPECTIVE_EVALUATION.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from neural_sat_speck.data.attack import (  # noqa: E402
    AttackInstanceConfig,
    generate_attack_instances,
    write_attack_shard,
)
from neural_sat_speck.data.format import canonical_json  # noqa: E402
from neural_sat_speck.data.seed_policy import (  # noqa: E402
    CLI_DEMO_SEED,
    HISTORICAL_SEEDS,
    PROTECTED_RESEARCH_SEEDS,
    SeedRole,
    validate_seed_for_role,
)
from neural_sat_speck.protocol.manifest_validator import (  # noqa: E402
    RETIRED_OR_OPENED_SEEDS,
    compute_sha256,
)

OUTPUT_DIR: Final[Path] = REPO_ROOT / "datasets" / "phase11_prospective"
DATASET_ID: Final[str] = "speck32_64_attack_phase11_prospective_r7"
SHARD_FILENAME: Final[str] = "prospective_evaluation.json"
MANIFEST_FILENAME: Final[str] = "manifest.json"
INSTANCE_COUNT: Final[int] = 128
ROUNDS: Final[int] = 7
PAIR_COUNT: Final[int] = 64
INPUT_DIFFERENCE: Final[tuple[int, int]] = (64, 0)
SCHEMA: Final[str] = "neural_sat_speck.attack_shard.v1"

ALL_PROHIBITED_SEEDS: Final[frozenset[int]] = frozenset(
    HISTORICAL_SEEDS
    | PROTECTED_RESEARCH_SEEDS
    | RETIRED_OR_OPENED_SEEDS
    | {
        CLI_DEMO_SEED,
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


def draw_csprng_seed() -> int:
    """Draw one safe integer seed from OS CSPRNG, rejecting all prohibited seeds."""
    sys_random = secrets.SystemRandom()
    for _ in range(10_000):
        candidate = sys_random.randint(100_000, 999_998)
        if candidate in ALL_PROHIBITED_SEEDS:
            continue
        try:
            validate_seed_for_role(candidate, SeedRole.PROSPECTIVE_VALIDATION)
            return candidate
        except ValueError:
            continue
    raise RuntimeError("Failed to draw a valid non-prohibited seed from CSPRNG.")


def compute_generation_config_hash(seed: int) -> str:
    """Compute the authoritative generation-configuration hash."""
    config_dict = {
        "cipher": "SPECK32/64",
        "dataset_id": DATASET_ID,
        "input_difference": list(INPUT_DIFFERENCE),
        "instance_count": INSTANCE_COUNT,
        "key_generation_policy": "one_random_master_key_per_instance",
        "pair_construction": "plaintext_difference",
        "pair_count": PAIR_COUNT,
        "rounds": ROUNDS,
        "schema": SCHEMA,
        "seed": seed,
        "split_role": "validation",
    }
    payload = canonical_json(config_dict).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_and_verify_dataset(
    seed: int,
    *,
    timestamp: str,
    git_commit: str,
) -> tuple[Any, str, str]:
    """Generate the prospective dataset, verify replay and cryptographic consistency."""
    config = AttackInstanceConfig(
        instance_count=INSTANCE_COUNT,
        pair_count=PAIR_COUNT,
        rounds=ROUNDS,
        seed=seed,
        split_role="validation",
        input_difference=INPUT_DIFFERENCE,
    )

    print(f"Generating {INSTANCE_COUNT} prospective attack instances (seed={seed}, rounds={ROUNDS}, pairs={PAIR_COUNT})...")
    instances_a = generate_attack_instances(
        config,
        generated_at=timestamp,
        repository_commit=git_commit,
    )

    print("Verifying deterministic replay consistency...")
    instances_b = generate_attack_instances(
        config,
        generated_at=timestamp,
        repository_commit=git_commit,
    )

    checksum_a = instances_a.metadata["data_checksum"]
    checksum_b = instances_b.metadata["data_checksum"]
    if checksum_a != checksum_b:
        raise RuntimeError(f"Deterministic replay failed: checksum {checksum_a} != {checksum_b}")

    config_hash = compute_generation_config_hash(seed)
    return instances_a, checksum_a, config_hash


def generate_prospective_dataset(
    *,
    reproduce: bool = False,
    verify_only: bool = False,
) -> dict[str, Any]:
    """Execute the controlled prospective dataset generation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_DIR / MANIFEST_FILENAME
    shard_path = OUTPUT_DIR / SHARD_FILENAME
    sidecar_path = OUTPUT_DIR / f"{SHARD_FILENAME.replace('.json', '.sha256')}"

    # Get repository commit
    git_commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
        .decode("utf-8")
        .strip()
    )

    if verify_only:
        if not shard_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(f"Cannot verify: {shard_path} or {manifest_path} does not exist.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual_sha = compute_sha256(shard_path)
        expected_sha = manifest["file_sha256"]
        if actual_sha != expected_sha:
            raise ValueError(f"Integrity failure: computed SHA-256 {actual_sha} != manifest {expected_sha}")
        print(f"[✓] Verification succeeded! Shard SHA-256: {actual_sha}")
        return manifest

    if shard_path.exists() and not reproduce:
        raise FileExistsError(
            f"Protected dataset {shard_path} already exists! Overwrite is strictly prohibited. "
            "To verify deterministic replay against the recorded seed, pass --reproduce."
        )

    if reproduce:
        if not manifest_path.exists():
            raise FileNotFoundError(f"--reproduce requires existing manifest at {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        seed = manifest["seed"]
        timestamp = manifest["generated_at"]
        print(f"[✓] Reproducing dataset from pre-recorded CSPRNG seed: {seed}")
    else:
        seed = draw_csprng_seed()
        timestamp = datetime.now(UTC).isoformat()
        print(f"[✓] Drawn fresh CSPRNG seed: {seed}")

        # Pre-record the seed in the dataset manifest BEFORE generation
        pre_manifest = {
            "dataset_id": DATASET_ID,
            "evidence_class": "PROSPECTIVE_EVALUATION",
            "protocol_id": "speck32_64_seven_round_comparative_protocol_v1",
            "authorization_decision": "D-031",
            "authorized_by": "F.M. Ashfaq, Project Author and Supervisor",
            "seed": seed,
            "seed_source": "secrets.SystemRandom",
            "sample_size_n": INSTANCE_COUNT,
            "rounds": ROUNDS,
            "ranking_differential_pairs": PAIR_COUNT,
            "input_difference": list(INPUT_DIFFERENCE),
            "generated_at": timestamp,
            "git_commit": git_commit,
            "status": "PRE_GENERATION_RECORDED",
        }
        manifest_path.write_text(json.dumps(pre_manifest, indent=2) + "\n", encoding="utf-8")
        print(f"[✓] Pre-recorded seed {seed} in {manifest_path}")

    # Generate and verify instances
    instances, data_checksum, config_hash = build_and_verify_dataset(
        seed,
        timestamp=timestamp,
        git_commit=git_commit,
    )

    # Atomic write of attack shard
    temp_shard_path = shard_path.with_suffix(".tmp")
    write_attack_shard(
        instances,
        temp_shard_path,
        dataset_id=DATASET_ID,
    )
    temp_shard_path.replace(shard_path)
    actual_file_sha256 = compute_sha256(shard_path)
    print(f"[✓] Atomically wrote protected dataset to {shard_path} (SHA-256: {actual_file_sha256})")

    # Generate sidecar
    rel_shard_path = shard_path.relative_to(REPO_ROOT)
    sidecar_content = f"{actual_file_sha256}  {rel_shard_path}\n"
    sidecar_path.write_text(sidecar_content, encoding="utf-8")
    print(f"[✓] Wrote detached sidecar to {sidecar_path}")

    # Finalize manifest
    final_manifest = {
        "dataset_id": DATASET_ID,
        "evidence_class": "PROSPECTIVE_EVALUATION",
        "protocol_id": "speck32_64_seven_round_comparative_protocol_v1",
        "authorization_decision": "D-031",
        "authorized_by": "F.M. Ashfaq, Project Author and Supervisor",
        "seed": seed,
        "seed_source": "secrets.SystemRandom",
        "sample_size_n": INSTANCE_COUNT,
        "rounds": ROUNDS,
        "ranking_differential_pairs": PAIR_COUNT,
        "input_difference": list(INPUT_DIFFERENCE),
        "generated_at": timestamp,
        "git_commit": git_commit,
        "file_name": SHARD_FILENAME,
        "file_sha256": actual_file_sha256,
        "data_checksum": data_checksum,
        "generation_config_hash": config_hash,
        "schema": SCHEMA,
        "status": "FROZEN_PROSPECTIVE_EVALUATION",
    }
    manifest_path.write_text(json.dumps(final_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[✓] Finalized prospective dataset manifest at {manifest_path}")

    return final_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the 7-round SPECK32/64 prospective evaluation dataset (Decision D-031)."
    )
    parser.add_argument(
        "--reproduce",
        action="store_true",
        help="Reproduce and verify dataset from existing recorded seed without drawing a new seed.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify integrity of existing dataset on disk without re-generating.",
    )
    args = parser.parse_args(argv)

    print("=== 7-Round SPECK32/64 Prospective Evaluation Dataset Generator ===")
    print("Authority: Decision D-031 (F.M. Ashfaq, Project Author and Supervisor)")
    print(f"Cohort: N={INSTANCE_COUNT}, Rounds={ROUNDS}, Pairs={PAIR_COUNT}\n")

    manifest = generate_prospective_dataset(
        reproduce=args.reproduce,
        verify_only=args.verify_only,
    )

    print("\n--- Prospective Dataset Summary ---")
    print(f"Dataset ID:             {manifest['dataset_id']}")
    print(f"Drawn Seed:             {manifest['seed']}")
    print(f"Dataset File SHA-256:   {manifest['file_sha256']}")
    print(f"Semantic Data Checksum: {manifest['data_checksum']}")
    print(f"Config Hash:            {manifest['generation_config_hash']}")
    print("----------------------------------\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
