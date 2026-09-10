#!/usr/bin/env python3
"""Deterministic Generation for Phase 10-G2 Prospective Validation Family (N=256, Seed 201001).

Strictly adheres to:
- Seed policy validation (Seed 201001, Role: PHASE10_G2_PROSPECTIVE_VALIDATION).
- Rejection of all quarantined, historical, development, and prior prospective seeds:
  (98001, 97001, 99001, 100001, 101001, 200001).
- Complete metadata tracking: git commit, protocol, and dataset checksum.
- Verification of deterministic replay before export.
- Frozen manifest in configs/experiments/phase10_g2_prospective_manifest.json.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from neural_sat_speck.data.attack import (
    AttackInstanceConfig,
    generate_attack_instances,
    write_attack_shard,
)
from neural_sat_speck.data.seed_policy import (
    HISTORICAL_PHASE7_SEED,
    PHASE10_DEVELOPMENT_SEED,
    PHASE10_G2_DEVELOPMENT_SEED,
    PHASE10_G2_PROSPECTIVE_SEED,
    QUARANTINED_PHASE10_SEED,
    RETIRED_BLIND_SEED,
    SeedRole,
    validate_seed_for_role,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "datasets" / "phase10_g2_prospective_r6"

INSTANCE_COUNT = 256
ROUNDS = 6
PAIR_COUNT = 64
SEED = PHASE10_G2_PROSPECTIVE_SEED  # 201001


def main() -> None:
    print("=== Phase 10-G2 Prospective Validation Dataset Generator (N=256, Seed=201001) ===")

    # 1. Enforce Seed Policy
    print(f"Validating seed {SEED} for role {SeedRole.PHASE10_G2_PROSPECTIVE_VALIDATION}...")
    validate_seed_for_role(SEED, SeedRole.PHASE10_G2_PROSPECTIVE_VALIDATION)

    # Reject quarantined, historical, and previous seeds
    prohibited_seeds = (
        HISTORICAL_PHASE7_SEED,
        QUARANTINED_PHASE10_SEED,
        RETIRED_BLIND_SEED,
        PHASE10_DEVELOPMENT_SEED,
        101001,
        PHASE10_G2_DEVELOPMENT_SEED,
    )
    for q_seed in prohibited_seeds:
        try:
            validate_seed_for_role(q_seed, SeedRole.PHASE10_G2_PROSPECTIVE_VALIDATION)
            raise RuntimeError(f"Security breach: seed {q_seed} was not rejected!")
        except ValueError:
            pass
    print("[✓] Seed validation and quarantine checks passed.")

    # 2. Get Repository Commit
    git_commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
        .decode("utf-8")
        .strip()
    )
    print(f"Git Commit: {git_commit}")

    # 3. Configure Dataset (N=256, 6 rounds, 64 pairs)
    config = AttackInstanceConfig(
        instance_count=INSTANCE_COUNT,
        pair_count=PAIR_COUNT,
        rounds=ROUNDS,
        seed=SEED,
        split_role="validation",
        input_difference=(64, 0),
    )

    print(f"Generating {INSTANCE_COUNT} prospective validation attack instances...")
    instances_a = generate_attack_instances(
        config,
        generated_at="2026-09-10T06:00:00+08:00",
        repository_commit=git_commit,
    )

    # 4. Deterministic Replay Check
    print("Verifying deterministic replay...")
    instances_b = generate_attack_instances(
        config,
        generated_at="2026-09-10T06:00:00+08:00",
        repository_commit=git_commit,
    )
    assert (
        instances_a.metadata["data_checksum"] == instances_b.metadata["data_checksum"]
    ), "Deterministic replay failed: checksums do not match!"
    print(f"[✓] Checksum matches: {instances_a.metadata['data_checksum']}")

    # 5. Write Shard
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shard_path = OUTPUT_DIR / "validation.json"
    shard_record = write_attack_shard(
        instances_a,
        shard_path,
        dataset_id="speck32_64_attack_phase10_g2_prospective_r6",
    )
    print(f"[✓] Successfully wrote validation shard to {shard_path}")

    # Compute SHA256
    with open(shard_path, "rb") as f:
        shard_sha256 = hashlib.sha256(f.read()).hexdigest()
    print(f"Shard SHA256: {shard_sha256}")

    # 6. Write Dataset Manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_content = {
        "dataset_id": "speck32_64_attack_phase10_g2_prospective_r6",
        "description": "Locked Phase 10-G2 prospective validation family for SPECK32/64 reduced to 6 rounds.",
        "generated_at": "2026-09-10T06:00:00+08:00",
        "git_commit": git_commit,
        "seed": SEED,
        "seed_role": SeedRole.PHASE10_G2_PROSPECTIVE_VALIDATION,
        "cipher": {
            "name": "SPECK32/64",
            "word_size": 16,
            "rounds": ROUNDS,
            "alpha": 7,
            "beta": 2,
        },
        "attack_parameters": {
            "pairs_per_instance": PAIR_COUNT,
            "input_difference": [64, 0],
            "key_recovery_target": "final_round_subkey",
        },
        "splits": {
            "validation": shard_record,
        },
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_content, f, indent=2)
    print(f"[✓] Saved manifest to {manifest_path}")
    print("Prospective validation data generation complete!")


if __name__ == "__main__":
    main()
