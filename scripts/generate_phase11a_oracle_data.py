#!/usr/bin/env python3
"""Deterministic Generation for Phase 11A Oracle-SAT Evaluation Family (N=64, Seed 300001).

Enforces:
- Seed policy validation (Seed 300001, Role: PHASE11A_ORACLE_EVALUATION).
- Rejection of all quarantined and past seeds (98001, 97001, 99001, 100001, 101001, 200001, 201001).
- 7 rounds of SPECK32/64.
- Verification of deterministic replay before export.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
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
    PHASE11A_ORACLE_SEED,
    QUARANTINED_PHASE10_SEED,
    RETIRED_BLIND_SEED,
    SeedRole,
    validate_seed_for_role,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "datasets" / "phase11a_oracle_r7"

INSTANCE_COUNT = 64
ROUNDS = 7
PAIR_COUNT = 64
SEED = PHASE11A_ORACLE_SEED


def main() -> None:
    print("=== Phase 11A Oracle-SAT Dataset Generator (N=64, Seed=300001, 7 Rounds) ===")

    # 1. Enforce Seed Policy
    print(f"Validating seed {SEED} for role {SeedRole.PHASE11A_ORACLE_EVALUATION}...")
    validate_seed_for_role(SEED, SeedRole.PHASE11A_ORACLE_EVALUATION)

    # Reject past seeds
    prohibited = (
        HISTORICAL_PHASE7_SEED,
        QUARANTINED_PHASE10_SEED,
        RETIRED_BLIND_SEED,
        PHASE10_DEVELOPMENT_SEED,
        101001,
        PHASE10_G2_DEVELOPMENT_SEED,
        PHASE10_G2_PROSPECTIVE_SEED,
    )
    for q_seed in prohibited:
        try:
            validate_seed_for_role(q_seed, SeedRole.PHASE11A_ORACLE_EVALUATION)
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

    # 3. Configure Dataset (N=64, 7 rounds, 64 pairs)
    config = AttackInstanceConfig(
        instance_count=INSTANCE_COUNT,
        pair_count=PAIR_COUNT,
        rounds=ROUNDS,
        seed=SEED,
        split_role="train",
        input_difference=(64, 0),
    )

    print(f"Generating {INSTANCE_COUNT} attack instances at {ROUNDS} rounds...")
    instances_a = generate_attack_instances(
        config,
        generated_at="2026-09-10T08:00:00+08:00",
        repository_commit=git_commit,
    )

    # 4. Deterministic Replay Check
    print("Verifying deterministic replay...")
    instances_b = generate_attack_instances(
        config,
        generated_at="2026-09-10T08:00:00+08:00",
        repository_commit=git_commit,
    )
    assert (
        instances_a.metadata["data_checksum"] == instances_b.metadata["data_checksum"]
    ), "Deterministic replay failed: checksums do not match!"
    print(f"[✓] Checksum matches: {instances_a.metadata['data_checksum']}")

    # 5. Write Shard
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shard_path = OUTPUT_DIR / "oracle_evaluation.json"
    shard_record = write_attack_shard(
        instances_a,
        shard_path,
        dataset_id="speck32_64_attack_phase11a_oracle_r7",
    )
    shard_record["split_role"] = "phase11a_oracle_evaluation"

    # 6. Write Manifest
    manifest = {
        "dataset_id": "speck32_64_attack_phase11a_oracle_r7",
        "data_role": "phase11a_oracle_evaluation",
        "phase": "11A",
        "rounds": ROUNDS,
        "pair_count": PAIR_COUNT,
        "instance_count": INSTANCE_COUNT,
        "seed": SEED,
        "seed_role": SeedRole.PHASE11A_ORACLE_EVALUATION,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "input_difference": [64, 0],
        "pair_construction": "plaintext_difference",
        "key_generation_policy": "one_random_master_key_per_instance",
        "splits": {
            "oracle_evaluation": shard_record,
        },
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[✓] Wrote manifest to {manifest_path}")
    print(f"[✓] Wrote shard to {shard_path} (SHA256: {shard_record['file_sha256']})")
    print("Phase 11A Oracle Dataset Generation Complete!")


if __name__ == "__main__":
    main()
