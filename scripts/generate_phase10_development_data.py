#!/usr/bin/env python3
"""Deterministic generation and export for Phase 10 Development Attack Family (N=192).

Enforces:
- Seed policy validation (Seed 100001, Role: PHASE10_DEVELOPMENT).
- Rejection of quarantined seeds (98001, 97001, 99001).
- Complete metadata tracking: git commit, protocol SHA256, config template SHA256.
- Verification of deterministic replay.
- Verification that locked prospective validation (N=64) is NOT generated.
"""

from __future__ import annotations

import hashlib
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
    PHASE10_DEVELOPMENT_SEED,
    SeedRole,
    validate_seed_for_role,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "datasets" / "phase10_development_r6"
PROTOCOL_FILE = REPO_ROOT / "docs" / "PHASE10_G1_R1_PROSPECTIVE_PROTOCOL.md"
TEMPLATE_FILE = REPO_ROOT / "configs" / "experiments" / "phase10_protocol_template.json"


def main() -> None:
    print("=== Phase 10 Development Dataset Generator (N=192) ===")

    # 1. Enforce Seed Policy
    seed = PHASE10_DEVELOPMENT_SEED
    print(f"Validating seed {seed} for role {SeedRole.PHASE10_DEVELOPMENT}...")
    validate_seed_for_role(seed, SeedRole.PHASE10_DEVELOPMENT)

    # 2. Get Repository Commit & Protocol Hashes
    git_commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
        .decode("utf-8")
        .strip()
    )
    protocol_hash = hashlib.sha256(PROTOCOL_FILE.read_bytes()).hexdigest()
    template_hash = hashlib.sha256(TEMPLATE_FILE.read_bytes()).hexdigest()

    print(f"Git Commit: {git_commit}")
    print(f"Protocol SHA256: {protocol_hash}")
    print(f"Template SHA256: {template_hash}")

    # 3. Configure Dataset (N=192, 6 rounds, 64 pairs)
    config = AttackInstanceConfig(
        instance_count=192,
        pair_count=64,
        rounds=6,
        seed=seed,
        split_role="train",  # internal format compatibility
        input_difference=(64, 0),
    )

    print("Generating 192 attack instances...")
    instances_a = generate_attack_instances(
        config,
        generated_at="2026-09-03T20:00:00+08:00",
        repository_commit=git_commit,
    )

    # 4. Deterministic Replay Check
    print("Verifying deterministic replay...")
    instances_b = generate_attack_instances(
        config,
        generated_at="2026-09-03T20:00:00+08:00",
        repository_commit=git_commit,
    )
    assert (
        instances_a.metadata["data_checksum"] == instances_b.metadata["data_checksum"]
    ), "Deterministic replay failed: checksums do not match!"
    print(f"[✓] Checksum matches: {instances_a.metadata['data_checksum']}")

    # 5. Write Shard
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shard_path = OUTPUT_DIR / "development.json"
    shard_record = write_attack_shard(
        instances_a,
        shard_path,
        dataset_id="speck32_64_attack_phase10_development_r6",
    )
    # Update split_role in record to accurately describe Phase 10 development role
    shard_record["split_role"] = "phase10_development"

    # 6. Write Manifest
    manifest = {
        "dataset_id": "speck32_64_attack_phase10_development_r6",
        "data_role": "phase10_development",
        "phase": 10,
        "rounds": 6,
        "pair_count": 64,
        "instance_count": 192,
        "seed": seed,
        "seed_role": SeedRole.PHASE10_DEVELOPMENT,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "protocol_document": "docs/PHASE10_G1_R1_PROSPECTIVE_PROTOCOL.md",
        "protocol_sha256": protocol_hash,
        "config_template_sha256": template_hash,
        "input_difference": [64, 0],
        "pair_construction": "plaintext_difference",
        "key_generation_policy": "one_random_master_key_per_instance",
        "splits": {
            "development": shard_record,
        },
        "locked_validation_opened": False,
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[✓] Wrote manifest to {manifest_path}")
    print(f"[✓] Wrote development shard to {shard_path}")
    print(f"File SHA256: {shard_record['file_sha256']}")
    print("Phase 10 Development Dataset Generation Complete!")


if __name__ == "__main__":
    main()
