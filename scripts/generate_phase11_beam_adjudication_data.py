#!/usr/bin/env python3
"""Deterministic Generation for Phase 11 Beam Adjudication Development Cohort (N=64, 7 Rounds).

Evidence Class: DEVELOPMENT.
Supervisor Authorization: Decision D-030 (2026-09-19).

Enforces:
- Seed drawn exactly once from OS CSPRNG (secrets.SystemRandom).
- Pre-generation rejection of all existing opened, retired, quarantined, or protected seeds.
- Pre-generation recording of the drawn seed in the development manifest.
- 7 rounds of SPECK32/64 with 64 differential pairs per instance (128 observations).
- Plaintext input difference (0x0040, 0x0000) = (64, 0).
- Deterministic replay verification before saving.
- Permanent retirement of the generated development dataset.
"""

from __future__ import annotations

import json
import secrets
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from neural_sat_speck.data.attack import (  # noqa: E402
    AttackInstanceConfig,
    generate_attack_instances,
    write_attack_shard,
)
from neural_sat_speck.data.seed_policy import (  # noqa: E402
    CLI_DEMO_SEED,
    PROTECTED_RESEARCH_SEEDS,
)
from neural_sat_speck.protocol.manifest_validator import (  # noqa: E402
    RETIRED_OR_OPENED_SEEDS,
)

OUTPUT_DIR: Final[Path] = REPO_ROOT / "datasets" / "phase11_beam_adjudication_r7"
DATASET_ID: Final[str] = "speck32_64_attack_phase11_beam_adjudication_r7"
INSTANCE_COUNT: Final[int] = 64
ROUNDS: Final[int] = 7
PAIR_COUNT: Final[int] = 64
INPUT_DIFFERENCE: Final[tuple[int, int]] = (64, 0)

ALL_PROHIBITED_SEEDS: Final[frozenset[int]] = frozenset(
    PROTECTED_RESEARCH_SEEDS
    | RETIRED_OR_OPENED_SEEDS
    | {CLI_DEMO_SEED, 101001, 201001, 300001, 400001, 400002}
)


def draw_csprng_seed() -> int:
    """Draw one safe integer seed from OS CSPRNG, rejecting all prohibited seeds."""
    sys_random = secrets.SystemRandom()
    for _ in range(1000):
        # Draw a 30-bit positive integer
        candidate = sys_random.randint(100_000, 999_998)
        if candidate not in ALL_PROHIBITED_SEEDS:
            return candidate
    raise RuntimeError("Failed to draw a non-prohibited seed from CSPRNG.")


def main() -> int:
    print("=== Phase 11 Beam Adjudication Dataset Generator (N=64, 7 Rounds) ===")
    print("Evidence class: DEVELOPMENT (Decision D-030)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_DIR / "manifest.json"
    shard_path = OUTPUT_DIR / "attack_instances.json"

    # Get repository commit
    git_commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
        .decode("utf-8")
        .strip()
    )

    # If manifest already has a seed recorded, preserve it
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        seed = manifest["seed"]
        print(f"[✓] Using recorded CSPRNG seed: {seed}")
        timestamp = manifest.get("generated_at", datetime.now(UTC).isoformat())
        pre_manifest = manifest
    else:
        # 1. Draw one fresh seed from OS CSPRNG
        seed = draw_csprng_seed()
        assert seed not in ALL_PROHIBITED_SEEDS, f"Drawn seed {seed} violates quarantine policy!"
        print(f"[✓] Drawn fresh CSPRNG seed: {seed}")
        timestamp = datetime.now(UTC).isoformat()
        pre_manifest = {
            "dataset_id": DATASET_ID,
            "evidence_class": "DEVELOPMENT",
            "protocol_status": "OPENED_PERMANENTLY_RETIRED",
            "data_role": "development_beam_adjudication",
            "rounds": ROUNDS,
            "pair_count": PAIR_COUNT,
            "instance_count": INSTANCE_COUNT,
            "seed": seed,
            "seed_source": "os.urandom / secrets.SystemRandom",
            "generated_at": timestamp,
            "git_commit": git_commit,
            "input_difference": list(INPUT_DIFFERENCE),
            "pair_construction": "plaintext_difference",
            "key_generation_policy": "one_random_master_key_per_instance",
            "supervisor_authorization": "Decision D-030 (2026-09-19) - Development-Only Beam Adjudication",
            "status": "OPENED_PERMANENTLY_RETIRED",
            "note": "Development cohort only. Never to be cited as prospective or publication evidence.",
            "splits": {},
        }
        manifest_path.write_text(json.dumps(pre_manifest, indent=2) + "\n", encoding="utf-8")
        print(f"[✓] Pre-recorded seed {seed} in manifest {manifest_path}")

    # 4. Configure dataset (split_role='train' for AttackInstanceConfig, marked 'development' in shard)
    config = AttackInstanceConfig(
        instance_count=INSTANCE_COUNT,
        pair_count=PAIR_COUNT,
        rounds=ROUNDS,
        seed=seed,
        split_role="train",
        input_difference=INPUT_DIFFERENCE,
    )

    print(f"Generating {INSTANCE_COUNT} attack instances (seed={seed}, rounds={ROUNDS}, pairs={PAIR_COUNT})...")
    instances_a = generate_attack_instances(
        config,
        generated_at=timestamp,
        repository_commit=git_commit,
    )

    # 5. Deterministic replay verification
    print("Verifying deterministic replay...")
    instances_b = generate_attack_instances(
        config,
        generated_at=timestamp,
        repository_commit=git_commit,
    )
    assert (
        instances_a.metadata["data_checksum"] == instances_b.metadata["data_checksum"]
    ), "Deterministic replay failed: checksums do not match!"
    print(f"[✓] Checksum matches: {instances_a.metadata['data_checksum']}")

    # 6. Write shard
    shard_record = write_attack_shard(
        instances_a,
        shard_path,
        dataset_id=DATASET_ID,
    )
    shard_record["split_role"] = "development"
    print(f"[✓] Wrote attack shard to {shard_path} (SHA256: {shard_record['file_sha256']})")

    # 7. Finalize manifest with file hash
    pre_manifest["splits"]["attack_instances"] = shard_record
    manifest_path.write_text(json.dumps(pre_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[✓] Finalized manifest at {manifest_path}")
    print(f"Generation complete! Seed: {seed}, SHA256: {shard_record['file_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
