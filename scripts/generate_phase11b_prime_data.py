#!/usr/bin/env python3
"""Deterministic Generation of the Phase 11B' Six-Round Stage-1 Pair-Group Family.

Development-family data only.  This family trains the six-round stage-1 ranker of
the (unfrozen) Phase 11B' hierarchical ranking protocol; it is **not**
prospective validation and **not** blind confirmation, and it opens no locked
data.  It mirrors the frozen Phase 6A five-round family exactly except for the
round count: 64 pairs per group, ``(0x0040, 0x0000)`` plaintext difference, one
random master key per group, task ``group_related_vs_random``.

Enforces:
- Seed policy (Role: PHASE11B_PRIME_STAGE1_DEVELOPMENT; owned seeds 400001 train
  and 400002 validation).
- Rejection of every historical, quarantined, retired, or already-opened seed.
- Six rounds (:data:`PAIR_GROUP_ROUNDS_SIX`) via the explicit opt-in
  :class:`SixRoundPairGroupConfig`; the frozen five-round path is untouched.
- Deterministic replay: the whole family is regenerated and its per-split data
  checksums compared against the written manifest before the run is accepted.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Final

from neural_sat_speck.data.pair_groups import (
    PAIR_GROUP_ROUNDS_SIX,
    PAIR_GROUP_SIX_ROUND_DATASET_ID,
    PAIR_GROUP_SIZE,
    GeneratedPairGroups,
    SixRoundPairGroupConfig,
    assert_disjoint_pair_group_keys,
    export_pair_group_benchmark,
    generate_pair_groups,
    verify_pair_group_benchmark,
)
from neural_sat_speck.data.seed_policy import (
    HISTORICAL_PHASE7_SEED,
    HISTORICAL_SEEDS,
    PHASE10_DEVELOPMENT_SEED,
    PHASE10_G2_DEVELOPMENT_SEED,
    PHASE10_G2_PROSPECTIVE_SEED,
    PHASE11A_ORACLE_SEED,
    PHASE11B_PRIME_SIX_ROUND_TRAIN_SEED,
    PHASE11B_PRIME_SIX_ROUND_VALIDATION_SEED,
    PROTECTED_RESEARCH_SEEDS,
    QUARANTINED_PHASE10_SEED,
    RETIRED_BLIND_SEED,
    SeedRole,
    validate_seed_for_role,
)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR: Final[Path] = (
    REPO_ROOT / "datasets" / "phase11b_prime_pair_group_r6_g64"
)
#: The frozen Phase 6A five-round family must never be written by this script.
FROZEN_FIVE_ROUND_DIR: Final[Path] = REPO_ROOT / "datasets" / "phase6a_pair_group_r5_g64"

DATASET_ID: Final[str] = PAIR_GROUP_SIX_ROUND_DATASET_ID
ROUNDS: Final[int] = PAIR_GROUP_ROUNDS_SIX
GROUP_SIZE: Final[int] = PAIR_GROUP_SIZE
TRAIN_GROUP_COUNT: Final[int] = 4096
VALIDATION_GROUP_COUNT: Final[int] = 1024
TRAIN_SEED: Final[int] = PHASE11B_PRIME_SIX_ROUND_TRAIN_SEED
VALIDATION_SEED: Final[int] = PHASE11B_PRIME_SIX_ROUND_VALIDATION_SEED
SEED_ROLE: Final[SeedRole] = SeedRole.PHASE11B_PRIME_STAGE1_DEVELOPMENT
#: Pinned so that the whole family replays byte-for-byte.
DEFAULT_GENERATED_AT: Final[str] = "2026-09-10T21:00:00+08:00"

#: Seeds that must be rejected by this role, in addition to HISTORICAL_SEEDS.
PROHIBITED_OPENED_SEEDS: Final[tuple[int, ...]] = (
    HISTORICAL_PHASE7_SEED,
    QUARANTINED_PHASE10_SEED,
    RETIRED_BLIND_SEED,
    PHASE10_DEVELOPMENT_SEED,
    101001,
    PHASE10_G2_DEVELOPMENT_SEED,
    PHASE10_G2_PROSPECTIVE_SEED,
    PHASE11A_ORACLE_SEED,
    # Frozen Phase 6A five-round family seeds.
    61001,
    62001,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the Phase 11B' six-round stage-1 pair-group development "
            "family (development data only; no validation or confirmation role)."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--train-groups",
        type=int,
        default=TRAIN_GROUP_COUNT,
        help=f"number of train groups (default: {TRAIN_GROUP_COUNT})",
    )
    parser.add_argument(
        "--validation-groups",
        type=int,
        default=VALIDATION_GROUP_COUNT,
        help=f"number of validation groups (default: {VALIDATION_GROUP_COUNT})",
    )
    parser.add_argument(
        "--generated-at",
        type=str,
        default=DEFAULT_GENERATED_AT,
        help=(
            "timestamp recorded in shard and manifest metadata "
            f"(default, pinned for replay: {DEFAULT_GENERATED_AT})"
        ),
    )
    return parser.parse_args()


def _repository_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _validate_seed_policy() -> None:
    """Validate the owned seeds and prove the prohibited ones are rejected."""

    print(f"Validating seeds for role {SEED_ROLE}...")
    for seed in (TRAIN_SEED, VALIDATION_SEED):
        validate_seed_for_role(seed, SEED_ROLE)

    prohibited = tuple(dict.fromkeys((*PROHIBITED_OPENED_SEEDS, *sorted(HISTORICAL_SEEDS))))
    for seed in prohibited:
        if seed in (TRAIN_SEED, VALIDATION_SEED):
            raise SystemExit(
                f"Governance breach: owned seed {seed} is listed as prohibited."
            )
        try:
            validate_seed_for_role(seed, SEED_ROLE)
        except ValueError:
            continue
        raise SystemExit(f"Security breach: seed {seed} was not rejected!")
    if TRAIN_SEED in PROTECTED_RESEARCH_SEEDS or VALIDATION_SEED in PROTECTED_RESEARCH_SEEDS:
        print(
            "[i] Both owned seeds are registered in PROTECTED_RESEARCH_SEEDS, so no "
            "other role can adopt them; this role accepts them explicitly."
        )
    print(f"[✓] Seed validation and quarantine checks passed ({len(prohibited)} seeds rejected).")


def _resolve_output_dir(value: Path) -> Path:
    output_dir = (value if value.is_absolute() else REPO_ROOT / value).resolve()
    frozen = FROZEN_FIVE_ROUND_DIR.resolve()
    if output_dir == frozen or frozen in output_dir.parents:
        raise SystemExit(
            f"refusing to write into the frozen five-round family at {frozen}"
        )
    return output_dir


def _configs(
    train_groups: int,
    validation_groups: int,
) -> tuple[SixRoundPairGroupConfig, SixRoundPairGroupConfig]:
    try:
        return (
            SixRoundPairGroupConfig(
                group_count=train_groups,
                group_size=GROUP_SIZE,
                rounds=ROUNDS,
                seed=TRAIN_SEED,
                split_role="train",
            ),
            SixRoundPairGroupConfig(
                group_count=validation_groups,
                group_size=GROUP_SIZE,
                rounds=ROUNDS,
                seed=VALIDATION_SEED,
                split_role="validation",
            ),
        )
    except (TypeError, ValueError) as error:
        raise SystemExit(f"invalid six-round pair-group configuration: {error}") from error


def _replay_split(
    config: SixRoundPairGroupConfig,
    *,
    generated_at: str,
    repository_commit: str | None,
) -> GeneratedPairGroups:
    return generate_pair_groups(
        config,
        generated_at=generated_at,
        repository_commit=repository_commit,
    )


def main() -> None:
    args = _parse_args()
    print("=== Phase 11B' Six-Round Stage-1 Pair-Group Generator (development family) ===")

    _validate_seed_policy()

    repository_commit = _repository_commit()
    print(f"Git commit: {repository_commit}")

    output_dir = _resolve_output_dir(args.output_dir)
    configs = _configs(args.train_groups, args.validation_groups)
    for config in configs:
        print(
            f"  {config.split_role}: {config.group_count} groups x {config.group_size} pairs "
            f"x {config.rounds} rounds, seed {config.seed}"
        )

    manifest = export_pair_group_benchmark(
        configs,
        output_dir,
        dataset_id=DATASET_ID,
        generated_at=args.generated_at,
        repository_commit=repository_commit,
        rounds=ROUNDS,
    )
    split_records = manifest["splits"]
    assert isinstance(split_records, dict)
    print(f"[✓] Wrote shards and manifest to {output_dir}")

    # Deterministic replay: regenerate the whole family from its seeds and
    # require every per-split data checksum to match the written manifest.
    print("Verifying deterministic replay (full regeneration)...")
    replayed = [
        _replay_split(
            config,
            generated_at=args.generated_at,
            repository_commit=repository_commit,
        )
        for config in configs
    ]
    assert_disjoint_pair_group_keys(replayed)
    for dataset in replayed:
        role = str(dataset.metadata["split_role"])
        record = split_records[role]
        checksum = dataset.metadata["data_checksum"]
        if checksum != record["data_checksum"]:
            raise SystemExit(
                f"Deterministic replay failed for {role}: {checksum} != "
                f"{record['data_checksum']}"
            )
        print(f"[✓] {role} replay checksum: {checksum}")

    verified = verify_pair_group_benchmark(
        output_dir / "manifest.json",
        expected_dataset_id=DATASET_ID,
        expected_rounds=ROUNDS,
    )
    if verified["rounds"] != ROUNDS:
        raise SystemExit("manifest round count is not six")
    if verified["generated_at"] != args.generated_at:
        raise SystemExit("manifest generated_at does not match the requested timestamp")
    print(
        f"[✓] Verified {verified['dataset_id']} "
        f"({verified['rounds']} rounds, generated_at {verified['generated_at']})"
    )
    for role, record in verified["splits"].items():
        print(
            f"  {role}: {record['group_count']} groups, seed {record['seed']}, "
            f"role {record['split_role']}, data_checksum {record['data_checksum']}, "
            f"file_sha256 {record['file_sha256']}"
        )
    print(f"Manifest: {output_dir / 'manifest.json'}")
    print("Phase 11B' six-round stage-1 development family generation complete.")


if __name__ == "__main__":
    sys.exit(main())
