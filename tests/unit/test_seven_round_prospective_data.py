"""Unit tests for seven-round prospective dataset generation and governance (D-031)."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from generate_seven_round_prospective_data import (  # noqa: E402
    ALL_PROHIBITED_SEEDS,
    DATASET_ID,
    INPUT_DIFFERENCE,
    PAIR_COUNT,
    ROUNDS,
    compute_generation_config_hash,
    draw_csprng_seed,
)

from neural_sat_speck.data.seed_policy import (  # noqa: E402
    SeedRole,
    validate_seed_for_role,
)


def test_draw_csprng_seed_rejection_and_policy() -> None:
    """draw_csprng_seed must return an integer satisfying strict prospective role validation."""
    for _ in range(10):
        seed = draw_csprng_seed()
        assert isinstance(seed, int)
        assert 100_000 <= seed <= 999_998
        assert seed not in ALL_PROHIBITED_SEEDS
        # Centralized policy must accept the drawn seed for prospective validation
        validate_seed_for_role(seed, SeedRole.PROSPECTIVE_VALIDATION)


def test_generation_config_hash_invariance() -> None:
    """Generation config hash must be deterministic and sensitive to parameters."""
    h1 = compute_generation_config_hash(555555)
    h2 = compute_generation_config_hash(555555)
    h3 = compute_generation_config_hash(555556)
    assert len(h1) == 64
    assert h1 == h2
    assert h1 != h3


def test_prohibited_seeds_contain_all_critical_seeds() -> None:
    """ALL_PROHIBITED_SEEDS must contain historical, opened, retired, development, and adjudication seeds."""
    mandatory_rejected = {
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
        999999,
    }
    for s in mandatory_rejected:
        assert s in ALL_PROHIBITED_SEEDS, f"Seed {s} missing from ALL_PROHIBITED_SEEDS"


def test_small_cohort_generation_and_deterministic_replay(tmp_path: Path) -> None:
    """Verify deterministic replay and schema invariants on an isolated cohort."""
    from neural_sat_speck.data.attack import (
        AttackInstanceConfig,
        generate_attack_instances,
        load_attack_shard,
        write_attack_shard,
    )

    test_seed = 888888
    config = AttackInstanceConfig(
        instance_count=4,
        pair_count=PAIR_COUNT,
        rounds=ROUNDS,
        seed=test_seed,
        split_role="validation",
        input_difference=INPUT_DIFFERENCE,
    )

    inst_a = generate_attack_instances(config, generated_at="2026-09-21T00:00:00Z")
    inst_b = generate_attack_instances(config, generated_at="2026-09-21T00:00:00Z")

    assert inst_a.metadata["data_checksum"] == inst_b.metadata["data_checksum"]

    shard_path = tmp_path / "test_shard.json"
    write_attack_shard(inst_a, shard_path, dataset_id=DATASET_ID)

    shard = load_attack_shard(shard_path, expected_dataset_id=DATASET_ID)
    assert len(shard.instances) == 4
    for inst in shard.instances:
        assert len(inst.pairs) == 64
        # Check input difference on each pair
        for pair in inst.pairs:
            diff_l = pair.plaintext0[0] ^ pair.plaintext1[0]
            diff_r = pair.plaintext0[1] ^ pair.plaintext1[1]
            assert (diff_l, diff_r) == INPUT_DIFFERENCE
