"""Deterministic single-key attack-instance generation and Dataset v1 export."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias

from neural_sat_speck.speck.cipher import Block, encrypt_block
from neural_sat_speck.speck.key_schedule import SPECK32_64_ROUND_COUNT, expand_key
from neural_sat_speck.speck.primitives import mask_word

from .format import (
    ATTACK_DATASET_ID,
    ATTACK_MANIFEST_SCHEMA,
    ATTACK_SHARD_SCHEMA,
    ATTACK_TASK,
    CIPHER_NAME,
    DATASET_FORMAT_VERSION,
    DATASET_LICENSE,
    PLAINTEXT_DIFFERENCE_POLICY,
    canonical_json,
    hex_to_word,
    hex_to_words,
    sha256_bytes,
    word_to_hex,
    words_to_hex,
)
from .generator import DEFAULT_INPUT_DIFFERENCE, SPLIT_ROLES, WordPair
from .metadata import BIT_ORDER, GENERATOR_VERSION, WORD_ORDER

MasterKey: TypeAlias = tuple[int, int, int, int]


def _random_word(rng: random.Random) -> int:
    return rng.randrange(1 << 16)


def _random_block(rng: random.Random) -> Block:
    return _random_word(rng), _random_word(rng)


def _random_master_key(rng: random.Random) -> MasterKey:
    return (
        _random_word(rng),
        _random_word(rng),
        _random_word(rng),
        _random_word(rng),
    )


def _attack_checksum(instances: Sequence[AttackInstance]) -> str:
    """Hash instance identifiers, keys, and word-level pairs in order."""

    digest = hashlib.sha256()
    for instance in instances:
        digest.update(instance.instance_id.encode("utf-8"))
        for word in (*instance.master_key, instance.final_round_subkey):
            digest.update(word.to_bytes(2, "big"))
        for pair in instance.pairs:
            for block in (
                pair.plaintext0,
                pair.plaintext1,
                pair.ciphertext0,
                pair.ciphertext1,
            ):
                for word in block:
                    digest.update(word.to_bytes(2, "big"))
    return digest.hexdigest()


@dataclass(frozen=True)
class AttackInstanceConfig:
    """Configuration for one split of single-key attack instances."""

    instance_count: int
    pair_count: int
    rounds: int
    seed: int
    split_role: str
    input_difference: WordPair = DEFAULT_INPUT_DIFFERENCE
    pair_construction: str = PLAINTEXT_DIFFERENCE_POLICY

    def __post_init__(self) -> None:
        if isinstance(self.instance_count, bool) or not isinstance(
            self.instance_count, int
        ):
            raise TypeError("instance_count must be an integer")
        if self.instance_count <= 0:
            raise ValueError("instance_count must be a positive integer")

        if isinstance(self.pair_count, bool) or not isinstance(self.pair_count, int):
            raise TypeError("pair_count must be an integer")
        if self.pair_count <= 0:
            raise ValueError("pair_count must be a positive integer")

        if isinstance(self.rounds, bool) or not isinstance(self.rounds, int):
            raise TypeError("rounds must be an integer")
        if not 1 <= self.rounds <= SPECK32_64_ROUND_COUNT:
            raise ValueError(
                f"rounds must be between 1 and {SPECK32_64_ROUND_COUNT}"
            )

        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")

        if self.split_role not in SPLIT_ROLES:
            raise ValueError(
                f"split_role must be one of {sorted(SPLIT_ROLES)}"
            )
        if self.pair_construction != PLAINTEXT_DIFFERENCE_POLICY:
            raise ValueError(
                "only the documented plaintext-difference pair construction "
                "is supported in Dataset v1"
            )

        try:
            difference = tuple(self.input_difference)
        except TypeError as error:
            raise TypeError(
                "input_difference must be a sequence of two integers"
            ) from error
        if len(difference) != 2:
            raise ValueError("input_difference must contain two words")
        object.__setattr__(
            self,
            "input_difference",
            (mask_word(difference[0]), mask_word(difference[1])),
        )


@dataclass(frozen=True)
class AttackPair:
    """One related plaintext/ciphertext pair under a shared master key."""

    plaintext0: Block
    plaintext1: Block
    ciphertext0: Block
    ciphertext1: Block


@dataclass(frozen=True)
class AttackInstance:
    """One secret-key attack instance with several related pairs."""

    instance_id: str
    master_key: MasterKey
    final_round_subkey: int
    pairs: tuple[AttackPair, ...]


@dataclass(frozen=True)
class AttackInstanceSet:
    """A split of attack instances plus machine-readable metadata."""

    instances: tuple[AttackInstance, ...]
    metadata: dict[str, object]


def validate_attack_split_configs(
    configs: Sequence[AttackInstanceConfig],
) -> None:
    """Validate that attack splits use distinct seeds and one protocol."""

    config_tuple = tuple(configs)
    roles = {config.split_role for config in config_tuple}
    if roles != SPLIT_ROLES or len(config_tuple) != len(SPLIT_ROLES):
        raise ValueError(
            "configs must contain exactly one train, validation, and test "
            "configuration"
        )

    seeds = [config.seed for config in config_tuple]
    if len(set(seeds)) != len(seeds):
        raise ValueError("train, validation, and test seeds must be distinct")

    reference = config_tuple[0]
    for config in config_tuple[1:]:
        if config.rounds != reference.rounds:
            raise ValueError("split configurations must use the same rounds")
        if config.input_difference != reference.input_difference:
            raise ValueError(
                "split configurations must use the same input difference"
            )
        if config.pair_construction != reference.pair_construction:
            raise ValueError(
                "split configurations must use the same pair construction"
            )


def generate_attack_instances(
    config: AttackInstanceConfig,
    *,
    generated_at: str | None = None,
    repository_commit: str | None = None,
) -> AttackInstanceSet:
    """Generate related ciphertext pairs that share one secret key per instance.

    Dataset v1 uses the plaintext-difference construction: each pair satisfies
    P1 = P0 XOR input_difference and both blocks are encrypted under the same
    independently sampled master key. This is not Gohr's later first-round
    difference arrangement used in key ranking.
    """

    rng = random.Random(config.seed)
    instances: list[AttackInstance] = []
    for index in range(config.instance_count):
        master_key = _random_master_key(rng)
        final_round_subkey = expand_key(master_key, config.rounds)[-1]
        pairs: list[AttackPair] = []
        for _ in range(config.pair_count):
            plaintext0 = _random_block(rng)
            plaintext1 = (
                mask_word(plaintext0[0] ^ config.input_difference[0]),
                mask_word(plaintext0[1] ^ config.input_difference[1]),
            )
            pairs.append(
                AttackPair(
                    plaintext0=plaintext0,
                    plaintext1=plaintext1,
                    ciphertext0=encrypt_block(
                        plaintext0, master_key, config.rounds
                    ),
                    ciphertext1=encrypt_block(
                        plaintext1, master_key, config.rounds
                    ),
                )
            )
        instances.append(
            AttackInstance(
                instance_id=f"{config.split_role}-{index:04d}",
                master_key=master_key,
                final_round_subkey=final_round_subkey,
                pairs=tuple(pairs),
            )
        )

    timestamp = generated_at or datetime.now(UTC).isoformat()
    metadata = {
        "bit_order": BIT_ORDER,
        "data_checksum": _attack_checksum(instances),
        "generated_at": timestamp,
        "generator_version": GENERATOR_VERSION,
        "input_difference": list(config.input_difference),
        "instance_count": config.instance_count,
        "key_generation_policy": "one_random_master_key_per_instance",
        "pair_construction": config.pair_construction,
        "pair_count": config.pair_count,
        "repository_commit": repository_commit,
        "rounds": config.rounds,
        "seed": config.seed,
        "split_role": config.split_role,
        "word_order": WORD_ORDER,
    }
    return AttackInstanceSet(instances=tuple(instances), metadata=metadata)


def assert_disjoint_attack_keys(sets: Sequence[AttackInstanceSet]) -> None:
    """Require attack-instance master keys from separate splits to be disjoint."""

    seen: set[MasterKey] = set()
    for instance_set in sets:
        for instance in instance_set.instances:
            if instance.master_key in seen:
                raise ValueError(
                    "train, validation, and test attack keys must be disjoint"
                )
            seen.add(instance.master_key)


def attack_instance_document(instance: AttackInstance) -> dict[str, object]:
    """Serialize one attack instance to hexadecimal JSON fields."""

    return {
        "final_round_subkey": word_to_hex(instance.final_round_subkey),
        "instance_id": instance.instance_id,
        "master_key": words_to_hex(instance.master_key),
        "pairs": [
            {
                "ciphertext0": words_to_hex(pair.ciphertext0),
                "ciphertext1": words_to_hex(pair.ciphertext1),
                "plaintext0": words_to_hex(pair.plaintext0),
                "plaintext1": words_to_hex(pair.plaintext1),
            }
            for pair in instance.pairs
        ],
    }


def attack_shard_document(
    instance_set: AttackInstanceSet,
    *,
    dataset_id: str = ATTACK_DATASET_ID,
) -> dict[str, object]:
    """Return the architecture-neutral JSON document for one attack split."""

    return {
        "cipher": CIPHER_NAME,
        "dataset_format_version": DATASET_FORMAT_VERSION,
        "dataset_id": dataset_id,
        "instances": [
            attack_instance_document(instance)
            for instance in instance_set.instances
        ],
        "license": DATASET_LICENSE,
        "metadata": instance_set.metadata,
        "schema": ATTACK_SHARD_SCHEMA,
        "task": ATTACK_TASK,
    }


def attack_shard_bytes(instance_set: AttackInstanceSet, **kwargs: object) -> bytes:
    """Serialize one attack split to canonical UTF-8 JSON bytes."""

    return canonical_json(attack_shard_document(instance_set, **kwargs)).encode(
        "utf-8"
    )


def write_attack_shard(
    instance_set: AttackInstanceSet,
    path: Path,
    **kwargs: object,
) -> dict[str, object]:
    """Write one attack shard and return its manifest record."""

    payload = attack_shard_bytes(instance_set, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    metadata = instance_set.metadata
    return {
        "data_checksum": metadata["data_checksum"],
        "file_sha256": sha256_bytes(payload),
        "instance_count": metadata["instance_count"],
        "pair_count": metadata["pair_count"],
        "path": path.name,
        "seed": metadata["seed"],
        "split_role": metadata["split_role"],
    }


def _parse_attack_instance(document: Mapping[str, Any]) -> AttackInstance:
    pairs = tuple(
        AttackPair(
            plaintext0=(
                hex_to_word(pair["plaintext0"][0]),
                hex_to_word(pair["plaintext0"][1]),
            ),
            plaintext1=(
                hex_to_word(pair["plaintext1"][0]),
                hex_to_word(pair["plaintext1"][1]),
            ),
            ciphertext0=(
                hex_to_word(pair["ciphertext0"][0]),
                hex_to_word(pair["ciphertext0"][1]),
            ),
            ciphertext1=(
                hex_to_word(pair["ciphertext1"][0]),
                hex_to_word(pair["ciphertext1"][1]),
            ),
        )
        for pair in document["pairs"]
    )
    master_key = hex_to_words(document["master_key"])
    if len(master_key) != 4:
        raise ValueError("attack instance master key must contain four words")
    return AttackInstance(
        instance_id=str(document["instance_id"]),
        master_key=(master_key[0], master_key[1], master_key[2], master_key[3]),
        final_round_subkey=hex_to_word(str(document["final_round_subkey"])),
        pairs=pairs,
    )


def load_attack_shard(path: Path) -> AttackInstanceSet:
    """Load an attack shard without any neural-framework dependency."""

    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != ATTACK_SHARD_SCHEMA:
        raise ValueError("attack shard schema is not recognised")
    if document.get("task") != ATTACK_TASK:
        raise ValueError("attack shard task is not single_key_related_pairs")
    instances = tuple(
        _parse_attack_instance(instance) for instance in document["instances"]
    )
    return AttackInstanceSet(
        instances=instances,
        metadata=dict(document["metadata"]),
    )


def export_attack_benchmark(
    configs: Sequence[AttackInstanceConfig],
    output_dir: Path,
    *,
    dataset_id: str = ATTACK_DATASET_ID,
    generated_at: str,
    repository_commit: str | None = None,
) -> dict[str, object]:
    """Generate disjoint attack splits and write shards plus a manifest."""

    validate_attack_split_configs(configs)
    instance_sets = [
        generate_attack_instances(
            config,
            generated_at=generated_at,
            repository_commit=repository_commit,
        )
        for config in configs
    ]
    assert_disjoint_attack_keys(instance_sets)

    output_dir.mkdir(parents=True, exist_ok=True)
    split_records: dict[str, dict[str, object]] = {}
    for instance_set in instance_sets:
        role = str(instance_set.metadata["split_role"])
        shard_path = output_dir / f"{role}.json"
        split_records[role] = write_attack_shard(
            instance_set,
            shard_path,
            dataset_id=dataset_id,
        )

    reference = instance_sets[0].metadata
    manifest = {
        "cipher": CIPHER_NAME,
        "dataset_format_version": DATASET_FORMAT_VERSION,
        "dataset_id": dataset_id,
        "generated_at": generated_at,
        "generator_version": GENERATOR_VERSION,
        "input_difference": reference["input_difference"],
        "license": DATASET_LICENSE,
        "pair_construction": reference["pair_construction"],
        "repository_commit": repository_commit,
        "rounds": reference["rounds"],
        "schema": ATTACK_MANIFEST_SCHEMA,
        "splits": split_records,
        "task": ATTACK_TASK,
        "word_order": WORD_ORDER,
    }
    (output_dir / "manifest.json").write_bytes(
        canonical_json(manifest).encode("utf-8")
    )
    return manifest
