"""Deterministic Gohr-style synthetic pair generation."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TypeAlias

from neural_sat_speck.speck.cipher import Block, encrypt_block
from neural_sat_speck.speck.key_schedule import SPECK32_64_ROUND_COUNT
from neural_sat_speck.speck.primitives import mask_word

from .metadata import (
    BIT_ORDER,
    FEATURE_DTYPE,
    FEATURE_REPRESENTATION,
    FEATURE_WIDTH,
    GENERATOR_VERSION,
    KEY_GENERATION_POLICY,
    WORD_ORDER,
    DatasetMetadata,
)

WordPair: TypeAlias = tuple[int, int]
CiphertextPair: TypeAlias = tuple[Block, Block]
PlaintextPair: TypeAlias = tuple[Block, Block]
FeatureVector: TypeAlias = tuple[int, ...]

DEFAULT_INPUT_DIFFERENCE: WordPair = (0x0040, 0x0000)
SPLIT_ROLES = frozenset({"train", "validation", "test"})


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for one balanced deterministic dataset."""

    sample_count: int
    rounds: int
    seed: int
    split_role: str
    input_difference: WordPair = DEFAULT_INPUT_DIFFERENCE
    key_generation_policy: str = KEY_GENERATION_POLICY

    def __post_init__(self) -> None:
        if isinstance(self.sample_count, bool) or not isinstance(
            self.sample_count, int
        ):
            raise TypeError("sample_count must be an integer")
        if self.sample_count <= 0 or self.sample_count % 2:
            raise ValueError("sample_count must be a positive even integer")

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
        if self.key_generation_policy != KEY_GENERATION_POLICY:
            raise ValueError(
                "only the verified independent per-sample key policy is "
                "supported"
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


MasterKey: TypeAlias = tuple[int, int, int, int]


@dataclass(frozen=True)
class GeneratedDataset:
    """Generated features, labels, optional debug pairs, and metadata."""

    features: tuple[FeatureVector, ...]
    labels: tuple[int, ...]
    metadata: DatasetMetadata
    plaintext_pairs: tuple[PlaintextPair, ...] | None = None
    ciphertext_pairs: tuple[CiphertextPair, ...] | None = None
    master_keys: tuple[MasterKey, ...] | None = None


def _normalise_word_pair(block: Sequence[int]) -> WordPair:
    try:
        words = tuple(block)
    except TypeError as error:
        raise TypeError("block must be a sequence of two integers") from error
    if len(words) != 2:
        raise ValueError("block must contain exactly two words")
    return mask_word(words[0]), mask_word(words[1])


def word_to_bits_msb(word: int) -> tuple[int, ...]:
    """Encode one 16-bit word from its most-significant bit to its least."""

    value = mask_word(word)
    return tuple((value >> bit) & 1 for bit in range(15, -1, -1))


def encode_ciphertext_pair(
    first_ciphertext: Sequence[int],
    second_ciphertext: Sequence[int],
) -> FeatureVector:
    """Encode a ciphertext pair in the selected 64-bit input order."""

    first_x, first_y = _normalise_word_pair(first_ciphertext)
    second_x, second_y = _normalise_word_pair(second_ciphertext)
    return (
        word_to_bits_msb(first_x)
        + word_to_bits_msb(first_y)
        + word_to_bits_msb(second_x)
        + word_to_bits_msb(second_y)
    )


def _random_word(rng: random.Random) -> int:
    return rng.randrange(1 << 16)


def _random_block(rng: random.Random) -> Block:
    return _random_word(rng), _random_word(rng)


def _random_master_key(rng: random.Random) -> tuple[int, int, int, int]:
    return (
        _random_word(rng),
        _random_word(rng),
        _random_word(rng),
        _random_word(rng),
    )


def dataset_content_checksum(
    features: Sequence[FeatureVector],
    labels: Sequence[int],
) -> str:
    """Hash labels and binary features in dataset order."""

    digest = hashlib.sha256()
    for label, feature in zip(labels, features, strict=True):
        digest.update(bytes((label,)))
        digest.update(bytes(feature))
    return digest.hexdigest()


def validate_split_configs(configs: Sequence[DatasetConfig]) -> None:
    """Validate train/validation/test separation rules."""

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
        if config.key_generation_policy != reference.key_generation_policy:
            raise ValueError(
                "split configurations must use the same key policy"
            )


def generate_dataset(
    config: DatasetConfig,
    *,
    include_debug_pairs: bool = False,
    include_master_keys: bool = False,
    generated_at: str | None = None,
    repository_commit: str | None = None,
) -> GeneratedDataset:
    """Generate a balanced, deterministic real-versus-random dataset.

    Positive samples use P1 = P0 XOR input_difference. Negative samples use a
    fresh random P1. Each sample has an independently generated master key.
    Samples are shuffled with the same local seeded RNG after construction.
    """

    rng = random.Random(config.seed)
    half_count = config.sample_count // 2
    examples: list[
        tuple[int, FeatureVector, PlaintextPair, CiphertextPair, MasterKey]
    ] = []

    for label in (1, 0):
        for _ in range(half_count):
            master_key = _random_master_key(rng)
            first_plaintext = _random_block(rng)
            if label == 1:
                second_plaintext = (
                    mask_word(first_plaintext[0] ^ config.input_difference[0]),
                    mask_word(first_plaintext[1] ^ config.input_difference[1]),
                )
            else:
                second_plaintext = _random_block(rng)

            first_ciphertext = encrypt_block(
                first_plaintext,
                master_key,
                config.rounds,
            )
            second_ciphertext = encrypt_block(
                second_plaintext,
                master_key,
                config.rounds,
            )
            ciphertext_pair = (first_ciphertext, second_ciphertext)
            examples.append(
                (
                    label,
                    encode_ciphertext_pair(*ciphertext_pair),
                    (first_plaintext, second_plaintext),
                    ciphertext_pair,
                    master_key,
                )
            )

    order = list(range(config.sample_count))
    rng.shuffle(order)
    ordered_examples = [examples[index] for index in order]
    ordered_features = tuple(example[1] for example in ordered_examples)
    ordered_labels = tuple(example[0] for example in ordered_examples)

    timestamp = generated_at or datetime.now(UTC).isoformat()
    metadata = DatasetMetadata(
        generator_version=GENERATOR_VERSION,
        repository_commit=repository_commit,
        generated_at=timestamp,
        sample_count=config.sample_count,
        class_balance=(half_count, half_count),
        rounds=config.rounds,
        input_difference=config.input_difference,
        key_generation_policy=config.key_generation_policy,
        seed=config.seed,
        split_role=config.split_role,
        word_order=WORD_ORDER,
        bit_order=BIT_ORDER,
        feature_representation=FEATURE_REPRESENTATION,
        feature_dtype=FEATURE_DTYPE,
        feature_shape=(config.sample_count, FEATURE_WIDTH),
        data_checksum=dataset_content_checksum(ordered_features, ordered_labels),
    )

    return GeneratedDataset(
        features=ordered_features,
        labels=ordered_labels,
        metadata=metadata,
        plaintext_pairs=(
            tuple(example[2] for example in ordered_examples)
            if include_debug_pairs
            else None
        ),
        ciphertext_pairs=(
            tuple(example[3] for example in ordered_examples)
            if include_debug_pairs
            else None
        ),
        master_keys=(
            tuple(example[4] for example in ordered_examples)
            if include_master_keys
            else None
        ),
    )


def assert_disjoint_master_keys(datasets: Sequence[GeneratedDataset]) -> None:
    """Require collected master keys from separate splits to be disjoint."""

    seen: set[MasterKey] = set()
    for dataset in datasets:
        if dataset.master_keys is None:
            raise ValueError("master keys were not collected")
        for master_key in dataset.master_keys:
            if master_key in seen:
                raise ValueError(
                    "train, validation, and test master keys must be disjoint"
                )
            seen.add(master_key)
