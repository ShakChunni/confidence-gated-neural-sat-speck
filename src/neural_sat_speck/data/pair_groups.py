"""Deterministic same-key ciphertext-pair groups for the Phase 6 study.

This module deliberately defines a new format rather than changing Dataset v1.
The group data are stored in NumPy ``.npz`` shards with an embedded metadata
document.  The arrays are architecture-neutral: consumers receive binary
MSB-first features, labels, and audit-only master-key words.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Final

import numpy as np

from neural_sat_speck.speck.cipher import encrypt_block
from neural_sat_speck.speck.primitives import WORD_MASK, mask_word

from .format import (
    CIPHER_NAME,
    DATASET_FORMAT_VERSION,
    DATASET_LICENSE,
    canonical_json,
    sha256_bytes,
)
from .generator import DEFAULT_INPUT_DIFFERENCE, WordPair
from .metadata import BIT_ORDER, WORD_ORDER

PAIR_GROUP_DATASET_ID: Final[str] = "speck32_64_pair_group_phase6a_r5_g64"
PAIR_GROUP_TASK: Final[str] = "group_related_vs_random"
PAIR_GROUP_SHARD_SCHEMA: Final[str] = "neural_sat_speck.pair_group_shard.v1"
PAIR_GROUP_MANIFEST_SCHEMA: Final[str] = (
    "neural_sat_speck.pair_group_manifest.v1"
)
PAIR_GROUP_GENERATOR_VERSION: Final[str] = "0.1.0"
PAIR_GROUP_SIZE: Final[int] = 64
PAIR_GROUP_ROUNDS: Final[int] = 5

# Six-round stage-1 development family for the unfrozen Phase 11B' hierarchical
# ranking protocol.  This is deliberately a separate constant: the frozen
# five-round dataset id, checksums, and verification path are unchanged, and the
# six-round family is reachable only through an explicit opt-in.
PAIR_GROUP_ROUNDS_SIX: Final[int] = 6
PAIR_GROUP_SIX_ROUND_DATASET_ID: Final[str] = (
    "speck32_64_pair_group_phase11b_prime_r6_g64"
)

#: Round count owned by each known pair-group dataset id.  Loaders use this to
#: resolve their round expectation when a dataset id is supplied; ids outside
#: this registry keep the frozen five-round expectation.
PAIR_GROUP_DATASET_ROUNDS: Final[dict[str, int]] = {
    PAIR_GROUP_DATASET_ID: PAIR_GROUP_ROUNDS,
    PAIR_GROUP_SIX_ROUND_DATASET_ID: PAIR_GROUP_ROUNDS_SIX,
}

PAIR_GROUP_SPLIT_ROLES: tuple[str, str] = ("train", "validation")
PAIR_GROUP_KEY_POLICY: Final[str] = "one_random_master_key_per_group"
PAIR_GROUP_FEATURE_DTYPE: Final[str] = "uint8"
PAIR_GROUP_MASTER_KEY_DTYPE: Final[str] = "uint16"
PAIR_GROUP_PAIR_CONSTRUCTION: Final[str] = (
    "positive_difference_or_independent_random_second_plaintext"
)


def _resolve_expected_rounds(
    expected_dataset_id: str | None,
    expected_rounds: int | None,
) -> int:
    """Resolve the round expectation, falling back to the frozen five rounds.

    An explicit ``expected_rounds`` always wins.  Otherwise the dataset id
    selects its registered round count; unknown or unspecified ids keep the
    frozen five-round expectation, so five-round behaviour is unchanged.
    """

    if expected_rounds is not None:
        return expected_rounds
    if expected_dataset_id is None:
        return PAIR_GROUP_ROUNDS
    return PAIR_GROUP_DATASET_ROUNDS.get(expected_dataset_id, PAIR_GROUP_ROUNDS)


def _validate_word_pair(value: Sequence[int]) -> WordPair:
    if isinstance(value, (str, bytes)):
        raise TypeError("input_difference must be a sequence of two words")
    words = tuple(value)
    if len(words) != 2:
        raise ValueError("input_difference must contain two words")
    if any(
        isinstance(word, bool)
        or not isinstance(word, int)
        or not 0 <= word <= WORD_MASK
        for word in words
    ):
        raise ValueError("input_difference words must be 16-bit integers")
    return mask_word(words[0]), mask_word(words[1])


@dataclass(frozen=True)
class PairGroupConfig:
    """Configuration for one Phase 6 pair-group split.

    This class is frozen at :data:`PAIR_GROUP_ROUNDS` rounds.  Six-round
    configurations must use :class:`SixRoundPairGroupConfig`, which shares the
    same fields, validation, and generator entry point.
    """

    #: Round count this configuration accepts.  Declared as a ``ClassVar`` so
    #: the dataclass fields and the constructor signature are unchanged, and
    #: overridden by :class:`SixRoundPairGroupConfig`.
    required_rounds: ClassVar[int] = PAIR_GROUP_ROUNDS

    #: Dataset id recorded by :func:`generate_pair_groups`; writers may still
    #: override it explicitly.
    default_dataset_id: ClassVar[str] = PAIR_GROUP_DATASET_ID

    group_count: int
    group_size: int
    rounds: int
    seed: int
    split_role: str
    input_difference: WordPair = DEFAULT_INPUT_DIFFERENCE

    def __post_init__(self) -> None:
        for name, value in (
            ("group_count", self.group_count),
            ("group_size", self.group_size),
            ("rounds", self.rounds),
            ("seed", self.seed),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.group_count <= 0 or self.group_count % 2:
            raise ValueError("group_count must be a positive even integer")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.group_size != PAIR_GROUP_SIZE:
            raise ValueError(f"group_size must be exactly {PAIR_GROUP_SIZE}")
        if self.rounds != self.required_rounds:
            raise ValueError(f"rounds must be exactly {self.required_rounds}")
        if self.split_role not in PAIR_GROUP_SPLIT_ROLES:
            raise ValueError(
                f"split_role must be one of {sorted(PAIR_GROUP_SPLIT_ROLES)}"
            )
        difference = _validate_word_pair(self.input_difference)
        if difference != DEFAULT_INPUT_DIFFERENCE:
            raise ValueError(
                "Phase 6 pair groups require the verified difference "
                "(0x0040, 0x0000)"
            )
        object.__setattr__(self, "input_difference", difference)


@dataclass(frozen=True)
class SixRoundPairGroupConfig(PairGroupConfig):
    """Configuration for the six-round stage-1 pair-group development family.

    Identical to :class:`PairGroupConfig` in fields, validation, and generator
    entry point, except that the round count must be exactly
    :data:`PAIR_GROUP_ROUNDS_SIX`.  The frozen five-round configuration and its
    verification path are unaffected.
    """

    required_rounds: ClassVar[int] = PAIR_GROUP_ROUNDS_SIX
    default_dataset_id: ClassVar[str] = PAIR_GROUP_SIX_ROUND_DATASET_ID


@dataclass(frozen=True)
class GeneratedPairGroups:
    """Generated group features, labels, audit keys, and metadata."""

    features: np.ndarray
    labels: np.ndarray
    master_keys: np.ndarray
    metadata: dict[str, object]
    plaintext_pairs: np.ndarray | None = None
    ciphertext_pairs: np.ndarray | None = None


def _array_digest(digest: hashlib._Hash, name: str, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    digest.update(name.encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))


def _content_checksum(
    features: np.ndarray,
    labels: np.ndarray,
    master_keys: np.ndarray,
) -> str:
    """Hash all stored arrays with their shape and dtype."""

    digest = hashlib.sha256()
    _array_digest(digest, "features", features)
    _array_digest(digest, "labels", labels)
    _array_digest(digest, "master_keys", master_keys)
    return digest.hexdigest()


def _word_features(ciphertext_pairs: np.ndarray) -> np.ndarray:
    """Encode ``(N,G,2,2)`` word pairs as ``(N,G,4,16)`` MSB-first bits."""

    if ciphertext_pairs.shape[-2:] != (2, 2):
        raise ValueError("ciphertext_pairs must end with (2, 2)")
    words = np.asarray(ciphertext_pairs, dtype=np.uint16).reshape(
        *ciphertext_pairs.shape[:-2],
        4,
    )
    shifts = np.arange(15, -1, -1, dtype=np.uint16)
    return np.bitwise_and(
        np.right_shift(words[..., None], shifts),
        1,
    ).astype(np.uint8, copy=False)


def generate_pair_groups(
    config: PairGroupConfig,
    *,
    generated_at: str | None = None,
    repository_commit: str | None = None,
    include_debug_pairs: bool = False,
) -> GeneratedPairGroups:
    """Generate balanced same-key positive and negative ciphertext groups.

    The local ``numpy.random.Generator`` is the only source of randomness.
    Positive groups use the verified plaintext difference; negative groups use
    independent random second plaintexts.  A deterministic permutation hides
    the class ordering while preserving each group's key and pairs.
    """

    rng = np.random.default_rng(config.seed)
    group_count = config.group_count
    group_size = config.group_size

    master_keys = rng.integers(
        0,
        WORD_MASK + 1,
        size=(group_count, 4),
        dtype=np.uint16,
    )
    plaintext0 = rng.integers(
        0,
        WORD_MASK + 1,
        size=(group_count, group_size, 2),
        dtype=np.uint16,
    )
    plaintext1 = np.empty_like(plaintext0)
    half = group_count // 2
    difference = np.asarray(config.input_difference, dtype=np.uint16)
    plaintext1[:half] = np.bitwise_xor(plaintext0[:half], difference)
    plaintext1[half:] = rng.integers(
        0,
        WORD_MASK + 1,
        size=(group_count - half, group_size, 2),
        dtype=np.uint16,
    )
    labels = np.zeros(group_count, dtype=np.uint8)
    labels[:half] = 1

    permutation = rng.permutation(group_count)
    master_keys = master_keys[permutation]
    plaintext0 = plaintext0[permutation]
    plaintext1 = plaintext1[permutation]
    labels = labels[permutation]

    ciphertext0 = np.empty_like(plaintext0)
    ciphertext1 = np.empty_like(plaintext1)
    for group_index in range(group_count):
        master_key = tuple(int(word) for word in master_keys[group_index])
        for pair_index in range(group_size):
            ciphertext0[group_index, pair_index] = encrypt_block(
                tuple(int(word) for word in plaintext0[group_index, pair_index]),
                master_key,
                rounds=config.rounds,
            )
            ciphertext1[group_index, pair_index] = encrypt_block(
                tuple(int(word) for word in plaintext1[group_index, pair_index]),
                master_key,
                rounds=config.rounds,
            )

    ciphertext_words = np.stack((ciphertext0, ciphertext1), axis=2)
    features = _word_features(ciphertext_words)
    timestamp = generated_at or datetime.now(UTC).isoformat()
    metadata: dict[str, object] = {
        "bit_order": BIT_ORDER,
        "cipher": CIPHER_NAME,
        "class_balance": {
            "0": int(np.count_nonzero(labels == 0)),
            "1": int(np.count_nonzero(labels == 1)),
        },
        "data_checksum": _content_checksum(features, labels, master_keys),
        "dataset_format_version": DATASET_FORMAT_VERSION,
        "dataset_id": config.default_dataset_id,
        "feature_dtype": str(features.dtype),
        "feature_shape": list(features.shape),
        "generated_at": timestamp,
        "generator_version": PAIR_GROUP_GENERATOR_VERSION,
        "group_count": group_count,
        "group_size": group_size,
        "input_difference": list(config.input_difference),
        "key_generation_policy": PAIR_GROUP_KEY_POLICY,
        "license": DATASET_LICENSE,
        "master_key_dtype": str(master_keys.dtype),
        "master_key_shape": list(master_keys.shape),
        "pair_construction": PAIR_GROUP_PAIR_CONSTRUCTION,
        "repository_commit": repository_commit,
        "rounds": config.rounds,
        "schema": PAIR_GROUP_SHARD_SCHEMA,
        "seed": config.seed,
        "split_role": config.split_role,
        "task": PAIR_GROUP_TASK,
        "word_order": WORD_ORDER,
    }
    return GeneratedPairGroups(
        features=features,
        labels=labels,
        master_keys=master_keys,
        metadata=metadata,
        plaintext_pairs=(
            np.stack((plaintext0, plaintext1), axis=2)
            if include_debug_pairs
            else None
        ),
        ciphertext_pairs=(
            np.stack((ciphertext0, ciphertext1), axis=2)
            if include_debug_pairs
            else None
        ),
    )


def assert_disjoint_pair_group_keys(
    datasets: Sequence[GeneratedPairGroups],
) -> None:
    """Require distinct master keys across generated group splits."""

    seen: set[tuple[int, int, int, int]] = set()
    for dataset in datasets:
        for row in dataset.master_keys:
            key = tuple(int(word) for word in row)
            if key in seen:
                raise ValueError("pair-group split master keys overlap")
            seen.add(key)


def validate_pair_group_split_configs(
    configs: Sequence[PairGroupConfig],
) -> None:
    """Validate the fixed train/validation group protocol."""

    config_tuple = tuple(configs)
    roles = {config.split_role for config in config_tuple}
    if roles != set(PAIR_GROUP_SPLIT_ROLES) or len(config_tuple) != 2:
        raise ValueError(
            "configs must contain exactly one train and one validation split"
        )
    seeds = [config.seed for config in config_tuple]
    if len(set(seeds)) != len(seeds):
        raise ValueError("train and validation seeds must be distinct")
    reference = config_tuple[0]
    for config in config_tuple[1:]:
        if config.group_size != reference.group_size:
            raise ValueError("group configurations must use the same group size")
        if config.rounds != reference.rounds:
            raise ValueError("group configurations must use the same rounds")
        if config.input_difference != reference.input_difference:
            raise ValueError(
                "group configurations must use the same input difference"
            )


def _validate_arrays(
    features: np.ndarray,
    labels: np.ndarray,
    master_keys: np.ndarray,
    metadata: Mapping[str, object],
) -> None:
    if features.dtype != np.dtype(np.uint8):
        raise ValueError("pair-group features must have dtype uint8")
    if labels.dtype != np.dtype(np.uint8):
        raise ValueError("pair-group labels must have dtype uint8")
    if master_keys.dtype != np.dtype(np.uint16):
        raise ValueError("pair-group master_keys must have dtype uint16")
    if features.ndim != 4 or features.shape[1:] != (64, 4, 16):
        raise ValueError("pair-group features must have shape (N,64,4,16)")
    if labels.shape != (features.shape[0],):
        raise ValueError("pair-group labels shape does not match features")
    if master_keys.shape != (features.shape[0], 4):
        raise ValueError("pair-group master_keys shape does not match features")
    if not np.isin(features, (0, 1)).all():
        raise ValueError("pair-group features must contain only binary values")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("pair-group labels must contain only 0/1 values")
    zero_count = int(np.count_nonzero(labels == 0))
    one_count = int(np.count_nonzero(labels == 1))
    if zero_count != one_count:
        raise ValueError("pair-group labels must be balanced")
    class_balance = metadata.get("class_balance")
    if not isinstance(class_balance, Mapping):
        raise ValueError("pair-group class balance metadata is incomplete")
    if class_balance.get("0") != zero_count or class_balance.get("1") != one_count:
        raise ValueError("pair-group class balance metadata does not match")
    if metadata.get("group_count") != int(features.shape[0]):
        raise ValueError("pair-group count metadata does not match contents")
    if metadata.get("feature_shape") != list(features.shape):
        raise ValueError("pair-group feature shape metadata does not match")
    if metadata.get("feature_dtype") != str(features.dtype):
        raise ValueError("pair-group feature dtype metadata does not match")
    if metadata.get("master_key_shape") != list(master_keys.shape):
        raise ValueError("pair-group master-key shape metadata does not match")
    if metadata.get("master_key_dtype") != str(master_keys.dtype):
        raise ValueError("pair-group master-key dtype metadata does not match")
    expected_checksum = _content_checksum(features, labels, master_keys)
    if metadata.get("data_checksum") != expected_checksum:
        raise ValueError("pair-group data checksum does not match contents")


def _shard_metadata(
    dataset: GeneratedPairGroups,
    *,
    dataset_id: str,
) -> dict[str, object]:
    metadata = dict(dataset.metadata)
    metadata["dataset_id"] = dataset_id
    metadata["group_count"] = int(dataset.features.shape[0])
    metadata["group_size"] = int(dataset.features.shape[1])
    _validate_arrays(dataset.features, dataset.labels, dataset.master_keys, metadata)
    return metadata


def write_pair_group_shard(
    dataset: GeneratedPairGroups,
    path: Path,
    *,
    dataset_id: str = PAIR_GROUP_DATASET_ID,
) -> dict[str, object]:
    """Write one group shard and return its manifest record."""

    metadata = _shard_metadata(dataset, dataset_id=dataset_id)
    metadata_json = canonical_json(metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=np.ascontiguousarray(dataset.features),
        labels=np.ascontiguousarray(dataset.labels),
        master_keys=np.ascontiguousarray(dataset.master_keys),
        metadata_json=np.asarray(metadata_json),
    )
    payload = path.read_bytes()
    return {
        "data_checksum": metadata["data_checksum"],
        "file_sha256": sha256_bytes(payload),
        "group_count": metadata["group_count"],
        "group_size": metadata["group_size"],
        "path": path.name,
        "seed": metadata["seed"],
        "split_role": metadata["split_role"],
    }


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {
                "features",
                "labels",
                "master_keys",
                "metadata_json",
            }:
                raise ValueError("pair-group shard arrays are incomplete")
            features = np.asarray(archive["features"]).copy()
            labels = np.asarray(archive["labels"]).copy()
            master_keys = np.asarray(archive["master_keys"]).copy()
            metadata_raw = archive["metadata_json"]
            if metadata_raw.ndim != 0 or not isinstance(metadata_raw.item(), str):
                raise ValueError("pair-group metadata_json must be a string scalar")
            metadata_value = json.loads(metadata_raw.item())
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read pair-group shard {path}") from error
    if not isinstance(metadata_value, Mapping):
        raise ValueError("pair-group shard metadata must be an object")
    metadata = dict(metadata_value)
    return features, labels, master_keys, metadata


def load_pair_group_shard(
    path: Path,
    *,
    expected_dataset_id: str | None = PAIR_GROUP_DATASET_ID,
    expected_rounds: int | None = None,
) -> GeneratedPairGroups:
    """Load and validate one architecture-neutral group shard.

    ``expected_rounds`` is resolved from ``expected_dataset_id`` when it is not
    given; ids outside :data:`PAIR_GROUP_DATASET_ROUNDS` keep the frozen
    five-round expectation.
    """

    features, labels, master_keys, metadata = _load_npz(path)
    if metadata.get("schema") not in (None, PAIR_GROUP_SHARD_SCHEMA):
        raise ValueError("pair-group shard schema is not recognised")
    if metadata.get("dataset_id") != expected_dataset_id and expected_dataset_id is not None:
        raise ValueError(f"pair-group dataset_id must be {expected_dataset_id!r}")
    required = (
        "bit_order",
        "cipher",
        "class_balance",
        "data_checksum",
        "dataset_format_version",
        "dataset_id",
        "feature_dtype",
        "feature_shape",
        "generated_at",
        "generator_version",
        "group_count",
        "group_size",
        "input_difference",
        "key_generation_policy",
        "license",
        "master_key_dtype",
        "master_key_shape",
        "pair_construction",
        "repository_commit",
        "rounds",
        "schema",
        "seed",
        "split_role",
        "task",
        "word_order",
    )
    if any(field not in metadata for field in required):
        raise ValueError("pair-group shard metadata is incomplete")
    if metadata["cipher"] != CIPHER_NAME or metadata["task"] != PAIR_GROUP_TASK:
        raise ValueError("pair-group cipher or task is unsupported")
    if metadata["dataset_format_version"] != DATASET_FORMAT_VERSION:
        raise ValueError("pair-group format version is unsupported")
    if metadata["license"] != DATASET_LICENSE:
        raise ValueError("pair-group licence is not MIT")
    if metadata["generator_version"] != PAIR_GROUP_GENERATOR_VERSION:
        raise ValueError("pair-group generator version is unsupported")
    if metadata["bit_order"] != BIT_ORDER or metadata["word_order"] != WORD_ORDER:
        raise ValueError("pair-group word or bit order is unsupported")
    if (
        metadata["group_size"] != PAIR_GROUP_SIZE
        or metadata["rounds"]
        != _resolve_expected_rounds(expected_dataset_id, expected_rounds)
    ):
        raise ValueError("pair-group size or rounds are unsupported")
    if metadata["split_role"] not in PAIR_GROUP_SPLIT_ROLES:
        raise ValueError("pair-group split role is invalid")
    if metadata["input_difference"] != list(DEFAULT_INPUT_DIFFERENCE):
        raise ValueError("pair-group input difference is unsupported")
    _validate_arrays(features, labels, master_keys, metadata)
    return GeneratedPairGroups(
        features=features,
        labels=labels,
        master_keys=master_keys,
        metadata=metadata,
    )


def verify_pair_group_shard(
    path: Path,
    *,
    expected_dataset_id: str | None = PAIR_GROUP_DATASET_ID,
    expected_manifest_record: Mapping[str, object] | None = None,
    expected_rounds: int | None = None,
) -> GeneratedPairGroups:
    """Verify a shard and its optional manifest record."""

    dataset = load_pair_group_shard(
        path,
        expected_dataset_id=expected_dataset_id,
        expected_rounds=expected_rounds,
    )
    if expected_manifest_record is not None:
        record = expected_manifest_record
        if record.get("path") != path.name:
            raise ValueError("pair-group manifest path does not match shard")
        if record.get("file_sha256") != sha256_bytes(path.read_bytes()):
            raise ValueError("pair-group manifest file hash does not match shard")
        for field in ("data_checksum", "group_count", "group_size", "seed", "split_role"):
            if record.get(field) != dataset.metadata.get(field):
                raise ValueError(f"pair-group manifest {field} does not match shard")
    return dataset


def load_pair_group_manifest(
    path: Path,
    *,
    expected_dataset_id: str | None = PAIR_GROUP_DATASET_ID,
    expected_rounds: int | None = None,
) -> dict[str, Any]:
    """Load and validate the two-split group manifest.

    ``expected_rounds`` is resolved from ``expected_dataset_id`` when it is not
    given; ids outside :data:`PAIR_GROUP_DATASET_ROUNDS` keep the frozen
    five-round expectation.
    """

    try:
        payload = path.read_bytes()
        document = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read pair-group manifest {path}") from error
    if not isinstance(document, Mapping):
        raise ValueError("pair-group manifest must be an object")
    if document.get("schema") != PAIR_GROUP_MANIFEST_SCHEMA:
        raise ValueError("pair-group manifest schema is not recognised")
    if document.get("dataset_id") != expected_dataset_id and expected_dataset_id is not None:
        raise ValueError(f"pair-group manifest dataset_id must be {expected_dataset_id!r}")
    if document.get("cipher") != CIPHER_NAME or document.get("task") != PAIR_GROUP_TASK:
        raise ValueError("pair-group manifest cipher or task is unsupported")
    if document.get("dataset_format_version") != DATASET_FORMAT_VERSION:
        raise ValueError("pair-group manifest format version is unsupported")
    if document.get("license") != DATASET_LICENSE:
        raise ValueError("pair-group manifest licence is not MIT")
    if document.get("generator_version") != PAIR_GROUP_GENERATOR_VERSION:
        raise ValueError("pair-group manifest generator version is unsupported")
    if document.get("rounds") != _resolve_expected_rounds(
        expected_dataset_id, expected_rounds
    ):
        raise ValueError("pair-group manifest rounds are unsupported")
    if document.get("group_size") != PAIR_GROUP_SIZE:
        raise ValueError("pair-group manifest group size is unsupported")
    if document.get("input_difference") != list(DEFAULT_INPUT_DIFFERENCE):
        raise ValueError("pair-group manifest input difference is unsupported")
    if document.get("bit_order") != BIT_ORDER or document.get("word_order") != WORD_ORDER:
        raise ValueError("pair-group manifest word or bit order is unsupported")
    splits = document.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(PAIR_GROUP_SPLIT_ROLES):
        raise ValueError("pair-group manifest must contain train and validation")
    seeds: list[int] = []
    for role in PAIR_GROUP_SPLIT_ROLES:
        record = splits[role]
        if not isinstance(record, Mapping):
            raise ValueError(f"pair-group record for {role} must be an object")
        for field in (
            "data_checksum",
            "file_sha256",
            "group_count",
            "group_size",
            "path",
            "seed",
            "split_role",
        ):
            if field not in record:
                raise ValueError(f"pair-group record for {role} is incomplete")
        if record["split_role"] != role:
            raise ValueError(f"pair-group split role mismatch for {role}")
        shard_path = Path(str(record["path"]))
        if shard_path.name != record["path"] or shard_path.is_absolute():
            raise ValueError("pair-group shard path must be a local filename")
        seed = record["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("pair-group split seed must be an integer")
        seeds.append(seed)
    if len(set(seeds)) != len(seeds):
        raise ValueError("pair-group split seeds must be distinct")
    if payload != canonical_json(document).encode("utf-8"):
        raise ValueError("pair-group manifest is not canonical JSON")
    return dict(document)


def verify_pair_group_benchmark(
    manifest_path: Path,
    *,
    expected_dataset_id: str | None = PAIR_GROUP_DATASET_ID,
    expected_rounds: int | None = None,
) -> dict[str, Any]:
    """Verify both group shards and disjoint master-key provenance.

    The round expectation is resolved once from ``expected_dataset_id`` (or the
    explicit ``expected_rounds``) and applied to the manifest and both shards.
    """

    resolved_rounds = _resolve_expected_rounds(expected_dataset_id, expected_rounds)
    manifest = load_pair_group_manifest(
        manifest_path,
        expected_dataset_id=expected_dataset_id,
        expected_rounds=resolved_rounds,
    )
    root = manifest_path.parent
    split_records = manifest["splits"]
    datasets: list[GeneratedPairGroups] = []
    for role in PAIR_GROUP_SPLIT_ROLES:
        record = split_records[role]
        datasets.append(
            verify_pair_group_shard(
                root / str(record["path"]),
                expected_dataset_id=str(manifest["dataset_id"]),
                expected_manifest_record=record,
                expected_rounds=resolved_rounds,
            )
        )
    assert_disjoint_pair_group_keys(datasets)
    return manifest


def export_pair_group_benchmark(
    configs: Sequence[PairGroupConfig],
    output_dir: Path,
    *,
    dataset_id: str = PAIR_GROUP_DATASET_ID,
    generated_at: str,
    repository_commit: str | None = None,
    rounds: int | None = None,
) -> dict[str, object]:
    """Generate, write, and manifest the fixed train/validation group data.

    ``rounds`` is resolved from ``dataset_id`` when it is not given (the frozen
    five-round id keeps five rounds); every supplied configuration must match
    the resolved value, so the six-round family is written only on explicit
    opt-in with :data:`PAIR_GROUP_ROUNDS_SIX` or
    :data:`PAIR_GROUP_SIX_ROUND_DATASET_ID`.
    """

    resolved_rounds = _resolve_expected_rounds(dataset_id, rounds)
    for config in configs:
        if config.rounds != resolved_rounds:
            raise ValueError(
                "pair-group configurations must use "
                f"{resolved_rounds} rounds for this export"
            )
    validate_pair_group_split_configs(configs)
    generated = [
        generate_pair_groups(
            config,
            generated_at=generated_at,
            repository_commit=repository_commit,
        )
        for config in configs
    ]
    assert_disjoint_pair_group_keys(generated)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_records: dict[str, dict[str, object]] = {}
    for dataset in generated:
        role = str(dataset.metadata["split_role"])
        split_records[role] = write_pair_group_shard(
            dataset,
            output_dir / f"{role}.npz",
            dataset_id=dataset_id,
        )
    reference = generated[0].metadata
    manifest: dict[str, object] = {
        "bit_order": BIT_ORDER,
        "cipher": CIPHER_NAME,
        "dataset_format_version": DATASET_FORMAT_VERSION,
        "dataset_id": dataset_id,
        "generated_at": generated_at,
        "generator_version": PAIR_GROUP_GENERATOR_VERSION,
        "group_size": PAIR_GROUP_SIZE,
        "input_difference": list(reference["input_difference"]),
        "license": DATASET_LICENSE,
        "pair_construction": PAIR_GROUP_PAIR_CONSTRUCTION,
        "repository_commit": repository_commit,
        "rounds": resolved_rounds,
        "schema": PAIR_GROUP_MANIFEST_SCHEMA,
        "splits": split_records,
        "task": PAIR_GROUP_TASK,
        "word_order": WORD_ORDER,
    }
    (output_dir / "manifest.json").write_bytes(
        canonical_json(manifest).encode("utf-8")
    )
    return manifest


__all__ = [
    "PAIR_GROUP_DATASET_ID",
    "PAIR_GROUP_DATASET_ROUNDS",
    "PAIR_GROUP_GENERATOR_VERSION",
    "PAIR_GROUP_MANIFEST_SCHEMA",
    "PAIR_GROUP_ROUNDS",
    "PAIR_GROUP_ROUNDS_SIX",
    "PAIR_GROUP_SHARD_SCHEMA",
    "PAIR_GROUP_SIX_ROUND_DATASET_ID",
    "PAIR_GROUP_SIZE",
    "PAIR_GROUP_SPLIT_ROLES",
    "PAIR_GROUP_TASK",
    "GeneratedPairGroups",
    "PairGroupConfig",
    "SixRoundPairGroupConfig",
    "assert_disjoint_pair_group_keys",
    "export_pair_group_benchmark",
    "generate_pair_groups",
    "load_pair_group_manifest",
    "load_pair_group_shard",
    "validate_pair_group_split_configs",
    "verify_pair_group_benchmark",
    "verify_pair_group_shard",
    "write_pair_group_shard",
]
