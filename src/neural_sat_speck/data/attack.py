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
from neural_sat_speck.speck.primitives import WORD_MASK, mask_word

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
from .metadata import (
    BIT_ORDER,
    GENERATOR_VERSION,
    WORD_ORDER,
)

MasterKey: TypeAlias = tuple[int, int, int, int]
ATTACK_SPLIT_ROLES: tuple[str, str, str] = ("train", "validation", "test")
ATTACK_KEY_GENERATION_POLICY: str = "one_random_master_key_per_instance"


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
        "key_generation_policy": ATTACK_KEY_GENERATION_POLICY,
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
    if not isinstance(document, Mapping):
        raise ValueError("attack instance must be a JSON object")
    raw_pairs = document.get("pairs")
    if not isinstance(raw_pairs, Sequence) or isinstance(raw_pairs, (str, bytes)):
        raise ValueError("attack instance pairs must be a sequence")
    pairs_list: list[AttackPair] = []
    for pair in raw_pairs:
        if not isinstance(pair, Mapping):
            raise ValueError("attack pair must be a JSON object")
        try:
            plaintext0 = hex_to_words(pair["plaintext0"])
            plaintext1 = hex_to_words(pair["plaintext1"])
            ciphertext0 = hex_to_words(pair["ciphertext0"])
            ciphertext1 = hex_to_words(pair["ciphertext1"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("attack pair has invalid word fields") from error
        if any(len(block) != 2 for block in (plaintext0, plaintext1, ciphertext0, ciphertext1)):
            raise ValueError("attack pair blocks must contain two words")
        pairs_list.append(
            AttackPair(
                plaintext0=(plaintext0[0], plaintext0[1]),
                plaintext1=(plaintext1[0], plaintext1[1]),
                ciphertext0=(ciphertext0[0], ciphertext0[1]),
                ciphertext1=(ciphertext1[0], ciphertext1[1]),
            )
        )
    pairs = tuple(pairs_list)
    raw_master_key = document.get("master_key")
    if not isinstance(raw_master_key, Sequence) or isinstance(
        raw_master_key, (str, bytes)
    ):
        raise ValueError("attack instance master_key must be a word sequence")
    master_key = hex_to_words(raw_master_key)
    if len(master_key) != 4:
        raise ValueError("attack instance master key must contain four words")
    instance_id = document.get("instance_id")
    if not isinstance(instance_id, str) or not instance_id:
        raise ValueError("attack instance_id must be a non-empty string")
    final_round_subkey = document.get("final_round_subkey")
    if not isinstance(final_round_subkey, str):
        raise ValueError("attack instance final_round_subkey must be a string")
    return AttackInstance(
        instance_id=instance_id,
        master_key=(master_key[0], master_key[1], master_key[2], master_key[3]),
        final_round_subkey=hex_to_word(final_round_subkey),
        pairs=pairs,
    )


def _validate_attack_document(
    document: Mapping[str, Any],
    *,
    expected_dataset_id: str | None,
) -> tuple[tuple[AttackInstance, ...], dict[str, object]]:
    if document.get("schema") != ATTACK_SHARD_SCHEMA:
        raise ValueError("attack shard schema is not recognised")
    if document.get("task") != ATTACK_TASK:
        raise ValueError("attack shard task is not single_key_related_pairs")
    if document.get("cipher") != CIPHER_NAME:
        raise ValueError("attack shard cipher is not SPECK32/64")
    if document.get("dataset_format_version") != DATASET_FORMAT_VERSION:
        raise ValueError("attack shard format version is not supported")
    if document.get("license") != DATASET_LICENSE:
        raise ValueError("attack shard licence is not MIT")
    dataset_id = document.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("attack shard dataset_id must be non-empty")
    if expected_dataset_id is not None and dataset_id != expected_dataset_id:
        raise ValueError(f"attack shard dataset_id must be {expected_dataset_id!r}")
    metadata_value = document.get("metadata")
    if not isinstance(metadata_value, Mapping):
        raise ValueError("attack shard metadata must be a JSON object")
    metadata = dict(metadata_value)
    required_fields = (
        "bit_order",
        "data_checksum",
        "generated_at",
        "generator_version",
        "input_difference",
        "instance_count",
        "key_generation_policy",
        "pair_construction",
        "pair_count",
        "repository_commit",
        "rounds",
        "seed",
        "split_role",
        "word_order",
    )
    if any(field not in metadata for field in required_fields):
        raise ValueError("attack shard metadata is incomplete")
    if metadata.get("generator_version") != GENERATOR_VERSION:
        raise ValueError("attack shard generator version is unsupported")
    if metadata.get("bit_order") != BIT_ORDER or metadata.get("word_order") != WORD_ORDER:
        raise ValueError("attack shard word or bit order is unsupported")
    repository_commit = metadata.get("repository_commit")
    if repository_commit is not None and not isinstance(repository_commit, str):
        raise ValueError("attack shard repository_commit must be a string or null")
    if not isinstance(metadata.get("generated_at"), str):
        raise ValueError("attack shard generated_at must be a string")
    if metadata.get("pair_construction") != PLAINTEXT_DIFFERENCE_POLICY:
        raise ValueError("attack shard pair construction is unsupported")
    if metadata.get("key_generation_policy") != ATTACK_KEY_GENERATION_POLICY:
        raise ValueError("attack shard key-generation policy is unsupported")
    for field in ("instance_count", "pair_count", "rounds", "seed"):
        value = metadata[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"attack shard metadata {field} must be an integer")
    if not 1 <= metadata["rounds"] <= SPECK32_64_ROUND_COUNT:
        raise ValueError("attack shard rounds are outside the SPECK32/64 range")
    if metadata["instance_count"] <= 0 or metadata["pair_count"] <= 0:
        raise ValueError("attack shard counts must be positive")
    if metadata["split_role"] not in ATTACK_SPLIT_ROLES:
        raise ValueError("attack shard split_role is invalid")
    if (
        not isinstance(metadata["input_difference"], Sequence)
        or isinstance(metadata["input_difference"], (str, bytes))
        or len(metadata["input_difference"]) != 2
        or any(
            isinstance(word, bool)
            or not isinstance(word, int)
            or not 0 <= word <= WORD_MASK
            for word in metadata["input_difference"]
        )
    ):
        raise ValueError("attack shard input_difference must contain two words")
    instances_value = document.get("instances")
    if not isinstance(instances_value, Sequence) or isinstance(
        instances_value, (str, bytes)
    ):
        raise ValueError("attack shard instances must be a sequence")
    instances = tuple(
        _parse_attack_instance(instance) for instance in document["instances"]
    )
    if len(instances) != int(metadata["instance_count"]):
        raise ValueError("attack shard instance count does not match metadata")
    if int(metadata["pair_count"]) <= 0:
        raise ValueError("attack shard pair_count must be positive")
    if any(len(instance.pairs) != int(metadata["pair_count"]) for instance in instances):
        raise ValueError("attack shard pair count does not match metadata")
    input_difference = (
        mask_word(int(metadata["input_difference"][0])),
        mask_word(int(metadata["input_difference"][1])),
    )
    rounds = int(metadata["rounds"])
    for instance in instances:
        if instance.final_round_subkey != expand_key(instance.master_key, rounds)[-1]:
            raise ValueError("attack shard final-round subkey is inconsistent")
        for pair in instance.pairs:
            expected_plaintext1 = (
                mask_word(pair.plaintext0[0] ^ input_difference[0]),
                mask_word(pair.plaintext0[1] ^ input_difference[1]),
            )
            if pair.plaintext1 != expected_plaintext1:
                raise ValueError("attack shard plaintext difference is invalid")
            if pair.ciphertext0 != encrypt_block(
                pair.plaintext0, instance.master_key, rounds
            ) or pair.ciphertext1 != encrypt_block(
                pair.plaintext1, instance.master_key, rounds
            ):
                raise ValueError("attack shard ciphertext does not match its key")
    if _attack_checksum(instances) != metadata["data_checksum"]:
        raise ValueError("attack shard data checksum does not match contents")
    return instances, metadata


def load_attack_shard(
    path: Path,
    *,
    expected_dataset_id: str | None = ATTACK_DATASET_ID,
) -> AttackInstanceSet:
    """Load and validate a word-level attack shard without PyTorch."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read attack shard {path}") from error
    if not isinstance(document, Mapping):
        raise ValueError("attack shard must be a JSON object")
    instances, metadata = _validate_attack_document(
        document,
        expected_dataset_id=expected_dataset_id,
    )
    return AttackInstanceSet(instances=instances, metadata=metadata)


def verify_attack_shard(
    path: Path,
    *,
    expected_dataset_id: str | None = ATTACK_DATASET_ID,
    expected_manifest_record: Mapping[str, object] | None = None,
) -> AttackInstanceSet:
    """Verify attack content, canonical JSON, and an optional manifest record."""

    payload = path.read_bytes()
    instance_set = load_attack_shard(
        path,
        expected_dataset_id=expected_dataset_id,
    )
    document = json.loads(payload.decode("utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("attack shard must be a JSON object")
    dataset_id = document.get("dataset_id")
    if not isinstance(dataset_id, str):
        raise ValueError("attack shard dataset_id must be a string")
    if payload != attack_shard_bytes(
        instance_set,
        dataset_id=dataset_id,
    ):
        raise ValueError("attack shard is not in canonical JSON form")
    if expected_manifest_record is not None:
        record = expected_manifest_record
        if record.get("path") != path.name:
            raise ValueError("manifest shard path does not match the shard file")
        if record.get("file_sha256") != sha256_bytes(payload):
            raise ValueError("manifest file_sha256 does not match shard bytes")
        if record.get("data_checksum") != instance_set.metadata["data_checksum"]:
            raise ValueError("manifest data_checksum does not match shard contents")
        if record.get("instance_count") != instance_set.metadata["instance_count"]:
            raise ValueError("manifest instance_count does not match shard metadata")
        if record.get("pair_count") != instance_set.metadata["pair_count"]:
            raise ValueError("manifest pair_count does not match shard metadata")
        if record.get("seed") != instance_set.metadata["seed"]:
            raise ValueError("manifest seed does not match shard metadata")
        if record.get("split_role") != instance_set.metadata["split_role"]:
            raise ValueError("manifest split_role does not match shard metadata")
    return instance_set


def load_attack_manifest(
    path: Path,
    *,
    expected_dataset_id: str | None = ATTACK_DATASET_ID,
) -> dict[str, Any]:
    """Load and validate the top-level attack manifest schema."""

    try:
        payload = path.read_bytes()
        document = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read attack manifest {path}") from error
    if not isinstance(document, Mapping):
        raise ValueError("attack manifest must be a JSON object")
    if document.get("schema") != ATTACK_MANIFEST_SCHEMA:
        raise ValueError("attack manifest schema is not recognised")
    if document.get("cipher") != CIPHER_NAME:
        raise ValueError("attack manifest cipher is not SPECK32/64")
    if document.get("dataset_format_version") != DATASET_FORMAT_VERSION:
        raise ValueError("attack manifest format version is not supported")
    if document.get("task") != ATTACK_TASK:
        raise ValueError("attack manifest task is not single_key_related_pairs")
    if document.get("license") != DATASET_LICENSE:
        raise ValueError("attack manifest licence is not MIT")
    if document.get("generator_version") != GENERATOR_VERSION:
        raise ValueError("attack manifest generator version is unsupported")
    if document.get("word_order") != WORD_ORDER:
        raise ValueError("attack manifest word order is unsupported")
    if document.get("pair_construction") != PLAINTEXT_DIFFERENCE_POLICY:
        raise ValueError("attack manifest pair construction is unsupported")
    repository_commit = document.get("repository_commit")
    if repository_commit is not None and not isinstance(repository_commit, str):
        raise ValueError("attack manifest repository_commit must be a string or null")
    if not isinstance(document.get("generated_at"), str):
        raise ValueError("attack manifest generated_at must be a string")
    dataset_id = document.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("attack manifest dataset_id must be non-empty")
    if expected_dataset_id is not None and dataset_id != expected_dataset_id:
        raise ValueError(f"attack manifest dataset_id must be {expected_dataset_id!r}")
    rounds = document.get("rounds")
    if (
        isinstance(rounds, bool)
        or not isinstance(rounds, int)
        or not 1 <= rounds <= SPECK32_64_ROUND_COUNT
    ):
        raise ValueError("attack manifest rounds are outside the SPECK32/64 range")
    input_difference = document.get("input_difference")
    if (
        not isinstance(input_difference, Sequence)
        or isinstance(input_difference, (str, bytes))
        or len(input_difference) != 2
        or any(
            isinstance(word, bool)
            or not isinstance(word, int)
            or not 0 <= word <= WORD_MASK
            for word in input_difference
        )
    ):
        raise ValueError("attack manifest input_difference must contain two words")
    splits = document.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(ATTACK_SPLIT_ROLES):
        raise ValueError("attack manifest must contain train, validation, and test")
    seeds: list[int] = []
    for role in ATTACK_SPLIT_ROLES:
        record = splits[role]
        if not isinstance(record, Mapping):
            raise ValueError(f"attack manifest record for {role} must be an object")
        for field in (
            "path",
            "data_checksum",
            "file_sha256",
            "instance_count",
            "pair_count",
            "seed",
            "split_role",
        ):
            if field not in record:
                raise ValueError(f"attack manifest record is missing {field}")
        if record["split_role"] != role:
            raise ValueError(f"attack manifest split role mismatch for {role}")
        if not isinstance(record["path"], str):
            raise ValueError("attack manifest shard path must be a string")
        shard_path = Path(record["path"])
        if shard_path.name != record["path"] or shard_path.is_absolute():
            raise ValueError("attack manifest shard path must be a local filename")
        if isinstance(record["seed"], bool) or not isinstance(record["seed"], int):
            raise ValueError("attack manifest seed must be an integer")
        if (
            isinstance(record["instance_count"], bool)
            or not isinstance(record["instance_count"], int)
            or record["instance_count"] <= 0
        ):
            raise ValueError("attack manifest instance_count must be positive")
        if (
            isinstance(record["pair_count"], bool)
            or not isinstance(record["pair_count"], int)
            or record["pair_count"] <= 0
        ):
            raise ValueError("attack manifest pair_count must be positive")
        seeds.append(record["seed"])
    if len(set(seeds)) != len(seeds):
        raise ValueError("attack manifest split seeds must be distinct")
    if payload != canonical_json(document).encode("utf-8"):
        raise ValueError("attack manifest is not in canonical JSON form")
    return dict(document)


def verify_attack_benchmark(
    manifest_path: Path,
    *,
    expected_dataset_id: str | None = ATTACK_DATASET_ID,
    expected_rounds: int | None = None,
    expected_input_difference: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Verify every attack shard against its manifest and cipher equations."""

    manifest = load_attack_manifest(
        manifest_path,
        expected_dataset_id=expected_dataset_id,
    )
    if expected_rounds is not None and manifest["rounds"] != expected_rounds:
        raise ValueError("attack benchmark rounds do not match the requested value")
    if expected_input_difference is not None:
        if manifest["input_difference"] != list(expected_input_difference):
            raise ValueError(
                "attack benchmark input_difference does not match the requested value"
            )
    root = manifest_path.parent
    splits = manifest["splits"]
    seen_keys: set[MasterKey] = set()
    reference_metadata: dict[str, object] | None = None
    for role in ATTACK_SPLIT_ROLES:
        record = splits[role]
        instance_set = verify_attack_shard(
            root / str(record["path"]),
            expected_dataset_id=str(manifest["dataset_id"]),
            expected_manifest_record=record,
        )
        metadata = instance_set.metadata
        if metadata["split_role"] != role:
            raise ValueError(f"attack shard metadata role mismatch for {role}")
        if metadata["rounds"] != manifest["rounds"]:
            raise ValueError(f"attack shard rounds mismatch for {role}")
        if metadata["input_difference"] != manifest["input_difference"]:
            raise ValueError(f"attack shard input difference mismatch for {role}")
        if metadata["generator_version"] != manifest["generator_version"]:
            raise ValueError(f"attack shard generator version mismatch for {role}")
        if metadata["repository_commit"] != manifest["repository_commit"]:
            raise ValueError(f"attack shard repository commit mismatch for {role}")
        if metadata["generated_at"] != manifest["generated_at"]:
            raise ValueError(f"attack shard timestamp mismatch for {role}")
        if reference_metadata is None:
            reference_metadata = metadata
        else:
            for field in (
                "bit_order",
                "word_order",
                "pair_construction",
                "key_generation_policy",
            ):
                if metadata[field] != reference_metadata[field]:
                    raise ValueError(f"attack shards disagree on {field}")
        for instance in instance_set.instances:
            if instance.master_key in seen_keys:
                raise ValueError("attack shard master keys overlap across splits")
            seen_keys.add(instance.master_key)
    return manifest


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
