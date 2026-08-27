"""Deterministic Dataset v1 export for the classifier benchmark."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from neural_sat_speck.speck.key_schedule import SPECK32_64_ROUND_COUNT
from neural_sat_speck.speck.primitives import WORD_MASK

from .format import (
    CIPHER_NAME,
    CLASSIFIER_DATASET_ID,
    CLASSIFIER_MANIFEST_SCHEMA,
    CLASSIFIER_SHARD_SCHEMA,
    CLASSIFIER_TASK,
    DATASET_FORMAT_VERSION,
    DATASET_LICENSE,
    canonical_json,
    sha256_bytes,
)
from .generator import (
    DatasetConfig,
    GeneratedDataset,
    dataset_content_checksum,
    generate_dataset,
    validate_split_configs,
)
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

CLASSIFIER_SPLIT_ROLES: tuple[str, str, str] = (
    "train",
    "validation",
    "test",
)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    """Return a JSON object or raise a useful schema error."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _parse_metadata(metadata_mapping: object) -> DatasetMetadata:
    """Parse and structurally validate Dataset v1 shard metadata."""

    metadata = _mapping(metadata_mapping, "metadata")
    try:
        class_balance = _mapping(metadata["class_balance"], "class_balance")
        input_difference = metadata["input_difference"]
        feature_shape = metadata["feature_shape"]
        if not isinstance(input_difference, Sequence) or isinstance(
            input_difference, (str, bytes)
        ):
            raise ValueError("input_difference must be a two-item sequence")
        if not isinstance(feature_shape, Sequence) or isinstance(
            feature_shape, (str, bytes)
        ):
            raise ValueError("feature_shape must be a two-item sequence")
        if len(input_difference) != 2 or len(feature_shape) != 2:
            raise ValueError("input_difference and feature_shape must have two items")
        repository_commit = metadata["repository_commit"]
        if repository_commit is not None and not isinstance(
            repository_commit, str
        ):
            raise ValueError("repository_commit must be a string or null")
        parsed = DatasetMetadata(
            generator_version=str(metadata["generator_version"]),
            repository_commit=repository_commit,
            generated_at=str(metadata["generated_at"]),
            sample_count=int(metadata["sample_count"]),
            class_balance=(
                int(class_balance["0"]),
                int(class_balance["1"]),
            ),
            rounds=int(metadata["rounds"]),
            input_difference=(
                int(input_difference[0]),
                int(input_difference[1]),
            ),
            key_generation_policy=str(metadata["key_generation_policy"]),
            seed=int(metadata["seed"]),
            split_role=str(metadata["split_role"]),
            word_order=str(metadata["word_order"]),
            bit_order=str(metadata["bit_order"]),
            feature_representation=str(metadata["feature_representation"]),
            feature_dtype=str(metadata["feature_dtype"]),
            feature_shape=(
                int(feature_shape[0]),
                int(feature_shape[1]),
            ),
            data_checksum=str(metadata["data_checksum"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("classifier shard metadata is incomplete") from error

    if parsed.sample_count <= 0:
        raise ValueError("metadata sample_count must be positive")
    if parsed.class_balance[0] < 0 or parsed.class_balance[1] < 0:
        raise ValueError("metadata class_balance cannot be negative")
    if sum(parsed.class_balance) != parsed.sample_count:
        raise ValueError("metadata class_balance must sum to sample_count")
    if not 1 <= parsed.rounds <= SPECK32_64_ROUND_COUNT:
        raise ValueError(
            "metadata rounds must be between 1 and "
            f"{SPECK32_64_ROUND_COUNT}"
        )
    if parsed.feature_shape != (parsed.sample_count, FEATURE_WIDTH):
        raise ValueError(
            "metadata feature_shape must equal (sample_count, feature width)"
        )
    if parsed.feature_dtype != FEATURE_DTYPE:
        raise ValueError(
            f"Dataset v1 classifier features must have dtype {FEATURE_DTYPE}"
        )
    if parsed.generator_version != GENERATOR_VERSION:
        raise ValueError("classifier shard generator version is unsupported")
    if parsed.key_generation_policy != KEY_GENERATION_POLICY:
        raise ValueError("classifier shard key-generation policy is unsupported")
    if parsed.word_order != WORD_ORDER:
        raise ValueError("classifier shard word order is unsupported")
    if parsed.bit_order != BIT_ORDER:
        raise ValueError("classifier shard bit order is unsupported")
    if parsed.feature_representation != FEATURE_REPRESENTATION:
        raise ValueError("classifier shard feature representation is unsupported")
    if parsed.split_role not in CLASSIFIER_SPLIT_ROLES:
        raise ValueError("metadata split_role is not a classifier split")
    return parsed


def _validate_classifier_document(
    document: Mapping[str, Any],
    *,
    expected_dataset_id: str | None,
) -> DatasetMetadata:
    """Validate fixed classifier-shard fields and return parsed metadata."""

    if document.get("schema") != CLASSIFIER_SHARD_SCHEMA:
        raise ValueError("classifier shard schema is not recognised")
    if document.get("task") != CLASSIFIER_TASK:
        raise ValueError("classifier shard task is not related_vs_random")
    if document.get("cipher") != CIPHER_NAME:
        raise ValueError("classifier shard cipher is not SPECK32/64")
    if document.get("dataset_format_version") != DATASET_FORMAT_VERSION:
        raise ValueError("classifier shard format version is not supported")
    if document.get("license") != DATASET_LICENSE:
        raise ValueError("classifier shard licence is not MIT")
    dataset_id = document.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("classifier shard dataset_id must be a non-empty string")
    if expected_dataset_id is not None and dataset_id != expected_dataset_id:
        raise ValueError(
            f"classifier shard dataset_id must be {expected_dataset_id!r}"
        )
    metadata = _parse_metadata(document.get("metadata"))
    features = document.get("features")
    labels = document.get("labels")
    if not isinstance(features, Sequence) or isinstance(features, (str, bytes)):
        raise ValueError("classifier shard features must be a sequence")
    if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
        raise ValueError("classifier shard labels must be a sequence")
    if len(features) != metadata.sample_count or len(labels) != metadata.sample_count:
        raise ValueError("feature and label counts must match metadata sample_count")
    return metadata


def _validate_classifier_content(
    features: Sequence[Sequence[object]],
    labels: Sequence[object],
    metadata: DatasetMetadata,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Validate feature/label values and return immutable typed content."""

    typed_features: list[tuple[int, ...]] = []
    typed_labels: list[int] = []
    for row in features:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise ValueError("each classifier feature must be a sequence")
        if len(row) != FEATURE_WIDTH:
            raise ValueError(
                f"each classifier feature must contain {FEATURE_WIDTH} bits"
            )
        typed_row: list[int] = []
        for bit in row:
            if isinstance(bit, bool) or not isinstance(bit, int) or bit not in (0, 1):
                raise ValueError("classifier features must contain only integer 0/1 bits")
            typed_row.append(bit)
        typed_features.append(tuple(typed_row))

    for label in labels:
        if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
            raise ValueError("classifier labels must contain only integer 0/1 values")
        typed_labels.append(label)

    actual_balance = (typed_labels.count(0), typed_labels.count(1))
    if actual_balance != metadata.class_balance:
        raise ValueError("metadata class_balance does not match labels")
    return tuple(typed_features), tuple(typed_labels)


def classifier_shard_document(
    dataset: GeneratedDataset,
    *,
    dataset_id: str = CLASSIFIER_DATASET_ID,
) -> dict[str, object]:
    """Return the architecture-neutral JSON document for one classifier split."""

    return {
        "cipher": CIPHER_NAME,
        "dataset_format_version": DATASET_FORMAT_VERSION,
        "dataset_id": dataset_id,
        "features": [list(feature) for feature in dataset.features],
        "labels": list(dataset.labels),
        "license": DATASET_LICENSE,
        "metadata": dataset.metadata.to_dict(),
        "schema": CLASSIFIER_SHARD_SCHEMA,
        "task": CLASSIFIER_TASK,
    }


def classifier_shard_bytes(dataset: GeneratedDataset, **kwargs: object) -> bytes:
    """Serialize one classifier split to canonical UTF-8 JSON bytes."""

    return canonical_json(classifier_shard_document(dataset, **kwargs)).encode(
        "utf-8"
    )


def write_classifier_shard(
    dataset: GeneratedDataset,
    path: Path,
    **kwargs: object,
) -> dict[str, object]:
    """Write one classifier shard and return its manifest record."""

    payload = classifier_shard_bytes(dataset, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    metadata = dataset.metadata
    return {
        "class_balance": metadata.to_dict()["class_balance"],
        "data_checksum": metadata.data_checksum,
        "file_sha256": sha256_bytes(payload),
        "path": path.name,
        "sample_count": metadata.sample_count,
        "seed": metadata.seed,
        "split_role": metadata.split_role,
    }


def load_classifier_shard(
    path: Path,
    *,
    expected_dataset_id: str | None = CLASSIFIER_DATASET_ID,
) -> GeneratedDataset:
    """Load and schema-validate a classifier shard without PyTorch."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read classifier shard {path}") from error
    if not isinstance(document, Mapping):
        raise ValueError("classifier shard must be a JSON object")
    metadata = _validate_classifier_document(
        document,
        expected_dataset_id=expected_dataset_id,
    )
    features_value = document["features"]
    labels_value = document["labels"]
    # _validate_classifier_document has checked these container types.
    features, labels = _validate_classifier_content(
        features_value,  # type: ignore[arg-type]
        labels_value,  # type: ignore[arg-type]
        metadata,
    )
    return GeneratedDataset(
        features=features,
        labels=labels,
        metadata=metadata,
    )


def verify_classifier_shard(
    path: Path,
    *,
    expected_dataset_id: str | None = CLASSIFIER_DATASET_ID,
    expected_manifest_record: Mapping[str, object] | None = None,
) -> GeneratedDataset:
    """Load a shard and verify content, canonical bytes, and manifest data."""

    payload = path.read_bytes()
    dataset = load_classifier_shard(
        path,
        expected_dataset_id=expected_dataset_id,
    )
    expected = dataset.metadata.data_checksum
    actual = dataset_content_checksum(dataset.features, dataset.labels)
    if actual != expected:
        raise ValueError("classifier shard data checksum does not match contents")

    document = json.loads(payload.decode("utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("classifier shard must be a JSON object")
    dataset_id = document.get("dataset_id")
    if not isinstance(dataset_id, str):
        raise ValueError("classifier shard dataset_id must be a string")
    reconstructed = classifier_shard_bytes(dataset, dataset_id=dataset_id)
    if payload != reconstructed:
        raise ValueError("classifier shard is not in canonical JSON form")

    if expected_manifest_record is not None:
        record = expected_manifest_record
        actual_file_sha256 = sha256_bytes(payload)
        if record.get("path") != path.name:
            raise ValueError("manifest shard path does not match the shard file")
        if record.get("file_sha256") != actual_file_sha256:
            raise ValueError("manifest file_sha256 does not match shard bytes")
        if record.get("data_checksum") != dataset.metadata.data_checksum:
            raise ValueError("manifest data_checksum does not match shard contents")
        if record.get("class_balance") != dataset.metadata.to_dict()["class_balance"]:
            raise ValueError("manifest class_balance does not match shard metadata")
        if int(record.get("sample_count", -1)) != dataset.metadata.sample_count:
            raise ValueError("manifest sample_count does not match shard metadata")
        if int(record.get("seed", -1)) != dataset.metadata.seed:
            raise ValueError("manifest seed does not match shard metadata")
        if record.get("split_role") != dataset.metadata.split_role:
            raise ValueError("manifest split_role does not match shard metadata")
    return dataset


def classifier_manifest_document(
    split_records: Mapping[str, Mapping[str, object]],
    *,
    dataset_id: str = CLASSIFIER_DATASET_ID,
    rounds: int,
    input_difference: Sequence[int],
    word_order: str,
    bit_order: str,
    feature_representation: str,
    feature_dtype: str,
    key_generation_policy: str,
    generator_version: str = GENERATOR_VERSION,
    repository_commit: str | None,
    generated_at: str,
) -> dict[str, object]:
    """Return the classifier-benchmark manifest document."""

    return {
        "bit_order": bit_order,
        "cipher": CIPHER_NAME,
        "dataset_format_version": DATASET_FORMAT_VERSION,
        "dataset_id": dataset_id,
        "feature_dtype": feature_dtype,
        "feature_representation": feature_representation,
        "generated_at": generated_at,
        "generator_version": generator_version,
        "input_difference": list(input_difference),
        "key_generation_policy": key_generation_policy,
        "license": DATASET_LICENSE,
        "repository_commit": repository_commit,
        "rounds": rounds,
        "schema": CLASSIFIER_MANIFEST_SCHEMA,
        "splits": dict(split_records),
        "task": CLASSIFIER_TASK,
        "word_order": word_order,
    }


def export_classifier_benchmark(
    configs: Sequence[DatasetConfig],
    output_dir: Path,
    *,
    dataset_id: str = CLASSIFIER_DATASET_ID,
    generated_at: str,
    repository_commit: str | None = None,
) -> dict[str, object]:
    """Generate disjoint classifier splits and write shards plus a manifest."""

    validate_split_configs(configs)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_records: dict[str, dict[str, object]] = {}
    seen_master_keys: set[tuple[int, int, int, int]] = set()
    reference: DatasetMetadata | None = None
    for config in configs:
        dataset = generate_dataset(
            config,
            include_master_keys=True,
            generated_at=generated_at,
            repository_commit=repository_commit,
        )
        if dataset.master_keys is None:
            raise ValueError("classifier export requires generated master keys")
        for master_key in dataset.master_keys:
            if master_key in seen_master_keys:
                raise ValueError(
                    "train, validation, and test master keys must be disjoint"
                )
            seen_master_keys.add(master_key)
        shard_path = output_dir / f"{dataset.metadata.split_role}.json"
        split_records[dataset.metadata.split_role] = write_classifier_shard(
            dataset,
            shard_path,
            dataset_id=dataset_id,
        )
        if reference is None:
            reference = dataset.metadata

    if reference is None:
        raise ValueError("at least one classifier split is required")
    manifest = classifier_manifest_document(
        split_records,
        dataset_id=dataset_id,
        rounds=reference.rounds,
        input_difference=reference.input_difference,
        word_order=reference.word_order,
        bit_order=reference.bit_order,
        feature_representation=reference.feature_representation,
        feature_dtype=reference.feature_dtype,
        key_generation_policy=reference.key_generation_policy,
        generator_version=reference.generator_version,
        repository_commit=repository_commit,
        generated_at=generated_at,
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_bytes(canonical_json(manifest).encode("utf-8"))
    return manifest


def load_classifier_manifest(
    path: Path,
    *,
    expected_dataset_id: str | None = CLASSIFIER_DATASET_ID,
) -> dict[str, Any]:
    """Load and validate the top-level classifier manifest schema."""

    try:
        payload = path.read_bytes()
        document = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read classifier manifest {path}") from error
    if not isinstance(document, Mapping):
        raise ValueError("classifier manifest must be a JSON object")
    if document.get("schema") != CLASSIFIER_MANIFEST_SCHEMA:
        raise ValueError("classifier manifest schema is not recognised")
    if document.get("cipher") != CIPHER_NAME:
        raise ValueError("classifier manifest cipher is not SPECK32/64")
    if document.get("dataset_format_version") != DATASET_FORMAT_VERSION:
        raise ValueError("classifier manifest format version is not supported")
    if document.get("task") != CLASSIFIER_TASK:
        raise ValueError("classifier manifest task is not related_vs_random")
    if document.get("license") != DATASET_LICENSE:
        raise ValueError("classifier manifest licence is not MIT")
    if document.get("generator_version") != GENERATOR_VERSION:
        raise ValueError("classifier manifest generator version is unsupported")
    if document.get("key_generation_policy") != KEY_GENERATION_POLICY:
        raise ValueError("classifier manifest key-generation policy is unsupported")
    if document.get("feature_representation") != FEATURE_REPRESENTATION:
        raise ValueError("classifier manifest feature representation is unsupported")
    repository_commit = document.get("repository_commit")
    if repository_commit is not None and not isinstance(repository_commit, str):
        raise ValueError("classifier manifest repository_commit must be a string or null")
    if not isinstance(document.get("generated_at"), str):
        raise ValueError("classifier manifest generated_at must be a string")
    dataset_id = document.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("classifier manifest dataset_id must be non-empty")
    if expected_dataset_id is not None and dataset_id != expected_dataset_id:
        raise ValueError(
            f"classifier manifest dataset_id must be {expected_dataset_id!r}"
        )
    if (
        isinstance(document.get("rounds"), bool)
        or not isinstance(document.get("rounds"), int)
        or not 1 <= document["rounds"] <= SPECK32_64_ROUND_COUNT
    ):
        raise ValueError(
            "classifier manifest rounds must be between 1 and "
            f"{SPECK32_64_ROUND_COUNT}"
        )
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
        raise ValueError("classifier manifest input_difference must have two words")
    splits = document.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(CLASSIFIER_SPLIT_ROLES):
        raise ValueError("classifier manifest must contain train, validation, and test")
    seeds: list[int] = []
    for role in CLASSIFIER_SPLIT_ROLES:
        record = _mapping(splits[role], f"splits[{role}]")
        for field in (
            "path",
            "data_checksum",
            "file_sha256",
            "class_balance",
            "sample_count",
            "seed",
            "split_role",
        ):
            if field not in record:
                raise ValueError(f"classifier manifest record is missing {field}")
        if record["split_role"] != role:
            raise ValueError(f"classifier manifest split role mismatch for {role}")
        if not isinstance(record["path"], str):
            raise ValueError("classifier manifest shard path must be a string")
        shard_path = Path(record["path"])
        if shard_path.name != record["path"] or shard_path.is_absolute():
            raise ValueError("classifier manifest shard path must be a local filename")
        if isinstance(record["seed"], bool) or not isinstance(record["seed"], int):
            raise ValueError("classifier manifest seed must be an integer")
        if not isinstance(record["sample_count"], int) or record["sample_count"] <= 0:
            raise ValueError("classifier manifest sample_count must be positive")
        class_balance = record["class_balance"]
        if not isinstance(class_balance, Mapping) or set(class_balance) != {"0", "1"}:
            raise ValueError("classifier manifest class_balance is invalid")
        if any(
            isinstance(class_balance[key], bool)
            or not isinstance(class_balance[key], int)
            or class_balance[key] < 0
            for key in ("0", "1")
        ):
            raise ValueError("classifier manifest class_balance is invalid")
        if sum(class_balance.values()) != record["sample_count"]:
            raise ValueError("classifier manifest class_balance is inconsistent")
        seeds.append(record["seed"])
    if len(set(seeds)) != len(seeds):
        raise ValueError("classifier manifest split seeds must be distinct")
    if payload != canonical_json(document).encode("utf-8"):
        raise ValueError("classifier manifest is not in canonical JSON form")
    return dict(document)


def verify_classifier_benchmark(
    manifest_path: Path,
    *,
    expected_dataset_id: str | None = CLASSIFIER_DATASET_ID,
    expected_rounds: int | None = None,
    expected_input_difference: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Verify every classifier shard against its manifest and shared protocol."""

    manifest = load_classifier_manifest(
        manifest_path,
        expected_dataset_id=expected_dataset_id,
    )
    if expected_rounds is not None and manifest["rounds"] != expected_rounds:
        raise ValueError("classifier benchmark rounds do not match the requested value")
    if expected_input_difference is not None:
        expected_difference = list(expected_input_difference)
        if manifest["input_difference"] != expected_difference:
            raise ValueError(
                "classifier benchmark input_difference does not match the requested value"
            )

    root = manifest_path.parent
    reference_metadata: DatasetMetadata | None = None
    splits = _mapping(manifest["splits"], "splits")
    for role in CLASSIFIER_SPLIT_ROLES:
        record = _mapping(splits[role], f"splits[{role}]")
        dataset = verify_classifier_shard(
            root / str(record["path"]),
            expected_dataset_id=str(manifest["dataset_id"]),
            expected_manifest_record=record,
        )
        if dataset.metadata.split_role != role:
            raise ValueError(f"classifier shard metadata role mismatch for {role}")
        if dataset.metadata.rounds != manifest["rounds"]:
            raise ValueError(f"classifier shard rounds mismatch for {role}")
        if list(dataset.metadata.input_difference) != manifest["input_difference"]:
            raise ValueError(f"classifier shard input difference mismatch for {role}")
        if dataset.metadata.generator_version != manifest["generator_version"]:
            raise ValueError(f"classifier shard generator version mismatch for {role}")
        if dataset.metadata.repository_commit != manifest["repository_commit"]:
            raise ValueError(f"classifier shard repository commit mismatch for {role}")
        if dataset.metadata.generated_at != manifest["generated_at"]:
            raise ValueError(f"classifier shard timestamp mismatch for {role}")
        if dataset.metadata.key_generation_policy != manifest["key_generation_policy"]:
            raise ValueError(f"classifier shard key policy mismatch for {role}")
        if dataset.metadata.feature_representation != manifest["feature_representation"]:
            raise ValueError(f"classifier shard feature representation mismatch for {role}")
        if reference_metadata is None:
            reference_metadata = dataset.metadata
        else:
            if dataset.metadata.word_order != reference_metadata.word_order:
                raise ValueError("classifier shards disagree on word order")
            if dataset.metadata.bit_order != reference_metadata.bit_order:
                raise ValueError("classifier shards disagree on bit order")
            if dataset.metadata.feature_dtype != reference_metadata.feature_dtype:
                raise ValueError("classifier shards disagree on feature dtype")
    return manifest
