"""Deterministic Dataset v1 export for the classifier benchmark."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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
    assert_disjoint_master_keys,
    dataset_content_checksum,
    generate_dataset,
    validate_split_configs,
)
from .metadata import GENERATOR_VERSION, DatasetMetadata


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


def load_classifier_shard(path: Path) -> GeneratedDataset:
    """Load a classifier shard without any neural-framework dependency."""

    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != CLASSIFIER_SHARD_SCHEMA:
        raise ValueError("classifier shard schema is not recognised")
    if document.get("task") != CLASSIFIER_TASK:
        raise ValueError("classifier shard task is not related_vs_random")

    metadata_mapping = document["metadata"]
    metadata = DatasetMetadata(
        generator_version=str(metadata_mapping["generator_version"]),
        repository_commit=metadata_mapping["repository_commit"],
        generated_at=str(metadata_mapping["generated_at"]),
        sample_count=int(metadata_mapping["sample_count"]),
        class_balance=(
            int(metadata_mapping["class_balance"]["0"]),
            int(metadata_mapping["class_balance"]["1"]),
        ),
        rounds=int(metadata_mapping["rounds"]),
        input_difference=(
            int(metadata_mapping["input_difference"][0]),
            int(metadata_mapping["input_difference"][1]),
        ),
        key_generation_policy=str(metadata_mapping["key_generation_policy"]),
        seed=int(metadata_mapping["seed"]),
        split_role=str(metadata_mapping["split_role"]),
        word_order=str(metadata_mapping["word_order"]),
        bit_order=str(metadata_mapping["bit_order"]),
        feature_representation=str(metadata_mapping["feature_representation"]),
        feature_dtype=str(metadata_mapping["feature_dtype"]),
        feature_shape=(
            int(metadata_mapping["feature_shape"][0]),
            int(metadata_mapping["feature_shape"][1]),
        ),
        data_checksum=str(metadata_mapping["data_checksum"]),
    )
    features = tuple(tuple(int(bit) for bit in row) for row in document["features"])
    labels = tuple(int(label) for label in document["labels"])
    return GeneratedDataset(
        features=features,
        labels=labels,
        metadata=metadata,
    )


def verify_classifier_shard(path: Path) -> GeneratedDataset:
    """Load a shard and check its file and content checksums."""

    payload = path.read_bytes()
    dataset = load_classifier_shard(path)
    expected = dataset.metadata.data_checksum
    actual = dataset_content_checksum(dataset.features, dataset.labels)
    if actual != expected:
        raise ValueError("classifier shard data checksum does not match contents")

    document = json.loads(payload.decode("utf-8"))
    reconstructed = classifier_shard_bytes(dataset, dataset_id=document["dataset_id"])
    if sha256_bytes(payload) != sha256_bytes(reconstructed):
        raise ValueError("classifier shard is not in canonical JSON form")
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
    datasets = [
        generate_dataset(
            config,
            include_master_keys=True,
            generated_at=generated_at,
            repository_commit=repository_commit,
        )
        for config in configs
    ]
    assert_disjoint_master_keys(datasets)

    output_dir.mkdir(parents=True, exist_ok=True)
    split_records: dict[str, dict[str, object]] = {}
    for dataset in datasets:
        shard_path = output_dir / f"{dataset.metadata.split_role}.json"
        split_records[dataset.metadata.split_role] = write_classifier_shard(
            dataset,
            shard_path,
            dataset_id=dataset_id,
        )

    reference = datasets[0].metadata
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


def load_classifier_manifest(path: Path) -> dict[str, Any]:
    """Load a classifier manifest from disk."""

    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != CLASSIFIER_MANIFEST_SCHEMA:
        raise ValueError("classifier manifest schema is not recognised")
    return document
