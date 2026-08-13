"""Deterministic synthetic data interfaces for the Phase 1 baseline."""

from .attack import (
    AttackInstance,
    AttackInstanceConfig,
    AttackInstanceSet,
    AttackPair,
    export_attack_benchmark,
    generate_attack_instances,
    load_attack_shard,
    validate_attack_split_configs,
)
from .export import (
    export_classifier_benchmark,
    load_classifier_manifest,
    load_classifier_shard,
    verify_classifier_shard,
)
from .format import (
    ATTACK_DATASET_ID,
    CLASSIFIER_DATASET_ID,
    DATASET_FORMAT_VERSION,
    DATASET_LICENSE,
)
from .generator import (
    DEFAULT_INPUT_DIFFERENCE,
    SPLIT_ROLES,
    DatasetConfig,
    GeneratedDataset,
    assert_disjoint_master_keys,
    encode_ciphertext_pair,
    generate_dataset,
    validate_split_configs,
    word_to_bits_msb,
)
from .metadata import (
    FEATURE_WIDTH,
    GENERATOR_VERSION,
    KEY_GENERATION_POLICY,
    DatasetMetadata,
    metadata_to_json,
)

__all__ = [
    "ATTACK_DATASET_ID",
    "AttackInstance",
    "AttackInstanceConfig",
    "AttackInstanceSet",
    "AttackPair",
    "CLASSIFIER_DATASET_ID",
    "DATASET_FORMAT_VERSION",
    "DATASET_LICENSE",
    "DatasetConfig",
    "DatasetMetadata",
    "DEFAULT_INPUT_DIFFERENCE",
    "GENERATOR_VERSION",
    "GeneratedDataset",
    "KEY_GENERATION_POLICY",
    "FEATURE_WIDTH",
    "SPLIT_ROLES",
    "assert_disjoint_master_keys",
    "encode_ciphertext_pair",
    "export_attack_benchmark",
    "export_classifier_benchmark",
    "generate_attack_instances",
    "generate_dataset",
    "load_attack_shard",
    "load_classifier_manifest",
    "load_classifier_shard",
    "metadata_to_json",
    "validate_attack_split_configs",
    "validate_split_configs",
    "verify_classifier_shard",
    "word_to_bits_msb",
]
