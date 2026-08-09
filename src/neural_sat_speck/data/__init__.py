"""Deterministic synthetic data interfaces for the Phase 1 baseline."""

from .generator import (
    DEFAULT_INPUT_DIFFERENCE,
    SPLIT_ROLES,
    DatasetConfig,
    GeneratedDataset,
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
    "DatasetConfig",
    "DatasetMetadata",
    "DEFAULT_INPUT_DIFFERENCE",
    "GENERATOR_VERSION",
    "GeneratedDataset",
    "KEY_GENERATION_POLICY",
    "FEATURE_WIDTH",
    "SPLIT_ROLES",
    "encode_ciphertext_pair",
    "generate_dataset",
    "metadata_to_json",
    "validate_split_configs",
    "word_to_bits_msb",
]
