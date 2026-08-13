"""Architecture-neutral Dataset v1 serialization helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Final

from neural_sat_speck.speck.primitives import mask_word

DATASET_FORMAT_VERSION: Final[str] = "1.0.0"
CIPHER_NAME: Final[str] = "SPECK32/64"
DATASET_LICENSE: Final[str] = "MIT"

CLASSIFIER_DATASET_ID: Final[str] = "speck32_64_classifier_v1"
CLASSIFIER_TASK: Final[str] = "related_vs_random"
CLASSIFIER_SHARD_SCHEMA: Final[str] = "neural_sat_speck.classifier_shard.v1"
CLASSIFIER_MANIFEST_SCHEMA: Final[str] = (
    "neural_sat_speck.classifier_manifest.v1"
)

ATTACK_DATASET_ID: Final[str] = "speck32_64_attack_v1"
ATTACK_TASK: Final[str] = "single_key_related_pairs"
ATTACK_SHARD_SCHEMA: Final[str] = "neural_sat_speck.attack_shard.v1"
ATTACK_MANIFEST_SCHEMA: Final[str] = "neural_sat_speck.attack_manifest.v1"
PLAINTEXT_DIFFERENCE_POLICY: Final[str] = "plaintext_difference"

JSON_SEPARATORS: Final[tuple[str, str]] = (",", ": ")


def sha256_bytes(payload: bytes) -> str:
    """Return the lowercase SHA-256 hex digest of a byte string."""

    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Mapping[str, object] | Sequence[object]) -> str:
    """Serialize a JSON value with stable key order and a trailing newline."""

    return json.dumps(value, indent=2, sort_keys=True, separators=JSON_SEPARATORS) + "\n"


def word_to_hex(word: int) -> str:
    """Encode one masked 16-bit word as four lowercase hexadecimal digits."""

    return f"{mask_word(word):04x}"


def hex_to_word(value: str) -> int:
    """Decode a four-digit hexadecimal word into a masked 16-bit integer."""

    if not isinstance(value, str):
        raise TypeError("hex word must be a string")
    if len(value) != 4:
        raise ValueError("hex word must contain exactly four digits")
    try:
        parsed = int(value, 16)
    except ValueError as error:
        raise ValueError("hex word must be hexadecimal") from error
    return mask_word(parsed)


def words_to_hex(words: Sequence[int]) -> list[str]:
    """Encode a sequence of 16-bit words as hexadecimal strings."""

    return [word_to_hex(word) for word in words]


def hex_to_words(values: Sequence[str]) -> tuple[int, ...]:
    """Decode a sequence of hexadecimal words."""

    return tuple(hex_to_word(value) for value in values)
