"""Machine-readable metadata for deterministic synthetic datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Final

GENERATOR_VERSION: Final[str] = "0.1.0"
KEY_GENERATION_POLICY: Final[str] = "independent_random_key_per_sample"
WORD_ORDER: Final[str] = (
    "block=(x,y); master_key=(l[2],l[1],l[0],k[0])"
)
BIT_ORDER: Final[str] = (
    "first_ciphertext_x_msb_to_lsb, first_ciphertext_y_msb_to_lsb, "
    "second_ciphertext_x_msb_to_lsb, second_ciphertext_y_msb_to_lsb"
)
FEATURE_REPRESENTATION: Final[str] = (
    "four 16-bit ciphertext words encoded as 64 binary uint8 values"
)
FEATURE_DTYPE: Final[str] = "uint8"
FEATURE_WIDTH: Final[int] = 64


@dataclass(frozen=True)
class DatasetMetadata:
    """Metadata describing one generated dataset."""

    generator_version: str
    repository_commit: str | None
    generated_at: str
    sample_count: int
    class_balance: tuple[int, int]
    rounds: int
    input_difference: tuple[int, int]
    key_generation_policy: str
    seed: int
    split_role: str
    word_order: str
    bit_order: str
    feature_representation: str
    feature_dtype: str
    feature_shape: tuple[int, int]
    data_checksum: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible metadata mapping."""

        return {
            "generator_version": self.generator_version,
            "repository_commit": self.repository_commit,
            "generated_at": self.generated_at,
            "sample_count": self.sample_count,
            "class_balance": {
                "0": self.class_balance[0],
                "1": self.class_balance[1],
            },
            "rounds": self.rounds,
            "input_difference": list(self.input_difference),
            "key_generation_policy": self.key_generation_policy,
            "seed": self.seed,
            "split_role": self.split_role,
            "word_order": self.word_order,
            "bit_order": self.bit_order,
            "feature_representation": self.feature_representation,
            "feature_dtype": self.feature_dtype,
            "feature_shape": list(self.feature_shape),
            "data_checksum": self.data_checksum,
        }


def metadata_to_json(
    metadata: DatasetMetadata,
    *,
    indent: int | None = 2,
) -> str:
    """Serialize dataset metadata deterministically as JSON."""

    return json.dumps(metadata.to_dict(), indent=indent, sort_keys=True)
