"""Validation-only attack manifests for the Phase 6A/6B ranker study."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from .attack import (
    ATTACK_TASK,
    AttackInstanceConfig,
    generate_attack_instances,
    verify_attack_shard,
    write_attack_shard,
)
from .format import (
    CIPHER_NAME,
    DATASET_FORMAT_VERSION,
    DATASET_LICENSE,
    canonical_json,
)
from .generator import DEFAULT_INPUT_DIFFERENCE
from .metadata import BIT_ORDER, GENERATOR_VERSION, WORD_ORDER

PHASE6A_ATTACK_MANIFEST_SCHEMA: Final[str] = (
    "neural_sat_speck.phase6a_validation_attack_manifest.v1"
)
PHASE6A_ATTACK_DATASET_PREFIX: Final[str] = "speck32_64_attack_phase6a"
PHASE6A_ATTACK_ROUNDS: Final[int] = 6
PHASE6A_ATTACK_PAIR_COUNT: Final[int] = 64


@dataclass(frozen=True)
class Phase6AValidationAttackConfig:
    """Configuration for one validation-only 64-pair attack role."""

    instance_count: int
    seed: int
    split_role: str = "validation"
    rounds: int = PHASE6A_ATTACK_ROUNDS
    pair_count: int = PHASE6A_ATTACK_PAIR_COUNT

    def __post_init__(self) -> None:
        if isinstance(self.instance_count, bool) or not isinstance(self.instance_count, int):
            raise TypeError("instance_count must be an integer")
        if self.instance_count <= 0:
            raise ValueError("instance_count must be positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.split_role != "validation":
            raise ValueError("Phase 6A attack material must have validation role")
        if self.rounds != PHASE6A_ATTACK_ROUNDS:
            raise ValueError(f"rounds must be exactly {PHASE6A_ATTACK_ROUNDS}")
        if self.pair_count != PHASE6A_ATTACK_PAIR_COUNT:
            raise ValueError(
                f"pair_count must be exactly {PHASE6A_ATTACK_PAIR_COUNT}"
            )


def export_phase6a_validation_attack(
    config: Phase6AValidationAttackConfig,
    output_dir: Path,
    *,
    dataset_id: str,
    generated_at: str,
    repository_commit: str | None = None,
) -> dict[str, object]:
    """Generate a validation-only shard without creating a test split."""

    instance_set = generate_attack_instances(
        AttackInstanceConfig(
            instance_count=config.instance_count,
            pair_count=config.pair_count,
            rounds=config.rounds,
            seed=config.seed,
            split_role=config.split_role,
        ),
        generated_at=generated_at,
        repository_commit=repository_commit,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = output_dir / "validation.json"
    record = write_attack_shard(
        instance_set,
        shard_path,
        dataset_id=dataset_id,
    )
    manifest: dict[str, object] = {
        "bit_order": BIT_ORDER,
        "cipher": CIPHER_NAME,
        "dataset_format_version": DATASET_FORMAT_VERSION,
        "dataset_id": dataset_id,
        "generated_at": generated_at,
        "generator_version": GENERATOR_VERSION,
        "held_out_test_generated": False,
        "input_difference": list(DEFAULT_INPUT_DIFFERENCE),
        "license": DATASET_LICENSE,
        "pair_construction": "plaintext_difference",
        "repository_commit": repository_commit,
        "rounds": PHASE6A_ATTACK_ROUNDS,
        "schema": PHASE6A_ATTACK_MANIFEST_SCHEMA,
        "splits": {"validation": record},
        "task": ATTACK_TASK,
        "validation_only": True,
        "word_order": WORD_ORDER,
    }
    (output_dir / "manifest.json").write_bytes(
        canonical_json(manifest).encode("utf-8")
    )
    return manifest


def load_phase6a_validation_manifest(
    path: Path,
    *,
    expected_dataset_id: str | None = None,
) -> dict[str, Any]:
    """Load and validate a single-validation-role manifest."""

    try:
        payload = path.read_bytes()
        document = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read Phase 6A manifest {path}") from error
    if not isinstance(document, Mapping):
        raise ValueError("Phase 6A manifest must be an object")
    if document.get("schema") != PHASE6A_ATTACK_MANIFEST_SCHEMA:
        raise ValueError("Phase 6A manifest schema is not recognised")
    if expected_dataset_id is not None and document.get("dataset_id") != expected_dataset_id:
        raise ValueError("Phase 6A manifest dataset_id does not match")
    if document.get("cipher") != CIPHER_NAME or document.get("task") != ATTACK_TASK:
        raise ValueError("Phase 6A manifest cipher or task is unsupported")
    if document.get("dataset_format_version") != DATASET_FORMAT_VERSION:
        raise ValueError("Phase 6A manifest format version is unsupported")
    if document.get("license") != DATASET_LICENSE:
        raise ValueError("Phase 6A manifest licence is not MIT")
    if document.get("generator_version") != GENERATOR_VERSION:
        raise ValueError("Phase 6A manifest generator version is unsupported")
    if document.get("rounds") != PHASE6A_ATTACK_ROUNDS:
        raise ValueError("Phase 6A manifest rounds are unsupported")
    if document.get("input_difference") != list(DEFAULT_INPUT_DIFFERENCE):
        raise ValueError("Phase 6A manifest input difference is unsupported")
    if document.get("bit_order") != BIT_ORDER or document.get("word_order") != WORD_ORDER:
        raise ValueError("Phase 6A manifest word or bit order is unsupported")
    if document.get("validation_only") is not True or document.get("held_out_test_generated") is not False:
        raise ValueError("Phase 6A manifest must explicitly be validation-only")
    splits = document.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {"validation"}:
        raise ValueError("Phase 6A manifest must contain only validation")
    record = splits["validation"]
    if not isinstance(record, Mapping):
        raise ValueError("Phase 6A validation record must be an object")
    for field in (
        "data_checksum",
        "file_sha256",
        "instance_count",
        "pair_count",
        "path",
        "seed",
        "split_role",
    ):
        if field not in record:
            raise ValueError("Phase 6A validation record is incomplete")
    if record["split_role"] != "validation":
        raise ValueError("Phase 6A validation record has the wrong role")
    shard_path = Path(str(record["path"]))
    if shard_path.name != record["path"] or shard_path.is_absolute():
        raise ValueError("Phase 6A shard path must be a local filename")
    if payload != canonical_json(document).encode("utf-8"):
        raise ValueError("Phase 6A manifest is not canonical JSON")
    return dict(document)


def verify_phase6a_validation_attack(
    manifest_path: Path,
    *,
    expected_dataset_id: str | None = None,
) -> dict[str, Any]:
    """Verify the validation-only manifest and its standard attack shard."""

    manifest = load_phase6a_validation_manifest(
        manifest_path,
        expected_dataset_id=expected_dataset_id,
    )
    record = manifest["splits"]["validation"]
    verify_attack_shard(
        manifest_path.parent / str(record["path"]),
        expected_dataset_id=str(manifest["dataset_id"]),
        expected_manifest_record=record,
    )
    return manifest


__all__ = [
    "PHASE6A_ATTACK_DATASET_PREFIX",
    "PHASE6A_ATTACK_MANIFEST_SCHEMA",
    "PHASE6A_ATTACK_PAIR_COUNT",
    "PHASE6A_ATTACK_ROUNDS",
    "Phase6AValidationAttackConfig",
    "export_phase6a_validation_attack",
    "load_phase6a_validation_manifest",
    "verify_phase6a_validation_attack",
]
