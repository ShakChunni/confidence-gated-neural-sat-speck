"""Export one validation-only 64-pair Phase 6A/6B attack role."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from neural_sat_speck.data.phase6a import (
    Phase6AValidationAttackConfig,
    export_phase6a_validation_attack,
    verify_phase6a_validation_attack,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT / "configs" / "data" / "phase6a_selection_attack_r6.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"could not read configuration {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit("configuration must contain a JSON object")
    return value


def _repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _repository_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a validation-only Phase 6A attack benchmark."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--include-commit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = _repository_path(args.config).resolve()
    config = _load_json(config_path)
    if config.get("schema") != "neural_sat_speck.phase6a_validation_attack_config.v1":
        raise SystemExit("Phase 6A attack configuration schema is not recognised")
    try:
        phase_config = Phase6AValidationAttackConfig(
            instance_count=int(config["instance_count"]),
            seed=int(config["seed"]),
            split_role=str(config.get("split_role", "validation")),
            rounds=int(config.get("rounds", 6)),
            pair_count=int(config.get("pair_count", 64)),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"invalid Phase 6A attack configuration: {error}") from error
    dataset_id = config.get("dataset_id")
    generated_at = config.get("generated_at")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise SystemExit("dataset_id must be a non-empty string")
    if not isinstance(generated_at, str) or not generated_at:
        raise SystemExit("generated_at must be a non-empty string")
    output_dir = _repository_path(
        args.output_dir if args.output_dir is not None else str(config["output_dir"])
    ).resolve()
    manifest = export_phase6a_validation_attack(
        phase_config,
        output_dir,
        dataset_id=dataset_id,
        generated_at=generated_at,
        repository_commit=_repository_commit() if args.include_commit else None,
    )
    verify_phase6a_validation_attack(
        output_dir / "manifest.json",
        expected_dataset_id=dataset_id,
    )
    record = manifest["splits"]["validation"]
    print(f"Verified validation-only attack dataset: {dataset_id}")
    print(f"Manifest: {output_dir / 'manifest.json'}")
    print(
        f"validation: {record['instance_count']} instances x "
        f"{record['pair_count']} pairs; seed={record['seed']}"
    )


if __name__ == "__main__":
    main()
