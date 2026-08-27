"""Export and verify the Phase 6A/6B same-key group dataset."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from neural_sat_speck.data.pair_groups import (
    PAIR_GROUP_DATASET_ID,
    PairGroupConfig,
    export_pair_group_benchmark,
    verify_pair_group_benchmark,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "data" / "phase6a_pair_group_r5_g64.json"


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
        description="Export the new Phase 6 pair-aware group dataset."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--include-commit",
        action="store_true",
        help="record the current repository commit in generated metadata",
    )
    return parser.parse_args()


def _config_split(config: dict[str, Any], role: str) -> PairGroupConfig:
    splits = config.get("splits")
    if not isinstance(splits, dict) or role not in splits:
        raise SystemExit(f"configuration is missing the {role} split")
    record = splits[role]
    if not isinstance(record, dict):
        raise SystemExit(f"configuration split {role} must be an object")
    try:
        return PairGroupConfig(
            group_count=int(record["group_count"]),
            group_size=int(config["group_size"]),
            rounds=int(config["rounds"]),
            seed=int(record["seed"]),
            split_role=role,
            input_difference=tuple(config["input_difference"]),  # type: ignore[arg-type]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"invalid {role} pair-group configuration: {error}") from error


def main() -> None:
    args = _parse_args()
    config_path = _repository_path(args.config).resolve()
    config = _load_json(config_path)
    if config.get("schema") != "neural_sat_speck.phase6a_pair_group_config.v1":
        raise SystemExit("pair-group configuration schema is not recognised")
    dataset_id = str(config.get("dataset_id", PAIR_GROUP_DATASET_ID))
    generated_at = config.get("generated_at")
    if not isinstance(generated_at, str) or not generated_at:
        raise SystemExit("generated_at must be a non-empty string")
    output_dir = _repository_path(
        args.output_dir if args.output_dir is not None else str(config["output_dir"])
    ).resolve()
    train_config = _config_split(config, "train")
    validation_config = _config_split(config, "validation")
    repository_commit = _repository_commit() if args.include_commit else None
    manifest = export_pair_group_benchmark(
        (train_config, validation_config),
        output_dir,
        dataset_id=dataset_id,
        generated_at=generated_at,
        repository_commit=repository_commit,
    )
    verify_pair_group_benchmark(
        output_dir / "manifest.json",
        expected_dataset_id=dataset_id,
    )
    print(f"Verified pair-group dataset: {manifest['dataset_id']}")
    print(f"Manifest: {output_dir / 'manifest.json'}")
    for role, record in manifest["splits"].items():
        print(
            f"{role}: {record['group_count']} groups x {record['group_size']} pairs; "
            f"seed={record['seed']}"
        )


if __name__ == "__main__":
    main()
