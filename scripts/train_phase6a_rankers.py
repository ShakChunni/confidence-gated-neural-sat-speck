"""Train the predeclared Phase 6A/6B B1, B2, and B3 rankers."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch

from neural_sat_speck.data.pair_groups import (
    PAIR_GROUP_DATASET_ID,
    load_pair_group_shard,
    verify_pair_group_benchmark,
)
from neural_sat_speck.neural.group_dataset import PairGroupDataset
from neural_sat_speck.neural.group_model import PAIR_GROUP_VARIANTS
from neural_sat_speck.neural.group_train import (
    PairGroupTrainingConfig,
    build_transferred_pair_group_model,
    save_pair_group_checkpoint,
    train_pair_group_ranker,
)
from neural_sat_speck.neural.train import load_checkpoint

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT / "configs" / "training" / "phase6a_pair_group_rankers.json"
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


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the frozen B1/B2/B3 pair-group ranker family."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda"),
        default=None,
        help="override the frozen training device for a feasibility run",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = _repository_path(args.config).resolve()
    config = _load_json(config_path)
    if config.get("schema") != "neural_sat_speck.phase6a_group_training_config.v1":
        raise SystemExit("Phase 6A group training schema is not recognised")

    dataset_root = _repository_path(str(config["dataset_root"])).resolve()
    expected_dataset_id = str(config.get("expected_dataset_id", PAIR_GROUP_DATASET_ID))
    manifest = verify_pair_group_benchmark(
        dataset_root / "manifest.json",
        expected_dataset_id=expected_dataset_id,
    )
    train_shard = load_pair_group_shard(
        dataset_root / str(manifest["splits"]["train"]["path"]),
        expected_dataset_id=expected_dataset_id,
    )
    validation_shard = load_pair_group_shard(
        dataset_root / str(manifest["splits"]["validation"]["path"]),
        expected_dataset_id=expected_dataset_id,
    )
    train_dataset = PairGroupDataset.from_generated_dataset(train_shard)
    validation_dataset = PairGroupDataset.from_generated_dataset(validation_shard)

    b0_path = _repository_path(str(config["baseline_checkpoint"])).resolve()
    if not b0_path.is_file():
        raise SystemExit(f"frozen B0 checkpoint does not exist: {b0_path}")
    baseline, baseline_metadata = load_checkpoint(b0_path, device="cpu")
    if baseline_metadata.get("phase") != 3:
        raise SystemExit("baseline checkpoint is not the Phase 3 B0 model")
    if baseline_metadata.get("dataset_id") != config.get("expected_model_dataset_id"):
        raise SystemExit("baseline checkpoint dataset_id is not the expected B0 dataset")
    if baseline_metadata.get("dataset", {}).get("validation", {}).get("rounds") != config.get("expected_model_rounds"):
        raise SystemExit("baseline checkpoint rounds do not match the B1 transfer source")

    training_record = config.get("training")
    if not isinstance(training_record, dict):
        raise SystemExit("training configuration must be an object")
    if args.device is not None:
        training_record = {**training_record, "device": args.device}
    training_config = PairGroupTrainingConfig(
        epochs=int(training_record["epochs"]),
        batch_size=int(training_record["batch_size"]),
        learning_rate=float(training_record["learning_rate"]),
        weight_decay=float(training_record["weight_decay"]),
        lr_decay=float(training_record["lr_decay"]),
        seed=int(training_record["seed"]),
        device=str(training_record["device"]),
        deterministic=bool(training_record["deterministic"]),
    )

    output_paths = config.get("outputs")
    if not isinstance(output_paths, dict):
        raise SystemExit("outputs must map model variants to checkpoint paths")
    variants = tuple(PAIR_GROUP_VARIANTS)
    print(f"Verified group dataset: {expected_dataset_id}")
    print(
        f"train={len(train_dataset)} groups, validation={len(validation_dataset)} groups"
    )
    print(f"B0 transfer checkpoint: {b0_path}")
    for variant in variants:
        output_path = _repository_path(str(output_paths[variant])).resolve()
        started = time.perf_counter()
        model = build_transferred_pair_group_model(
            baseline,
            variant=variant,
            config=training_config,
        )
        training_result = train_pair_group_ranker(
            model,
            train_dataset,
            validation_dataset,
            training_config,
        )
        training_seconds = time.perf_counter() - started
        metadata: dict[str, Any] = {
            "phase": 6,
            "study": "phase6a_b1_b2_b3_pair_aware_rankers",
            "model_id": model.model_id,
            "variant": variant,
            "dataset_id": expected_dataset_id,
            "dataset_manifest": _relative_or_absolute(dataset_root / "manifest.json"),
            "dataset_manifest_sha256": _sha256_file(dataset_root / "manifest.json"),
            "dataset_splits": manifest["splits"],
            "baseline_checkpoint": _relative_or_absolute(b0_path),
            "baseline_checkpoint_sha256": _sha256_file(b0_path),
            "transfer_initialization": "frozen Phase 3 B0 encoder",
            "checkpoint_selection": "minimum group-validation BCE; earliest exact tie",
            "training_result": training_result,
            "training_time_seconds": training_seconds,
            "provenance": {
                "config_path": _relative_or_absolute(config_path),
                "config_sha256": _sha256_file(config_path),
                "python_version": sys.version,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "torch_version": str(torch.__version__),
                "device": training_config.device,
                "command": [sys.executable, *sys.argv],
            },
        }
        save_pair_group_checkpoint(model, output_path, metadata=metadata)
        metadata["checkpoint_path"] = _relative_or_absolute(output_path)
        metadata["checkpoint_sha256"] = _sha256_file(output_path)
        summary_path = output_path.with_suffix(".json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"{variant}: best epoch={training_result['best_epoch']}, "
            f"val BCE={training_result['best_val_bce']:.6f}, "
            f"val AUC={training_result['history'][training_result['best_epoch'] - 1]['val_auc']:.6f}, "
            f"time={training_seconds:.2f}s"
        )
        print(f"  checkpoint: {output_path}")


if __name__ == "__main__":
    main()
