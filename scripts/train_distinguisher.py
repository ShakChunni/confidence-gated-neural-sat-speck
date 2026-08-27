"""Run the reproducible Phase 3 classifier baseline.

The script consumes an already exported and checksummed classifier benchmark.
Dataset generation is intentionally a separate command so training cannot
silently change the examples used by a reported result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from neural_sat_speck.neural.dataset import (
    ClassifierSplits,
    create_dataloader,
    load_classifier_splits,
)
from neural_sat_speck.neural.model import GohrResNetConfig
from neural_sat_speck.neural.train import (
    TrainingConfig,
    build_model,
    evaluate,
    save_checkpoint,
    select_device,
    train_distinguisher,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "training" / "phase3_baseline.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"Could not read JSON configuration {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"Configuration {path} must contain a JSON object")
    return value


def _repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        description=(
            "Train the Phase 3 Gohr-style distinguisher from verified "
            "Dataset v1-compatible classifier shards."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Committed training JSON (default: {DEFAULT_CONFIG.relative_to(REPOSITORY_ROOT)})",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Override the classifier shard directory from the config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the checkpoint path from the config.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default=None,
        help="Override the configured device.",
    )
    return parser.parse_args()


def _dataset_metadata(splits: ClassifierSplits) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    manifest_splits = splits.manifest["splits"]
    for role, dataset in splits.as_dict().items():
        metadata = dataset.metadata
        manifest_record = manifest_splits[role]
        result[role] = {
            "sample_count": len(dataset),
            "class_balance": metadata.get("class_balance"),
            "data_checksum": metadata.get("data_checksum"),
            "shard_path": manifest_record["path"],
            "shard_file_sha256": manifest_record["file_sha256"],
            "feature_dtype": metadata.get("feature_dtype"),
            "feature_shape": metadata.get("feature_shape"),
            "generated_at": metadata.get("generated_at"),
            "generator_version": metadata.get("generator_version"),
            "input_difference": metadata.get("input_difference"),
            "repository_commit": metadata.get("repository_commit"),
            "rounds": metadata.get("rounds"),
            "seed": metadata.get("seed"),
            "split_role": metadata.get("split_role"),
            "bit_order": metadata.get("bit_order"),
            "word_order": metadata.get("word_order"),
        }
    return result


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    args = _parse_args()
    config_path = _repository_path(args.config).resolve()
    config = _load_json(config_path)
    if config.get("schema") != "neural_sat_speck.phase3_training_config.v1":
        raise SystemExit("training config schema is not recognised")
    if config.get("phase") != 3:
        raise SystemExit("the selected configuration is not a Phase 3 config")

    dataset_root = (
        args.dataset_root
        if args.dataset_root is not None
        else Path(str(config["dataset_root"]))
    )
    dataset_root = _repository_path(dataset_root).resolve()
    output_path = (
        args.output
        if args.output is not None
        else Path(str(config["output"]))
    )
    output_path = _repository_path(output_path).resolve()

    expected_difference = tuple(
        int(word) for word in config["expected_input_difference"]
    )
    if len(expected_difference) != 2:
        raise SystemExit("expected_input_difference must contain two words")
    model_config = GohrResNetConfig.from_dict(dict(config["model"]))
    training_config = TrainingConfig.from_dict(dict(config["training"]))
    if args.device is not None:
        training_config = replace(training_config, device=args.device)

    print("=" * 70)
    print("SPECK32/64 NEURAL DISTINGUISHER TRAINING (PHASE 3)")
    print(f"Config: {config_path}")
    print(f"Dataset manifest: {dataset_root / 'manifest.json'}")
    print(f"Expected rounds: {config['expected_rounds']}")
    print(f"Expected input difference: {expected_difference}")
    print("=" * 70)

    started = time.perf_counter()
    splits = load_classifier_splits(
        dataset_root,
        expected_dataset_id=str(config["expected_dataset_id"]),
        expected_rounds=int(config["expected_rounds"]),
        expected_input_difference=expected_difference,
    )
    if bool(config.get("require_repository_commit", False)) and not splits.manifest.get(
        "repository_commit"
    ):
        raise SystemExit(
            "the Phase 3 baseline requires a dataset manifest with a repository "
            "commit; export with --include-commit"
        )
    load_seconds = time.perf_counter() - started
    print(
        "Loaded and verified train/validation/test shards in "
        f"{load_seconds:.3f}s: "
        f"{len(splits.train)}/{len(splits.validation)}/{len(splits.test)} samples"
    )

    model = build_model(
        model_config,
        seed=training_config.seed,
        deterministic=training_config.deterministic,
    )
    print(
        f"Model input shape: {model.input_shape}; "
        f"trainable parameters: {model.parameter_count:,}"
    )
    print(f"Training configuration: {training_config.to_dict()}")

    started = time.perf_counter()
    training_result = train_distinguisher(
        model,
        splits.train,
        splits.validation,
        training_config,
    )
    train_seconds = time.perf_counter() - started
    device = select_device(str(training_result["device"]))
    criterion = nn.BCEWithLogitsLoss()

    train_metrics = evaluate(
        model,
        create_dataloader(
            splits.train,
            batch_size=training_config.batch_size,
            shuffle=False,
        ),
        criterion,
        device,
    )
    test_metrics = evaluate(
        model,
        create_dataloader(
            splits.test,
            batch_size=training_config.batch_size,
            shuffle=False,
        ),
        criterion,
        device,
    )

    print("\n--- Validation history ---")
    for epoch_data in training_result["history"]:
        print(
            f"Epoch {epoch_data['epoch']:2d}/{training_config.epochs} | "
            f"Train loss {epoch_data['train_loss']:.6f} | "
            f"Val loss {epoch_data['val_loss']:.6f} | "
            f"Val accuracy {epoch_data['val_accuracy']:.4f} | "
            f"Val AUC {epoch_data['val_auc']:.6f}"
        )
    print(f"Training time: {train_seconds:.3f}s")
    print(f"Selected validation AUC: {training_result['best_val_auc']:.6f}")
    print(f"Held-out test AUC: {test_metrics['auc']:.6f}")

    manifest_path = dataset_root / "manifest.json"
    metadata: dict[str, Any] = {
        "phase": 3,
        "protocol": "Dataset v1-compatible related_vs_random classifier",
        "dataset_id": str(config["expected_dataset_id"]),
        "dataset_manifest": _relative_or_absolute(manifest_path),
        "dataset_manifest_sha256": _sha256_file(manifest_path),
        "dataset": _dataset_metadata(splits),
        "model_config": model.config.to_dict(),
        "training_config": training_config.to_dict(),
        "training_result": training_result,
        "train_metrics_after_selection": train_metrics,
        "test_metrics_after_selection": test_metrics,
        "evaluation_protocol": {
            "selection_split": "validation",
            "final_report_split": "test",
            "test_used_for_model_selection": False,
            "random_baseline": {
                "accuracy": 0.5,
                "auc": 0.5,
                "definition": "chance expectation for balanced binary classification",
            },
        },
        "provenance": {
            "repository_commit": _repository_commit(),
            "config_path": _relative_or_absolute(config_path),
            "config_sha256": _sha256_file(config_path),
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy_version": np.__version__,
            "torch_version": str(torch.__version__),
            "device": str(device),
            "deterministic_requested": training_config.deterministic,
            "command": [sys.executable, *sys.argv],
        },
    }
    save_checkpoint(model, output_path, metadata=metadata)
    metadata["checkpoint_path"] = _relative_or_absolute(output_path)
    metadata["checkpoint_sha256"] = _sha256_file(output_path)
    summary_path = output_path.with_suffix(".json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Checkpoint saved to: {output_path}")
    print(f"Run summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
