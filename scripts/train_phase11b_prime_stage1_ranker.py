#!/usr/bin/env python3
"""Train the Phase 11B-prime stage-1 (six-round) pair-group ranker.

The hierarchical seven-round attack ranks the last round key by decrypting one
round and asking a *six-round* model whether the resulting pair is related; the
round key below it is then ranked by the frozen five-round model on the descended
instance (see `docs/PHASE11_SCALING_STRATEGY.md`). This script trains that
six-round ranker on the six-round pair-group family.

The shared encoder is warm-started from the frozen five-round B0 checkpoint by
default, which is a deliberate round-count transfer: the encoder recognises ARX
differential structure at the shorter round count and is then adapted to the
six-round residual. Pass `--no-transfer` to train from random initialization.

The resulting checkpoint keeps the `b1_mean` variant identity, so the six-round
provenance is recorded in the sidecar manifest written next to the checkpoint
(rounds, dataset id, baseline hash, training configuration).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from neural_sat_speck.data.pair_groups import verify_pair_group_benchmark
from neural_sat_speck.neural.group_dataset import PairGroupDataset
from neural_sat_speck.neural.group_model import PairGroupRanker, build_pair_group_model
from neural_sat_speck.neural.group_train import (
    PairGroupTrainingConfig,
    save_pair_group_checkpoint,
    train_pair_group_ranker,
)
from neural_sat_speck.neural.train import load_checkpoint

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPOSITORY_ROOT / "datasets" / "phase11b_prime_pair_group_r6_g64",
    )
    parser.add_argument("--expected-dataset-id", type=str, required=True)
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=REPOSITORY_ROOT / "models" / "speck32_64_phase3_r5_distinguisher.pt",
        help="frozen five-round B0 checkpoint used for warm-start transfer",
    )
    parser.add_argument("--no-transfer", action="store_true")
    parser.add_argument("--variant", type=str, default="b1_mean")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "models" / "speck32_64_phase11b_prime_stage1_r6_b1_mean.pt",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-decay", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=64001)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root: Path = args.dataset_root
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"dataset manifest not found at {manifest_path}; "
            "run scripts/generate_phase11b_prime_data.py first"
        )

    manifest = verify_pair_group_benchmark(
        manifest_path,
        expected_dataset_id=args.expected_dataset_id,
    )
    train_dataset = PairGroupDataset.from_shard(
        dataset_root / "train.npz",
        expected_dataset_id=args.expected_dataset_id,
    )
    validation_dataset = PairGroupDataset.from_shard(
        dataset_root / "validation.npz",
        expected_dataset_id=args.expected_dataset_id,
    )

    training_config = PairGroupTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_decay=args.lr_decay,
        seed=args.seed,
        device=args.device,
        deterministic=True,
    )

    baseline_sha256: str | None = None
    if args.no_transfer:
        model = PairGroupRanker(variant=args.variant)
        initialization = "random"
    else:
        baseline_path: Path = args.baseline_checkpoint
        if not baseline_path.exists():
            raise SystemExit(f"baseline checkpoint not found at {baseline_path}")
        baseline_sha256 = _sha256_file(baseline_path)
        baseline_payload = torch.load(baseline_path, map_location="cpu", weights_only=False)
        if isinstance(baseline_payload, dict) and "variant" in baseline_payload:
            # Warm start from an existing pair-group ranker: reuse its shared encoder.
            source = PairGroupRanker(variant=str(baseline_payload["variant"]))
            source.load_state_dict(baseline_payload["state_dict"])
            model = PairGroupRanker(variant=args.variant)
            model.encoder.load_state_dict(source.encoder.state_dict())
            initialization = "transferred five-round pair-group encoder"
        else:
            baseline, _ = load_checkpoint(baseline_path)
            model = build_pair_group_model(
                baseline,
                variant=args.variant,
                seed=training_config.seed,
                deterministic=training_config.deterministic,
            )
            initialization = "transferred five-round B0 encoder"

    print(
        f"training stage-1 ranker: variant={args.variant} initialization={initialization} "
        f"train_groups={len(train_dataset)} validation_groups={len(validation_dataset)}"
    )
    started = time.perf_counter()
    training_result = train_pair_group_ranker(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        config=training_config,
    )
    training_seconds = time.perf_counter() - started

    best_epoch = int(training_result["best_epoch"])
    metadata: dict[str, Any] = {
        "phase": "11B-prime",
        "study": "phase11b_prime_hierarchical_seven_round_stage1",
        "stage": 1,
        "model_id": model.model_id,
        "variant": args.variant,
        "rounds": int(manifest["rounds"]),
        "dataset_id": args.expected_dataset_id,
        "dataset_manifest": _relative_or_absolute(manifest_path),
        "dataset_manifest_sha256": _sha256_file(manifest_path),
        "dataset_splits": manifest["splits"],
        "initialization": initialization,
        "baseline_checkpoint": (
            None if args.no_transfer else _relative_or_absolute(args.baseline_checkpoint)
        ),
        "baseline_checkpoint_sha256": baseline_sha256,
        "checkpoint_selection": "minimum group-validation BCE; earliest exact tie",
        "training_result": training_result,
        "training_time_seconds": training_seconds,
        "provenance": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch_version": str(torch.__version__),
            "device": training_config.device,
            "git_commit": _git_commit(),
            "command": [sys.executable, *sys.argv],
        },
    }
    save_pair_group_checkpoint(model, args.output, metadata=metadata)
    metadata["checkpoint_path"] = _relative_or_absolute(args.output)
    metadata["checkpoint_sha256"] = _sha256_file(args.output)
    summary_path = args.output.with_suffix(".json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    history_entry = training_result["history"][best_epoch - 1]
    print(
        f"best epoch={best_epoch}, val BCE={training_result['best_val_bce']:.6f}, "
        f"val AUC={history_entry['val_auc']:.6f}, time={training_seconds:.2f}s"
    )
    print(f"  checkpoint: {_relative_or_absolute(args.output)}")
    print(f"  manifest:   {_relative_or_absolute(summary_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
