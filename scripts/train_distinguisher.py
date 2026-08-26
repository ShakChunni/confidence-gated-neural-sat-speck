"""Train and evaluate a compact GohrResNet differential distinguisher on SPECK32/64."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from neural_sat_speck.data.generator import DatasetConfig, generate_dataset
from neural_sat_speck.neural.dataset import SpeckClassifierDataset
from neural_sat_speck.neural.model import GohrResNet, GohrResNetConfig
from neural_sat_speck.neural.train import (
    TrainingConfig,
    save_checkpoint,
    train_distinguisher,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a GohrResNet differential distinguisher on SPECK32/64."
    )
    parser.add_argument("--rounds", type=int, default=7, help="Cipher round count (default: 7)")
    parser.add_argument("--train-samples", type=int, default=10000, help="Number of training samples (default: 10000)")
    parser.add_argument("--val-samples", type=int, default=2000, help="Number of validation samples (default: 2000)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--filters", type=int, default=32, help="Convolutional filter count (default: 32)")
    parser.add_argument("--blocks", type=int, default=4, help="Residual block count (default: 4)")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'auto', 'mps', 'cpu' (default: auto)")
    parser.add_argument("--output", type=str, default="models/speck32_64_r7_distinguisher.pt", help="Checkpoint path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = REPOSITORY_ROOT / args.output

    print("=" * 70)
    print("SPECK32/64 NEURAL DISTINGUISHER TRAINING (PHASE 3)")
    print(f"Rounds: {args.rounds} | Train Samples: {args.train_samples} | Val Samples: {args.val_samples}")
    print(f"Filters: {args.filters} | Blocks: {args.blocks} | Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print("=" * 70)

    print("\n[1/3] Generating synthetic Dataset v1 classifier samples...")
    t0 = time.perf_counter()
    train_config = DatasetConfig(
        sample_count=args.train_samples,
        rounds=args.rounds,
        seed=1001,
        split_role="train",
    )
    val_config = DatasetConfig(
        sample_count=args.val_samples,
        rounds=args.rounds,
        seed=2001,
        split_role="validation",
    )

    train_gen = generate_dataset(train_config)
    val_gen = generate_dataset(val_config)

    train_dataset = SpeckClassifierDataset.from_generated_dataset(train_gen)
    val_dataset = SpeckClassifierDataset.from_generated_dataset(val_gen)
    gen_time = time.perf_counter() - t0
    print(f"  -> Generated {len(train_dataset)} train and {len(val_dataset)} val pairs in {gen_time:.2f}s")

    print("\n[2/3] Initializing GohrResNet architecture...")
    model_config = GohrResNetConfig(
        num_filters=args.filters,
        num_blocks=args.blocks,
    )
    model = GohrResNet(model_config)
    print(f"  -> Total trainable parameters: {model.parameter_count:,}")

    train_config_obj = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )

    print(f"\n[3/3] Training model across {args.epochs} epochs...")
    t_train = time.perf_counter()
    results = train_distinguisher(model, train_dataset, val_dataset, train_config_obj)
    train_duration = time.perf_counter() - t_train

    print("\n--- Training History ---")
    for epoch_data in results["history"]:
        print(
            f"Epoch {epoch_data['epoch']:2d}/{args.epochs} | "
            f"Train Loss: {epoch_data['train_loss']:.4f} | "
            f"Val Loss: {epoch_data['val_loss']:.4f} | "
            f"Val Acc: {epoch_data['val_accuracy'] * 100:.2f}% | "
            f"Val AUC: {epoch_data['val_auc']:.4f}"
        )

    print(f"\nTraining completed in {train_duration:.2f}s")
    print(f"Best Validation ROC-AUC: {results['best_val_auc']:.4f}")

    metadata = {
        "rounds": args.rounds,
        "best_val_auc": results["best_val_auc"],
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "history": results["history"],
    }
    save_checkpoint(model, output_path, metadata=metadata)
    print(f"Model checkpoint saved to: {output_path}")

    # Output JSON summary
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Training metadata saved to: {summary_path}")


if __name__ == "__main__":
    main()
