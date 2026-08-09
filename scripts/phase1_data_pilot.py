"""Run a small reproducible Phase 1 data-generation pilot."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import tracemalloc
from pathlib import Path

from neural_sat_speck.data.generator import DatasetConfig, generate_dataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _repository_commit() -> str | None:
    """Return the current Git commit when the repository is available."""

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
    return result.stdout.strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and measure a small SPECK data pilot."
    )
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument(
        "--split-role",
        choices=("train", "validation", "test"),
        default="train",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = DatasetConfig(
        sample_count=args.sample_count,
        rounds=args.rounds,
        seed=args.seed,
        split_role=args.split_role,
    )

    tracemalloc.start()
    started = time.perf_counter()
    dataset = generate_dataset(
        config,
        repository_commit=_repository_commit(),
    )
    elapsed = time.perf_counter() - started
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    output = {
        "sample_count": len(dataset.features),
        "rounds": config.rounds,
        "seed": config.seed,
        "split_role": config.split_role,
        "runtime_seconds": elapsed,
        "peak_traced_memory_bytes": peak_memory,
        "feature_shape": list(dataset.metadata.feature_shape),
        "class_balance": dataset.metadata.to_dict()["class_balance"],
        "metadata": dataset.metadata.to_dict(),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
