#!/usr/bin/env python3
"""CLI entry point for running the authorized cross-cipher residual-depth empirical map.

Executes under decision D-043 with strict parent monotonic deadline, process group isolation,
Darwin QoS certification, and crash-resilient journaling with SHA-256 sidecars.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_sat_speck.experiments.depth_map_runner import (
    DEFAULT_DATASET,
    DEFAULT_MANIFEST,
    DEFAULT_RESULTS,
    DEFAULT_TMPDIR,
    run_depth_map,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cross-cipher residual-depth empirical map (D-043).")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Manifest path")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Dataset path")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="Results output path")
    parser.add_argument("--tmpdir", type=Path, default=DEFAULT_TMPDIR, help="Persistent worker TMPDIR")
    args = parser.parse_args()

    run_depth_map(
        manifest_path=args.manifest,
        dataset_path=args.dataset,
        results_path=args.results,
        repo_root=ROOT,
        tmpdir=args.tmpdir,
    )


if __name__ == "__main__":
    main()
