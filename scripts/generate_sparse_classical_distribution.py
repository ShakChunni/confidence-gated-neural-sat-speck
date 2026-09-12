#!/usr/bin/env python3
"""Deterministic generator for sparse empirical difference-distribution table.

Generates penultimate-round difference distribution weights using vectorized
Monte Carlo simulation under independent random round keys (Markov assumption).
Refuses to overwrite existing artifacts unless --allow-overwrite is passed.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

from neural_sat_speck.attacks.classical_ranking import (
    compute_empirical_speck_distribution,
)
from neural_sat_speck.data.seed_policy import SeedRole, validate_seed_for_role

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = REPOSITORY_ROOT / "models" / "classical_dist_r5_sparse_v2.json"
RANKING_IMPL_PATH = (
    REPOSITORY_ROOT / "src" / "neural_sat_speck" / "attacks" / "classical_ranking.py"
)


def file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=5, help="Number of encryption rounds (default 5)")
    parser.add_argument("--samples", type=int, default=10_000_000, help="Number of synthetic pairs (default 10,000,000)")
    parser.add_argument("--min-count", type=int, default=5, help="Minimum observed difference count threshold (default 5)")
    parser.add_argument("--seed", type=int, default=999999, help="Random seed for generation (default 999999)")
    parser.add_argument(
        "--seed-role",
        type=str,
        default=SeedRole.DEMO.value,
        choices=[r.value for r in SeedRole],
        help="Governing seed role (default: demo)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Destination JSON path")
    parser.add_argument("--allow-overwrite", action="store_true", help="Allow overwriting existing artifact")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = args.output.resolve()

    # Enforce seed governance policy
    validate_seed_for_role(args.seed, args.seed_role)

    if output_path.exists() and not args.allow_overwrite:
        print(f"[!] Refusing to overwrite existing file: {output_path}")
        print("    Pass --allow-overwrite if intentional.")
        return 1

    generator_script_path = Path(__file__).resolve()
    generator_script_hash = file_sha256(generator_script_path)
    ranking_impl_hash = file_sha256(RANKING_IMPL_PATH) if RANKING_IMPL_PATH.exists() else None

    print("=== SPARSE EMPIRICAL DIFFERENCE DISTRIBUTION GENERATOR ===")
    print(f"Rounds: {args.rounds} | Samples: {args.samples:,} | Min Count: {args.min_count} | Seed: {args.seed} ({args.seed_role})")
    print(f"Output: {output_path}\n")

    t0 = time.perf_counter()
    dist = compute_empirical_speck_distribution(
        rounds=args.rounds,
        sample_count=args.samples,
        min_count=args.min_count,
        seed=args.seed,
        seed_role=args.seed_role,
        generator_script_hash=generator_script_hash,
        ranking_implementation_hash=ranking_impl_hash,
    )
    elapsed = time.perf_counter() - t0

    # Verify deterministic equality of weights against an independent rerun
    print("[*] Running independent rerun to verify deterministic equality...")
    dist_rerun = compute_empirical_speck_distribution(
        rounds=args.rounds,
        sample_count=args.samples,
        min_count=args.min_count,
        seed=args.seed,
        seed_role=args.seed_role,
    )
    if dist.weights != dist_rerun.weights:
        raise RuntimeError("Deterministic equality check failed: independent rerun produced different weights!")
    if dist.weights_digest != dist_rerun.weights_digest:
        raise RuntimeError("Deterministic weights digest check failed!")
    print(f"[✓] Deterministic equality verified ({len(dist.weights):,} weights, digest: {dist.weights_digest[:16]}...)")

    dist.save(output_path, allow_overwrite=args.allow_overwrite)
    sha256_hash = file_sha256(output_path)

    print(f"[✓] Generated and verified {len(dist.weights):,} sparse non-zero difference weights in {elapsed:.2f} s")
    print(f"[✓] Saved to {output_path}")
    print(f"[✓] SHA-256: {sha256_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
