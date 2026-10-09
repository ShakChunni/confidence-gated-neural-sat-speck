#!/usr/bin/env python3
"""Two-pass sparse empirical SPECK difference table for deeper rounds (D-061 attempt).

The one-pass generator (generate_sparse_classical_distribution.py) holds every
sampled difference in memory, which is fine for 5 rounds at 10^7 samples but not
for 6 rounds, where the distribution is far flatter. This generator uses the same
Markov model (independent uniform round keys, input difference (0x0040, 0)) and
the same clipped positive-LLR weights, in two independent passes:

1. Selection: sample N1 pairs in chunks; a difference is a candidate if it occurs
   at least ``select_min`` times inside any one chunk.
2. Counting: sample N2 fresh pairs (independent stream) and count only candidates.
   Keep candidates with count >= ``min_count``; weight = log2(count/N2) + 32,
   kept only if positive.

Selection and counting use disjoint random streams, so the stored counts are
unbiased for the selected differences. The result is a sparse surrogate, not an
exact Markov table: differences rarer than about select_min / chunk are mostly
absent (weight 0). Refuses to overwrite existing files.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np

from neural_sat_speck.attacks.classical_ranking import ClassicalDistributionTable
from neural_sat_speck.data.seed_policy import SeedRole, validate_seed_for_role

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RANKING_IMPL_PATH = (
    REPOSITORY_ROOT / "src" / "neural_sat_speck" / "attacks" / "classical_ranking.py"
)
INPUT_DIFFERENCE = (0x0040, 0x0000)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def sample_differences(rng: np.random.Generator, rounds: int, n: int) -> np.ndarray:
    """Packed (dx << 16) | dy output differences, as in compute_empirical_speck_distribution."""
    x0 = rng.integers(0, 65536, size=n, dtype=np.uint16)
    y0 = rng.integers(0, 65536, size=n, dtype=np.uint16)
    x1 = x0 ^ np.uint16(INPUT_DIFFERENCE[0])
    y1 = y0 ^ np.uint16(INPUT_DIFFERENCE[1])
    for _ in range(rounds):
        kr = rng.integers(0, 65536, size=n, dtype=np.uint16)
        x0 = (((x0 >> 7) | (x0 << 9)) + y0) ^ kr
        y0 = ((y0 << 2) | (y0 >> 14)) ^ x0
        x1 = (((x1 >> 7) | (x1 << 9)) + y1) ^ kr
        y1 = ((y1 << 2) | (y1 >> 14)) ^ x1
    dx = (x0 ^ x1).astype(np.uint64)
    dy = (y0 ^ y1).astype(np.uint64)
    return (dx << np.uint64(16)) | dy


def generate(
    rounds: int,
    select_chunks: int,
    count_chunks: int,
    chunk_log2: int,
    select_min: int,
    min_count: int,
    seed: int,
) -> tuple[dict[int, float], dict]:
    chunk = 1 << chunk_log2
    streams = np.random.SeedSequence(seed).spawn(select_chunks + count_chunks)
    candidates = np.empty(0, dtype=np.uint64)
    t0 = time.perf_counter()
    for i in range(select_chunks):
        diffs = sample_differences(np.random.default_rng(streams[i]), rounds, chunk)
        values, counts = np.unique(diffs, return_counts=True)
        candidates = np.union1d(candidates, values[counts >= select_min])
    select_seconds = time.perf_counter() - t0
    totals = np.zeros(candidates.size, dtype=np.int64)
    t1 = time.perf_counter()
    for j in range(count_chunks):
        rng = np.random.default_rng(streams[select_chunks + j])
        diffs = sample_differences(rng, rounds, chunk)
        pos = np.searchsorted(candidates, diffs)
        pos[pos == candidates.size] = 0
        hit = candidates[pos] == diffs
        totals += np.bincount(pos[hit], minlength=candidates.size)
    count_seconds = time.perf_counter() - t1
    n2 = count_chunks * chunk
    keep = totals >= min_count
    llr = np.log2(totals[keep] / n2) + 32.0
    positive = llr > 0.0
    diffs_kept = candidates[keep][positive]
    weights = {int(d): float(w) for d, w in zip(diffs_kept, llr[positive], strict=True)}
    meta = {
        "method": "two_pass_sparse_empirical_llr",
        "rounds": rounds,
        "chunk_samples": chunk,
        "select_chunks": select_chunks,
        "select_samples": select_chunks * chunk,
        "select_min_count_within_chunk": select_min,
        "count_samples": n2,
        "min_count": min_count,
        "candidates_selected": int(candidates.size),
        "weights_kept": len(weights),
        "select_seconds": select_seconds,
        "count_seconds": count_seconds,
    }
    return weights, meta


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--chunk-log2", type=int, default=26)
    p.add_argument("--select-chunks", type=int, default=4)
    p.add_argument("--count-chunks", type=int, default=16)
    p.add_argument("--select-min", type=int, default=2)
    p.add_argument("--min-count", type=int, default=5)
    p.add_argument("--seed", type=int, default=999999)
    p.add_argument("--seed-role", default=SeedRole.DEMO.value, choices=[r.value for r in SeedRole])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--skip-rerun", action="store_true", help="Skip the determinism rerun")
    args = p.parse_args(argv)
    validate_seed_for_role(args.seed, args.seed_role)
    out = args.output.resolve()
    if out.exists():
        print(f"[!] Refusing to overwrite existing file: {out}")
        return 1
    params = dict(
        rounds=args.rounds,
        select_chunks=args.select_chunks,
        count_chunks=args.count_chunks,
        chunk_log2=args.chunk_log2,
        select_min=args.select_min,
        min_count=args.min_count,
        seed=args.seed,
    )
    weights, meta = generate(**params)
    print(f"[*] {meta}")
    if not args.skip_rerun:
        rerun, _ = generate(**params)
        if rerun != weights:
            raise RuntimeError("Determinism check failed: rerun produced different weights")
        print("[✓] Deterministic rerun matches")
    table = ClassicalDistributionTable(
        rounds=args.rounds,
        weights=weights,
        sample_count=meta["count_samples"],
        min_count=args.min_count,
        seed=args.seed,
        seed_role=args.seed_role,
        input_difference=INPUT_DIFFERENCE,
        independent_round_keys=True,
        generator_script_hash=file_sha256(Path(__file__).resolve()),
        ranking_implementation_hash=file_sha256(RANKING_IMPL_PATH),
    )
    table.save(out)
    meta_path = out.with_suffix(".generation.json")
    import json

    meta_path.write_text(json.dumps({**meta, "seed": args.seed, "seed_role": args.seed_role,
                                     "weights_digest": table.weights_digest,
                                     "table_sha256": file_sha256(out)}, indent=2))
    print(f"[✓] {len(weights):,} weights → {out}\n[✓] SHA-256 {file_sha256(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
