"""Truth-blind SIMON candidate-feature and synthetic-sort throughput probe.

No cipher key, plaintext, trained model, truth label, or DDT is loaded. The
synthetic score is only a timing placeholder; its order has no attack meaning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from neural_sat_speck.simon.ranking_features import (
    bao_full_value_bits,
    bao_value_difference_bits,
    candidate_difference_words,
    candidate_transformed_words,
    candidate_value_words,
)


def _synthetic_pairs(count: int) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    rng = np.random.default_rng(260926)
    words = rng.integers(0, 1 << 16, size=(count, 2, 2), dtype=np.uint16)
    return tuple(
        (
            (int(words[i, 0, 0]), int(words[i, 0, 1])),
            (int(words[i, 1, 0]), int(words[i, 1, 1])),
        )
        for i in range(count)
    )


def _benchmark_domain(
    pairs: tuple[tuple[tuple[int, int], tuple[int, int]], ...],
    batch_size: int,
    final_key_candidate: int | None,
    representation: str,
) -> dict[str, float | int]:
    candidate_count = 1 << 16
    scores = np.empty(candidate_count, dtype=np.int32)
    transform_seconds = 0.0
    encoding_seconds = 0.0
    placeholder_seconds = 0.0
    peak_batch_bytes = 0
    started = time.perf_counter()
    for start in range(0, candidate_count, batch_size):
        keys = range(start, min(start + batch_size, candidate_count))
        t0 = time.perf_counter()
        if representation == "vv":
            states = candidate_value_words(
                pairs, keys, final_key_candidate=final_key_candidate
            )
        else:
            states = candidate_transformed_words(
                pairs, keys, final_key_candidate=final_key_candidate
            )
        transform_seconds += time.perf_counter() - t0
        t0 = time.perf_counter()
        if representation == "vv":
            bits = bao_full_value_bits(states)
            auxiliary_bytes = 0
        else:
            bits = bao_value_difference_bits(states)
            differences = candidate_difference_words(states)
            auxiliary_bytes = differences.nbytes
        encoding_seconds += time.perf_counter() - t0
        peak_batch_bytes = max(
            peak_batch_bytes, states.nbytes + bits.nbytes + auxiliary_bytes
        )
        t0 = time.perf_counter()
        scores[start : start + len(keys)] = bits.sum(axis=(1, 2, 3), dtype=np.int32)
        placeholder_seconds += time.perf_counter() - t0

    t0 = time.perf_counter()
    order = np.lexsort((np.arange(candidate_count), -scores))
    sort_seconds = time.perf_counter() - t0
    assert len(order) == candidate_count
    return {
        "candidate_count": candidate_count,
        "representation": representation,
        "pair_count": len(pairs),
        "candidate_pair_count": candidate_count * len(pairs),
        "transform_seconds": transform_seconds,
        "encoding_seconds": encoding_seconds,
        "synthetic_score_seconds": placeholder_seconds,
        "sort_seconds": sort_seconds,
        "total_seconds": time.perf_counter() - started,
        "peak_live_batch_array_bytes_lower_bound": peak_batch_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--stage-two-domains", type=int, default=2)
    parser.add_argument("--representation", choices=("vv", "vd_ddt"), default="vv")
    args = parser.parse_args()
    if args.pairs <= 0 or not 1 <= args.batch_size <= 65536:
        parser.error("pairs must be positive and batch-size must be 1..65536")
    if not 0 <= args.stage_two_domains <= 65536:
        parser.error("stage-two-domains must be 0..65536")

    pairs = _synthetic_pairs(args.pairs)
    stage_one = _benchmark_domain(pairs, args.batch_size, None, args.representation)
    stage_two = [
        _benchmark_domain(pairs, args.batch_size, candidate, args.representation)
        for candidate in range(args.stage_two_domains)
    ]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    feature_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "neural_sat_speck"
        / "simon"
        / "ranking_features.py"
    )
    print(
        json.dumps(
            {
                "classification": "ENGINEERING_TRUTH_BLIND_FEATURE_THROUGHPUT",
                "warning": "Synthetic score is not neural inference or DDT lookup; no ranking-quality claim.",
                "source_commit": commit,
                "feature_source_sha256": hashlib.sha256(
                    feature_path.read_bytes()
                ).hexdigest(),
                "benchmark_script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "batch_size": args.batch_size,
                "stage_one": stage_one,
                "stage_two": stage_two,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
