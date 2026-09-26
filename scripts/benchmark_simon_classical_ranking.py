"""Truth-blind complete SIMON classical 16-to-512-to-32 ranking benchmark.

Uses public complete 7R/6R Markov tables and synthetic ciphertext words only.
It does not generate keys, read attack data, or inspect true candidate ranks.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import resource
import subprocess
import time
from pathlib import Path

import numpy as np

from neural_sat_speck.simon.classical_ranking import (
    FullPredictedDdt,
    rank_classical_domain,
)


def _synthetic_pairs() -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    rng = np.random.default_rng(260926)
    words = rng.integers(0, 1 << 16, size=(32, 2, 2), dtype=np.uint16)
    return tuple(
        (
            (int(words[i, 0, 0]), int(words[i, 0, 1])),
            (int(words[i, 1, 0]), int(words[i, 1, 1])),
        )
        for i in range(32)
    )


def _summarize_domain(domain: object) -> dict[str, float | int]:
    return {
        "total_seconds": domain.total_seconds,
        "row_load_seconds": domain.row_load_seconds,
        "screen_seconds": domain.screen_seconds,
        "rescore_seconds": domain.rescore_seconds,
        "sort_seconds": domain.sort_seconds,
        "unique_left_rows": domain.unique_left_rows,
        "screen_tie_count": domain.screen_tie_count,
        "full_tie_count": domain.full_tie_count,
    }


def _one_run(sparse6: Path, dense7: Path) -> dict[str, object]:
    started = time.perf_counter()
    pairs = _synthetic_pairs()
    t0 = time.perf_counter()
    stage1_table = FullPredictedDdt(dense7, 7)
    stage2_table = FullPredictedDdt(sparse6, 6)
    table_open_seconds = time.perf_counter() - t0
    stage1 = rank_classical_domain(pairs, stage1_table)
    parents = [int(key) for key in stage1.order[:8]]
    stage2 = [
        rank_classical_domain(pairs, stage2_table, final_key_candidate=parent)
        for parent in parents
    ]
    total_seconds = time.perf_counter() - started
    stage1_summary = _summarize_domain(stage1)
    stage2_summaries = [_summarize_domain(domain) for domain in stage2]
    # Method-domain output words and scores are deliberately omitted.
    del stage1_table, stage2_table, stage1, stage2
    gc.collect()
    return {
        "total_ranking_seconds": total_seconds,
        "table_open_seconds": table_open_seconds,
        "stage1": stage1_summary,
        "stage2": stage2_summaries,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sparse6", type=Path)
    parser.add_argument("dense7", type=Path)
    parser.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 3:
        parser.error("repetitions must be 1..3")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    runs = [_one_run(args.sparse6, args.dense7) for _ in range(args.repetitions)]
    print(
        json.dumps(
            {
                "classification": "ENGINEERING_TRUTH_BLIND_CLASSICAL_THROUGHPUT",
                "source_commit": commit,
                "benchmark_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
                "sparse6_bytes": args.sparse6.stat().st_size,
                "dense7_bytes": args.dense7.stat().st_size,
                "filesystem_cache": "uncontrolled; new-process is not cold-disk",
                "pair_count": 32,
                "screen_pairs": 16,
                "rescore_top": 512,
                "stage1_width": 8,
                "stage2_width": 128,
                "stage2_domains": 8,
                "batch_size": 1024,
                "runs": runs,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
