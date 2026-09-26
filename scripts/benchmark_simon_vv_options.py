"""Truth-blind complete-ranking cost for two unapproved SIMON method options.

The hybrid is the proposed neural Stage 1 plus classical Stage 2. The
one-parent fully neural option changes the proposed width of eight and is
measured only for an author protocol decision. No attack truth is loaded.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from neural_sat_speck.simon.classical_ranking import (
    FullPredictedDdt,
    rank_classical_domain,
)
from neural_sat_speck.simon.published_vv import load_published_vv_checkpoint
from neural_sat_speck.simon.published_vv_ranking import rank_neural_domain


def _synthetic_pairs() -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    words = np.random.default_rng(260926).integers(0, 1 << 16, size=(32, 2, 2), dtype=np.uint16)
    return tuple(
        (
            (int(words[i, 0, 0]), int(words[i, 0, 1])),
            (int(words[i, 1, 0]), int(words[i, 1, 1])),
        )
        for i in range(32)
    )


def _summary(rank: object) -> dict[str, float | int]:
    fields = (
        "total_seconds", "feature_seconds", "inference_seconds",
        "aggregation_seconds", "row_load_seconds", "screen_seconds",
        "rescore_seconds", "sort_seconds", "screen_tie_count", "full_tie_count",
    )
    return {field: getattr(rank, field) for field in fields if hasattr(rank, field)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("proposed_hybrid", "one_parent_neural"))
    parser.add_argument("checkpoint_directory", type=Path)
    parser.add_argument("sparse6", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    pairs = _synthetic_pairs()
    vv8, audit8 = load_published_vv_checkpoint(
        args.checkpoint_directory / "ND_VV_Simon32_8R.h5", 8
    )
    if args.method == "proposed_hybrid":
        table = FullPredictedDdt(args.sparse6, 6)
        vv7 = audit7 = None
    else:
        vv7, audit7 = load_published_vv_checkpoint(
            args.checkpoint_directory / "ND_VV_Simon32_7R.h5", 7
        )
        table = None
    load_seconds = time.perf_counter() - started
    stage1 = rank_neural_domain(pairs, vv8)
    parents = [int(word) for word in stage1.order[: (8 if table is not None else 1)]]
    if table is not None:
        stage2 = [rank_classical_domain(pairs, table, final_key_candidate=parent) for parent in parents]
    else:
        assert vv7 is not None
        stage2 = [rank_neural_domain(pairs, vv7, final_key_candidate=parents[0])]
    print(
        json.dumps(
            {
                "classification": "ENGINEERING_TRUTH_BLIND_COMPLETE_RANKING_OPTION",
                "method": args.method,
                "status": "under_120_seconds" if time.perf_counter() - started <= 120 else "over_120_seconds",
                "total_ranking_seconds": time.perf_counter() - started,
                "load_seconds": load_seconds,
                "stage1": _summary(stage1),
                "stage2": [_summary(domain) for domain in stage2],
                "stage1_parent_width": len(parents),
                "stage2_width_per_parent": 128,
                "screen_pairs": 16,
                "full_pairs": 32,
                "rescore_top": 512,
                "feature_batch_size": 1024,
                "inference_batch_size": 16384,
                "candidate_scores_or_words_reported": False,
                "checkpoint_audit": {"8": audit8, "7": audit7},
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "tensorflow": tf.__version__,
                "tf_intraop_threads": os.environ.get("TF_NUM_INTRAOP_THREADS"),
                "tf_interop_threads": os.environ.get("TF_NUM_INTEROP_THREADS"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
