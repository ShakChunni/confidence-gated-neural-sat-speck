"""Truth-blind optimistic SIMON VV inference-cost falsification probe.

Reuses one synthetic 16-pair bit matrix for all nine proposed screen domains.
It omits feature extraction, the 512-candidate full rescores, sorting, SAT,
and verification, so observed time is below a complete neural attack path
under this exact runtime/backend. It stops once the 120-second ranking gate is
exceeded; no truth labels or attack data are loaded.
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

from neural_sat_speck.simon.published_vv import load_published_vv_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_directory", type=Path)
    parser.add_argument("--ranking-cutoff", type=float, default=120.0)
    args = parser.parse_args()
    if args.ranking_cutoff <= 0:
        parser.error("ranking cutoff must be positive")
    started = time.perf_counter()
    models = {}
    audit = {}
    for rounds in (7, 8):
        path = args.checkpoint_directory / f"ND_VV_Simon32_{rounds}R.h5"
        models[rounds], audit[rounds] = load_published_vv_checkpoint(path, rounds)
    load_seconds = time.perf_counter() - started
    rng = np.random.default_rng(260926)
    bits = rng.integers(0, 2, size=(1 << 20, 64), dtype=np.uint8).astype(np.float32)
    allocation_seconds = time.perf_counter() - started - load_seconds
    domain_records = []
    for domain in range(9):
        rounds = 8 if domain == 0 else 7
        t0 = time.perf_counter()
        output = models[rounds].predict(bits, batch_size=1 << 14, verbose=0)
        elapsed = time.perf_counter() - t0
        if output.shape != ((1 << 20), 1) or not np.all(np.isfinite(output)):
            raise ValueError("published VV inference returned malformed/nonfinite output")
        record = {
            "screen_domain": domain + 1,
            "checkpoint_rounds": rounds,
            "samples": 1 << 20,
            "inference_seconds": elapsed,
            "elapsed_online_seconds": time.perf_counter() - started,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
        domain_records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if record["elapsed_online_seconds"] > args.ranking_cutoff:
            break
    print(
        json.dumps(
            {
                "classification": "ENGINEERING_TRUTH_BLIND_OPTIMISTIC_INFERENCE_ONLY",
                "status": (
                    "ranking_cutoff_exceeded"
                    if domain_records[-1]["elapsed_online_seconds"] > args.ranking_cutoff
                    else "all_nine_screen_domains_completed"
                ),
                "checkpoint_audit": audit,
                "tensorflow": tf.__version__,
                "numpy": np.__version__,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "tf_intraop_threads": os.environ.get("TF_NUM_INTRAOP_THREADS"),
                "tf_interop_threads": os.environ.get("TF_NUM_INTEROP_THREADS"),
                "batch_size": 1 << 14,
                "ranking_cutoff_seconds": args.ranking_cutoff,
                "checkpoint_load_seconds": load_seconds,
                "synthetic_input_allocation_seconds": allocation_seconds,
                "completed_screen_domains": len(domain_records),
                "required_screen_domains": 9,
                "omitted_online_components": [
                    "candidate_feature_extraction",
                    "512-candidate_full_rescores_per_domain",
                    "ranking_sort_and_queue",
                    "SAT_and_verification",
                ],
                "domain_records": domain_records,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
