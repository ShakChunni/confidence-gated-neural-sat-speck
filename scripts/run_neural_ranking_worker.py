"""Standalone neural ranking worker for SIMON32/64 R=10 comparisons.

Executes in the pinned TensorFlow virtual environment (/private/tmp/simon_tf_venv)
under Python 3.12 with TensorFlow 2.16.2 and Keras 3.15.1.

Supports:
- fully_neural_1parent: Stage 1 VV9 (all 65,536 screened to 512, rescored on all attack pairs)
  -> Top 1 parent k9 -> Stage 2 VV8 (conditioned on k9) -> Top 128 children k8
  -> 128 candidate tuples (k8, k9) in chronological order.
- neural_classical_hybrid_8parent: Stage 1 VV9
  -> Top 8 parents k9 -> Stage 2 Classical 7R DDT (conditioned on each k9)
  -> Top 128 children k8 per parent -> 1,024 candidate tuples (k8, k9) in chronological order.

Invariants:
1. No secret keys or true labels accepted or accessed (truth-blind).
2. Candidate tuples returned in chronological order (k8, k9) matching SAT assumptions.
3. Accurate timing and peak RSS measurement reported.
"""

# ruff: noqa: E402, I001 - environment and TensorFlow threading are set before imports

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

# Ensure repo root src is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

# Configure TensorFlow threading for optimal CPU inference before TF init
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf  # noqa: E402 - thread settings must precede TensorFlow import

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(1)

import numpy as np  # noqa: E402

from neural_sat_speck.simon.classical_ranking import (  # noqa: E402
    FullPredictedDdt,
    rank_classical_domain,
)
from neural_sat_speck.simon.published_vv import load_published_vv_checkpoint  # noqa: E402
from neural_sat_speck.simon.published_vv_ranking import rank_neural_domain  # noqa: E402

_DEFAULT_RUNTIME = repo_root / "runtime"
_DEFAULT_CHECKPOINT_DIR = (
    str(_DEFAULT_RUNTIME / "bao-simon-audit" / "simonNDvsDD" / "SENet")
    if (_DEFAULT_RUNTIME / "bao-simon-audit" / "simonNDvsDD" / "SENet").is_dir()
    else "/private/tmp/bao-simon-audit/simonNDvsDD/SENet"
)
_DEFAULT_DDT_PATH = (
    str(_DEFAULT_RUNTIME / "resources" / "simon_predicted_ddt_7r.bin")
    if (_DEFAULT_RUNTIME / "resources" / "simon_predicted_ddt_7r.bin").is_file()
    else "/private/tmp/simon_predicted_ddt_7r.bin"
)


def get_peak_rss_gib() -> float:
    """Measure peak RSS in GiB on macOS/Linux."""
    scale = (1024.0 ** 3) if platform.system() == "Darwin" else (1024.0 ** 2)
    self_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    child_rss = float(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    return max(0.001, max(self_rss, child_rss) / scale)


def run_ranking(
    method: str,
    pairs: tuple[tuple[tuple[int, int], tuple[int, int]], ...],
    *,
    checkpoint_dir: Path | str = _DEFAULT_CHECKPOINT_DIR,
    vv9_path: Path | str | None = None,
    vv8_path: Path | str | None = None,
    ddt_7r_path: Path | str = _DEFAULT_DDT_PATH,
    time_limit: float = 120.0,
) -> dict[str, Any]:
    if method not in ("fully_neural_1parent", "neural_classical_hybrid_8parent"):
        raise ValueError(f"Unsupported neural ranking method: {method}")

    t_start = time.perf_counter()
    chk_dir = Path(checkpoint_dir)
    vv9_file = Path(vv9_path) if vv9_path else chk_dir / "ND_VV_Simon32_9R.h5"
    vv8_file = Path(vv8_path) if vv8_path else chk_dir / "ND_VV_Simon32_8R.h5"

    # Stage 1: Load VV9 and rank all 65,536 candidates
    t_s1_start = time.perf_counter()
    vv9_model, _ = load_published_vv_checkpoint(vv9_file, 9)
    stage1 = rank_neural_domain(pairs, vv9_model)
    s1_time = time.perf_counter() - t_s1_start

    elapsed_after_s1 = time.perf_counter() - t_start
    if elapsed_after_s1 > time_limit:
        return {
            "status": "ranking_timeout",
            "method": method,
            "queue": [],
            "queue_length": 0,
            "ranking_wall_seconds": elapsed_after_s1,
            "stage1_seconds": s1_time,
            "stage2_seconds": 0.0,
            "peak_rss_gib": get_peak_rss_gib(),
            "error_message": f"Stage 1 exceeded time limit ({elapsed_after_s1:.2f}s > {time_limit:.2f}s)",
        }

    t_s2_start = time.perf_counter()
    queue: list[tuple[int, int]] = []

    if method == "fully_neural_1parent":
        parent_k9 = int(stage1.order[0])
        vv8_model, _ = load_published_vv_checkpoint(vv8_file, 8)
        stage2 = rank_neural_domain(pairs, vv8_model, final_key_candidate=parent_k9)
        s2_time = time.perf_counter() - t_s2_start

        children_k8 = [int(k) for k in stage2.order[:128]]
        # Chronological order: (k8, k9)
        queue = [(int(child_k8), parent_k9) for child_k8 in children_k8]

    elif method == "neural_classical_hybrid_8parent":
        parents = [int(k) for k in stage1.order[:8]]
        table_7r = FullPredictedDdt(ddt_7r_path, 7)
        stage2_results = [
            rank_classical_domain(
                pairs,
                table_7r,
                final_key_candidate=parent,
                expected_rounds=7,
            )
            for parent in parents
        ]
        s2_time = time.perf_counter() - t_s2_start

        raw_tuples = []
        for r1, parent_k9 in enumerate(parents):
            children_k8 = stage2_results[r1].order[:128]
            for r2, child_k8 in enumerate(children_k8):
                raw_tuples.append((r1 + r2, r1, r2, int(parent_k9), int(child_k8)))

        raw_tuples.sort()
        seen = set()
        for _, _, _, parent_k9, child_k8 in raw_tuples:
            cand_tuple = (int(child_k8), int(parent_k9))  # Chronological order: (k8, k9)
            if cand_tuple not in seen:
                seen.add(cand_tuple)
                queue.append(cand_tuple)

    total_time = time.perf_counter() - t_start
    peak_rss = get_peak_rss_gib()

    return {
        "status": "ranking_timeout" if total_time > time_limit else "ok",
        "method": method,
        "queue": [list(t) for t in queue],
        "queue_length": len(queue),
        "ranking_wall_seconds": total_time,
        "stage1_seconds": s1_time,
        "stage2_seconds": s2_time,
        "peak_rss_gib": peak_rss,
        "full_pairs_used": len(pairs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Neural ranking worker for SIMON32/64")
    parser.add_argument("--method", choices=("fully_neural_1parent", "neural_classical_hybrid_8parent"), default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=_DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--vv9-path", type=str, default=None)
    parser.add_argument("--vv8-path", type=str, default=None)
    parser.add_argument("--ddt-7r-path", type=str, default=_DEFAULT_DDT_PATH)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--pair-count", type=int, choices=(32, 64), default=32, help="Pair count for synthetic pairs (32 or 64)")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic ciphertext pairs for testing")
    parser.add_argument("--stdin", action="store_true", help="Read input JSON payload from stdin")

    args = parser.parse_args()

    # Determine input mode
    if args.stdin or (not sys.stdin.isatty() and args.method is None and not args.synthetic):
        raw = sys.stdin.read()
        payload = json.loads(raw)
        method = payload["method"]
        attack_pairs_raw = payload["attack_pairs"]
        pairs = tuple(
            (
                (int(p[0][0]), int(p[0][1])),
                (int(p[1][0]), int(p[1][1])),
            )
            for p in attack_pairs_raw
        )
        checkpoint_dir = payload.get("checkpoint_dir", args.checkpoint_dir)
        vv9_path = payload.get("vv9_path", args.vv9_path)
        vv8_path = payload.get("vv8_path", args.vv8_path)
        ddt_7r_path = payload.get("ddt_7r_path", args.ddt_7r_path)
        time_limit = float(payload.get("time_limit", args.time_limit))
    elif args.synthetic:
        method = args.method or "neural_classical_hybrid_8parent"
        rng = np.random.default_rng(260926)
        pair_cnt = args.pair_count
        words = rng.integers(0, 1 << 16, size=(pair_cnt, 2, 2), dtype=np.uint16)
        pairs = tuple(
            (
                (int(words[i, 0, 0]), int(words[i, 0, 1])),
                (int(words[i, 1, 0]), int(words[i, 1, 1])),
            )
            for i in range(pair_cnt)
        )
        checkpoint_dir = args.checkpoint_dir
        vv9_path = args.vv9_path
        vv8_path = args.vv8_path
        ddt_7r_path = args.ddt_7r_path
        time_limit = args.time_limit
    else:
        sys.stderr.write("Error: either --stdin, piped input, or --synthetic required.\n")
        sys.exit(1)

    try:
        result = run_ranking(
            method=method,
            pairs=pairs,
            checkpoint_dir=checkpoint_dir,
            vv9_path=vv9_path,
            vv8_path=vv8_path,
            ddt_7r_path=ddt_7r_path,
            time_limit=time_limit,
        )
        sys.stdout.write(json.dumps(result))
        sys.stdout.flush()
        sys.exit(0 if result["status"] == "ok" else (2 if result["status"] == "ranking_timeout" else 1))
    except Exception as exc:
        err_res = {
            "status": "error",
            "method": method,
            "queue": [],
            "queue_length": 0,
            "ranking_wall_seconds": 0.0,
            "stage1_seconds": 0.0,
            "stage2_seconds": 0.0,
            "peak_rss_gib": get_peak_rss_gib(),
            "error_message": f"{type(exc).__name__}: {exc}",
        }
        sys.stdout.write(json.dumps(err_res))
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
