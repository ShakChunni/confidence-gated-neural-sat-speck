#!/usr/bin/env python3
"""Deterministic retrospective evaluation of sparse empirical classical ranker.

Evaluates the sparse empirical differential log-likelihood ranker on opened
historical Seed 98001 (Phase 7 held-out test shard, N=64 instances, 64 pairs).
Computes tie-aware rank statistics (best rank, worst rank, midrank, sorted rank),
raw unrounded timings, and cryptographic SHA-256 hashes of all inputs, code dependencies,
and git state. Refuses to overwrite existing artifacts unless --allow-overwrite is passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from neural_sat_speck.attacks.classical_ranking import (
    ClassicalDistributionTable,
    compute_rank_interval,
    rank_classical_subkeys,
)
from neural_sat_speck.data.attack import load_attack_shard
from neural_sat_speck.data.seed_policy import (
    HISTORICAL_PHASE7_SEED,
    PHASE10_G2_PROSPECTIVE_SEED,
    PHASE10_PROSPECTIVE_SEED,
    QUARANTINED_PHASE10_SEED,
    RETIRED_BLIND_SEED,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = REPOSITORY_ROOT / "datasets" / "phase7_held_out_r6" / "test.json"
DEFAULT_MODEL_PATH = REPOSITORY_ROOT / "models" / "classical_dist_r5_sparse_v2.json"
DEFAULT_OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "results"
    / "classical_comparator"
    / "classical_6round_retrospective_evaluation_v3.json"
)

DEPENDENCY_PATHS = [
    REPOSITORY_ROOT / "src" / "neural_sat_speck" / "attacks" / "classical_ranking.py",
    REPOSITORY_ROOT / "src" / "neural_sat_speck" / "attacks" / "ranking.py",
    REPOSITORY_ROOT / "src" / "neural_sat_speck" / "data" / "attack.py",
    REPOSITORY_ROOT / "src" / "neural_sat_speck" / "data" / "seed_policy.py",
    REPOSITORY_ROOT / "src" / "neural_sat_speck" / "speck" / "speck32_64.py",
]


def file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_state() -> dict[str, Any]:
    """Capture current repository commit, dirty status, and working-tree diff digest."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=REPOSITORY_ROOT
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, cwd=REPOSITORY_ROOT
        ).strip()
        is_dirty = len(status) > 0
        diff_sha256 = (
            hashlib.sha256(
                subprocess.check_output(
                    ["git", "diff", "HEAD"], text=True, cwd=REPOSITORY_ROOT
                ).encode("utf-8")
            ).hexdigest()
            if is_dirty
            else None
        )
        return {
            "commit": commit,
            "is_dirty": is_dirty,
            "diff_sha256": diff_sha256,
        }
    except Exception as e:
        return {"commit": None, "is_dirty": None, "error": str(e)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="Path to evaluation dataset shard")
    parser.add_argument("--distribution", type=Path, default=DEFAULT_MODEL_PATH, help="Path to distribution table JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Destination results JSON path")
    parser.add_argument("--allow-diagnostic-dataset", action="store_true", help="Allow running on non-Seed-98001 diagnostic dataset")
    parser.add_argument("--allow-overwrite", action="store_true", help="Allow overwriting existing artifact")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = args.output.resolve()

    if output_path.exists() and not args.allow_overwrite:
        print(f"[!] Refusing to overwrite existing file: {output_path}")
        print("    Pass --allow-overwrite if intentional.")
        return 1

    dataset_path = args.dataset.resolve()
    dist_path = args.distribution.resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset shard not found at {dataset_path}")
    if not dist_path.exists():
        raise FileNotFoundError(f"Distribution table not found at {dist_path}")

    shard = load_attack_shard(dataset_path, expected_dataset_id=None)
    meta = shard.metadata
    seed = int(meta.get("seed", 0))
    rounds = int(meta.get("rounds", 0))
    pair_count = int(meta.get("pair_count", 0))
    split_role = str(meta.get("split_role", ""))

    # Seed and data governance enforcement
    if seed in (QUARANTINED_PHASE10_SEED, RETIRED_BLIND_SEED):
        raise ValueError(f"Dataset uses quarantined ({QUARANTINED_PHASE10_SEED}) or retired ({RETIRED_BLIND_SEED}) seed: {seed}")
    if seed in (PHASE10_PROSPECTIVE_SEED, PHASE10_G2_PROSPECTIVE_SEED):
        raise ValueError(f"Dataset uses retired prospective validation seed: {seed}. Classical comparison is retrospective only.")

    if not args.allow_diagnostic_dataset:
        if seed != HISTORICAL_PHASE7_SEED:
            raise ValueError(
                f"P1 evaluator requires opened historical Seed {HISTORICAL_PHASE7_SEED}, got {seed}. "
                "Pass --allow-diagnostic-dataset to deliberately evaluate other opened diagnostic data."
            )
        if rounds != 6:
            raise ValueError(f"P1 retrospective evaluation requires rounds=6, got {rounds}")
        if pair_count != 64:
            raise ValueError(f"P1 retrospective evaluation requires pair_count=64, got {pair_count}")
        if split_role not in ("test", "held_out"):
            raise ValueError(f"P1 retrospective evaluation requires test split, got {split_role}")
        evidence_class = "RETROSPECTIVE"
    else:
        evidence_class = "RETROSPECTIVE" if seed == HISTORICAL_PHASE7_SEED else "DEVELOPMENT / DIAGNOSTIC"

    dataset_sha256 = file_sha256(dataset_path)
    dist_sha256 = file_sha256(dist_path)
    script_sha256 = file_sha256(Path(__file__).resolve())
    dep_hashes = {str(p.relative_to(REPOSITORY_ROOT)): file_sha256(p) for p in DEPENDENCY_PATHS if p.exists()}
    git_state = get_git_state()

    print("=== SPARSE EMPIRICAL CLASSICAL RANKER EVALUATION (PACKAGE P1) ===")
    print(f"Dataset: {dataset_path} (Seed: {seed}, {evidence_class})")
    print(f"Distribution: {dist_path}")
    print(f"Output: {output_path}\n")

    dist = ClassicalDistributionTable.load(dist_path)

    records: list[dict[str, Any]] = []
    sorted_ranks: list[int] = []
    best_ranks: list[int] = []
    worst_ranks: list[int] = []
    midranks: list[float] = []
    timings: list[float] = []

    t_eval_start = time.perf_counter()

    for idx, inst in enumerate(shard.instances):
        true_k = inst.final_round_subkey
        ranking = rank_classical_subkeys(inst, dist)
        interval = compute_rank_interval(ranking, true_k)

        sorted_ranks.append(interval.sorted_rank)
        best_ranks.append(interval.best_rank)
        worst_ranks.append(interval.worst_rank)
        midranks.append(interval.midrank)
        timings.append(ranking.total_seconds)

        records.append({
            "instance_index": idx,
            "instance_id": inst.instance_id,
            "true_final_round_subkey": true_k,
            "classical_rank": interval.sorted_rank,
            "best_rank": interval.best_rank,
            "worst_rank": interval.worst_rank,
            "midrank": interval.midrank,
            "tied_candidates_count": interval.tied_count,
            "classical_score": interval.score,
            "max_score": float(max(ranking.candidate_scores)),
            "top_16_containment": interval.sorted_rank <= 16,
            "timing_seconds": ranking.total_seconds,
            "partial_decryption_seconds": ranking.partial_decryption_seconds,
            "inference_seconds": ranking.inference_seconds,
            "sorting_seconds": ranking.sorting_seconds,
        })

    total_eval_time = time.perf_counter() - t_eval_start
    n = len(records)

    containment = {
        f"top{k}": sum(1 for r in sorted_ranks if r <= k)
        for k in (1, 2, 4, 8, 16, 32, 64)
    }
    containment_rates = {
        f"top{k}_rate": containment[f"top{k}"] / n
        for k in (1, 2, 4, 8, 16, 32, 64)
    }

    deterministic_evaluation = {
        "sample_size": n,
        "rounds": rounds,
        "penultimate_rounds": rounds - 1,
        "pair_count": pair_count,
        "seed": seed,
        "evidence_class": evidence_class,
        "containment": {**containment, **containment_rates},
        "rank_statistics": {
            "sorted_rank": {
                "min": min(sorted_ranks),
                "max": max(sorted_ranks),
                "mean": statistics.mean(sorted_ranks),
                "median": statistics.median(sorted_ranks),
            },
            "best_rank": {
                "min": min(best_ranks),
                "max": max(best_ranks),
                "mean": statistics.mean(best_ranks),
                "median": statistics.median(best_ranks),
            },
            "worst_rank": {
                "min": min(worst_ranks),
                "max": max(worst_ranks),
                "mean": statistics.mean(worst_ranks),
                "median": statistics.median(worst_ranks),
            },
            "midrank": {
                "min": min(midranks),
                "max": max(midranks),
                "mean": statistics.mean(midranks),
                "median": statistics.median(midranks),
            },
        },
        "deterministic_records": [
            {
                "instance_index": r["instance_index"],
                "instance_id": r["instance_id"],
                "true_final_round_subkey": r["true_final_round_subkey"],
                "classical_rank": r["classical_rank"],
                "best_rank": r["best_rank"],
                "worst_rank": r["worst_rank"],
                "midrank": r["midrank"],
                "tied_candidates_count": r["tied_candidates_count"],
                "classical_score": r["classical_score"],
                "max_score": r["max_score"],
                "top_16_containment": r["top_16_containment"],
            }
            for r in records
        ],
    }

    environment_performance = {
        "note": "Timings naturally vary across execution environments and runs; not byte-deterministic.",
        "timing_statistics": {
            "mean_seconds": statistics.mean(timings),
            "median_seconds": statistics.median(timings),
            "min_seconds": min(timings),
            "max_seconds": max(timings),
            "total_seconds": sum(timings),
            "total_wall_seconds": total_eval_time,
        },
        "per_instance_timings": [
            {
                "instance_index": r["instance_index"],
                "instance_id": r["instance_id"],
                "timing_seconds": r["timing_seconds"],
                "partial_decryption_seconds": r["partial_decryption_seconds"],
                "inference_seconds": r["inference_seconds"],
                "sorting_seconds": r["sorting_seconds"],
            }
            for r in records
        ],
    }

    result_doc = {
        "schema": "neural_sat_speck.classical_comparator_evaluation.v3",
        "date": "2026-09-12",
        "method": "sparse_empirical_monte_carlo_llr_surrogate",
        "cipher": "SPECK32/64",
        "provenance": {
            "dataset_path": str(dataset_path),
            "dataset_sha256": dataset_sha256,
            "distribution_path": str(dist_path),
            "distribution_sha256": dist_sha256,
            "distribution_weights_digest": dist.weights_digest,
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": script_sha256,
            "dependency_sha256": dep_hashes,
            "git_state": git_state,
        },
        "distribution_table_metadata": {
            "weights_count": len(dist.weights),
            "sample_count": dist.sample_count,
            "min_count": dist.min_count,
            "generation_seed": dist.seed,
            "generation_seed_role": dist.seed_role,
            "input_difference": list(dist.input_difference),
        },
        # Backwards compatibility flat summaries
        "containment": {**containment, **containment_rates},
        "rank_statistics": deterministic_evaluation["rank_statistics"],
        "timing_statistics": environment_performance["timing_statistics"],
        "deterministic_evaluation": deterministic_evaluation,
        "environment_performance": environment_performance,
        "records": records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(f".tmp.{os.getpid()}")
    try:
        temp_path.write_text(json.dumps(result_doc, indent=2), encoding="utf-8")
        temp_path.replace(output_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    out_hash = file_sha256(output_path)

    print(f"[✓] Evaluated {n} instances in {total_eval_time:.2f} s")
    print(f"[✓] Containment: Top-1={containment['top1']}/{n} ({containment_rates['top1_rate']*100:.1f}%), "
          f"Top-4={containment['top4']}/{n} ({containment_rates['top4_rate']*100:.1f}%), "
          f"Top-16={containment['top16']}/{n} ({containment_rates['top16_rate']*100:.1f}%)")
    print(f"[✓] Sorted Rank: Mean={statistics.mean(sorted_ranks):.3f}, Median={statistics.median(sorted_ranks):.1f}, Range=[{min(sorted_ranks)}, {max(sorted_ranks)}]")
    print(f"[✓] Midrank: Mean={statistics.mean(midranks):.3f}, Median={statistics.median(midranks):.1f}, Range=[{min(midranks)}, {max(midranks)}]")
    print(f"[✓] Results written to {output_path}")
    print(f"[✓] Output SHA-256: {out_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
