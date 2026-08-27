"""Summarize the predeclared Phase 6A/6B validation ranking study."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

from neural_sat_speck.neural.group_model import PAIR_GROUP_VARIANTS

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "experiments" / "phase6a_ranking.json"
METHODS = ("b0", *PAIR_GROUP_VARIANTS)
ROLES = ("selection", "confirmation")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"could not read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"JSON document {path} must contain an object")
    return value


def _repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize complete-domain B0/B1/B2/B3 rankings."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _result_path(root: Path, role: str, method: str, index: int) -> Path:
    return root / role / method / f"ranking_{index:04d}.json"


def _bootstrap_interval(
    values: list[float],
    *,
    seed: int,
    repetitions: int = 10_000,
) -> list[float]:
    if not values:
        raise ValueError("cannot bootstrap an empty list")
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(repetitions):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    lower_index = int(0.025 * repetitions)
    upper_index = int(0.975 * repetitions) - 1
    return [means[lower_index], means[upper_index]]


def _method_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = [record["evaluation"]["true_rank"] for record in records]
    top_k = {
        str(k): sum(
            bool(record["evaluation"]["top_k_membership"].get(str(k), False))
            for record in records
        )
        for k in (1, 16, 32, 64)
    }
    return {
        "instance_count": len(records),
        "ranks": ranks,
        "log2_ranks": [math.log2(rank) for rank in ranks],
        "top_k_hits": top_k,
        "top_k_recall": {key: value / len(records) for key, value in top_k.items()},
        "median_rank": sorted(ranks)[len(ranks) // 2]
        if len(ranks) % 2
        else (sorted(ranks)[len(ranks) // 2 - 1] + sorted(ranks)[len(ranks) // 2]) / 2,
        "geometric_mean_rank": math.exp(sum(math.log(rank) for rank in ranks) / len(ranks)),
        "mean_log2_rank": sum(math.log2(rank) for rank in ranks) / len(ranks),
        "timing_seconds": {
            "partial_decryption_mean": sum(
                record["ranking"]["timing_seconds"]["partial_decryption"]
                for record in records
            )
            / len(records),
            "inference_mean": sum(
                record["ranking"]["timing_seconds"]["inference"]
                for record in records
            )
            / len(records),
            "sorting_mean": sum(
                record["ranking"]["timing_seconds"]["sorting"]
                for record in records
            )
            / len(records),
            "total_mean": sum(
                record["ranking"]["timing_seconds"]["total"] for record in records
            )
            / len(records),
        },
        "peak_feature_tensor_bytes": max(
            record["ranking"]["peak_feature_tensor_bytes"] for record in records
        ),
    }


def _paired_summary(
    candidate_records: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    candidate_ranks = [record["evaluation"]["true_rank"] for record in candidate_records]
    baseline_ranks = [record["evaluation"]["true_rank"] for record in baseline_records]
    differences = [base - candidate for candidate, base in zip(candidate_ranks, baseline_ranks, strict=True)]
    wins = sum(difference > 0 for difference in differences)
    ties = sum(difference == 0 for difference in differences)
    losses = sum(difference < 0 for difference in differences)
    top16_differences = [
        int(candidate["evaluation"]["top_k_membership"]["16"])
        - int(baseline["evaluation"]["top_k_membership"]["16"])
        for candidate, baseline in zip(candidate_records, baseline_records, strict=True)
    ]
    return {
        "rank_win_tie_loss": {"win": wins, "tie": ties, "loss": losses},
        "rank_difference_baseline_minus_candidate": differences,
        "median_rank_difference": (
            sorted(differences)[len(differences) // 2]
            if len(differences) % 2
            else (
                sorted(differences)[len(differences) // 2 - 1]
                + sorted(differences)[len(differences) // 2]
            )
            / 2
        ),
        "top16_hit_difference_candidate_minus_baseline": sum(top16_differences),
        "top16_recall_difference": sum(top16_differences) / len(top16_differences),
        "top16_recall_difference_bootstrap_95_ci": _bootstrap_interval(
            [float(value) for value in top16_differences],
            seed=seed,
        ),
    }


def main() -> None:
    args = _parse_args()
    config_path = _repository_path(args.config).resolve()
    config = _load_json(config_path)
    if config.get("schema") != "neural_sat_speck.phase6a_ranking_config.v1":
        raise SystemExit("Phase 6A ranking configuration schema is not recognised")
    output_root = _repository_path(str(config["output_root"])).resolve()
    instance_count = int(config.get("validation_instance_count", 32))
    all_records: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for role in ROLES:
        all_records[role] = {}
        for method in METHODS:
            records: list[dict[str, Any]] = []
            for index in range(instance_count):
                result_path = _result_path(output_root, role, method, index)
                if not result_path.is_file():
                    raise SystemExit(f"missing ranking result: {result_path}")
                record = _load_json(result_path)
                if record.get("method") != method:
                    raise SystemExit(f"result method mismatch in {result_path}")
                if record.get("instance", {}).get("validation_role") != role:
                    raise SystemExit(f"result validation role mismatch in {result_path}")
                if record.get("candidate_domain", {}).get("complete_16_bit_domain") is not True:
                    raise SystemExit(f"result is not complete-domain ranking: {result_path}")
                records.append(record)
            all_records[role][method] = records

    summaries: dict[str, Any] = {"selection": {}, "confirmation": {}}
    for role in ROLES:
        for method in METHODS:
            summaries[role][method] = _method_summary(all_records[role][method])
        baseline = all_records[role]["b0"]
        for method in PAIR_GROUP_VARIANTS:
            summaries[role][method]["paired_vs_b0"] = _paired_summary(
                all_records[role][method],
                baseline,
                seed=63001 + ROLES.index(role) * 100 + PAIR_GROUP_VARIANTS.index(method),
            )

    selection_scores = {
        method: (
            summaries["selection"][method]["top_k_hits"]["16"],
            -summaries["selection"][method]["median_rank"],
            -summaries["selection"][method]["geometric_mean_rank"],
            method,
        )
        for method in PAIR_GROUP_VARIANTS
    }
    selected_method = max(selection_scores, key=selection_scores.get)
    selection_summary = summaries["selection"][selected_method]
    baseline_selection = summaries["selection"]["b0"]
    selection_pass = (
        selection_summary["top_k_hits"]["16"]
        >= baseline_selection["top_k_hits"]["16"] + 4
        and selection_summary["median_rank"] < baseline_selection["median_rank"]
    )
    confirmation_summary = summaries["confirmation"][selected_method]
    baseline_confirmation = summaries["confirmation"]["b0"]
    confirmation_pass = (
        confirmation_summary["top_k_hits"]["16"]
        >= baseline_confirmation["top_k_hits"]["16"] + 4
        and confirmation_summary["median_rank"] < baseline_confirmation["median_rank"]
    )
    report: dict[str, Any] = {
        "schema": "neural_sat_speck.phase6a_summary.v1",
        "study": "phase6a_b1_b2_b3_pair_aware_rankers",
        "config_path": str(config_path),
        "candidate_selection_rule": (
            "highest selection top16 hits, then lower median rank, then lower "
            "geometric-mean rank, then lexical method id"
        ),
        "selected_method_by_selection": selected_method,
        "selection_gate": {
            "required_top16_hit_margin": 4,
            "required_lower_median_rank": True,
            "passed": selection_pass,
        },
        "confirmation_gate_for_selected_method": {
            "passed": confirmation_pass,
            "required_top16_hit_margin": 4,
            "required_lower_median_rank": True,
        },
        "promotion_to_sat_pilot": bool(selection_pass and confirmation_pass),
        "results": summaries,
    }
    output_path = (
        _repository_path(args.output).resolve()
        if args.output is not None
        else output_root / "phase6a_summary.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Selection method: {selected_method}")
    print(f"Selection gate passed: {selection_pass}")
    print(f"Confirmation gate passed: {confirmation_pass}")
    print(f"SAT pilot promotion: {report['promotion_to_sat_pilot']}")
    for role in ROLES:
        print(f"{role}:")
        for method in METHODS:
            summary = summaries[role][method]
            print(
                f"  {method}: top16 {summary['top_k_hits']['16']}/{instance_count}, "
                f"median rank {summary['median_rank']}, "
                f"geomean {summary['geometric_mean_rank']:.3f}"
            )
    print(f"Summary: {output_path}")


if __name__ == "__main__":
    main()
