"""Compare paired ranking results formed from 32 and 64 related pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREFIX_GLOB = (
    "results/phase6_pair64_prefix32_ranking_r6_validation_*.json"
)
DEFAULT_FULL_GLOB = "results/phase6_pair64_ranking_r6_validation_*.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"Could not read ranking result {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"Ranking result {path} must contain an object")
    return value


def _repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_result(path: Path, *, expected_pair_count: int) -> dict[str, Any]:
    document = _load_json(path)
    if document.get("schema") != "neural_sat_speck.phase4_ranking_result.v1":
        raise SystemExit(f"unsupported ranking result schema in {path}")
    instance = document.get("instance")
    evaluation = document.get("evaluation")
    domain = document.get("candidate_domain")
    ranking = document.get("ranking")
    dataset = document.get("dataset")
    if not all(
        isinstance(value, dict)
        for value in (instance, evaluation, domain, ranking, dataset)
    ):
        raise SystemExit(f"ranking result is incomplete: {path}")
    if instance.get("split_role") != "validation":
        raise SystemExit(f"paired comparison accepts validation results only: {path}")
    if instance.get("pair_count") != expected_pair_count:
        raise SystemExit(f"ranking result pair count does not match {path}")
    if (
        domain.get("count") != 1 << 16
        or domain.get("complete_16_bit_domain") is not True
    ):
        raise SystemExit(f"ranking result is not a complete domain: {path}")
    order = ranking.get("candidate_order")
    scores = ranking.get("candidate_scores")
    if (
        not isinstance(order, list)
        or len(order) != 1 << 16
        or len(set(order)) != 1 << 16
        or not isinstance(scores, list)
        or len(scores) != 1 << 16
    ):
        raise SystemExit(f"ranking result payload is incomplete: {path}")
    true_subkey = evaluation.get("true_final_round_subkey")
    true_rank = evaluation.get("true_rank")
    if (
        isinstance(true_subkey, bool)
        or not isinstance(true_subkey, int)
        or isinstance(true_rank, bool)
        or not isinstance(true_rank, int)
    ):
        raise SystemExit(f"ranking result evaluation is invalid: {path}")
    return {
        "path": _relative_or_absolute(path),
        "sha256": _sha256_file(path),
        "dataset_id": dataset.get("dataset_id"),
        "instance_id": instance.get("instance_id"),
        "true_final_round_subkey": true_subkey,
        "true_rank": true_rank,
        "pair_count": expected_pair_count,
        "ranking_seconds": ranking.get("timing_seconds", {}).get("total"),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare paired full-domain ranking results for 32 and 64 pairs."
    )
    parser.add_argument("--prefix-glob", default=DEFAULT_PREFIX_GLOB)
    parser.add_argument("--full-glob", default=DEFAULT_FULL_GLOB)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "phase6_pair_count_comparison.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prefix_root = _repository_path(args.prefix_glob)
    full_root = _repository_path(args.full_glob)
    prefix_paths = sorted(prefix_root.parent.glob(prefix_root.name))
    full_paths = sorted(full_root.parent.glob(full_root.name))
    if not prefix_paths or not full_paths:
        raise SystemExit("both paired ranking result globs must match files")

    prefix_records = {
        record["instance_id"]: record
        for record in (
            _validated_result(path, expected_pair_count=32)
            for path in prefix_paths
        )
    }
    full_records = {
        record["instance_id"]: record
        for record in (
            _validated_result(path, expected_pair_count=64) for path in full_paths
        )
    }
    if set(prefix_records) != set(full_records):
        raise SystemExit("32-pair and 64-pair result instances do not match")
    dataset_ids = {
        record["dataset_id"]
        for record in (*prefix_records.values(), *full_records.values())
    }
    if len(dataset_ids) != 1:
        raise SystemExit("paired results use different dataset ids")

    records: list[dict[str, Any]] = []
    for instance_id in sorted(prefix_records):
        prefix = prefix_records[instance_id]
        full = full_records[instance_id]
        if (
            prefix["true_final_round_subkey"]
            != full["true_final_round_subkey"]
            or prefix["true_rank"] < 1
            or full["true_rank"] < 1
        ):
            raise SystemExit(f"paired result truth metadata disagrees: {instance_id}")
        records.append(
            {
                "instance_id": instance_id,
                "prefix32_result": prefix["path"],
                "prefix32_sha256": prefix["sha256"],
                "full64_result": full["path"],
                "full64_sha256": full["sha256"],
                "rank_prefix32": prefix["true_rank"],
                "rank_full64": full["true_rank"],
                "rank_change_full_minus_prefix": full["true_rank"]
                - prefix["true_rank"],
                "top16_prefix32": prefix["true_rank"] <= 16,
                "top16_full64": full["true_rank"] <= 16,
                "ranking_seconds_prefix32": prefix["ranking_seconds"],
                "ranking_seconds_full64": full["ranking_seconds"],
            }
        )

    rank_changes = [record["rank_change_full_minus_prefix"] for record in records]
    output: dict[str, Any] = {
        "schema": "neural_sat_speck.phase6_pair_count_comparison.v1",
        "phase": 6,
        "purpose": (
            "paired descriptive comparison of 32-pair and 64-pair full-domain "
            "neural subkey rankings"
        ),
        "protocol": {
            "split_role": "validation",
            "same_key_and_pair_prefix": True,
            "prefix_pair_count": 32,
            "full_pair_count": 64,
            "candidate_count": 1 << 16,
            "no_test_results_read": True,
        },
        "dataset_id": next(iter(dataset_ids)),
        "instances": records,
        "summary": {
            "instance_count": len(records),
            "top16_prefix32_count": sum(record["top16_prefix32"] for record in records),
            "top16_full64_count": sum(record["top16_full64"] for record in records),
            "full64_rank_improved_count": sum(change < 0 for change in rank_changes),
            "full64_rank_unchanged_count": sum(change == 0 for change in rank_changes),
            "full64_rank_worsened_count": sum(change > 0 for change in rank_changes),
            "median_rank_prefix32": statistics.median(
                record["rank_prefix32"] for record in records
            ),
            "median_rank_full64": statistics.median(
                record["rank_full64"] for record in records
            ),
            "median_rank_change_full_minus_prefix": statistics.median(rank_changes),
        },
    }
    output_path = _repository_path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = output["summary"]
    print(
        f"Compared {summary['instance_count']} paired validation instances: "
        f"top16 {summary['top16_prefix32_count']}/{summary['instance_count']} -> "
        f"{summary['top16_full64_count']}/{summary['instance_count']}"
    )
    print(f"Median rank: {summary['median_rank_prefix32']} -> {summary['median_rank_full64']}")
    print(f"Comparison saved to: {output_path}")


if __name__ == "__main__":
    main()
