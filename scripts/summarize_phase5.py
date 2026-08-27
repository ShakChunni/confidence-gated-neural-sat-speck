"""Summarize local Phase 5 integration results without tuning them."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GLOB = "results/phase5_integration_validation_*.json"
METHODS = ("sat_only", "random", "neural")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"Could not read result {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"Result {path} must contain a JSON object")
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


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"{name} must be numeric")
    return float(value)


def _ranking_rank(result: dict[str, Any]) -> int | None:
    provenance = result.get("ranking_provenance")
    if not isinstance(provenance, dict):
        return None
    raw_path = provenance.get("path")
    if not isinstance(raw_path, str):
        return None
    ranking_path = _repository_path(raw_path)
    if not ranking_path.is_file():
        return None
    ranking = _load_json(ranking_path)
    evaluation = ranking.get("evaluation")
    if not isinstance(evaluation, dict):
        return None
    value = evaluation.get("true_rank")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _instance_record(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    if result.get("schema") != "neural_sat_speck.phase5_integration_result.v1":
        raise SystemExit(f"unsupported Phase 5 result schema in {path}")
    instance = result.get("instance")
    methods = result.get("methods")
    if not isinstance(instance, dict) or not isinstance(methods, dict):
        raise SystemExit(f"Phase 5 result is incomplete: {path}")
    if set(methods) != set(METHODS):
        raise SystemExit(f"Phase 5 result methods are incomplete: {path}")
    method_records: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        record = methods[method]
        if not isinstance(record, dict):
            raise SystemExit(f"Phase 5 method record is invalid: {path}")
        timing = record.get("timing_seconds")
        if not isinstance(timing, dict):
            raise SystemExit(f"Phase 5 method timing is missing: {path}")
        method_records[method] = {
            "status": record.get("status"),
            "success": record.get("success"),
            "winning_candidate_position": record.get("winning_candidate_position"),
            "candidates_attempted": record.get("candidates_attempted"),
            "solver_calls": record.get("solver_calls"),
            "solver_seconds": _number(timing.get("solver"), f"{method}.solver"),
            "verification_seconds": _number(
                timing.get("verification"),
                f"{method}.verification",
            ),
            "online_total_seconds": _number(
                timing.get("online_total"),
                f"{method}.online_total",
            ),
        }
    return {
        "source_result": _relative_or_absolute(path),
        "source_result_sha256": _sha256_file(path),
        "split_role": instance.get("split_role"),
        "instance_index": instance.get("index"),
        "instance_id": instance.get("instance_id"),
        "true_final_round_subkey": instance.get("true_final_round_subkey"),
        "true_subkey_rank": _ranking_rank(result),
        "methods": method_records,
    }


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    method_summary: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        method_records = [record["methods"][method] for record in records]
        statuses = Counter(record["status"] for record in method_records)
        method_summary[method] = {
            "instance_count": len(method_records),
            "success_count": sum(record["success"] is True for record in method_records),
            "success_rate": (
                sum(record["success"] is True for record in method_records)
                / len(method_records)
            ),
            "status_counts": dict(sorted(statuses.items())),
            "median_candidates_attempted": statistics.median(
                record["candidates_attempted"] for record in method_records
            ),
            "median_solver_seconds": statistics.median(
                record["solver_seconds"] for record in method_records
            ),
            "median_online_total_seconds": statistics.median(
                record["online_total_seconds"] for record in method_records
            ),
        }
    return method_summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize local Phase 5 result JSON files without retuning them."
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=None,
        help="Explicit result files; default is the validation-result glob.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "phase5_validation_summary.json",
        help="Summary JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.input is None:
        paths = sorted(REPOSITORY_ROOT.glob(DEFAULT_GLOB))
    else:
        paths = sorted(_repository_path(path).resolve() for path in args.input)
    if not paths:
        raise SystemExit("No Phase 5 result files were found")
    records = [_instance_record(path, _load_json(path)) for path in paths]
    split_roles = {record["split_role"] for record in records}
    result: dict[str, Any] = {
        "schema": "neural_sat_speck.phase5_summary.v1",
        "phase": 5,
        "purpose": "descriptive aggregation of frozen Phase 5 result files",
        "instance_count": len(records),
        "split_roles": sorted(split_roles),
        "instances": records,
        "methods": _summary(records),
        "limitations": [
            "This summary does not select a timeout, budget, threshold, or method.",
            "The validation neural ranking time is precomputed input to each result.",
            "Time-limit-ambiguous solver outcomes remain inconclusive.",
        ],
    }
    output_path = _repository_path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Summarized {len(records)} result files")
    for method, summary in result["methods"].items():
        print(
            f"{method:>8}: {summary['success_count']}/{summary['instance_count']} "
            f"verified, statuses={summary['status_counts']}"
        )
    print(f"Summary saved to: {output_path}")


if __name__ == "__main__":
    main()
