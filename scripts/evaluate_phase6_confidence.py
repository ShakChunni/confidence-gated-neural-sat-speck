"""Evaluate frozen confidence features on validation ranking results only."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from neural_sat_speck.confidence import (
    CONFIDENCE_FEATURE_NAMES,
    EXTENDED_CONFIDENCE_FEATURE_NAMES,
    ConfidenceObservation,
    all_risk_coverage_curves,
    candidate_is_in_top_k,
    extract_serialized_ranking_confidence,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT / "configs" / "experiments" / "phase6_confidence_r6.json"
)
FULL_CANDIDATE_COUNT = 1 << 16


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"Could not read JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"JSON file {path} must contain an object")
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


def _repository_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _require_int(config: dict[str, Any], field: str) -> int:
    value = config.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f"configuration field {field!r} must be an integer")
    return value


def _validated_observation(
    path: Path,
    *,
    expected_dataset_id: str,
    expected_rounds: int,
    expected_input_difference: tuple[int, int],
    expected_split_role: str,
    top_k: int,
) -> ConfidenceObservation:
    document = _load_json(path)
    if document.get("schema") != "neural_sat_speck.phase4_ranking_result.v1":
        raise SystemExit(f"unsupported ranking result schema in {path}")

    dataset = document.get("dataset")
    if not isinstance(dataset, dict):
        raise SystemExit(f"ranking result dataset metadata is missing: {path}")
    if dataset.get("dataset_id") != expected_dataset_id:
        raise SystemExit(f"ranking result dataset id does not match: {path}")
    if dataset.get("rounds") != expected_rounds:
        raise SystemExit(f"ranking result round count does not match: {path}")
    if dataset.get("input_difference") != list(expected_input_difference):
        raise SystemExit(f"ranking result input difference does not match: {path}")

    instance = document.get("instance")
    if not isinstance(instance, dict):
        raise SystemExit(f"ranking result instance metadata is missing: {path}")
    if instance.get("split_role") != expected_split_role:
        raise SystemExit(
            f"Phase 6 accepts validation results only; got {path}"
        )
    instance_index = instance.get("index")
    instance_id = instance.get("instance_id")
    if (
        isinstance(instance_index, bool)
        or not isinstance(instance_index, int)
        or not isinstance(instance_id, str)
        or not instance_id
    ):
        raise SystemExit(f"ranking result instance identity is invalid: {path}")

    domain = document.get("candidate_domain")
    if not isinstance(domain, dict):
        raise SystemExit(f"ranking result candidate domain is missing: {path}")
    if (
        domain.get("complete_16_bit_domain") is not True
        or domain.get("count") != FULL_CANDIDATE_COUNT
        or domain.get("first") != 0
        or domain.get("last") != FULL_CANDIDATE_COUNT - 1
    ):
        raise SystemExit(f"ranking result is not a complete 16-bit ranking: {path}")

    try:
        features = extract_serialized_ranking_confidence(document, top_k=top_k)
    except (TypeError, ValueError, KeyError) as error:
        raise SystemExit(f"invalid ranking payload in {path}: {error}") from error

    evaluation = document.get("evaluation")
    if not isinstance(evaluation, dict):
        raise SystemExit(f"ranking result evaluation is missing: {path}")
    true_subkey = evaluation.get("true_final_round_subkey")
    true_rank = evaluation.get("true_rank")
    if (
        isinstance(true_subkey, bool)
        or not isinstance(true_subkey, int)
        or not 0 <= true_subkey <= 0xFFFF
        or isinstance(true_rank, bool)
        or not isinstance(true_rank, int)
        or not 1 <= true_rank <= FULL_CANDIDATE_COUNT
    ):
        raise SystemExit(f"ranking result true-subkey evaluation is invalid: {path}")
    target = candidate_is_in_top_k(features, true_subkey)
    if target != (true_rank <= top_k):
        raise SystemExit(f"ranking result target disagrees with true rank: {path}")

    return ConfidenceObservation(
        source_result=_relative_or_absolute(path),
        source_result_sha256=_sha256_file(path),
        split_role=expected_split_role,
        instance_index=instance_index,
        instance_id=instance_id,
        true_final_round_subkey=true_subkey,
        true_rank=true_rank,
        target_in_top_k=target,
        features=features,
    )


def _curve_summary(curve: tuple[Any, ...]) -> dict[str, Any]:
    zero_risk_points = [point for point in curve if point.risk == 0.0]
    return {
        "point_count": len(curve),
        "full_coverage": curve[-1].coverage,
        "full_coverage_risk": curve[-1].risk,
        "max_zero_empirical_risk_accepted": (
            max(point.accepted_count for point in zero_risk_points)
            if zero_risk_points
            else 0
        ),
        "max_zero_empirical_risk_coverage": (
            max(point.coverage for point in zero_risk_points)
            if zero_risk_points
            else 0.0
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Phase 6 confidence features on validation rankings; "
            "do not fit a threshold or read the held-out test split."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Phase 6 JSON configuration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the local JSON output path from the configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = _repository_path(args.config).resolve()
    config = _load_json(config_path)
    if config.get("schema") != "neural_sat_speck.phase6_confidence_config.v1":
        raise SystemExit("confidence config schema is not recognised")
    if config.get("phase") != 6:
        raise SystemExit("the selected configuration is not a Phase 6 config")

    split_role = config.get("split_role")
    if split_role != "validation":
        raise SystemExit("Phase 6 confidence evaluation is validation-only")
    top_k = _require_int(config, "top_k")
    if not 2 <= top_k <= FULL_CANDIDATE_COUNT:
        raise SystemExit("top_k must be between 2 and 65,536")
    feature_names = config.get("feature_names")
    if feature_names not in (
        list(CONFIDENCE_FEATURE_NAMES),
        list(EXTENDED_CONFIDENCE_FEATURE_NAMES),
    ):
        raise SystemExit(
            "feature_names must exactly match a frozen Phase 6 feature list"
        )
    primary_feature = config.get("primary_confidence_feature")
    if primary_feature not in feature_names:
        raise SystemExit("primary_confidence_feature is not recognised")

    glob_value = config.get("ranking_results_glob")
    if not isinstance(glob_value, str) or not glob_value:
        raise SystemExit("ranking_results_glob must be a non-empty string")
    glob_root = _repository_path(glob_value)
    paths = sorted(glob_root.parent.glob(glob_root.name))
    if not paths:
        raise SystemExit(f"no ranking results matched {glob_value!r}")

    difference_value = config.get("expected_input_difference")
    if (
        not isinstance(difference_value, list)
        or len(difference_value) != 2
        or any(
            isinstance(word, bool) or not isinstance(word, int)
            for word in difference_value
        )
    ):
        raise SystemExit("expected_input_difference must contain two integers")

    observations = [
        _validated_observation(
            path,
            expected_dataset_id=str(config["expected_dataset_id"]),
            expected_rounds=_require_int(config, "expected_rounds"),
            expected_input_difference=tuple(difference_value),
            expected_split_role=split_role,
            top_k=top_k,
        )
        for path in paths
    ]
    instance_ids = [observation.instance_id for observation in observations]
    if len(set(instance_ids)) != len(instance_ids):
        raise SystemExit("validation rankings contain duplicate instance ids")

    curves = all_risk_coverage_curves(
        observations,
        feature_names=tuple(feature_names),
    )
    output: dict[str, Any] = {
        "schema": "neural_sat_speck.phase6_confidence_result.v1",
        "phase": 6,
        "purpose": (
            "descriptive validation-only risk-coverage analysis for frozen "
            "post-hoc ranking features"
        ),
        "protocol": {
            "split_role": split_role,
            "top_k": top_k,
            "feature_names": list(feature_names),
            "primary_confidence_feature": primary_feature,
            "threshold_selection_performed": False,
            "threshold_selection_note": (
                "Four validation instances are treated as a pilot; no "
                "threshold or fallback is selected from this result."
            ),
        },
        "dataset": {
            "dataset_id": config["expected_dataset_id"],
            "rounds": config["expected_rounds"],
            "input_difference": difference_value,
        },
        "observations": [observation.to_dict() for observation in observations],
        "target_summary": {
            "instance_count": len(observations),
            "target_positive_count": sum(
                observation.target_in_top_k for observation in observations
            ),
            "target_positive_rate": sum(
                observation.target_in_top_k for observation in observations
            )
            / len(observations),
            "true_ranks": [observation.true_rank for observation in observations],
        },
        "risk_coverage": {
            feature_name: [point.to_dict() for point in curve]
            for feature_name, curve in curves.items()
        },
        "descriptive_summary": {
            feature_name: _curve_summary(curve)
            for feature_name, curve in curves.items()
        },
        "provenance": {
            "repository_commit": _repository_commit(),
            "config_path": _relative_or_absolute(config_path),
            "config_sha256": _sha256_file(config_path),
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "command": [sys.executable, *sys.argv],
        },
    }
    output_path = _repository_path(
        args.output if args.output is not None else str(config["output"])
    ).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    positive_count = output["target_summary"]["target_positive_count"]
    total_count = output["target_summary"]["instance_count"]
    primary_curve = curves[primary_feature]
    print(f"Evaluated {total_count} validation rankings with top_k={top_k}")
    print(f"Top-k target positives: {positive_count}/{total_count}")
    print(
        f"Primary feature: {primary_feature}; "
        f"zero-risk prefix={_curve_summary(primary_curve)['max_zero_empirical_risk_accepted']}"
    )
    print("No confidence threshold or fallback was selected.")
    print(f"Result saved to: {output_path}")


if __name__ == "__main__":
    main()
