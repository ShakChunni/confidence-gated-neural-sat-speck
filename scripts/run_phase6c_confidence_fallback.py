"""Run the frozen Phase 6C-A confidence/fallback validation protocol.

The default ``--calibrate`` action is read-only: it audits the authoritative
eager Phase 6B validation records, fits the one-feature calibrator, and emits
the frozen threshold decision.  The optional dataset/ranking actions are kept
here for the later reviewed launch, but the runner always refuses to proceed
when calibration has no feasible threshold.  It never reads or creates a test
split and never invokes CryptoMiniSat.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from neural_sat_speck.attacks.diversity_fallback import (
    PHASE6C_FALLBACK_BUDGET,
    PHASE6C_FALLBACK_HEAD_SIZE,
    PHASE6C_TRUSTED_BUDGET,
    choose_phase6c_route,
    compose_diversity_fallback,
)
from neural_sat_speck.attacks.group_ranking import (
    PairGroupSubkeyRanking,
    evaluate_pair_group_subkey_ranking,
    rank_pair_group_subkeys,
)
from neural_sat_speck.attacks.ranking import (
    ALL_FINAL_ROUND_SUBKEYS,
    SubkeyRanking,
    evaluate_subkey_ranking,
    rank_final_round_subkeys,
)
from neural_sat_speck.confidence.calibration import (
    CalibrationObservation,
    PlattCalibrator,
    ThresholdSelection,
    bootstrap_mean_interval,
    calibration_report,
    fit_platt_calibrator,
    select_threshold,
    top8_score_separation,
)
from neural_sat_speck.data.attack import (
    ATTACK_KEY_GENERATION_POLICY,
    ATTACK_TASK,
    AttackInstance,
    AttackInstanceConfig,
    generate_attack_instances,
    load_attack_shard,
    verify_attack_shard,
    write_attack_shard,
)
from neural_sat_speck.data.format import (
    CIPHER_NAME,
    DATASET_FORMAT_VERSION,
    DATASET_LICENSE,
    PLAINTEXT_DIFFERENCE_POLICY,
    canonical_json,
)
from neural_sat_speck.data.generator import DEFAULT_INPUT_DIFFERENCE
from neural_sat_speck.data.metadata import BIT_ORDER, GENERATOR_VERSION, WORD_ORDER
from neural_sat_speck.neural.group_train import load_pair_group_checkpoint
from neural_sat_speck.neural.train import load_checkpoint

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "configs"
    / "experiments"
    / "phase6c_confidence_fallback_r6.json"
)
CONFIG_SCHEMA = "neural_sat_speck.phase6c_confidence_fallback_config.v1"
MANIFEST_SCHEMA = "neural_sat_speck.phase6c_confidence_fallback_manifest.v1"
RESULT_SCHEMA = "neural_sat_speck.phase6c_confidence_fallback_result.v1"
SUMMARY_SCHEMA = "neural_sat_speck.phase6c_confidence_fallback_summary.v1"
CALIBRATION_SCHEMA = "neural_sat_speck.phase6c_calibration_report.v1"
PHASE6B_RESULT_SCHEMA = "neural_sat_speck.phase6b_h1_adjudication_result.v1"
PHASE6B_MANIFEST_SCHEMA = "neural_sat_speck.phase6b_h1_adjudication_manifest.v1"
METHODS = ("b0", "b1")

# These are data-generation or attack-instance seeds already present in the
# repository.  Metric-resampling seeds are tracked separately in the config.
EXISTING_DATA_SEEDS = frozenset(
    {
        1001,
        2001,
        3001,
        4001,
        5001,
        6001,
        51001,
        52001,
        53001,
        61001,
        62001,
        63001,
        64001,
        65001,
        72001,
        73001,
        81001,
        82001,
        83001,
        91001,
        92001,
        93001,
        96001,
    }
)


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


def _peak_rss_bytes() -> int:
    """Return the process peak RSS in bytes on macOS and Linux."""

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


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


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SystemExit(f"configuration field {name!r} must be an object")
    return value


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f"configuration field {name!r} must be an integer")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SystemExit(f"configuration field {name!r} must be non-empty text")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    """Reject protocol drift before reading calibration or attack data."""

    if config.get("schema") != CONFIG_SCHEMA:
        raise SystemExit("Phase 6C configuration schema is not recognised")
    if _require_int(config.get("phase"), "phase") != 6:
        raise SystemExit("Phase 6C configuration must have phase=6")
    if config.get("validation_only") is not True:
        raise SystemExit("Phase 6C configuration must be validation-only")
    if config.get("held_out_test_generated") is not False:
        raise SystemExit("Phase 6C configuration must forbid test generation")

    calibration = _require_mapping(config.get("calibration"), "calibration")
    if calibration.get("source_dataset_id") != (
        "speck32_64_attack_phase6b_h1_adjudication_r6"
    ):
        raise SystemExit("calibration must use the authoritative Phase 6B dataset")
    if _require_int(calibration.get("source_seed"), "calibration.source_seed") != 96001:
        raise SystemExit("calibration source seed must be 96001")
    if _require_int(
        calibration.get("source_record_count"), "calibration.source_record_count"
    ) != 64:
        raise SystemExit("calibration must use all 64 Phase 6B records")
    if calibration.get("source_backend") != "eager":
        raise SystemExit("calibration must use eager Phase 6B records")
    if calibration.get("source_split_role") != "validation":
        raise SystemExit("calibration source must be validation-only")
    if calibration.get("target") != "true_pair_aware_rank <= 8":
        raise SystemExit("Phase 6C target must be pair-aware top-8 containment")
    if _require_int(calibration.get("top_k"), "calibration.top_k") != 8:
        raise SystemExit("Phase 6C calibration top_k must be eight")
    if calibration.get("feature_name") != "top1_top8_margin_over_score_std":
        raise SystemExit("Phase 6C feature identity is not frozen")
    if calibration.get("feature_formula") != (
        "(ordered_score_1 - ordered_score_8) / max(score_std_over_all_65536_candidates, 1e-12)"
    ):
        raise SystemExit("Phase 6C feature formula is not frozen")
    if calibration.get("score_std_epsilon") != 1e-12:
        raise SystemExit("Phase 6C score standard-deviation epsilon is not frozen")
    calibrator_config = _require_mapping(
        calibration.get("calibrator"), "calibration.calibrator"
    )
    if calibrator_config.get("type") != "one_dimensional_platt_logistic":
        raise SystemExit("Phase 6C calibrator type is not frozen")
    if calibrator_config.get("arithmetic") != "float64":
        raise SystemExit("Phase 6C calibrator must use float64 arithmetic")
    if calibrator_config.get("intercept_penalty") != "none":
        raise SystemExit("Phase 6C intercept must be unpenalized")
    if calibrator_config.get("slope_penalty") != "l2":
        raise SystemExit("Phase 6C slope penalty must be L2")
    if calibrator_config.get("l2_lambda") != 1.0:
        raise SystemExit("Phase 6C L2 lambda must be exactly 1.0")
    if _require_int(
        calibrator_config.get("max_iterations"), "calibration.max_iterations"
    ) <= 0:
        raise SystemExit("calibration.max_iterations must be positive")
    for name in ("gradient_tolerance", "step_tolerance"):
        value = calibrator_config.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise SystemExit(f"calibration.{name} must be positive and finite")
    threshold = _require_mapping(
        calibration.get("threshold"), "calibration.threshold"
    )
    if threshold.get("risk_limit") != 0.1:
        raise SystemExit("Phase 6C risk limit must be exactly 0.10")
    if _require_int(
        threshold.get("minimum_accepted"), "calibration.threshold.minimum_accepted"
    ) != 16:
        raise SystemExit("Phase 6C minimum accepted count must be 16")
    if threshold.get("accept_rule") != "estimated_probability >= selected_threshold":
        raise SystemExit("Phase 6C probability acceptance rule is not frozen")

    dataset = _require_mapping(config.get("dataset"), "dataset")
    if not _require_string(dataset.get("dataset_id"), "dataset.dataset_id").startswith(
        "speck32_64_attack_phase6c_confidence_fallback"
    ):
        raise SystemExit("Phase 6C dataset id is not a new validation identifier")
    if _require_int(dataset.get("rounds"), "dataset.rounds") != 6:
        raise SystemExit("Phase 6C requires six cipher rounds")
    if _require_int(dataset.get("pair_count"), "dataset.pair_count") != 64:
        raise SystemExit("Phase 6C requires exactly 64 related pairs")
    if _require_int(dataset.get("instance_count"), "dataset.instance_count") != 64:
        raise SystemExit("Phase 6C requires exactly 64 validation instances")
    if _require_int(dataset.get("seed"), "dataset.seed") != 97001:
        raise SystemExit("Phase 6C fresh validation seed must be 97001")
    if dataset.get("seed_namespace") != "validation_attack_instance_seed":
        raise SystemExit("Phase 6C dataset seed namespace is not frozen")
    if dataset.get("split_role") != "validation":
        raise SystemExit("Phase 6C dataset must have validation role")
    if dataset.get("input_difference") != list(DEFAULT_INPUT_DIFFERENCE):
        raise SystemExit("Phase 6C input difference is not frozen")
    if dataset.get("pair_construction") != PLAINTEXT_DIFFERENCE_POLICY:
        raise SystemExit("Phase 6C pair construction is not frozen")
    if dataset.get("key_generation_policy") != ATTACK_KEY_GENERATION_POLICY:
        raise SystemExit("Phase 6C key policy is not frozen")
    if int(dataset["seed"]) in EXISTING_DATA_SEEDS:
        raise SystemExit("Phase 6C dataset seed overlaps an existing data seed")

    domain = _require_mapping(config.get("candidate_domain"), "candidate_domain")
    if (
        _require_int(domain.get("first"), "candidate_domain.first") != 0
        or _require_int(domain.get("last"), "candidate_domain.last") != 0xFFFF
        or _require_int(domain.get("count"), "candidate_domain.count") != 1 << 16
        or domain.get("complete_16_bit_domain") is not True
    ):
        raise SystemExit("Phase 6C candidate domain must be complete 16-bit")

    methods = _require_mapping(config.get("methods"), "methods")
    b0 = _require_mapping(methods.get("b0"), "methods.b0")
    b1 = _require_mapping(methods.get("b1"), "methods.b1")
    if b0.get("method_id") != "b0_sum_logits" or b0.get("score_aggregation") != "sum_logits":
        raise SystemExit("B0 identity or score convention is not frozen")
    if b1.get("method_id") != "b1_group_logit" or b1.get("score_aggregation") != "group_logit":
        raise SystemExit("B1 identity or score convention is not frozen")
    if _require_int(b1.get("group_size"), "methods.b1.group_size") != 64:
        raise SystemExit("B1 group size must be 64")
    for method in (b0, b1):
        if not _require_string(method.get("checkpoint_path"), "checkpoint_path"):
            raise SystemExit("checkpoint path is missing")
        if len(_require_string(method.get("checkpoint_sha256"), "checkpoint_sha256")) != 64:
            raise SystemExit("checkpoint SHA-256 must be 64 characters")

    ranking = _require_mapping(config.get("ranking"), "ranking")
    if _require_int(ranking.get("pair_count"), "ranking.pair_count") != 64:
        raise SystemExit("ranking pair count must be 64")
    if ranking.get("candidate_domain") != "complete_16_bit_domain":
        raise SystemExit("ranking candidate domain is not frozen")
    if ranking.get("execution_order") != (
        "b0_then_b1_for_even_validation_index_b1_then_b0_for_odd"
    ):
        raise SystemExit("ranking execution order is not counterbalanced")
    for name in ("b0_candidate_batch_size", "b1_candidate_batch_size"):
        if _require_int(ranking.get(name), f"ranking.{name}") != 4096:
            raise SystemExit(f"{name} must be exactly 4096")

    fallback = _require_mapping(config.get("fallback"), "fallback")
    if fallback.get("trusted_method") != "b1_group_logit":
        raise SystemExit("trusted route must use B1")
    if _require_int(fallback.get("trusted_budget"), "fallback.trusted_budget") != 8:
        raise SystemExit("trusted route budget must be 8")
    if _require_int(fallback.get("head_size"), "fallback.head_size") != 16:
        raise SystemExit("fallback head size must be 16")
    if _require_int(fallback.get("budget"), "fallback.budget") != 32:
        raise SystemExit("fallback budget must be 32")
    if fallback.get("comparator") != "b1_top32":
        raise SystemExit("fallback comparator must be B1 top-32")
    if fallback.get("descriptive_baseline") != "b0_top32":
        raise SystemExit("B0 top-32 descriptive baseline is required")

    backend = _require_mapping(config.get("backend_policy"), "backend_policy")
    if backend.get("reference") != "eager_pytorch" or backend.get("device") != "cpu":
        raise SystemExit("Phase 6C reference backend must be eager PyTorch CPU")
    if backend.get("numpy_authoritative") is not False or backend.get("tolerance_relaxation") is not False:
        raise SystemExit("NumPy cannot be authoritative and tolerances cannot relax")
    runtime = _require_mapping(config.get("runtime"), "runtime")
    if runtime.get("device") != "cpu" or runtime.get("single_process_sequential") is not True:
        raise SystemExit("Phase 6C runtime must be one sequential CPU process")
    if runtime.get("concurrent_ranking_jobs") is not False:
        raise SystemExit("concurrent ranking jobs are forbidden")

    metrics = _require_mapping(config.get("metrics"), "metrics")
    if metrics.get("primary") != "route_specific_capped_attempts":
        raise SystemExit("Phase 6C primary metric is not frozen")
    if metrics.get("top_k_values") != [8, 16, 32]:
        raise SystemExit("Phase 6C top-k values are not frozen")
    if _require_int(metrics.get("bootstrap_replicates"), "metrics.bootstrap_replicates") <= 0:
        raise SystemExit("bootstrap_replicates must be positive")
    if _require_int(metrics.get("bootstrap_seed"), "metrics.bootstrap_seed") != 97101:
        raise SystemExit("Phase 6C bootstrap seed is not frozen")
    if metrics.get("bootstrap_seed_namespace") != "metric_resampling_only":
        raise SystemExit("bootstrap seed namespace must remain non-data")
    confidence = metrics.get("bootstrap_confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 < float(confidence) < 1.0
    ):
        raise SystemExit("metrics.bootstrap_confidence must be between zero and one")

    exact_recovery = _require_mapping(config.get("exact_recovery"), "exact_recovery")
    if exact_recovery.get("sat_executed") is not False or exact_recovery.get("verification_executed") is not False:
        raise SystemExit("Phase 6C-A must not execute SAT or verification")
    later_solver = _require_mapping(
        exact_recovery.get("later_common_solver_settings"),
        "exact_recovery.later_common_solver_settings",
    )
    if later_solver.get("reference_config") != "configs/experiments/phase5_integration_r6.json":
        raise SystemExit("later SAT settings must reference the common Phase 5 config")
    if later_solver.get("status") != "frozen_reference_only_not_run_in_phase6c_a":
        raise SystemExit("later SAT settings must remain deferred")
    timing = _require_mapping(config.get("timing_policy"), "timing_policy")
    if timing.get("fallback_online_cost_includes_b0_ranking") is not True:
        raise SystemExit("fallback online timing must include B0 ranking")
    if timing.get("sat_time_recorded_as") != "not_run" or timing.get("verification_time_recorded_as") != "not_run":
        raise SystemExit("SAT and verification timing status must be not_run")
    if timing.get("execution_order") != ranking.get("execution_order"):
        raise SystemExit("timing and ranking execution-order contracts disagree")
    if timing.get("single_process_sequential") is not True or timing.get("concurrent_ranking_jobs") is not False:
        raise SystemExit("Phase 6C timing must be sequential")

    exclusions = config.get("seed_exclusions")
    if not isinstance(exclusions, Sequence) or isinstance(exclusions, (str, bytes)):
        raise SystemExit("seed_exclusions must be a sequence")
    exclusion_values = tuple(_require_int(value, "seed_exclusions item") for value in exclusions)
    if len(set(exclusion_values)) != len(exclusion_values):
        raise SystemExit("seed_exclusions must not contain duplicates")
    if set(exclusion_values) != set(EXISTING_DATA_SEEDS):
        raise SystemExit("seed_exclusions must enumerate all known data seeds exactly")
    non_data = config.get("known_non_data_seed_namespaces")
    if not isinstance(non_data, Sequence) or isinstance(non_data, (str, bytes)):
        raise SystemExit("known_non_data_seed_namespaces must be a sequence")
    if not any(
        isinstance(item, Mapping)
        and item.get("seed") == 97001
        and item.get("namespace") == "phase6b_metric_bootstrap"
        and item.get("does_not_generate_keys") is True
        for item in non_data
    ):
        raise SystemExit("Phase 6B bootstrap namespace must be documented")


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Write canonical JSON through a same-directory atomic replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(value).encode("utf-8")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            temporary_path = Path(stream.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _validate_full_ranking_payload(payload: Mapping[str, Any], name: str) -> None:
    values = payload.get("candidate_values")
    scores = payload.get("candidate_scores")
    order = payload.get("candidate_order")
    if not all(isinstance(item, Sequence) and not isinstance(item, (str, bytes)) for item in (values, scores, order)):
        raise ValueError(f"{name} ranking arrays are missing")
    candidates = tuple(values)
    candidate_order = tuple(order)
    if candidates != ALL_FINAL_ROUND_SUBKEYS:
        raise ValueError(f"{name} does not describe the complete ascending domain")
    if any(
        isinstance(candidate, bool) or not isinstance(candidate, int)
        for candidate in candidate_order
    ):
        raise ValueError(f"{name} candidate order contains a non-integer")
    if len(scores) != len(candidates):
        raise ValueError(f"{name} ranking array lengths are invalid")
    if len(candidate_order) != len(candidates) or set(candidate_order) != set(candidates):
        raise ValueError(f"{name} candidate order is not a permutation")
    if any(
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        for score in scores
    ):
        raise ValueError(f"{name} contains non-finite scores")


def _read_source_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    calibration = _require_mapping(config["calibration"], "calibration")
    path = _repository_path(str(calibration["source_manifest_path"])).resolve()
    if not path.is_file():
        raise SystemExit(f"calibration source manifest is missing: {path}")
    actual_hash = _sha256_file(path)
    if actual_hash != calibration.get("source_manifest_sha256"):
        raise SystemExit("calibration source manifest hash does not match config")
    manifest = _load_json(path)
    if manifest.get("schema") != PHASE6B_MANIFEST_SCHEMA:
        raise SystemExit("calibration source manifest schema is not Phase 6B")
    if manifest.get("dataset_id") != calibration.get("source_dataset_id"):
        raise SystemExit("calibration source manifest dataset id is wrong")
    if manifest.get("validation_only") is not True or manifest.get("held_out_test_generated") is not False:
        raise SystemExit("calibration source manifest is not validation-only")
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {"validation"}:
        raise SystemExit("calibration source manifest must contain validation only")
    record = splits["validation"]
    if not isinstance(record, Mapping) or record.get("seed") != 96001 or record.get("split_role") != "validation":
        raise SystemExit("calibration source manifest validation seed/role is wrong")
    if record.get("instance_count") != 64 or record.get("pair_count") != 64:
        raise SystemExit("calibration source manifest counts are wrong")
    return manifest


def _calibration_observations(
    config: Mapping[str, Any],
) -> tuple[tuple[CalibrationObservation, ...], dict[str, Any]]:
    """Read and validate exactly the frozen Phase 6B eager records."""

    calibration = _require_mapping(config["calibration"], "calibration")
    manifest = _read_source_manifest(config)
    source_glob = _repository_path(str(calibration["source_result_glob"])).resolve()
    paths = tuple(sorted(source_glob.parent.glob(source_glob.name)))
    if len(paths) != int(calibration["source_record_count"]):
        raise SystemExit("calibration source does not contain exactly 64 records")
    observations: list[CalibrationObservation] = []
    source_records: list[dict[str, Any]] = []
    indices: set[int] = set()
    for path in paths:
        record = _load_json(path)
        if record.get("schema") != PHASE6B_RESULT_SCHEMA:
            raise SystemExit(f"calibration record {path} has the wrong schema")
        if record.get("phase") != 6 or record.get("study") != "phase6b_h1_b0_b1_adjudication":
            raise SystemExit(f"calibration record {path} is not Phase 6B H1")
        dataset = _require_mapping(record.get("dataset"), "calibration.dataset")
        if dataset.get("dataset_id") != calibration.get("source_dataset_id") or dataset.get("validation_only") is not True or dataset.get("held_out_test_generated") is not False:
            raise SystemExit(f"calibration record {path} is not validation-only Phase 6B")
        if dataset.get("manifest_sha256") != calibration.get("source_manifest_sha256"):
            raise SystemExit(f"calibration record {path} uses another source manifest")
        provenance = _require_mapping(record.get("provenance"), "calibration.provenance")
        if provenance.get("inference_backend") != "eager" or provenance.get("b1_inference_backend") != "pytorch":
            raise SystemExit(f"calibration record {path} is not eager PyTorch")
        instance = _require_mapping(record.get("instance"), "calibration.instance")
        index = instance.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 64:
            raise SystemExit(f"calibration record {path} has invalid index")
        if index in indices or instance.get("split_role") != "validation":
            raise SystemExit("calibration records do not form one validation permutation")
        indices.add(index)
        instance_id = instance.get("instance_id")
        if instance_id != f"validation-{index:04d}" or instance.get("pair_count") != 64:
            raise SystemExit(f"calibration record {path} has invalid instance metadata")
        methods = _require_mapping(record.get("methods"), "calibration.methods")
        b1 = _require_mapping(methods.get("b1"), "calibration.methods.b1")
        if b1.get("method_id") != "b1_group_logit":
            raise SystemExit(f"calibration record {path} has the wrong B1 method")
        ranking = _require_mapping(b1.get("ranking"), "calibration.methods.b1.ranking")
        _validate_full_ranking_payload(ranking, f"calibration record {path}")
        evaluation = _require_mapping(b1.get("evaluation"), "calibration.methods.b1.evaluation")
        true_rank = evaluation.get("true_rank")
        if isinstance(true_rank, bool) or not isinstance(true_rank, int) or not 1 <= true_rank <= 65536:
            raise SystemExit(f"calibration record {path} has invalid B1 true rank")
        feature = top8_score_separation(ranking)
        observations.append(
            CalibrationObservation(
                instance_id=str(instance_id),
                feature=feature,
                target_top8=true_rank <= 8,
            )
        )
        source_records.append(
            {
                "path": _relative_or_absolute(path),
                "sha256": _sha256_file(path),
                "instance_id": instance_id,
                "index": index,
            }
        )
    if indices != set(range(64)):
        raise SystemExit("calibration records do not cover all validation indices")
    ordered = tuple(sorted(observations, key=lambda item: item.instance_id))
    if sum(item.target_top8 for item in ordered) != 41:
        raise SystemExit("Phase 6B top-8 calibration target is not the frozen 41/64")
    return ordered, {
        "manifest_path": calibration["source_manifest_path"],
        "manifest_sha256": calibration["source_manifest_sha256"],
        "record_count": len(ordered),
        "records": source_records,
        "manifest": {
            "dataset_id": manifest["dataset_id"],
            "seed": manifest["splits"]["validation"]["seed"],
            "split_role": manifest["splits"]["validation"]["split_role"],
        },
    }


def _fit_calibration(
    config: Mapping[str, Any],
    *,
    config_path: Path | None = None,
) -> tuple[dict[str, Any], PlattCalibrator, ThresholdSelection, tuple[CalibrationObservation, ...]]:
    """Fit and report the frozen calibrator without changing experimental data."""

    observations, source = _calibration_observations(config)
    calibration = _require_mapping(config["calibration"], "calibration")
    calibrator_config = _require_mapping(calibration["calibrator"], "calibration.calibrator")
    calibrator = fit_platt_calibrator(
        tuple(item.feature for item in observations),
        tuple(item.target_top8 for item in observations),
        l2_lambda=float(calibrator_config["l2_lambda"]),
        max_iterations=int(calibrator_config["max_iterations"]),
        gradient_tolerance=float(calibrator_config["gradient_tolerance"]),
        step_tolerance=float(calibrator_config["step_tolerance"]),
    )
    probabilities = calibrator.predict_many(tuple(item.feature for item in observations))
    targets = tuple(item.target_top8 for item in observations)
    report = calibration_report(
        probabilities,
        targets,
        instance_ids=tuple(item.instance_id for item in observations),
    )
    threshold_config = _require_mapping(calibration["threshold"], "calibration.threshold")
    selection = select_threshold(
        probabilities,
        targets,
        risk_limit=float(threshold_config["risk_limit"]),
        minimum_accepted=int(threshold_config["minimum_accepted"]),
    )
    result: dict[str, Any] = {
        "schema": CALIBRATION_SCHEMA,
        "validation_only": True,
        "held_out_test_generated": False,
        "source": source,
        "observations": [item.to_dict() for item in observations],
        "probabilities": [
            {"instance_id": item.instance_id, "probability": probability, "target_top8": item.target_top8}
            for item, probability in zip(observations, probabilities, strict=True)
        ],
        "calibrator": calibrator.to_dict(),
        "metrics": report,
        "threshold_selection": selection.to_dict(),
        "provenance": {
            "config_path": _relative_or_absolute(config_path) if config_path else None,
            "config_sha256": _sha256_file(config_path) if config_path else None,
            "calibration_source_is_validation_only": True,
            "held_out_accessed": False,
            "repository_commit": _repository_commit(),
        },
    }
    return result, calibrator, selection, observations


def _manifest_document(
    instance_set: Any,
    *,
    config: Mapping[str, Any],
    repository_commit: str | None,
) -> dict[str, Any]:
    """Build the future Phase 6C validation-only attack manifest."""

    dataset = _require_mapping(config["dataset"], "dataset")
    return {
        "schema": MANIFEST_SCHEMA,
        "phase": 6,
        "study": "phase6c_confidence_gated_pair_aware_fallback",
        "validation_only": True,
        "held_out_test_generated": False,
        "bit_order": BIT_ORDER,
        "cipher": CIPHER_NAME,
        "dataset_format_version": DATASET_FORMAT_VERSION,
        "dataset_id": dataset["dataset_id"],
        "generated_at": dataset["generated_at"],
        "seed_namespace": dataset["seed_namespace"],
        "seed_exclusions": sorted(EXISTING_DATA_SEEDS),
        "generator_version": GENERATOR_VERSION,
        "input_difference": list(DEFAULT_INPUT_DIFFERENCE),
        "key_generation_policy": ATTACK_KEY_GENERATION_POLICY,
        "license": DATASET_LICENSE,
        "pair_construction": PLAINTEXT_DIFFERENCE_POLICY,
        "repository_commit": repository_commit,
        "rounds": 6,
        "splits": {
            "validation": {
                "data_checksum": instance_set.metadata["data_checksum"],
                "file_sha256": None,
                "instance_count": instance_set.metadata["instance_count"],
                "pair_count": instance_set.metadata["pair_count"],
                "path": "validation.json",
                "seed": instance_set.metadata["seed"],
                "split_role": "validation",
            }
        },
        "task": ATTACK_TASK,
        "word_order": WORD_ORDER,
    }


def _verify_manifest(
    manifest: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    manifest_path: Path,
) -> None:
    dataset = _require_mapping(config["dataset"], "dataset")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise SystemExit("Phase 6C manifest schema is not recognised")
    expected = {
        "phase": 6,
        "study": "phase6c_confidence_gated_pair_aware_fallback",
        "validation_only": True,
        "held_out_test_generated": False,
        "cipher": CIPHER_NAME,
        "dataset_format_version": DATASET_FORMAT_VERSION,
        "dataset_id": dataset["dataset_id"],
        "seed_namespace": dataset["seed_namespace"],
        "generator_version": GENERATOR_VERSION,
        "input_difference": list(DEFAULT_INPUT_DIFFERENCE),
        "key_generation_policy": ATTACK_KEY_GENERATION_POLICY,
        "license": DATASET_LICENSE,
        "pair_construction": PLAINTEXT_DIFFERENCE_POLICY,
        "rounds": 6,
        "task": ATTACK_TASK,
        "word_order": WORD_ORDER,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise SystemExit(f"Phase 6C manifest field {field!r} is not frozen")
    if manifest.get("bit_order") != BIT_ORDER or manifest.get("generated_at") != dataset.get("generated_at"):
        raise SystemExit("Phase 6C manifest metadata is not frozen")
    if manifest.get("seed_exclusions") != sorted(EXISTING_DATA_SEEDS):
        raise SystemExit("Phase 6C manifest seed exclusions are not frozen")
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {"validation"}:
        raise SystemExit("Phase 6C manifest must contain validation only")
    record = splits["validation"]
    if not isinstance(record, Mapping):
        raise SystemExit("Phase 6C validation manifest record is invalid")
    for field, value in {
        "instance_count": dataset["instance_count"],
        "pair_count": dataset["pair_count"],
        "seed": dataset["seed"],
        "split_role": "validation",
        "path": "validation.json",
    }.items():
        if record.get(field) != value:
            raise SystemExit(f"Phase 6C validation manifest field {field!r} is wrong")
    if not manifest_path.is_file():
        raise SystemExit("Phase 6C manifest disappeared during verification")


def _write_or_verify_dataset(config: Mapping[str, Any], *, generate: bool) -> Path:
    """Create or verify one validation shard, never a test shard."""

    dataset = _require_mapping(config["dataset"], "dataset")
    dataset_id = str(dataset["dataset_id"])
    output_dir = _repository_path(str(dataset["output_dir"])).resolve()
    manifest_path = _repository_path(str(dataset["manifest_path"])).resolve()
    if manifest_path.parent != output_dir:
        raise SystemExit("Phase 6C manifest must be inside dataset.output_dir")
    shard_path = output_dir / "validation.json"
    if (output_dir / "test.json").exists():
        raise SystemExit("refusing to operate on a Phase 6C output containing test.json")
    if generate:
        if manifest_path.exists() or shard_path.exists():
            raise SystemExit("refusing to overwrite an existing Phase 6C dataset")
        instance_set = generate_attack_instances(
            AttackInstanceConfig(
                instance_count=int(dataset["instance_count"]),
                pair_count=int(dataset["pair_count"]),
                rounds=int(dataset["rounds"]),
                seed=int(dataset["seed"]),
                split_role=str(dataset["split_role"]),
                input_difference=tuple(dataset["input_difference"]),
            ),
            generated_at=str(dataset["generated_at"]),
            repository_commit=_repository_commit(),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        temporary_shard = output_dir / ".validation.json.pending"
        try:
            record = write_attack_shard(instance_set, temporary_shard, dataset_id=dataset_id)
            os.replace(temporary_shard, shard_path)
        finally:
            temporary_shard.unlink(missing_ok=True)
        manifest = _manifest_document(
            instance_set,
            config=config,
            repository_commit=instance_set.metadata["repository_commit"],
        )
        manifest["splits"]["validation"]["file_sha256"] = record["file_sha256"]
        _write_atomic(manifest_path, manifest)
    if not manifest_path.is_file() or not shard_path.is_file():
        raise SystemExit("Phase 6C dataset is missing; use --generate at the reviewed launch")
    payload = manifest_path.read_bytes()
    manifest = _load_json(manifest_path)
    if payload != canonical_json(manifest).encode("utf-8"):
        raise SystemExit("Phase 6C manifest is not canonical JSON")
    _verify_manifest(manifest, config=config, manifest_path=manifest_path)
    verify_attack_shard(
        shard_path,
        expected_dataset_id=dataset_id,
        expected_manifest_record=manifest["splits"]["validation"],
    )
    return manifest_path


def _load_models(config: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    """Load and verify the frozen B0 and B1 checkpoints."""

    methods = _require_mapping(config["methods"], "methods")
    b0_info = _require_mapping(methods["b0"], "methods.b0")
    b1_info = _require_mapping(methods["b1"], "methods.b1")
    b0_path = _repository_path(str(b0_info["checkpoint_path"])).resolve()
    b1_path = _repository_path(str(b1_info["checkpoint_path"])).resolve()
    if not b0_path.is_file() or not b1_path.is_file():
        raise SystemExit("one or more Phase 6C frozen checkpoints are missing")
    b0_hash = _sha256_file(b0_path)
    b1_hash = _sha256_file(b1_path)
    if b0_hash != b0_info.get("checkpoint_sha256") or b1_hash != b1_info.get("checkpoint_sha256"):
        raise SystemExit("Phase 6C checkpoint hash does not match the frozen config")
    b0_model, b0_metadata = load_checkpoint(b0_path, device="cpu")
    b1_model, b1_metadata = load_pair_group_checkpoint(b1_path, device="cpu")
    if b0_metadata.get("phase") != 3:
        raise SystemExit("Phase 6C B0 checkpoint is not the frozen Phase 3 model")
    if b1_metadata.get("model_id") != b1_info.get("model_id") or getattr(b1_model, "model_id", None) != b1_info.get("model_id"):
        raise SystemExit("Phase 6C B1 model identity does not match config")
    if getattr(b1_model, "group_size", None) != 64:
        raise SystemExit("Phase 6C B1 group size is not 64")
    b0_model.eval()
    b1_model.eval()
    metadata = {
        "b0": {
            "checkpoint_path": _relative_or_absolute(b0_path),
            "checkpoint_sha256": b0_hash,
            "model_id": b0_info["model_id"],
            "metadata": b0_metadata,
        },
        "b1": {
            "checkpoint_path": _relative_or_absolute(b1_path),
            "checkpoint_sha256": b1_hash,
            "model_id": b1_info["model_id"],
            "metadata": b1_metadata,
        },
    }
    return b0_model, b1_model, metadata


def _counterbalanced_execution_order(index: int) -> tuple[str, str]:
    """Counterbalance B0-first and B1-first deterministically by index."""

    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("validation index must be a non-negative integer")
    return ("b0", "b1") if index % 2 == 0 else ("b1", "b0")


def _rank_complete_methods(
    instance: AttackInstance,
    b0_model: torch.nn.Module,
    b1_model: torch.nn.Module,
    *,
    config: Mapping[str, Any],
    index: int,
) -> tuple[SubkeyRanking, PairGroupSubkeyRanking, tuple[str, str]]:
    """Rank B0 and B1 over the full domain in the frozen order."""

    ranking = _require_mapping(config["ranking"], "ranking")
    execution_order = _counterbalanced_execution_order(index)
    results: dict[str, SubkeyRanking | PairGroupSubkeyRanking] = {}
    for method in execution_order:
        if method == "b0":
            results[method] = rank_final_round_subkeys(
                instance,
                b0_model,
                candidate_values=ALL_FINAL_ROUND_SUBKEYS,
                candidate_batch_size=int(ranking["b0_candidate_batch_size"]),
                device="cpu",
            )
        else:
            results[method] = rank_pair_group_subkeys(
                instance,
                b1_model,  # type: ignore[arg-type]
                candidate_values=ALL_FINAL_ROUND_SUBKEYS,
                candidate_batch_size=int(ranking["b1_candidate_batch_size"]),
                device="cpu",
            )
    b0_result = results["b0"]
    b1_result = results["b1"]
    if not isinstance(b0_result, SubkeyRanking) or not isinstance(b1_result, PairGroupSubkeyRanking):
        raise RuntimeError("Phase 6C ranking returned unexpected result types")
    return b0_result, b1_result, execution_order


def _evaluate_order(
    candidate_order: Sequence[int],
    true_subkey: int,
    *,
    budget: int,
    top_k_values: Sequence[int] = (8, 16, 32),
) -> dict[str, Any]:
    """Evaluate a route order post-hoc, including a route-specific cap."""

    order = tuple(candidate_order)
    rank = order.index(true_subkey) + 1 if true_subkey in order else None
    capped = budget + 1 if rank is None else min(rank, budget + 1)
    return {
        "true_final_round_subkey": true_subkey,
        "true_rank": rank,
        "true_subkey_in_candidate_set": rank is not None,
        "top_k_membership": {
            str(k): bool(rank is not None and rank <= k) for k in top_k_values
        },
        "capped_attempts": capped,
        "budget": budget,
    }


def _record_determinism_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    """Remove timing and host metadata before comparing a deterministic rerun."""

    methods = _require_mapping(record["methods"], "record.methods")
    return {
        "methods": {
            method: {
                "ranking": {
                    "candidate_scores": methods[method]["ranking"]["candidate_scores"],
                    "candidate_order": methods[method]["ranking"]["candidate_order"],
                },
                "evaluation": methods[method]["evaluation"],
            }
            for method in METHODS
        },
        "confidence": record["confidence"],
        "route": record["route"],
        "comparators": record["comparators"],
    }


def _build_record(
    instance: AttackInstance,
    b0_ranking: SubkeyRanking,
    b1_ranking: PairGroupSubkeyRanking,
    *,
    config: Mapping[str, Any],
    config_path: Path,
    manifest_path: Path,
    model_metadata: Mapping[str, Any],
    calibration: PlattCalibrator,
    threshold: ThresholdSelection,
    index: int,
    execution_order: Sequence[str],
) -> dict[str, Any]:
    """Build one atomic-result payload without SAT or verification calls."""

    if not threshold.feasible or threshold.threshold is None:
        raise RuntimeError("cannot build Phase 6C records without a selected threshold")
    metrics = _require_mapping(config["metrics"], "metrics")
    confidence_started = time.perf_counter()
    b1_feature = top8_score_separation(b1_ranking.to_dict())
    probability = calibration.predict(b1_feature)
    route_name = choose_phase6c_route(probability, threshold.threshold)
    confidence_seconds = time.perf_counter() - confidence_started
    composition_started = time.perf_counter()
    if route_name == "trusted_top8":
        route_order = b1_ranking.top_k(PHASE6C_TRUSTED_BUDGET)
        route_budget = PHASE6C_TRUSTED_BUDGET
    else:
        route_order = compose_diversity_fallback(
            b1_ranking.candidate_order,
            b0_ranking.candidate_order,
        )
        route_budget = PHASE6C_FALLBACK_BUDGET
    composition_seconds = time.perf_counter() - composition_started
    top_k_values = tuple(int(value) for value in metrics["top_k_values"])
    b0_eval = evaluate_subkey_ranking(
        b0_ranking,
        instance.final_round_subkey,
        top_k_values=top_k_values,
    )
    b1_eval = evaluate_pair_group_subkey_ranking(
        b1_ranking,
        instance.final_round_subkey,
        top_k_values=top_k_values,
    )
    route_eval = _evaluate_order(
        route_order,
        instance.final_round_subkey,
        budget=route_budget,
        top_k_values=top_k_values,
    )
    b1_top32_eval = _evaluate_order(
        b1_ranking.top_k(32),
        instance.final_round_subkey,
        budget=32,
        top_k_values=top_k_values,
    )
    b0_top32_eval = _evaluate_order(
        b0_ranking.top_k(32),
        instance.final_round_subkey,
        budget=32,
        top_k_values=top_k_values,
    )
    fallback_order = compose_diversity_fallback(
        b1_ranking.candidate_order,
        b0_ranking.candidate_order,
    )
    if len(fallback_order) != PHASE6C_FALLBACK_BUDGET or len(set(fallback_order)) != PHASE6C_FALLBACK_BUDGET:
        raise RuntimeError("Phase 6C fallback is not a unique 32-candidate order")
    methods = _require_mapping(config["methods"], "methods")
    dataset = _require_mapping(config["dataset"], "dataset")
    b0_seconds = float(b0_ranking.total_seconds)
    b1_seconds = float(b1_ranking.total_seconds)
    route_online_seconds = b1_seconds + confidence_seconds + composition_seconds
    if route_name == "diversity_fallback":
        route_online_seconds += b0_seconds
    ranking_only_total = b0_seconds + b1_seconds + confidence_seconds + composition_seconds
    return {
        "schema": RESULT_SCHEMA,
        "phase": 6,
        "study": "phase6c_confidence_gated_pair_aware_fallback",
        "validation_only": True,
        "held_out_test_generated": False,
        "dataset": {
            "dataset_id": dataset["dataset_id"],
            "manifest_path": _relative_or_absolute(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "validation_only": True,
            "held_out_test_generated": False,
        },
        "instance": {
            "split_role": "validation",
            "index": index,
            "instance_id": instance.instance_id,
            "pair_count": len(instance.pairs),
            "execution_order": list(execution_order),
        },
        "method_order": ["b0", "b1"],
        "methods": {
            "b0": {
                "method_id": methods["b0"]["method_id"],
                "ranking": b0_ranking.to_dict(),
                "evaluation": b0_eval,
            },
            "b1": {
                "method_id": methods["b1"]["method_id"],
                "ranking": b1_ranking.to_dict(),
                "evaluation": b1_eval,
            },
        },
        "confidence": {
            "feature_name": "top1_top8_margin_over_score_std",
            "feature": b1_feature,
            "estimated_probability": probability,
            "selected_threshold": threshold.threshold,
            "target": "true_pair_aware_rank <= 8",
        },
        "route": {
            "route_id": route_name,
            "budget": route_budget,
            "candidate_order": list(route_order),
            "evaluation": route_eval,
            "composition_seconds": composition_seconds,
            "false_accept": route_name == "trusted_top8" and b1_eval["true_rank"] > 8,
            "false_abstention": route_name == "diversity_fallback" and b1_eval["true_rank"] <= 8,
        },
        "fallback": {
            "method_id": "phase6c_b1_b0_alternating_top16_fill_b1_to32",
            "head_size": PHASE6C_FALLBACK_HEAD_SIZE,
            "budget": PHASE6C_FALLBACK_BUDGET,
            "candidate_order": list(fallback_order),
            "unique": True,
        },
        "comparators": {
            "b1_top32": b1_top32_eval,
            "b0_top32": b0_top32_eval,
        },
        "timing": {
            "b0_ranking_seconds": b0_seconds,
            "b1_ranking_seconds": b1_seconds,
            "confidence_seconds": confidence_seconds,
            "fallback_composition_seconds": composition_seconds,
            "route_online_seconds": route_online_seconds,
            "ranking_only_total_seconds": ranking_only_total,
            "sat_seconds": None,
            "verification_seconds": None,
            "sat_status": "not_run",
            "verification_status": "not_run",
            "total_seconds": ranking_only_total,
        },
        "provenance": {
            "config_path": _relative_or_absolute(config_path),
            "config_sha256": _sha256_file(config_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "repository_commit": _repository_commit(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch_version": str(torch.__version__),
            "device": "cpu",
            "inference_backend": "eager_pytorch",
            "execution_order": list(execution_order),
            "checkpoints": model_metadata,
            "held_out_accessed": False,
            "peak_rss_bytes": _peak_rss_bytes(),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        },
    }


def _validate_saved_result(
    record: Mapping[str, Any],
    *,
    index: int,
    config_path: Path,
    manifest_path: Path,
) -> None:
    """Reject stale or incomplete records before resumable skipping."""

    if record.get("schema") != RESULT_SCHEMA or record.get("validation_only") is not True or record.get("held_out_test_generated") is not False:
        raise SystemExit("saved Phase 6C result has the wrong schema/role")
    instance = _require_mapping(record.get("instance"), "result.instance")
    if instance.get("index") != index or instance.get("split_role") != "validation":
        raise SystemExit("saved Phase 6C result has the wrong validation index")
    if instance.get("execution_order") != list(_counterbalanced_execution_order(index)):
        raise SystemExit("saved Phase 6C result has the wrong counterbalanced order")
    dataset = _require_mapping(record.get("dataset"), "result.dataset")
    if dataset.get("manifest_sha256") != _sha256_file(manifest_path):
        raise SystemExit("saved Phase 6C result uses another manifest")
    provenance = _require_mapping(record.get("provenance"), "result.provenance")
    if provenance.get("config_sha256") != _sha256_file(config_path) or provenance.get("inference_backend") != "eager_pytorch":
        raise SystemExit("saved Phase 6C result uses another config/backend")
    if provenance.get("held_out_accessed") is not False:
        raise SystemExit("saved Phase 6C result does not prove held-out isolation")
    rss = provenance.get("peak_rss_bytes")
    if isinstance(rss, bool) or not isinstance(rss, int) or rss <= 0:
        raise SystemExit("saved Phase 6C result has no peak RSS")
    methods = _require_mapping(record.get("methods"), "result.methods")
    for method in METHODS:
        method_record = _require_mapping(methods.get(method), f"result.methods.{method}")
        _validate_full_ranking_payload(
            _require_mapping(method_record.get("ranking"), f"result.methods.{method}.ranking"),
            f"saved {method}",
        )
    confidence = _require_mapping(record.get("confidence"), "result.confidence")
    probability = confidence.get("estimated_probability")
    if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not 0.0 <= float(probability) <= 1.0:
        raise SystemExit("saved Phase 6C result has an invalid probability")
    route = _require_mapping(record.get("route"), "result.route")
    order = route.get("candidate_order")
    route_id = route.get("route_id")
    if route_id not in ("trusted_top8", "diversity_fallback"):
        raise SystemExit("saved Phase 6C result has an unknown route")
    if not isinstance(order, Sequence) or isinstance(order, (str, bytes)):
        raise SystemExit("saved Phase 6C route has no candidate order")
    if any(
        isinstance(candidate, bool)
        or not isinstance(candidate, int)
        or not 0 <= candidate <= 0xFFFF
        for candidate in order
    ) or len(set(order)) != len(order):
        raise SystemExit("saved Phase 6C route is not a unique 16-bit order")
    if route_id == "trusted_top8" and len(order) != 8:
        raise SystemExit("saved trusted route has the wrong budget")
    if route_id == "diversity_fallback" and len(order) != 32:
        raise SystemExit("saved fallback route has the wrong budget")


def _preflight_path(config: Mapping[str, Any]) -> Path:
    """Return the marker used to authorize a later full-domain run."""

    return _repository_path(str(config["output_root"])).resolve() / "eager" / "preflight.json"


def _write_preflight(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    manifest_path: Path,
    record: Mapping[str, Any],
) -> None:
    """Persist evidence from a deterministic full-domain smoke rerun."""

    rss = int(record["provenance"]["peak_rss_bytes"])
    smoke = _require_mapping(config["smoke"], "smoke")
    limit = int(smoke["peak_rss_safe_limit_bytes"])
    _write_atomic(
        _preflight_path(config),
        {
            "schema": "neural_sat_speck.phase6c_preflight.v1",
            "full_domain": True,
            "deterministic_rerun": True,
            "passed": rss <= limit,
            "peak_rss_bytes": rss,
            "peak_rss_safe_limit_bytes": limit,
            "config_path": _relative_or_absolute(config_path),
            "config_sha256": _sha256_file(config_path),
            "manifest_path": _relative_or_absolute(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "held_out_accessed": False,
        },
    )


def _require_preflight(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    manifest_path: Path,
) -> None:
    """Require current full-domain smoke and resource evidence before 64 rows."""

    path = _preflight_path(config)
    if not path.is_file():
        raise SystemExit(
            "full Phase 6C run requires a passing full-domain deterministic smoke first"
        )
    record = _load_json(path)
    if (
        record.get("schema") != "neural_sat_speck.phase6c_preflight.v1"
        or record.get("full_domain") is not True
        or record.get("deterministic_rerun") is not True
        or record.get("passed") is not True
        or record.get("held_out_accessed") is not False
        or record.get("config_sha256") != _sha256_file(config_path)
        or record.get("manifest_sha256") != _sha256_file(manifest_path)
    ):
        raise SystemExit("Phase 6C full-run preflight evidence is stale or failed")
    smoke = _require_mapping(config["smoke"], "smoke")
    rss = record.get("peak_rss_bytes")
    if isinstance(rss, bool) or not isinstance(rss, int) or rss > int(smoke["peak_rss_safe_limit_bytes"]):
        raise SystemExit("Phase 6C smoke peak RSS exceeds the frozen safe limit")


def _stats(
    evaluations: Sequence[Mapping[str, Any]],
    *,
    budget: int,
    top_k_values: Sequence[int] = (8, 16, 32),
) -> dict[str, Any]:
    if not evaluations:
        return {
            "count": 0,
            "true_positions": [],
            "capped_attempts": [],
            "top_k_containment": {str(k): 0 for k in top_k_values},
            "median_capped_attempts": None,
            "mean_capped_attempts": None,
            "worst_position": None,
            "budget": budget,
        }
    positions = [evaluation.get("true_rank") for evaluation in evaluations]
    capped = [
        budget + 1 if position is None else min(int(position), budget + 1)
        for position in positions
    ]
    finite_positions = [int(position) for position in positions if position is not None]
    ordered_capped = sorted(capped)
    middle = len(ordered_capped) // 2
    median = (
        ordered_capped[middle]
        if len(ordered_capped) % 2
        else (ordered_capped[middle - 1] + ordered_capped[middle]) / 2
    )
    return {
        "count": len(evaluations),
        "true_positions": positions,
        "capped_attempts": capped,
        "top_k_containment": {
            str(k): sum(position is not None and int(position) <= k for position in positions)
            for k in top_k_values
        },
        "median_capped_attempts": median,
        "mean_capped_attempts": sum(capped) / len(capped),
        "worst_position": max(finite_positions, default=None),
        "budget": budget,
    }


def _win_tie_loss(first: Sequence[int], second: Sequence[int]) -> dict[str, int]:
    if len(first) != len(second):
        raise ValueError("paired comparisons require equal lengths")
    return {
        "wins": sum(a < b for a, b in zip(first, second, strict=True)),
        "ties": sum(a == b for a, b in zip(first, second, strict=True)),
        "losses": sum(a > b for a, b in zip(first, second, strict=True)),
    }


def _summary(
    records: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    config_path: Path,
    manifest_path: Path,
    calibration_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate route, comparator, timing, and paired metrics after all 64."""

    if len(records) != 64:
        raise ValueError("Phase 6C summary requires all 64 records")
    for index, record in enumerate(sorted(records, key=lambda item: item["instance"]["index"])):
        _validate_saved_result(
            record,
            index=index,
            config_path=config_path,
            manifest_path=manifest_path,
        )
    ordered = tuple(sorted(records, key=lambda item: item["instance"]["index"]))
    b1_top32 = tuple(record["comparators"]["b1_top32"] for record in ordered)
    b0_top32 = tuple(record["comparators"]["b0_top32"] for record in ordered)
    # Fallback-vs-B1 is compared on every row using the stored fallback order,
    # regardless of whether the gate selected it for that row.
    if any("true_final_round_subkey" not in record["methods"]["b0"]["evaluation"] for record in ordered):
        raise ValueError("Phase 6C records must retain the post-hoc true subkey for summary")
    fallback_positions = [
        _evaluate_order(
            record["fallback"]["candidate_order"],
            int(record["methods"]["b0"]["evaluation"]["true_final_round_subkey"]),
            budget=32,
        )
        for record in ordered
    ]
    top_k_values = tuple(int(value) for value in config["metrics"]["top_k_values"])
    b1_top32_stats = _stats(b1_top32, budget=32, top_k_values=top_k_values)
    b0_top32_stats = _stats(b0_top32, budget=32, top_k_values=top_k_values)
    fallback_stats = _stats(fallback_positions, budget=32, top_k_values=top_k_values)
    fallback_capped = fallback_stats["capped_attempts"]
    b1_capped = b1_top32_stats["capped_attempts"]
    differences = [
        fallback - b1
        for fallback, b1 in zip(fallback_capped, b1_capped, strict=True)
    ]
    routes = [record["route"]["route_id"] for record in ordered]
    false_accepts = sum(
        route == "trusted_top8" and not bool(record["methods"]["b1"]["evaluation"]["top_k_membership"]["8"])
        for route, record in zip(routes, ordered, strict=True)
    )
    false_abstentions = sum(
        route == "diversity_fallback" and bool(record["methods"]["b1"]["evaluation"]["top_k_membership"]["8"])
        for route, record in zip(routes, ordered, strict=True)
    )
    timings = [record["timing"] for record in ordered]
    peak_rss = [int(record["provenance"]["peak_rss_bytes"]) for record in ordered]
    calibration_selection = calibration_result["threshold_selection"]
    trusted_evaluations = tuple(
        record["route"]["evaluation"]
        for record in ordered
        if record["route"]["route_id"] == "trusted_top8"
    )
    selected_fallback_evaluations = tuple(
        record["route"]["evaluation"]
        for record in ordered
        if record["route"]["route_id"] == "diversity_fallback"
    )
    trusted_count = routes.count("trusted_top8")
    fallback_count = routes.count("diversity_fallback")
    trusted_containment = sum(
        route == "trusted_top8"
        and bool(record["methods"]["b1"]["evaluation"]["top_k_membership"]["8"])
        for route, record in zip(routes, ordered, strict=True)
    )
    fallback_containment = sum(
        route == "diversity_fallback"
        and bool(record["route"]["evaluation"]["true_subkey_in_candidate_set"])
        for route, record in zip(routes, ordered, strict=True)
    )
    return {
        "schema": SUMMARY_SCHEMA,
        "phase": 6,
        "study": "phase6c_confidence_gated_pair_aware_fallback",
        "validation_only": True,
        "held_out_test_generated": False,
        "instance_count": len(ordered),
        "calibration": calibration_result,
        "routes": {
            "trusted_top8": {
                "selected_count": trusted_count,
                "coverage": trusted_count / len(routes),
                "containment": trusted_containment,
                "containment_rate": trusted_containment / trusted_count
                if trusted_count
                else None,
                "false_accepts": false_accepts,
            },
            "diversity_fallback": {
                "selected_count": fallback_count,
                "coverage": fallback_count / len(routes),
                "containment": fallback_containment,
                "containment_rate": fallback_containment / fallback_count
                if fallback_count
                else None,
                "false_abstentions": false_abstentions,
            },
            "false_accepts": false_accepts,
            "false_abstentions": false_abstentions,
        },
        "comparators": {
            "b1_top32": b1_top32_stats,
            "b0_top32": b0_top32_stats,
            "diversity_fallback_all_rows": fallback_stats,
            "fallback_vs_b1_top32_win_tie_loss": _win_tie_loss(
                fallback_capped,
                b1_capped,
            ),
            "paired_capped_attempt_mean_difference_fallback_minus_b1_top32": sum(differences) / len(differences),
            "paired_bootstrap_95_percent_interval": list(
                bootstrap_mean_interval(
                    differences,
                    replicates=int(config["metrics"]["bootstrap_replicates"]),
                    seed=int(config["metrics"]["bootstrap_seed"]),
                    confidence=float(config["metrics"]["bootstrap_confidence"]),
                )
            ),
        },
        "route_selected_stats": {
            "trusted_top8": _stats(
                trusted_evaluations,
                budget=PHASE6C_TRUSTED_BUDGET,
                top_k_values=top_k_values,
            ),
            "diversity_fallback": _stats(
                selected_fallback_evaluations,
                budget=PHASE6C_FALLBACK_BUDGET,
                top_k_values=top_k_values,
            ),
        },
        "execution_order_counts": {
            "b0_then_b1": sum(tuple(record["instance"]["execution_order"]) == ("b0", "b1") for record in ordered),
            "b1_then_b0": sum(tuple(record["instance"]["execution_order"]) == ("b1", "b0") for record in ordered),
        },
        "timing_seconds": {
            "mean_b0_ranking": sum(float(item["b0_ranking_seconds"]) for item in timings) / len(timings),
            "mean_b1_ranking": sum(float(item["b1_ranking_seconds"]) for item in timings) / len(timings),
            "mean_confidence": sum(float(item["confidence_seconds"]) for item in timings) / len(timings),
            "mean_route_online": sum(float(item["route_online_seconds"]) for item in timings) / len(timings),
            "mean_total": sum(float(item["total_seconds"]) for item in timings) / len(timings),
            "sat_status": "not_run",
            "verification_status": "not_run",
            "peak_rss_bytes_mean": sum(peak_rss) / len(peak_rss),
            "peak_rss_bytes_max": max(peak_rss),
        },
        "gate": {
            "threshold_selection": calibration_selection,
            "selected_threshold_is_frozen": True,
            "auto_rescue_false_accepts": False,
        },
        "provenance": {
            "config_path": _relative_or_absolute(config_path),
            "config_sha256": _sha256_file(config_path),
            "manifest_path": _relative_or_absolute(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "repository_commit": _repository_commit(),
            "held_out_accessed": False,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen Phase 6C-A protocol.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--calibrate", action="store_true", help="read-only calibration dry-run")
    parser.add_argument("--generate", action="store_true", help="generate the reviewed validation shard")
    parser.add_argument("--run", action="store_true", help="rank the existing validation shard")
    parser.add_argument("--smoke", action="store_true", help="rank one existing validation instance")
    parser.add_argument("--determinism-check", action="store_true", help="compare a smoke rerun")
    parser.add_argument("--first-instance", type=int, default=0)
    parser.add_argument("--last-instance", type=int, default=None)
    parser.add_argument("--rerun", action="store_true", help="ignore matching saved records")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = _repository_path(args.config).resolve()
    config = _load_json(config_path)
    _validate_config(config)
    calibration_result, calibrator, threshold, _ = _fit_calibration(
        config,
        config_path=config_path,
    )
    if args.calibrate or not (args.generate or args.run or args.smoke):
        print(json.dumps(calibration_result, indent=2, sort_keys=True), flush=True)
        return
    if not threshold.feasible or threshold.threshold is None:
        raise SystemExit(
            "frozen Phase 6C gate selection failed; no fresh data or ranking run is permitted"
        )
    if args.generate and not args.smoke:
        raise SystemExit("--generate must be combined with the one-instance smoke checkpoint")
    if args.determinism_check and not args.smoke:
        raise SystemExit("--determinism-check requires --smoke")
    runtime = _require_mapping(config["runtime"], "runtime")
    torch.set_num_threads(int(runtime["torch_num_threads"]))
    torch.set_num_interop_threads(int(runtime["torch_num_interop_threads"]))
    manifest_path = _write_or_verify_dataset(config, generate=args.generate)
    dataset = _require_mapping(config["dataset"], "dataset")
    attack_set = load_attack_shard(
        manifest_path.parent / "validation.json",
        expected_dataset_id=str(dataset["dataset_id"]),
    )
    if len(attack_set.instances) != 64 or any(len(item.pairs) != 64 for item in attack_set.instances):
        raise SystemExit("Phase 6C validation shard does not contain 64x64 data")
    b0_model, b1_model, model_metadata = _load_models(config)
    first_instance = args.first_instance
    last_instance = first_instance if args.smoke else (63 if args.last_instance is None else args.last_instance)
    if not 0 <= first_instance <= last_instance < 64:
        raise SystemExit("Phase 6C instance range is outside 0..63")
    output_root = _repository_path(str(config["output_root"])).resolve() / "eager"
    if not args.smoke:
        _require_preflight(
            config,
            config_path=config_path,
            manifest_path=manifest_path,
        )
    if args.smoke:
        output_root = output_root / "smoke"
    records: list[dict[str, Any]] = []
    for index in range(first_instance, last_instance + 1):
        output_path = output_root / f"ranking_{index:04d}.json"
        if output_path.is_file() and not args.rerun and not args.determinism_check:
            record = _load_json(output_path)
            _validate_saved_result(record, index=index, config_path=config_path, manifest_path=manifest_path)
            records.append(record)
            print(f"skip [{index + 1}/{last_instance + 1}] {output_path}", flush=True)
            continue
        instance = attack_set.instances[index]
        b0_ranking, b1_ranking, execution_order = _rank_complete_methods(
            instance,
            b0_model,
            b1_model,
            config=config,
            index=index,
        )
        record = _build_record(
            instance,
            b0_ranking,
            b1_ranking,
            config=config,
            config_path=config_path,
            manifest_path=manifest_path,
            model_metadata=model_metadata,
            calibration=calibrator,
            threshold=threshold,
            index=index,
            execution_order=execution_order,
        )
        _write_atomic(output_path, record)
        records.append(record)
        print(f"ranked [{index + 1}/{last_instance + 1}] {instance.instance_id}", flush=True)
        if args.determinism_check:
            second_b0, second_b1, second_order = _rank_complete_methods(
                instance,
                b0_model,
                b1_model,
                config=config,
                index=index,
            )
            second_record = _build_record(
                instance,
                second_b0,
                second_b1,
                config=config,
                config_path=config_path,
                manifest_path=manifest_path,
                model_metadata=model_metadata,
                calibration=calibrator,
                threshold=threshold,
                index=index,
                execution_order=second_order,
            )
            if _record_determinism_projection(record) != _record_determinism_projection(second_record):
                raise SystemExit("Phase 6C deterministic rerun failed")
            print(f"determinism check passed for {instance.instance_id}", flush=True)
            _write_preflight(
                config,
                config_path=config_path,
                manifest_path=manifest_path,
                record=record,
            )
    if not args.smoke and len(records) == 64:
        summary = _summary(
            records,
            config=config,
            config_path=config_path,
            manifest_path=manifest_path,
            calibration_result=calibration_result,
        )
        _write_atomic(output_root / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
