"""Execute the frozen Phase 7 held-out comparative evaluation across all 5 methods."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neural_sat_speck.attacks.diversity_fallback import (
    choose_phase6c_route,
    compose_diversity_fallback,
)
from neural_sat_speck.attacks.group_ranking import (
    evaluate_pair_group_subkey_ranking,
    rank_pair_group_subkeys,
)
from neural_sat_speck.attacks.ranking import (
    evaluate_subkey_ranking,
    rank_final_round_subkeys,
)
from neural_sat_speck.attacks.recovery import (
    independent_verification_pairs,
    known_pairs_for_attack_instance,
    random_candidate_order,
    run_candidate_recovery,
    run_sat_only_recovery,
)
from neural_sat_speck.confidence.calibration import (
    PlattCalibrator,
    top8_score_separation,
)
from neural_sat_speck.data.attack import (
    AttackInstance,
    AttackInstanceConfig,
    generate_attack_instances,
    load_attack_shard,
    write_attack_shard,
)
from neural_sat_speck.evaluation.metrics import (
    calculate_global_par2,
    exact_mcnemar_test,
    wilson_score_interval,
)
from neural_sat_speck.neural.group_model import PairGroupRanker
from neural_sat_speck.neural.model import GohrResNet
from neural_sat_speck.neural.numpy_inference import (
    NumpyInferenceModule,
    NumpyRankerInference,
)
from neural_sat_speck.sat.isolation import IsolatedSolverFactory

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "configs"
    / "experiments"
    / "phase7_final_evaluation.json"
)
CONFIG_SCHEMA = "neural_sat_speck.phase7_final_evaluation_config.v1"
MANIFEST_SCHEMA = "neural_sat_speck.phase7_final_evaluation_manifest.v1"
RESULT_SCHEMA = "neural_sat_speck.phase7_final_evaluation_result.v1"
SUMMARY_SCHEMA = "neural_sat_speck.phase7_final_evaluation_summary.v1"
METHODS = ("sat_only", "random_sat", "b0_sat", "b1_sat", "gated_b1_fallback_sat")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as err:
        raise SystemExit(f"could not read JSON file {path}: {err}") from err
    if not isinstance(data, dict):
        raise SystemExit(f"JSON file {path} must contain an object")
    return data


def _repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
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


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SystemExit(f"configuration section {name!r} must be an object")
    return value


def _write_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, sort_keys=True) + "\n"
    temp_dir = path.parent
    with tempfile.NamedTemporaryFile(
        "w",
        dir=temp_dir,
        delete=False,
        encoding="utf-8",
        prefix=f".{path.stem}_",
        suffix=".pending",
    ) as temp_file:
        temp_file.write(payload)
        temp_file.flush()
        os.fsync(temp_file.fileno())
        temp_path = Path(temp_file.name)
    temp_path.replace(path)


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise SystemExit(f"config schema must be {CONFIG_SCHEMA}")
    if config.get("phase") != 7:
        raise SystemExit("config phase must be 7")
    dataset = _require_mapping(config.get("dataset"), "dataset")
    if dataset.get("split_role") != "test":
        raise SystemExit("Phase 7 dataset split_role must be 'test'")
    seed = dataset.get("seed")
    if not isinstance(seed, int) or seed in config.get("seed_exclusions", []):
        raise SystemExit(f"Phase 7 seed {seed} collides with exclusion list")


def generate_phase7_dataset(config: Mapping[str, Any]) -> Path:
    """Generate the held-out test dataset for Phase 7."""
    _validate_config(config)
    dataset_cfg = _require_mapping(config["dataset"], "dataset")
    manifest_path = _repository_path(str(dataset_cfg["manifest_path"]))
    output_dir = _repository_path(str(dataset_cfg["output_dir"]))

    if manifest_path.is_file():
        return manifest_path

    input_diff = tuple(dataset_cfg["input_difference"])
    instance_config = AttackInstanceConfig(
        instance_count=int(dataset_cfg["instance_count"]),
        pair_count=int(dataset_cfg["pair_count"]),
        rounds=int(dataset_cfg["rounds"]),
        seed=int(dataset_cfg["seed"]),
        split_role="test",
        input_difference=(int(input_diff[0]), int(input_diff[1])),
    )
    collection = generate_attack_instances(
        instance_config,
        generated_at=str(dataset_cfg["generated_at"]),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    test_json = output_dir / "test.json"
    write_attack_shard(collection, test_json)

    manifest = {
        "dataset_id": dataset_cfg["dataset_id"],
        "schema": MANIFEST_SCHEMA,
        "phase": 7,
        "validation_only": False,
        "held_out_test_generated": True,
        "rounds": instance_config.rounds,
        "pair_count": instance_config.pair_count,
        "instance_count": instance_config.instance_count,
        "seed": instance_config.seed,
        "input_difference": list(instance_config.input_difference),
        "splits": {
            "test": {
                "split_role": "test",
                "instance_count": instance_config.instance_count,
                "pair_count": instance_config.pair_count,
                "seed": instance_config.seed,
                "path": "test.json",
                "file_sha256": _sha256_file(test_json),
            }
        },
    }
    _write_atomic(manifest_path, manifest)
    print(f"Generated Phase 7 held-out test dataset in {output_dir}")
    return manifest_path


def _counterbalanced_methods_order(index: int) -> tuple[str, ...]:
    shift = index % len(METHODS)
    return tuple(METHODS[shift:] + METHODS[:shift])


def evaluate_single_instance(
    instance: AttackInstance,
    index: int,
    b0_infer: NumpyInferenceModule,
    b1_infer: NumpyInferenceModule,
    calibrator: PlattCalibrator,
    config: Mapping[str, Any],
    manifest_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    """Evaluate all 5 attack methods on a single held-out test key instance."""
    solver_cfg = _require_mapping(config["solver"], "solver")
    conf_cfg = _require_mapping(config["confidence_and_fallback"], "confidence_and_fallback")
    timeout = float(solver_cfg["solver_timeout_seconds"])
    rounds = int(config["dataset"]["rounds"])

    # 1. Prepare known pairs (first 4 pairs) and independent verification pairs
    all_pairs = known_pairs_for_attack_instance(instance)
    known_pairs = all_pairs[: int(solver_cfg["known_pairs_used"])]
    verif_seed = int(config["dataset"]["seed"]) + 10000 + index
    verif_pairs = independent_verification_pairs(
        instance.master_key,
        rounds=rounds,
        pair_count=int(solver_cfg["independent_verification_pairs_count"]),
        seed=verif_seed,
    )

    # 2. Build SAT Model Isolation Factory
    factory = IsolatedSolverFactory.from_pairs(
        known_pairs, rounds=rounds, instance_id=instance.instance_id
    )

    # 3. Check ranking cache to avoid redundant neural computation
    ranking_cache_dir = REPOSITORY_ROOT / "results" / "phase7_final_evaluation" / "ranking_cache"
    ranking_cache_file = ranking_cache_dir / f"instance_{index:04d}.json"

    if ranking_cache_file.is_file():
        cached = _load_json(ranking_cache_file)
        b0_order = tuple(cached["b0_order"])
        t_rank_b0 = float(cached["t_rank_b0"])
        b0_eval = cached["b0_eval"]
        b1_order = tuple(cached["b1_order"])
        t_rank_b1 = float(cached["t_rank_b1"])
        b1_eval = cached["b1_eval"]
        conf_feature = float(cached["conf_feature"])
        t_conf = float(cached["t_conf"])
        prob = calibrator.predict(conf_feature)
        route = choose_phase6c_route(prob, float(conf_cfg["decision_threshold"] or 0.5))
        if route == "trusted_top8":
            gated_order = b1_order[: int(conf_cfg["trusted_budget"])]
            gated_budget = int(conf_cfg["trusted_budget"])
            t_gated_rank = t_rank_b1
        else:
            gated_order = compose_diversity_fallback(b1_order, b0_order)
            gated_budget = int(conf_cfg["fallback_budget"])
            t_gated_rank = t_rank_b1 + t_rank_b0
    else:
        # Pre-run neural rankings for B0 and B1
        b0_ranking = rank_final_round_subkeys(instance, b0_infer, candidate_batch_size=1024)
        t_rank_b0 = b0_ranking.total_seconds
        b0_order = b0_ranking.candidate_order
        b0_eval = evaluate_subkey_ranking(b0_ranking, instance.final_round_subkey)

        b1_ranking = rank_pair_group_subkeys(instance, b1_infer, candidate_batch_size=1024)
        t_rank_b1 = b1_ranking.total_seconds
        b1_order = b1_ranking.candidate_order
        b1_eval = evaluate_pair_group_subkey_ranking(b1_ranking, instance.final_round_subkey)

        t0 = time.perf_counter()
        b1_dict = {
            "ranking": {
                "candidate_values": list(b1_ranking.candidate_values),
                "candidate_scores": list(b1_ranking.candidate_scores),
                "candidate_order": list(b1_ranking.candidate_order),
            }
        }
        conf_feature = top8_score_separation(b1_dict)
        prob = calibrator.predict(conf_feature)
        t_conf = time.perf_counter() - t0

        route = choose_phase6c_route(prob, float(conf_cfg["decision_threshold"] or 0.5))
        if route == "trusted_top8":
            gated_order = b1_order[: int(conf_cfg["trusted_budget"])]
            gated_budget = int(conf_cfg["trusted_budget"])
            t_gated_rank = t_rank_b1
        else:
            gated_order = compose_diversity_fallback(b1_order, b0_order)
            gated_budget = int(conf_cfg["fallback_budget"])
            t_gated_rank = t_rank_b1 + t_rank_b0

    # Method order for counterbalancing
    method_order = _counterbalanced_methods_order(index)
    method_results: dict[str, Any] = {}
    active_models: dict[str, Any] = {}

    for method in method_order:
        method_model = factory.build_isolated_model(method)
        active_models[method] = method_model
        solver_obj_id = id(method_model.solver)

        if method == "sat_only":
            rec = run_sat_only_recovery(
                method_model,
                known_pairs,
                verif_pairs,
                solver_timeout=timeout,
                strategy="sat_only",
            )
            t_online = rec.solver_seconds + rec.verification_seconds
            method_results["sat_only"] = {
                "success": rec.success,
                "status": rec.status,
                "candidates_attempted": 0,
                "winning_candidate": rec.winning_candidate,
                "t_rank": 0.0,
                "t_conf": 0.0,
                "t_sat": rec.solver_seconds,
                "t_ver": rec.verification_seconds,
                "t_online": t_online,
                "solver_object_id": solver_obj_id,
            }

        elif method == "random_sat":
            rnd_order = random_candidate_order(
                seed=int(config["dataset"]["seed"]) + 20000 + index,
            )
            rec = run_candidate_recovery(
                method_model,
                rnd_order,
                known_pairs,
                verif_pairs,
                candidate_budget=16,
                solver_timeout=timeout,
                strategy="random_sat",
            )
            t_online = rec.solver_seconds + rec.verification_seconds
            method_results["random_sat"] = {
                "success": rec.success,
                "status": rec.status,
                "candidates_attempted": rec.candidates_attempted,
                "winning_candidate": rec.winning_candidate,
                "winning_candidate_position": rec.winning_candidate_position,
                "t_rank": 0.0,
                "t_conf": 0.0,
                "t_sat": rec.solver_seconds,
                "t_ver": rec.verification_seconds,
                "t_online": t_online,
                "solver_object_id": solver_obj_id,
            }

        elif method == "b0_sat":
            rec = run_candidate_recovery(
                method_model,
                b0_order,
                known_pairs,
                verif_pairs,
                candidate_budget=16,
                solver_timeout=timeout,
                strategy="b0_sat",
            )
            t_online = t_rank_b0 + rec.solver_seconds + rec.verification_seconds
            method_results["b0_sat"] = {
                "success": rec.success,
                "status": rec.status,
                "exact_true_rank": b0_eval["true_rank"],
                "candidates_attempted": rec.candidates_attempted,
                "winning_candidate": rec.winning_candidate,
                "winning_candidate_position": rec.winning_candidate_position,
                "t_rank": t_rank_b0,
                "t_conf": 0.0,
                "t_sat": rec.solver_seconds,
                "t_ver": rec.verification_seconds,
                "t_online": t_online,
                "solver_object_id": solver_obj_id,
            }

        elif method == "b1_sat":
            rec = run_candidate_recovery(
                method_model,
                b1_order,
                known_pairs,
                verif_pairs,
                candidate_budget=16,
                solver_timeout=timeout,
                strategy="b1_sat",
            )
            t_online = t_rank_b1 + rec.solver_seconds + rec.verification_seconds
            method_results["b1_sat"] = {
                "success": rec.success,
                "status": rec.status,
                "exact_true_rank": b1_eval["true_rank"],
                "candidates_attempted": rec.candidates_attempted,
                "winning_candidate": rec.winning_candidate,
                "winning_candidate_position": rec.winning_candidate_position,
                "t_rank": t_rank_b1,
                "t_conf": 0.0,
                "t_sat": rec.solver_seconds,
                "t_ver": rec.verification_seconds,
                "t_online": t_online,
                "solver_object_id": solver_obj_id,
            }

        elif method == "gated_b1_fallback_sat":
            rec = run_candidate_recovery(
                method_model,
                gated_order,
                known_pairs,
                verif_pairs,
                candidate_budget=gated_budget,
                solver_timeout=timeout,
                strategy="gated_b1_fallback_sat",
            )
            t_online = t_gated_rank + t_conf + rec.solver_seconds + rec.verification_seconds
            method_results["gated_b1_fallback_sat"] = {
                "success": rec.success,
                "status": rec.status,
                "route_taken": route,
                "confidence_probability": prob,
                "confidence_feature": conf_feature,
                "candidates_attempted": rec.candidates_attempted,
                "winning_candidate": rec.winning_candidate,
                "winning_candidate_position": rec.winning_candidate_position,
                "t_rank": t_gated_rank,
                "t_conf": t_conf,
                "t_sat": rec.solver_seconds,
                "t_ver": rec.verification_seconds,
                "t_online": t_online,
                "solver_object_id": solver_obj_id,
            }

    # Oracle True Subkey Diagnostic (Fresh Model)
    oracle_model = factory.build_isolated_model("oracle_true_subkey")
    active_models["oracle_true_subkey"] = oracle_model
    rec_oracle = run_candidate_recovery(
        oracle_model,
        [instance.final_round_subkey],
        known_pairs,
        verif_pairs,
        candidate_budget=1,
        solver_timeout=timeout,
        strategy="oracle_true_subkey",
    )
    diagnostics = {
        "oracle_true_subkey": {
            "success": rec_oracle.success,
            "status": rec_oracle.status,
            "solver_seconds": rec_oracle.solver_seconds,
            "verification_seconds": rec_oracle.verification_seconds,
            "total_seconds": rec_oracle.solver_seconds + rec_oracle.verification_seconds,
            "solver_object_id": id(oracle_model.solver),
        }
    }

    # Verify complete object isolation across all allocated models
    factory.verify_isolation(active_models)

    return {
        "schema": RESULT_SCHEMA,
        "instance_id": instance.instance_id,
        "instance_index": index,
        "master_key": list(instance.master_key),
        "true_final_round_subkey": instance.final_round_subkey,
        "execution_order": list(method_order),
        "provenance": {
            "config_sha256": config_sha256,
            "manifest_sha256": manifest_sha256,
            "git_commit": _git_commit(),
            "platform": platform.platform(),
            "python_version": sys.version,
        },
        "methods": method_results,
        "diagnostics": diagnostics,
        "isolation_verified": True,
    }


def run_evaluation(
    config_path: Path = DEFAULT_CONFIG,
    smoke: bool = False,
) -> Path:
    config = _load_json(config_path)
    _validate_config(config)
    config_sha256 = _sha256_file(config_path)

    # 1. Ensure dataset exists
    manifest_path = generate_phase7_dataset(config)
    manifest_sha256 = _sha256_file(manifest_path)
    dataset_dir = manifest_path.parent
    collection = load_attack_shard(dataset_dir / "test.json")

    # 2. Load models
    models_cfg = config["models"]
    b0_path = _repository_path(models_cfg["b0"]["checkpoint_path"])
    b1_path = _repository_path(models_cfg["b1"]["checkpoint_path"])

    print("Loading neural models (B0 and B1)...")
    b0_torch = GohrResNet()
    b0_torch.load_state_dict(torch.load(b0_path, weights_only=False)["state_dict"])
    b0_torch.eval()
    b0_infer = NumpyInferenceModule(NumpyRankerInference(b0_torch))

    b1_data = torch.load(b1_path, weights_only=False)
    b1_torch = PairGroupRanker(variant=b1_data["variant"])
    b1_torch.load_state_dict(b1_data["state_dict"])
    b1_torch.eval()
    b1_infer = NumpyInferenceModule(NumpyRankerInference(b1_torch))

    # 3. Setup calibrator
    cal_cfg = config["confidence_and_fallback"]["calibrator"]
    calibrator = PlattCalibrator(
        feature_mean=float(cal_cfg["feature_mean"]),
        feature_std=float(cal_cfg["feature_std"]),
        intercept=float(cal_cfg["intercept"]),
        slope=float(cal_cfg["slope"]),
        l2_lambda=1.0,
        fit_count=64,
        positive_count=41,
        iterations=10,
        gradient_tolerance=1e-10,
        step_tolerance=1e-12,
    )

    records_dir = _repository_path(config["outputs"]["records_dir"])
    records_dir.mkdir(parents=True, exist_ok=True)
    summary_path = _repository_path(config["outputs"]["summary_path"])

    instances = collection.instances[:2] if smoke else collection.instances
    total = len(instances)
    print(f"Starting Phase 7 evaluation across {total} test instances...")

    for index, instance in enumerate(instances):
        record_file = records_dir / f"instance_{index:04d}.json"
        if record_file.is_file():
            print(f"[{index+1}/{total}] Skipping already completed {instance.instance_id}")
            continue

        print(f"[{index+1}/{total}] Evaluating {instance.instance_id} across all 5 methods...")
        t_start = time.perf_counter()
        result = evaluate_single_instance(
            instance,
            index,
            b0_infer,
            b1_infer,
            calibrator,
            config,
            manifest_sha256,
            config_sha256,
        )
        _write_atomic(record_file, result)
        gc.collect()
        t_elapsed = time.perf_counter() - t_start
        print(f"  ✓ Finished {instance.instance_id} in {t_elapsed:.2f}s")

    # 4. Generate summary
    print("Generating Phase 7 final summary...")
    summary = summarize_records(records_dir, config, manifest_sha256, config_sha256)
    _write_atomic(summary_path, summary)
    print(f"Summary written to {summary_path}")
    return summary_path


def summarize_records(
    records_dir: Path,
    config: Mapping[str, Any],
    manifest_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    records = []
    for path in sorted(records_dir.glob("instance_*.json")):
        records.append(_load_json(path))

    count = len(records)
    summary_methods: dict[str, Any] = {}

    for method in METHODS:
        success_list = [bool(r["methods"][method]["success"]) for r in records]
        successes = sum(1 for s in success_list if s)
        t_onlines = [float(r["methods"][method]["t_online"]) for r in records]
        attempts = [int(r["methods"][method]["candidates_attempted"]) for r in records]

        # Wilson 95% confidence interval
        ci_lower, ci_upper = wilson_score_interval(successes, count, confidence=0.95)

        # Global PAR-2 under explicit cutoffs (SAT Competition Standard)
        par2_80 = calculate_global_par2(t_onlines, success_list, global_cutoff=80.0)
        par2_100 = calculate_global_par2(t_onlines, success_list, global_cutoff=100.0)
        par2_160 = calculate_global_par2(t_onlines, success_list, global_cutoff=160.0)
        par2_200 = calculate_global_par2(t_onlines, success_list, global_cutoff=200.0)
        par2_300 = calculate_global_par2(t_onlines, success_list, global_cutoff=300.0)

        # Historical local PAR-2 (2x local solver timeout) preserved for reference
        timeout = float(config["solver"]["solver_timeout_seconds"])
        par2_local = [
            t if succ else 2 * timeout
            for t, succ in zip(t_onlines, success_list, strict=True)
        ]

        summary_methods[method] = {
            "success_count": successes,
            "success_rate": successes / count if count > 0 else 0.0,
            "wilson_ci_95": [ci_lower, ci_upper],
            "mean_online_seconds": float(np.mean(t_onlines)) if t_onlines else 0.0,
            "median_online_seconds": float(np.median(t_onlines)) if t_onlines else 0.0,
            "mean_candidates_attempted": float(np.mean(attempts)) if attempts else 0.0,
            "median_candidates_attempted": float(np.median(attempts)) if attempts else 0.0,
            "par2_seconds": par2_200,
            "par2_global_200s": par2_200,
            "par2_by_cutoff": {
                "80s": par2_80,
                "100s": par2_100,
                "160s": par2_160,
                "200s": par2_200,
                "300s": par2_300,
            },
            "par2_local_timeout_seconds": float(np.mean(par2_local)) if par2_local else 0.0,
        }

    # McNemar tests
    b1_succ = [r["methods"]["b1_sat"]["success"] for r in records]
    b0_succ = [r["methods"]["b0_sat"]["success"] for r in records]
    gated_succ = [r["methods"]["gated_b1_fallback_sat"]["success"] for r in records]
    sat_succ = [r["methods"]["sat_only"]["success"] for r in records]

    def _discordant(s1, s2):
        b = sum(1 for a, b_ in zip(s1, s2, strict=True) if a and not b_)
        c = sum(1 for a, b_ in zip(s1, s2, strict=True) if not a and b_)
        return exact_mcnemar_test(b, c)

    mcnemar_tests = {
        "b1_vs_b0": _discordant(b1_succ, b0_succ),
        "gated_vs_b1": _discordant(gated_succ, b1_succ),
        "b1_vs_sat_only": _discordant(b1_succ, sat_succ),
        "b0_vs_sat_only": _discordant(b0_succ, sat_succ),
    }

    # Oracle True Subkey Diagnostic Summary
    oracle_times = [
        r.get("diagnostics", {}).get("oracle_true_subkey", {}).get("total_seconds", 0.0)
        for r in records
        if "diagnostics" in r and "oracle_true_subkey" in r["diagnostics"]
    ]
    oracle_summary = {}
    if oracle_times:
        oracle_summary = {
            "mean_seconds": float(np.mean(oracle_times)),
            "median_seconds": float(np.median(oracle_times)),
            "min_seconds": float(np.min(oracle_times)),
            "max_seconds": float(np.max(oracle_times)),
            "count_under_100ms": sum(1 for t in oracle_times if t < 0.1),
        }

    return {
        "schema": SUMMARY_SCHEMA,
        "phase": 7,
        "study": "phase7_held_out_comparative_evaluation",
        "instance_count": count,
        "status_notice": (
            "CORRECTED / AUDITED — Every method evaluated under strict CryptoMiniSat solver isolation. "
            "Historical sat_only result (12/64 = 18.75%) was contaminated by shared solver state."
        ),
        "provenance": {
            "config_sha256": config_sha256,
            "manifest_sha256": manifest_sha256,
            "git_commit": _git_commit(),
        },
        "methods": summary_methods,
        "mcnemar_paired_tests": mcnemar_tests,
        "oracle_diagnostic": oracle_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7 Held-Out Evaluation Runner")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--generate-dataset", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else REPOSITORY_ROOT / args.config
    config = _load_json(config_path)

    if args.generate_dataset:
        generate_phase7_dataset(config)
        return

    run_evaluation(config_path, smoke=args.smoke)


if __name__ == "__main__":
    main()
