"""Run the bounded Phase 5 SAT-only, random, and neural-order comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from neural_sat_speck.attacks.recovery import (
    independent_verification_pairs,
    known_pairs_for_attack_instance,
    random_candidate_order,
    run_candidate_recovery,
    run_sat_only_recovery,
)
from neural_sat_speck.data.attack import (
    ATTACK_SPLIT_ROLES,
    load_attack_shard,
    verify_attack_benchmark,
)
from neural_sat_speck.sat.model import SpeckSatModel

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT / "configs" / "experiments" / "phase5_integration_r6.json"
)
METHODS = ("sat_only", "random", "neural")
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


def _sha256_candidate_order(candidate_order: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    for candidate in candidate_order:
        digest.update(candidate.to_bytes(2, "big"))
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


def _require_positive_int(config: dict[str, Any], field: str) -> int:
    value = _require_int(config, field)
    if value <= 0:
        raise SystemExit(f"configuration field {field!r} must be positive")
    return value


def _require_positive_number(config: dict[str, Any], field: str) -> float:
    value = config.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"configuration field {field!r} must be a number")
    if value <= 0:
        raise SystemExit(f"configuration field {field!r} must be positive")
    return float(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare SAT-only, seeded random-order, and Phase 4 neural-order "
            "candidate recovery for reduced-round SPECK32/64."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=(
            "Phase 5 JSON configuration "
            f"(default: {DEFAULT_CONFIG.relative_to(REPOSITORY_ROOT)})"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the JSON result path from the configuration.",
    )
    parser.add_argument(
        "--instance-index",
        type=int,
        default=None,
        help="Override the zero-based validation/attack instance index.",
    )
    parser.add_argument(
        "--ranking-result",
        type=Path,
        default=None,
        help="Override the complete Phase 4 ranking result path.",
    )
    parser.add_argument(
        "--split-role",
        choices=ATTACK_SPLIT_ROLES,
        default=None,
        help="Override the configured attack split role.",
    )
    return parser.parse_args()


def _validated_neural_order(
    path: Path,
    *,
    expected_dataset_id: str,
    split_role: str,
    instance_index: int,
) -> tuple[tuple[int, ...], float, dict[str, Any]]:
    document = _load_json(path)
    if document.get("schema") != "neural_sat_speck.phase4_ranking_result.v1":
        raise SystemExit("ranking result schema is not recognised")
    dataset = document.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("dataset_id") != expected_dataset_id:
        raise SystemExit("ranking result dataset does not match Phase 5")
    instance = document.get("instance")
    if not isinstance(instance, dict):
        raise SystemExit("ranking result instance metadata is missing")
    if instance.get("split_role") != split_role or instance.get("index") != instance_index:
        raise SystemExit("ranking result instance does not match Phase 5")
    domain = document.get("candidate_domain")
    if not isinstance(domain, dict):
        raise SystemExit("ranking result candidate domain is missing")
    if (
        domain.get("first") != 0
        or domain.get("last") != 0xFFFF
        or domain.get("count") != FULL_CANDIDATE_COUNT
        or domain.get("complete_16_bit_domain") is not True
    ):
        raise SystemExit("neural ranking must cover the complete 16-bit domain")
    ranking = document.get("ranking")
    if not isinstance(ranking, dict):
        raise SystemExit("ranking result ranking payload is missing")
    raw_order = ranking.get("candidate_order")
    if not isinstance(raw_order, list) or len(raw_order) != FULL_CANDIDATE_COUNT:
        raise SystemExit("ranking result must contain 65,536 ordered candidates")
    if any(
        isinstance(candidate, bool)
        or not isinstance(candidate, int)
        or not 0 <= candidate <= 0xFFFF
        for candidate in raw_order
    ) or set(raw_order) != set(range(FULL_CANDIDATE_COUNT)):
        raise SystemExit("ranking result candidate order is not a full permutation")
    timing = ranking.get("timing_seconds")
    if not isinstance(timing, dict):
        raise SystemExit("ranking result timing metadata is missing")
    total = timing.get("total")
    if isinstance(total, bool) or not isinstance(total, (int, float)) or total < 0:
        raise SystemExit("ranking result total timing is invalid")
    return tuple(raw_order), float(total), document


def _method_record(
    recovery: Any,
    *,
    model_build_seconds: float,
    candidate_order_seconds: float,
    ranking_seconds: float,
    candidate_order_source: str,
    candidate_order: tuple[int, ...],
    known_pair_count: int,
    verification_pair_count: int,
) -> dict[str, Any]:
    record = recovery.to_dict()
    recovery_timing = record["timing_seconds"]
    record["known_pair_count"] = known_pair_count
    record["known_observation_count"] = known_pair_count * 2
    record["verification_pair_count"] = verification_pair_count
    record["candidate_order_source"] = candidate_order_source
    record["candidate_order_count"] = len(candidate_order)
    record["candidate_order_sha256"] = _sha256_candidate_order(candidate_order)
    record["candidate_order_prefix"] = list(candidate_order[:16])
    record["timing_seconds"] = {
        "model_build": model_build_seconds,
        "candidate_order": candidate_order_seconds,
        "ranking_reused": ranking_seconds,
        "solver": recovery_timing["solver"],
        "verification": recovery_timing["verification"],
        "recovery_total": recovery_timing["total"],
        "online_total": (
            model_build_seconds
            + candidate_order_seconds
            + ranking_seconds
            + recovery_timing["total"]
        ),
    }
    record["ranking_reused_precomputation"] = ranking_seconds > 0.0
    return record


def main() -> None:
    args = _parse_args()
    config_path = _repository_path(args.config).resolve()
    config = _load_json(config_path)
    if config.get("schema") != "neural_sat_speck.phase5_integration_config.v1":
        raise SystemExit("integration config schema is not recognised")
    if config.get("phase") != 5:
        raise SystemExit("the selected configuration is not a Phase 5 config")

    split_role = args.split_role or config.get("split_role")
    if split_role not in ATTACK_SPLIT_ROLES:
        raise SystemExit("split_role must be train, validation, or test")
    instance_index = (
        args.instance_index
        if args.instance_index is not None
        else _require_int(config, "instance_index")
    )
    if instance_index < 0:
        raise SystemExit("instance_index must be non-negative")
    known_pair_count = _require_positive_int(config, "known_pair_count")
    candidate_budget = _require_positive_int(config, "candidate_budget")
    solver_timeout = _require_positive_number(config, "solver_timeout_seconds")
    max_solutions = _require_positive_int(config, "max_solutions_per_candidate")
    verification_pair_count = _require_positive_int(
        config,
        "verification_pair_count",
    )
    verification_seed = _require_int(config, "verification_seed")
    random_seed = _require_int(config, "random_seed")
    methods_value = config.get("methods")
    if methods_value != list(METHODS):
        raise SystemExit(f"methods must be exactly {list(METHODS)}")

    expected_difference_value = config.get("expected_input_difference")
    if (
        not isinstance(expected_difference_value, list)
        or len(expected_difference_value) != 2
        or any(
            isinstance(word, bool) or not isinstance(word, int)
            for word in expected_difference_value
        )
    ):
        raise SystemExit("expected_input_difference must contain two integers")
    expected_difference = tuple(expected_difference_value)

    attack_root = _repository_path(str(config["attack_dataset_root"])).resolve()
    manifest_path = attack_root / "manifest.json"
    expected_dataset_id = str(config["expected_attack_dataset_id"])
    attack_manifest = verify_attack_benchmark(
        manifest_path,
        expected_dataset_id=expected_dataset_id,
        expected_rounds=_require_int(config, "expected_rounds"),
        expected_input_difference=expected_difference,
    )
    split_record = attack_manifest["splits"][split_role]
    attack_set = load_attack_shard(
        attack_root / str(split_record["path"]),
        expected_dataset_id=expected_dataset_id,
    )
    if instance_index >= len(attack_set.instances):
        raise SystemExit(
            f"instance_index {instance_index} is outside the {split_role} split"
        )
    instance = attack_set.instances[instance_index]
    if known_pair_count > len(instance.pairs):
        raise SystemExit("known_pair_count exceeds the attack instance pair count")
    known_pairs = known_pairs_for_attack_instance(
        instance,
        pair_count=known_pair_count,
    )
    verification_pairs = independent_verification_pairs(
        instance.master_key,
        int(attack_manifest["rounds"]),
        verification_pair_count,
        verification_seed,
    )

    ranking_path = _repository_path(
        args.ranking_result
        if args.ranking_result is not None
        else str(config["ranking_result"])
    ).resolve()
    neural_order, ranking_seconds, ranking_document = _validated_neural_order(
        ranking_path,
        expected_dataset_id=expected_dataset_id,
        split_role=split_role,
        instance_index=instance_index,
    )
    output_path = _repository_path(
        args.output if args.output is not None else str(config["output"])
    ).resolve()
    print("=" * 72)
    print("SPECK32/64 INCREMENTAL NEURAL-SAT INTEGRATION (PHASE 5)")
    print(f"Config: {config_path}")
    print(f"Attack manifest: {manifest_path}")
    print(f"Ranking result: {ranking_path}")
    print(f"Split and instance: {split_role}[{instance_index}]")
    print(f"Known related pairs: {known_pair_count}")
    print(f"Candidate budget: {candidate_budget}")
    print(f"Solver timeout per call: {solver_timeout:.3f}s")
    print("=" * 72)

    method_records: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        method_started = time.perf_counter()
        if method == "sat_only":
            candidate_order: tuple[int, ...] = ()
            candidate_order_seconds = 0.0
            ranking_cost = 0.0
        elif method == "random":
            order_started = time.perf_counter()
            candidate_order = random_candidate_order(random_seed)
            candidate_order_seconds = time.perf_counter() - order_started
            ranking_cost = 0.0
        else:
            candidate_order = neural_order
            candidate_order_seconds = 0.0
            ranking_cost = ranking_seconds

        build_started = time.perf_counter()
        model = SpeckSatModel.from_known_pairs(
            known_pairs,
            rounds=int(attack_manifest["rounds"]),
        )
        model_build_seconds = time.perf_counter() - build_started

        if method == "sat_only":
            recovery = run_sat_only_recovery(
                model,
                known_pairs,
                verification_pairs,
                solver_timeout=solver_timeout,
            )
        else:
            recovery = run_candidate_recovery(
                model,
                candidate_order,
                known_pairs,
                verification_pairs,
                candidate_budget=candidate_budget,
                solver_timeout=solver_timeout,
                max_solutions_per_candidate=max_solutions,
                strategy=method,
            )
        record = _method_record(
            recovery,
            model_build_seconds=model_build_seconds,
            candidate_order_seconds=candidate_order_seconds,
            ranking_seconds=ranking_cost,
            candidate_order_source=(
                "none" if method == "sat_only" else f"{method}_full_domain"
            ),
            candidate_order=candidate_order,
            known_pair_count=known_pair_count,
            verification_pair_count=verification_pair_count,
        )
        record["model_variable_count"] = model.variable_count
        record["method_wall_seconds"] = time.perf_counter() - method_started
        method_records[method] = record
        print(
            f"{method:>8}: status={record['status']}, "
            f"success={record['success']}, "
            f"online={record['timing_seconds']['online_total']:.3f}s"
        )

    result: dict[str, Any] = {
        "schema": "neural_sat_speck.phase5_integration_result.v1",
        "phase": 5,
        "protocol": (
            "same exact reduced-round SPECK32/64 model under SAT-only, seeded "
            "random-order, and Phase 4 neural-order recovery"
        ),
        "cipher": attack_manifest["cipher"],
        "task": attack_manifest["task"],
        "dataset": {
            "dataset_id": attack_manifest["dataset_id"],
            "manifest_path": _relative_or_absolute(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "repository_commit": attack_manifest["repository_commit"],
            "generated_at": attack_manifest["generated_at"],
            "rounds": attack_manifest["rounds"],
            "input_difference": attack_manifest["input_difference"],
            "word_order": attack_manifest["word_order"],
        },
        "instance": {
            "split_role": split_role,
            "index": instance_index,
            "instance_id": instance.instance_id,
            "available_pair_count": len(instance.pairs),
            "known_pair_count": known_pair_count,
            "known_observation_count": len(known_pairs),
            "true_final_round_subkey": instance.final_round_subkey,
        },
        "verification": {
            "pair_count": verification_pair_count,
            "seed": verification_seed,
            "purpose": "fresh independent plaintext/ciphertext pairs",
        },
        "candidate_testing": {
            "candidate_budget": candidate_budget,
            "solver_timeout_seconds": solver_timeout,
            "max_solutions_per_candidate": max_solutions,
            "random_seed": random_seed,
            "candidate_domain_count": FULL_CANDIDATE_COUNT,
            "candidate_order_difference_is_only_method_difference": True,
        },
        "ranking_provenance": {
            "path": _relative_or_absolute(ranking_path),
            "sha256": _sha256_file(ranking_path),
            "schema": ranking_document["schema"],
            "reported_total_seconds": ranking_seconds,
            "precomputed_before_integration": True,
        },
        "methods": method_records,
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Result saved to: {output_path}")


if __name__ == "__main__":
    main()
