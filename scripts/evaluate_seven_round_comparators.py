#!/usr/bin/env python3
"""Evaluate seven-round recovery comparators under strict solver isolation.

This script executes a controlled comparative study of 7-round SPECK32/64 recovery
strategies on the opened development dataset (Seed 300001, N=64):
1. hierarchical_neural_screened: Neural 6-round stage 1 (M1=32), Neural 5-round stage 2 (M2=32)
   with multi-fidelity screening (16 pairs, 512 survivors).
2. hierarchical_neural_full: Neural 6-round stage 1 (M1=32), Neural 5-round stage 2 (M2=32)
   with full scoring on all 64 pairs.
3. hierarchical_neural_classical: Neural 6-round stage 1 (M1=32), Classical 5-round DDT stage 2 (M2=32).
   Aliased as hierarchical_classical.
4. random_sat: Random candidate order for (k6, k5) tuples under equivalent candidate budget (C=1024).
5. sat_only_4obs: Exact SAT solving on 7 rounds with 4 observations (2 differential pairs equivalent).
   Aliased as sat_only.
6. sat_only_all_observations: Exact SAT solving on 7 rounds with all 128 observations (64 differential pairs).

SCIENTIFIC STATUS:
- DEVELOPMENT/DIAGNOSTIC evidence only.
- Evaluated on opened Seed 300001; this dataset cannot be used for confirmation.
- Governed by AGENTS.md, PLAN.md, and D-027.
- Strict per-method solver isolation enforced via IsolatedSolverFactory.
- Monotonic per-instance global wall-clock deadline enforced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import torch  # noqa: E402

from neural_sat_speck.attacks.classical_ranking import (  # noqa: E402
    ClassicalDistributionTable,
)
from neural_sat_speck.attacks.hierarchical_multi_round import (  # noqa: E402
    HierarchicalStage,
    run_hierarchical_multi_round_attack,
)
from neural_sat_speck.attacks.multi_round_recovery import (  # noqa: E402
    run_multi_round_sat_only_recovery,
    run_random_multi_round_recovery,
)
from neural_sat_speck.attacks.recovery import (  # noqa: E402
    independent_verification_pairs,
    known_pairs_for_attack_instance,
)
from neural_sat_speck.data.attack import load_attack_shard  # noqa: E402
from neural_sat_speck.evaluation.metrics import wilson_score_interval  # noqa: E402
from neural_sat_speck.neural.group_model import PairGroupRanker  # noqa: E402
from neural_sat_speck.neural.numpy_inference import (  # noqa: E402
    NumpyInferenceModule,
    NumpyRankerInference,
)
from neural_sat_speck.protocol.manifest_validator import (  # noqa: E402
    ManifestValidationError,
    classify_attack_outcome,
    validate_manifest_for_development_execution,
    validate_manifest_for_prospective_execution,
    verify_attack_dataset_integrity,
)
from neural_sat_speck.sat.isolation import IsolatedSolverFactory  # noqa: E402

DEFAULT_DATASET_PATH = (
    REPOSITORY_ROOT / "datasets" / "phase11a_oracle_r7" / "oracle_evaluation.json"
)
DATASET_ID = "speck32_64_attack_phase11a_oracle_r7"
RESULTS_DIR = REPOSITORY_ROOT / "results" / "phase11_scaling_diagnostics"
ROUNDS = 7
RANKING_DIFFERENTIAL_PAIRS = 64
RANKING_TOTAL_OBSERVATIONS = 128
SAT_OBSERVATIONS_4OBS = 4
SAT_OBSERVATIONS_ALL = 128
VERIFICATION_PAIRS = 2
VERIFICATION_SEED = 999999
SCHEMA_VERSION = "2.0.0"

DEFAULT_STAGE1_PATH = (
    REPOSITORY_ROOT / "models" / "speck32_64_phase11b_prime_stage1_r6_b1_mean.pt"
)
DEFAULT_STAGE2_PATH = (
    REPOSITORY_ROOT / "models" / "speck32_64_phase6a_b1_mean_g64.pt"
)
DEFAULT_CLASSICAL_PATH = (
    REPOSITORY_ROOT / "models" / "classical_dist_r5_sparse_v2.json"
)

COMPARATOR_METHODS = (
    "hierarchical_neural_screened",
    "hierarchical_neural_full",
    "hierarchical_neural_classical",
    "random_sat",
    "sat_only_4obs",
    "sat_only_all_observations",
)

METHOD_ALIASES = {
    "hierarchical_classical": "hierarchical_neural_classical",
    "sat_only": "sat_only_4obs",
}

METHOD_CHOICES = list(COMPARATOR_METHODS) + list(METHOD_ALIASES.keys())

METHOD_DISPLAY_NAMES = {
    "hierarchical_neural_screened": "Neural Screened (M1=32, M2=32)",
    "hierarchical_neural_full": "Neural Full (M1=32, M2=32)",
    "hierarchical_neural_classical": "Neural-Classical Hybrid (M1=32, M2=32)",
    "random_sat": "Random Candidate SAT",
    "sat_only_4obs": "Exact SAT Only (4 obs, formula-matched)",
    "sat_only_all_observations": "Exact SAT Only (128 obs, info-matched)",
}

REQUIRED_METADATA_FIELDS = {
    "schema_version",
    "dataset_path",
    "dataset_sha256",
    "dataset_id",
    "stage1_model_path",
    "stage1_model_sha256",
    "stage2_model_path",
    "stage2_model_sha256",
    "classical_dist_path",
    "classical_dist_sha256",
    "rounds",
    "ranking_differential_pairs",
    "ranking_total_observations",
    "sat_observations_4obs",
    "sat_observations_all",
    "verification_pairs",
    "verification_seed",
    "m1",
    "m2",
    "candidate_budget",
    "solver_timeout",
    "global_timeout",
    "device",
    "methods",
}

REQUIRED_RECORD_FIELDS = {
    "instance_id",
    "instance_index",
    "method",
    "method_display_name",
    "strategy",
    "success",
    "status",
    "failure_cause",
    "candidates_attempted",
    "recovered_master_key",
    "timing_seconds",
    "wall_seconds",
    "method_wall_seconds",
    "cutoff_seconds",
    "cutoff_overshoot_seconds",
    "preprocessing_seconds",
    "ranking_screening_sorting_seconds",
    "solver_seconds",
    "verification_seconds",
    "counts_as_verified_recovery",
    "counts_as_unconditional_failure",
    "is_resolved_sat_state",
    "failure_domain",
}


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHOD_CHOICES,
        default=list(COMPARATOR_METHODS),
        help="Methods to evaluate (default: all 6 comparators)",
    )
    parser.add_argument(
        "--stage1-model",
        type=Path,
        default=DEFAULT_STAGE1_PATH,
        help="Stage 1 6-round ResNet checkpoint",
    )
    parser.add_argument(
        "--stage2-model",
        type=Path,
        default=DEFAULT_STAGE2_PATH,
        help="Stage 2 5-round ResNet checkpoint",
    )
    parser.add_argument(
        "--classical-dist",
        type=Path,
        default=DEFAULT_CLASSICAL_PATH,
        help="Stage 2 5-round classical distribution JSON",
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=16,
        help="Number of instances to evaluate",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index in the dataset",
    )
    parser.add_argument(
        "--m1",
        type=int,
        default=32,
        help="Stage 1 beam width (default: 32)",
    )
    parser.add_argument(
        "--m2",
        type=int,
        default=32,
        help="Stage 2 beam width (default: 32)",
    )
    parser.add_argument(
        "--candidate-budget",
        type=int,
        default=None,
        help="Maximum candidate budget for random multi-round recovery (default: m1 * m2)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Per-candidate SAT solver timeout in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--global-timeout",
        type=float,
        default=300.0,
        help="Per-instance global wall-clock deadline in seconds (default: 300.0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Candidate batch size for neural ranking",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="PyTorch device (e.g. mps, cpu, cuda)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing run, skipping completed instance-method pairs (requires --output)",
    )
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        default=None,
        help="Path to protocol manifest JSON (enforced strictly for --prospective)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to attack dataset JSON (required for --prospective; defaults to opened Seed 300001 dataset in development mode)",
    )
    parser.add_argument(
        "--prospective",
        action="store_true",
        help="Flag indicating prospective evaluation on protected data (requires FROZEN_AUTHORIZED)",
    )
    parser.add_argument(
        "--development-mode",
        action="store_true",
        help="Flag indicating development/diagnostic run on opened data",
    )
    args = parser.parse_args(argv)

    if args.prospective and args.development_mode:
        raise SystemExit("Error: Cannot specify both --prospective and --development-mode.")

    if args.prospective and args.protocol_manifest is None:
        raise SystemExit("Error: --prospective requires an explicit --protocol-manifest.")

    if args.prospective and args.dataset is None:
        raise SystemExit("Error: --prospective requires an explicit --dataset path.")

    if args.dataset is None:
        args.dataset = DEFAULT_DATASET_PATH

    # Normalize method aliases to canonical names while preserving user order
    normalized_methods: list[str] = []
    for m in args.methods:
        canonical = METHOD_ALIASES.get(m, m)
        if canonical not in normalized_methods:
            normalized_methods.append(canonical)
    args.methods = normalized_methods

    if args.resume and args.output is None:
        raise SystemExit("Error: --resume requires an explicit --output path.")

    return args


def load_pytorch_ranker(path: Path, device: torch.device) -> PairGroupRanker:
    if not path.exists():
        raise SystemExit(f"checkpoint not found at {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    variant = payload.get("variant", "b1_mean")
    model = PairGroupRanker(variant=variant)
    state = payload["state_dict"] if "state_dict" in payload else payload
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_numpy_ranker(path: Path) -> NumpyInferenceModule:
    if not path.exists():
        raise SystemExit(f"checkpoint not found at {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    variant = payload.get("variant", "b1_mean")
    model = PairGroupRanker(variant=variant)
    state = payload["state_dict"] if "state_dict" in payload else payload
    model.load_state_dict(state)
    model.eval()
    return NumpyInferenceModule(NumpyRankerInference(model))


def _save_atomic(data: dict[str, Any], target_path: Path) -> None:
    """Atomically save dictionary as JSON via a temporary file with fsync."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(f".tmp.{os.getpid()}")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def compute_method_summary(
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Compute aggregate statistics for a set of instance records."""
    if not records:
        return {}
    total = len(records)
    successes = sum(1 for r in records if r["success"])
    low, high = wilson_score_interval(successes, total)
    wall_times = [r["wall_seconds"] for r in records]
    median_wall = statistics.median(wall_times)
    method_wall_times = [
        r.get("method_wall_seconds", r["wall_seconds"]) for r in records
    ]
    median_method_wall = statistics.median(method_wall_times)
    mean_candidates = sum(r["candidates_attempted"] for r in records) / total
    overshoots = [r.get("cutoff_overshoot_seconds", 0.0) for r in records]
    max_overshoot = max(overshoots) if overshoots else 0.0
    mean_overshoot = sum(overshoots) / total if total > 0 else 0.0

    causes: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    for r in records:
        dom = r.get("failure_domain", "none" if r.get("success") else "unknown")
        domain_counts[dom] = domain_counts.get(dom, 0) + 1
        if not r["success"]:
            cause = r.get("failure_cause") or r.get("status") or "unknown"
            causes[cause] = causes.get(cause, 0) + 1

    return {
        "instances_evaluated": total,
        "verified_recoveries": successes,
        "verified_recovery_rate": round(successes / total, 4),
        "wilson_95_ci": [round(low, 4), round(high, 4)],
        "unconditional_failures": sum(1 for r in records if r.get("counts_as_unconditional_failure")),
        "resolved_sat_states": sum(1 for r in records if r.get("is_resolved_sat_state")),
        "failure_domains": domain_counts,
        "median_wall_seconds": round(median_wall, 3),
        "median_method_wall_seconds": round(median_method_wall, 3),
        "max_cutoff_overshoot_seconds": round(max_overshoot, 6),
        "mean_cutoff_overshoot_seconds": round(mean_overshoot, 6),
        "mean_candidates_attempted": round(mean_candidates, 1),
        "failure_causes": causes,
        "total_wall_seconds": round(sum(r["wall_seconds"] for r in records), 3),
        "total_method_wall_seconds": round(
            sum(r.get("method_wall_seconds", r["wall_seconds"]) for r in records),
            3,
        ),
        "total_preprocessing_seconds": round(
            sum(r.get("preprocessing_seconds", 0.0) for r in records), 3
        ),
        "total_ranking_seconds": round(
            sum(
                r.get(
                    "ranking_screening_sorting_seconds",
                    r.get("timing_seconds", {}).get("ranking", 0.0),
                )
                for r in records
            ),
            3,
        ),
        "total_solver_seconds": round(
            sum(
                r.get(
                    "solver_seconds",
                    r.get("timing_seconds", {}).get("solver", 0.0),
                )
                for r in records
            ),
            3,
        ),
        "total_verification_seconds": round(
            sum(
                r.get(
                    "verification_seconds",
                    r.get("timing_seconds", {}).get("verification", 0.0),
                )
                for r in records
            ),
            3,
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Strict Pre-Execution Manifest Gate and Dataset Integrity Check
    manifest_data = None
    if args.prospective:
        try:
            manifest_data = validate_manifest_for_prospective_execution(
                args.protocol_manifest,
                cli_args=args,
                base_dir=REPOSITORY_ROOT,
            )
            dataset_info = verify_attack_dataset_integrity(
                args.dataset,
                manifest_data=manifest_data,
                is_prospective=True,
            )
            print("[INFO] Manifest validated for PROSPECTIVE execution (FROZEN_AUTHORIZED).")
            print(
                f"[INFO] Prospective dataset validated: {dataset_info['dataset_id']} "
                f"(N={dataset_info['instance_count']}, seed={dataset_info['seed']})."
            )
        except ManifestValidationError as err:
            raise SystemExit(f"PROSPECTIVE EXECUTION REFUSED: {err}") from err
    elif args.protocol_manifest is not None:
        try:
            manifest_data = validate_manifest_for_development_execution(
                args.protocol_manifest,
                base_dir=REPOSITORY_ROOT,
            )
            dataset_info = verify_attack_dataset_integrity(
                args.dataset,
                manifest_data=manifest_data,
                is_prospective=False,
            )
            print(
                f"[INFO] Manifest validated in DEVELOPMENT mode "
                f"(status: {manifest_data.get('protocol_status')})."
            )
        except ManifestValidationError as err:
            raise SystemExit(f"DEVELOPMENT MANIFEST VALIDATION FAILED: {err}") from err
    else:
        dataset_info = verify_attack_dataset_integrity(
            args.dataset,
            manifest_data=None,
            is_prospective=False,
        )

    # Determine execution device
    device_obj = torch.device(
        args.device
        if (
            (args.device == "mps" and torch.backends.mps.is_available())
            or (args.device == "cuda" and torch.cuda.is_available())
        )
        else "cpu"
    )

    # Output file path
    output_path = args.output
    if output_path is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_path = RESULTS_DIR / f"seven_round_comparators_{stamp}.json"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate or compute hashes
    dataset_hash = dataset_info["sha256"]
    dataset_id = dataset_info["dataset_id"]
    stage1_hash = _sha256_file(args.stage1_model)
    stage2_hash = _sha256_file(args.stage2_model)
    classical_hash = _sha256_file(args.classical_dist)

    effective_candidate_budget = (
        args.candidate_budget
        if args.candidate_budget is not None
        else (args.m1 * args.m2)
    )

    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_manifest_path": str(args.protocol_manifest) if args.protocol_manifest else None,
        "prospective": args.prospective,
        "protocol_status": manifest_data.get("protocol_status") if manifest_data else "UNSPECIFIED_DEVELOPMENT",
        "dataset_path": str(args.dataset),
        "dataset_sha256": dataset_hash,
        "dataset_id": dataset_id,
        "stage1_model_path": str(args.stage1_model),
        "stage1_model_sha256": stage1_hash,
        "stage2_model_path": str(args.stage2_model),
        "stage2_model_sha256": stage2_hash,
        "classical_dist_path": str(args.classical_dist),
        "classical_dist_sha256": classical_hash,
        "rounds": ROUNDS,
        "ranking_differential_pairs": RANKING_DIFFERENTIAL_PAIRS,
        "ranking_total_observations": RANKING_TOTAL_OBSERVATIONS,
        "sat_observations_4obs": SAT_OBSERVATIONS_4OBS,
        "sat_observations_all": SAT_OBSERVATIONS_ALL,
        "sat_observations_by_method": {
            "hierarchical_neural_screened": SAT_OBSERVATIONS_4OBS,
            "hierarchical_neural_full": SAT_OBSERVATIONS_4OBS,
            "hierarchical_neural_classical": SAT_OBSERVATIONS_4OBS,
            "random_sat": SAT_OBSERVATIONS_4OBS,
            "sat_only_4obs": SAT_OBSERVATIONS_4OBS,
            "sat_only_all_observations": SAT_OBSERVATIONS_ALL,
        },
        "formula_metrics_4obs": {
            "variables": 2368,
            "cnf_clauses": 3350,
            "xor_clauses": 1632,
            "total_clauses": 4982,
        },
        "formula_metrics_all_observations": {
            "variables": 61888,
            "cnf_clauses": 90274,
            "xor_clauses": 43296,
            "total_clauses": 133570,
        },
        "verification_pairs": VERIFICATION_PAIRS,
        "verification_seed": VERIFICATION_SEED,
        "m1": args.m1,
        "m2": args.m2,
        "candidate_budget": effective_candidate_budget,
        "solver_timeout": args.timeout,
        "global_timeout": args.global_timeout,
        "device": str(device_obj),
        "methods": args.methods,
    }

    # Load existing results if resuming (fail-closed on error or mismatch)
    records_by_method: dict[str, list[dict[str, Any]]] = {}
    if args.resume:
        if not output_path.exists():
            raise SystemExit(
                f"Error: --resume specified but output path {output_path} does not exist."
            )
        try:
            with output_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
        except Exception as err:
            raise SystemExit(
                f"Error: Failed to parse JSON from resume file {output_path}: {err}"
            )

        if not isinstance(loaded, dict):
            raise SystemExit(
                f"Error: resume file {output_path} top level must be a JSON object."
            )

        loaded_metadata = loaded.get("metadata")
        if not isinstance(loaded_metadata, dict):
            raise SystemExit(
                f"Error: resume file {output_path} missing or invalid 'metadata' section."
            )

        # Check all required metadata fields exist
        missing_metadata = sorted(
            REQUIRED_METADATA_FIELDS - set(loaded_metadata.keys())
        )
        if missing_metadata:
            raise SystemExit(
                f"Error: resume file {output_path} metadata missing required fields: {missing_metadata}"
            )

        # Check schema version
        if loaded_metadata.get("schema_version") != SCHEMA_VERSION:
            raise SystemExit(
                f"Error: resume file schema_version mismatch: expected {SCHEMA_VERSION!r}, "
                f"found {loaded_metadata.get('schema_version')!r}."
            )

        config_checks = [
            ("dataset_sha256", dataset_hash),
            ("stage1_model_sha256", stage1_hash),
            ("stage2_model_sha256", stage2_hash),
            ("classical_dist_sha256", classical_hash),
            ("rounds", ROUNDS),
            ("ranking_differential_pairs", RANKING_DIFFERENTIAL_PAIRS),
            ("ranking_total_observations", RANKING_TOTAL_OBSERVATIONS),
            ("sat_observations_4obs", SAT_OBSERVATIONS_4OBS),
            ("sat_observations_all", SAT_OBSERVATIONS_ALL),
            ("verification_pairs", VERIFICATION_PAIRS),
            ("verification_seed", VERIFICATION_SEED),
            ("m1", args.m1),
            ("m2", args.m2),
            ("candidate_budget", effective_candidate_budget),
            ("solver_timeout", args.timeout),
            ("global_timeout", args.global_timeout),
        ]
        for key, expected in config_checks:
            actual = loaded_metadata.get(key)
            if actual != expected:
                raise SystemExit(
                    f"Error: --resume configuration mismatch for '{key}': "
                    f"expected {expected!r}, found {actual!r}. "
                    "Refusing to resume into an incompatible run."
                )

        records_by_method_raw = loaded.get("records_by_method")
        if not isinstance(records_by_method_raw, dict):
            raise SystemExit(
                f"Error: resume file {output_path} missing or invalid 'records_by_method'."
            )

        for m_name, m_records in records_by_method_raw.items():
            if m_name not in COMPARATOR_METHODS and m_name not in METHOD_ALIASES:
                raise SystemExit(
                    f"Error: resume file contains unrecognized method {m_name!r}."
                )
            if not isinstance(m_records, list):
                raise SystemExit(
                    f"Error: records for method {m_name} must be a list."
                )
            seen_instances: set[str] = set()
            for rec in m_records:
                if not isinstance(rec, dict):
                    raise SystemExit(
                        f"Error: record in method {m_name} is not an object."
                    )
                missing_rec_fields = sorted(
                    REQUIRED_RECORD_FIELDS - set(rec.keys())
                )
                if missing_rec_fields:
                    raise SystemExit(
                        f"Error: incomplete record for instance {rec.get('instance_id')!r} "
                        f"in method {m_name!r}: missing {missing_rec_fields}."
                    )
                inst_id = rec["instance_id"]
                if inst_id in seen_instances:
                    raise SystemExit(
                        f"Error: duplicate record for instance {inst_id!r} in method {m_name!r}."
                    )
                seen_instances.add(inst_id)

        records_by_method = {
            m: list(records_by_method_raw.get(m, [])) for m in args.methods
        }
        print(
            f"Resuming from {output_path} with "
            f"{sum(len(r) for r in records_by_method.values())} validated records across "
            f"{len(records_by_method)} methods."
        )

    for m in args.methods:
        records_by_method.setdefault(m, [])

    # Load models (offline setup timing recorded separately)
    t_load_start = time.perf_counter()
    print(f"Loading Stage 1 model from {args.stage1_model}...")
    stage1_pyt = load_pytorch_ranker(args.stage1_model, device=device_obj)

    print(f"Loading Stage 2 neural model from {args.stage2_model}...")
    stage2_pyt = load_pytorch_ranker(args.stage2_model, device=device_obj)

    print(f"Loading Stage 2 classical distribution from {args.classical_dist}...")
    classical_table = ClassicalDistributionTable.load(args.classical_dist)
    model_init_seconds = time.perf_counter() - t_load_start
    metadata["model_initialization_seconds"] = round(model_init_seconds, 3)
    metadata["timing_convention"] = (
        "Online attack time T_online is measured per-instance with rankers preloaded: "
        "T_online = T_preprocess + T_rank + T_screen + T_sort + T_SAT + T_verify. "
        "Offline model initialization is recorded separately in model_initialization_seconds."
    )

    # Load dataset
    shard = load_attack_shard(args.dataset, expected_dataset_id=dataset_id)
    selected_instances = shard.instances[args.start : args.start + args.instances]
    if not selected_instances:
        raise SystemExit("No instances selected for evaluation.")

    print(
        f"\nStarting 7-round comparator evaluation: {len(selected_instances)} instances "
        f"(indices {args.start}..{args.start + len(selected_instances) - 1}) on {len(args.methods)} methods."
    )
    print(
        f"Parameters: M1={args.m1}, M2={args.m2}, Budget={effective_candidate_budget}, "
        f"SolverTimeout={args.timeout}s, GlobalTimeout={args.global_timeout}s\n"
    )

    summary_by_method: dict[str, Any] = {}

    for idx, instance in enumerate(selected_instances):
        global_idx = args.start + idx
        instance_id = instance.instance_id
        known_all = known_pairs_for_attack_instance(instance)  # 128 obs (64 pairs)
        known_4 = known_all[:SAT_OBSERVATIONS_4OBS]  # 4 obs (2 pairs)
        verification = independent_verification_pairs(
            instance.master_key,
            rounds=ROUNDS,
            pair_count=VERIFICATION_PAIRS,
            seed=VERIFICATION_SEED,
        )

        print(f"--- Instance [{global_idx:02d}] {instance_id} ---")

        for method in args.methods:
            # Check if already recorded in resume mode
            if any(
                r.get("instance_id") == instance_id
                for r in records_by_method[method]
            ):
                print(f"  {method:<32} [ALREADY RECORDED]")
                continue

            # Absolute method-level deadline: every method independently receives the global budget
            method_started = time.perf_counter()
            cutoff_seconds = args.global_timeout
            method_deadline = method_started + cutoff_seconds

            rec: dict[str, Any]

            if method == "hierarchical_neural_screened":
                t_prep_start = time.perf_counter()
                stages = (
                    HierarchicalStage(
                        round_index=ROUNDS - 1,
                        model=stage1_pyt,
                        beam_width=args.m1,
                    ),
                    HierarchicalStage(
                        round_index=ROUNDS - 2,
                        model=stage2_pyt,
                        beam_width=args.m2,
                        screened=True,
                        screen_pairs=16,
                        screen_top_k=512,
                    ),
                )
                preprocessing_seconds = time.perf_counter() - t_prep_start
                remaining = max(0.0, method_deadline - time.perf_counter())

                if remaining <= 0.0:
                    rec = {
                        "strategy": "hierarchical_multi_round_beam",
                        "success": False,
                        "status": "global_timeout",
                        "failure_cause": "global_timeout",
                        "stages": [],
                        "candidates_attempted": 0,
                        "attempted_paths": [],
                        "recovered_master_key": None,
                        "timing_seconds": {
                            "ranking": 0.0,
                            "solver": 0.0,
                            "verification": 0.0,
                            "total": 0.0,
                        },
                        "wall_seconds": 0.0,
                        "preprocessing_seconds": round(preprocessing_seconds, 6),
                        "ranking_screening_sorting_seconds": 0.0,
                        "solver_seconds": 0.0,
                        "verification_seconds": 0.0,
                    }
                else:
                    eff_timeout = min(args.timeout, remaining)
                    t_run_start = time.perf_counter()
                    res = run_hierarchical_multi_round_attack(
                        instance,
                        rounds=ROUNDS,
                        stages=stages,
                        known_pairs=known_4,
                        verification_pairs=verification,
                        solver_timeout=eff_timeout,
                        global_timeout=remaining,
                        candidate_batch_size=args.batch_size,
                        device=device_obj,
                    )
                    wall_sec = time.perf_counter() - t_run_start
                    rec = res.to_dict()
                    rec["wall_seconds"] = round(wall_sec, 3)
                    rec["preprocessing_seconds"] = round(preprocessing_seconds, 6)
                    rec["ranking_screening_sorting_seconds"] = round(
                        res.ranking_seconds, 6
                    )
                    rec["solver_seconds"] = round(res.solver_seconds, 6)
                    rec["verification_seconds"] = round(
                        res.verification_seconds, 6
                    )

            elif method == "hierarchical_neural_full":
                t_prep_start = time.perf_counter()
                stages = (
                    HierarchicalStage(
                        round_index=ROUNDS - 1,
                        model=stage1_pyt,
                        beam_width=args.m1,
                    ),
                    HierarchicalStage(
                        round_index=ROUNDS - 2,
                        model=stage2_pyt,
                        beam_width=args.m2,
                        screened=False,
                    ),
                )
                preprocessing_seconds = time.perf_counter() - t_prep_start
                remaining = max(0.0, method_deadline - time.perf_counter())

                if remaining <= 0.0:
                    rec = {
                        "strategy": "hierarchical_multi_round_beam",
                        "success": False,
                        "status": "global_timeout",
                        "failure_cause": "global_timeout",
                        "stages": [],
                        "candidates_attempted": 0,
                        "attempted_paths": [],
                        "recovered_master_key": None,
                        "timing_seconds": {
                            "ranking": 0.0,
                            "solver": 0.0,
                            "verification": 0.0,
                            "total": 0.0,
                        },
                        "wall_seconds": 0.0,
                        "preprocessing_seconds": round(preprocessing_seconds, 6),
                        "ranking_screening_sorting_seconds": 0.0,
                        "solver_seconds": 0.0,
                        "verification_seconds": 0.0,
                    }
                else:
                    eff_timeout = min(args.timeout, remaining)
                    t_run_start = time.perf_counter()
                    res = run_hierarchical_multi_round_attack(
                        instance,
                        rounds=ROUNDS,
                        stages=stages,
                        known_pairs=known_4,
                        verification_pairs=verification,
                        solver_timeout=eff_timeout,
                        global_timeout=remaining,
                        candidate_batch_size=args.batch_size,
                        device=device_obj,
                    )
                    wall_sec = time.perf_counter() - t_run_start
                    rec = res.to_dict()
                    rec["wall_seconds"] = round(wall_sec, 3)
                    rec["preprocessing_seconds"] = round(preprocessing_seconds, 6)
                    rec["ranking_screening_sorting_seconds"] = round(
                        res.ranking_seconds, 6
                    )
                    rec["solver_seconds"] = round(res.solver_seconds, 6)
                    rec["verification_seconds"] = round(
                        res.verification_seconds, 6
                    )

            elif method in ("hierarchical_neural_classical", "hierarchical_classical"):
                t_prep_start = time.perf_counter()
                stages = (
                    HierarchicalStage(
                        round_index=ROUNDS - 1,
                        model=stage1_pyt,
                        beam_width=args.m1,
                    ),
                    HierarchicalStage(
                        round_index=ROUNDS - 2,
                        model=classical_table,
                        beam_width=args.m2,
                        screened=False,
                    ),
                )
                preprocessing_seconds = time.perf_counter() - t_prep_start
                remaining = max(0.0, method_deadline - time.perf_counter())

                if remaining <= 0.0:
                    rec = {
                        "strategy": "hierarchical_multi_round_beam",
                        "success": False,
                        "status": "global_timeout",
                        "failure_cause": "global_timeout",
                        "stages": [],
                        "candidates_attempted": 0,
                        "attempted_paths": [],
                        "recovered_master_key": None,
                        "timing_seconds": {
                            "ranking": 0.0,
                            "solver": 0.0,
                            "verification": 0.0,
                            "total": 0.0,
                        },
                        "wall_seconds": 0.0,
                        "preprocessing_seconds": round(preprocessing_seconds, 6),
                        "ranking_screening_sorting_seconds": 0.0,
                        "solver_seconds": 0.0,
                        "verification_seconds": 0.0,
                    }
                else:
                    eff_timeout = min(args.timeout, remaining)
                    t_run_start = time.perf_counter()
                    res = run_hierarchical_multi_round_attack(
                        instance,
                        rounds=ROUNDS,
                        stages=stages,
                        known_pairs=known_4,
                        verification_pairs=verification,
                        solver_timeout=eff_timeout,
                        global_timeout=remaining,
                        candidate_batch_size=args.batch_size,
                        device=device_obj,
                    )
                    wall_sec = time.perf_counter() - t_run_start
                    rec = res.to_dict()
                    rec["wall_seconds"] = round(wall_sec, 3)
                    rec["preprocessing_seconds"] = round(preprocessing_seconds, 6)
                    rec["ranking_screening_sorting_seconds"] = round(
                        res.ranking_seconds, 6
                    )
                    rec["solver_seconds"] = round(res.solver_seconds, 6)
                    rec["verification_seconds"] = round(
                        res.verification_seconds, 6
                    )

            elif method == "random_sat":
                t_prep_start = time.perf_counter()
                method_factory = IsolatedSolverFactory.from_pairs(
                    known_4,
                    rounds=ROUNDS,
                    instance_id=f"{instance_id}_{method}",
                )
                model = method_factory.build_isolated_model("random_multi_round_sat")
                preprocessing_seconds = time.perf_counter() - t_prep_start
                remaining = max(0.0, method_deadline - time.perf_counter())

                if remaining <= 0.0:
                    rec = {
                        "strategy": "random_multi_round_sat",
                        "success": False,
                        "status": "global_timeout",
                        "failure_cause": "global_timeout",
                        "candidates_attempted": 0,
                        "recovered_master_key": None,
                        "timing_seconds": {
                            "ranking": 0.0,
                            "solver": 0.0,
                            "verification": 0.0,
                            "total": 0.0,
                        },
                        "wall_seconds": 0.0,
                        "preprocessing_seconds": round(preprocessing_seconds, 6),
                        "ranking_screening_sorting_seconds": 0.0,
                        "solver_seconds": 0.0,
                        "verification_seconds": 0.0,
                    }
                else:
                    rand_seed = 100_000 + global_idx
                    eff_timeout = min(args.timeout, remaining)
                    t_run_start = time.perf_counter()
                    recovery_res = run_random_multi_round_recovery(
                        model,
                        known_pairs=known_4,
                        verification_pairs=verification,
                        rounds=ROUNDS,
                        round_indices=(ROUNDS - 1, ROUNDS - 2),
                        candidate_budget=effective_candidate_budget,
                        seed=rand_seed,
                        solver_timeout=eff_timeout,
                        global_timeout=remaining,
                        strategy="random_multi_round_sat",
                        solver_factory=method_factory,
                    )
                    wall_sec = time.perf_counter() - t_run_start
                    rec = {
                        "strategy": recovery_res.strategy,
                        "success": recovery_res.success,
                        "status": recovery_res.status,
                        "failure_cause": (
                            None
                            if recovery_res.success
                            else recovery_res.status
                        ),
                        "candidates_attempted": recovery_res.candidates_attempted,
                        "recovered_master_key": (
                            list(recovery_res.recovered_master_key)
                            if recovery_res.recovered_master_key is not None
                            else None
                        ),
                        "timing_seconds": {
                            "ranking": 0.0,
                            "solver": recovery_res.solver_seconds,
                            "verification": recovery_res.verification_seconds,
                            "total": wall_sec,
                        },
                        "wall_seconds": round(wall_sec, 3),
                        "preprocessing_seconds": round(preprocessing_seconds, 6),
                        "ranking_screening_sorting_seconds": 0.0,
                        "solver_seconds": round(recovery_res.solver_seconds, 6),
                        "verification_seconds": round(
                            recovery_res.verification_seconds, 6
                        ),
                    }

            elif method in ("sat_only_4obs", "sat_only"):
                t_prep_start = time.perf_counter()
                method_factory = IsolatedSolverFactory.from_pairs(
                    known_4,
                    rounds=ROUNDS,
                    instance_id=f"{instance_id}_{method}",
                )
                model = method_factory.build_isolated_model("sat_only_4obs")
                preprocessing_seconds = time.perf_counter() - t_prep_start
                remaining = max(0.0, method_deadline - time.perf_counter())

                if remaining <= 0.0:
                    rec = {
                        "strategy": "sat_only_4obs",
                        "success": False,
                        "status": "global_timeout",
                        "failure_cause": "global_timeout",
                        "candidates_attempted": 0,
                        "recovered_master_key": None,
                        "timing_seconds": {
                            "ranking": 0.0,
                            "solver": 0.0,
                            "verification": 0.0,
                            "total": 0.0,
                        },
                        "wall_seconds": 0.0,
                        "preprocessing_seconds": round(preprocessing_seconds, 6),
                        "ranking_screening_sorting_seconds": 0.0,
                        "solver_seconds": 0.0,
                        "verification_seconds": 0.0,
                    }
                else:
                    t_run_start = time.perf_counter()
                    recovery_res = run_multi_round_sat_only_recovery(
                        model,
                        known_pairs=known_4,
                        verification_pairs=verification,
                        rounds=ROUNDS,
                        solver_timeout=remaining,
                        global_timeout=remaining,
                        strategy="sat_only_4obs",
                    )
                    wall_sec = time.perf_counter() - t_run_start
                    rec = {
                        "strategy": recovery_res.strategy,
                        "success": recovery_res.success,
                        "status": recovery_res.status,
                        "failure_cause": (
                            None
                            if recovery_res.success
                            else recovery_res.status
                        ),
                        "candidates_attempted": recovery_res.candidates_attempted,
                        "recovered_master_key": (
                            list(recovery_res.recovered_master_key)
                            if recovery_res.recovered_master_key is not None
                            else None
                        ),
                        "timing_seconds": {
                            "ranking": 0.0,
                            "solver": recovery_res.solver_seconds,
                            "verification": recovery_res.verification_seconds,
                            "total": wall_sec,
                        },
                        "wall_seconds": round(wall_sec, 3),
                        "preprocessing_seconds": round(preprocessing_seconds, 6),
                        "ranking_screening_sorting_seconds": 0.0,
                        "solver_seconds": round(recovery_res.solver_seconds, 6),
                        "verification_seconds": round(
                            recovery_res.verification_seconds, 6
                        ),
                    }

            elif method == "sat_only_all_observations":
                t_prep_start = time.perf_counter()
                method_factory = IsolatedSolverFactory.from_pairs(
                    known_all,
                    rounds=ROUNDS,
                    instance_id=f"{instance_id}_{method}",
                )
                model = method_factory.build_isolated_model("sat_only_all_observations")
                preprocessing_seconds = time.perf_counter() - t_prep_start
                remaining = max(0.0, method_deadline - time.perf_counter())

                if remaining <= 0.0:
                    rec = {
                        "strategy": "sat_only_all_observations",
                        "success": False,
                        "status": "global_timeout",
                        "failure_cause": "global_timeout",
                        "candidates_attempted": 0,
                        "recovered_master_key": None,
                        "timing_seconds": {
                            "ranking": 0.0,
                            "solver": 0.0,
                            "verification": 0.0,
                            "total": 0.0,
                        },
                        "wall_seconds": 0.0,
                        "preprocessing_seconds": round(preprocessing_seconds, 6),
                        "ranking_screening_sorting_seconds": 0.0,
                        "solver_seconds": 0.0,
                        "verification_seconds": 0.0,
                    }
                else:
                    t_run_start = time.perf_counter()
                    recovery_res = run_multi_round_sat_only_recovery(
                        model,
                        known_pairs=known_all,
                        verification_pairs=verification,
                        rounds=ROUNDS,
                        solver_timeout=remaining,
                        global_timeout=remaining,
                        strategy="sat_only_all_observations",
                    )
                    wall_sec = time.perf_counter() - t_run_start
                    rec = {
                        "strategy": recovery_res.strategy,
                        "success": recovery_res.success,
                        "status": recovery_res.status,
                        "failure_cause": (
                            None
                            if recovery_res.success
                            else recovery_res.status
                        ),
                        "candidates_attempted": recovery_res.candidates_attempted,
                        "recovered_master_key": (
                            list(recovery_res.recovered_master_key)
                            if recovery_res.recovered_master_key is not None
                            else None
                        ),
                        "timing_seconds": {
                            "ranking": 0.0,
                            "solver": recovery_res.solver_seconds,
                            "verification": recovery_res.verification_seconds,
                            "total": wall_sec,
                        },
                        "wall_seconds": round(wall_sec, 3),
                        "preprocessing_seconds": round(preprocessing_seconds, 6),
                        "ranking_screening_sorting_seconds": 0.0,
                        "solver_seconds": round(recovery_res.solver_seconds, 6),
                        "verification_seconds": round(
                            recovery_res.verification_seconds, 6
                        ),
                    }
            else:
                raise ValueError(f"Unknown method {method!r}")

            method_wall_sec = time.perf_counter() - method_started
            classified = classify_attack_outcome(
                rec["status"],
                success=bool(rec["success"]),
                unresolved_timeouts=1 if rec["status"] == "solver_timeout" else 0,
            )
            rec.update(
                {
                    "instance_id": instance_id,
                    "instance_index": global_idx,
                    "method": method,
                    "method_display_name": METHOD_DISPLAY_NAMES.get(method, method),
                    "method_wall_seconds": round(method_wall_sec, 6),
                    "cutoff_seconds": cutoff_seconds,
                    "cutoff_overshoot_seconds": round(
                        max(0.0, method_wall_sec - cutoff_seconds), 6
                    ),
                    "counts_as_verified_recovery": classified["counts_as_verified_recovery"],
                    "counts_as_unconditional_failure": classified["counts_as_unconditional_failure"],
                    "is_resolved_sat_state": classified["is_resolved_sat_state"],
                    "failure_domain": classified["failure_domain"],
                }
            )

            records_by_method[method].append(rec)
            status_str = f"success={rec['success']} status={rec['status']} cause={rec['failure_cause']}"
            tuples_str = f"tuples={rec['candidates_attempted']}"
            overshoot_str = f"overshoot={rec['cutoff_overshoot_seconds']:.4f}s"
            print(
                f"  {method:<30} {status_str:<45} {tuples_str:<12} {overshoot_str:<18} wall={method_wall_sec:6.2f}s"
            )

        # Save checkpoint after each instance preserving all methods
        summary_by_method = {
            m: compute_method_summary(records_by_method[m]) for m in records_by_method
        }
        checkpoint_data = {
            "metadata": metadata,
            "evidence_class": "DEVELOPMENT_DIAGNOSTIC",
            "protocol_status": "PROPOSED_DEVELOPMENT_ONLY",
            "generated_at": datetime.now(UTC).isoformat(),
            "summary_by_method": summary_by_method,
            "records_by_method": records_by_method,
        }
        _save_atomic(checkpoint_data, output_path)

    # Print final summary table
    print("\n" + "=" * 105)
    print(f"{'7-ROUND COMPARATOR EVALUATION SUMMARY':^105}")
    print("=" * 105)
    print(
        f"{'Method':<32} | {'Success':<8} | {'Rate':<8} | {'95% CI':<14} | {'Med Wall':<9} | {'Max Over':<9} | {'Mean Tuples':<11}"
    )
    print("-" * 105)
    for method in args.methods:
        sm = summary_by_method.get(method, {})
        if not sm:
            continue
        ci_str = f"[{sm['wilson_95_ci'][0]:.3f}, {sm['wilson_95_ci'][1]:.3f}]"
        print(
            f"{method:<32} | "
            f"{sm['verified_recoveries']:>2}/{sm['instances_evaluated']:<2}   | "
            f"{sm['verified_recovery_rate']*100:>5.1f}%  | "
            f"{ci_str:<14} | "
            f"{sm['median_method_wall_seconds']:>7.2f}s | "
            f"{sm['max_cutoff_overshoot_seconds']:>7.3f}s | "
            f"{sm['mean_candidates_attempted']:>11.1f}"
        )
    print("=" * 105)
    print(f"Output saved to: {output_path}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
