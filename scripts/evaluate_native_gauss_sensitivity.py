#!/usr/bin/env python3
"""Evaluate native CryptoMiniSat Gaussian elimination sensitivity baselines.

Predeclared secondary sensitivity treatments:
1. sat_only_native_gauss_4obs: Native cryptominisat5 CLI with 4 SAT observations (2 pairs).
2. sat_only_native_gauss_all_observations: Native cryptominisat5 CLI with 128 SAT observations (64 pairs).

Accounting rules:
- Full wall-clock accounting includes DIMACS serialization, subprocess launch,
  pipe communication, log parsing, and independent fresh-pair verification.
- Outcome mapped to the 4 orthogonal failure taxonomy fields:
  counts_as_verified_recovery, counts_as_unconditional_failure,
  is_resolved_sat_state, failure_domain.
- Reports matrix detection, dimension, and usage directly from solver stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from neural_sat_speck.data.attack import load_attack_shard  # noqa: E402
from neural_sat_speck.evaluation.metrics import wilson_score_interval  # noqa: E402
from neural_sat_speck.protocol.manifest_validator import (  # noqa: E402
    ManifestValidationError,
    validate_manifest_for_development_execution,
    validate_manifest_for_prospective_execution,
    verify_attack_dataset_integrity,
)
from neural_sat_speck.sat.native_sensitivity import (  # noqa: E402
    run_native_gauss_sensitivity,
)

DEFAULT_DATASET_PATH = (
    REPOSITORY_ROOT / "datasets" / "phase11a_oracle_r7" / "oracle_evaluation.json"
)
DATASET_ID = "speck32_64_attack_phase11a_oracle_r7"
RESULTS_DIR = REPOSITORY_ROOT / "results" / "phase11_scaling_diagnostics"
MANIFEST_PATH = REPOSITORY_ROOT / "docs" / "seven_round_protocol_manifest.json"

SENSITIVITY_METHODS = (
    "sat_only_native_gauss_4obs",
    "sat_only_native_gauss_all_observations",
)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _save_atomic(data: dict[str, Any], target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(f".tmp.{os.getpid()}")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(target_path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=SENSITIVITY_METHODS,
        default=list(SENSITIVITY_METHODS),
        help="Sensitivity methods to evaluate (default: both 4obs and all_observations)",
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=3,
        help="Number of instances to evaluate (default: 3)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index in the dataset (default: 0)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-instance SAT solver timeout in seconds (default: 300.0)",
    )
    parser.add_argument(
        "--maxmatrixcols",
        type=int,
        default=2500,
        help="Maximum matrix column count for Gauss elimination (default: 2500)",
    )
    parser.add_argument(
        "--autodisablegauss",
        type=int,
        default=0,
        choices=[0, 1],
        help="Whether to autodisable Gauss elimination (0=no, 1=yes; default: 0)",
    )
    parser.add_argument(
        "--maxmatrixrows",
        type=int,
        default=None,
        help="Optional maximum matrix rows (e.g. 50000 for 128 observations)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to attack dataset JSON (required for --prospective; defaults to opened Seed 300001 dataset in development mode)",
    )
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        default=None,
        help="Path to protocol manifest JSON (enforced strictly for --prospective)",
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing run, skipping completed instance-method pairs (requires --output)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file path",
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

    if args.resume and args.output is None:
        raise SystemExit("Error: --resume requires an explicit --output path.")

    return args


def compute_sensitivity_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    n = len(records)
    verified = sum(1 for r in records if r.get("counts_as_verified_recovery"))
    uncond_fail = sum(1 for r in records if r.get("counts_as_unconditional_failure"))
    resolved_sat = sum(1 for r in records if r.get("is_resolved_sat_state"))
    matrix_used_count = sum(
        1 for r in records if r.get("matrix_diagnostics", {}).get("matrix_used")
    )
    wall_times = [r.get("timing_seconds", {}).get("total_wall", 0.0) for r in records]
    solver_cpu = [r.get("timing_seconds", {}).get("solver_cpu", 0.0) for r in records]

    # Failure domain breakdown
    failure_domains: dict[str, int] = {}
    for r in records:
        dom = r.get("failure_domain", "unknown")
        failure_domains[dom] = failure_domains.get(dom, 0) + 1

    return {
        "instances_evaluated": n,
        "verified_recoveries": verified,
        "verified_recovery_rate": round(verified / n, 4),
        "wilson_95_ci": [round(x, 4) for x in wilson_score_interval(verified, n)],
        "unconditional_failures": uncond_fail,
        "resolved_sat_states": resolved_sat,
        "matrix_used_instances": matrix_used_count,
        "matrix_used_rate": round(matrix_used_count / n, 4),
        "mean_wall_seconds": round(sum(wall_times) / n, 3) if n else 0.0,
        "median_wall_seconds": round(statistics.median(wall_times), 3) if wall_times else 0.0,
        "mean_solver_cpu_seconds": round(sum(solver_cpu) / n, 3) if n else 0.0,
        "failure_domain_distribution": failure_domains,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Output file path
    output_path = args.output
    if output_path is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_path = RESULTS_DIR / f"native_gauss_sensitivity_{stamp}.json"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Manifest and dataset validation
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

    metadata = {
        "schema_version": "1.0.0",
        "evidence_class": "PROSPECTIVE_SENSITIVITY" if args.prospective else "DEVELOPMENT_SENSITIVITY",
        "dataset_path": str(args.dataset),
        "dataset_sha256": dataset_info["sha256"],
        "dataset_id": dataset_info["dataset_id"],
        "rounds": 7,
        "timeout_seconds": args.timeout,
        "maxmatrixcols": args.maxmatrixcols,
        "autodisablegauss": args.autodisablegauss,
        "maxmatrixrows": args.maxmatrixrows,
        "threads": 1,
        "methods": args.methods,
        "protocol_manifest_path": str(args.protocol_manifest) if args.protocol_manifest else None,
        "protocol_status": manifest_data.get("protocol_status") if manifest_data else "UNSPECIFIED_DEVELOPMENT",
        "formula_metrics": {
            "sat_only_native_gauss_4obs": {
                "observations": 4,
                "differential_pairs_equivalent": 2,
                "variables": 2368,
                "cnf_clauses_unpinned": 3350,
                "xor_clauses": 1632,
                "total_clauses_unpinned": 4982,
            },
            "sat_only_native_gauss_all_observations": {
                "observations": 128,
                "differential_pairs_equivalent": 64,
                "variables": 61888,
                "cnf_clauses_unpinned": 90274,
                "xor_clauses": 43296,
                "total_clauses_unpinned": 133570,
            },
        },
        "environment_provenance": {
            "python_version": sys.version.split()[0],
            "cryptominisat_cli_version": "5.14.7",
            "platform": "macOS-27.0-arm64-arm-64bit-Mach-O",
            "machine": "arm64",
        },
    }

    records_by_method: dict[str, list[dict[str, Any]]] = {m: [] for m in args.methods}
    completed_instances: set[str] = set()

    if args.resume and output_path.exists():
        try:
            with output_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            ex_meta = existing.get("metadata", {})
            if ex_meta.get("dataset_sha256") != dataset_info["sha256"]:
                raise SystemExit(
                    f"Resume refused: dataset SHA-256 mismatch in {output_path} "
                    f"(existing={ex_meta.get('dataset_sha256')}, current={dataset_info['sha256']})"
                )
            for m in args.methods:
                if m in existing.get("records_by_method", {}):
                    records_by_method[m] = existing["records_by_method"][m]
            method_sets = [
                {r["instance_id"] for r in records_by_method[m]} for m in args.methods
            ]
            if method_sets:
                completed_instances = set.intersection(*method_sets)
            print(f"[INFO] Resumed run: {len(completed_instances)} instances already completed.")
        except Exception as err:
            raise SystemExit(f"Failed to resume from {output_path}: {err}") from err

    shard = load_attack_shard(args.dataset, expected_dataset_id=dataset_info["dataset_id"])
    selected_instances = shard.instances[args.start : args.start + args.instances]
    if not selected_instances:
        raise SystemExit("No instances selected for evaluation.")

    print("\n" + "=" * 95)
    print(f"{'CRYPTO-MINISAT NATIVE GAUSS SENSITIVITY EVALUATION':^95}")
    print("=" * 95)
    print(f"Instances: {len(selected_instances)} | Timeout: {args.timeout}s | Threads: 1")
    print(f"Flags: --maxmatrixcols {args.maxmatrixcols} --autodisablegauss {args.autodisablegauss}")
    print(f"Output: {output_path}\n")

    for idx, instance in enumerate(selected_instances):
        global_idx = args.start + idx
        if instance.instance_id in completed_instances:
            print(f"--- Instance {global_idx} ({instance.instance_id}): already evaluated, skipping ---")
            continue

        print(f"--- Instance {global_idx} ({instance.instance_id}): True k6={instance.final_round_subkey:#06x} ---")

        for method in args.methods:
            res = run_native_gauss_sensitivity(
                instance,
                method_id=method,
                rounds=7,
                timeout_seconds=args.timeout,
                maxmatrixcols=args.maxmatrixcols,
                autodisablegauss=args.autodisablegauss,
                maxmatrixrows=args.maxmatrixrows,
                threads=1,
            )
            rec = res.to_dict()
            rec["instance_index"] = global_idx
            records_by_method[method].append(rec)

            matrix_info = "YES" if res.matrix_diagnostics.get("matrix_used") else f"NO ({res.matrix_diagnostics.get('unused_reason') or 'unused'})"
            print(
                f"  {method:<38} status={res.status:<18} "
                f"matrix={matrix_info:<22} wall={res.timing_seconds['total_wall']:6.2f}s"
            )

        # Atomic checkpoint save after each instance
        summary_by_method = {
            m: compute_sensitivity_summary(records_by_method[m]) for m in records_by_method
        }
        payload = {
            "metadata": metadata,
            "summary_by_method": summary_by_method,
            "records_by_method": records_by_method,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        _save_atomic(payload, output_path)

    print("\n" + "=" * 95)
    print(f"{'SENSITIVITY EVALUATION SUMMARY':^95}")
    print("=" * 95)
    print(
        f"{'Method':<40} | {'Success':<8} | {'Rate':<8} | {'95% CI':<14} | {'Matrix Used':<12} | {'Med Wall':<9}"
    )
    print("-" * 95)
    for method in args.methods:
        sm = summary_by_method.get(method, {})
        if not sm:
            continue
        ci_str = f"[{sm['wilson_95_ci'][0]:.3f}, {sm['wilson_95_ci'][1]:.3f}]"
        print(
            f"{method:<40} | "
            f"{sm['verified_recoveries']:>2}/{sm['instances_evaluated']:<2}   | "
            f"{sm['verified_recovery_rate']*100:>5.1f}%  | "
            f"{ci_str:<14} | "
            f"{sm['matrix_used_instances']:>2}/{sm['instances_evaluated']:<2} ({sm['matrix_used_rate']*100:.0f}%)   | "
            f"{sm['median_wall_seconds']:>7.2f}s"
        )
    print("=" * 95)
    print(f"Output saved to: {output_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
