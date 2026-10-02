#!/usr/bin/env python3
"""Dedicated manifest-enforced runner for SIMON32/64 R=10 unseen confirmation.

Governed by AGENTS.md, D-029, D-034, D-035, D-036, D-037, and D-038.
Enforces strict manifest approval, seed governance, disjoint-family certificates,
exact provenance hashing, and execution-identity-bound resume before evaluating
any outcome-bearing key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from neural_sat_speck.simon.attack_instance import (  # noqa: E402
    SimonAttackInstanceConfig,
    generate_simon_attack_instances,
    load_simon_attack_instances,
)
from neural_sat_speck.simon.comparison_input import SimonComparisonInput  # noqa: E402
from neural_sat_speck.simon.confirmation_guard import (  # noqa: E402
    ALL_CONFIRMATION_METHODS,
    ConfirmationPreflightError,
    DisjointFamilyCertificateError,
    ProvenanceVerificationError,
    ResumeIntegrityError,
    compute_execution_identity,
    file_sha256,
    validate_and_filter_resume,
    validate_confirmation_manifest,
    verify_disjoint_family_certificate,
    verify_independent_fresh,
    verify_runtime_provenance,
)
from neural_sat_speck.simon.key_schedule import expand_key  # noqa: E402
from neural_sat_speck.simon.method_worker import (  # noqa: E402
    MethodWorkerOutcome,
    run_method_worker,
)
from neural_sat_speck.simon.verification import verify_simon_master_key  # noqa: E402


def counterbalance_methods(instance_index: int, methods: tuple[str, ...]) -> tuple[str, ...]:
    """Apply Latin-square cyclic shift to autonomous comparators; diagnostic oracle is always evaluated last."""
    autonomous = [m for m in methods if not m.startswith("oracle_")]
    oracles = [m for m in methods if m.startswith("oracle_")]
    shift = instance_index % len(autonomous)
    rotated = autonomous[shift:] + autonomous[:shift]
    return tuple(rotated + oracles)


def main() -> int:
    preflight_started = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Manifest-enforced execution runner for SIMON32/64 R=10 unseen confirmation."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to the frozen confirmation manifest JSON.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to the confirmation attack instances JSON.",
    )
    parser.add_argument(
        "--certificate",
        type=Path,
        default=None,
        help="Path to the disjoint-family certificate JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the output confirmation ledger JSON.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume execution from an existing ledger, verifying execution identity.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute synthetic plumbing check only; does not produce confirmation evidence.",
    )
    parser.add_argument("--log-dir", type=Path, default=None, help="Directory for durable per-cell monitoring journal.")
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=REPOSITORY_ROOT / "runtime",
        help="Path to persistent workspace runtime directory.",
    )
    args = parser.parse_args()

    print("================================================================================")
    print("SIMON32/64 R=10 UNSEEN CONFIRMATION RUNNER")
    print("================================================================================")

    # 1. Manifest Validation
    print(f"[*] Validating manifest at {args.manifest}...")
    try:
        if args.dry_run:
            with open(args.manifest, encoding="utf-8") as f:
                manifest_dict = json.load(f)
            # In dry-run mode, check if manifest has basic schema
            if manifest_dict.get("protocol_id") not in (
                "simon32_64_r10_unseen_confirmation_v1",
                "simon32_64_r10_unseen_confirmation_replacement_v1",
            ):
                raise ConfirmationPreflightError("Invalid protocol_id in manifest")
            manifest = manifest_dict
            print("[+] Dry-run mode: manifest syntax validated.")
        else:
            manifest = validate_confirmation_manifest(args.manifest)
            print("[+] Manifest fully validated and approved.")
    except ConfirmationPreflightError as exc:
        print(f"[!] PREFLIGHT REJECTION: {exc}", file=sys.stderr)
        return 1

    # 2. Provenance Verification
    print("[*] Verifying runtime provenance and artifact hashes...")
    try:
        resource_locs = dict(manifest.get("development_resource_locations", {}))
        permanent_runtime = (
            args.runtime_dir
            if getattr(args, "runtime_dir", None)
            else (REPOSITORY_ROOT / "runtime")
        )
        if permanent_runtime.is_dir():
            permanent_tf = permanent_runtime / "simon_tf_venv" / "bin" / "python3.12"
            if not permanent_tf.is_file():
                permanent_tf = permanent_runtime / "simon_tf_venv" / "bin" / "python"
            cur_tf = Path(resource_locs.get("tensorflow_python", ""))
            if permanent_tf.is_file() and (not cur_tf.is_file() or str(cur_tf).startswith(("/tmp", "/private/tmp"))):
                resource_locs["tensorflow_python"] = str(permanent_tf)

            permanent_ddt = permanent_runtime / "resources" / "simon_predicted_ddt_7r.bin"
            if not permanent_ddt.is_file():
                permanent_ddt = permanent_runtime / "simon_predicted_ddt_7r.bin"
            cur_ddt = Path(resource_locs.get("ddt_7r", ""))
            if permanent_ddt.is_file() and (not cur_ddt.is_file() or str(cur_ddt).startswith(("/tmp", "/private/tmp"))):
                resource_locs["ddt_7r"] = str(permanent_ddt)

            permanent_8r = permanent_runtime / "resources" / "simon_queried_row_8r"
            if not permanent_8r.is_file():
                permanent_8r = permanent_runtime / "simon_queried_row_8r"
            cur_8r = Path(resource_locs.get("queried_row_8r_binary", ""))
            if permanent_8r.is_file() and (not cur_8r.is_file() or str(cur_8r).startswith(("/tmp", "/private/tmp"))):
                resource_locs["queried_row_8r_binary"] = str(permanent_8r)

            permanent_bao = permanent_runtime / "bao-simon-audit" / "simonNDvsDD" / "SENet"
            if not permanent_bao.is_dir():
                permanent_bao = permanent_runtime / "bao" / "simonNDvsDD" / "SENet"
            cur_bao = Path(resource_locs.get("bao_vv_directory", ""))
            if permanent_bao.is_dir() and (not cur_bao.is_dir() or str(cur_bao).startswith(("/tmp", "/private/tmp"))):
                resource_locs["bao_vv_directory"] = str(permanent_bao)

        manifest = dict(manifest)
        manifest["development_resource_locations"] = resource_locs
        try:
            resolved_paths = verify_runtime_provenance(manifest, resource_locs)
        except TypeError:
            resolved_paths = verify_runtime_provenance(manifest)
        print("[+] All runtime tools, tables, models, and libraries verified.")
    except ProvenanceVerificationError as exc:
        print(f"[!] PROVENANCE REJECTION: {exc}", file=sys.stderr)
        return 2

    # 3. Dataset and Certificate Verification
    if args.dry_run:
        print("[*] Dry-run synthetic fixture mode active.")
        cfg = SimonAttackInstanceConfig(
            instance_count=1,
            pair_count=64,
            rounds=10,
            seed=999999,
            split_role="demo",
        )
        instances = generate_simon_attack_instances(cfg)
        dataset_sha256 = "0" * 64
        evidence_class = "DRY_RUN_SYNTHETIC"
    else:
        if args.dataset is None or not args.dataset.is_file():
            print(f"[!] DATASET REJECTION: dataset file missing: {args.dataset}", file=sys.stderr)
            return 3
        if args.certificate is None or not args.certificate.is_file():
            print(f"[!] CERTIFICATE REJECTION: certificate file missing: {args.certificate}", file=sys.stderr)
            return 3

        print(f"[*] Loading confirmation dataset from {args.dataset}...")
        sidecar = args.dataset.with_name(args.dataset.name + ".sha256")
        if not sidecar.is_file() or sidecar.read_text().split()[0] != file_sha256(args.dataset):
            print("[!] DATASET REJECTION: missing or mismatched SHA-256 sidecar", file=sys.stderr)
            return 3
        instances = load_simon_attack_instances(args.dataset)
        for inst in instances:
            if inst.split_role != "confirmation":
                print(
                    f"[!] DATA ROLE REJECTION: Instance {inst.instance_id} has role {inst.split_role!r}; "
                    "confirmation runner strictly rejects non-confirmation data!",
                    file=sys.stderr,
                )
                return 4

        print(f"[*] Verifying disjoint-family certificate at {args.certificate}...")
        try:
            verify_disjoint_family_certificate(manifest, instances, args.certificate)
            print("[+] Disjoint-family certificate verified: zero prior-key overlap.")
        except DisjointFamilyCertificateError as exc:
            print(f"[!] DISJOINT FAMILY REJECTION: {exc}", file=sys.stderr)
            return 5

        dataset_sha256 = file_sha256(args.dataset)
        evidence_class = "CONFIRMATION"

    # 4. Execution Identity & Resume Validation
    execution_identity = compute_execution_identity(manifest, dataset_sha256)
    methods = ALL_CONFIRMATION_METHODS

    try:
        completed_cells, pending_cells = validate_and_filter_resume(
            ledger_path=args.output,
            expected_identity=execution_identity,
            instances=instances,
            methods=methods,
            resume_flag=args.resume,
        )
    except ResumeIntegrityError as exc:
        print(f"[!] RESUME INTEGRITY REJECTION: {exc}", file=sys.stderr)
        return 6

    print(
        f"[+] Schedule: {len(instances)} instances x {len(methods)} methods = "
        f"{len(instances) * len(methods)} total cells. "
        f"Completed: {len(completed_cells)}, Pending: {len(pending_cells)}."
    )

    if args.dry_run:
        print("[+] Dry-run preflight checks passed completely. Zero worker executions performed.")
        return 0

    def git_output(*arguments):
        return subprocess.run(["git", *arguments], cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    run_provenance = json.loads(args.output.read_text()).get("run_provenance", {}) if args.resume and args.output.exists() else {}
    run_provenance.setdefault("invocations", []).append({
        "argv": [sys.executable, *sys.argv], "cwd": str(Path.cwd()),
        "git_head": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short"),
        "started_at": datetime.now(UTC).isoformat(),
        "common_preflight_wall_seconds": time.perf_counter() - preflight_started,
        "common_preflight_accounting": "Shared provenance/data integrity checks outside the per-method online cutoff; worker startup/loading/verification remain inside 300s.",
    })

    # 5. Outcome-Bearing Cell Execution Loop
    output_records: list[dict[str, Any]] = list(completed_cells.values())
    control_seed = manifest["random_control_stream"]["control_seed"]
    b1_checkpoint = resolved_paths["b1_checkpoint"]
    b1_checkpoint_sha256 = manifest["provenance_hashes"]["b1_checkpoint_sha256"]

    for inst_idx, inst in enumerate(instances):
        inst_methods = counterbalance_methods(inst_idx, methods)
        for method in inst_methods:
            cell = (inst.instance_id, method)
            if cell in completed_cells:
                continue

            print(f"[*] Running {inst.instance_id} | {method}...")
            comp_input = SimonComparisonInput.from_attack_instance(
                inst,
                manifest_seed=manifest["seed_governance"]["seed"],
                pair_count=64,
            )

            oracle_keys = (
                expand_key(inst.master_key, rounds=inst.rounds)[-2:]
                if method.startswith("oracle_")
                else None
            )

            t0 = time.perf_counter()
            outer_exception = None
            try:
                outcome: MethodWorkerOutcome = run_method_worker(
                    method=method,
                    comparison_input=comp_input,
                    global_deadline_seconds=manifest["resource_limits"]["global_deadline_seconds"],
                    candidate_cutoff_soft_seconds=manifest["resource_limits"]["candidate_cutoff_soft_seconds"],
                    ranking_time_limit=manifest["resource_limits"]["ranking_time_limit_seconds"],
                    b1_checkpoint_path=b1_checkpoint if "b1" in method else None,
                    b1_checkpoint_sha256=b1_checkpoint_sha256 if "b1" in method else None,
                    queue_control_seed=control_seed,
                    oracle_trailing_keys=oracle_keys,
                    ddt_7r_path=resolved_paths["ddt_7r"],
                    tool_8r_binary_path=resolved_paths["queried_row_8r_binary"],
                    bao_checkpoint_dir=resolved_paths["bao_vv_directory"],
                    tf_python_path=resolved_paths["tensorflow_python"],
                    use_parent_watchdog=True,
                )
            except Exception as exc:
                outer_exception = f"{type(exc).__name__}: {exc}"
                outcome = MethodWorkerOutcome(
                    method=method, outcome="exception",
                    complete_online_wall_seconds=time.perf_counter() - t0,
                    ranking_wall_seconds=0.0, solving_wall_seconds=0.0,
                    verification_wall_seconds=0.0, candidates_tested_count=0,
                    candidates_remaining_count=0, recovered_master_key=None,
                    verified_on_fresh_pairs=False, candidate_attempts=(),
                    peak_rss_gib=0.0, error_message=outer_exception,
                )
            elapsed = time.perf_counter() - t0

            # Audit verification is separate from the online worker's fresh check.
            verification_diagnostic = None
            if outcome.outcome == "verified_recovery":
                try:
                    if outcome.recovered_master_key is None:
                        raise ValueError("Claimed success missing recovered key")
                    v_res = verify_simon_master_key(
                        master_key=outcome.recovered_master_key, known_pairs=inst.known_pairs,
                        verification_pairs=inst.verification_pairs[:2], rounds=10,
                    )
                    if outcome.recovered_master_key != inst.master_key or not v_res.verified or not verify_independent_fresh(outcome.recovered_master_key, inst):
                        raise ValueError("Claimed success failed harness key/independent fresh verification")
                except Exception as exc:
                    verification_diagnostic = {"claimed_master_key": outcome.recovered_master_key, "error": f"{type(exc).__name__}: {exc}"}
                    outcome = replace(outcome, outcome="verification_failure", recovered_master_key=None, verified_on_fresh_pairs=False, error_message=verification_diagnostic["error"])

            rec = outcome.to_dict()
            if verification_diagnostic is not None:
                rec["harness_verification_diagnostic"] = verification_diagnostic
            if outer_exception is not None:
                rec["telemetry_status"] = "UNAVAILABLE_OUTER_WORKER_EXCEPTION"
                for field in ("ranking_wall_seconds", "solving_wall_seconds", "verification_wall_seconds"):
                    rec[field] = None
            rec.update(instance_id=inst.instance_id, instance_index=inst_idx, recorded_at=datetime.now(UTC).isoformat())
            output_records.append(rec)
            completed_cells[cell] = rec

            # Stream save after every cell
            _save_ledger_atomic(args.output, output_records, execution_identity, evidence_class, run_provenance)
            if args.log_dir is not None:
                args.log_dir.mkdir(parents=True, exist_ok=True)
                with (args.log_dir / "confirmation_stream.jsonl").open("a") as log:
                    log.write(json.dumps({key: rec[key] for key in (
                        "instance_id", "method", "outcome", "complete_online_wall_seconds", "recorded_at",
                    )}) + "\n")
                    log.flush()
                    os.fsync(log.fileno())
            print(f"[+] Completed {cell}: outcome={outcome.outcome} ({elapsed:.2f}s)")
            if outcome.outcome in ("exception", "verification_failure", "sat_unverified"):
                print("[!] Infrastructure/integrity issue persisted; suspension requires reviewer inspection.", file=sys.stderr)
                return 8

    # 6. Grid Completion Validation: require exactly N x 9 validated cells
    expected_total_cells = len(instances) * len(methods)
    if len(output_records) != expected_total_cells:
        print(
            f"[!] LEDGER INCOMPLETE: expected exactly {expected_total_cells} cells ({len(instances)} instances x {len(methods)} methods), "
            f"got {len(output_records)} records.",
            file=sys.stderr,
        )
        return 7

    observed_cells = {(r["instance_id"], r["method"]) for r in output_records}
    all_expected_cells = {(inst.instance_id, m) for inst in instances for m in methods}
    if observed_cells != all_expected_cells:
        print("[!] LEDGER MISMATCH: recorded cells do not match expected confirmation grid!", file=sys.stderr)
        return 7

    validate_and_filter_resume(args.output, execution_identity, instances, methods, resume_flag=True)

    print(f"[+] All {len(output_records)} cells ({len(instances)} x {len(methods)}) completed and verified. Ledger persisted to {args.output}.")
    return 0


def _save_ledger_atomic(
    path: Path,
    records: list[dict[str, Any]],
    execution_identity: dict[str, Any],
    evidence_class: str,
    run_provenance: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    data = {
        "evidence_class": evidence_class,
        "run_provenance": run_provenance or {},
        "execution_identity": execution_identity,
        "completed_count": len(records),
        "updated_at": datetime.now(UTC).isoformat(),
        "records": records,
    }
    raw = json.dumps(data, indent=2)
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)

    sha_path = path.parent / f"{path.name}.sha256"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    tmp_sha_path = path.parent / f"{path.name}.sha256.tmp"
    with open(tmp_sha_path, "w", encoding="utf-8") as f:
        f.write(f"{digest}  {path.name}\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_sha_path, sha_path)


if __name__ == "__main__":
    sys.exit(main())
