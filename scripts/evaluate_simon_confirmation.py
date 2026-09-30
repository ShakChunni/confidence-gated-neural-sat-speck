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
import sys
import time
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
    derive_confirmation_queue_seed,
    file_sha256,
    validate_and_filter_resume,
    validate_confirmation_manifest,
    verify_disjoint_family_certificate,
    verify_runtime_provenance,
)
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
            if manifest_dict.get("protocol_id") != "simon32_64_r10_unseen_confirmation_v1":
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
            seed=1234567,
            split_role="confirmation",
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
            comp_input = SimonComparisonInput(
                instance_id=inst.instance_id,
                ciphertext_pairs=tuple((p.ciphertext0, p.ciphertext1) for p in inst.pairs),
                known_pairs=inst.known_pairs,
                verification_pairs=inst.verification_pairs[:2],
                rounds=10,
                manifest_seed=manifest["seed_governance"]["seed"],
            )

            oracle_keys = (
                (inst.master_key[2], inst.master_key[3])
                if method.startswith("oracle_")
                else None
            )

            random_seed = (
                derive_confirmation_queue_seed(control_seed, inst.instance_id, method)
                if method.startswith("random_control_")
                else None
            )

            t0 = time.perf_counter()
            outcome: MethodWorkerOutcome = run_method_worker(
                method=method,
                comparison_input=comp_input,
                global_deadline_seconds=manifest["resource_limits"]["global_deadline_seconds"],
                candidate_cutoff_soft_seconds=manifest["resource_limits"]["candidate_cutoff_soft_seconds"],
                ranking_time_limit=manifest["resource_limits"]["ranking_time_limit_seconds"],
                b1_checkpoint_path=b1_checkpoint if "b1" in method else None,
                b1_checkpoint_sha256=b1_checkpoint_sha256 if "b1" in method else None,
                random_seed=random_seed,
                oracle_trailing_keys=oracle_keys,
                use_parent_watchdog=True,
            )
            elapsed = time.perf_counter() - t0

            # Independent fresh-verification check on claimed recoveries
            if outcome.outcome == "verified_recovery":
                if outcome.recovered_master_key is None:
                    raise RuntimeError(f"Claimed success for {cell} missing recovered key")
                v_res = verify_simon_master_key(
                    master_key=outcome.recovered_master_key,
                    known_pairs=inst.known_pairs,
                    verification_pairs=inst.verification_pairs[:2],
                    rounds=10,
                )
                if not v_res.verified:
                    raise RuntimeError(f"Claimed success for {cell} failed harness independent verification")

            rec = {
                "instance_id": inst.instance_id,
                "instance_index": inst_idx,
                "method": method,
                "outcome": outcome.outcome,
                "complete_online_wall_seconds": outcome.complete_online_wall_seconds,
                "ranking_wall_seconds": outcome.ranking_wall_seconds,
                "solving_wall_seconds": outcome.solving_wall_seconds,
                "verification_wall_seconds": outcome.verification_wall_seconds,
                "candidates_tested_count": outcome.candidates_tested_count,
                "candidates_remaining_count": outcome.candidates_remaining_count,
                "recovered_master_key": list(outcome.recovered_master_key) if outcome.recovered_master_key else None,
                "verified_on_fresh_pairs": outcome.verified_on_fresh_pairs,
                "peak_rss_gib": outcome.peak_rss_gib,
                "full_pairs_used": outcome.full_pairs_used,
                "error_message": outcome.error_message,
                "candidate_attempts": [a.to_dict() for a in outcome.candidate_attempts],
                "recorded_at": datetime.now(UTC).isoformat(),
            }
            output_records.append(rec)
            completed_cells[cell] = rec

            # Stream save after every cell
            _save_ledger_atomic(args.output, output_records, execution_identity, evidence_class)
            print(f"[+] Completed {cell}: outcome={outcome.outcome} ({elapsed:.2f}s)")

    print(f"[+] All {len(output_records)} cells completed. Ledger persisted to {args.output}.")
    return 0


def _save_ledger_atomic(
    path: Path,
    records: list[dict[str, Any]],
    execution_identity: dict[str, Any],
    evidence_class: str,
) -> None:
    tmp_path = path.with_suffix(".tmp")
    data = {
        "evidence_class": evidence_class,
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

    sha_path = path.with_suffix(".json.sha256")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    with open(sha_path, "w", encoding="utf-8") as f:
        f.write(f"{digest}  {path.name}\n")


if __name__ == "__main__":
    sys.exit(main())
