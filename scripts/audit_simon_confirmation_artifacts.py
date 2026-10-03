#!/usr/bin/env python3
"""Read-only confirmation integrity audit; never dispatches attack workers.

The executed ledger stays immutable. Recovered keys are checked against the
clean-room cipher on every attack and fresh observation, not just stored flags.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "ledger", "dataset", "certificate", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    inputs = (args.manifest, args.ledger, args.dataset, args.certificate)
    if args.output.resolve() in {path.resolve() for path in inputs}:
        parser.error("The audit receipt must not overwrite an input artifact")

    from neural_sat_speck.simon.confirmation_profile import THREAD_ENVIRONMENT

    # Reproduce the frozen numerical environment for identity probes only.
    # This process does not solve formulas, rank candidates, or generate data.
    os.environ.update(THREAD_ENVIRONMENT)
    from neural_sat_speck.simon.attack_instance import load_simon_attack_instances
    from neural_sat_speck.simon.confirmation_guard import (
        ALL_CONFIRMATION_METHODS,
        compute_execution_identity,
        file_sha256,
        validate_and_filter_resume,
        validate_confirmation_manifest,
        verify_disjoint_family_certificate,
        verify_runtime_provenance,
    )

    manifest = validate_confirmation_manifest(args.manifest)
    print("Checking frozen runtime/source/resource identities...", flush=True)
    paths = verify_runtime_provenance(manifest)
    instances = load_simon_attack_instances(args.dataset)
    verify_disjoint_family_certificate(manifest, instances, args.certificate)
    dataset_hash = file_sha256(args.dataset)
    sidecar = args.dataset.with_name(args.dataset.name + ".sha256")
    if sidecar.read_text().split()[0] != dataset_hash:
        raise ValueError("Dataset sidecar mismatch")
    completed, pending = validate_and_filter_resume(
        args.ledger,
        compute_execution_identity(manifest, dataset_hash),
        instances,
        ALL_CONFIRMATION_METHODS,
    )
    expected = len(instances) * len(ALL_CONFIRMATION_METHODS)
    if pending or len(completed) != expected:
        raise ValueError("Confirmation grid is incomplete")

    reference = ROOT / "tests/fixtures/simon_reference_oracle.py"
    spec = importlib.util.spec_from_file_location("clean_room_simon", reference)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    encrypt = oracle.oracle_simon32_64_encrypt
    schedule = oracle.oracle_simon32_64_key_schedule

    dataset_checks = 0
    success_known_checks = 0
    success_fresh_checks = 0
    by_id = {instance.instance_id: instance for instance in instances}
    for inst in instances:
        if len(inst.pairs) != 64 or len(inst.verification_pairs) != 2:
            raise ValueError("Observation count differs from frozen profile")
        plaintexts = [pt for pt, _ in (*inst.known_pairs, *inst.verification_pairs)]
        if len(set(plaintexts)) != len(plaintexts):
            raise ValueError("Attack/fresh plaintexts are not distinct")
        for pt, ct in (*inst.known_pairs, *inst.verification_pairs):
            if encrypt(pt, inst.master_key, rounds=inst.rounds) != ct:
                raise ValueError("Dataset disagrees with independent cipher")
            dataset_checks += 1

    methods = {}
    successful = 0
    for method in ALL_CONFIRMATION_METHODS:
        rows = [row for (_, arm), row in completed.items() if arm == method]
        if len(rows) != len(instances):
            raise ValueError("Method denominator differs from cohort size")
        outcomes = Counter(row["outcome"] for row in rows)
        for row in rows:
            inst = by_id[row["instance_id"]]
            if method.startswith("oracle_"):
                true_keys = schedule(inst.master_key, rounds=inst.rounds)[-2:]
                if row["full_queue"] != [list(true_keys)]:
                    raise ValueError("Oracle did not receive the independent true tuple")
            elapsed = row["complete_online_wall_seconds"]
            components = sum(row[name] for name in (
                "ranking_wall_seconds", "solving_wall_seconds", "verification_wall_seconds"
            ))
            if components > elapsed + 0.05:
                raise ValueError("Component accounting exceeds complete online time")
            if row["outcome"] != "verified_recovery":
                continue
            successful += 1
            key = tuple(row["recovered_master_key"])
            if key != inst.master_key or not 0 < elapsed <= 300:
                raise ValueError("Invalid exact recovery or deadline")
            keys = schedule(key, rounds=inst.rounds)
            true_tuple = (keys[-2], keys[-1])
            attempts = row["candidate_attempts"]
            if not method.startswith("sat_only_"):
                if not attempts or attempts[-1]["status"] != "sat":
                    raise ValueError("Recovery lacks a completed SAT candidate")
                if (attempts[-1]["candidate_k8"], attempts[-1]["candidate_k9"]) != true_tuple:
                    raise ValueError("Recovered candidate disagrees with clean-room key schedule")
            for pt, ct in inst.known_pairs:
                if encrypt(pt, key, rounds=inst.rounds) != ct:
                    raise ValueError("Recovery fails a known observation")
                success_known_checks += 1
            for pt, ct in inst.verification_pairs:
                if encrypt(pt, key, rounds=inst.rounds) != ct:
                    raise ValueError("Recovery fails a fresh observation")
                success_fresh_checks += 1
        methods[method] = {"records": len(rows), "outcomes": dict(sorted(outcomes.items()))}

    ledger = json.loads(args.ledger.read_text())
    artifacts = {str(path.resolve().relative_to(ROOT)): file_sha256(path) for path in inputs}
    artifacts[str(reference.relative_to(ROOT))] = file_sha256(reference)
    artifacts[str(Path(__file__).relative_to(ROOT))] = file_sha256(Path(__file__))
    receipt = {
        "audit_version": 1,
        "audited_at_utc": datetime.now(UTC).isoformat(),
        "scope": "Integrity/cipher verification only; scheduling, seed-freeze chronology and scientific interpretation require separate review.",
        "records": len(completed),
        "pending": len(pending),
        "unique_keys": len({inst.master_key for inst in instances}),
        "independently_verified_recoveries": successful,
        "independent_dataset_observation_checks": dataset_checks,
        "independent_recovery_known_observation_checks": success_known_checks,
        "independent_recovery_fresh_observation_checks": success_fresh_checks,
        "methods": methods,
        "artifact_sha256": artifacts,
        "verified_resource_paths": {name: str(path) for name, path in paths.items()},
        "run_provenance": ledger.get("run_provenance"),
        "ledger_updated_at": ledger.get("updated_at"),
        "status": "PASS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "PASS", "records": len(completed), "recoveries": successful,
                      "known_checks": success_known_checks, "fresh_checks": success_fresh_checks}))


if __name__ == "__main__":
    main()
