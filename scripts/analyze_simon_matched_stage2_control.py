"""Validate all assigned closures before an opened-data Stage 2 truth join."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_sat_speck.backend.guard import atomic_json, read_checked_json  # noqa: E402
from neural_sat_speck.followups.stage2_control import (  # noqa: E402
    analyze_cells,
    assemble_queue,
    fixed_jobs,
    ordered_hash,
    public_case,
    streaming_sha256,
    validate_manifest,
    validate_public_case,
)
from neural_sat_speck.followups.stage2_receipts import (  # noqa: E402
    validate_resource_receipts,
)
from neural_sat_speck.followups.stage2_runner import (  # noqa: E402
    normalize_job,
    validate_execution_identity,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original-ledger", type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_name(args.output.name + ".sha256").exists():
        raise FileExistsError("Version new analyses; never overwrite evidence")
    manifest = read_checked_json(args.manifest)
    paths = validate_manifest(manifest, ROOT)
    ledger = read_checked_json(args.ledger)
    if (
        manifest["mode"] != "full"
        or ledger.get("mode") != "full"
        or ledger.get("status") not in {"COMPLETE", "STOPPED_ALARM"}
        or ledger.get("source_commit") != manifest["source_commit"]
        or ledger.get("manifest_sha256") != streaming_sha256(args.manifest)
        or ledger.get("profile") != manifest["profile"]
        or set(ledger.get("records", {})) != {j["job_id"] for j in fixed_jobs()}
    ):
        raise ValueError("Analysis requires a closed full grid with exact identity")
    sealed = "sealed_from" in ledger
    if sealed:
        if args.original_ledger is None:
            raise ValueError("Sealed analysis requires its preserved original ledger")
        from neural_sat_speck.followups.stage2_seal import seal

        # Reconstruct the seal from checked historical evidence. Only completion
        # labels change with the audit time; no rank/cost measurement is repeated.
        with tempfile.TemporaryDirectory(
            dir=args.output.parent, prefix="stage2_seal_check_"
        ) as temporary:
            reconstructed = seal(
                args.manifest,
                args.original_ledger,
                Path(temporary) / "checked_seal.json",
                ROOT,
            )

        def stable(value):
            if isinstance(value, dict):
                return {
                    key: stable(item)
                    for key, item in value.items()
                    if key != "completed_utc"
                }
            if isinstance(value, list):
                return [stable(item) for item in value]
            return value

        if stable(reconstructed) != stable(ledger):
            raise ValueError("Sealed ledger differs from checked historical evidence")
    elif args.original_ledger is not None:
        raise ValueError("Original-ledger argument applies only to sealed evidence")
    parents = {}
    public_cases = {}
    for job in fixed_jobs():
        row = ledger["records"][job["job_id"]]
        if row.get("guardian_receipt") is not None:
            directory = Path(row["directory"])
            if not sealed:
                path = directory / "supervisor_result.json"
                if not path.exists() and row["outcome"] == "campaign_timeout":
                    path = directory / "supervisor_alarm.json"
                if read_checked_json(path) != row:
                    raise ValueError("Ledger and invocation artifact differ")
            payload = read_checked_json(directory / "payload.json")
            case = payload["public_case"]
            validate_public_case(case)
            previous = public_cases.setdefault(job["key_index"], case)
            if case != previous:
                raise ValueError("Compared scorers received different ciphertext pairs")
            guardian = read_checked_json(directory / "guardian.json")
            stopped = (
                row["outcome"]
                if row["outcome"]
                in {"rss_stop", "environment_alarm", "interrupted", "campaign_timeout"}
                else None
            )
            normalized = normalize_job(
                job, directory, guardian["job_started_monotonic"], stopped
            )
            for name in (
                "calls",
                "load_wall_seconds",
                "worker_wall_seconds",
                "accepted_events",
                "guardian_receipt",
            ):
                if normalized[name] != row[name]:
                    raise ValueError("Ledger differs from checked raw phase artifacts")
            selected = [] if job["operation"] == "stage1" else parents[job["key_index"]]
            validate_execution_identity(
                manifest,
                row,
                case,
                selected,
                ledger["campaign_deadline_monotonic"],
            )
            validate_resource_receipts(
                manifest,
                {**row, "sleep_assertion": ledger["sleep_assertion"]},
                require_worker_sample=row["outcome"] == "completed",
            )
            for width_key, measured in row["queue_measurements"].items():
                width = int(width_key)
                orders = {
                    call["ordinal"]: call["ordered_words"]
                    for call in row["calls"]
                    if call["ordinal"] < width
                }
                queue = assemble_queue(selected, orders, width)
                persisted = read_checked_json(directory / f"queue_m{width}.json")
                if persisted.get("candidate_tuples") != [
                    list(value) for value in queue
                ] or measured["queue_sha256"] != ordered_hash(queue):
                    raise ValueError(
                        "Queue artifact differs from its constituent ranks"
                    )
        elif row.get("dispatched") is True and not sealed:
            raise ValueError(
                "Integrity alarm lacks validated evidence: independent review required"
            )
        if job["operation"] == "stage1" and row["calls"]:
            parents[job["key_index"]] = row["calls"][0]["ordered_words"][:8]
    dataset = json.loads(paths["files"]["cohort"].read_bytes())
    for job in fixed_jobs():
        if job["key_index"] in public_cases and public_case(dataset, job) != public_cases[
            job["key_index"]
        ]:
            raise ValueError("Worker ciphertext projection differs from pinned cohort")
    analysis = analyze_cells(dataset, list(ledger["records"].values()))
    atomic_json(
        args.output,
        {
            "schema": "stage2_control_analysis_v1",
            "manifest_sha256": streaming_sha256(args.manifest),
            "ledger_sha256": streaming_sha256(args.ledger),
            "source_commit": manifest["source_commit"],
            "analysis": analysis,
            "limits": manifest["limits"],
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()
