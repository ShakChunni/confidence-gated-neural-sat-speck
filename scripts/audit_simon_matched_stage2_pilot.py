"""Check real pilot contracts and receipts without a true-key rank join."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_sat_speck.backend.guard import atomic_json, read_checked_json  # noqa: E402
from neural_sat_speck.followups.stage2_control import (  # noqa: E402
    PROFILE,
    assemble_queue,
    fixed_jobs,
    ordered_hash,
    public_case,
    streaming_sha256,
    validate_manifest,
)
from neural_sat_speck.followups.stage2_receipts import (  # noqa: E402
    validate_resource_receipts,
)
from neural_sat_speck.followups.stage2_runner import (  # noqa: E402
    normalize_job,
    validate_execution_identity,
)


def audit(manifest_path: Path, ledger_path: Path) -> dict:
    manifest = read_checked_json(manifest_path)
    paths = validate_manifest(manifest, ROOT)
    ledger = read_checked_json(ledger_path)
    jobs = fixed_jobs()[:3]
    if (
        manifest["mode"] != "pilot"
        or ledger.get("mode") != "pilot"
        or ledger.get("status") != "COMPLETE"
        or ledger.get("manifest_sha256") != streaming_sha256(manifest_path)
        or ledger.get("profile") != PROFILE
        or ledger.get("source_commit") != manifest["source_commit"]
        or set(ledger.get("records", {})) != {j["job_id"] for j in jobs}
        or set(ledger.get("starts", {})) != {j["job_id"] for j in jobs}
    ):
        raise ValueError("Pilot identity, closure or grid differs from frozen profile")
    dataset = json.loads(paths["files"]["cohort"].read_bytes())
    sidecars, count = 0, 0
    parents = []
    for job in jobs:
        row = ledger["records"][job["job_id"]]
        directory = Path(row["directory"])
        start = ledger["starts"][job["job_id"]]
        if (
            start["directory"] != str(directory)
            or start["invocation"] != directory.name
            or row["outcome"] != "completed"
            or row.get("dispatched") is not True
        ):
            raise ValueError(
                "Pilot worker did not mechanically complete its assigned path"
            )
        for sidecar in directory.rglob("*.json.sha256"):
            read_checked_json(sidecar.with_name(sidecar.name.removesuffix(".sha256")))
            sidecars += 1
        guardian = read_checked_json(directory / "guardian.json")
        normalized = normalize_job(
            job, directory, guardian["job_started_monotonic"], None
        )
        for key in (
            "outcome",
            "calls",
            "load_wall_seconds",
            "worker_wall_seconds",
            "accepted_events",
            "guardian_receipt",
        ):
            if normalized[key] != row[key]:
                raise ValueError(
                    "Ledger differs from independently normalized pilot artifacts"
                )
        validate_execution_identity(
            manifest,
            row,
            public_case(dataset, job),
            parents,
            ledger["campaign_deadline_monotonic"],
        )
        validate_resource_receipts(
            manifest, {**row, "sleep_assertion": ledger["sleep_assertion"]}
        )
        if read_checked_json(directory / "supervisor_result.json") != row:
            raise ValueError(
                "Pilot ledger differs from its persisted supervisor result"
            )
        final = read_checked_json(directory / "worker_result.json")
        if (
            final.get("status") != "complete"
            or final.get("job") != job
            or final.get("invocation") != directory.name
            or final.get("parents") != parents
            or final.get("completed_call_count") != len(row["calls"])
            or final.get("telemetry", {}).get("git_commit") != manifest["source_commit"]
        ):
            raise ValueError("Pilot worker final metadata is not source/job-bound")
        expected = 1 if job["operation"] == "stage1" else 8
        if len(row["calls"]) != expected:
            raise ValueError("Pilot did not exercise all 17 actual rank calls")
        count += len(row["calls"])
        if job["operation"] == "stage1":
            parents = row["calls"][0]["ordered_words"][:8]
        else:
            for width in (1, 8):
                calls = {
                    call["ordinal"]: call["ordered_words"]
                    for call in row["calls"]
                    if call["ordinal"] < width
                }
                queue = assemble_queue(parents, calls, width)
                persisted = read_checked_json(directory / f"queue_m{width}.json")
                if persisted.get("candidate_tuples") != [
                    list(value) for value in queue
                ] or row["queue_measurements"][str(width)][
                    "queue_sha256"
                ] != ordered_hash(queue):
                    raise ValueError(
                        "Pilot queue differs from shared-parent/scorer results"
                    )
    if count != 17:
        raise ValueError("Pilot call count differs")
    return {
        "schema": "stage2_control_pilot_report_v1",
        "status": "PASS",
        "mechanical_pass": True,
        "source_commit": manifest["source_commit"],
        "profile": PROFILE,
        "profile_sha256": ordered_hash(PROFILE),
        "manifest_sha256": streaming_sha256(manifest_path),
        "ledger_sha256": streaming_sha256(ledger_path),
        "jobs": [job["job_id"] for job in jobs],
        "completed_calls": count,
        "completed_workers": 3,
        "all_groups_absent": True,
        "truth_join_performed": False,
        "raw_sidecars_checked": sidecars,
        "physical_worker_seconds": sum(
            row["worker_wall_seconds"] for row in ledger["records"].values()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_name(args.output.name + ".sha256").exists():
        raise FileExistsError("Never overwrite a pilot audit")
    report = audit(args.manifest, args.ledger)
    atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "completed_calls": report["completed_calls"],
                "raw_sidecars_checked": report["raw_sidecars_checked"],
            }
        )
    )


if __name__ == "__main__":
    main()
