"""An immutable old execution prefix for a narrowly reviewed AC-stop resume."""

from __future__ import annotations

import copy
import re
from pathlib import Path

from .guard import (
    canonical_bytes,
    checked_path,
    read_checked_json,
    sha256_file,
    validate_authorization,
    validate_manifest,
    validate_record,
)

_ALLOWED_SOURCE_AMENDMENTS = {
    "source:src/neural_sat_speck/backend/runner.py",
    "source:src/neural_sat_speck/backend/evidence.py",
    "source:scripts/analyze_backend_comparison.py",
}
_NEW_SOURCE = "source:src/neural_sat_speck/backend/reviewed_resume.py"


def load_prefix(manifest: dict, root: Path, output: Path) -> dict | None:
    plan = manifest.get("reviewed_power_resume")
    if plan is None:
        return None
    if (
        not isinstance(plan, dict)
        or plan.get("schema") != "backend_ac_resume_v1"
        or plan.get("status") != "AUTHOR_APPROVED"
        or not re.fullmatch(r"D-[0-9]+", str(plan.get("decision", "")))
        or plan.get("output_path") != str(output.resolve())
    ):
        raise ValueError("Unapproved or mismatched AC-resume plan")
    prior_root = Path(plan["prior_source_root"])
    prior_path = checked_path(root, plan["prior_manifest"])
    prior = read_checked_json(prior_path)
    if "reviewed_power_resume" in prior:
        raise ValueError("Nested continuation needs a separate design review")
    prior_paths = validate_manifest(prior, prior_root)
    auth = checked_path(root, plan["prior_authorization"])
    validate_authorization(auth, sha256_file(prior_path), prior_root, output=output)
    archived_path = checked_path(root, plan["stopped_ledger"])
    archived = read_checked_json(archived_path)
    if (
        archived.get("schema") != "backend_ledger_v1"
        or archived.get("status") != "safety_stop"
        or archived.get("manifest_sha256") != sha256_file(prior_path)
        or not isinstance(archived.get("starts"), dict)
        or not isinstance(archived.get("records"), dict)
        or not archived["records"]
        or set(archived["records"]) != set(archived["starts"])
    ):
        raise ValueError("Incomplete or mismatched stopped execution prefix")
    cells = manifest["cells"]
    expected = {c["cell_id"] for c in cells[: len(archived["records"])]}
    if set(archived["records"]) != expected:
        raise ValueError("Stopped cells are not the exact fixed schedule prefix")
    if canonical_bytes(prior["cells"]) != canonical_bytes(cells):
        raise ValueError("Scientific cell matrix changed at resume")
    # Guard validation establishes the fixed scientific fields. All non-source
    # identities and scientific source bytes must additionally remain identical.
    for key, identity in prior["files"].items():
        if key not in manifest["files"]:
            raise ValueError("Resume dropped a prior source/runtime identity")
        current = manifest["files"][key]
        if key.startswith("source:"):
            if key not in _ALLOWED_SOURCE_AMENDMENTS and identity != current:
                raise ValueError(f"Scientific source drift at resume: {key}")
        elif identity != current:
            raise ValueError(f"Runtime/cohort/artifact drift at resume: {key}")
    extra = set(manifest["files"]) - set(prior["files"])
    if extra - {_NEW_SOURCE}:
        raise ValueError("Unexpected additional resume identities")
    stop_ids = plan.get("power_stop_cells")
    alarms = {
        ident
        for ident, row in archived["records"].items()
        if row["outcome"] not in {"recovered", "global_deadline", "unknown"}
    }
    if (
        not isinstance(stop_ids, list)
        or len(stop_ids) != 1
        or type(stop_ids[0]) is not str
        or alarms != set(stop_ids)
    ):
        raise ValueError("Resume review must name the sole recorded AC-stop alarm")
    by_id = {c["cell_id"]: c for c in cells}
    for ident, row in archived["records"].items():
        validate_record(row, by_id[ident])
        if row.get("invocation") != archived["starts"][ident].get("invocation"):
            raise ValueError("Prior start/result invocation mismatch")
    row = archived["records"][stop_ids[0]]
    if (
        row["outcome"] != "environment_rejected"
        or row.get("error") != "Observed power/sleep policy violation"
        or row.get("guardian_healthy") is not True
        or row.get("process_group_absent") is not True
        or row.get("recovered_key") is not None
        or any(
            row[f] is not False
            for f in ("verified_known", "verified_fresh", "assigned_key_match")
        )
    ):
        raise ValueError("Prior alarm is not a cleanly stopped AC interruption")
    power_path = Path(row["worker_directory"]) / "power_latest.json"
    if row.get("raw_artifacts", {}).get(power_path.name) != sha256_file(power_path):
        raise ValueError("AC-stop sample is not sealed in the original evidence")
    power = read_checked_json(power_path)
    if (
        power.get("accepted") is not False
        or power.get("ac_power_observed") is not False
        or power.get("sleep_assertion_observed") is not True
        or type(power.get("assertions_returncode")) is not int
        or power["assertions_returncode"] != 0
        or type(power.get("power_returncode")) is not int
        or power["power_returncode"] != 0
        or "Battery Power" not in power.get("raw_power", "")
    ):
        raise ValueError("Original sample does not establish the recorded AC loss")
    return {
        "manifest": prior,
        "paths": prior_paths,
        "ledger": archived,
        "ledger_sha256": sha256_file(archived_path),
        "power_stop_cells": set(stop_ids),
        "decision": plan["decision"],
    }


def validate_preserved_prefix(ledger: dict, prefix: dict | None) -> None:
    if prefix is None:
        return
    for section in ("starts", "records"):
        for ident, original in prefix["ledger"][section].items():
            if canonical_bytes(ledger.get(section, {}).get(ident)) != canonical_bytes(
                original
            ):
                raise ValueError(f"Resumed ledger changed a prior {section} row")


def original_stop_only(ledger: dict, prefix: dict | None) -> bool:
    if prefix is None:
        return False
    validate_preserved_prefix(ledger, prefix)
    if canonical_bytes(ledger.get("environment_stops")) != canonical_bytes(
        prefix["ledger"].get("environment_stops")
    ):
        return False
    return all(
        set(ledger[s]) == set(prefix["ledger"][s]) for s in ("starts", "records")
    )


def transition_identity(ledger: dict, prefix: dict | None, manifest_sha: str) -> dict:
    """Transition only the byte-archived stop; preserve every prior cell."""
    if prefix is None or ledger.get("manifest_sha256") == manifest_sha:
        return ledger
    if canonical_bytes(ledger) != canonical_bytes(
        prefix["ledger"]
    ) or not original_stop_only(ledger, prefix):
        raise ValueError(
            "Only the unchanged archived stop can change manifest identity"
        )
    transitioned = copy.deepcopy(ledger)
    transitioned["manifest_sha256"] = manifest_sha
    transitioned.setdefault("resume_reviews", []).append(
        expected_review(prefix, manifest_sha)
    )
    return transitioned


def expected_review(prefix: dict, manifest_sha: str) -> dict:
    return {
        "decision": prefix["decision"],
        "stopped_ledger_sha256": prefix["ledger_sha256"],
        "prior_manifest_sha256": prefix["ledger"]["manifest_sha256"],
        "continued_manifest_sha256": manifest_sha,
        "retained_power_stop_cells": sorted(prefix["power_stop_cells"]),
    }


def validate_review_history(
    ledger: dict, prefix: dict | None, manifest_sha: str
) -> None:
    expected = [] if prefix is None else [expected_review(prefix, manifest_sha)]
    if canonical_bytes(ledger.get("resume_reviews", [])) != canonical_bytes(expected):
        raise ValueError(
            "Resume review history does not match approved execution identities"
        )


def provenance(
    ident: str, manifest: dict, paths: dict, prefix: dict | None
) -> tuple[dict, dict]:
    if prefix is not None and ident in prefix["ledger"]["records"]:
        return prefix["manifest"], prefix["paths"]
    return manifest, paths
