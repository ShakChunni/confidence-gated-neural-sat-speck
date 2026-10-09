"""Audit D-057's authorized two-segment study without rewriting either ledger."""

from __future__ import annotations

from pathlib import Path

from neural_sat_speck.backend.artifacts import cipher_functions, verify_key
from neural_sat_speck.backend.guard import read_checked_json, sha256_file
from neural_sat_speck.followups.conditioning_control import digest
from neural_sat_speck.followups.ranking_depth_audit import require, validate_cell
from neural_sat_speck.followups.ranking_depth_control import (
    load_manifest,
    optional_terminal_json,
    verify_cases,
)
from neural_sat_speck.followups.ranking_depth_recovery import (
    ORIGINAL_TERMINAL_RECORDS,
    _check_unlocked,
    _load_original,
    _validate_source_rebase,
)
from neural_sat_speck.followups.stage2_receipts import _validate_sleep_assertion

ORIGINAL_LEDGER_SHA256 = (
    "ac53ea48efeb18b9c765d3e40b67254999a2071ed3aaf45ab9e10e2e4839937d"
)
CONTINUATION_MANIFEST_SHA256 = (
    "eae9919af8f0b8280c8beaf830b2a13ad611a8f41e24d5c54b4ed0d89e7cb192"
)
CONTINUATION_SOURCE_COMMIT = "9a00e05de3b3cf7e48dddfb1da4939d8c6116f5a"


def require_assigned_key(checked: dict) -> None:
    require(
        checked.get("assigned_key_match") is True,
        "Claimed exact recovery does not match the assigned master key",
    )


def validate_segment_layout(original: dict, continuation: dict, old: dict, new: dict):
    """Check the assigned grid before exposing truth to the scientific analysis."""
    require(
        all(original[k] == continuation[k] for k in ("profile", "cells", "cases")),
        "Scientific profile, case or schedule drift between segments",
    )
    cells = original["cells"]
    ids = [c["cell_id"] for c in cells]
    prefix, suffix = ids[:ORIGINAL_TERMINAL_RECORDS], ids[ORIGINAL_TERMINAL_RECORDS:]
    require(len(ids) == len(set(ids)) == 256, "Duplicate or incomplete assigned grid")
    require(
        old.get("status") == "INTERRUPTED"
        and old.get("execution_order") == ids[: ORIGINAL_TERMINAL_RECORDS + 1]
        and set(old["starts"]) == set(ids[: ORIGINAL_TERMINAL_RECORDS + 1])
        and set(old["records"]) == set(prefix),
        "Original segment is not the preserved terminal prefix",
    )
    require(
        new.get("schema") == "backend_ranking_depth_d056_continuation_results_v1"
        and new.get("status") == "COMPLETE"
        and new.get("execution_order") == suffix
        and set(new["records"]) == set(new["starts"]) == set(suffix)
        and new.get("cells") == cells[ORIGINAL_TERMINAL_RECORDS:]
        and continuation["recovery"].get("suffix_cells")
        == cells[ORIGINAL_TERMINAL_RECORDS:]
        and continuation["recovery"].get("suffix_sha256")
        == digest(cells[ORIGINAL_TERMINAL_RECORDS:]),
        "Incomplete, repeated or reordered continuation suffix",
    )
    seen, previous_end = set(), None
    for ordinal, cell in enumerate(cells):
        ledger = old if ordinal < ORIGINAL_TERMINAL_RECORDS else new
        ident = cell["cell_id"]
        row, start = ledger["records"][ident], ledger["starts"][ident]
        directory = Path(row["directory"])
        require(
            row["cell"] == cell
            and start["directory"] == str(directory)
            and start["invocation_id"] == directory.name
            and directory.name not in seen,
            "Cell/start identity mismatch or duplicate invocation",
        )
        require(
            previous_end is None or row["started_monotonic"] >= previous_end,
            "Overlapping scientific cells across segments",
        )
        seen.add(directory.name)
        previous_end = row["guardian_receipt"]["cleanup_completed_monotonic"]


def audit_continuation(ledger_path: Path, manifest_path: Path) -> dict:
    """Validate each row against its actual source; retain the unavailable attempt."""
    ledger_path, manifest_path = ledger_path.absolute(), manifest_path.absolute()
    new = read_checked_json(ledger_path)
    continuation = read_checked_json(manifest_path)
    recovery = continuation["recovery"]
    require(
        recovery.get("ledger_sha256") == ORIGINAL_LEDGER_SHA256
        and sha256_file(manifest_path) == CONTINUATION_MANIFEST_SHA256
        and continuation.get("source_commit") == CONTINUATION_SOURCE_COMMIT,
        "Segment identities differ from D-057's recorded launch freeze",
    )
    original, old, _, _, _, binding = _load_original(
        Path(recovery["original_root"]), recovery["ledger_sha256"]
    )
    require(
        all(recovery.get(k) == v for k, v in binding.items())
        and new.get("failed_cell_replacement_binding") == binding["failed_invocation"],
        "Original segment/incident binding changed",
    )
    for key in (
        "original_manifest_path",
        "original_manifest_sha256",
        "original_ledger_path",
        "original_ledger_sha256",
        "original_launch_receipt_path",
        "original_launch_receipt_sha256",
    ):
        require(
            new.get(key) == recovery[key.removeprefix("original_")],
            f"Continuation ledger provenance differs: {key}",
        )
    require(
        new.get("source_commit") == continuation["source_commit"]
        and new.get("source_root") == continuation["source_root"]
        and new.get("manifest_path") == str(manifest_path)
        and new.get("manifest_sha256") == sha256_file(manifest_path),
        "Continuation ledger identity differs",
    )
    validate_segment_layout(original, continuation, old, new)
    _check_unlocked(ledger_path.parent / "coordinator.lock")
    for ledger in (old, new):
        _validate_sleep_assertion(ledger["sleep_assertion"])
    # Hash the shared 32 GiB DDT only once. Rebase validation hashes every new
    # source file and demands identical identities for shared runtime/assets.
    _, old_paths = load_manifest(Path(binding["manifest_path"]), hashes=True)
    files, source, commit = _validate_source_rebase(
        original, Path(continuation["source_root"])
    )
    require(
        files == continuation["files"]
        and source == recovery["continuation_source"]
        and commit == continuation["source_commit"],
        "Continuation source rebase differs from reviewed repair",
    )
    _, new_paths = load_manifest(manifest_path, hashes=False)
    verify_cases(original, old_paths)
    inspected, assigned = {}, 0
    for ordinal, cell in enumerate(original["cells"]):
        manifest, ledger, paths = (
            (original, old, old_paths)
            if ordinal < ORIGINAL_TERMINAL_RECORDS
            else (continuation, new, new_paths)
        )
        ident = cell["cell_id"]
        row = ledger["records"][ident]
        inspected[ident] = validate_cell(row, manifest, paths)
        truth = read_checked_json(paths[cell["case_id"] + ":truth"])
        _, expand = cipher_functions(cell["cipher"])
        true_tuple = list(
            expand(tuple(truth["master_key"]), cell["rounds"])[-cell["trailing_keys"] :]
        )
        for attempt in Path(row["directory"]).glob("attempts/[0-9][0-9][0-9][0-9]"):
            record, _ = optional_terminal_json(
                attempt / "attempt.json", row["outcome"] == "global_deadline"
            )
            require(
                record is None
                or record["candidate"] != true_tuple
                or record["status"] != "unsat",
                "Correct trailing tuple returned false UNSAT",
            )
        if row["outcome"] == "recovered":
            checked = verify_key(
                cell["cipher"],
                cell["rounds"],
                inspected[ident]["recovered_key"],
                read_checked_json(paths[cell["case_id"] + ":public"]),
                truth,
            )
            require_assigned_key(checked)
            assigned += 1
    parity = []
    for case in original["cases"]:
        case_cells = [c for c in original["cells"] if c["case_id"] == case]
        require(
            len({inspected[c["cell_id"]]["stage1_sha256"] for c in case_cells} - {None})
            <= 1,
            "Stage1 prefix differs across segments/depth/backend",
        )
        queues = [inspected[c["cell_id"]]["queue"] for c in case_cells]
        for depth in (1, 2):
            fingerprints = {
                inspected[c["cell_id"]]["queue"]["queue_sha256"]
                for c in case_cells
                if c["trailing_keys"] == depth
                and inspected[c["cell_id"]]["queue"] is not None
            }
            require(len(fingerprints) <= 1, "G/X queues differ across segments")
        parity.append(
            {
                "case_id": case,
                "complete_queues": sum(q is not None for q in queues),
                "parity_checked": True,
            }
        )
    segments = []
    for label, path, mp, manifest, ledger in (
        (
            "original",
            Path(binding["ledger_path"]),
            Path(binding["manifest_path"]),
            original,
            old,
        ),
        ("continuation", ledger_path, manifest_path, continuation, new),
    ):
        segments.append(
            {
                "segment": label,
                "ledger_path": str(path),
                "ledger_sha256": sha256_file(path),
                "manifest_path": str(mp),
                "manifest_sha256": sha256_file(mp),
                "source_commit": manifest["source_commit"],
                "terminal_cells": len(ledger["records"]),
                "started_invocations": len(ledger["starts"]),
            }
        )
    return {
        "schema": "backend_ranking_depth_segment_audit_v1",
        "passed": True,
        "mode": "full",
        "evidence_role": "development",
        "segments": segments,
        "profile_sha256": digest(original["profile"]),
        "cells": 256,
        "assigned_verified_recoveries": assigned,
        "completed_attempts": sum(r["completed_attempts"] for r in inspected.values()),
        "queue_parity": parity,
        "unavailable_original_attempt": binding["failed_invocation"],
        "attempt_history": {
            "assigned_cells": 256,
            "starts": 257,
            "authorized_retries": 1,
        },
    }
