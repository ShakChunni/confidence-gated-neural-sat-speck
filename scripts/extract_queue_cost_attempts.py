#!/usr/bin/env python3
"""Extract per-attempt and per-cell tables for the N=32 backend x ranking-depth study.

Read-only with respect to the repository and every run directory. Writes only
attempts.csv, cells.csv and extraction_report.md into --output-dir, which must
not be under results/ or datasets/.

Run as:
    python3 -I scripts/extract_queue_cost_attempts.py --output-dir DIR [--repo-root DIR]
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # never create new .pyc files inside the snapshot trees

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
PROTECTED_OUTPUT_DIRS = ("results", "datasets")

# Module paths are bound by layout() in main(), before any of them is used.
BASE: Path
EVIDENCE: Path
ROOTS: dict[str, Path]
SNAPSHOTS: dict[str, Path]
OUT: Path

WORD_BITS = 16
EXPECTED = {"cells": 256, "completed": 6110, "unsat": 5992, "sat": 118}
ATTEMPT_DIR = re.compile(r"[0-9]{4}")
TOL = 1e-6

checks: list[tuple[str, bool, str]] = []
mismatches: dict[str, list[str]] = {}
diagnostics: dict[str, float | int] = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, bool(ok), detail))


def mismatch(name: str, item: str) -> None:
    mismatches.setdefault(name, []).append(item)


def load(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def opt_int(text: str):
    return None if text in ("", "None") else int(text)


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def decode_literals(literals: list[int], trailing: int) -> tuple[int, ...]:
    """Recover the candidate words from the start record's assumption literals.

    formula.py emits, for each word w (chronological order) and bit b, the literal
    +v if bit b of w is 1 and -v otherwise, bits in ascending order. So the sign
    pattern of each 16-literal group is the word.
    """
    if len(literals) != WORD_BITS * trailing:
        raise ValueError(
            f"expected {WORD_BITS * trailing} literals, got {len(literals)}"
        )
    if any(lit == 0 for lit in literals) or len({abs(lit) for lit in literals}) != len(
        literals
    ):
        raise ValueError("zero or repeated literal")
    words = []
    for w in range(trailing):
        group = literals[w * WORD_BITS : (w + 1) * WORD_BITS]
        words.append(sum(1 << b for b, lit in enumerate(group) if lit > 0))
    return tuple(words)


def load_expanders() -> dict:
    sys.path.insert(0, str(SNAPSHOTS["original"]))
    from neural_sat_speck.simon.key_schedule import expand_key as simon_expand
    from neural_sat_speck.speck.key_schedule import expand_key as speck_expand

    return {"simon32_64": simon_expand, "speck32_64": speck_expand}


def snapshot_consistency() -> None:
    rels = [
        "neural_sat_speck/simon/key_schedule.py",
        "neural_sat_speck/speck/key_schedule.py",
        "neural_sat_speck/simon/primitives.py",
        "neural_sat_speck/speck/primitives.py",
    ]
    for rel in rels:
        a = SNAPSHOTS["original"] / rel
        b = SNAPSHOTS["continuation"] / rel
        if a.exists() and b.exists():
            check(
                f"key schedule source identical across snapshots: {rel}",
                sha256(a) == sha256(b),
                f"{sha256(a)[:12]} vs {sha256(b)[:12]}",
            )


def layout(repo_root: Path, output_dir: Path) -> None:
    """Bind the module-level paths to a repository root and an output directory."""
    global BASE, EVIDENCE, ROOTS, SNAPSHOTS, OUT
    BASE = repo_root
    EVIDENCE = BASE / "docs/backend_ranking_depth_evidence_20261009"
    ROOTS = {
        "original": BASE / "results/backend_ranking_depth_20261008_full_d056",
        "continuation": BASE
        / "results/backend_ranking_depth_20261009_continuation_d057",
    }
    SNAPSHOTS = {
        "original": BASE / "runtime/backend_ranking_depth_snapshot_20261008_d056/src",
        "continuation": BASE
        / "runtime/backend_ranking_depth_recovery_snapshot_20261009_v2/src",
    }
    OUT = output_dir


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract per-attempt and per-cell tables for the N=32 study."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="repository root (default: the root containing this script)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory receiving attempts.csv, cells.csv and extraction_report.md",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for name in PROTECTED_OUTPUT_DIRS:
        protected = (repo_root / name).resolve()
        if output_dir == protected or output_dir.is_relative_to(protected):
            raise SystemExit(
                f"refusing to write into {protected}; choose another --output-dir"
            )
    layout(repo_root, output_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    expanders = load_expanders()
    snapshot_consistency()

    # ---- canonical cell list -------------------------------------------------
    canon = read_rows(EVIDENCE / "analysis_v2.csv")
    analysis = load(EVIDENCE / "analysis_v2.json")
    acells = {c["cell_id"]: c for c in analysis["cells"]}
    cids = [r["cell_id"] for r in canon]
    check(
        "canonical CSV has 256 rows with unique cell_id",
        len(canon) == EXPECTED["cells"] and len(set(cids)) == EXPECTED["cells"],
        f"rows={len(canon)}",
    )
    check(
        "analysis_v2.json cell set == analysis_v2.csv cell set",
        set(acells) == set(cids),
    )
    csv_vs_json_bad = []
    for r in canon:
        a = acells[r["cell_id"]]
        same = (
            r["segment"] == a["segment"]
            and r["cipher"] == a["cipher"]
            and int(r["key_index"]) == a["key_index"]
            and int(r["trailing_keys"]) == a["trailing_keys"]
            and r["profile"] == a["profile"]
        )
        if not same:
            csv_vs_json_bad.append(r["cell_id"])
    check(
        "analysis CSV attributes == analysis JSON attributes",
        not csv_vs_json_bad,
        ",".join(csv_vs_json_bad),
    )

    manifest = load(ROOTS["original"] / "inputs/manifest.json")
    mcells = {c["cell_id"]: c for c in manifest["cells"]}
    cont_manifest = load(ROOTS["continuation"] / "inputs/continuation_manifest.json")
    ccells = {c["cell_id"]: c for c in cont_manifest["cells"]}
    man_bad = []
    for cid in cids:
        m, a = mcells.get(cid), acells[cid]
        if m is None or any(
            m[k] != a[k]
            for k in ("cipher", "key_index", "trailing_keys", "profile", "rounds")
        ):
            man_bad.append(cid)
    check(
        "d056 manifest cells agree with analysis (cipher/key/depth/profile/rounds)",
        not man_bad and len(mcells) == 256,
        ",".join(man_bad),
    )
    cont_bad = [
        cid
        for cid in cids
        if cid not in ccells
        or any(
            ccells[cid][k] != acells[cid][k]
            for k in ("cipher", "key_index", "trailing_keys", "profile", "rounds")
        )
    ]
    check(
        "continuation manifest (full 256-cell grid) agrees with analysis",
        set(ccells) == set(cids) and not cont_bad,
        f"n={len(ccells)} bad={','.join(cont_bad)}",
    )

    # ---- designated run directories from the ledgers -------------------------
    ledgers = {seg: load(ROOTS[seg] / "execution/ledger.json") for seg in ROOTS}
    for seg in ROOTS:
        seg_ids = {r["cell_id"] for r in canon if r["segment"] == seg}
        check(
            f"{seg} ledger records == {seg}-segment cells",
            set(ledgers[seg]["records"]) == seg_ids,
            f"ledger={len(ledgers[seg]['records'])} segment={len(seg_ids)}",
        )
        # The original ledger also lists the failed original start of speck32_64_17_m1_cms_native_no_gauss
        # (no record); the continuation ledger's order is exactly its own segment.
        extra = set(ledgers[seg]["execution_order"]) - seg_ids
        expected_extra = (
            {"speck32_64_17_m1_cms_native_no_gauss"} if seg == "original" else set()
        )
        check(
            f"{seg} ledger execution_order == {seg}-segment cells (+ expected failed start)",
            set(ledgers[seg]["execution_order"]) >= seg_ids
            and extra == expected_extra
            and len(ledgers[seg]["execution_order"])
            == len(set(ledgers[seg]["execution_order"])),
            f"n={len(ledgers[seg]['execution_order'])} extra={sorted(extra)}",
        )

    def run_dir_for(seg: str, cid: str) -> Path:
        rec = ledgers[seg]["records"][cid]
        d = Path(rec["directory"])
        start_dir = Path(ledgers[seg]["starts"][cid]["directory"])
        ok = (
            d.is_dir()
            and d.parent == ROOTS[seg] / "execution/cells" / cid
            and start_dir == d
        )
        if not ok:
            mismatch("designated run directory", f"{cid} ({seg})")
        return d

    excluded = []
    for r in canon:
        cid = r["cell_id"]
        designated = run_dir_for(r["segment"], cid)
        for seg, root in ROOTS.items():
            cell_root = root / "execution/cells" / cid
            if cell_root.is_dir():
                for d in sorted(cell_root.iterdir()):
                    if d.is_dir() and d.name != designated.name:
                        excluded.append(f"{cid}: {d.name} ({seg} root)")
    check(
        "every non-designated run dir is accounted for (expected 1: failed original of speck32_64_17_m1_cms_native_no_gauss)",
        len(excluded) == 1
        and excluded[0].startswith("speck32_64_17_m1_cms_native_no_gauss"),
        "; ".join(excluded),
    )

    # ---- truth and true trailing tuples --------------------------------------
    truth_sha_bad = []
    true_tuples: dict[str, tuple[int, ...]] = {}
    for r in canon:
        cid = r["cell_id"]
        a = acells[cid]
        case = mcells[cid]["case_id"]
        tpath = (
            ROOTS["original"]
            / "inputs"
            / f"{a['cipher']}_{a['key_index']:02d}_truth.json"
        )
        if sha256(tpath) != manifest["cases"][case]["truth"]["sha256"]:
            truth_sha_bad.append(tpath.name)
        truth = load(tpath)
        round_keys = expanders[a["cipher"]](tuple(truth["master_key"]), a["rounds"])
        true_tuples[cid] = tuple(round_keys[-a["trailing_keys"] :])
    check(
        "truth files match manifest sha256",
        not truth_sha_bad,
        f"checked={len(canon)} bad={len(truth_sha_bad)}",
    )

    # ---- per-cell and per-attempt extraction ----------------------------------
    attempt_rows: list[dict] = []
    cell_rows: list[dict] = []
    total_attempts = {"unsat": 0, "sat": 0, "other": 0, "killed": 0}
    sat_true = 0
    unsat_true = 0
    recovered_cells = 0
    censor_vs_inflight = []
    kill_notes: list[str] = []
    start_clock_gap = []
    solve_clock_gap = []

    for r in canon:
        cid = r["cell_id"]
        seg = r["segment"]
        a = acells[cid]
        cipher, profile = a["cipher"], a["profile"]
        key_index, depth = a["key_index"], a["trailing_keys"]
        run = run_dir_for(seg, cid)
        true_tuple = true_tuples[cid]

        sup = load(run / "supervisor_result.json")
        parent_start = sup["started_monotonic"]
        deadline = sup["deadline_monotonic"]
        pa = sup["phase_accounting"]
        totals = dict(pa["completed_seconds"])
        inflight_phase = pa.get("inflight_phase")
        if inflight_phase:
            totals[inflight_phase] = (
                totals.get(inflight_phase, 0.0) + pa["inflight_seconds"]
            )
        stage1 = totals.get("stage1_ranking", 0.0)
        stage2 = totals.get("stage2_ranking", 0.0)
        startup = sup["parent_wall_seconds"] - sum(totals.values())

        queue_path = run / "queue.json"
        if queue_path.exists():
            qdoc = load(queue_path)
            queue = [tuple(q) for q in qdoc["queue"]]
            queue_ok = (
                qdoc["trailing_keys"] == depth
                and all(
                    len(q) == depth
                    and all(type(w) is int and 0 <= w < 65536 for w in q)
                    for q in queue
                )
                and len(set(queue)) == len(queue)
            )
            check(f"queue well-formed: {cid}", queue_ok, "")
        else:
            queue = []
            if a["candidate_attempts_started"] != 0:
                mismatch("queue.json missing but attempts were started", cid)

        att_dirs = sorted(
            (
                p
                for p in (run / "attempts").iterdir()
                if p.is_dir() and ATTEMPT_DIR.fullmatch(p.name)
            ),
            key=lambda p: p.name,
        )
        attempts = []
        for p in att_dirs:
            idx = int(p.name)
            start = load(p / "attempt_start.json")
            if start["index"] != idx:
                mismatch("attempt_start index != directory name", f"{cid} #{idx}")
            decoded = decode_literals(start["candidate_literals"], depth)
            start_mono = start["observed_monotonic"]
            done_path = p / "attempt.json"
            if done_path.exists():
                done = load(done_path)
                if done["index"] != idx:
                    mismatch("attempt.json index != directory name", f"{cid} #{idx}")
                cand = tuple(done["candidate"])
                status = (
                    done["status"] if done["status"] in ("unsat", "sat") else "other"
                )
                solve = done["solve_wall_seconds"]
                censored, censor = False, None
                start_clock_gap.append(abs(start_mono - done["started_monotonic"]))
                solve_clock_gap.append(
                    abs((done["ended_monotonic"] - done["started_monotonic"]) - solve)
                )
            else:
                cand = decoded
                status, solve, censored = "killed", None, True
                censor = deadline - start_mono
            attempts.append(
                dict(
                    idx=idx,
                    cand=cand,
                    decoded=decoded,
                    status=status,
                    solve=solve,
                    censored=censored,
                    censor=censor,
                    start_mono=start_mono,
                    offset=start_mono - parent_start,
                    is_true=cand == true_tuple,
                    qpos=queue.index(cand) if cand in queue else None,
                )
            )

        # ---- attempt-level validation ----
        for att in attempts:
            tag = f"{cid} #{att['idx']:04d}"
            if att["decoded"] != att["cand"]:
                mismatch("decoded assumption literals != attempt candidate", tag)
            if att["qpos"] != att["idx"]:
                mismatch(
                    "attempt candidate not at queue position == attempt index", tag
                )
            total_attempts[att["status"]] = total_attempts.get(att["status"], 0) + 1
            if att["status"] == "sat" and att["is_true"]:
                sat_true += 1
            if att["status"] == "sat" and not att["is_true"]:
                mismatch("SAT attempt that is not the true tuple", tag)
            if att["status"] == "unsat" and att["is_true"]:
                unsat_true += 1
                mismatch("UNSAT attempt on the true tuple", tag)
            if (
                att["status"] == "killed"
                and inflight_phase is not None
                and att is attempts[-1]
            ):
                censor_vs_inflight.append(abs(att["censor"] - pa["inflight_seconds"]))
        idxs = [att["idx"] for att in attempts]
        if idxs != list(range(len(attempts))):
            mismatch("attempt indices not contiguous from 0", cid)
        if any(
            attempts[i + 1]["start_mono"] <= attempts[i]["start_mono"]
            for i in range(len(attempts) - 1)
        ):
            mismatch("attempt starts not strictly increasing", cid)
        killed_positions = [
            i for i, att in enumerate(attempts) if att["status"] == "killed"
        ]
        if killed_positions not in ([], [len(attempts) - 1]):
            mismatch("killed attempt is not the last attempt", cid)

        # ---- cell-level validation against the analysis ----
        n_completed = sum(1 for att in attempts if att["status"] != "killed")
        true_pos = queue.index(true_tuple) if true_tuple in queue else None
        true_dispatched = any(att["is_true"] for att in attempts)
        sat_true_here = any(
            att["status"] == "sat" and att["is_true"] for att in attempts
        )
        recovered = sup["outcome"] == "recovered"
        recovered_cells += recovered

        if sup["outcome"] != a["outcome"]:
            mismatch("supervisor outcome != analysis outcome", cid)
        if abs(sup["parent_wall_seconds"] - a["parent_wall_seconds"]) > TOL:
            mismatch("parent_wall_seconds", cid)
        if abs(stage1 - a["stage1_seconds"]) > TOL:
            mismatch("stage1_seconds", cid)
        if abs(stage2 - a["stage2_seconds"]) > TOL:
            mismatch("stage2_seconds", cid)
        if abs(startup - a["startup_and_supervision_seconds"]) > TOL:
            mismatch("startup_and_supervision_seconds", cid)
        if len(attempts) != a["candidate_attempts_started"]:
            mismatch("n_attempts_started != candidate_attempts_started", cid)
        if len(attempts) != a["attempt_directories_retained"]:
            mismatch("attempt directories != attempt_directories_retained", cid)
        if true_pos != a["true_tuple_queue_index_zero_based"]:
            mismatch(
                "true_queue_position != analysis true_tuple_queue_index_zero_based",
                f"{cid}: {true_pos} vs {a['true_tuple_queue_index_zero_based']}",
            )
        if true_dispatched != a["true_tuple_dispatched"]:
            mismatch("true_tuple_dispatched", cid)
        if recovered != a["assigned_verified_recovery"] or (recovered != sat_true_here):
            mismatch("recovered flag disagrees (outcome / analysis / SAT-on-true)", cid)

        prep = attempts[0]["start_mono"] - parent_start if attempts else None
        if abs((deadline - parent_start) - 300.0) > TOL:
            mismatch(
                "deadline is not 300 s after parent start",
                f"{cid}: {deadline - parent_start!r}",
            )
        n_killed = sum(1 for att in attempts if att["status"] == "killed")
        if (sup["outcome"] == "global_deadline") != (n_killed == 1):
            kill_notes.append(
                f"{cid}: outcome={sup['outcome']} killed_attempts={n_killed}"
            )
        cell_rows.append(
            dict(
                cell_id=cid,
                cipher=cipher,
                key_index=key_index,
                depth=depth,
                profile=profile,
                recovered=recovered,
                parent_wall_seconds=sup["parent_wall_seconds"],
                prep_seconds=prep,
                stage1_seconds=stage1,
                stage2_seconds=stage2,
                startup_and_supervision_seconds=startup,
                queue_length=len(queue),
                true_queue_position=true_pos,
                n_attempts_started=len(attempts),
                n_attempts_completed=n_completed,
                deadline_seconds_from_start=deadline - parent_start,
                segment=seg,
                run_uuid=run.name,
            )
        )
        for att in attempts:
            attempt_rows.append(
                dict(
                    cell_id=cid,
                    cipher=cipher,
                    key_index=key_index,
                    depth=depth,
                    profile=profile,
                    attempt_index=att["idx"],
                    queue_position=att["qpos"],
                    is_true=att["is_true"],
                    status=att["status"],
                    solve_wall_seconds=att["solve"],
                    censored=att["censored"],
                    censor_seconds=att["censor"],
                    started_offset_seconds=att["offset"],
                    segment=seg,
                    run_uuid=run.name,
                )
            )

    # ---- global checks --------------------------------------------------------
    completed = (
        total_attempts["unsat"] + total_attempts["sat"] + total_attempts["other"]
    )
    check(
        "256 cells extracted", len(cell_rows) == EXPECTED["cells"], f"{len(cell_rows)}"
    )
    check(
        "completed attempts == 6110", completed == EXPECTED["completed"], f"{completed}"
    )
    check(
        "unsat attempts == 5992",
        total_attempts["unsat"] == EXPECTED["unsat"],
        f"{total_attempts['unsat']}",
    )
    check(
        "sat attempts == 118",
        total_attempts["sat"] == EXPECTED["sat"],
        f"{total_attempts['sat']}",
    )
    check(
        "no 'other' status values",
        total_attempts["other"] == 0,
        f"{total_attempts['other']}",
    )
    check(
        "every SAT attempt is the true tuple",
        total_attempts["sat"] == sat_true,
        f"sat={total_attempts['sat']} sat_on_true={sat_true}",
    )
    check("no UNSAT attempt is the true tuple", unsat_true == 0, f"{unsat_true}")
    check(
        "recovered cells == 118",
        recovered_cells == EXPECTED["sat"],
        f"{recovered_cells}",
    )
    check(
        "started attempts == sum(candidate_attempts_started) in analysis",
        len(attempt_rows)
        == sum(a["candidate_attempts_started"] for a in acells.values()),
        f"{len(attempt_rows)}",
    )
    diagnostics["killed_attempts"] = total_attempts["killed"]
    check(
        "global_deadline cells <=> exactly one killed (in-flight) attempt",
        not kill_notes,
        f"exceptions={len(kill_notes)}"
        + ("; " + "; ".join(kill_notes[:5]) if kill_notes else ""),
    )
    check(
        "deadline = parent start + 300 s for all cells",
        not mismatches.get("deadline is not 300 s after parent start"),
        "",
    )
    diagnostics["completed_attempts"] = completed
    diagnostics["unsat_attempts"] = total_attempts["unsat"]
    diagnostics["sat_attempts"] = total_attempts["sat"]
    diagnostics["max_censor_vs_supervisor_inflight_seconds"] = (
        max(censor_vs_inflight) if censor_vs_inflight else 0.0
    )
    diagnostics["max_start_observed_vs_attempt_started_seconds"] = (
        max(start_clock_gap) if start_clock_gap else 0.0
    )
    diagnostics["max_solve_wall_vs_ended_minus_started_seconds"] = (
        max(solve_clock_gap) if solve_clock_gap else 0.0
    )
    diagnostics["cells_with_true_tuple_not_in_queue"] = sum(
        1 for c in cell_rows if c["true_queue_position"] is None
    )
    diagnostics["cells_with_zero_attempts"] = sum(
        1 for c in cell_rows if c["n_attempts_started"] == 0
    )

    # ---- write CSVs -----------------------------------------------------------
    att_cols = [
        "cell_id",
        "cipher",
        "key_index",
        "depth",
        "profile",
        "attempt_index",
        "queue_position",
        "is_true",
        "status",
        "solve_wall_seconds",
        "censored",
        "censor_seconds",
        "started_offset_seconds",
        "segment",
        "run_uuid",
    ]
    cell_cols = [
        "cell_id",
        "cipher",
        "key_index",
        "depth",
        "profile",
        "recovered",
        "parent_wall_seconds",
        "prep_seconds",
        "stage1_seconds",
        "stage2_seconds",
        "startup_and_supervision_seconds",
        "queue_length",
        "true_queue_position",
        "n_attempts_started",
        "n_attempts_completed",
        "deadline_seconds_from_start",
        "segment",
        "run_uuid",
    ]
    with open(OUT / "attempts.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=att_cols)
        writer.writeheader()
        for row in attempt_rows:
            writer.writerow({k: fmt(row[k]) for k in att_cols})
    with open(OUT / "cells.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cell_cols)
        writer.writeheader()
        for row in cell_rows:
            writer.writerow({k: fmt(row[k]) for k in cell_cols})

    write_report(attempt_rows, cell_rows, excluded, kill_notes)


def write_report(attempt_rows, cell_rows, excluded, kill_notes) -> None:
    n_fail = sum(1 for _, ok, _ in checks if not ok)
    lines = []
    lines.append("# Extraction report: N=32 backend x ranking-depth per-attempt table")
    lines.append("")
    lines.append(
        "Generated by `scripts/extract_queue_cost_attempts.py`. Read-only over the repository and run directories."
    )
    lines.append("")
    lines.append("## Outputs")
    lines.append(
        f"- `attempts.csv`: {len(attempt_rows)} rows, one per started attempt."
    )
    lines.append(f"- `cells.csv`: {len(cell_rows)} rows, one per canonical cell.")
    lines.append("")
    lines.append("## Sources and designation")
    lines.append(
        f"- Canonical cells: `{EVIDENCE / 'analysis_v2.csv'}` (cross-checked against `analysis_v2.json`)."
    )
    lines.append(
        f"- Original segment run dirs: `{ROOTS['original'] / 'execution/cells'}/<cell_id>/<uuid>`, designated by `{ROOTS['original'] / 'execution/ledger.json'}` `records[cell_id].directory`."
    )
    lines.append(
        f"- Continuation segment run dirs: `{ROOTS['continuation'] / 'execution/cells'}/<cell_id>/<uuid>`, designated by `{ROOTS['continuation'] / 'execution/ledger.json'}` `records[cell_id].directory`."
    )
    lines.append(
        "- Excluded run dirs (present on disk, not designated by any ledger record):"
    )
    for item in excluded:
        if not item.startswith("speck32_64_17_m1_cms_native_no_gauss"):
            lines.append(f"  - {item}")
    lines.append(
        "  - `speck32_64_17_m1_cms_native_no_gauss` original-segment run `660e0a9d34944fbb9933663194f1b35f` (failed original attempt; the ledger has a start but no record). The cell uses continuation run `2ba20b903e984ca6b09611d8dcdc69a7`."
    )
    lines.append(
        f"- Truth: `{ROOTS['original'] / 'inputs'}/<cipher>_<KK>_truth.json` (continuation inputs contain no truth files). sha256 checked against `manifest.json` `cases[*].truth.sha256`."
    )
    lines.append(
        "- Key schedules: imported from the d056 snapshot `neural_sat_speck.simon.key_schedule.expand_key` and `neural_sat_speck.speck.key_schedule.expand_key`, `rounds` = manifest cell rounds, true tuple = `expand_key(master_key, rounds)[-trailing_keys:]` (same convention as `followups/ranking_depth_audit.py`)."
    )
    lines.append("")
    lines.append("## Rules")
    lines.append(
        "- Queue: `queue.json` `queue` list, in stored order. Queue position = index in that list."
    )
    lines.append(
        "- Attempt candidate: `attempts/NNNN/attempt.json` `candidate` when present. The candidate is also decoded from `attempt_start.json` `candidate_literals` (16 literals per word, chronological word order; literal > 0 means bit = 1). The decoded value must equal the attempt.json candidate, and both must sit at queue position NNNN."
    )
    lines.append(
        "- Status: `attempt.json` `status` if `unsat` or `sat`; any other value is `other`; no `attempt.json` means `killed` (right-censored at the deadline)."
    )
    lines.append(
        "- `is_true`: candidate == true tuple (for m1 the last round key; for m2 the last two round keys in chronological order `[k_{R-2}, k_{R-1}]`)."
    )
    lines.append(
        "- `solve_wall_seconds`: `attempt.json` `solve_wall_seconds`; blank when killed."
    )
    lines.append(
        "- `started_offset_seconds`: `attempt_start.json` `observed_monotonic` minus the cell's `supervisor_result.json` `started_monotonic` (parent start). The same start timestamp is used for every attempt, so the offset is uniform across finished and killed attempts."
    )
    lines.append(
        "- `censor_seconds` (killed only): `supervisor_result.json` `deadline_monotonic` minus `attempt_start.json` `observed_monotonic`. The deadline is `started_monotonic + 300 s` for every cell. The start record is written before solver launch and is the only start timestamp present for killed attempts; `attempt.json` `started_monotonic` differs from it by at most the amount in the diagnostics below. Using one timestamp for all attempts keeps the offsets and censor times on a single clock."
    )
    lines.append(
        "- `prep_seconds`: first attempt's `started_offset_seconds` (time from parent start to first attempt start); blank if no attempts."
    )
    lines.append(
        "- `stage1_seconds` / `stage2_seconds`: per phase total = `supervisor_result.json` `phase_accounting.completed_seconds[phase]` plus `inflight_seconds` if `inflight_phase` equals that phase. Stage1 is `stage1_ranking`, Stage2 is `stage2_ranking`; missing phases count as 0."
    )
    lines.append(
        "- `startup_and_supervision_seconds`: `parent_wall_seconds` minus the sum of all phase totals (completed plus in-flight), i.e. the unattributed remainder."
    )
    lines.append(
        "- `recovered`: supervisor `outcome == 'recovered'`. Must agree with `assigned_verified_recovery` and with the existence of a SAT attempt on the true tuple."
    )
    lines.append(
        "- `true_queue_position`: index of the true tuple in the queue; blank if absent."
    )
    lines.append(
        "- `n_attempts_started`: number of `NNNN` attempt directories; `n_attempts_completed`: those with `attempt.json`."
    )
    lines.append(
        "- `deadline_seconds_from_start`: `deadline_monotonic` minus `started_monotonic`."
    )
    lines.append(
        "- Extra trailing columns `segment` and `run_uuid` are added in both CSVs for traceability."
    )
    lines.append("")
    lines.append(f"## Validation checks ({len(checks)} run, {n_fail} failed)")
    lines.append("")
    lines.append("| Check | Result | Detail |")
    lines.append("|---|---|---|")
    for name, ok, detail in checks:
        if name.startswith("queue well-formed: "):
            continue
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    queue_checks = [c for c in checks if c[0].startswith("queue well-formed: ")]
    lines.append(
        f"| queue well-formed (all {len(queue_checks)} cells) | {'PASS' if all(ok for _, ok, _ in queue_checks) else 'FAIL'} | |"
    )
    lines.append("")
    lines.append("## Counts")
    lines.append(
        f"- Completed attempts: {diagnostics['completed_attempts']} (unsat {diagnostics['unsat_attempts']}, sat {diagnostics['sat_attempts']})."
    )
    lines.append(
        f"- Killed (right-censored) attempts: {diagnostics['killed_attempts']}."
    )
    lines.append(
        f"- Cells whose true tuple is not in the queue: {diagnostics['cells_with_true_tuple_not_in_queue']}."
    )
    lines.append(
        f"- Cells with zero started attempts: {diagnostics['cells_with_zero_attempts']}."
    )
    qlen = collections.Counter((c["depth"], c["queue_length"]) for c in cell_rows)
    lines.append(
        "- Queue length by depth (depth, length: cells): "
        + ", ".join(f"({d}, {n}): {k}" for (d, n), k in sorted(qlen.items()))
        + "."
    )
    lines.append("")
    lines.append("## Mismatches")
    if not mismatches:
        lines.append(
            "None. Every per-cell and per-attempt cross-check agreed with the analysis and with the raw files."
        )
    else:
        for name, items in mismatches.items():
            lines.append(f"- **{name}** ({len(items)}):")
            for item in items:
                lines.append(f"  - {item}")
    lines.append("")
    lines.append("## Diagnostics (clock and accounting consistency)")
    lines.append(
        f"- Max |censor_seconds - supervisor inflight_seconds| on the last (killed) attempt: {diagnostics['max_censor_vs_supervisor_inflight_seconds']:.6f} s. Expected to be a few ms, because the supervisor measures in-flight time from a different clock read."
    )
    lines.append(
        f"- Max |attempt_start observed_monotonic - attempt.json started_monotonic| over finished attempts: {diagnostics['max_start_observed_vs_attempt_started_seconds']:.6f} s."
    )
    lines.append(
        f"- Max |ended_monotonic - started_monotonic - solve_wall_seconds|: {diagnostics['max_solve_wall_vs_ended_minus_started_seconds']:.2e} s."
    )
    lines.append("")
    lines.append("## Notes")
    lines.append(
        f"- global_deadline / killed-attempt exceptions: {len(kill_notes)}"
        + (f" ({'; '.join(kill_notes)})" if kill_notes else "")
        + "."
    )
    lines.append(
        "- Only the d056 snapshot's key-schedule module was imported. The d057 recovery snapshot's source for these modules is compared by sha256 in the checks above."
    )
    lines.append(
        "- The report is generated from the same run as the CSVs, so every number above is reproducible by re-running `python3 -I scripts/extract_queue_cost_attempts.py --output-dir <dir>`."
    )
    lines.append("")
    (OUT / "extraction_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
