#!/usr/bin/env python3
"""One-command D-060/D-061 launcher: snapshot -> pilot -> gates -> full run.

Run from the repository root after the campaign commit, inside caffeinate:

    caffeinate -dimsu .venv/bin/python -u scripts/launch_prospective_queue_cost.py

Stages (each refuses to overwrite existing output, so a stopped stage is never
silently retried):
1. Check that HEAD contains the campaign code/config/protocol with no
   uncommitted changes to them, and that the cohort and R6 table match their pins.
2. Clone HEAD into runtime/prospective_queue_cost_snapshot_<sha7> (clean source).
3. Pilot: 10 cells on opened D-056 key 0, mechanics only (never analysed).
4. Pilot gates: audit passed; at least one capped attempt; no solver left alive.
5. Full run: 320 cells on the fresh 601001/601002 cohort, supervised with the
   100,000 s campaign ceiling. Receipts go to runtime/d060_*_receipt.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
TF_PYTHON = PROJECT / "runtime/simon_tf_venv/bin/python"
PYTHON = PROJECT / ".venv/bin/python"
PILOT_OUT = PROJECT / "results/prospective_queue_cost_d060_pilot"
FULL_OUT = PROJECT / "results/prospective_queue_cost_d060_full"
CONFIG_PILOT = "configs/protocol/prospective_queue_cost_d060_pilot_n32.json"
CONFIG_FULL = "configs/protocol/prospective_queue_cost_d060_n32.json"
CEILING = 100_000
CAMPAIGN_PATHS = [
    "src",
    "scripts",
    "configs/protocol",
    "tests",
    "docs/PROSPECTIVE_QUEUE_COST_TEST_PROTOCOL_20261009.md",
    "docs/DECISIONS.md",
    "docs/queue_cost_model_evidence_20261009",
]


def now() -> str:
    return datetime.now(UTC).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str, cwd: Path = PROJECT) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    os.replace(tmp, path)


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def check_commit() -> str:
    head = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--", *CAMPAIGN_PATHS)
    if dirty:
        raise SystemExit("Campaign files have uncommitted changes:\n" + dirty)
    full = json.loads((PROJECT / CONFIG_FULL).read_text())
    for name in ("cohort", "speck_classical_r6"):
        pin = full["files"][name]
        if sha256(PROJECT / pin["path"]) != pin["sha256"]:
            raise SystemExit(f"{name} does not match its pinned SHA-256")
    log(f"HEAD {head} clean for campaign paths; cohort and R6 table match pins")
    return head


def make_snapshot(head: str) -> Path:
    snap = PROJECT / f"runtime/prospective_queue_cost_snapshot_{head[:7]}"
    if not snap.exists():
        subprocess.run(["git", "clone", "--quiet", str(PROJECT), str(snap)], check=True)
        git("checkout", "--quiet", head, cwd=snap)
    if git("rev-parse", "HEAD", cwd=snap) != head or git(
        "status", "--porcelain", cwd=snap
    ):
        raise SystemExit(f"Snapshot {snap} is not a clean checkout of {head}")
    log(f"Snapshot {snap}")
    return snap


def environment(snap: Path) -> dict[str, str]:
    sys.path.insert(0, str(snap / "src"))
    from neural_sat_speck.followups.conditioning_runner import controlled_environment

    tmp = snap / "runtime/ranking_depth_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    return controlled_environment(snap, tmp)


def argv(
    snap: Path, mode: str, out: Path, config: str, pilot_audit: Path | None
) -> list[str]:
    a = [
        str(PYTHON),
        "-u",
        str(snap / "scripts/run_backend_ranking_depth.py"),
        "--source-root",
        str(snap),
        "--asset-root",
        str(PROJECT),
        "--tf-python",
        str(TF_PYTHON),
        "--output-root",
        str(out),
        "--mode",
        mode,
        "--config",
        str(snap / config),
        "--author-approval",
        "D-060",
    ]
    if pilot_audit is not None:
        a += ["--pilot-audit", str(pilot_audit)]
    return a


def supervise(
    name: str, args: list[str], env: dict, out: Path, head: str, ceiling: float
) -> int:
    receipt_path = PROJECT / f"runtime/d060_{name}_receipt.json"
    started = time.monotonic()
    child = subprocess.Popen(
        args, cwd=PROJECT, env=env, stdin=subprocess.DEVNULL, start_new_session=True
    )
    receipt = {
        "schema": "d060_supervised_launch_v1",
        "stage": name,
        "status": "RUNNING",
        "supervisor_pid": os.getpid(),
        "coordinator_pid": child.pid,
        "started_utc": now(),
        "argv": args,
        "source_commit": head,
        "output_root": str(out),
        "ceiling_seconds": ceiling,
        "author_instruction": "start the run (2026-10-09)",
    }
    write_json(receipt_path, receipt)
    log(f"{name} started, coordinator pid {child.pid}")
    try:
        child.wait(timeout=max(0.1, ceiling - 30))
    except subprocess.TimeoutExpired:
        receipt["status"] = "CAMPAIGN_CEILING_STOP"
        child.send_signal(signal.SIGINT)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)
        from neural_sat_speck.followups.ranking_depth_runner import (
            live_invocation_groups,
        )

        ledger = out / "execution/ledger.json"
        if ledger.exists():
            data = json.loads(ledger.read_text())
            for ident, row in data["starts"].items():
                if ident not in data["records"]:
                    for group in live_invocation_groups(row["invocation_id"]):
                        os.killpg(group, signal.SIGKILL)
    receipt.update(
        completed_utc=now(),
        exit_code=child.returncode,
        wall_seconds=time.monotonic() - started,
    )
    if receipt["status"] == "RUNNING":
        receipt["status"] = "EXITED" if child.returncode == 0 else "STOPPED"
    write_json(receipt_path, receipt)
    log(f"{name} finished: {receipt['status']} (exit {child.returncode})")
    return child.returncode


def pilot_gates() -> None:
    audit_path = PILOT_OUT / "execution/audit.json"
    if not audit_path.exists():
        raise SystemExit("Pilot produced no audit.json (stopped or failed)")
    audit = json.loads(audit_path.read_text())
    if audit.get("passed") is not True or audit.get("mode") != "pilot":
        raise SystemExit("Pilot audit did not pass")
    capped = sum(
        json.loads(p.read_text()).get("status") == "capped"
        for p in (PILOT_OUT / "execution/cells").rglob("attempt.json")
    )
    if capped == 0:
        raise SystemExit("Pilot showed no capped attempt; cap mechanics unverified")
    alive = subprocess.run(
        ["pgrep", "-f", "cryptominisat5"], capture_output=True, text=True
    )
    if alive.stdout.strip():
        raise SystemExit("Solver processes still alive after pilot: " + alive.stdout)
    write_json(
        PROJECT / "runtime/d060_pilot_gates.json",
        {
            "checked_utc": now(),
            "audit_passed": True,
            "capped_attempts": capped,
            "solvers_alive": 0,
            "audit_sha256": sha256(audit_path),
        },
    )
    log(f"Pilot gates passed ({capped} capped attempts, no solver alive)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("all", "pilot", "full"), default="all")
    args = p.parse_args()
    head = check_commit()
    snap = make_snapshot(head)
    env = environment(snap)
    if args.stage in ("all", "pilot"):
        if PILOT_OUT.exists():
            raise SystemExit(
                f"{PILOT_OUT} exists: no pilot retry without a new instruction"
            )
        supervise(
            "pilot",
            argv(snap, "pilot", PILOT_OUT, CONFIG_PILOT, None),
            env,
            PILOT_OUT,
            head,
            3600,
        )
        pilot_gates()
    if args.stage in ("all", "full"):
        if FULL_OUT.exists():
            raise SystemExit(
                f"{FULL_OUT} exists: no full-run retry without a new instruction"
            )
        code = supervise(
            "full",
            argv(
                snap, "full", FULL_OUT, CONFIG_FULL, PILOT_OUT / "execution/audit.json"
            ),
            env,
            FULL_OUT,
            head,
            CEILING,
        )
        sys.exit(code)


if __name__ == "__main__":
    main()
