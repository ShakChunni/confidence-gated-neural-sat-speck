"""Guarded runner and isolated worker for the cross-cipher residual-depth map.

Enforces:
1. Strict parent monotonic deadline (300.0s) per cell with process group SIGKILL.
2. Complete process group isolation (start_new_session=True) and clean solver lifecycle.
3. Pre-execution manifest verification against schema, D-043 decision, and dataset hash.
4. Worker execution telemetry certification (rejects Darwin QoS 9, background state, ephemeral tmpdir).
5. Comprehensive outcome taxonomy: verified_recovery, spurious_sat_distinct_key, verification_failure,
   solver_timeout, parent_watchdog_timeout, proven_unsat, worker_exception, infrastructure_interrupted.
6. Immediate correctness alarm halting on any UNSAT on instances satisfied by the assigned key.
7. Atomic crash-safe journaling with SHA-256 sidecar, preserving interrupted cells.
8. Fixed counterbalanced grid schedule across all 192 cells.
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
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neural_sat_speck.protocol.worker_telemetry import (
    WorkerTelemetry,
    certify_worker_or_raise,
    load_and_verify_worker_receipt,
    observe_worker_telemetry,
    save_worker_receipt,
)
from neural_sat_speck.sat.dimacs import export_speck_to_dimacs
from neural_sat_speck.sat.model import SpeckSatModel
from neural_sat_speck.sat.multi_round import assumptions_for_round_keys
from neural_sat_speck.simon.cipher import encrypt_block as simon_encrypt
from neural_sat_speck.simon.sat_model import SimonSatModel
from neural_sat_speck.speck.cipher import encrypt_block as speck_encrypt

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = ROOT / "configs" / "protocol" / "cross_cipher_residual_depth_manifest.json"
DEFAULT_DATASET = ROOT / "datasets" / "demo_cross_cipher_depth_r6_r11" / "dataset.json"
DEFAULT_RESULTS = ROOT / "results" / "phase11_scaling_diagnostics" / "cross_cipher_residual_depth_map.json"
DEFAULT_RUNTIME = ROOT / "runtime"
DEFAULT_TMPDIR = DEFAULT_RUNTIME / "worker_tmp"
DEFAULT_RECEIPTS = DEFAULT_RUNTIME / "receipts"

PER_CELL_CUTOFF_SECONDS = 300.0
TOTAL_CELLS_COUNT = 192

SPECK_POINTS = (
    (6, 0),
    (6, 1),
    (7, 1),
    (7, 2),
    (8, 2),
    (8, 3),
)

SIMON_POINTS = (
    (9, 0),
    (9, 1),
    (10, 1),
    (10, 2),
    (11, 2),
    (11, 3),
)


def get_git_commit(repo_root: Path = ROOT) -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_COMMIT"


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def validate_manifest_for_execution(manifest_path: Path = DEFAULT_MANIFEST, repo_root: Path = ROOT) -> dict[str, Any]:
    """Validate manifest schema, D-043 decision, authorized execution, and dataset hash."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("manifest_id") != "cross_cipher_residual_depth_map_v1":
        raise ValueError(f"Invalid manifest_id: {manifest.get('manifest_id')}")

    if manifest.get("execution_status") != "authorized_execution":
        raise ValueError(f"Manifest status is not 'authorized_execution': {manifest.get('execution_status')}")

    decisions = manifest.get("governing_decisions", [])
    if "D-043" not in decisions:
        raise ValueError(f"Manifest lacks governing decision D-043: {decisions}")

    dataset_rel = manifest.get("dataset", {}).get("relpath")
    if not dataset_rel:
        raise ValueError("Manifest missing dataset.relpath")

    dataset_path = repo_root / dataset_rel
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset specified in manifest does not exist: {dataset_path}")

    expected_ds_hash = manifest.get("dataset", {}).get("sha256")
    actual_ds_hash = compute_file_sha256(dataset_path)
    if expected_ds_hash and expected_ds_hash != actual_ds_hash:
        raise ValueError(
            f"Dataset SHA-256 mismatch: manifest expected {expected_ds_hash}, got {actual_ds_hash}"
        )

    solver_cfg = manifest.get("solver_configuration", {})
    if solver_cfg.get("backend") != "pycryptosat":
        raise ValueError(f"Expected backend pycryptosat, got {solver_cfg.get('backend')}")
    if solver_cfg.get("threads") != 1:
        raise ValueError(f"Expected single-threaded solver, got threads={solver_cfg.get('threads')}")
    if solver_cfg.get("per_cell_timeout_seconds") != PER_CELL_CUTOFF_SECONDS:
        raise ValueError(f"Expected 300.0s cutoff, got {solver_cfg.get('per_cell_timeout_seconds')}")

    return manifest


def build_counterbalanced_schedule() -> list[dict[str, Any]]:
    """Build the counterbalanced interleaved sequence of 192 cells.

    Alternates across key indices, observation levels, ciphers, and (R, m) points.
    """
    cells: list[dict[str, Any]] = []

    # Interleave to prevent clustering timeouts:
    # 8 key blocks; in each block, interleave (O=4, speck), (O=4, simon), (O=128, speck), (O=128, simon)
    for key_idx in range(8):
        for o_level in (4, 128):
            for pt_idx in range(6):
                # Speck cell
                r_sp, m_sp = SPECK_POINTS[pt_idx]
                cells.append({
                    "cell_id": f"speck32_64_r{r_sp}_m{m_sp}_o{o_level}_k{key_idx}",
                    "cipher": "speck32_64",
                    "rounds": r_sp,
                    "assumed_keys": m_sp,
                    "nominal_residual_depth": r_sp - m_sp,
                    "observations": o_level,
                    "instance_index": key_idx,
                })
                # Simon cell
                r_si, m_si = SIMON_POINTS[pt_idx]
                cells.append({
                    "cell_id": f"simon32_64_r{r_si}_m{m_si}_o{o_level}_k{key_idx}",
                    "cipher": "simon32_64",
                    "rounds": r_si,
                    "assumed_keys": m_si,
                    "nominal_residual_depth": r_si - m_si,
                    "observations": o_level,
                    "instance_index": key_idx,
                })

    assert len(cells) == TOTAL_CELLS_COUNT, f"Expected {TOTAL_CELLS_COUNT} cells, got {len(cells)}"
    assert len({c["cell_id"] for c in cells}) == TOTAL_CELLS_COUNT, "Cell IDs not unique"
    return cells


def load_dataset(dataset_path: Path = DEFAULT_DATASET) -> dict[str, Any]:
    with open(dataset_path, encoding="utf-8") as f:
        return json.load(f)


def prepare_cell_payload(
    cell: dict[str, Any],
    dataset: dict[str, Any],
    repo_root: Path,
    tmpdir: Path,
    expected_git_commit: str,
) -> dict[str, Any]:
    """Extract cell inputs from dataset and prepare payload for worker."""
    cipher = cell["cipher"]
    rounds = cell["rounds"]
    assumed_keys = cell["assumed_keys"]
    observations = cell["observations"]
    key_idx = cell["instance_index"]

    inst_data = dataset["ciphers"][cipher]["instances"][key_idx]
    master_key = inst_data["master_key"]
    r_data = inst_data["rounds_data"][str(rounds)]

    # Slicing observations: observations // 2 differential pairs
    num_pairs = observations // 2
    atk_pts = inst_data["attack_plaintexts"][:num_pairs]
    atk_cts = r_data["attack_ciphertexts"][:num_pairs]

    ver_pts = inst_data["verification_plaintexts"]
    ver_cts = r_data["verification_ciphertexts"]

    trailing = r_data["trailing_subkeys"][str(assumed_keys)]

    return {
        "cell_id": cell["cell_id"],
        "cipher": cipher,
        "rounds": rounds,
        "assumed_keys": assumed_keys,
        "nominal_residual_depth": cell["nominal_residual_depth"],
        "observations": observations,
        "instance_index": key_idx,
        "master_key": master_key,
        "attack_plaintexts": atk_pts,
        "attack_ciphertexts": atk_cts,
        "verification_plaintexts": ver_pts,
        "verification_ciphertexts": ver_cts,
        "trailing_subkeys": trailing,
        "cutoff_seconds": PER_CELL_CUTOFF_SECONDS,
        "repo_root": str(repo_root),
        "tmpdir": str(tmpdir),
        "expected_git_commit": expected_git_commit,
    }


def execute_worker_task(payload: dict[str, Any]) -> dict[str, Any]:
    """In-worker solver logic: models, assumptions, solve, verify, and telemetry receipt."""
    repo_root = Path(payload["repo_root"])
    tmpdir = payload["tmpdir"]
    expected_commit = payload["expected_git_commit"]
    cell_id = payload["cell_id"]

    # 1. Telemetry certification (fail closed immediately)
    telemetry = certify_worker_or_raise(repo_root=repo_root, tmpdir_override=tmpdir)

    cipher = payload["cipher"]
    rounds = payload["rounds"]
    assumed_keys = payload["assumed_keys"]
    observations = payload["observations"]
    master_key = tuple(payload["master_key"])
    trailing_subkeys = payload["trailing_subkeys"]
    cutoff_seconds = float(payload.get("cutoff_seconds", PER_CELL_CUTOFF_SECONDS))

    # Flatten attack pairs into (pt, ct) observations
    known_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for pt_pair, ct_pair in zip(payload["attack_plaintexts"], payload["attack_ciphertexts"], strict=True):
        known_pairs.append((tuple(pt_pair[0]), tuple(ct_pair[0])))
        known_pairs.append((tuple(pt_pair[1]), tuple(ct_pair[1])))

    ver_pairs = [
        (tuple(vp), tuple(vc))
        for vp, vc in zip(payload["verification_plaintexts"], payload["verification_ciphertexts"], strict=True)
    ]

    worker_start_monotonic = time.monotonic()
    worker_start_perf = time.perf_counter()

    # 2. Build model and extract formula metrics
    model_build_start = time.perf_counter()
    if cipher == "speck32_64":
        model = SpeckSatModel.from_known_pairs(known_pairs, rounds=rounds)
        model_build_seconds = time.perf_counter() - model_build_start

        # Metric extraction via DIMACS model
        dimacs_mod = export_speck_to_dimacs(known_pairs, rounds=rounds)
        variables_count = dimacs_mod.num_variables
        cnf_clauses_count = dimacs_mod.num_cnf_clauses
        xor_clauses_count = dimacs_mod.num_xor_clauses

        # Assumptions
        if assumed_keys > 0:
            round_indices = range(rounds - assumed_keys, rounds)
            assumed_map = {idx: trailing_subkeys[i] for i, idx in enumerate(round_indices)}
            assumptions = assumptions_for_round_keys(model, assumed_map)
        else:
            assumptions = ()

    elif cipher == "simon32_64":
        model = SimonSatModel.from_known_pairs(known_pairs, rounds=rounds)
        model_build_seconds = time.perf_counter() - model_build_start

        variables_count = model.variable_count
        cnf_clauses_count = model.cnf_clause_count
        xor_clauses_count = model.xor_clause_count

        if assumed_keys > 0:
            assumptions = model.assumptions_for_trailing_round_subkeys(trailing_subkeys)
        else:
            assumptions = ()
    else:
        raise ValueError(f"Unknown cipher: {cipher}")

    # 3. Advisory solver call
    remaining_time = max(0.001, cutoff_seconds - model_build_seconds)
    solve_start = time.perf_counter()
    solved = model.solve(assumptions, time_limit=remaining_time)
    solve_seconds = time.perf_counter() - solve_start

    # 4. Result interpretation and independent verification
    verification_seconds = 0.0
    recovered_key = None
    enc_fn = speck_encrypt if cipher == "speck32_64" else simon_encrypt

    if solved.satisfiable and solved.solution is not None:
        v_start = time.perf_counter()
        recovered_key = model.master_key_from_solution(solved.solution)

        # Independent encryption verification on all known attack pairs + fresh blocks
        all_blocks_match = True
        for pt, ct in known_pairs:
            if enc_fn(pt, recovered_key, rounds) != ct:
                all_blocks_match = False
                break
        if all_blocks_match:
            for vpt, vct in ver_pairs:
                if enc_fn(vpt, recovered_key, rounds) != vct:
                    all_blocks_match = False
                    break

        verification_seconds = time.perf_counter() - v_start

        if all_blocks_match:
            if recovered_key == master_key:
                outcome = "verified_recovery"
            else:
                outcome = "spurious_sat_distinct_key"
        else:
            outcome = "verification_failure"

    elif solved.status == "unsat":
        outcome = "proven_unsat"
    elif solved.timed_out is True or solved.status in {"solver_timeout", "unsat_or_timeout"}:
        outcome = "solver_timeout"
    else:
        outcome = f"solver_{solved.status}"

    worker_total_seconds = time.perf_counter() - worker_start_perf

    # 5. Persist worker telemetry receipt
    receipts_dir = repo_root / "runtime" / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    receipt_file = receipts_dir / f"{cell_id}.json"
    save_worker_receipt(receipt_file, telemetry)

    return {
        "cell_id": cell_id,
        "cipher": cipher,
        "rounds": rounds,
        "assumed_keys": assumed_keys,
        "observations": observations,
        "instance_index": payload["instance_index"],
        "master_key": list(master_key),
        "recovered_master_key": list(recovered_key) if recovered_key is not None else None,
        "outcome": outcome,
        "solve_status": solved.status,
        "model_build_seconds": model_build_seconds,
        "solve_wall_seconds": solve_seconds,
        "verification_seconds": verification_seconds,
        "worker_total_seconds": worker_total_seconds,
        "variables_count": variables_count,
        "cnf_clauses_count": cnf_clauses_count,
        "xor_clauses_count": xor_clauses_count,
        "receipt_file": str(receipt_file),
    }


def run_cell_with_parent_watchdog(
    payload: dict[str, Any],
    repo_root: Path = ROOT,
    worker_cmd: list[str] | None = None,
) -> dict[str, Any]:
    """Execute one cell with strict parent monotonic deadline and process group isolation."""
    cell_id = payload["cell_id"]
    cutoff = float(payload.get("cutoff_seconds", PER_CELL_CUTOFF_SECONDS))

    cmd = worker_cmd or [sys.executable, "-m", "neural_sat_speck.experiments.depth_map_runner", "--worker"]
    payload_bytes = json.dumps(payload).encode("utf-8")

    start_mono = time.monotonic()
    start_utc = datetime.now(UTC).isoformat()
    timed_out = False
    stdout_text = ""
    stderr_text = ""
    returncode = None

    # Spawn in fresh process group (start_new_session=True)
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        cwd=repo_root,
    )
    pgid = process.pid

    try:
        stdout_raw, stderr_raw = process.communicate(input=payload_bytes, timeout=cutoff)
        returncode = process.returncode
        stdout_text = stdout_raw.decode("utf-8", errors="replace")
        stderr_text = stderr_raw.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout_raw, stderr_raw = process.communicate()
        returncode = process.returncode
        stdout_text = stdout_raw.decode("utf-8", errors="replace")
        stderr_text = stderr_raw.decode("utf-8", errors="replace")

    parent_elapsed_mono = time.monotonic() - start_mono

    # Strict evaluation of deadline
    if timed_out or parent_elapsed_mono >= cutoff:
        outcome = "parent_watchdog_timeout"
        worker_record: dict[str, Any] = {}
    elif returncode != 0:
        outcome = "worker_exception"
        worker_record = {}
    else:
        try:
            worker_record = json.loads(stdout_text)
            outcome = worker_record.get("outcome", "worker_exception")

            if outcome == "proven_unsat":
                raise RuntimeError(
                    f"CORRECTNESS ALARM: Solver returned UNSAT on cell {cell_id} "
                    f"where assigned ground-truth master key satisfies the model!"
                )

            # Verify worker receipt
            receipt_file = Path(worker_record["receipt_file"])
            load_and_verify_worker_receipt(
                receipt_file,
                expected_pid=process.pid,
                expected_git_commit=payload["expected_git_commit"],
            )
        except RuntimeError:
            raise
        except Exception as exc:
            outcome = "worker_exception"
            stderr_text = f"{stderr_text}\nReceipt/WorkerDecodeError: {exc}"


    return {
        "cell_id": cell_id,
        "cipher": payload["cipher"],
        "rounds": payload["rounds"],
        "assumed_keys_count": payload["assumed_keys"],
        "nominal_residual_depth": payload["nominal_residual_depth"],
        "observations": payload["observations"],
        "instance_index": payload["instance_index"],
        "master_key": payload["master_key"],
        "outcome": outcome,
        "solve_status": worker_record.get("solve_status", outcome),
        "parent_wall_seconds": parent_elapsed_mono,
        "solve_wall_seconds": worker_record.get("solve_wall_seconds", 0.0),
        "model_build_seconds": worker_record.get("model_build_seconds", 0.0),
        "verification_seconds": worker_record.get("verification_seconds", 0.0),
        "variables_count": worker_record.get("variables_count", 0),
        "cnf_clauses_count": worker_record.get("cnf_clauses_count", 0),
        "xor_clauses_count": worker_record.get("xor_clauses_count", 0),
        "verified_recovery": outcome == "verified_recovery",
        "recovered_master_key": worker_record.get("recovered_master_key"),
        "receipt_file": worker_record.get("receipt_file"),
        "stderr_tail": stderr_text[-1000:] if stderr_text else None,
        "completed_at": datetime.now(UTC).isoformat(),
    }


class AtomicDepthJournal:
    """Manages atomic persistence and sidecar SHA-256 for the depth map study."""

    def __init__(self, journal_path: Path) -> None:
        self.path = journal_path
        self.sidecar = journal_path.with_name(f"{journal_path.name}.sha256")
        self.data: dict[str, Any] = {
            "schema": "cross_cipher_residual_depth_map_v1",
            "created_at": datetime.now(UTC).isoformat(),
            "manifest_id": "cross_cipher_residual_depth_map_v1",
            "total_cells_declared": TOTAL_CELLS_COUNT,
            "events": [],
            "cells": {},
        }
        self.load_or_initialize()

    def load_or_initialize(self) -> None:
        if self.path.is_file():
            # Verify sidecar
            if not self.sidecar.is_file():
                raise FileNotFoundError(f"Journal sidecar missing for existing journal: {self.sidecar}")
            content = self.path.read_bytes()
            expected_hash = hashlib.sha256(content).hexdigest()
            sidecar_hash = self.sidecar.read_text(encoding="utf-8").strip().split()[0]
            if expected_hash != sidecar_hash:
                raise ValueError("Journal SHA-256 sidecar mismatch on reload!")
            self.data = json.loads(content.decode("utf-8"))

            # Audit start-without-finish cells
            started_cells = [
                ev["cell_id"] for ev in self.data["events"] if ev["event"] == "cell_started"
            ]
            finished_cells = set(self.data["cells"].keys())

            for c_id in started_cells:
                if c_id not in finished_cells:
                    # Mark interrupted cell
                    self.data["cells"][c_id] = {
                        "cell_id": c_id,
                        "outcome": "infrastructure_interrupted",
                        "solve_status": "interrupted",
                        "verified_recovery": False,
                        "interrupted_at": datetime.now(UTC).isoformat(),
                    }
                    self.data["events"].append({
                        "event": "cell_interrupted",
                        "cell_id": c_id,
                        "timestamp": datetime.now(UTC).isoformat(),
                    })
            self._save()

    def record_start(self, cell_id: str) -> None:
        self.data["events"].append({
            "event": "cell_started",
            "cell_id": cell_id,
            "timestamp": datetime.now(UTC).isoformat(),
        })
        self._save()

    def record_finish(self, cell_id: str, record: dict[str, Any]) -> None:
        self.data["cells"][cell_id] = record
        self.data["events"].append({
            "event": "cell_finished",
            "cell_id": cell_id,
            "outcome": record["outcome"],
            "timestamp": datetime.now(UTC).isoformat(),
        })
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        raw_bytes = json.dumps(self.data, indent=2).encode("utf-8")
        temp_target = self.path.with_suffix(".tmp")
        temp_target.write_bytes(raw_bytes)
        temp_target.replace(self.path)

        digest = hashlib.sha256(raw_bytes).hexdigest()
        self.sidecar.write_text(f"{digest}  {self.path.name}\n", encoding="utf-8")

    def is_cell_completed(self, cell_id: str) -> bool:
        return cell_id in self.data["cells"]

    @property
    def completed_count(self) -> int:
        return len(self.data["cells"])


def run_depth_map(
    manifest_path: Path = DEFAULT_MANIFEST,
    dataset_path: Path = DEFAULT_DATASET,
    results_path: Path = DEFAULT_RESULTS,
    repo_root: Path = ROOT,
    tmpdir: Path = DEFAULT_TMPDIR,
    worker_cmd: list[str] | None = None,
    stop_on_first_error: bool = True,
) -> dict[str, Any]:
    """Execute the full 192-cell bounded cross-cipher depth map."""
    print("=" * 70)
    print("Cross-Cipher Residual-Depth Bounded Empirical Map Runner")
    print(f"Time (AWST): {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 70)

    # 1. Validate manifest
    manifest = validate_manifest_for_execution(manifest_path, repo_root=repo_root)
    git_commit = get_git_commit(repo_root)
    dataset_sha256 = compute_file_sha256(dataset_path)

    # Ensure persistent directories
    tmpdir.mkdir(parents=True, exist_ok=True)
    DEFAULT_RECEIPTS.mkdir(parents=True, exist_ok=True)

    # 2. Load dataset and schedule
    dataset = load_dataset(dataset_path)
    schedule = build_counterbalanced_schedule()
    journal = AtomicDepthJournal(results_path)

    print(f"Git commit: {git_commit}")
    print(f"Dataset SHA-256: {dataset_sha256}")
    print(f"Total cells declared: {len(schedule)}")
    print(f"Already completed: {journal.completed_count}/{len(schedule)}")

    # 3. Serial cell dispatch
    for index, cell in enumerate(schedule, 1):
        cell_id = cell["cell_id"]
        if journal.is_cell_completed(cell_id):
            continue

        print(f"\n[{index:03d}/{len(schedule)}] Starting cell {cell_id} ...")
        payload = prepare_cell_payload(cell, dataset, repo_root, tmpdir, git_commit)

        journal.record_start(cell_id)

        try:
            cell_result = run_cell_with_parent_watchdog(payload, repo_root=repo_root, worker_cmd=worker_cmd)
        except RuntimeError as alarm:
            print(f"\nCRITICAL STOP: {alarm}")
            if stop_on_first_error:
                raise
            break

        journal.record_finish(cell_id, cell_result)

        print(
            f"  -> Outcome: {cell_result['outcome']} "
            f"| Parent: {cell_result['parent_wall_seconds']:.2f}s "
            f"| Solve: {cell_result['solve_wall_seconds']:.2f}s "
            f"| Recovered: {cell_result['verified_recovery']}"
        )

    print("\n" + "=" * 70)
    print(f"Depth map execution run finished. Total cells recorded: {journal.completed_count}/192")
    print(f"Results file: {results_path}")
    print("=" * 70)
    return journal.data


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-cipher residual depth map runner & worker.")
    parser.add_argument("--worker", action="store_true", help="Run in worker mode (reads stdin)")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Manifest path")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Dataset path")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="Results output path")
    parser.add_argument("--tmpdir", type=Path, default=DEFAULT_TMPDIR, help="Persistent worker TMPDIR")
    args = parser.parse_args()

    if args.worker:
        raw_input = sys.stdin.read()
        payload = json.loads(raw_input)
        result = execute_worker_task(payload)
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()
    else:
        run_depth_map(
            manifest_path=args.manifest,
            dataset_path=args.dataset,
            results_path=args.results,
            tmpdir=args.tmpdir,
        )


if __name__ == "__main__":
    main()
