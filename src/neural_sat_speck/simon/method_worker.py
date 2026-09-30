"""Isolated method worker and execution harness for SIMON32/64 R=10 comparisons.

Supports all eight compared methods under common global resource accounting:
- M1: fully_neural_1parent (VV9 -> VV8, 128 candidate tuples)
- M2: neural_classical_hybrid_8parent (VV9 -> 7R DDT, 1,024 candidate tuples)
- M3: fully_classical_8parent (8R on-demand queried rows -> 7R DDT, 1,024 candidate tuples)
- M4a: random_control_128 (128 uniform random tuples)
- M4b: random_control_1024 (1,024 uniform random tuples)
- M5: sat_only_n4 (4-observation unguided formula control)
- M6: sat_only_n64 (64-observation unguided all-observation control)
- M7: faithful_b1_hybrid (faithful SIMON B1 ranker -> 7R DDT, 1,024 candidate tuples)
Plus diagnostic oracle:
- M_oracle: oracle_a2_n4 (oracle trailing keys control for testing harness mechanics)

Test-only methods:
- test_mock_queue: accepts mock candidate queue for testing
- test_slow_ranker: deliberately slow ranker to test watchdog termination
- test_slow_solver: deliberately slow solver to test watchdog termination
- test_verification_failure: tests solver returning SAT that fails verification
- test_unknown_solver_status: tests solver returning unknown/unsat_or_timeout status

Tuple convention:
- Chronological order throughout: (k8, k9).
- Stage 1 ranks final-round subkey k9 (parent).
- Stage 2 ranks penultimate-round subkey k8 (child) conditioned on k9.
- SAT assumptions require chronological order: (k8, k9).
- Conversion from ranking: (parent_k9, child_k8) -> (child_k8, parent_k9).

Invariants:
1. Eight comparators cannot access oracle keys or mock candidate queues.
2. Parent process watchdog (300.0s default) covering startup, table/checkpoint loading, ranking, formula build, every solve, and verification.
3. Strict per-method-instance solver isolation (fresh pycryptosat instance per method and after timeout).
4. Measured wall time and peak memory (never peak_rss_gib=0.0).
5. Fine-grained outcome taxonomy with rebuild-after-unresolved-candidate and verification-failure handling.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import platform
import random
import resource
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

from neural_sat_speck.simon.attack_instance import (
    DEFAULT_SIMON_INPUT_DIFFERENCE,
    KnownPair,
)
from neural_sat_speck.simon.cipher import Block
from neural_sat_speck.simon.classical_ranking import (
    FullPredictedDdt,
    rank_classical_domain,
    rank_stage1_ondemand,
)
from neural_sat_speck.simon.comparison_input import (
    SimonComparisonInput,
    derive_random_control_seed,
)
from neural_sat_speck.simon.faithful_b1_model import (
    SimonPairGroupRanker,
    generate_b1_hybrid_8parent_queue,
)
from neural_sat_speck.simon.published_vv import load_published_vv_checkpoint
from neural_sat_speck.simon.published_vv_ranking import rank_neural_domain
from neural_sat_speck.simon.sat_model import SatSolveResult, SimonSatModel
from neural_sat_speck.simon.verification import (
    SimonVerificationResult,
    verify_simon_master_key,
)

EIGHT_COMPARATORS: Final[frozenset[str]] = frozenset(
    {
        "fully_neural_1parent",
        "neural_classical_hybrid_8parent",
        "faithful_b1_hybrid",
        "fully_classical_8parent",
        "random_control_128",
        "random_control_1024",
        "sat_only_n4",
        "sat_only_n128",
    }
)

SEVEN_COMPARATORS: Final[frozenset[str]] = frozenset(
    EIGHT_COMPARATORS - {"faithful_b1_hybrid"}
)

COMPARATOR_METHODS: Final[frozenset[str]] = frozenset(
    EIGHT_COMPARATORS | {"faithful_b1_8parent", "sat_only_n64"}
)

DIAGNOSTIC_ORACLE: Final[str] = "oracle_a2_n4"
DIAGNOSTIC_ORACLES: Final[frozenset[str]] = frozenset(
    {"oracle_a2_n4", "oracle_2key_n4"}
)

TEST_ONLY_METHODS: Final[frozenset[str]] = frozenset(
    {
        "test_mock_queue",
        "test_slow_ranker",
        "test_hanging_ranking_child",
        "test_hanging_tf_child",
        "test_slow_solver",
        "test_slow_rebuild",
        "test_slow_verification",
        "test_verification_failure",
        "test_multiple_verification_failures",
        "test_unknown_solver_status",
        "test_sat_only",
    }
)

ALL_PERMITTED_METHODS: Final[frozenset[str]] = frozenset(
    COMPARATOR_METHODS | DIAGNOSTIC_ORACLES | TEST_ONLY_METHODS
)


def get_process_peak_rss_gib() -> float:
    """Measure peak RSS in GiB honestly across concurrently alive process tree."""
    scale = (1024.0 ** 3) if platform.system() == "Darwin" else (1024.0 ** 2)
    self_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    child_rss = float(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    # Honest measurement without artificial floors
    return (self_rss + child_rss) / scale


@dataclass
class _RecoveredProgress:
    ranking_wall_seconds: float = 0.0
    solving_wall_seconds: float = 0.0
    verification_wall_seconds: float = 0.0
    candidates_tested_count: int = 0
    candidates_remaining_count: int = 0
    attempts: list[CandidateAttemptLog] = field(default_factory=list)
    in_flight_attempt: CandidateAttemptLog | None = None
    queue_sample: tuple[tuple[int, int], ...] = ()
    full_queue: tuple[tuple[int, int], ...] = ()
    peak_rss_gib: float = 0.0
    last_phase: str = "initialization"
    child_pids: list[int] = field(default_factory=list)
    full_pairs_used: int | None = None
    total_queue_count: int = 0
    journal_error: str | None = None


def _log_progress_event(progress_file: str | None, event: dict[str, Any]) -> None:
    if not progress_file:
        return
    with open(progress_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _read_progress_file(
    path: str | Path | None,
    parent_elapsed: float | None = None,
    *,
    stopped_at_monotonic: float | None = None,
) -> _RecoveredProgress:
    rec = _RecoveredProgress()
    if path is None:
        return rec
    if not os.path.exists(path):
        rec.journal_error = f"Progress journal missing: {path}"
        return rec
    last_attempt_start: dict[str, Any] | None = None
    last_candidate_solve_complete: dict[str, Any] | None = None
    last_solving_start: dict[str, Any] | None = None
    last_verification_start: dict[str, Any] | None = None
    last_rebuild_start: dict[str, Any] | None = None
    last_construction_start: dict[str, Any] | None = None

    try:
        with open(path, encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                    if not isinstance(ev, dict) or not isinstance(ev.get("event"), str):
                        raise ValueError("event must be a JSON object with an event type")
                except (json.JSONDecodeError, ValueError) as exc:
                    rec.journal_error = f"Malformed progress journal line {line_number}: {exc}"
                    continue
                event_type = ev["event"]
                try:
                    _apply_progress_event(rec, ev, event_type, line_number)
                except (KeyError, IndexError, TypeError, ValueError, OverflowError) as exc:
                    rec.journal_error = f"Invalid progress journal line {line_number}: {exc}"
                    continue
                if event_type == "attempt_start":
                    if last_attempt_start is not None:
                        rec.journal_error = f"Overlapping attempt_start at line {line_number}"
                    last_attempt_start = ev
                    last_candidate_solve_complete = None
                    last_verification_start = None
                    last_rebuild_start = None
                elif event_type == "candidate_solve_complete":
                    last_candidate_solve_complete = ev
                elif event_type == "attempt":
                    last_attempt_start = None
                    last_candidate_solve_complete = None
                    last_verification_start = None
                elif event_type == "rebuild_complete":
                    last_rebuild_start = None
                elif event_type == "initial_formula_construction_complete":
                    last_construction_start = None
                elif event_type == "solve_complete":
                    last_solving_start = None
                    rec.solving_wall_seconds = float(ev["solving_wall_seconds"])
                elif event_type == "phase":
                    phase_name = ev.get("phase")
                    if phase_name == "solving" and ev.get("solving_kind") == "unguided":
                        last_solving_start = ev
                    elif phase_name == "verification":
                        last_verification_start = ev
                    elif phase_name == "rebuilding":
                        last_rebuild_start = ev
                    elif phase_name == "initial_formula_construction":
                        last_construction_start = ev
                elif event_type == "completed":
                    last_solving_start = None
                    last_verification_start = None
                    last_rebuild_start = None
                    last_construction_start = None
                    last_attempt_start = None
                    last_candidate_solve_complete = None
    except OSError as exc:
        rec.journal_error = f"Cannot read progress journal: {exc}"

    # Handle interruptions across different phases:
    if last_attempt_start is not None:
        # Interrupted during a candidate attempt (solving or verification)
        if last_candidate_solve_complete is not None and rec.last_phase == "verification":
            # Solve completed before interruption; interrupted during verification
            try:
                cand_solve_time = float(last_candidate_solve_complete["solve_wall_seconds"])
                rec.in_flight_attempt = CandidateAttemptLog(
                    candidate_k8=int(last_attempt_start["candidate_k8"]),
                    candidate_k9=int(last_attempt_start["candidate_k9"]),
                    solve_wall_seconds=cand_solve_time,
                    status="in_flight",
                    rebuild_occurred=False,
                )
                rec.solving_wall_seconds += cand_solve_time
                if last_verification_start is not None:
                    verif_time = _interrupted_duration(
                        last_verification_start, parent_elapsed, stopped_at_monotonic,
                    )
                    rec.verification_wall_seconds += verif_time
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                rec.journal_error = f"Invalid in-flight attempt during verification: {exc}"
        else:
            # Interrupted during candidate solving
            try:
                in_flight_time = _interrupted_duration(
                    last_attempt_start, parent_elapsed, stopped_at_monotonic,
                )
                rec.in_flight_attempt = CandidateAttemptLog(
                    candidate_k8=int(last_attempt_start["candidate_k8"]),
                    candidate_k9=int(last_attempt_start["candidate_k9"]),
                    solve_wall_seconds=in_flight_time,
                    status="in_flight",
                    rebuild_occurred=False,
                )
                rec.solving_wall_seconds += in_flight_time
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                rec.journal_error = f"Invalid in-flight attempt: {exc}"
    elif last_solving_start is not None and rec.last_phase == "solving":
        # Unguided SAT solving interrupted
        try:
            # perf_counter has a shared monotonic origin across processes on
            # supported runtimes. Child-relative elapsed time cannot be
            # subtracted from parent elapsed time: that would charge startup
            # to solving. Only an explicitly unguided phase uses this path.
            rec.solving_wall_seconds = _interrupted_duration(
                last_solving_start, parent_elapsed, stopped_at_monotonic,
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            rec.journal_error = f"Invalid in-flight solving phase: {exc}"
    elif last_verification_start is not None and rec.last_phase == "verification":
        # Unguided SAT verification interrupted (solving completed earlier)
        try:
            rec.verification_wall_seconds += _interrupted_duration(
                last_verification_start, parent_elapsed, stopped_at_monotonic,
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            rec.journal_error = f"Invalid in-flight unguided verification phase: {exc}"
    elif last_rebuild_start is not None and rec.last_phase == "rebuilding":
        # Interrupted during guided rebuild phase
        try:
            rec.solving_wall_seconds += _interrupted_duration(
                last_rebuild_start, parent_elapsed, stopped_at_monotonic,
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            rec.journal_error = f"Invalid in-flight rebuilding phase: {exc}"
    elif last_construction_start is not None and rec.last_phase == "initial_formula_construction":
        # Interrupted during guided initial formula construction
        try:
            rec.solving_wall_seconds += _interrupted_duration(
                last_construction_start, parent_elapsed, stopped_at_monotonic,
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            rec.journal_error = f"Invalid in-flight initial formula construction phase: {exc}"

    in_flight_count = 1 if rec.in_flight_attempt is not None else 0
    rec.candidates_tested_count = len(rec.attempts)
    rec.candidates_remaining_count = max(0, rec.total_queue_count - rec.candidates_tested_count - in_flight_count)

    if rec.solving_wall_seconds == 0.0 and rec.attempts:
        rec.solving_wall_seconds = sum(a.solve_wall_seconds for a in rec.attempts)
    return rec


def _interrupted_duration(
    event: dict[str, Any], parent_elapsed: float | None, stopped_at: float | None,
) -> float:
    timestamp = event["started_at_monotonic"]
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
        raise ValueError("monotonic solving timestamp must be a JSON number")
    started_at = float(timestamp)
    stopped_at = time.perf_counter() if stopped_at is None else float(stopped_at)
    if not math.isfinite(started_at) or not math.isfinite(stopped_at):
        raise ValueError("non-finite monotonic solving timestamp")
    if not 0.0 <= started_at <= stopped_at:
        raise ValueError("invalid monotonic solving timestamp")
    duration = stopped_at - started_at
    if parent_elapsed is not None:
        if not math.isfinite(parent_elapsed) or parent_elapsed < 0.0:
            raise ValueError("invalid parent duration")
        if duration > parent_elapsed + 1e-6:
            raise ValueError("solving timestamp precedes parent execution")
        duration = min(duration, parent_elapsed)
    return duration


def _apply_progress_event(
    rec: _RecoveredProgress, ev: dict[str, Any], event_type: str, line_number: int
) -> None:
    """Apply one validated journal record; invalid records must remain auditable."""
    if event_type == "child_pid":
        rec.child_pids.append(int(ev["pid"]))
    elif event_type == "queue":
        rec.ranking_wall_seconds = float(ev.get("ranking_wall_seconds", 0.0))
        rec.total_queue_count = int(ev["total_queue_count"])
        rec.queue_sample = tuple((int(t[0]), int(t[1])) for t in ev.get("queue_sample", ()))
        rec.full_queue = tuple((int(t[0]), int(t[1])) for t in ev.get("full_queue", ()))
        if len(rec.full_queue) != rec.total_queue_count:
            raise ValueError("full queue length differs from recorded queue count")
        rec.peak_rss_gib = max(rec.peak_rss_gib, float(ev.get("peak_rss_gib", 0.0)))
        if "full_pairs_used" in ev:
            rec.full_pairs_used = int(ev["full_pairs_used"])
    elif event_type == "attempt_start":
        rec.peak_rss_gib = max(rec.peak_rss_gib, float(ev.get("peak_rss_gib", 0.0)))
    elif event_type == "attempt":
        rec.attempts.append(
            CandidateAttemptLog(
                candidate_k8=int(ev["candidate_k8"]),
                candidate_k9=int(ev["candidate_k9"]),
                solve_wall_seconds=float(ev["solve_wall_seconds"]),
                status=str(ev["status"]),
                rebuild_occurred=bool(ev["rebuild_occurred"]),
            )
        )
        rec.solving_wall_seconds = max(
            rec.solving_wall_seconds, float(ev.get("current_solving_wall_seconds", 0.0))
        )
        rec.verification_wall_seconds = max(
            rec.verification_wall_seconds, float(ev.get("current_verification_wall_seconds", 0.0))
        )
        rec.peak_rss_gib = max(rec.peak_rss_gib, float(ev.get("peak_rss_gib", 0.0)))
    elif event_type == "rebuild_complete":
        if not rec.attempts:
            raise ValueError("rebuild_complete without a prior attempt")
        previous = rec.attempts[-1]
        if (previous.candidate_k8, previous.candidate_k9) != (
            int(ev["candidate_k8"]), int(ev["candidate_k9"])
        ):
            raise ValueError("rebuild_complete candidate does not match prior attempt")
        rec.attempts[-1] = CandidateAttemptLog(
            previous.candidate_k8, previous.candidate_k9, previous.solve_wall_seconds,
            previous.status, True,
        )
    elif event_type == "candidate_solve_complete":
        pass
    elif event_type == "initial_formula_construction_complete":
        pass
    elif event_type == "solve_complete":
        duration = float(ev["solving_wall_seconds"])
        if not math.isfinite(duration) or duration < 0.0:
            raise ValueError("invalid solve_complete duration")
        rec.solving_wall_seconds = duration
    elif event_type == "phase":
        rec.last_phase = str(ev.get("phase", rec.last_phase))
        if "full_pairs_used" in ev:
            rec.full_pairs_used = int(ev["full_pairs_used"])
    elif event_type == "completed":
        if "solving_wall_seconds" in ev:
            duration = float(ev["solving_wall_seconds"])
            if not math.isfinite(duration) or duration < 0.0:
                raise ValueError("invalid completed solving duration")
            rec.solving_wall_seconds = duration
        if "verification_wall_seconds" in ev:
            v_duration = float(ev["verification_wall_seconds"])
            if not math.isfinite(v_duration) or v_duration < 0.0:
                raise ValueError("invalid completed verification duration")
            rec.verification_wall_seconds = v_duration
    elif event_type not in {"init", "exception"}:
        raise ValueError(f"unknown event {event_type!r} at line {line_number}")


@dataclass(frozen=True)
class CandidateAttemptLog:
    candidate_k8: int
    candidate_k9: int
    solve_wall_seconds: float
    status: str
    rebuild_occurred: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateAttemptLog:
        return cls(
            candidate_k8=int(data["candidate_k8"]),
            candidate_k9=int(data["candidate_k9"]),
            solve_wall_seconds=float(data["solve_wall_seconds"]),
            status=str(data["status"]),
            rebuild_occurred=bool(data["rebuild_occurred"]),
        )


@dataclass(frozen=True)
class MethodWorkerOutcome:
    method: str
    outcome: str  # verified_recovery, sat_unverified, proven_unsat, candidate_timeout, queue_exhausted, global_deadline, ranking_timeout, verification_failure, exception, solver_timeout
    complete_online_wall_seconds: float
    ranking_wall_seconds: float
    solving_wall_seconds: float
    verification_wall_seconds: float
    candidates_tested_count: int
    candidates_remaining_count: int
    recovered_master_key: tuple[int, int, int, int] | None
    verified_on_fresh_pairs: bool
    candidate_attempts: tuple[CandidateAttemptLog, ...]
    peak_rss_gib: float
    error_message: str | None = None
    queue_sample: tuple[tuple[int, int], ...] = ()
    full_queue: tuple[tuple[int, int], ...] = ()
    in_flight_attempt: CandidateAttemptLog | None = None
    child_pids: tuple[int, ...] = ()
    full_pairs_used: int | None = None

    @property
    def untested_remaining_count(self) -> int:
        """Untested candidates remaining (candidates never started)."""
        return self.candidates_remaining_count

    @property
    def all_attempts(self) -> tuple[CandidateAttemptLog, ...]:
        """All candidate attempts including in-flight attempt if present."""
        if self.in_flight_attempt is not None:
            return self.candidate_attempts + (self.in_flight_attempt,)
        return self.candidate_attempts

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["candidate_attempts"] = [a.to_dict() for a in self.candidate_attempts]
        if self.in_flight_attempt is not None:
            d["in_flight_attempt"] = self.in_flight_attempt.to_dict()
        if self.recovered_master_key is not None:
            d["recovered_master_key"] = list(self.recovered_master_key)
        d["queue_sample"] = [list(t) for t in self.queue_sample]
        d["full_queue"] = [list(t) for t in self.full_queue]
        d["child_pids"] = list(self.child_pids)
        d["full_pairs_used"] = self.full_pairs_used
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MethodWorkerOutcome:
        attempts = tuple(CandidateAttemptLog.from_dict(a) for a in data.get("candidate_attempts", ()))
        in_flight_raw = data.get("in_flight_attempt")
        in_flight = CandidateAttemptLog.from_dict(in_flight_raw) if in_flight_raw is not None else None
        key_raw = data.get("recovered_master_key")
        recovered_key = (
            (int(key_raw[0]), int(key_raw[1]), int(key_raw[2]), int(key_raw[3]))
            if key_raw is not None
            else None
        )
        sample_raw = data.get("queue_sample", ())
        sample = tuple((int(t[0]), int(t[1])) for t in sample_raw)
        full_raw = data.get("full_queue", ())
        full_q = tuple((int(t[0]), int(t[1])) for t in full_raw)
        child_pids_raw = data.get("child_pids", ())
        child_pids = tuple(int(p) for p in child_pids_raw)
        return cls(
            method=str(data["method"]),
            outcome=str(data["outcome"]),
            complete_online_wall_seconds=float(data["complete_online_wall_seconds"]),
            ranking_wall_seconds=float(data["ranking_wall_seconds"]),
            solving_wall_seconds=float(data["solving_wall_seconds"]),
            verification_wall_seconds=float(data["verification_wall_seconds"]),
            candidates_tested_count=int(data["candidates_tested_count"]),
            candidates_remaining_count=int(data["candidates_remaining_count"]),
            recovered_master_key=recovered_key,
            verified_on_fresh_pairs=bool(data["verified_on_fresh_pairs"]),
            candidate_attempts=attempts,
            peak_rss_gib=float(data["peak_rss_gib"]),
            error_message=data.get("error_message"),
            queue_sample=sample,
            full_queue=full_q,
            in_flight_attempt=in_flight,
            child_pids=child_pids,
            full_pairs_used=data.get("full_pairs_used"),
        )


def generate_random_tuples(count: int, seed: int) -> list[tuple[int, int]]:
    """Draw count distinct (k8, k9) tuples uniformly at random."""
    rng = random.Random(seed)
    seen = set()
    tuples = []
    while len(tuples) < count:
        cand = (rng.randint(0, 0xFFFF), rng.randint(0, 0xFFFF))
        if cand not in seen:
            seen.add(cand)
            tuples.append(cand)
    return tuples


def generate_classical_8parent_queue(
    pairs: Sequence[tuple[Block, Block]],
    ddt_7r_path: Path | str = "/private/tmp/simon_predicted_ddt_7r.bin",
    tool_path: Path | str = "/private/tmp/simon_queried_row_8r",
    table_7r: FullPredictedDdt | None = None,
    *,
    return_diagnostics: bool = False,
) -> list[tuple[int, int]] | tuple[list[tuple[int, int]], dict[str, Any]]:
    """Generate 1,024 candidate tuples (k8, k9) in chronological SAT order from pairs alone."""
    t_load0 = time.perf_counter()
    if table_7r is None:
        table_7r = FullPredictedDdt(ddt_7r_path, 7)
    load_seconds = time.perf_counter() - t_load0

    stage1, diag1 = rank_stage1_ondemand(pairs, tool_path=tool_path, ddt_7r_path=ddt_7r_path)
    parents = [int(k) for k in stage1.order[:8]]  # Final-round subkeys k9

    t_s2_0 = time.perf_counter()
    stage2_results = [
        rank_classical_domain(
            pairs,
            table_7r,
            final_key_candidate=parent,
            expected_rounds=7,
        )
        for parent in parents
    ]
    stage2_seconds = time.perf_counter() - t_s2_0

    t_q0 = time.perf_counter()
    raw_tuples = []
    for r1, parent_k9 in enumerate(parents):
        children_k8 = stage2_results[r1].order[:128]  # Penultimate-round subkeys k8
        for r2, child_k8 in enumerate(children_k8):
            raw_tuples.append((r1 + r2, r1, r2, int(parent_k9), int(child_k8)))

    # Sort by (sum_rank, r1, r2, parent_k9, child_k8)
    raw_tuples.sort()

    seen = set()
    queue = []
    duplicate_count = 0
    for _, _, _, parent_k9, child_k8 in raw_tuples:
        cand_tuple = (int(child_k8), int(parent_k9))  # Chronological order: (k8, k9)
        if cand_tuple in seen:
            duplicate_count += 1
            continue
        seen.add(cand_tuple)
        queue.append(cand_tuple)
    queue_seconds = time.perf_counter() - t_q0

    if not return_diagnostics:
        return queue

    diagnostics = {
        "startup_and_7r_load_seconds": load_seconds,
        "stage1_seconds": stage1.total_seconds,
        "stage1_breakdown": diag1,
        "stage2_total_seconds": stage2_seconds,
        "stage2_per_parent_seconds": [d.total_seconds for d in stage2_results],
        "queue_construction_seconds": queue_seconds,
        "candidate_tuples_produced": len(queue),
        "candidate_duplicates_suppressed": duplicate_count,
        "table_7r": table_7r,
        "stage1_ranking": stage1,
        "stage2_rankings": stage2_results,
        "full_pairs_used": len(pairs),
    }
    return queue, diagnostics


def run_neural_ranking_subprocess(
    method: str,
    pairs: Sequence[tuple[Block, Block]],
    *,
    tf_python_path: Path | str = "/private/tmp/simon_tf_venv/bin/python3.12",
    worker_script_path: Path | str | None = None,
    ranking_time_limit: float = 120.0,
    checkpoint_dir: Path | str = "/private/tmp/bao-simon-audit/simonNDvsDD/SENet",
    ddt_7r_path: Path | str = "/private/tmp/simon_predicted_ddt_7r.bin",
    progress_file: str | None = None,
    return_diagnostics: bool = False,
) -> list[tuple[int, int]] | tuple[list[tuple[int, int]], dict[str, Any]]:
    """Execute neural ranking in the isolated TF environment and return candidate queue."""
    tf_python = Path(tf_python_path)
    if not tf_python.is_file():
        raise RuntimeError(
            f"Pinned TensorFlow Python binary not found at {tf_python}. "
            "Please ensure /private/tmp/simon_tf_venv is configured."
        )

    if worker_script_path is None:
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        script_file = repo_root / "scripts" / "run_neural_ranking_worker.py"
    else:
        script_file = Path(worker_script_path)

    if not script_file.is_file():
        raise RuntimeError(f"Neural ranking worker script not found at {script_file}")

    payload = {
        "method": method,
        "attack_pairs": [
            [[int(p[0][0]), int(p[0][1])], [int(p[1][0]), int(p[1][1])]]
            for p in pairs
        ],
        "checkpoint_dir": str(checkpoint_dir),
        "ddt_7r_path": str(ddt_7r_path),
        "time_limit": float(ranking_time_limit),
    }

    env = dict(os.environ)
    repo_src = str(Path(__file__).resolve().parent.parent.parent)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_src}:{existing_pythonpath}" if existing_pythonpath else repo_src

    proc = subprocess.Popen(
        [str(tf_python), str(script_file), "--stdin"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        # Note: start_new_session=True is intentionally NOT set so that the child
        # shares the worker process group, allowing the parent watchdog to kill
        # the entire process tree reliably.
    )

    if progress_file:
        _log_progress_event(progress_file, {"event": "child_pid", "pid": proc.pid})

    try:
        stdout, stderr = proc.communicate(
            input=json.dumps(payload),
            timeout=ranking_time_limit + 15.0,
        )
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass
        proc.communicate()
        raise TimeoutError(f"Neural ranking subprocess timed out after {ranking_time_limit}s")

    if proc.returncode != 0 and proc.returncode != 2:
        raise RuntimeError(f"Neural ranking worker exited with code {proc.returncode}: {stderr[-1000:]}")

    data = json.loads(stdout)
    if data.get("status") == "ranking_timeout":
        raise TimeoutError("Neural ranking subprocess reached its ranking limit")
    if data.get("status") != "ok":
        raise RuntimeError(f"Neural ranking failed: {data.get('error_message')}")

    if "full_pairs_used" in data:
        assert data["full_pairs_used"] == len(pairs), (
            f"Neural ranking subprocess evaluated {data['full_pairs_used']} pairs instead of {len(pairs)}"
        )
    else:
        data["full_pairs_used"] = len(pairs)

    queue = [(int(t[0]), int(t[1])) for t in data["queue"]]
    if return_diagnostics:
        return queue, data
    return queue


def generate_hybrid_8parent_queue(
    pairs: Sequence[tuple[Block, Block]],
    vv9_model: Any | None = None,
    ddt_7r_path: Path | str = "/private/tmp/simon_predicted_ddt_7r.bin",
    *,
    tf_python_path: Path | str = "/private/tmp/simon_tf_venv/bin/python3.12",
    worker_script_path: Path | str | None = None,
    ranking_time_limit: float = 120.0,
    progress_file: str | None = None,
    return_diagnostics: bool = False,
) -> list[tuple[int, int]] | tuple[list[tuple[int, int]], dict[str, Any]]:
    """Generate 1,024 hybrid candidate tuples (k8, k9) in chronological order from pairs alone."""
    if vv9_model is None:
        try:
            import tensorflow  # noqa: F401
            has_tf = True
        except ImportError:
            has_tf = False

        if not has_tf:
            return run_neural_ranking_subprocess(
                method="neural_classical_hybrid_8parent",
                pairs=pairs,
                tf_python_path=tf_python_path,
                worker_script_path=worker_script_path,
                ranking_time_limit=ranking_time_limit,
                ddt_7r_path=ddt_7r_path,
                progress_file=progress_file,
                return_diagnostics=return_diagnostics,
            )

        vv_dir = Path("/private/tmp/bao-simon-audit/simonNDvsDD/SENet")
        vv9_model, _ = load_published_vv_checkpoint(vv_dir / "ND_VV_Simon32_9R.h5", 9)
    table_7r = FullPredictedDdt(ddt_7r_path, 7)

    stage1 = rank_neural_domain(pairs, vv9_model)
    parents = [int(k) for k in stage1.order[:8]]  # Final-round subkeys k9

    stage2_results = [
        rank_classical_domain(
            pairs,
            table_7r,
            final_key_candidate=parent,
            expected_rounds=7,
        )
        for parent in parents
    ]

    raw_tuples = []
    for r1, parent_k9 in enumerate(parents):
        children_k8 = stage2_results[r1].order[:128]
        for r2, child_k8 in enumerate(children_k8):
            raw_tuples.append((r1 + r2, r1, r2, int(parent_k9), int(child_k8)))

    raw_tuples.sort()
    seen = set()
    queue = []
    for _, _, _, parent_k9, child_k8 in raw_tuples:
        cand_tuple = (int(child_k8), int(parent_k9))  # Chronological order: (k8, k9)
        if cand_tuple not in seen:
            seen.add(cand_tuple)
            queue.append(cand_tuple)

    if return_diagnostics:
        diagnostics = {
            "stage1_ranking": stage1,
            "stage2_rankings": stage2_results,
            "full_pairs_used": len(pairs),
            "candidate_tuples_produced": len(queue),
        }
        return queue, diagnostics
    return queue


def generate_neural_1parent_queue(
    pairs: Sequence[tuple[Block, Block]],
    vv9_model: Any | None = None,
    vv8_model: Any | None = None,
    *,
    tf_python_path: Path | str = "/private/tmp/simon_tf_venv/bin/python3.12",
    worker_script_path: Path | str | None = None,
    ranking_time_limit: float = 180.0,
    progress_file: str | None = None,
    return_diagnostics: bool = False,
) -> list[tuple[int, int]] | tuple[list[tuple[int, int]], dict[str, Any]]:
    """Generate 128 neural candidate tuples (k8, k9) in chronological order from pairs alone."""
    if vv9_model is None or vv8_model is None:
        try:
            import tensorflow  # noqa: F401
            has_tf = True
        except ImportError:
            has_tf = False

        if not has_tf:
            return run_neural_ranking_subprocess(
                method="fully_neural_1parent",
                pairs=pairs,
                tf_python_path=tf_python_path,
                worker_script_path=worker_script_path,
                ranking_time_limit=ranking_time_limit,
                progress_file=progress_file,
                return_diagnostics=return_diagnostics,
            )

        vv_dir = Path("/private/tmp/bao-simon-audit/simonNDvsDD/SENet")
        if vv9_model is None:
            vv9_model, _ = load_published_vv_checkpoint(vv_dir / "ND_VV_Simon32_9R.h5", 9)
        if vv8_model is None:
            vv8_model, _ = load_published_vv_checkpoint(vv_dir / "ND_VV_Simon32_8R.h5", 8)

    stage1 = rank_neural_domain(pairs, vv9_model)
    parent_k9 = int(stage1.order[0])  # Top 1 final-round subkey k9

    stage2 = rank_neural_domain(pairs, vv8_model, final_key_candidate=parent_k9)
    children_k8 = [int(k) for k in stage2.order[:128]]  # Top 128 penultimate-round subkeys k8

    queue = [(int(child_k8), parent_k9) for child_k8 in children_k8]  # Chronological order: (k8, k9)
    if return_diagnostics:
        diagnostics = {
            "stage1_ranking": stage1,
            "stage2_ranking": stage2,
            "full_pairs_used": len(pairs),
            "candidate_tuples_produced": len(queue),
        }
        return queue, diagnostics
    return queue


def generate_faithful_b1_hybrid_queue(
    pairs: Sequence[tuple[Block, Block]],
    b1_model: SimonPairGroupRanker | None = None,
    ddt_7r_path: Path | str = "/private/tmp/simon_predicted_ddt_7r.bin",
    table_7r: FullPredictedDdt | None = None,
    *,
    checkpoint_path: Path | str | None = None,
    checkpoint_sha256: str | None = None,
    screen_pairs: int = 16,
    full_pairs: int | None = None,
    parent_count: int = 8,
    children_per_parent: int = 128,
    return_diagnostics: bool = False,
) -> list[tuple[int, int]] | tuple[list[tuple[int, int]], dict[str, Any]]:
    """Generate 1,024 candidate tuples (k8, k9) using faithful SIMON B1 ranker + 7R DDT."""
    t_load0 = time.perf_counter()
    if b1_model is None:
        if checkpoint_path is None or checkpoint_sha256 is None:
            raise ValueError("Production B1 ranking requires a checkpoint path and SHA-256")
        if len(checkpoint_sha256) != 64 or any(c not in "0123456789abcdef" for c in checkpoint_sha256):
            raise ValueError("checkpoint_sha256 must be a lowercase 64-character SHA-256 digest")
        checkpoint = Path(checkpoint_path)
        digest = hashlib.sha256()
        with checkpoint.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if not hmac.compare_digest(digest.hexdigest(), checkpoint_sha256):
            raise ValueError("B1 checkpoint SHA-256 mismatch")
        import torch

        from neural_sat_speck.simon.b1_data import load_simon_b1_checkpoint

        b1_model, metadata = load_simon_b1_checkpoint(checkpoint)
        if b1_model.head_variant != "speck_b1":
            raise ValueError("Production B1 checkpoint has the wrong head architecture")
        if not isinstance(metadata, dict) or metadata.get("training_status") != "complete":
            raise ValueError("Production B1 checkpoint lacks completed-training provenance")
        if metadata.get("smoke_test") is True or metadata.get("performance_label") == "smoke_test":
            raise ValueError("Smoke-test checkpoints cannot be used in production comparison")
        for field_name in (
            "training_manifest_sha256",
            "train_dataset_checksum",
            "validation_dataset_checksum",
        ):
            digest_value = metadata.get(field_name)
            if not isinstance(digest_value, str) or len(digest_value) != 64 or any(
                char not in "0123456789abcdef" for char in digest_value
            ):
                raise ValueError(f"Production B1 checkpoint has invalid {field_name}")
        if type(metadata.get("selected_epoch")) is not int or metadata["selected_epoch"] < 1:
            raise ValueError("Production B1 checkpoint has invalid selected_epoch")
        if any(
            not torch.isfinite(tensor).all()
            for tensor in b1_model.state_dict().values()
            if tensor.is_floating_point()
        ):
            raise ValueError("Production B1 checkpoint contains non-finite weights or buffers")
    elif checkpoint_path is not None or checkpoint_sha256 is not None:
        raise ValueError("An injected B1 model cannot also receive a checkpoint")
    b1_model.eval()

    if table_7r is None:
        table_7r = FullPredictedDdt(ddt_7r_path, 7)
    load_seconds = time.perf_counter() - t_load0

    actual_full_pairs = len(pairs) if full_pairs is None else full_pairs

    res = generate_b1_hybrid_8parent_queue(
        pairs=pairs,
        b1_model=b1_model,
        table_7r=table_7r,
        screen_pairs=screen_pairs,
        full_pairs=actual_full_pairs,
        parent_count=parent_count,
        children_per_parent=children_per_parent,
        return_diagnostics=return_diagnostics,
    )
    if return_diagnostics:
        queue, diag = res
        diag["startup_and_load_seconds"] = load_seconds
        return queue, diag
    return res


def _run_method_worker_direct(
    method: str,
    rounds: int,
    attack_pairs: Sequence[tuple[Block, Block]],
    known_pairs: Sequence[KnownPair],
    verification_pairs: Sequence[KnownPair],
    *,
    global_deadline_seconds: float = 300.0,
    candidate_cutoff_seconds: float = 45.0,
    candidate_cutoff_soft_seconds: float | None = None,
    ranking_time_limit: float = 120.0,
    b1_checkpoint_path: Path | str | None = None,
    b1_checkpoint_sha256: str | None = None,
    random_seed: int = 260926,
    oracle_trailing_keys: tuple[int, int] | None = None,
    mock_candidate_queue: list[tuple[int, int]] | None = None,
    solver_stub: Callable[[SimonSatModel, Sequence[int], float], SatSolveResult] | None = None,
    progress_file: str | None = None,
    test_interruption_point: str | None = None,
) -> MethodWorkerOutcome:
    """Execute isolated worker logic directly within the worker process."""
    if method not in ALL_PERMITTED_METHODS:
        raise ValueError(f"Unknown or unauthorized method: {method}")
    if len(verification_pairs) < 2:
        raise ValueError("At least 2 fresh verification pairs required")

    cutoff_seconds = (
        candidate_cutoff_soft_seconds
        if candidate_cutoff_soft_seconds is not None
        else candidate_cutoff_seconds
    )

    # Invariant 1: Compared methods CANNOT receive oracle keys, mock queues, stubs, or test interruptions
    if method in COMPARATOR_METHODS:
        if oracle_trailing_keys is not None:
            raise ValueError(f"Compared method {method} cannot receive oracle trailing keys")
        if mock_candidate_queue is not None:
            raise ValueError(f"Compared method {method} cannot receive mock candidate queue")
        if solver_stub is not None:
            raise ValueError(f"Production comparator {method} cannot use solver_stub")
        if test_interruption_point is not None:
            raise ValueError(f"Production comparator {method} cannot use test_interruption_point")

    if mock_candidate_queue is not None and len(mock_candidate_queue) == 0:
        raise ValueError("mock_candidate_queue cannot be empty; non-empty queue required")

    # Invariant: Verification observations are NEVER passed to ranker functions
    verif_cts = {v[1] for v in verification_pairs}
    verif_pts = {v[0] for v in verification_pairs}
    for p0, p1 in attack_pairs:
        if p0 in verif_cts or p1 in verif_cts or p0 in verif_pts or p1 in verif_pts:
            raise ValueError("Verification observation leaked into ranking attack pairs")

    _log_progress_event(progress_file, {"event": "init", "method": method, "rounds": rounds})

    t_start = time.perf_counter()
    ranking_time = 0.0
    solving_time = 0.0
    verif_time = 0.0
    attempts: list[CandidateAttemptLog] = []
    recovered_key = None
    verified = False
    outcome = "unknown"
    error_msg = None
    has_verification_failure = False
    queue: list[tuple[int, int]] = []

    try:
        # 1. SAT-only methods (no candidate queue)
        if method in ("sat_only_n4", "sat_only_n64", "sat_only_n128", "test_sat_only"):
            if method == "sat_only_n4":
                obs_count = 4
            elif method in ("sat_only_n64", "sat_only_n128"):
                obs_count = 128 if (method == "sat_only_n128" or len(known_pairs) >= 128) else 64
            else:
                obs_count = min(64, len(known_pairs))
            _log_progress_event(progress_file, {
                "event": "phase",
                "phase": "solving",
                "solving_kind": "unguided",
                "started_at_monotonic": time.perf_counter(),
                "full_pairs_used": obs_count // 2,
            })
            if test_interruption_point == "solving":
                time.sleep(10.0)
            obs_pairs = tuple(known_pairs[:obs_count])
            t0 = time.perf_counter()
            model = SimonSatModel.from_known_pairs(known_pairs=obs_pairs, rounds=rounds)
            remaining = max(0.001, global_deadline_seconds - (time.perf_counter() - t_start))
            if solver_stub is not None:
                solve_res = solver_stub(model, (), remaining)
            else:
                solve_res = model.solve(time_limit=remaining)
            solving_time = time.perf_counter() - t0

            if solve_res.status == "sat":
                sol_key = model.master_key_from_solution(solve_res.solution)
                recovered_key = sol_key
                _log_progress_event(progress_file, {
                    "event": "solve_complete",
                    "solving_wall_seconds": solving_time,
                })
                _log_progress_event(progress_file, {
                    "event": "phase",
                    "phase": "verification",
                    "started_at_monotonic": time.perf_counter(),
                })
                if test_interruption_point == "verification":
                    time.sleep(10.0)
                t_v0 = time.perf_counter()
                v_res = verify_simon_master_key(
                    master_key=sol_key,
                    known_pairs=known_pairs,
                    verification_pairs=verification_pairs[:2],
                    rounds=rounds,
                )
                verif_time = time.perf_counter() - t_v0
                verified = v_res.verified
                outcome = "verified_recovery" if verified else "verification_failure"
            elif solve_res.status == "solver_timeout":
                elapsed = time.perf_counter() - t_start
                outcome = "global_deadline" if elapsed >= global_deadline_seconds - 0.05 else "solver_timeout"
            elif solve_res.status == "unsat":
                outcome = "proven_unsat"
            else:
                outcome = f"solver_{solve_res.status}"

            _log_progress_event(progress_file, {
                "event": "completed",
                "outcome": outcome,
                "solving_wall_seconds": solving_time,
                "verification_wall_seconds": verif_time,
            })
            return MethodWorkerOutcome(
                method=method,
                outcome=outcome,
                complete_online_wall_seconds=time.perf_counter() - t_start,
                ranking_wall_seconds=0.0,
                solving_wall_seconds=solving_time,
                verification_wall_seconds=verif_time,
                candidates_tested_count=0,
                candidates_remaining_count=0,
                recovered_master_key=recovered_key,
                verified_on_fresh_pairs=verified,
                candidate_attempts=(),
                peak_rss_gib=get_process_peak_rss_gib(),
                queue_sample=(),
                full_queue=(),
                in_flight_attempt=None,
                child_pids=(),
                full_pairs_used=obs_count // 2,
            )

        # 2. Candidate Queue Generation
        t_rank0 = time.perf_counter()
        _log_progress_event(progress_file, {"event": "phase", "phase": "ranking"})
        if test_interruption_point == "ranking" or method == "test_slow_ranker":
            time.sleep(10.0)

        if method in DIAGNOSTIC_ORACLES:
            if oracle_trailing_keys is None:
                raise ValueError(f"oracle_trailing_keys required for {method}")
            queue = [oracle_trailing_keys]
        elif method == "test_mock_queue":
            if mock_candidate_queue is None:
                raise ValueError("mock_candidate_queue required for test_mock_queue")
            queue = list(mock_candidate_queue)
        elif method == "test_hanging_ranking_child":
            child_proc = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
            )
            if progress_file:
                _log_progress_event(progress_file, {"event": "child_pid", "pid": child_proc.pid})
            child_proc.wait()
            queue = [(0x1234, 0x5678)]
        elif method == "test_hanging_tf_child":
            tf_py = Path("/private/tmp/simon_tf_venv/bin/python3.12")
            py_bin = str(tf_py) if tf_py.is_file() else sys.executable
            child_proc = subprocess.Popen(
                [py_bin, "-c", "import time; time.sleep(120)"],
            )
            if progress_file:
                _log_progress_event(progress_file, {"event": "child_pid", "pid": child_proc.pid})
            child_proc.wait()
            queue = [(0x1234, 0x5678)]
        elif method == "test_slow_ranker":
            queue = [(0x1234, 0x5678)]
        elif method == "test_slow_solver":
            queue = [(0x1234, 0x5678), (0x2345, 0x6789), (0x3456, 0x789A)]
        elif method in (
            "test_slow_rebuild",
            "test_slow_verification",
            "test_verification_failure",
            "test_unknown_solver_status",
        ):
            queue = [(0x1234, 0x5678), (0x2345, 0x6789)]
        elif method == "test_multiple_verification_failures":
            queue = list(mock_candidate_queue) if mock_candidate_queue else [(0x1111, 0x2222), (0x3333, 0x4444), (0x5555, 0x6666)]
        elif method == "random_control_128":
            queue = generate_random_tuples(128, seed=random_seed)
        elif method == "random_control_1024":
            queue = generate_random_tuples(1024, seed=random_seed)
        elif method == "fully_classical_8parent":
            queue = generate_classical_8parent_queue(attack_pairs)
        elif method == "neural_classical_hybrid_8parent":
            queue = generate_hybrid_8parent_queue(
                attack_pairs, ranking_time_limit=ranking_time_limit, progress_file=progress_file
            )
        elif method == "fully_neural_1parent":
            queue = generate_neural_1parent_queue(
                attack_pairs, ranking_time_limit=ranking_time_limit, progress_file=progress_file
            )
        elif method in ("faithful_b1_hybrid", "faithful_b1_8parent"):
            queue = generate_faithful_b1_hybrid_queue(
                attack_pairs,
                checkpoint_path=b1_checkpoint_path,
                checkpoint_sha256=b1_checkpoint_sha256,
            )
        else:
            raise ValueError(f"Unhandled method queue: {method}")

        if len(queue) == 0:
            raise ValueError(f"Method {method} produced an empty candidate queue; non-empty queue required")

        ranking_time = time.perf_counter() - t_rank0
        _log_progress_event(progress_file, {
            "event": "queue",
            "ranking_wall_seconds": ranking_time,
            "total_queue_count": len(queue),
            "full_queue": [list(t) for t in queue],
            "queue_sample": [list(t) for t in queue[:5]],
            "peak_rss_gib": get_process_peak_rss_gib(),
            "full_pairs_used": len(attack_pairs),
        })

        if ranking_time > ranking_time_limit:
            _log_progress_event(progress_file, {"event": "completed", "outcome": "ranking_timeout"})
            return MethodWorkerOutcome(
                method=method,
                outcome="ranking_timeout",
                complete_online_wall_seconds=time.perf_counter() - t_start,
                ranking_wall_seconds=ranking_time,
                solving_wall_seconds=0.0,
                verification_wall_seconds=0.0,
                candidates_tested_count=0,
                candidates_remaining_count=len(queue),
                recovered_master_key=None,
                verified_on_fresh_pairs=False,
                candidate_attempts=(),
                peak_rss_gib=get_process_peak_rss_gib(),
                queue_sample=tuple(queue[:5]),
                full_queue=tuple(queue),
                in_flight_attempt=None,
                child_pids=(),
                full_pairs_used=len(attack_pairs),
            )

        # 3. Candidate Solving Loop on 4-observation formula
        known_4obs = tuple(known_pairs[:4])
        _log_progress_event(progress_file, {
            "event": "phase",
            "phase": "initial_formula_construction",
            "started_at_monotonic": time.perf_counter(),
        })
        if test_interruption_point == "initial_formula_construction":
            time.sleep(10.0)
        t_model0 = time.perf_counter()
        model = SimonSatModel.from_known_pairs(known_pairs=known_4obs, rounds=rounds)
        initial_construction_time = time.perf_counter() - t_model0
        _log_progress_event(progress_file, {
            "event": "initial_formula_construction_complete",
            "initial_construction_wall_seconds": initial_construction_time,
        })

        t_solve_start = time.perf_counter()
        candidates_tested = 0
        total_in_queue = len(queue)

        for idx, (cand_k8, cand_k9) in enumerate(queue):
            elapsed_total = time.perf_counter() - t_start
            remaining_global = global_deadline_seconds - elapsed_total
            if remaining_global <= 0.01:
                outcome = "global_deadline"
                break

            cand_timeout = min(cutoff_seconds, remaining_global)
            assumptions = model.assumptions_for_trailing_round_subkeys([cand_k8, cand_k9])

            _log_progress_event(progress_file, {
                "event": "attempt_start",
                "candidate_index": idx,
                "candidate_k8": cand_k8,
                "candidate_k9": cand_k9,
                "elapsed_so_far": time.perf_counter() - t_start,
                "started_at_monotonic": time.perf_counter(),
                "peak_rss_gib": get_process_peak_rss_gib(),
            })

            _log_progress_event(progress_file, {
                "event": "phase",
                "phase": "solving",
                "candidate_index": idx,
                "candidate_k8": cand_k8,
                "candidate_k9": cand_k9,
                "started_at_monotonic": time.perf_counter(),
            })

            if test_interruption_point == "solving" and idx == 1:
                time.sleep(10.0)

            t0 = time.perf_counter()
            if method == "test_slow_solver":
                if idx == 0:
                    solve_res = SatSolveResult(False, None, 0.001, False, "unsat")
                else:
                    time.sleep(10.0)
                    solve_res = SatSolveResult(False, None, 10.0, True, "solver_timeout")
            elif method == "test_slow_rebuild":
                solve_res = SatSolveResult(False, None, 0.001, True, "solver_timeout")
            elif method in ("test_slow_verification", "test_multiple_verification_failures"):
                sol = [False] * (model.variable_count + 1)
                solve_res = SatSolveResult(True, sol, 0.001, False, "sat")
            elif solver_stub is not None:
                solve_res = solver_stub(model, assumptions, cand_timeout)
            else:
                solve_res = model.solve(assumptions, time_limit=cand_timeout)
            cand_solve_time = time.perf_counter() - t0
            candidates_tested += 1

            rebuilt = False
            cand_status = solve_res.status

            if solve_res.status == "sat":
                sol_key = model.master_key_from_solution(solve_res.solution)
                _log_progress_event(progress_file, {
                    "event": "candidate_solve_complete",
                    "candidate_index": idx,
                    "candidate_k8": cand_k8,
                    "candidate_k9": cand_k9,
                    "solve_wall_seconds": cand_solve_time,
                    "status": "sat",
                })
                _log_progress_event(progress_file, {
                    "event": "phase",
                    "phase": "verification",
                    "candidate_index": idx,
                    "candidate_k8": cand_k8,
                    "candidate_k9": cand_k9,
                    "started_at_monotonic": time.perf_counter(),
                })
                if test_interruption_point == "verification" or method == "test_slow_verification":
                    time.sleep(10.0)

                t_v0 = time.perf_counter()
                if method == "test_multiple_verification_failures":
                    is_verified = (idx >= 2)
                    v_res = SimonVerificationResult(is_verified, is_verified)
                else:
                    v_res = verify_simon_master_key(
                        master_key=sol_key,
                        known_pairs=known_pairs,
                        verification_pairs=verification_pairs[:2],
                        rounds=rounds,
                    )
                verif_time += time.perf_counter() - t_v0
                current_solve_wall = max(0.0, (time.perf_counter() - t_solve_start) - verif_time)

                if v_res.verified:
                    recovered_key = sol_key
                    verified = True
                    outcome = "verified_recovery"
                    cand_status = "sat"
                    att = CandidateAttemptLog(
                        candidate_k8=cand_k8,
                        candidate_k9=cand_k9,
                        solve_wall_seconds=cand_solve_time,
                        status=cand_status,
                        rebuild_occurred=False,
                    )
                    attempts.append(att)
                    _log_progress_event(progress_file, {
                        "event": "attempt",
                        "candidate_k8": cand_k8,
                        "candidate_k9": cand_k9,
                        "solve_wall_seconds": cand_solve_time,
                        "status": cand_status,
                        "rebuild_occurred": False,
                        "candidates_tested_count": len(attempts),
                        "current_solving_wall_seconds": current_solve_wall,
                        "current_verification_wall_seconds": verif_time,
                        "peak_rss_gib": get_process_peak_rss_gib(),
                    })
                    break
                else:
                    cand_status = "verification_failure"
                    has_verification_failure = True
                    rebuilt = True
                    att = CandidateAttemptLog(
                        candidate_k8=cand_k8,
                        candidate_k9=cand_k9,
                        solve_wall_seconds=cand_solve_time,
                        status=cand_status,
                        rebuild_occurred=False,
                    )
                    attempts.append(att)
                    _log_progress_event(progress_file, {
                        "event": "attempt",
                        "candidate_k8": cand_k8,
                        "candidate_k9": cand_k9,
                        "solve_wall_seconds": cand_solve_time,
                        "status": cand_status,
                        "rebuild_occurred": False,
                        "candidates_tested_count": len(attempts),
                        "current_solving_wall_seconds": current_solve_wall,
                        "current_verification_wall_seconds": verif_time,
                        "peak_rss_gib": get_process_peak_rss_gib(),
                    })
                    _log_progress_event(progress_file, {
                        "event": "phase",
                        "phase": "rebuilding",
                        "candidate_index": idx,
                        "candidate_k8": cand_k8,
                        "candidate_k9": cand_k9,
                        "started_at_monotonic": time.perf_counter(),
                    })
                    if test_interruption_point == "rebuilding" or method == "test_slow_rebuild":
                        time.sleep(10.0)
                    t_reb0 = time.perf_counter()
                    model = SimonSatModel.from_known_pairs(known_pairs=known_4obs, rounds=rounds)
                    reb_time = time.perf_counter() - t_reb0
                    attempts[-1] = CandidateAttemptLog(
                        cand_k8, cand_k9, cand_solve_time, cand_status, True
                    )
                    _log_progress_event(progress_file, {
                        "event": "rebuild_complete",
                        "candidate_k8": cand_k8,
                        "candidate_k9": cand_k9,
                        "rebuild_wall_seconds": reb_time,
                    })
            elif solve_res.status == "unsat":
                cand_status = "unsat"
                rebuilt = False
                att = CandidateAttemptLog(
                    candidate_k8=cand_k8,
                    candidate_k9=cand_k9,
                    solve_wall_seconds=cand_solve_time,
                    status=cand_status,
                    rebuild_occurred=False,
                )
                attempts.append(att)
                current_solve_wall = max(0.0, (time.perf_counter() - t_solve_start) - verif_time)
                _log_progress_event(progress_file, {
                    "event": "attempt",
                    "candidate_k8": cand_k8,
                    "candidate_k9": cand_k9,
                    "solve_wall_seconds": cand_solve_time,
                    "status": cand_status,
                    "rebuild_occurred": False,
                    "candidates_tested_count": len(attempts),
                    "current_solving_wall_seconds": current_solve_wall,
                    "current_verification_wall_seconds": verif_time,
                    "peak_rss_gib": get_process_peak_rss_gib(),
                })
            else:
                # timeout, unsat_or_timeout, or unknown
                cand_status = solve_res.status if solve_res.status in ("solver_timeout", "unsat_or_timeout") else f"unknown_{solve_res.status}"
                rebuilt = True
                att = CandidateAttemptLog(
                    candidate_k8=cand_k8,
                    candidate_k9=cand_k9,
                    solve_wall_seconds=cand_solve_time,
                    status=cand_status,
                    rebuild_occurred=False,
                )
                attempts.append(att)
                current_solve_wall = max(0.0, (time.perf_counter() - t_solve_start) - verif_time)
                _log_progress_event(progress_file, {
                    "event": "attempt",
                    "candidate_k8": cand_k8,
                    "candidate_k9": cand_k9,
                    "solve_wall_seconds": cand_solve_time,
                    "status": cand_status,
                    "rebuild_occurred": False,
                    "candidates_tested_count": len(attempts),
                    "current_solving_wall_seconds": current_solve_wall,
                    "current_verification_wall_seconds": verif_time,
                    "peak_rss_gib": get_process_peak_rss_gib(),
                })
                _log_progress_event(progress_file, {
                    "event": "phase",
                    "phase": "rebuilding",
                    "candidate_index": idx,
                    "candidate_k8": cand_k8,
                    "candidate_k9": cand_k9,
                    "started_at_monotonic": time.perf_counter(),
                })
                if test_interruption_point == "rebuilding" or method == "test_slow_rebuild":
                    time.sleep(10.0)
                t_reb0 = time.perf_counter()
                model = SimonSatModel.from_known_pairs(known_pairs=known_4obs, rounds=rounds)
                reb_time = time.perf_counter() - t_reb0
                attempts[-1] = CandidateAttemptLog(
                    cand_k8, cand_k9, cand_solve_time, cand_status, True
                )
                _log_progress_event(progress_file, {
                    "event": "rebuild_complete",
                    "candidate_k8": cand_k8,
                    "candidate_k9": cand_k9,
                    "rebuild_wall_seconds": reb_time,
                })

        solving_time = max(0.0, (time.perf_counter() - t_solve_start) - verif_time)

        if outcome == "unknown":
            if verified:
                outcome = "verified_recovery"
            elif any(
                attempt.status in ("solver_timeout", "unsat_or_timeout")
                or attempt.status.startswith("unknown_")
                for attempt in attempts
            ):
                outcome = "candidate_unresolved"
            elif has_verification_failure and candidates_tested == total_in_queue:
                outcome = "verification_failure"
            elif candidates_tested == total_in_queue:
                outcome = "queue_exhausted"
            else:
                outcome = "global_deadline"

        _log_progress_event(progress_file, {
            "event": "completed",
            "outcome": outcome,
            "solving_wall_seconds": solving_time,
            "verification_wall_seconds": verif_time,
        })

    except TimeoutError as exc:
        outcome = "ranking_timeout"
        ranking_time = time.perf_counter() - t_rank0 if "t_rank0" in locals() else 0.0
        error_msg = f"{type(exc).__name__}: {exc}"
        _log_progress_event(progress_file, {"event": "completed", "outcome": outcome})
    except Exception as exc:
        outcome = "exception"
        error_msg = f"{type(exc).__name__}: {exc}"
        _log_progress_event(progress_file, {"event": "exception", "error": error_msg})

    return MethodWorkerOutcome(
        method=method,
        outcome=outcome,
        complete_online_wall_seconds=time.perf_counter() - t_start,
        ranking_wall_seconds=ranking_time,
        solving_wall_seconds=solving_time,
        verification_wall_seconds=verif_time,
        candidates_tested_count=len(attempts),
        candidates_remaining_count=len(queue) - len(attempts) if 'queue' in locals() else 0,
        recovered_master_key=recovered_key,
        verified_on_fresh_pairs=verified,
        candidate_attempts=tuple(attempts),
        peak_rss_gib=get_process_peak_rss_gib(),
        error_message=error_msg,
        queue_sample=tuple(queue[:5]) if 'queue' in locals() else (),
        full_queue=tuple(queue) if 'queue' in locals() else (),
        in_flight_attempt=None,
        child_pids=(),
        full_pairs_used=len(attack_pairs),
    )


def run_method_worker(
    method: str,
    rounds: int | None = None,
    attack_pairs: Sequence[tuple[Block, Block]] | None = None,
    known_pairs: Sequence[KnownPair] | None = None,
    verification_pairs: Sequence[KnownPair] | None = None,
    *,
    comparison_input: SimonComparisonInput | None = None,
    manifest_seed: int = 260926,
    instance_id: str = "simon32_64_instance_0000",
    global_deadline_seconds: float = 300.0,
    candidate_cutoff_seconds: float = 45.0,
    candidate_cutoff_soft_seconds: float | None = None,
    ranking_time_limit: float = 120.0,
    b1_checkpoint_path: Path | str | None = None,
    b1_checkpoint_sha256: str | None = None,
    random_seed: int | None = None,
    oracle_trailing_keys: tuple[int, int] | None = None,
    mock_candidate_queue: list[tuple[int, int]] | None = None,
    solver_stub: Callable[[SimonSatModel, Sequence[int], float], SatSolveResult] | None = None,
    use_parent_watchdog: bool = True,
    test_interruption_point: str | None = None,
) -> MethodWorkerOutcome:
    """Execute one method under a parent process watchdog or directly in-process.

    Cutoff semantics:
    - candidate_cutoff_soft_seconds = 45.0 (default): soft request passed to pycryptosat
      cooperative solver, checked periodically at conflict checkpoints.
    - global_deadline_seconds = 300.0 (default): hard parent process watchdog limit.
      On expiration, the parent watchdog forcibly terminates the worker process group
      and child PID tree via SIGKILL.
    """
    # Invariant: Start reported online clock in parent BEFORE payload preparation
    t_start = time.perf_counter()

    if comparison_input is not None:
        if rounds is not None and rounds != comparison_input.rounds:
            raise ValueError("rounds conflict with comparison_input")
        if attack_pairs is not None and tuple(attack_pairs) != comparison_input.ciphertext_pairs:
            raise ValueError("attack_pairs conflict with comparison_input")
        if known_pairs is not None and tuple(known_pairs) != comparison_input.known_pairs:
            raise ValueError("known_pairs conflict with comparison_input")
        if (
            verification_pairs is not None
            and tuple(verification_pairs) != tuple(comparison_input.verification_pairs)
        ):
            raise ValueError("verification_pairs conflict with comparison_input")
        rounds = comparison_input.rounds
        attack_pairs = comparison_input.ciphertext_pairs
        known_pairs = comparison_input.known_pairs
        verification_pairs = comparison_input.verification_pairs
        instance_id = comparison_input.instance_id
        manifest_seed = comparison_input.manifest_seed

    if method not in ALL_PERMITTED_METHODS:
        raise ValueError(f"Unknown or unauthorized method: {method}")

    effective_cutoff = (
        candidate_cutoff_soft_seconds
        if candidate_cutoff_soft_seconds is not None
        else candidate_cutoff_seconds
    )

    # Production comparator invariants and strict validation
    if method in COMPARATOR_METHODS:
        if oracle_trailing_keys is not None:
            raise ValueError(f"Compared method {method} cannot receive oracle trailing keys")
        if mock_candidate_queue is not None:
            raise ValueError(f"Compared method {method} cannot receive mock candidate queue")
        if solver_stub is not None:
            raise ValueError(f"Production comparator {method} cannot use solver_stub")
        if not use_parent_watchdog:
            raise ValueError(
                f"Production comparator {method} cannot bypass parent watchdog (use_parent_watchdog must be True)"
            )
        if test_interruption_point is not None:
            raise ValueError(f"Production comparator {method} cannot use test_interruption_point")
        if method.startswith("random_control_") and random_seed is not None:
            raise ValueError("Production random controls require the manifest-derived seed")

        if rounds != 10:
            raise ValueError(f"Production comparator {method} requires R=10 rounds, got {rounds}")
        if attack_pairs is None or len(attack_pairs) not in (32, 64):
            raise ValueError(
                f"Production comparator {method} requires 32 or 64 attack pairs, "
                f"got {len(attack_pairs) if attack_pairs else None}"
            )
        if known_pairs is None or len(known_pairs) != 2 * len(attack_pairs):
            raise ValueError(
                f"Production comparator {method} requires exactly {2 * len(attack_pairs)} known observations"
            )
        if verification_pairs is None or len(verification_pairs) != 2:
            raise ValueError(f"Production comparator {method} requires exactly 2 verification observations")
        attack_plaintexts: list[Block] = []
        for index, (ct0, ct1) in enumerate(attack_pairs):
            (pt0, sat_ct0), (pt1, sat_ct1) = known_pairs[2 * index : 2 * index + 2]
            if tuple(ct0) != tuple(sat_ct0) or tuple(ct1) != tuple(sat_ct1):
                raise ValueError("Ranking ciphertext pairs must match SAT observations in order")
            if (pt0[0] ^ pt1[0], pt0[1] ^ pt1[1]) != DEFAULT_SIMON_INPUT_DIFFERENCE:
                raise ValueError("Attack plaintext pair has wrong input difference")
            attack_plaintexts.extend((tuple(pt0), tuple(pt1)))
        verification_plaintexts = [tuple(pair[0]) for pair in verification_pairs]
        if (
            len(set(attack_plaintexts)) != len(attack_plaintexts)
            or len(set(verification_plaintexts)) != len(verification_plaintexts)
            or set(attack_plaintexts).intersection(verification_plaintexts)
        ):
            raise ValueError("Attack and verification plaintexts must be disjoint and unique")
        if method in ("faithful_b1_hybrid", "faithful_b1_8parent"):
            if b1_checkpoint_path is None or b1_checkpoint_sha256 is None:
                raise ValueError("Production B1 method requires a checkpoint path and SHA-256")
        elif b1_checkpoint_path is not None or b1_checkpoint_sha256 is not None:
            raise ValueError(f"Comparator {method} cannot receive a B1 checkpoint")
    elif method in DIAGNOSTIC_ORACLES:
        if oracle_trailing_keys is None:
            raise ValueError(f"Diagnostic oracle {method} requires oracle_trailing_keys")

    if rounds is None:
        raise ValueError("rounds is required")
    if attack_pairs is None:
        raise ValueError("attack_pairs is required")
    if known_pairs is None:
        raise ValueError("known_pairs is required")
    if verification_pairs is None:
        raise ValueError("verification_pairs is required")

    # Invariant: Verification observations are NEVER passed to ranker functions
    verif_cts = {v[1] for v in verification_pairs}
    verif_pts = {v[0] for v in verification_pairs}
    for p0, p1 in attack_pairs:
        if p0 in verif_cts or p1 in verif_cts or p0 in verif_pts or p1 in verif_pts:
            raise ValueError("Verification observation leaked into ranking attack pairs")

    # Deterministic random control stream derivation from manifest seed and instance/method IDs
    if method.startswith("random_control_"):
        if random_seed is None:
            random_seed = derive_random_control_seed(manifest_seed, instance_id, method)
    elif random_seed is None:
        random_seed = 260926

    # Test-only / diagnostic bypass
    if not use_parent_watchdog or solver_stub is not None:
        return _run_method_worker_direct(
            method=method,
            rounds=rounds,
            attack_pairs=attack_pairs,
            known_pairs=known_pairs,
            verification_pairs=verification_pairs,
            global_deadline_seconds=global_deadline_seconds,
            candidate_cutoff_seconds=effective_cutoff,
            candidate_cutoff_soft_seconds=effective_cutoff,
            ranking_time_limit=ranking_time_limit,
            b1_checkpoint_path=b1_checkpoint_path,
            b1_checkpoint_sha256=b1_checkpoint_sha256,
            random_seed=random_seed,
            oracle_trailing_keys=oracle_trailing_keys,
            mock_candidate_queue=mock_candidate_queue,
            solver_stub=solver_stub,
            test_interruption_point=test_interruption_point,
        )

    # Subprocess execution under OS process-group watchdog with external stream persistence
    progress_fd, progress_path = tempfile.mkstemp(prefix="simon_worker_progress_", suffix=".jsonl")
    os.close(progress_fd)
    preserve_progress = False

    try:
        prep_elapsed = time.perf_counter() - t_start
        child_deadline = max(0.001, global_deadline_seconds - prep_elapsed)

        payload = {
            "method": method,
            "rounds": rounds,
            "attack_pairs": [list(p) for p in attack_pairs],
            "known_pairs": [list(p) for p in known_pairs],
            "verification_pairs": [list(p) for p in verification_pairs],
            "global_deadline_seconds": child_deadline,
            "candidate_cutoff_seconds": effective_cutoff,
            "candidate_cutoff_soft_seconds": effective_cutoff,
            "ranking_time_limit": ranking_time_limit,
            "b1_checkpoint_path": str(b1_checkpoint_path) if b1_checkpoint_path is not None else None,
            "b1_checkpoint_sha256": b1_checkpoint_sha256,
            "random_seed": random_seed,
            "oracle_trailing_keys": list(oracle_trailing_keys) if oracle_trailing_keys else None,
            "mock_candidate_queue": [list(t) for t in mock_candidate_queue] if mock_candidate_queue else None,
            "progress_file": progress_path,
            "test_interruption_point": test_interruption_point,
        }

        cmd = [sys.executable, "-m", "neural_sat_speck.simon.method_worker", "--worker"]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,  # Separate process group for clean group kill
        )

        try:
            remaining_timeout = max(0.001, global_deadline_seconds - (time.perf_counter() - t_start))
            stdout, stderr = proc.communicate(input=json.dumps(payload), timeout=remaining_timeout)
            stopped_at = time.perf_counter()
            elapsed = stopped_at - t_start

            # Measure process tree memory honestly across concurrently alive processes
            parent_self_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            parent_child_rss = float(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
            scale = (1024.0 ** 3) if platform.system() == "Darwin" else (1024.0 ** 2)

            if proc.returncode == 0:
                result_dict = json.loads(stdout)
                child_peak = float(result_dict.get("peak_rss_gib", 0.0))
                tree_peak_rss = (parent_self_rss + max(parent_child_rss, child_peak * scale)) / scale
                result_dict["complete_online_wall_seconds"] = elapsed
                result_dict["peak_rss_gib"] = tree_peak_rss
                return MethodWorkerOutcome.from_dict(result_dict)
            else:
                rec = _read_progress_file(
                    progress_path, parent_elapsed=elapsed, stopped_at_monotonic=stopped_at,
                )
                preserve_progress = rec.journal_error is not None
                child_peak = max(parent_child_rss, rec.peak_rss_gib * scale)
                tree_peak_rss = (parent_self_rss + child_peak) / scale
                return MethodWorkerOutcome(
                    method=method,
                    outcome="exception",
                    complete_online_wall_seconds=elapsed,
                    ranking_wall_seconds=rec.ranking_wall_seconds,
                    solving_wall_seconds=rec.solving_wall_seconds,
                    verification_wall_seconds=rec.verification_wall_seconds,
                    candidates_tested_count=rec.candidates_tested_count,
                    candidates_remaining_count=rec.candidates_remaining_count,
                    recovered_master_key=None,
                    verified_on_fresh_pairs=False,
                    candidate_attempts=tuple(rec.attempts),
                    peak_rss_gib=tree_peak_rss,
                    error_message=(
                        f"Progress journal integrity failure at {progress_path}: {rec.journal_error}; "
                        if rec.journal_error else ""
                    ) + (stderr[-1000:] if stderr else f"Worker exited with code {proc.returncode}"),
                    queue_sample=rec.queue_sample,
                    full_queue=rec.full_queue,
                    in_flight_attempt=rec.in_flight_attempt,
                    child_pids=tuple(rec.child_pids),
                    full_pairs_used=rec.full_pairs_used,
                )
        except subprocess.TimeoutExpired:
            # Kill first. A corrupt journal must never prevent process-tree cleanup.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
            proc.communicate()
            stopped_at = time.perf_counter()
            elapsed = stopped_at - t_start
            rec = _read_progress_file(
                progress_path, parent_elapsed=elapsed, stopped_at_monotonic=stopped_at,
            )
            preserve_progress = rec.journal_error is not None

            parent_self_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            parent_child_rss = float(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
            scale = (1024.0 ** 3) if platform.system() == "Darwin" else (1024.0 ** 2)
            child_peak = max(parent_child_rss, rec.peak_rss_gib * scale)
            tree_peak_rss = (parent_self_rss + child_peak) / scale

            return MethodWorkerOutcome(
                method=method,
                outcome="exception" if rec.journal_error else "global_deadline",
                complete_online_wall_seconds=elapsed,
                ranking_wall_seconds=rec.ranking_wall_seconds,
                solving_wall_seconds=rec.solving_wall_seconds,
                verification_wall_seconds=rec.verification_wall_seconds,
                candidates_tested_count=rec.candidates_tested_count,
                candidates_remaining_count=rec.candidates_remaining_count,
                recovered_master_key=None,
                verified_on_fresh_pairs=False,
                candidate_attempts=tuple(rec.attempts),
                peak_rss_gib=tree_peak_rss,
                error_message=(
                    f"Progress journal integrity failure at {progress_path}: {rec.journal_error}; "
                    if rec.journal_error else ""
                ) + (
                    f"Terminated by parent process watchdog after {elapsed:.3f}s "
                    f"(deadline {global_deadline_seconds}s; interrupted during {rec.last_phase})"
                ),
                queue_sample=rec.queue_sample,
                full_queue=rec.full_queue,
                in_flight_attempt=rec.in_flight_attempt,
                child_pids=tuple(rec.child_pids),
                full_pairs_used=rec.full_pairs_used,
            )
    finally:
        if not preserve_progress and progress_path and os.path.exists(progress_path):
            try:
                os.remove(progress_path)
            except OSError:
                pass


def _worker_main() -> None:
    """CLI entrypoint for isolated worker process."""
    raw = sys.stdin.read()
    payload = json.loads(raw)

    attack_pairs = tuple(
        (tuple(p[0]), tuple(p[1])) for p in payload["attack_pairs"]
    )
    known_pairs = tuple(
        (tuple(p[0]), tuple(p[1])) for p in payload["known_pairs"]
    )
    verification_pairs = tuple(
        (tuple(p[0]), tuple(p[1])) for p in payload["verification_pairs"]
    )
    oracle_keys = (
        tuple(payload["oracle_trailing_keys"])
        if payload.get("oracle_trailing_keys")
        else None
    )
    mock_queue = (
        [tuple(t) for t in payload["mock_candidate_queue"]]
        if payload.get("mock_candidate_queue")
        else None
    )

    outcome = _run_method_worker_direct(
        method=payload["method"],
        rounds=payload["rounds"],
        attack_pairs=attack_pairs,
        known_pairs=known_pairs,
        verification_pairs=verification_pairs,
        global_deadline_seconds=payload["global_deadline_seconds"],
        candidate_cutoff_seconds=payload["candidate_cutoff_seconds"],
        candidate_cutoff_soft_seconds=payload.get("candidate_cutoff_soft_seconds"),
        ranking_time_limit=payload["ranking_time_limit"],
        b1_checkpoint_path=payload.get("b1_checkpoint_path"),
        b1_checkpoint_sha256=payload.get("b1_checkpoint_sha256"),
        random_seed=payload["random_seed"],
        oracle_trailing_keys=oracle_keys,
        mock_candidate_queue=mock_queue,
        progress_file=payload.get("progress_file"),
        test_interruption_point=payload.get("test_interruption_point"),
    )

    sys.stdout.write(json.dumps(outcome.to_dict()))
    sys.stdout.flush()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _worker_main()
