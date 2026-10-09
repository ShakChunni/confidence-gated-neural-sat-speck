"""Per-candidate cap, classical Stage 1, and capped-attempt audit rules (D-060/D-061)."""

# ruff: noqa: F811  (pytest fixtures are imported by name from sibling test modules)

import os
import random
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
from test_ranking_depth_audit import evidence  # noqa: F401  (fixture re-export)
from test_ranking_depth_execution import toy, worker_case  # noqa: F401

from neural_sat_speck.attacks.classical_ranking import ClassicalDistributionTable
from neural_sat_speck.backend.formula import build_formula
from neural_sat_speck.backend.guard import atomic_json, read_checked_json
from neural_sat_speck.backend.solver_io import solver_arguments
from neural_sat_speck.backend.worker import Journal
from neural_sat_speck.followups import ranking_depth_audit as audit
from neural_sat_speck.followups import ranking_depth_ranking as ranking
from neural_sat_speck.followups import ranking_depth_worker as worker
from neural_sat_speck.followups.ranking_depth_worker import conditioned_formula

SLEEP = [sys.executable, "-c", "import time; time.sleep(60)"]


# --- Real-process cap -------------------------------------------------------------


def test_cap_kills_only_the_candidate_pid_never_the_group(monkeypatch):
    sibling = subprocess.Popen(SLEEP)  # same process group as this test process
    solver = subprocess.Popen(SLEEP)
    monkeypatch.setattr(
        worker.os, "killpg", lambda *_: pytest.fail("killpg must never be used")
    )
    begun = time.perf_counter()
    try:
        code, capped = worker.wait_with_cap(solver, 0.3)
        assert capped is True and code == -9
        assert time.perf_counter() - begun >= 0.3
        assert solver.poll() == -9
        assert sibling.poll() is None  # the shared group member is untouched
    finally:
        sibling.kill()
        sibling.wait()


class _RecordingProcess:
    def __init__(self):
        self.calls = []

    def wait(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return 0

    def kill(self):
        pytest.fail("the uncapped path never kills")


def test_uncapped_wait_is_the_unchanged_plain_wait():
    proc = _RecordingProcess()
    assert worker.wait_with_cap(proc, None) == (0, False)
    assert proc.calls == [((), {})]


def test_solve_candidate_caps_a_running_process_and_records_no_receipt(
    tmp_path, monkeypatch
):
    real_popen = subprocess.Popen

    def stand_in(_argv, **kwargs):
        # Stand-in for the launcher: a process that never finishes on its own.
        return real_popen(SLEEP, **kwargs)

    monkeypatch.setattr(worker.subprocess, "Popen", stand_in)
    formula = {
        "native_dimacs": "p cnf 1 1\n1 0\n",
        "assumption_literals": [1],
        "counts": {},
        "num_vars": 1,
    }
    payload = {
        "cell": {"profile": "cms_native_gauss", "candidate_cap_seconds": 0.2},
        "paths": {"cms": "/nonexistent/cms"},
        "repo_root": str(tmp_path),
        "source_commit": "a" * 40,
    }
    directory = tmp_path / "0003"
    result = worker.solve_candidate(
        formula, payload, directory, Journal(tmp_path / "journal.json")
    )
    assert result["parsed"] == {"status": "capped"}
    assert result["returncode"] == -9
    assert result["candidate_cap_seconds"] == 0.2
    assert result["solve_wall_seconds"] >= 0.2
    assert result["native_launch_receipt"] is None  # killed before any receipt
    pid = read_checked_json(directory / "native_process.json")["pid"]
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)  # reaped: the solver PID is gone


@pytest.mark.parametrize("cap", [10, True, 0.0, -1.0, 300.0, float("nan"), "1.5"])
def test_cap_must_be_none_or_a_float_in_the_open_range(cap):
    with pytest.raises(ValueError):
        worker.checked_candidate_cap({"candidate_cap_seconds": cap})


def test_absent_or_none_cap_is_uncapped():
    assert worker.checked_candidate_cap({}) is None
    assert worker.checked_candidate_cap({"candidate_cap_seconds": None}) is None
    assert worker.checked_candidate_cap({"candidate_cap_seconds": 1.5}) == 1.5


# --- Worker: capped candidates, Stage 1 pass-through ---------------------------------


def test_capped_true_candidate_is_never_counted_as_recovered(worker_case, monkeypatch):
    payload, _, base, directory = worker_case
    _, _, _, keys = toy("speck32_64")
    payload["cell"]["candidate_cap_seconds"] = 1.5
    true_literals = conditioned_formula(base, [keys[-1]])["assumption_literals"]
    calls = []

    def solve(formula, payload, attempt_dir, journal):
        attempt_dir.mkdir(parents=True)
        calls.append(formula["assumption_literals"])
        if formula["assumption_literals"] == true_literals:
            return {
                "parsed": {"status": "capped"},
                "returncode": -9,
                "solver_arguments": [],
                "solve_wall_seconds": 1.5,
                "native_launch_receipt": None,
                "candidate_cap_seconds": 1.5,
            }
        return {
            "parsed": {"status": "unsat", "assignment": {}},
            "returncode": 20,
            "solver_arguments": [],
            "solve_wall_seconds": 0.1,
            "native_launch_receipt": None,
        }

    monkeypatch.setattr(worker, "solve_candidate", solve)
    result = worker.execute(payload)
    assert result["outcome"] == "queue_exhausted"
    assert result["recovered_key"] is None and result["attempt_count"] == 32
    assert len(calls) == 32  # the queue moves on past the capped candidate
    capped = read_checked_json(directory / "attempts/0001/attempt.json")
    assert capped["status"] == "capped"
    assert (
        read_checked_json(directory / "attempts/0000/attempt.json")["status"] == "unsat"
    )


def test_worker_rejects_classical_stage1_outside_speck(worker_case, monkeypatch):
    payload, _, _, _ = worker_case
    payload["cell"]["cipher"] = "simon32_64"
    payload["cell"]["stage1"] = "classical_sparse_r6"
    monkeypatch.setattr(
        worker, "solve_candidate", lambda *_: pytest.fail("Unexpected solver call")
    )
    result = worker.execute(payload)
    assert result["outcome"] == "worker_exception"
    assert "Invalid Stage 1 option" in result["error"]


def test_worker_passes_classical_stage1_only_when_requested(worker_case, monkeypatch):
    payload, _, _, _ = worker_case
    seen = []

    def spy_prospective(cipher, pairs, depth, paths, directory, callback, *, stage1):
        seen.append(stage1)
        raise RuntimeError("stop after the dispatch is observed")

    monkeypatch.setitem(
        sys.modules,
        "neural_sat_speck.followups.ranking_depth_ranking",
        SimpleNamespace(
            rank_queue=lambda *a, **k: pytest.fail("neural path must not run"),
            rank_prospective_queue=spy_prospective,
        ),
    )
    payload["cell"]["stage1"] = "classical_sparse_r6"
    result = worker.execute(payload)
    assert seen == ["classical_sparse_r6"]
    assert result["outcome"] == "worker_exception"


# --- Classical Stage 1 on a tiny fake table -------------------------------------------


def _fake_pairs(seed=7):
    rng = random.Random(seed)
    return tuple(
        (
            (rng.randrange(65536), rng.randrange(65536)),
            (rng.randrange(65536), rng.randrange(65536)),
        )
        for _ in range(64)
    )


def _tiny_table(path, rounds=6, weights=None):
    table = ClassicalDistributionTable(
        rounds=rounds,
        weights=weights if weights is not None else {0x00400000: 2.5, 0x00010002: 1.25},
        sample_count=1000,
        min_count=20,
    )
    table.save(path)
    return path


def test_classical_stage1_parents_are_deterministic_top_32(tmp_path):
    table = _tiny_table(tmp_path / "table.json")
    pairs = _fake_pairs()
    first = ranking.rank_prospective_queue(
        "speck32_64",
        pairs,
        1,
        {"speck_classical_r6": table},
        tmp_path / "run_a",
        stage1="classical_sparse_r6",
    )
    second = ranking.rank_prospective_queue(
        "speck32_64",
        pairs,
        1,
        {"speck_classical_r6": table},
        tmp_path / "run_b",
        stage1="classical_sparse_r6",
    )
    assert first["stage1_parents"] == second["stage1_parents"]
    assert len(set(first["stage1_parents"])) == 32
    assert first["queue"] == [[p] for p in first["stage1_parents"]]
    stage1 = first["ranking_receipts"]["stage1"]
    assert stage1["ranker"]["table"] == "sparseR6"
    assert stage1["ranker"]["table_rounds"] == 6
    assert set(stage1["assets"]) == {"speck_classical_r6"}
    assert stage1["stage1_parent_count"] == 32


def test_classical_stage1_ties_break_by_ascending_word(tmp_path):
    table = _tiny_table(tmp_path / "zero.json", weights={})
    result = ranking.rank_prospective_queue(
        "speck32_64",
        _fake_pairs(),
        1,
        {"speck_classical_r6": table},
        tmp_path / "zero_run",
        stage1="classical_sparse_r6",
    )
    assert result["stage1_parents"] == list(range(32))


def test_classical_stage1_refuses_a_non_six_round_table(tmp_path):
    table = _tiny_table(tmp_path / "r5.json", rounds=5)
    with pytest.raises(ValueError, match="6-round"):
        ranking.rank_prospective_queue(
            "speck32_64",
            _fake_pairs(),
            1,
            {"speck_classical_r6": table},
            tmp_path / "bad",
            stage1="classical_sparse_r6",
        )


# --- Audit: capped attempts ------------------------------------------------------------


def make_capped_run(
    evidence, *, cap=0.01, cell_cap=0.01, wall=None, returncode=-9, attempt_cap=None
):
    """Rewrite the synthetic evidence as a fully capped 32-candidate queue."""
    row, manifest, paths, directory, _attempt = evidence
    cell = {**row["cell"], "candidate_cap_seconds": cell_cap}
    manifest["cells"] = [cell]
    row["cell"] = cell
    payload = read_checked_json(directory / "payload.json")
    payload["cell"] = cell
    atomic_json(directory / "payload.json", payload)
    queue = read_checked_json(directory / "queue.json")["queue"]
    public = read_checked_json(paths["case:public"])
    base = build_formula(cell["cipher"], public, 4, 0, include_cnf=False)
    shutil.rmtree(directory / "attempts")
    seconds = cap if wall is None else wall
    for index, candidate in enumerate(queue):
        formula = conditioned_formula(base, candidate)
        attempt = directory / "attempts" / f"{index:04d}"
        attempt.mkdir(parents=True)
        native = attempt / "input.dimacs"
        native.write_text(formula["native_dimacs"])
        (attempt / "solver_stdout.txt").write_text("")
        (attempt / "solver_stderr.txt").write_text("")
        argv = solver_arguments(cell["profile"], str(paths["cms"]), "", str(native))
        start = 1001.0 + index * 0.02
        atomic_json(
            attempt / "attempt_start.json",
            {
                "candidate_literals": formula["assumption_literals"],
                "native_sha256": audit.sha256_file(native),
                "formula_counts": formula["counts"],
            },
        )
        atomic_json(
            attempt / "native_process.json",
            {"pid": 60000 + index, "process_group": 12345, "argv": argv},
        )
        atomic_json(
            attempt / "attempt.json",
            {
                "index": index,
                "candidate": candidate,
                "status": "capped",
                "returncode": returncode,
                "native_sha256": audit.sha256_file(native),
                "stdout_sha256": audit.sha256_file(attempt / "solver_stdout.txt"),
                "stderr_sha256": audit.sha256_file(attempt / "solver_stderr.txt"),
                "solver_arguments": argv,
                "native_launch_receipt": None,
                "started_monotonic": start,
                "ended_monotonic": start + seconds,
                "solve_wall_seconds": seconds,
                "candidate_cap_seconds": cap if attempt_cap is None else attempt_cap,
            },
        )
    journal = read_checked_json(directory / "journal.json")
    raw = {
        "outcome": "queue_exhausted",
        "attempt_count": len(queue),
        "completed_phase_seconds": journal["completed_seconds"],
    }
    row["outcome"] = "queue_exhausted"
    row["worker_result"] = raw
    atomic_json(directory / "worker_result.json", raw)
    atomic_json(directory / "supervisor_result.json", row)
    return row, manifest, paths


def test_audit_accepts_capped_attempts_in_a_capped_cell(evidence):
    row, manifest, paths = make_capped_run(evidence)
    checked = audit.validate_cell(row, manifest, paths)
    assert checked["recovered_key"] is None
    assert checked["completed_attempts"] == 32


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("uncapped_cell", "Capped attempt in uncapped cell or cap differs"),
        ("wrong_returncode", "Capped attempt was not SIGKILL-terminated"),
        ("finished_before_cap", "Capped attempt finished before its cap"),
        ("cap_mismatch", "Capped attempt in uncapped cell or cap differs"),
    ],
)
def test_audit_rejects_capped_attempts_that_break_the_cap_rules(
    evidence, damage, message
):
    kwargs = {}
    if damage == "uncapped_cell":
        kwargs["cell_cap"] = None
    elif damage == "wrong_returncode":
        kwargs["returncode"] = 10
    elif damage == "finished_before_cap":
        kwargs["wall"] = 0.005
    else:
        kwargs["attempt_cap"] = 0.02
    row, manifest, paths = make_capped_run(evidence, **kwargs)
    with pytest.raises(ValueError, match=message):
        audit.validate_cell(row, manifest, paths)


def test_capped_true_candidate_cannot_be_claimed_recovered(evidence):
    row, manifest, paths = make_capped_run(evidence)
    _, _, _, directory, _ = evidence
    # A consistent but false claim: the raw record and supervisor row say recovered.
    claim = {
        **row["worker_result"],
        "outcome": "recovered",
        "recovered_key": None,
        "verified_known": False,
        "verified_fresh": False,
    }
    row["outcome"] = "recovered"
    row["worker_result"] = claim
    atomic_json(directory / "worker_result.json", claim)
    atomic_json(directory / "supervisor_result.json", row)
    with pytest.raises(ValueError, match="lacks raw model/fresh verification"):
        audit.validate_cell(row, manifest, paths)
