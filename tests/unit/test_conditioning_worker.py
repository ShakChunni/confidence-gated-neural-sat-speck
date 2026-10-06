"""Synthetic correctness and interface tests for conditioning-worker cells."""

from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from neural_sat_speck.backend.formula import build_formula
from neural_sat_speck.backend.guard import (
    atomic_json,
    canonical_bytes,
    read_checked_json,
    sha256_file,
)
from neural_sat_speck.followups import conditioning_worker as worker


def _cipher_api(cipher: str):
    if cipher == "speck32_64":
        from neural_sat_speck.speck.cipher import encrypt_block
        from neural_sat_speck.speck.key_schedule import expand_key
    else:
        from neural_sat_speck.simon.cipher import encrypt_block
        from neural_sat_speck.simon.key_schedule import expand_key
    return encrypt_block, expand_key


def _case(
    cipher: str, key_index: int = 0
) -> tuple[dict[str, Any], dict[str, Any], tuple[int, ...]]:
    rounds = worker.ROUND_PROFILE[cipher]
    key = (0x1234 + key_index, 0x5678, 0x9ABC, 0xDEF0)
    encrypt, expand = _cipher_api(cipher)
    points = [
        ((0x0001, 0x0203), (0x0405, 0x0607)),
        ((0x1111, 0x2222), (0x3333, 0x4444)),
    ]
    pairs = [
        {
            "plaintext0": list(p0),
            "plaintext1": list(p1),
            "ciphertext0": list(encrypt(p0, key, rounds)),
            "ciphertext1": list(encrypt(p1, key, rounds)),
        }
        for p0, p1 in points
    ]
    public = {
        "instance_id": f"{cipher}_k{key_index}_o4_m1",
        "rounds": rounds,
        "pairs": pairs,
        "trailing_round_keys": [expand(key, rounds)[-1]],
    }
    fresh_pts = [(0xABCD, 0x1234), (0xFEDC, 0x5678)]
    truth = {
        "master_key": list(key),
        "verification_pairs": [
            [list(pt), list(encrypt(pt, key, rounds))] for pt in fresh_pts
        ],
    }
    return public, truth, key


def _cell(cipher: str, key_index: int, arm: str) -> dict[str, Any]:
    case_id = f"{cipher}_k{key_index}_o4_m1"
    return {
        "cell_id": f"{case_id}__{arm}",
        "case_id": case_id,
        "cipher": cipher,
        "rounds": worker.ROUND_PROFILE[cipher],
        "key_index": key_index,
        "observations": 4,
        "trailing_keys": 1,
        "arm": arm,
    }


def _formula_sha(formula: dict[str, Any]) -> str:
    identity = {
        name: formula[name]
        for name in (
            "num_vars",
            "cnf_clauses",
            "xor_clauses",
            "assumption_literals",
            "varmap",
        )
    }
    return hashlib.sha256(canonical_bytes(identity)).hexdigest()


def _satisfying_model(formula: dict[str, Any], key: tuple[int, ...]):
    from pycryptosat import Solver

    solver = Solver(verbose=0, threads=1)
    for clause in formula["cnf_clauses"]:
        solver.add_clause(clause)
    for record in formula["xor_clauses"]:
        solver.add_xor_clause(record["literals"], record["rhs"])
    for word, value in zip(formula["varmap"]["master_key"], key, strict=True):
        for bit, variable in enumerate(word):
            solver.add_clause([variable if (value >> bit) & 1 else -variable])
    for literal in formula["assumption_literals"]:
        solver.add_clause([literal])
    sat, solution = solver.solve()
    assert sat is True
    return solution


class _SpySolver:
    def __init__(self, model):
        self.model = model
        self.cnf: list[list[int]] = []
        self.xors: list[tuple[list[int], bool]] = []
        self.units: list[list[int]] = []
        self.solve_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def add_clause(self, clause):
        copied = list(clause)
        self.cnf.append(copied)
        if len(copied) == 1:
            self.units.append(copied)
        return True

    def add_xor_clause(self, literals, rhs):
        self.xors.append((list(literals), rhs))
        return True

    def solve(self, *args, **kwargs):
        self.solve_calls.append((args, kwargs))
        return True, self.model


def _payload(tmp_path: Path, cipher: str, key_index: int, arm: str):
    public, truth, key = _case(cipher, key_index)
    directory = tmp_path / f"{key_index:032x}"
    directory.mkdir()
    public_path = tmp_path / f"{cipher}_{key_index}_public.json"
    truth_path = tmp_path / f"{cipher}_{key_index}_truth.json"
    atomic_json(public_path, public)
    atomic_json(truth_path, truth)
    formula = build_formula(cipher, public, 4, 1, include_cnf=False)
    payload = {
        "cell": _cell(cipher, key_index, arm),
        "invocation_id": directory.name,
        "directory": str(directory),
        "repo_root": str(tmp_path),
        "source_commit": "1" * 40,
        "python_path": sys.executable,
        "public_path": str(public_path),
        "public_sha256": sha256_file(public_path),
        "truth_path": str(truth_path),
        "truth_sha256": sha256_file(truth_path),
        "formula_sha256": _formula_sha(formula),
        "deadline_monotonic": 1.0e12,
    }
    return payload, public, truth, key, formula


def _accepted_environment(monkeypatch: pytest.MonkeyPatch, commit: str) -> None:
    telemetry = SimpleNamespace(
        is_certified=True,
        rejection_reason=None,
        git_commit=commit,
        to_dict=lambda: {
            "is_certified": True,
            "git_commit": commit,
            "pid": 123,
            "parent_pid": 456,
            "python_executable": sys.executable,
            "qos_class": 33,
            "darwin_priority": 0,
            "darwin_bg_state": 0,
            "io_policy_state": 0,
            "thread_environment": {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            },
            "tmpdir_safe": True,
        },
    )
    monkeypatch.setattr(worker, "observe_worker_telemetry", lambda *_: telemetry)
    monkeypatch.setattr(worker, "certify_worker_or_raise", lambda *_: telemetry)
    monkeypatch.setattr(
        worker,
        "power_sample",
        lambda: {
            "accepted": True,
            "power_source": "Battery Power",
            "sleep_assertion_observed": True,
        },
    )


def test_production_solver_factory_uses_the_fixed_wrapper_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    sentinel = object()

    def construct(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setitem(sys.modules, "pycryptosat", SimpleNamespace(Solver=construct))
    assert worker.new_solver() is sentinel
    assert calls == [{"verbose": 0, "threads": 1}]


@pytest.mark.parametrize("cipher", ["speck32_64", "simon32_64"])
@pytest.mark.parametrize("arm", ["permanent_units", "temporary_assumptions"])
def test_worker_loads_identical_formula_and_changes_only_conditioning_interface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cipher: str, arm: str
) -> None:
    payload, _public_value, _truth_value, key, formula = _payload(
        tmp_path, cipher, 0, arm
    )
    model = _satisfying_model(formula, key)
    spy = _SpySolver(model)
    _accepted_environment(monkeypatch, payload["source_commit"])
    constructors = []

    def factory():
        constructors.append({"verbose": 0, "threads": 1})
        return spy

    monkeypatch.setattr(worker, "new_solver", factory)
    events = []
    original_read = worker.read_checked_json

    def checked_read(path):
        if Path(path) == Path(payload["truth_path"]):
            events.append("truth")
        return original_read(path)

    monkeypatch.setattr(worker, "read_checked_json", checked_read)
    original_known = worker._reference_known_check

    def known(*args):
        events.append("known")
        return original_known(*args)

    monkeypatch.setattr(worker, "_reference_known_check", known)
    result = worker.execute(payload)

    assert result["outcome"] == "recovered"
    assert result["solver_status"] == "sat"
    assert result["formula_valid"] is True
    assert result["verified_known"] is True
    assert result["verified_fresh"] is True
    assert result["assigned_key_match"] is True
    assert result["recovered_key"] == list(key)
    assert result["model"][0] is None
    assert len(result["model"]) == formula["num_vars"] + 1
    assert constructors == [{"verbose": 0, "threads": 1}]
    assert spy.cnf[: len(formula["cnf_clauses"])] == formula["cnf_clauses"]
    assert spy.xors == [
        (record["literals"], record["rhs"]) for record in formula["xor_clauses"]
    ]
    true_literals = formula["assumption_literals"]
    permanent_units = [
        clause for clause in spy.cnf[len(formula["cnf_clauses"]) :] if len(clause) == 1
    ]
    if arm == "permanent_units":
        assert permanent_units == [[literal] for literal in true_literals]
        assert spy.solve_calls == [((), {})]
    else:
        assert permanent_units == []
        assert spy.solve_calls == [(tuple([true_literals]), {})]
    dispatch = read_checked_json(Path(payload["directory"]) / "solver_dispatch.json")
    assert dispatch == {
        "invocation_id": payload["invocation_id"],
        "cell_id": payload["cell"]["cell_id"],
        "formula_sha256": payload["formula_sha256"],
        "arm": arm,
        "assumptions": true_literals if arm == "temporary_assumptions" else [],
        "permanent_unit_count": 16 if arm == "permanent_units" else 0,
        "solver_configuration": {"verbose": 0, "threads": 1},
        "requested_at_monotonic": dispatch["requested_at_monotonic"],
    }
    assert type(dispatch["requested_at_monotonic"]) is float
    assert dispatch["requested_at_monotonic"] <= result["completed_at_monotonic"]
    assert events == ["known", "truth"]
    assert "solving" in result["phase_seconds"]
    assert result["elapsed_seconds"] >= result["phase_seconds"]["solving"]
    assert (
        read_checked_json(Path(payload["directory"]) / "journal.json")["phase"] is None
    )


def test_temporary_assumptions_are_removed_and_permanent_units_persist() -> None:
    from pycryptosat import Solver

    temporary = Solver(verbose=0, threads=1)
    temporary.add_clause([1])
    assert temporary.solve([-1])[0] is False
    assert temporary.solve()[0] is True

    permanent = Solver(verbose=0, threads=1)
    permanent.add_clause([1])
    permanent.add_clause([-1])
    assert permanent.solve()[0] is False


@pytest.mark.parametrize("cipher", ["speck32_64", "simon32_64"])
def test_reference_known_check_uses_cipher_and_key_schedule(cipher: str) -> None:
    public, _truth, key = _case(cipher)
    assert worker._reference_known_check(
        cipher, worker.ROUND_PROFILE[cipher], list(key), public
    )
    wrong = list(key)
    wrong[0] ^= 1
    assert not worker._reference_known_check(
        cipher, worker.ROUND_PROFILE[cipher], wrong, public
    )


def test_invalid_model_is_correctness_alarm_and_truth_is_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, _public_value, _truth_value, key, formula = _payload(
        tmp_path, "speck32_64", 0, "temporary_assumptions"
    )
    model = list(_satisfying_model(formula, key))
    model[1] = not model[1]
    spy = _SpySolver(model)
    _accepted_environment(monkeypatch, payload["source_commit"])
    monkeypatch.setattr(worker, "new_solver", lambda: spy)
    original_read = worker.read_checked_json
    truth_reads = []

    def checked_read(path):
        if Path(path) == Path(payload["truth_path"]):
            truth_reads.append(path)
        return original_read(path)

    monkeypatch.setattr(worker, "read_checked_json", checked_read)
    result = worker.execute(payload)
    assert result["outcome"] == "correctness_alarm"
    assert result["formula_valid"] is False
    assert result["recovered_key"] is None
    assert truth_reads == []


def test_known_check_failure_stops_before_truth_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, _public_value, _truth_value, key, formula = _payload(
        tmp_path, "simon32_64", 0, "permanent_units"
    )
    monkeypatch.setattr(
        worker, "new_solver", lambda: _SpySolver(_satisfying_model(formula, key))
    )
    _accepted_environment(monkeypatch, payload["source_commit"])
    monkeypatch.setattr(worker, "_reference_known_check", lambda *_: False)
    original_read = worker.read_checked_json
    truth_reads = []

    def checked_read(path):
        if Path(path) == Path(payload["truth_path"]):
            truth_reads.append(path)
        return original_read(path)

    monkeypatch.setattr(worker, "read_checked_json", checked_read)
    result = worker.execute(payload)
    assert result["outcome"] == "correctness_alarm"
    assert result["formula_valid"] is True
    assert result["verified_known"] is False
    assert truth_reads == []


def test_valid_sat_model_without_fresh_match_is_retained_as_nonunique_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, _public_value, _truth_value, key, formula = _payload(
        tmp_path, "speck32_64", 0, "temporary_assumptions"
    )
    monkeypatch.setattr(
        worker, "new_solver", lambda: _SpySolver(_satisfying_model(formula, key))
    )
    _accepted_environment(monkeypatch, payload["source_commit"])
    monkeypatch.setattr(
        worker,
        "verify_key",
        lambda *_: {
            "verified_known": True,
            "verified_fresh": False,
            "assigned_key_match": False,
        },
    )
    result = worker.execute(payload)
    assert result["outcome"] == "sat_model_not_fresh_verified"
    assert result["formula_valid"] is True
    assert result["verified_known"] is True
    assert result["verified_fresh"] is False
    assert result["assigned_key_match"] is False
    assert result["recovered_key"] == list(key)


def test_truth_hash_is_checked_only_after_known_model_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, _public_value, _truth_value, key, formula = _payload(
        tmp_path, "simon32_64", 0, "temporary_assumptions"
    )
    payload["truth_sha256"] = "0" * 64
    monkeypatch.setattr(
        worker, "new_solver", lambda: _SpySolver(_satisfying_model(formula, key))
    )
    _accepted_environment(monkeypatch, payload["source_commit"])
    original_read = worker.read_checked_json
    events = []

    def checked_read(path):
        if Path(path) == Path(payload["truth_path"]):
            events.append("truth")
        return original_read(path)

    monkeypatch.setattr(worker, "read_checked_json", checked_read)
    original_known = worker._reference_known_check

    def known(*args):
        events.append("known")
        return original_known(*args)

    monkeypatch.setattr(worker, "_reference_known_check", known)
    result = worker.execute(payload)
    assert result["outcome"] == "worker_exception"
    assert result["formula_valid"] is True
    assert events == ["known", "truth"]


def test_true_key_unsat_is_a_distinct_correctness_alarm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, _public_value, _truth_value, _key, formula = _payload(
        tmp_path, "simon32_64", 0, "permanent_units"
    )

    class UnsatSpy(_SpySolver):
        def solve(self, *args, **kwargs):
            self.solve_calls.append((args, kwargs))
            return False, None

    _accepted_environment(monkeypatch, payload["source_commit"])
    spy = UnsatSpy(None)
    monkeypatch.setattr(worker, "new_solver", lambda: spy)
    result = worker.execute(payload)
    assert result["outcome"] == "correctness_alarm"
    assert result["solver_status"] == "unsat"
    assert result["verified_known"] is False
    assert spy.solve_calls == [((), {})]


def test_environment_rejection_keeps_initial_telemetry_and_power_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, *_ = _payload(tmp_path, "speck32_64", 0, "permanent_units")
    telemetry = SimpleNamespace(
        is_certified=False,
        rejection_reason="synthetic rejection",
        git_commit=payload["source_commit"],
        to_dict=lambda: {
            "is_certified": False,
            "rejection_reason": "synthetic rejection",
        },
    )
    monkeypatch.setattr(worker, "observe_worker_telemetry", lambda *_: telemetry)
    monkeypatch.setattr(
        worker,
        "power_sample",
        lambda: {"accepted": True, "power_source": "Battery Power"},
    )
    monkeypatch.setattr(
        worker,
        "certify_worker_or_raise",
        lambda *_: pytest.fail("rejected initial telemetry must fail closed"),
    )
    monkeypatch.setattr(
        worker, "new_solver", lambda: pytest.fail("solver must not start")
    )
    result = worker.execute(payload)
    directory = Path(payload["directory"])
    assert result["outcome"] == "environment_rejected"
    assert (
        read_checked_json(directory / "python_worker_observation.json")["is_certified"]
        is False
    )
    assert (
        read_checked_json(directory / "power_at_worker_start.json")["power_source"]
        == "Battery Power"
    )
    assert not (directory / "python_worker_receipt.json").exists()


def test_thread_environment_mismatch_stops_before_solver_certification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, *_ = _payload(tmp_path, "speck32_64", 0, "temporary_assumptions")
    telemetry = SimpleNamespace(
        is_certified=True,
        rejection_reason=None,
        git_commit=payload["source_commit"],
        to_dict=lambda: {
            "is_certified": True,
            "git_commit": payload["source_commit"],
            "python_executable": payload["python_path"],
            "thread_environment": {"OMP_NUM_THREADS": "8"},
        },
    )
    monkeypatch.setattr(worker, "observe_worker_telemetry", lambda *_: telemetry)
    monkeypatch.setattr(
        worker,
        "power_sample",
        lambda: {"accepted": True, "power_source": "AC Power"},
    )
    monkeypatch.setattr(
        worker,
        "certify_worker_or_raise",
        lambda *_: pytest.fail("thread mismatch must fail before certification"),
    )
    monkeypatch.setattr(
        worker, "new_solver", lambda: pytest.fail("solver must not start")
    )
    result = worker.execute(payload)
    assert result["outcome"] == "environment_rejected"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rounds", True),
        ("key_index", 8),
        ("observations", 2),
        ("trailing_keys", 0),
        ("arm", "unknown"),
    ],
)
def test_invalid_fixed_cell_is_rejected_before_environment_or_solver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    payload, *_ = _payload(tmp_path, "speck32_64", 0, "temporary_assumptions")
    broken = copy.deepcopy(payload)
    broken["cell"][field] = value
    monkeypatch.setattr(
        worker,
        "observe_worker_telemetry",
        lambda *_: pytest.fail("invalid cell reached environment observation"),
    )
    result = worker.execute(broken)
    assert result["outcome"] == "worker_exception"
    assert "not_started" == result["solver_status"]


def test_main_reads_checked_payload_and_atomically_persists_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload_path = tmp_path / "payload.json"
    payload = {"directory": str(tmp_path / f"{'a' * 32}")}
    payload_path.write_bytes(b"test payload")
    captured = {}
    monkeypatch.setattr(worker.sys, "argv", ["conditioning_worker", str(payload_path)])

    def checked_read(path):
        assert Path(path) == payload_path
        captured["checked_read"] = True
        return payload

    monkeypatch.setattr(worker, "read_checked_json", checked_read)
    monkeypatch.setattr(
        worker, "execute", lambda value: {"outcome": "solver_unknown", **value}
    )
    worker.main()
    result_path = Path(payload["directory"]) / "worker_result.json"
    assert captured["checked_read"] is True
    assert read_checked_json(result_path)["outcome"] == "solver_unknown"
