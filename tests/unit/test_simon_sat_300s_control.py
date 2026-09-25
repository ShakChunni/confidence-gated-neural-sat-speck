"""Boundary tests for the fixed demo-only SIMON 300-second SAT control."""

from __future__ import annotations

import json
import subprocess

import pytest

import scripts.audit_simon_sat_300s as control
from neural_sat_speck.sat.model import SatSolveResult


def test_worker_receives_only_attack_and_verification_observations() -> None:
    payload, key = control._payload(0)
    assert set(payload) == {"rounds", "known_pairs", "verification_pairs"}
    assert len(payload["known_pairs"]) == 4
    assert len(payload["verification_pairs"]) == 2
    assert "master_key" not in json.dumps(payload)
    assert key == control._instances(9)[0].master_key


def test_worker_classifies_limit_as_unresolved(monkeypatch) -> None:
    class LimitModel:
        variable_count = 1
        cnf_clause_count = 1
        xor_clause_count = 1

        @classmethod
        def from_known_pairs(cls, known, rounds):
            assert len(known) == 4 and rounds == 9
            return cls()

        def solve(self, assumptions, time_limit):
            assert assumptions == ()
            assert 0 < time_limit <= 300
            return SatSolveResult(
                satisfiable=False, solution=None, elapsed_seconds=300,
                timed_out=None, status="unsat_or_timeout",
            )

    monkeypatch.setattr(control, "SimonSatModel", LimitModel)
    result = control.worker_evaluate(control._payload(0)[0])
    assert result["outcome"] == "unresolved_solver_limit"
    assert result["solver_status"] == "unsat_or_timeout"
    assert result["recovered_master_key"] is None


def test_worker_rejects_truth_bearing_or_wrong_sized_payload() -> None:
    payload, key = control._payload(0)
    payload["master_key"] = key
    with pytest.raises(ValueError, match="unexpected"):
        control.worker_evaluate(payload)
    payload.pop("master_key")
    payload["known_pairs"] = payload["known_pairs"][:2]
    with pytest.raises(ValueError, match="fixed protocol"):
        control.worker_evaluate(payload)


def test_parent_watchdog_is_unconditional_failure(monkeypatch) -> None:
    def fake_run(command, **kwargs):
        assert command[-1] == "--worker"
        assert kwargs["timeout"] == 300.0
        payload = json.loads(kwargs["input"])
        assert set(payload) == {"rounds", "known_pairs", "verification_pairs"}
        raise subprocess.TimeoutExpired(command, 300.0)

    monkeypatch.setattr(control.subprocess, "run", fake_run)
    result = control.evaluate_index(0)
    assert result["outcome"] == "parent_wall_timeout"
    assert result["verified_recovery_within_cutoff"] is False


def test_resume_refuses_source_drift_and_duplicate_records(tmp_path) -> None:
    output = tmp_path / "control.json"
    data = control._new_result(output, resume=False)
    output.write_text(json.dumps(data), encoding="utf-8")
    assert control._new_result(output, resume=True)["records"] == []
    with pytest.raises(FileExistsError):
        control._new_result(output, resume=False)

    data["records"] = [{"instance_index": 0}, {"instance_index": 0}]
    output.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        control._new_result(output, resume=True)

    data["records"] = []
    data["configuration"]["cutoff_seconds"] = 10.0
    output.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="configuration"):
        control._new_result(output, resume=True)
