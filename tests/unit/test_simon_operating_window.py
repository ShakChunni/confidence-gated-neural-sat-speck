"""Protocol and failure-boundary tests for the demo-only SIMON SAT window."""

from __future__ import annotations

import pytest

import scripts.adjudicate_simon_operating_window as window
from neural_sat_speck.sat.model import SatSolveResult
from scripts.adjudicate_simon_operating_window import (
    INSTANCE_COUNT,
    SOLVE_CUTOFF,
    _instances,
    _load_or_create,
    evaluate_cell,
    grid_screen,
    known_observations,
    wrong_subkey_tuples,
)


def test_observations_are_ciphertexts_not_differential_pairs() -> None:
    instance = _instances(8)[0]
    selected = known_observations(instance, 4)
    assert len(selected) == 4
    assert selected == instance.known_pairs[:4]
    with pytest.raises(ValueError):
        known_observations(instance, 3)
    with pytest.raises(ValueError):
        known_observations(instance, 18)


def test_demo_keys_are_distinct_and_pairing_is_stable() -> None:
    first = _instances(9)
    second = _instances(9)
    assert first == second
    assert len(first) == INSTANCE_COUNT
    assert len({instance.master_key for instance in first}) == INSTANCE_COUNT
    assert all(instance.split_role == "demo" for instance in first)


def test_wrong_tuple_probe_is_unique_truth_excluded_and_reproducible() -> None:
    for truth in ((0x1234,), (0x1234, 0x5678)):
        candidates = wrong_subkey_tuples(truth, 9, 4, 0)
        assert candidates == wrong_subkey_tuples(truth, 9, 4, 0)
        assert len(candidates) == len(set(candidates)) == 32
        assert truth not in candidates
        assert all(len(item) == len(truth) for item in candidates)


def test_grid_screen_applies_paired_counts_and_censored_median() -> None:
    rows = []
    for idx in range(INSTANCE_COUNT):
        for assumed_keys in (0, 1, 2):
            success = (idx < 4) if assumed_keys == 0 else (idx < 7)
            solve_time = 0.1 if success else SOLVE_CUTOFF
            rows.append({
                "rounds": 9,
                "observations": 4,
                "assumed_keys": assumed_keys,
                "instance_index": idx,
                "verified_recovery_within_cutoff": success,
                "solve_wall_seconds": solve_time,
            })
    assert grid_screen(rows, (9,)) == [(9, 4, 1), (9, 4, 2)]
    next(
        row for row in rows
        if row["assumed_keys"] == 1 and row["instance_index"] == 6
    )["verified_recovery_within_cutoff"] = False
    assert grid_screen(rows, (9,)) == [(9, 4, 2)]


def test_checkpoint_refuses_overwrite_and_bad_resume(tmp_path) -> None:
    path = tmp_path / "window.json"
    created = _load_or_create(path, resume=False)
    assert created["status"] == "IN_PROGRESS"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _load_or_create(path, resume=False)
    with pytest.raises(ValueError):
        _load_or_create(path, resume=True)


def test_one_true_key_control_matches_reference_and_fresh_verification() -> None:
    instance = _instances(8)[0]
    row = evaluate_cell(instance, 0, 4, 2)
    assert row["assumed_keys"] == 2
    assert row["differential_pairs"] == 2
    assert row["solver_status"] == "sat"
    assert row["verified_recovery_within_cutoff"] is True
    assert row["outcome"] == "verified_recovery"


def test_solver_timeout_is_unresolved_not_unsat(monkeypatch) -> None:
    class TimeoutModel:
        variable_count = 1
        cnf_clause_count = 1
        xor_clause_count = 0

        @classmethod
        def from_known_pairs(cls, known, rounds):
            assert len(known) == 4 and rounds == 8
            return cls()

        def solve(self, assumptions, time_limit):
            assert assumptions == () and time_limit == SOLVE_CUTOFF
            return SatSolveResult(
                satisfiable=False, solution=None, elapsed_seconds=10.0,
                timed_out=True, status="solver_timeout",
            )

    monkeypatch.setattr(window, "SimonSatModel", TimeoutModel)
    row = evaluate_cell(_instances(8)[0], 0, 4, 0)
    assert row["solver_status"] == "solver_timeout"
    assert row["outcome"] == "solver_timeout_or_unknown"
    assert row["verified_recovery_within_cutoff"] is False


def test_detached_checksum_refuses_modified_artifact(tmp_path) -> None:
    output = tmp_path / "result.json"
    output.write_text('{"status": "COMPLETE"}\n', encoding="utf-8")
    window._write_checksum_sidecar(output)
    window._write_checksum_sidecar(output)
    output.write_text('{"status": "IN_PROGRESS"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="disagrees"):
        window._write_checksum_sidecar(output)
