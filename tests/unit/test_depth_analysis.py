"""Failure-denominator and unavailable-metric regressions; no cipher solving."""

import importlib.util
from pathlib import Path

import pytest

from neural_sat_speck.backend.guard import atomic_json, read_checked_json


@pytest.fixture
def analysis_module():
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "analyze_cross_cipher_depth_map.py"
    )
    spec = importlib.util.spec_from_file_location("depth_analysis_fixture", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def ledger(tmp_path):
    cells = {}
    points = {
        "speck32_64": [(6, 0), (6, 1), (7, 1), (7, 2), (8, 2), (8, 3)],
        "simon32_64": [(9, 0), (9, 1), (10, 1), (10, 2), (11, 2), (11, 3)],
    }
    for cipher, pairs in points.items():
        for r, m in pairs:
            for o in (4, 128):
                for key in range(8):
                    ident = f"{cipher}_r{r}_m{m}_o{o}_k{key}"
                    timeout = (
                        cipher == "simon32_64"
                        and o == 128
                        and ((r, m) == (10, 1) or ((r, m) == (9, 0) and key < 6))
                    )
                    solve = (
                        (20.0 if key == 6 else 40.0)
                        if cipher == "simon32_64"
                        and (r, m, o) == (9, 0, 128)
                        and not timeout
                        else 10.0
                    )
                    cells[ident] = {
                        "cell_id": ident,
                        "cipher": cipher,
                        "rounds": r,
                        "assumed_keys_count": m,
                        "nominal_residual_depth": r - m,
                        "observations": o,
                        "instance_index": key,
                        "outcome": "parent_watchdog_timeout"
                        if timeout
                        else "verified_recovery",
                        "verified_recovery": not timeout,
                        "parent_wall_seconds": 300.01 if timeout else solve + 1,
                        "solve_wall_seconds": 0.0 if timeout else solve,
                        "variables_count": 0 if timeout else 10,
                        "cnf_clauses_count": 0 if timeout else 20,
                        "xor_clauses_count": 0 if timeout else 30,
                    }
    path = tmp_path / "raw.json"
    atomic_json(path, {"cells": cells})
    return path


def test_timeouts_are_unavailable_and_keep_failure_penalty(
    analysis_module, ledger, tmp_path
):
    output = tmp_path / "reviewed.json"
    raw = ledger.read_bytes()
    result = analysis_module.analyze_map(ledger, output)
    mixed = next(
        r
        for r in result["grid_breakdown"]
        if (r["cipher"], r["rounds"], r["assumed_keys"], r["observations"])
        == ("simon32_64", 9, 0, 128)
    )
    assert mixed["median_solve_seconds"] == 30.0
    assert mixed["solve_duration_available_count"] == 2
    assert mixed["solve_duration_unavailable_count"] == 6
    assert mixed["median_parent_seconds"] == 300.01
    assert mixed["par2_seconds"] == (6 * 600 + 21 + 41) / 8
    assert (
        mixed["variables_count"],
        mixed["cnf_clauses_count"],
        mixed["xor_clauses_count"],
    ) == (10, 20, 30)
    missing = next(
        r
        for r in result["grid_breakdown"]
        if (r["cipher"], r["rounds"], r["assumed_keys"], r["observations"])
        == ("simon32_64", 10, 1, 128)
    )
    assert missing["median_solve_seconds"] is None
    assert missing["variables_count"] is None
    assert missing["par2_seconds"] == 600.0
    assert missing["formula_count_source"] == "unavailable_in_raw_workers"
    assert ledger.read_bytes() == raw
    assert read_checked_json(output) == result
    with pytest.raises(FileExistsError):
        analysis_module.analyze_map(ledger, output)


@pytest.mark.parametrize("defect", ["missing", "identity", "flag", "counts", "late"])
def test_analysis_rejects_grid_or_metric_inconsistency(
    analysis_module, ledger, tmp_path, defect
):
    data = read_checked_json(ledger)
    row = next(r for r in data["cells"].values() if r["outcome"] == "verified_recovery")
    if defect == "missing":
        data["cells"].pop(row["cell_id"])
    elif defect == "identity":
        row["instance_index"] = 1
    elif defect == "flag":
        row["verified_recovery"] = "true"
    elif defect == "counts":
        row["variables_count"] = 11
    else:
        row["parent_wall_seconds"] = 300.01
    atomic_json(ledger, data)
    with pytest.raises(ValueError):
        analysis_module.analyze_map(ledger, tmp_path / "invalid.json")


def test_hash_corruption_cannot_be_analyzed(analysis_module, ledger, tmp_path):
    ledger.write_text(ledger.read_text() + " ")
    with pytest.raises(ValueError, match="Checksum"):
        analysis_module.analyze_map(ledger, tmp_path / "invalid.json")
