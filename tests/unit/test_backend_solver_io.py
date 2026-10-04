"""Unit tests for backend argv profiles and strict solver-output parsing."""

from __future__ import annotations

import pytest

from neural_sat_speck.backend.solver_io import (
    parse_solver_output,
    solver_arguments,
)


def test_native_gauss_profiles_differ_only_by_matrix_limit():
    gauss = solver_arguments(
        "cms_native_gauss", "/bin/cms", "/bin/cadical", "/tmp/input formula.xor"
    )
    no_gauss = solver_arguments(
        "cms_native_no_gauss",
        "/bin/cms",
        "/bin/cadical",
        "/tmp/input formula.xor",
    )
    gauss_limit = gauss.index("--maxnummatrices") + 1
    no_gauss_limit = no_gauss.index("--maxnummatrices") + 1
    assert gauss[gauss_limit] == "5"
    assert no_gauss[no_gauss_limit] == "0"
    gauss[gauss_limit] = "MATRIX_LIMIT"
    no_gauss[no_gauss_limit] = "MATRIX_LIMIT"
    assert gauss == no_gauss
    assert gauss[-1] == "/tmp/input formula.xor"


def test_native_default_keeps_cms_defaults_and_pins_one_thread():
    assert solver_arguments(
        "cms_native_default", "/bin/cms", "/bin/cadical", "formula.xor"
    ) == ["/bin/cms", "--verb", "1", "--threads", "1", "formula.xor"]


def test_cnf_profile_changes_only_input_representation_from_x():
    no_gauss = solver_arguments(
        "cms_native_no_gauss", "/bin/cms", "/bin/cadical", "formula.cnf"
    )
    cnf = solver_arguments(
        "cms_cnf_no_gauss", "/bin/cms", "/bin/cadical", "formula.cnf"
    )
    assert cnf == no_gauss
    assert cnf[-3:] == ["--xor", "0", "formula.cnf"]


def test_cadical_uses_quiet_mode_and_preserves_path_as_one_argument():
    assert solver_arguments(
        "cadical_cnf", "/bin/cms", "/bin/cadical", "/tmp/input with spaces.cnf"
    ) == ["/bin/cadical", "-q", "/tmp/input with spaces.cnf"]


@pytest.mark.parametrize(
    ("profile", "cms_path", "cadical_path", "formula_path"),
    [
        ("unknown", "/bin/cms", "/bin/cadical", "formula.cnf"),
        ("cms_native_default", "", "/bin/cadical", "formula.cnf"),
        ("cadical_cnf", "/bin/cms", "", "formula.cnf"),
        ("cms_native_default", "/bin/cms", "/bin/cadical", ""),
    ],
)
def test_solver_arguments_reject_unknown_profiles_and_missing_paths(
    profile: str,
    cms_path: str,
    cadical_path: str,
    formula_path: str,
):
    with pytest.raises(ValueError):
        solver_arguments(profile, cms_path, cadical_path, formula_path)


def test_sat_model_can_span_lines_and_returns_raw_gauss_diagnostics():
    parsed = parse_solver_output(
        "cms",
        10,
        "c Gaussian matrix rows: 5\ns SATISFIABLE\nv 1 -2\nv 3 -4 0\nc done\n",
        required_variables=[1, 2, 3, 4],
        num_vars=7,
    )
    assert parsed["status"] == "sat"
    assert parsed["assignment"] == {1: True, 2: False, 3: True, 4: False}
    assert parsed["error"] is None
    assert parsed["gauss_evidence"] == {
        "matching_log_lines": ["c Gaussian matrix rows: 5"],
        "matrix_count_samples": [],
        "is_proof_of_gauss_use": False,
    }


def test_cadical_signed_dimacs_model_and_comments():
    parsed = parse_solver_output(
        "cadical",
        10,
        "c solver comment\ns SATISFIABLE\nv -1 +2 0\n",
        required_variables=[1, 2],
        num_vars=3,
    )
    assert parsed["status"] == "sat"
    assert parsed["assignment"] == {1: False, 2: True}


@pytest.mark.parametrize("engine", ["cms", "cadical"])
def test_empty_unsat_result_has_empty_assignment(engine: str):
    parsed = parse_solver_output(
        engine,
        20,
        "s UNSATISFIABLE\n",
        required_variables=[1, 2],
        num_vars=2,
    )
    assert parsed["status"] == "unsat"
    assert parsed["assignment"] == {}
    assert parsed["error"] is None


@pytest.mark.parametrize(
    ("engine", "returncode", "stdout"),
    [
        ("cms", 15, "s INDETERMINATE\n"),
        ("cadical", 0, "s UNKNOWN\n"),
    ],
)
def test_unknown_outcomes(engine: str, returncode: int, stdout: str):
    parsed = parse_solver_output(
        engine,
        returncode,
        stdout,
        required_variables=[1],
        num_vars=1,
    )
    assert parsed["status"] == "unknown"
    assert parsed["assignment"] == {}


@pytest.mark.parametrize(
    "engine,code,text", [("cms", 0, "s UNKNOWN\n"), ("cadical", 0, "")]
)
def test_unknown_needs_real_exit_semantics_and_explicit_output(engine, code, text):
    assert (
        parse_solver_output(engine, code, text, required_variables=[], num_vars=0)[
            "status"
        ]
        == "error"
    )


def test_exit_code_status_disagreement_is_error():
    parsed = parse_solver_output(
        "cadical",
        0,
        "s SATISFIABLE\n",
        required_variables=[],
        num_vars=0,
    )
    assert parsed["status"] == "error"
    assert parsed["assignment"] == {}
    assert "contradicts" in parsed["error"]


@pytest.mark.parametrize(
    ("returncode", "stdout", "required", "num_vars", "message"),
    [
        (10, "v 1 0\n", [1], 1, "no explicit solver status"),
        (10, "s SATISFIABLE\ns SATISFIABLE\nv 1 0\n", [1], 1, "repeated"),
        (10, "s SATISFIABLE\nv 1 0 2\n", [1], 2, "after terminal zero"),
        (10, "s SATISFIABLE\nv 1 0\nv -1 0\n", [1], 1, "after terminal zero"),
        (10, "s SATISFIABLE\nv 1 -1 0\n", [1], 1, "conflicting"),
        (10, "s SATISFIABLE\nv 1 1_0 0\n", [1], 10, "non-integer"),
        (10, "s SATISFIABLE\nv 4 0\n", [1], 3, "outside"),
        (10, "s SATISFIABLE\nv 1 -2\n", [1], 2, "truncated"),
        (10, "s SATISFIABLE\nv 1 0\n", [1, 2], 2, "omits required"),
        (20, "s UNSATISFIABLE\nv 0\n", [], 0, "UNSAT result unexpectedly"),
        (0, "s UNKNOWN\nv 1 0\n", [1], 1, "UNKNOWN result unexpectedly"),
        (10, "s SATISFIABLE extra\nv 1 0\n", [1], 1, "malformed status"),
    ],
)
def test_malformed_or_partial_solver_results_fail_closed(
    returncode: int,
    stdout: str,
    required: list[int],
    num_vars: int,
    message: str,
):
    parsed = parse_solver_output(
        "cadical",
        returncode,
        stdout,
        required_variables=required,
        num_vars=num_vars,
    )
    assert parsed["status"] == "error"
    assert parsed["assignment"] == {}
    assert message in parsed["error"]


@pytest.mark.parametrize(
    ("required", "num_vars"),
    [([0], 2), ([3], 2), ([1, 1], 2), ([True], 2), ([1], -1)],
)
def test_invalid_requested_variable_sets_are_rejected(required, num_vars):
    parsed = parse_solver_output(
        "cadical",
        10,
        "s SATISFIABLE\nv 1 0\n",
        required_variables=required,
        num_vars=num_vars,
    )
    assert parsed["status"] == "error"
    assert parsed["assignment"] == {}


def test_unknown_engine_and_unexpected_exit_code_are_errors():
    common = {"required_variables": [], "num_vars": 0}
    assert (
        parse_solver_output("other", 20, "s UNSATISFIABLE\n", **common)["status"]
        == "error"
    )
    assert parse_solver_output("cms", -9, "", **common)["status"] == "error"
    assert (
        parse_solver_output("cadical", 15, "s UNKNOWN\n", **common)["status"] == "error"
    )
