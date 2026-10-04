"""Independent truth-valued fixtures for saved DIMACS assignment auditing."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "backend_completed_evidence",
    Path(__file__).resolve().parents[2] / "scripts/audit_backend_completed_evidence.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("row", ["x1 2 0", "x 1 2 0"])
def test_signed_xor_parity_and_unit_clause(row):
    text = f"p cnf 2 2\n-1 0\n{row}\n"
    assert MODULE.check_dimacs_assignment(text, {1: False, 2: True})["xor_records"] == 1
    with pytest.raises(ValueError, match="violates"):
        MODULE.check_dimacs_assignment(text, {1: False, 2: False})


def test_negative_xor_literal_changes_parity():
    text = "p cnf 2 1\nx-1 2 0\n"
    MODULE.check_dimacs_assignment(text, {1: True, 2: True})
    with pytest.raises(ValueError, match="violates"):
        MODULE.check_dimacs_assignment(text, {1: True, 2: False})


@pytest.mark.parametrize("row", ["1 -2 0", "0", "x0"])
def test_unsatisfied_disjunction_or_empty_constraint_is_rejected(row):
    with pytest.raises(ValueError, match="violates"):
        MODULE.check_dimacs_assignment(f"p cnf 2 1\n{row}\n", {1: False, 2: True})


@pytest.mark.parametrize(
    "assignment", [{1: True}, {1: True, 2: 0}, {1: True, 2: False, 3: True}]
)
def test_incomplete_nonboolean_and_out_of_range_assignments_are_rejected(assignment):
    with pytest.raises(ValueError):
        MODULE.check_dimacs_assignment("p cnf 2 1\n1 0\n", assignment)


@pytest.mark.parametrize(
    "text", ["p cnf 2 2\n1 0\n", "p cnf 2 1\n3 0\n", "p cnf 2 1\n1 0 2 0\n"]
)
def test_malformed_header_range_or_record_is_rejected(text):
    with pytest.raises(ValueError):
        MODULE.check_dimacs_assignment(text, {1: True, 2: False})


def test_decode_is_lsb_first_and_uses_variable_map_order():
    assert MODULE.decode_words(
        {1: True, 2: False, 3: False, 4: True}, [[1, 2], [3, 4], [4, 3]]
    ) == [1, 2, 1]


def test_forced_extension_handles_negative_xor_and_auxiliary_clause():
    model = MODULE.extend_witness([[2, 3], [-2, -3]], [([-1, 2], True)], {1: True}, 3)
    assert model == {1: True, 2: True, 3: False}


@pytest.mark.parametrize("cnf,xors", [([[-1]], []), ([], [([1], False)])])
def test_contradictory_reference_witness_is_rejected(cnf, xors):
    with pytest.raises(ValueError, match="contradicts"):
        MODULE.extend_witness(cnf, xors, {1: True}, 1)


def test_unforced_reference_extension_is_unavailable_not_filled_arbitrarily():
    with pytest.raises(ValueError, match="incomplete"):
        MODULE.extend_witness([[1, 2]], [], {}, 2)
