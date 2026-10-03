"""Unit tests for XOR-to-CNF translation and semantic equivalence."""

from __future__ import annotations

import itertools

import pytest

from neural_sat_speck.sat.encoding import VariablePool
from neural_sat_speck.sat.xor_to_cnf import (
    convert_xor_clauses_to_cnf,
    xor_to_cnf_direct,
    xor_to_cnf_tseitin,
)


def eval_clause(clause: list[int], assignment: dict[int, bool]) -> bool:
    """Evaluate whether a CNF disjunctive clause is satisfied under assignment."""
    for lit in clause:
        var = abs(lit)
        val = assignment[var]
        lit_val = val if lit > 0 else not val
        if lit_val:
            return True
    return False


def eval_cnf(clauses: list[list[int]], assignment: dict[int, bool]) -> bool:
    """Evaluate whether a conjunction of CNF clauses is satisfied under assignment."""
    return all(eval_clause(clause, assignment) for clause in clauses)


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("rhs", [False, True])
def test_xor_to_cnf_direct_exhaustive_equivalence(k: int, rhs: bool):
    """Exhaustively verify that direct CNF expansion matches XOR semantics for all 2^k inputs."""
    vars = list(range(1, k + 1))
    clauses = xor_to_cnf_direct(vars, rhs)
    assert len(clauses) == 2 ** (k - 1)

    for bits in itertools.product((False, True), repeat=k):
        assignment = dict(zip(vars, bits, strict=True))
        expected_xor = (sum(bits) % 2) == (1 if rhs else 0)
        actual_cnf = eval_cnf(clauses, assignment)
        assert actual_cnf == expected_xor, (
            f"Failed for k={k}, rhs={rhs}, assignment={assignment}: "
            f"expected XOR={expected_xor}, got CNF={actual_cnf}"
        )


@pytest.mark.parametrize("k", [4, 5, 6, 7])
@pytest.mark.parametrize("rhs", [False, True])
def test_xor_to_cnf_tseitin_exhaustive_equivalence(k: int, rhs: bool):
    """Exhaustively verify that Tseitin CNF decomposition matches XOR semantics via projection."""
    pool = VariablePool(next_variable=1)
    input_vars = [pool.new() for _ in range(k)]
    clauses = xor_to_cnf_tseitin(input_vars, rhs, pool)

    all_vars = list(range(1, pool.variable_count + 1))
    aux_vars = [v for v in all_vars if v not in input_vars]

    for input_bits in itertools.product((False, True), repeat=k):
        input_assignment = dict(zip(input_vars, input_bits, strict=True))
        expected_xor = (sum(input_bits) % 2) == (1 if rhs else 0)

        # Check if there exists ANY assignment to aux_vars satisfying all clauses
        exists_satisfying_extension = False
        for aux_bits in itertools.product((False, True), repeat=len(aux_vars)):
            full_assignment = dict(input_assignment)
            full_assignment.update(zip(aux_vars, aux_bits, strict=True))
            if eval_cnf(clauses, full_assignment):
                exists_satisfying_extension = True
                break

        assert exists_satisfying_extension == expected_xor, (
            f"Tseitin projection failed for k={k}, rhs={rhs}, input={input_assignment}: "
            f"expected XOR={expected_xor}, got extension={exists_satisfying_extension}"
        )


def test_xor_to_cnf_edge_cases():
    """Verify empty literal list and corner cases."""
    assert xor_to_cnf_direct([], False) == []
    assert xor_to_cnf_direct([], True) == [[]]


def test_convert_xor_clauses_to_cnf():
    """Test converting a heterogeneous set of XOR clauses."""
    pool = VariablePool(next_variable=1)
    x1, x2, x3, x4, x5, x6 = [pool.new() for _ in range(6)]

    xor_clauses = [
        ([x1, x2], False),               # k=2 direct
        ([x2, x3, x4], True),            # k=3 direct
        ([x1, x3, x4, x5, x6], False),   # k=5 direct (max_direct_size=5)
    ]

    cnf_clauses, aux_count = convert_xor_clauses_to_cnf(xor_clauses, pool, max_direct_size=5)
    # k=2 -> 2 clauses; k=3 -> 4 clauses; k=5 -> 16 clauses. Total = 22 clauses.
    assert len(cnf_clauses) == 2 + 4 + 16
    assert aux_count == 0  # No aux needed for k <= 5
