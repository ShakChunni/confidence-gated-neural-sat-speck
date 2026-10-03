"""Independent truth-table and malformed-input checks for backend conversion."""

from __future__ import annotations

import itertools

import pytest

from neural_sat_speck.sat.pure_cnf import convert_native_xor_dimacs_to_pure_cnf
from neural_sat_speck.sat.xor_to_cnf import convert_xor_clauses_to_cnf


def _read_cnf(text: str) -> tuple[int, list[list[int]]]:
    lines = text.splitlines()
    header = lines[0].split()
    assert header[:2] == ["p", "cnf"]
    count, declared_clauses = map(int, header[2:])
    clauses = []
    for line in lines[1:]:
        record = list(map(int, line.split()))
        assert record[-1] == 0
        assert all(0 < abs(literal) <= count for literal in record[:-1])
        clauses.append(record[:-1])
    assert len(clauses) == declared_clauses
    return count, clauses


def _literal(literal: int, values: tuple[bool, ...]) -> bool:
    value = values[abs(literal) - 1]
    return value if literal > 0 else not value


def _has_extension(
    clauses: list[list[int]], source: tuple[bool, ...], total_variables: int
) -> bool:
    for suffix in itertools.product((False, True), repeat=total_variables - len(source)):
        values = source + suffix
        if all(any(_literal(lit, values) for lit in clause) for clause in clauses):
            return True
    return False


@pytest.mark.parametrize("record", ["0", "x 0", "x0"])
def test_empty_contradiction_survives_conversion(record: str) -> None:
    output, count = convert_native_xor_dimacs_to_pure_cnf(f"p cnf 0 1\n{record}\n", 0)
    assert count == 0
    assert _read_cnf(output) == (0, [[]])
    assert not _has_extension([[]], (), 0)


def test_empty_formula_is_tautology() -> None:
    output, count = convert_native_xor_dimacs_to_pure_cnf("p cnf 0 0\n", 0)
    assert count == 0
    assert _read_cnf(output) == (0, [])
    assert _has_extension([], (), 0)


@pytest.mark.parametrize("rhs", [False, True])
def test_generic_empty_xor_preserves_both_rhs_values(rhs: bool) -> None:
    clauses, auxiliaries = convert_xor_clauses_to_cnf([([], rhs)])
    assert auxiliaries == 0
    assert _has_extension(clauses, (), 0) == (not rhs)


@pytest.mark.parametrize(
    "literals",
    [
        [1], [-1], [1, -2], [-1, -2, 3], [1, 1], [1, -1],
        [-1, -1, 2], [1, -2, 3, -1, 2, -3],
        [-1, 2, -3, 4, -5, 6, 1],
    ],
)
@pytest.mark.parametrize("rhs", [False, True])
def test_signed_repeated_xors_have_exact_projected_semantics(
    literals: list[int], rhs: bool
) -> None:
    count = max(map(abs, literals))
    exported = list(literals)
    if not rhs:
        exported[0] = -exported[0]
    record = "x " + " ".join(map(str, exported)) + " 0"
    output, total = convert_native_xor_dimacs_to_pure_cnf(
        f"p cnf {count} 1\n{record}\n", count
    )
    cnf_count, clauses = _read_cnf(output)
    assert cnf_count == total
    for values in itertools.product((False, True), repeat=count):
        expected = (sum(_literal(lit, values) for lit in literals) % 2 == 1) == rhs
        assert _has_extension(clauses, values, total) == expected


def test_multiple_long_xors_use_disjoint_fresh_auxiliaries() -> None:
    # Both XORs require a Tseitin chain; their shared source variables are deliberate.
    output, total = convert_native_xor_dimacs_to_pure_cnf(
        "p cnf 6 3\nx 1 -2 3 4 5 6 0\nx -1 2 3 -4 5 6 0\n-6 0\n", 6
    )
    _, clauses = _read_cnf(output)
    assert total == 12
    for source in itertools.product((False, True), repeat=6):
        odd_a = sum(_literal(lit, source) for lit in [1, -2, 3, 4, 5, 6]) % 2 == 1
        odd_b = sum(_literal(lit, source) for lit in [-1, 2, 3, -4, 5, 6]) % 2 == 1
        assert _has_extension(clauses, source, total) == (odd_a and odd_b and not source[5])


def test_declared_unused_variables_are_reserved() -> None:
    output, total = convert_native_xor_dimacs_to_pure_cnf(
        "p cnf 8 1\nx 1 2 3 4 5 6 0\n", 8
    )
    _, clauses = _read_cnf(output)
    assert total == 11
    allocated = {abs(lit) for clause in clauses for lit in clause if abs(lit) > 6}
    assert allocated == {9, 10, 11}


def test_wrapped_and_multiple_records_preserve_semantics() -> None:
    output, _ = convert_native_xor_dimacs_to_pure_cnf(
        "cComment\np cnf 2 3\n1\nc within wrapped clause\n-2 0 -1 0\nx2 0\n", 2
    )
    _, clauses = _read_cnf(output)
    for values in itertools.product((False, True), repeat=2):
        expected = (values[0] or not values[1]) and not values[0] and values[1]
        assert _has_extension(clauses, values, 2) == expected


@pytest.mark.parametrize(
    ("text", "count", "reason"),
    [
        ("1 0\n", 1, "header"),
        ("", 0, "header"),
        ("p cnf 1 1\np cnf 1 1\n1 0\n", 1, "header"),
        ("p cnf 1 0\n1 0\n", 1, "record count"),
        ("p cnf 1 2\n1 0\n", 1, "record count"),
        ("p cnf 1 1\n1\n", 1, "unterminated"),
        ("p cnf 1 1\nx 1\n", 1, "unterminated"),
        ("p cnf 1 1\nx\n", 1, "unterminated"),
        ("p cnf 1 2\n1 x 1 0\n", 1, "unterminated"),
        ("p cnf 1 1\n2 0\n", 1, "range"),
        ("p cnf 1 1\nx -2 0\n", 1, "range"),
        ("p cnf 6 1\nx 1 2 3 4 5 6 0\n", 5, "match"),
        ("p cnf -1 0\n", 0, "negative"),
        ("p cnf 1 -1\n", 1, "negative"),
        ("p cnf one 0\n", 1, "noninteger"),
        ("p cnf 1_0 0\n", 10, "noninteger"),
        ("p cnf 1 1\n١ 0\n", 1, "noninteger"),
        ("p cnf 10 1\n1_0 0\n", 10, "noninteger"),
        ("p cnf 1 0 trailing\n", 1, "expected header"),
        ("p other 1 0\n", 1, "expected header"),
        ("p cnf 1 1\nword 0\n", 1, "noninteger"),
        ("p cnf 1 1\nx 1 0 0\n", 1, "record count"),
        ("p cnf 1 1\n1 0 1\n", 1, "unterminated"),
    ],
)
def test_malformed_input_rejected_before_conversion(
    text: str, count: int, reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        convert_native_xor_dimacs_to_pure_cnf(text, count)


@pytest.mark.parametrize("count", [True, False, -1, 1.0, "1", None])
def test_invalid_supplied_variable_count_rejected(count: object) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        convert_native_xor_dimacs_to_pure_cnf("p cnf 1 0\n", count)  # type: ignore[arg-type]
