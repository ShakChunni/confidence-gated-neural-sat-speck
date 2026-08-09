"""Tests for the exact Boolean primitive encoders."""

import pytest

from neural_sat_speck.sat.encoding import (
    VariablePool,
    add_modular_addition,
    add_xor_bits,
    decode_word,
)

pycryptosat = pytest.importorskip("pycryptosat")
Solver = pycryptosat.Solver


def test_four_bit_modular_addition_truth_table() -> None:
    solver = Solver()
    pool = VariablePool()
    left = pool.new_bits(4)
    right = pool.new_bits(4)
    output = add_modular_addition(solver, pool, left, right)

    for left_value in range(16):
        for right_value in range(16):
            assumptions = [
                variable if (value >> index) & 1 else -variable
                for bits, value in ((left, left_value), (right, right_value))
                for index, variable in enumerate(bits)
            ]
            satisfiable, solution = solver.solve(assumptions)
            assert satisfiable
            assert solution is not None
            assert decode_word(solution, output) == (left_value + right_value) & 0xF


def test_native_xor_encoder_matches_truth_table() -> None:
    solver = Solver()
    pool = VariablePool()
    left = pool.new_bits(4)
    right = pool.new_bits(4)
    output = add_xor_bits(solver, pool, left, right)

    for left_value in range(16):
        for right_value in range(16):
            assumptions = [
                variable if (value >> index) & 1 else -variable
                for bits, value in ((left, left_value), (right, right_value))
                for index, variable in enumerate(bits)
            ]
            satisfiable, solution = solver.solve(assumptions)
            assert satisfiable
            assert solution is not None
            assert decode_word(solution, output) == left_value ^ right_value
