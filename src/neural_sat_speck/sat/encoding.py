"""Small exact Boolean encoders used by the Phase 2 SAT pilot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

WordBits: TypeAlias = tuple[int, ...]
SolverLike: TypeAlias = Any


@dataclass
class VariablePool:
    """Allocate positive one-based SAT variable identifiers."""

    next_variable: int = 1

    def new(self) -> int:
        """Allocate and return one fresh variable identifier."""

        variable = self.next_variable
        self.next_variable += 1
        return variable

    def new_bits(self, width: int) -> WordBits:
        """Allocate a little-endian bit vector."""

        if width <= 0:
            raise ValueError("width must be positive")
        return tuple(self.new() for _ in range(width))

    @property
    def variable_count(self) -> int:
        """Return the number of allocated variables."""

        return self.next_variable - 1


def add_fixed_word(
    solver: SolverLike,
    bits: WordBits,
    value: int,
) -> None:
    """Constrain a little-endian bit vector to an integer value."""

    if value < 0:
        raise ValueError("value must be non-negative")
    for index, variable in enumerate(bits):
        solver.add_clause(
            [variable if (value >> index) & 1 else -variable]
        )


def add_xor_bits(
    solver: SolverLike,
    pool: VariablePool,
    left: WordBits,
    right: WordBits,
) -> WordBits:
    """Encode bitwise XOR using native CryptoMiniSat XOR clauses."""

    if len(left) != len(right):
        raise ValueError("XOR operands must have equal widths")
    output = pool.new_bits(len(left))
    for left_bit, right_bit, output_bit in zip(
        left,
        right,
        output,
        strict=True,
    ):
        solver.add_xor_clause(
            [left_bit, right_bit, output_bit],
            False,
        )
    return output


def add_xor_constant(
    solver: SolverLike,
    pool: VariablePool,
    bits: WordBits,
    value: int,
) -> WordBits:
    """Encode bitwise XOR of a vector with an integer constant."""

    if value < 0:
        raise ValueError("value must be non-negative")
    output = pool.new_bits(len(bits))
    for index, (input_bit, output_bit) in enumerate(zip(bits, output, strict=True)):
        solver.add_xor_clause(
            [input_bit, output_bit],
            bool((value >> index) & 1),
        )
    return output


def add_majority(
    solver: SolverLike,
    left: int,
    right: int,
    carry_in: int,
    carry_out: int,
) -> None:
    """Encode carry_out as the majority of three input bits."""

    solver.add_clause([-left, -right, carry_out])
    solver.add_clause([-left, -carry_in, carry_out])
    solver.add_clause([-right, -carry_in, carry_out])
    solver.add_clause([left, right, -carry_out])
    solver.add_clause([left, carry_in, -carry_out])
    solver.add_clause([right, carry_in, -carry_out])


def add_modular_addition(
    solver: SolverLike,
    pool: VariablePool,
    left: WordBits,
    right: WordBits,
) -> WordBits:
    """Encode little-endian addition modulo the operand width."""

    if len(left) != len(right):
        raise ValueError("addition operands must have equal widths")
    if not left:
        raise ValueError("addition operands must not be empty")

    output = pool.new_bits(len(left))
    carry = pool.new()
    solver.add_clause([-carry])

    for index, (left_bit, right_bit, output_bit) in enumerate(
        zip(left, right, output, strict=True)
    ):
        if index == 0:
            solver.add_xor_clause(
                [left_bit, right_bit, output_bit],
                False,
            )
        else:
            solver.add_xor_clause(
                [left_bit, right_bit, carry, output_bit],
                False,
            )

        if index < len(left) - 1:
            next_carry = pool.new()
            add_majority(
                solver,
                left_bit,
                right_bit,
                carry,
                next_carry,
            )
            carry = next_carry

    return output


def rotate_bits_right(bits: WordBits, amount: int) -> WordBits:
    """Wire a little-endian bit vector through a right rotation."""

    width = len(bits)
    shift = amount % width
    return tuple(bits[(index + shift) % width] for index in range(width))


def rotate_bits_left(bits: WordBits, amount: int) -> WordBits:
    """Wire a little-endian bit vector through a left rotation."""

    width = len(bits)
    shift = amount % width
    return tuple(bits[(index - shift) % width] for index in range(width))


def decode_word(
    solution: tuple[bool | None, ...],
    bits: WordBits,
) -> int:
    """Decode a solved bit vector into an integer."""

    value = 0
    for index, variable in enumerate(bits):
        bit = solution[variable]
        if bit is None:
            raise ValueError(f"solver left variable {variable} undefined")
        if bit:
            value |= 1 << index
    return value
