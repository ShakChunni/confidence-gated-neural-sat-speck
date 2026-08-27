"""Exact reduced-round SPECK constraints for the Phase 2 pilot."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import TypeAlias

from neural_sat_speck.speck.cipher import Block
from neural_sat_speck.speck.key_schedule import (
    SPECK32_64_ALPHA,
    SPECK32_64_BETA,
    SPECK32_64_ROUND_COUNT,
)
from neural_sat_speck.speck.primitives import WORD_MASK, mask_word

from .encoding import (
    VariablePool,
    WordBits,
    add_fixed_word,
    add_modular_addition,
    add_xor_bits,
    add_xor_constant,
    decode_word,
    rotate_bits_left,
    rotate_bits_right,
)

KnownPair: TypeAlias = tuple[Block, Block]
StateBits: TypeAlias = tuple[WordBits, WordBits]


def _new_solver():
    try:
        from pycryptosat import Solver
    except ImportError as error:
        raise RuntimeError(
            "CryptoMiniSat is required for the SAT pilot; install the "
            "project sat extra or pycryptosat"
        ) from error
    return Solver(verbose=0, threads=1)


def _validate_rounds(rounds: int) -> int:
    if isinstance(rounds, bool) or not isinstance(rounds, int):
        raise TypeError("rounds must be an integer")
    if not 1 <= rounds <= SPECK32_64_ROUND_COUNT:
        raise ValueError(
            f"rounds must be between 1 and {SPECK32_64_ROUND_COUNT}"
        )
    return rounds


def _normalise_block(block: Sequence[int]) -> Block:
    try:
        words = tuple(block)
    except TypeError as error:
        raise TypeError("block must be a sequence of two integers") from error
    if len(words) != 2:
        raise ValueError("block must contain exactly two words")
    return mask_word(words[0]), mask_word(words[1])


def _encode_round(
    solver,
    pool: VariablePool,
    state: StateBits,
    round_key: WordBits,
) -> StateBits:
    """Encode one exact SPECK round over little-endian bit vectors."""

    rotated_x = rotate_bits_right(state[0], SPECK32_64_ALPHA)
    added_x = add_modular_addition(solver, pool, rotated_x, state[1])
    next_x = add_xor_bits(solver, pool, added_x, round_key)

    rotated_y = rotate_bits_left(state[1], SPECK32_64_BETA)
    next_y = add_xor_bits(solver, pool, rotated_y, next_x)
    return next_x, next_y


@dataclass(frozen=True)
class SatSolveResult:
    """Result of one CryptoMiniSat solve call."""

    satisfiable: bool
    solution: tuple[bool | None, ...] | None
    elapsed_seconds: float
    timed_out: bool | None
    status: str


@dataclass
class SpeckSatModel:
    """A reduced-round SPECK model with known plaintext/ciphertext pairs."""

    solver: object
    pool: VariablePool
    known_pairs: tuple[KnownPair, ...]
    master_key_bits: tuple[WordBits, ...]
    round_keys: tuple[WordBits, ...]
    state_bits: tuple[tuple[StateBits, ...], ...]
    rounds: int

    @classmethod
    def from_known_pairs(
        cls,
        known_pairs: Sequence[KnownPair],
        rounds: int,
    ) -> SpeckSatModel:
        """Build a key-schedule-aware exact model from known pairs."""

        round_count = _validate_rounds(rounds)
        normalised_pairs = tuple(
            (
                _normalise_block(plaintext),
                _normalise_block(ciphertext),
            )
            for plaintext, ciphertext in known_pairs
        )
        if not normalised_pairs:
            raise ValueError("at least one known pair is required")

        solver = _new_solver()
        pool = VariablePool()
        master_key_bits = tuple(pool.new_bits(16) for _ in range(4))

        l_words = [
            master_key_bits[2],
            master_key_bits[1],
            master_key_bits[0],
        ]
        round_keys = [master_key_bits[3]]
        for index in range(round_count - 1):
            rotated_l = rotate_bits_right(
                l_words[index],
                SPECK32_64_ALPHA,
            )
            added_l = add_modular_addition(
                solver,
                pool,
                round_keys[index],
                rotated_l,
            )
            next_l = add_xor_constant(solver, pool, added_l, index)
            next_key = add_xor_bits(
                solver,
                pool,
                rotate_bits_left(
                    round_keys[index],
                    SPECK32_64_BETA,
                ),
                next_l,
            )
            l_words.append(next_l)
            round_keys.append(next_key)

        all_state_bits: list[tuple[StateBits, ...]] = []
        for plaintext, ciphertext in normalised_pairs:
            initial_state: StateBits = (
                pool.new_bits(16),
                pool.new_bits(16),
            )
            add_fixed_word(solver, initial_state[0], plaintext[0])
            add_fixed_word(solver, initial_state[1], plaintext[1])

            states: list[StateBits] = [initial_state]
            state = initial_state
            for round_key in round_keys:
                state = _encode_round(solver, pool, state, round_key)
                states.append(state)

            add_fixed_word(solver, state[0], ciphertext[0])
            add_fixed_word(solver, state[1], ciphertext[1])
            all_state_bits.append(tuple(states))

        return cls(
            solver=solver,
            pool=pool,
            known_pairs=normalised_pairs,
            master_key_bits=master_key_bits,
            round_keys=tuple(round_keys),
            state_bits=tuple(all_state_bits),
            rounds=round_count,
        )

    @property
    def variable_count(self) -> int:
        """Return the number of SAT variables allocated by the model."""

        return self.pool.variable_count

    def solve(
        self,
        assumptions: Sequence[int] = (),
        *,
        time_limit: float | None = None,
    ) -> SatSolveResult:
        """Solve once, optionally under temporary assumptions."""

        if time_limit is not None and time_limit <= 0:
            raise ValueError("time_limit must be positive")
        started = perf_counter()
        assumption_list = list(assumptions)
        if time_limit is None:
            if assumption_list:
                satisfiable, solution = self.solver.solve(assumption_list)
            else:
                satisfiable, solution = self.solver.solve()
        else:
            satisfiable, solution = self.solver.solve(
                assumption_list,
                time_limit=time_limit,
            )
        elapsed = perf_counter() - started
        if satisfiable:
            status = "sat"
            timed_out: bool | None = False
        elif time_limit is not None and elapsed >= time_limit:
            # pycryptosat returns (False, None) for both a proven UNSAT
            # result and a solve call interrupted by its time limit. Keep
            # this outcome explicitly ambiguous rather than mislabelling it.
            status = "unsat_or_timeout"
            timed_out = None
        else:
            status = "unsat"
            timed_out = False
        return SatSolveResult(
            satisfiable=bool(satisfiable),
            solution=(
                tuple(solution)
                if solution is not None
                else None
            ),
            elapsed_seconds=elapsed,
            timed_out=timed_out,
            status=status,
        )

    def master_key_from_solution(
        self,
        solution: tuple[bool | None, ...],
    ) -> tuple[int, int, int, int]:
        """Decode the four master-key words from a SAT model."""

        decoded = tuple(
            decode_word(solution, bits)
            for bits in self.master_key_bits
        )
        return decoded[0], decoded[1], decoded[2], decoded[3]

    def state_words(
        self,
        solution: tuple[bool | None, ...],
        pair_index: int,
        round_index: int,
    ) -> Block:
        """Decode one pair's state, where round zero is the plaintext."""

        if not 0 <= pair_index < len(self.state_bits):
            raise IndexError("pair_index is outside the model")
        if not 0 <= round_index <= self.rounds:
            raise IndexError("round_index is outside the model")
        state = self.state_bits[pair_index][round_index]
        return decode_word(solution, state[0]), decode_word(
            solution,
            state[1],
        )

    def assumptions_for_master_key(
        self,
        master_key: Sequence[int],
    ) -> tuple[int, ...]:
        """Return literals fixing the model's master-key variables."""

        words = tuple(master_key)
        if len(words) != 4:
            raise ValueError("master_key must contain four words")
        assumptions: list[int] = []
        for word, bits in zip(words, self.master_key_bits, strict=True):
            value = mask_word(word)
            assumptions.extend(
                variable if (value >> index) & 1 else -variable
                for index, variable in enumerate(bits)
            )
        return tuple(assumptions)

    def assumptions_for_final_round_subkey(
        self,
        final_round_subkey: int,
    ) -> tuple[int, ...]:
        """Return literals fixing the generated final round-key word.

        The returned literals are intended as temporary assumptions for one
        ``solve`` call. They constrain ``round_keys[-1]``, not the master-key
        variables directly, so a candidate is tested through the exact key
        schedule already present in this model.
        """

        if isinstance(final_round_subkey, bool) or not isinstance(
            final_round_subkey,
            int,
        ):
            raise TypeError("final_round_subkey must be an integer")
        if not 0 <= final_round_subkey <= WORD_MASK:
            raise ValueError("final_round_subkey must be a 16-bit word")
        final_key_bits = self.round_keys[-1]
        return tuple(
            variable if (final_round_subkey >> index) & 1 else -variable
            for index, variable in enumerate(final_key_bits)
        )

    def solve_final_round_subkey(
        self,
        final_round_subkey: int,
        *,
        time_limit: float | None = None,
    ) -> SatSolveResult:
        """Solve under one temporary final-round subkey assumption."""

        return self.solve(
            self.assumptions_for_final_round_subkey(final_round_subkey),
            time_limit=time_limit,
        )

    def add_blocking_clause(
        self,
        solution: tuple[bool | None, ...],
        variables: Sequence[int] | None = None,
    ) -> None:
        """Block one assignment of selected variables for enumeration tests."""

        selected = tuple(
            variables
            if variables is not None
            else (
                variable
                for word in self.master_key_bits
                for variable in word
            )
        )
        clause = []
        for variable in selected:
            value = solution[variable]
            if value is None:
                raise ValueError(f"solver left variable {variable} undefined")
            clause.append(-variable if value else variable)
        self.solver.add_clause(clause)
