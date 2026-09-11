"""Native-XOR DIMACS export and parser utilities for CryptoMiniSat CLI experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

from neural_sat_speck.sat.encoding import (
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
from neural_sat_speck.sat.model import (
    KnownPair,
    StateBits,
    _encode_round,
    _normalise_block,
    _validate_rounds,
)
from neural_sat_speck.speck.key_schedule import SPECK32_64_ALPHA, SPECK32_64_BETA

DimacsClause: TypeAlias = list[int]


@dataclass
class DimacsRecorder:
    """Capture CNF and XOR clauses in memory for export to DIMACS format."""

    clauses: list[DimacsClause] = field(default_factory=list)
    xor_clauses: list[tuple[list[int], bool]] = field(default_factory=list)

    def add_clause(self, clause: Sequence[int]) -> None:
        """Add a standard CNF disjunctive clause."""
        self.clauses.append(list(clause))

    def add_xor_clause(self, lits: Sequence[int], rhs: bool) -> None:
        """Add a native XOR clause: lits[0] ^ ... ^ lits[k-1] == rhs."""
        self.xor_clauses.append((list(lits), bool(rhs)))

    def to_dimacs(self, num_vars: int) -> str:
        """Render clauses into CryptoMiniSat native-XOR DIMACS format.

        CryptoMiniSat convention:
        - Header: p cnf <num_vars> <num_clauses + num_xor_clauses>
        - Standard CNF: lit1 lit2 ... 0
        - Native XOR: starts with 'x '. An unnegated XOR clause has odd parity (True).
          To represent parity False (rhs=False), exactly one literal is negated.
        """
        lines: list[str] = [
            f"p cnf {num_vars} {len(self.clauses) + len(self.xor_clauses)}"
        ]
        for clause in self.clauses:
            lines.append(" ".join(str(lit) for lit in clause) + " 0")

        for lits, rhs in self.xor_clauses:
            if not lits:
                continue
            if rhs:
                # Odd parity = True
                xor_str = "x " + " ".join(str(lit) for lit in lits) + " 0"
            else:
                # Even parity = False (negate the first literal)
                xor_str = (
                    f"x {-lits[0]} "
                    + " ".join(str(lit) for lit in lits[1:])
                    + " 0"
                )
            lines.append(xor_str)

        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class DimacsSpeckModel:
    """Exported SPECK SAT model with DIMACS text and decoding metadata."""

    dimacs_text: str
    num_variables: int
    master_key_bits: tuple[WordBits, ...]
    round_keys: tuple[WordBits, ...]
    rounds: int
    num_cnf_clauses: int
    num_xor_clauses: int


def export_speck_to_dimacs(
    known_pairs: Sequence[KnownPair],
    rounds: int,
    *,
    pinned_round_keys: dict[int, int] | None = None,
) -> DimacsSpeckModel:
    """Build an exact SPECK SAT model and export to native-XOR DIMACS format.

    Parameters
    ----------
    known_pairs : Sequence[KnownPair]
        Plaintext/ciphertext pairs constraining the instance.
    rounds : int
        Number of encryption rounds.
    pinned_round_keys : dict[int, int] | None
        Optional mapping of {round_index: subkey_value} to pin as unit clauses.
    """
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

    recorder = DimacsRecorder()
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
            recorder,
            pool,
            round_keys[index],
            rotated_l,
        )
        next_l = add_xor_constant(recorder, pool, added_l, index)
        next_key = add_xor_bits(
            recorder,
            pool,
            rotate_bits_left(
                round_keys[index],
                SPECK32_64_BETA,
            ),
            next_l,
        )
        l_words.append(next_l)
        round_keys.append(next_key)

    for plaintext, ciphertext in normalised_pairs:
        initial_state: StateBits = (
            pool.new_bits(16),
            pool.new_bits(16),
        )
        add_fixed_word(recorder, initial_state[0], plaintext[0])
        add_fixed_word(recorder, initial_state[1], plaintext[1])

        state = initial_state
        for round_key in round_keys:
            state = _encode_round(recorder, pool, state, round_key)

        add_fixed_word(recorder, state[0], ciphertext[0])
        add_fixed_word(recorder, state[1], ciphertext[1])

    if pinned_round_keys is not None:
        for round_idx, subkey_val in pinned_round_keys.items():
            if not 0 <= round_idx < round_count:
                raise IndexError(f"round_idx {round_idx} out of range (0..{round_count - 1})")
            add_fixed_word(recorder, round_keys[round_idx], subkey_val)

    num_vars = pool.variable_count
    dimacs_text = recorder.to_dimacs(num_vars)

    return DimacsSpeckModel(
        dimacs_text=dimacs_text,
        num_variables=num_vars,
        master_key_bits=master_key_bits,
        round_keys=tuple(round_keys),
        rounds=round_count,
        num_cnf_clauses=len(recorder.clauses),
        num_xor_clauses=len(recorder.xor_clauses),
    )


def decode_master_key_from_dimacs_solution(
    dimacs_output: str,
    master_key_bits: tuple[WordBits, ...],
) -> tuple[int, int, int, int] | None:
    """Parse satisfying variable assignments from CryptoMiniSat CLI output."""
    var_map: dict[int, bool] = {}
    is_sat = False

    for line in dimacs_output.splitlines():
        line = line.strip()
        if line.startswith("s SATISFIABLE"):
            is_sat = True
        elif line.startswith("v "):
            tokens = line[2:].split()
            for tok in tokens:
                try:
                    val = int(tok)
                except ValueError:
                    continue
                if val != 0:
                    var_map[abs(val)] = val > 0

    if not is_sat:
        return None

    # Construct solution tuple indexed by variable ID
    max_var = max(var_map.keys()) if var_map else 0
    solution: list[bool | None] = [None] * (max_var + 1)
    for v, val in var_map.items():
        solution[v] = val

    decoded = tuple(
        decode_word(tuple(solution), bits)
        for bits in master_key_bits
    )
    return decoded[0], decoded[1], decoded[2], decoded[3]
