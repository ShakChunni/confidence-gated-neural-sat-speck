"""Pure CNF DIMACS conversion for SPECK formulas without modifying frozen modules.

Parses CryptoMiniSat native-XOR DIMACS format and translates native XOR
constraints into standard CNF clauses using direct expansion and Tseitin decomposition.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from neural_sat_speck.sat.dimacs import (
    DimacsSpeckModel,
    export_speck_to_dimacs,
)
from neural_sat_speck.sat.encoding import VariablePool, WordBits
from neural_sat_speck.sat.model import KnownPair
from neural_sat_speck.sat.xor_to_cnf import convert_xor_clauses_to_cnf


def _parse_native_dimacs(
    dimacs_text: str, initial_num_vars: int
) -> tuple[list[list[int]], list[tuple[list[int], bool]]]:
    """Validate records before reserving fresh conversion variables.

    The header counts both CNF and native XOR records. Clauses may span lines;
    multiple zero-terminated clauses on one line are also accepted. In the
    CryptoMiniSat extension, XOR of the signed literals must have odd parity.
    """
    if type(initial_num_vars) is not int or initial_num_vars < 0:
        raise ValueError("initial_num_vars must be a nonnegative integer")
    header: tuple[int, int] | None = None
    cnf_clauses: list[list[int]] = []
    xor_clauses: list[tuple[list[int], bool]] = []
    kind: str | None = None
    literals: list[int] = []

    for line_number, raw_line in enumerate(dimacs_text.splitlines(), 1):
        tokens = raw_line.split()
        if not tokens or raw_line.lstrip().startswith("c"):
            continue
        if tokens[0] == "p":
            if header is not None or kind is not None:
                raise ValueError(f"duplicate or misplaced DIMACS header at line {line_number}")
            if len(tokens) != 4 or tokens[1] != "cnf":
                raise ValueError("expected header: p cnf <variables> <records>")
            if any(re.fullmatch(r"[+-]?[0-9]+", token) is None for token in tokens[2:]):
                raise ValueError("noninteger DIMACS header")
            try:
                num_vars, num_records = int(tokens[2]), int(tokens[3])
            except ValueError as error:
                raise ValueError("noninteger DIMACS header") from error
            if num_vars < 0 or num_records < 0:
                raise ValueError("negative DIMACS header count")
            if num_vars != initial_num_vars:
                raise ValueError("initial_num_vars does not match DIMACS header")
            header = (num_vars, num_records)
            continue
        if header is None:
            raise ValueError("DIMACS record before required header")

        for token in tokens:
            if token.startswith("x"):
                if kind is not None:
                    raise ValueError(f"unterminated record before XOR at line {line_number}")
                kind = "xor"
                token = token[1:]
                if not token:
                    continue
            elif kind is None:
                kind = "cnf"
            if re.fullmatch(r"[+-]?[0-9]+", token) is None:
                raise ValueError(f"noninteger DIMACS literal at line {line_number}")
            try:
                literal = int(token)
            except ValueError as error:
                raise ValueError(f"noninteger DIMACS literal at line {line_number}") from error
            if literal == 0:
                if kind == "cnf":
                    cnf_clauses.append(literals)
                else:
                    # Preserve the old normalization/order for valid exports.
                    # Empty XOR has odd RHS and is a contradiction, not a no-op.
                    if literals and literals[0] < 0:
                        xor_clauses.append(([abs(literals[0])] + literals[1:], False))
                    else:
                        xor_clauses.append((literals, True))
                kind, literals = None, []
            else:
                if abs(literal) > header[0]:
                    raise ValueError(f"DIMACS literal outside declared range at line {line_number}")
                literals.append(literal)

    if header is None:
        raise ValueError("missing DIMACS header")
    if kind is not None:
        raise ValueError("unterminated DIMACS record")
    if len(cnf_clauses) + len(xor_clauses) != header[1]:
        raise ValueError("DIMACS record count does not match header")
    return cnf_clauses, xor_clauses


def convert_native_xor_dimacs_to_pure_cnf(
    dimacs_text: str,
    initial_num_vars: int,
) -> tuple[str, int]:
    """Convert a native-XOR DIMACS string into standard pure-CNF DIMACS text.

    Parameters
    ----------
    dimacs_text : str
        DIMACS text containing standard clauses and 'x ... 0' native XOR clauses.
    initial_num_vars : int
        Declared variable count; must match the header. Unused IDs are allowed.

    Returns
    -------
    tuple[str, int]
        (pure_cnf_text, total_variables_including_aux)
    """
    cnf_clauses, xor_clauses = _parse_native_dimacs(dimacs_text, initial_num_vars)
    pool = VariablePool(next_variable=initial_num_vars + 1)
    expanded_cnf, _ = convert_xor_clauses_to_cnf(xor_clauses, pool=pool, max_direct_size=5)
    total_clauses = cnf_clauses + expanded_cnf
    final_num_vars = pool.variable_count

    output_lines = [f"p cnf {final_num_vars} {len(total_clauses)}"]
    for clause in total_clauses:
        output_lines.append(" ".join(str(lit) for lit in clause) + " 0")

    return "\n".join(output_lines) + "\n", final_num_vars


def export_speck_to_pure_cnf(
    known_pairs: Sequence[KnownPair],
    rounds: int,
    *,
    pinned_round_keys: dict[int, int] | None = None,
) -> tuple[str, int, tuple[WordBits, ...]]:
    """Generate exact SPECK constraints and export as pure standard CNF DIMACS.

    Returns (cnf_text, total_variables, master_key_bits).
    """
    model: DimacsSpeckModel = export_speck_to_dimacs(
        known_pairs=known_pairs,
        rounds=rounds,
        pinned_round_keys=pinned_round_keys,
    )
    cnf_text, total_vars = convert_native_xor_dimacs_to_pure_cnf(
        model.dimacs_text,
        model.num_variables,
    )
    return cnf_text, total_vars, model.master_key_bits
