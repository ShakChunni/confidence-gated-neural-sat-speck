"""XOR-to-CNF translation utilities with proven semantic equivalence.

Supports both direct truth-table clause expansion (for small XORs, k <= 5)
and linear Tseitin decomposition (for arbitrary k > 5) into chains of
3-variable XORs using fresh auxiliary variables from VariablePool.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neural_sat_speck.sat.encoding import VariablePool


def xor_to_cnf_direct(lits: Sequence[int], rhs: bool) -> list[list[int]]:
    """Convert an XOR constraint to CNF using direct truth-table expansion.

    Constraint semantics:
        (lits[0] ^ lits[1] ^ ... ^ lits[k-1]) == rhs

    An assignment (v_1, ..., v_k) violates the constraint iff
    parity(v_1, ..., v_k) != rhs. Each violating assignment is ruled
    out by exactly one disjunctive clause:
        OR_{i=1}^k (-lits[i] if v_i == 1 else lits[i])

    Produces exactly 2^(k-1) clauses of length k.
    """
    k = len(lits)
    if k == 0:
        # 0 literals: LHS is 0 (False).
        # If rhs is False, 0 == 0 is trivially True (0 clauses).
        # If rhs is True, 0 == 1 is UNSAT (empty clause []).
        return [] if not rhs else [[]]

    target_parity = 1 if rhs else 0
    clauses: list[list[int]] = []

    # Iterate over all 2^k possible bit assignments
    for bits in itertools.product((0, 1), repeat=k):
        parity = sum(bits) % 2
        if parity != target_parity:
            # Violating assignment: construct the clause that is False under this assignment
            clause = [-lit if bit == 1 else lit for lit, bit in zip(lits, bits, strict=True)]
            clauses.append(clause)

    return clauses


def xor_to_cnf_tseitin(
    lits: Sequence[int],
    rhs: bool,
    pool: VariablePool,
) -> list[list[int]]:
    """Convert an XOR constraint to CNF using linear Tseitin decomposition.

    For k <= 3, delegates to direct expansion (at most 4 clauses).
    For k > 3, decomposes the k-ary XOR into a chain of 3-variable XORs:
        lits[0] ^ lits[1] == aux_1
        aux_1 ^ lits[2]   == aux_2
        ...
        aux_{k-3} ^ lits[k-2] ^ lits[k-1] == rhs

    Allocates (k - 3) fresh variables from pool and produces 4*(k - 2) clauses of length 3.
    """
    k = len(lits)
    if k <= 3:
        return xor_to_cnf_direct(lits, rhs)

    clauses: list[list[int]] = []
    current_lhs_lit = lits[0]
    next_lit = lits[1]

    # First link: current_lhs_lit ^ next_lit == aux_var
    # i.e., current_lhs_lit ^ next_lit ^ aux_var == False
    aux_var = pool.new()
    clauses.extend(xor_to_cnf_direct([current_lhs_lit, next_lit, aux_var], False))
    current_lhs_lit = aux_var

    # Middle links
    for i in range(2, k - 2):
        next_lit = lits[i]
        aux_var = pool.new()
        clauses.extend(xor_to_cnf_direct([current_lhs_lit, next_lit, aux_var], False))
        current_lhs_lit = aux_var

    # Final link: current_lhs_lit ^ lits[k-2] ^ lits[k-1] == rhs
    clauses.extend(xor_to_cnf_direct([current_lhs_lit, lits[k - 2], lits[k - 1]], rhs))

    return clauses


def convert_xor_clauses_to_cnf(
    xor_clauses: Sequence[tuple[Sequence[int], bool]],
    pool: VariablePool | None = None,
    *,
    max_direct_size: int = 5,
) -> tuple[list[list[int]], int]:
    """Convert a collection of native XOR clauses to standard CNF clauses.

    Parameters
    ----------
    xor_clauses:
        Sequence of (literals, rhs) tuples.
    pool:
        VariablePool for auxiliary variables. Required if any XOR clause has length > max_direct_size.
    max_direct_size:
        Maximum XOR size to expand directly (default: 5, yielding 2^4 = 16 clauses).

    Returns
    -------
    tuple[list[list[int]], int]:
        (expanded_cnf_clauses, aux_variables_allocated)
    """
    initial_var_count = pool.variable_count if pool is not None else 0
    all_cnf_clauses: list[list[int]] = []

    for lits, rhs in xor_clauses:
        if not lits:
            if rhs:
                all_cnf_clauses.append([])  # UNSAT empty clause
            continue

        if len(lits) <= max_direct_size or pool is None:
            all_cnf_clauses.extend(xor_to_cnf_direct(lits, rhs))
        else:
            all_cnf_clauses.extend(xor_to_cnf_tseitin(lits, rhs, pool))

    aux_allocated = (pool.variable_count - initial_var_count) if pool is not None else 0
    return all_cnf_clauses, aux_allocated
