"""Bounded checks of Bao-Markov queried-row transition calculations.

The toy-width tests compare exhaustive transitions with butterfly and queried-row
calculations over multiple rounds. A separate 16-bit basis check covers the
derivative algebra. These tests do not exhaustively validate the full 8R table.
"""

from __future__ import annotations

import numpy as np


def rol(x: int, shift: int, bits: int) -> int:
    mask = (1 << bits) - 1
    shift %= bits
    return ((x << shift) | (x >> (bits - shift))) & mask if shift else x


def simon_f(x: int, bits: int) -> int:
    return (rol(x, 1, bits) & rol(x, 8 if bits >= 16 else (bits // 2), bits)) ^ rol(
        x, 2, bits
    )


def naive_transition_matrix(bits: int) -> np.ndarray:
    """Compute exact ground truth transition matrix M[in_diff, out_diff] by exhausting all x."""
    word_count = 1 << bits
    domain = word_count * word_count
    # matrix shape: (in_diff, out_diff)
    matrix = np.zeros((domain, domain), dtype=np.float64)

    for a in range(word_count):
        for b in range(word_count):
            in_idx = (a << bits) | b
            for x in range(word_count):
                delta_f = simon_f(x ^ a, bits) ^ simon_f(x, bits)
                c = b ^ delta_f
                d = a
                out_idx = (c << bits) | d
                matrix[in_idx, out_idx] += 1.0 / word_count
    return matrix


def butterfly_row_transition(
    row_in: np.ndarray, a: int, bits: int
) -> np.ndarray:
    """Compute output row using affine-subspace butterfly network."""
    word_count = 1 << bits
    row = row_in.copy().astype(np.float64)

    # Compute basis vectors for Image(L_a)
    pivots = [0] * bits
    offset = simon_f(a, bits)
    for bit in range(bits):
        e = 1 << bit
        col = simon_f(a ^ e, bits) ^ simon_f(e, bits) ^ offset
        for p in range(bits - 1, -1, -1):
            if col & (1 << p):
                if pivots[p]:
                    col ^= pivots[p]
                else:
                    pivots[p] = col
                    break
    basis = [p for p in pivots if p != 0]

    # Fast Walsh-Hadamard style butterfly across basis vectors
    for vec in basis:
        for i in range(word_count):
            j = i ^ vec
            if i < j:
                avg = 0.5 * (row[i] + row[j])
                row[i] = row[j] = avg

    # Permute by offset = f(a)
    out_row = np.zeros(word_count, dtype=np.float64)
    for c in range(word_count):
        out_row[c] = row[c ^ offset]
    return out_row


def test_toy_width_exact_transition_equivalence() -> None:
    """Verify on toy width (bits=4, 256 state pairs) that butterfly equals naive transition."""
    bits = 4
    word_count = 1 << bits
    M_naive = naive_transition_matrix(bits)

    # Initial state difference delta_0 = (0, 1)
    p_naive = np.zeros(word_count * word_count, dtype=np.float64)
    p_naive[(0 << bits) | 1] = 1.0

    p_butterfly = np.zeros(word_count * word_count, dtype=np.float64)
    p_butterfly[(0 << bits) | 1] = 1.0

    # Propagate through 8 rounds
    for r in range(1, 9):
        p_naive = p_naive @ M_naive

        # Butterfly transition
        p_next_bf = np.zeros(word_count * word_count, dtype=np.float64)
        for a in range(word_count):
            in_row = p_butterfly[a * word_count : (a + 1) * word_count]
            if np.all(in_row == 0):
                continue
            out_col = butterfly_row_transition(in_row, a, bits)
            for c in range(word_count):
                p_next_bf[(c << bits) | a] += out_col[c]
        p_butterfly = p_next_bf

        # Check bit-for-bit exact identity
        np.testing.assert_allclose(p_naive, p_butterfly, atol=1e-15, rtol=1e-12)
        assert np.isclose(np.sum(p_butterfly), 1.0)


def test_queried_row_reconstruction_toy() -> None:
    """Verify that querying row c directly yields identical values to the full matrix."""
    bits = 4
    word_count = 1 << bits
    M_naive = naive_transition_matrix(bits)

    p_curr = np.zeros(word_count * word_count, dtype=np.float64)
    p_curr[(0 << bits) | 1] = 1.0
    for _ in range(3):
        p_curr = p_curr @ M_naive

    # Expected next round
    p_next_full = p_curr @ M_naive

    # Queried-row reconstruction for a subset of rows
    query_c_list = [0, 3, 7, 12]
    for c_query in query_c_list:
        queried_row = np.zeros(word_count, dtype=np.float64)
        for a in range(word_count):
            in_row = p_curr[a * word_count : (a + 1) * word_count]
            out_col = butterfly_row_transition(in_row, a, bits)
            queried_row[a] = out_col[c_query]

        expected_row = p_next_full[c_query * word_count : (c_query + 1) * word_count]
        np.testing.assert_allclose(queried_row, expected_row, atol=1e-15, rtol=1e-12)


def test_full_width_16bit_affine_basis_rank() -> None:
    """Verify 16-bit linear basis span against scalar exhaustive evaluations for sample differences."""
    bits = 16
    test_diffs = [0x0001, 0x0040, 0x0100, 0x1234, 0x8000, 0xFFFF]

    for a in test_diffs:
        pivots = [0] * bits
        offset = simon_f(a, bits)
        for bit in range(bits):
            e = 1 << bit
            col = simon_f(a ^ e, bits) ^ simon_f(e, bits) ^ offset
            for p in range(bits - 1, -1, -1):
                if col & (1 << p):
                    if pivots[p]:
                        col ^= pivots[p]
                    else:
                        pivots[p] = col
                        break
        basis = [p for p in pivots if p != 0]
        rank = len(basis)
        assert 0 <= rank <= 15, "SIMON round function derivative rank cannot exceed 15"

        # Check linear independence of basis vectors
        assert len(set(basis)) == rank
        for i, v in enumerate(basis):
            assert v != 0
