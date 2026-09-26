"""Ciphertext-only SIMON candidate transforms for source-aligned ranking studies.

The zero-key inverse is essential for candidate-sensitive pair differences.
These functions do not score candidates or use attack ground truth.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import numpy as np

from .cipher import Block, decrypt_round


def _word(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer word")
    if not 0 <= value <= 0xFFFF:
        raise ValueError(f"{name} must be a 16-bit unsigned word")
    return int(value)


def _block(value: Sequence[int], name: str) -> Block:
    try:
        if len(value) != 2:
            raise ValueError(f"{name} must contain two words")
        return _word(value[0], name), _word(value[1], name)
    except TypeError as error:
        raise TypeError(f"{name} must contain two words") from error


def _rol16(values: np.ndarray, shift: int) -> np.ndarray:
    return ((values << shift) | (values >> (16 - shift))) & 0xFFFF


def _f16(values: np.ndarray) -> np.ndarray:
    return (_rol16(values, 1) & _rol16(values, 8)) ^ _rol16(values, 2)


def candidate_value_words(
    ciphertext_pairs: Sequence[tuple[Block, Block]],
    candidates: Sequence[int],
    *,
    final_key_candidate: int | None = None,
) -> np.ndarray:
    """Return post-candidate-inverse states, shape ``(key,pair,4)``.

    Each last axis is ``(left_a, right_a, left_b, right_b)``. At the
    conditional previous-key stage, ``final_key_candidate`` is first inverted
    from both ciphertexts. It is an attack-time candidate, never ground truth.
    This is the direct-pair full-value route for Bao's VV distinguishers.
    """
    if len(ciphertext_pairs) == 0:
        raise ValueError("at least one ciphertext pair is required")
    if len(candidates) == 0:
        raise ValueError("at least one candidate is required")
    if final_key_candidate is not None:
        final_key_candidate = _word(final_key_candidate, "final_key_candidate")

    keys = np.asarray([_word(key, "candidate") for key in candidates], dtype=np.uint16)
    pairs: list[tuple[Block, Block]] = []
    for index, pair in enumerate(ciphertext_pairs):
        try:
            pair_length = len(pair)
        except TypeError as error:
            raise TypeError(f"ciphertext pair {index} must contain two blocks") from error
        if pair_length != 2:
            raise ValueError(f"ciphertext pair {index} must contain two blocks")
        a = _block(pair[0], f"ciphertext pair {index} block a")
        b = _block(pair[1], f"ciphertext pair {index} block b")
        if final_key_candidate is not None:
            a = decrypt_round(a, final_key_candidate)
            b = decrypt_round(b, final_key_candidate)
        pairs.append((a, b))

    base = np.asarray(pairs, dtype=np.uint16)
    a_left, a_right = base[:, 0, 0], base[:, 0, 1]
    b_left, b_right = base[:, 1, 0], base[:, 1, 1]
    key_grid = keys[:, None]

    # D_k(L', R') = (R', L' xor f(R') xor k).
    one_right_a = (a_left ^ _f16(a_right))[None, :] ^ key_grid
    one_right_b = (b_left ^ _f16(b_right))[None, :] ^ key_grid
    one_left_a = np.broadcast_to(a_right[None, :], one_right_a.shape)
    one_left_b = np.broadcast_to(b_right[None, :], one_right_b.shape)
    return np.stack(
        (one_left_a, one_right_a, one_left_b, one_right_b),
        axis=2,
    ).astype(np.uint16, copy=False)


def candidate_transformed_words(
    ciphertext_pairs: Sequence[tuple[Block, Block]],
    candidates: Sequence[int],
    *,
    final_key_candidate: int | None = None,
) -> np.ndarray:
    """Return post-candidate, post-zero-inverse states, shape ``(key,pair,4)``.

    This route supplies Bao's value-difference layout and classical DDT lookup.
    At direct R=9 its pair difference has seven- and six-round depth by stage.
    """
    once = candidate_value_words(
        ciphertext_pairs, candidates, final_key_candidate=final_key_candidate
    )
    # D_0(L', R') = (R', L' xor f(R')).
    return np.stack(
        (
            once[:, :, 1],
            once[:, :, 0] ^ _f16(once[:, :, 1]),
            once[:, :, 3],
            once[:, :, 2] ^ _f16(once[:, :, 3]),
        ),
        axis=2,
    ).astype(np.uint16, copy=False)


def bao_full_value_bits(states: np.ndarray) -> np.ndarray:
    """Encode Bao's ``[c0a, c1a, c0b, c1b]`` as MSB-first 64 bits.

    Input is the output of :func:`candidate_value_words`; result shape is
    ``(key, pair, 4, 16)`` with unsigned byte entries in ``{0, 1}``.
    """
    if states.ndim != 3 or states.shape[2] != 4 or states.dtype != np.uint16:
        raise ValueError("states must be a uint16 (key, pair, 4) array")
    shifts = np.arange(15, -1, -1, dtype=np.uint16)
    return ((states[..., None] >> shifts) & 1).astype(np.uint8)


def bao_value_difference_bits(states: np.ndarray) -> np.ndarray:
    """Encode Bao's ``[c0a, c0b, c1a xor c1b]`` as MSB-first 48 bits.

    Input is the output of :func:`candidate_transformed_words`; result shape
    is ``(key, pair, 3, 16)`` with unsigned byte entries in ``{0, 1}``.
    """
    if states.ndim != 3 or states.shape[2] != 4 or states.dtype != np.uint16:
        raise ValueError("states must be a uint16 (key, pair, 4) array")
    words = np.stack(
        (states[:, :, 0], states[:, :, 2], states[:, :, 1] ^ states[:, :, 3]),
        axis=2,
    )
    shifts = np.arange(15, -1, -1, dtype=np.uint16)
    return ((words[..., None] >> shifts) & 1).astype(np.uint8)


def candidate_difference_words(states: np.ndarray) -> np.ndarray:
    """Return pair XOR differences after the extra zero-key inverse.

    Shape is ``(key, pair, 2)`` in ``(left, right)`` order, suitable for a
    separately audited full or explicitly labelled sampled distribution.
    """
    if states.ndim != 3 or states.shape[2] != 4 or states.dtype != np.uint16:
        raise ValueError("states must be a uint16 (key, pair, 4) array")
    return np.stack(
        (states[:, :, 0] ^ states[:, :, 2], states[:, :, 1] ^ states[:, :, 3]),
        axis=2,
    )
