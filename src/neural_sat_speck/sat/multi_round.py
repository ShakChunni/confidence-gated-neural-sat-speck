"""Temporary-assumption helpers for conditioning on several round keys.

The Phase 11A gate conditioned the exact model on the single final round key
``k6`` and every instance timed out. A post-gate residual-depth benchmark showed
that the exact backend completes an instance while

    rounds - (number of consecutive trailing round keys assumed) <= 5,

so each additional target round requires one further trailing round key to be
supplied as a candidate. These helpers build the assumption literal tuples from
the public round-key bit variables exposed by :class:`SpeckSatModel`; the frozen
solver model itself is unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping

from neural_sat_speck.sat.model import SpeckSatModel
from neural_sat_speck.speck.primitives import WORD_MASK


def assumptions_for_round_key(
    model: SpeckSatModel,
    round_index: int,
    value: int,
) -> tuple[int, ...]:
    """Return literals fixing one generated round key of ``model``.

    The round key is selected by position in the exact key schedule, so a
    candidate is tested through the same schedule the solver already encodes.
    """

    if not isinstance(model, SpeckSatModel):
        raise TypeError("model must be a SpeckSatModel")
    if isinstance(round_index, bool) or not isinstance(round_index, int):
        raise TypeError("round_index must be an integer")
    if not 0 <= round_index < len(model.round_keys):
        raise ValueError("round_index is outside the model's round keys")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("round key value must be an integer")
    if not 0 <= value <= WORD_MASK:
        raise ValueError("round key value must be a 16-bit word")
    return tuple(
        variable if (value >> bit) & 1 else -variable
        for bit, variable in enumerate(model.round_keys[round_index])
    )


def assumptions_for_round_keys(
    model: SpeckSatModel,
    round_key_values: Mapping[int, int],
) -> tuple[int, ...]:
    """Return one literal tuple fixing every ``{round_index: value}`` entry.

    Indices are emitted in ascending order so that identical candidate mappings
    always produce identical assumption tuples.
    """

    if not isinstance(round_key_values, Mapping):
        raise TypeError(
            "round_key_values must be a mapping of round index to round key value"
        )
    if not round_key_values:
        raise ValueError("round_key_values must contain at least one round key")
    assumptions: list[int] = []
    for round_index in sorted(round_key_values):
        assumptions.extend(
            assumptions_for_round_key(model, round_index, round_key_values[round_index])
        )
    return tuple(assumptions)


def trailing_round_key_indices(rounds: int, count: int) -> tuple[int, ...]:
    """Return the indices of the last ``count`` round keys of a ``rounds`` model."""

    if isinstance(rounds, bool) or not isinstance(rounds, int):
        raise TypeError("rounds must be an integer")
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be an integer")
    if not 0 <= count <= rounds:
        raise ValueError("count must be between zero and rounds")
    return tuple(range(rounds - count, rounds))
