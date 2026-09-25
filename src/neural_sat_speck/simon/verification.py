"""Fresh-pair master-key verification for SIMON32/64.

Verifies candidate 64-bit master keys against known attack observations
and fresh, independent plaintext/ciphertext pairs using the reference
scalar SIMON32/64 implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

from neural_sat_speck.simon.cipher import Block, _normalise_block, encrypt_block
from neural_sat_speck.simon.key_schedule import (
    _normalise_master_key,
    _validate_round_count,
)

KnownPair: TypeAlias = tuple[Block, Block]
MasterKey: TypeAlias = tuple[int, int, int, int]


@dataclass(frozen=True)
class SimonVerificationResult:
    """Outcome of independently checking a candidate SIMON32/64 master key."""

    known_pairs_verified: bool
    independent_pairs_verified: bool

    @property
    def verified(self) -> bool:
        """True only if candidate key satisfies all known and fresh pairs."""
        return self.known_pairs_verified and self.independent_pairs_verified

    def to_dict(self) -> dict[str, bool]:
        """Convert verification result to dictionary representation."""
        return {
            "known_pairs_verified": self.known_pairs_verified,
            "independent_pairs_verified": self.independent_pairs_verified,
            "verified": self.verified,
        }


def _materialise_pairs(pairs: Sequence[KnownPair], name: str) -> tuple[KnownPair, ...]:
    """Validate and normalise a sequence of (plaintext, ciphertext) pairs."""
    try:
        pair_seq = tuple(pairs)
    except TypeError as error:
        raise TypeError(f"{name} must be a sequence of pairs") from error

    normalised: list[KnownPair] = []
    for index, item in enumerate(pair_seq):
        try:
            pt, ct = item
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name}[{index}] must be a pair of blocks") from error
        normalised.append((_normalise_block(pt), _normalise_block(ct)))
    return tuple(normalised)


def verify_simon_master_key(
    master_key: Sequence[int],
    known_pairs: Sequence[KnownPair],
    verification_pairs: Sequence[KnownPair],
    rounds: int,
) -> SimonVerificationResult:
    """Verify a candidate SIMON32/64 master key against known and fresh pairs.

    Args:
        master_key: Four 16-bit words in order (k3, k2, k1, k0).
        known_pairs: Sequence of ((L_pt, R_pt), (L_ct, R_ct)) observations from attack.
        verification_pairs: Sequence of fresh, independently generated ((pt), (ct)) pairs.
        rounds: Number of cipher rounds evaluated.

    Returns:
        SimonVerificationResult detailing whether known and independent pairs matched.
    """
    key = _normalise_master_key(master_key)
    valid_rounds = _validate_round_count(rounds)
    known = _materialise_pairs(known_pairs, "known_pairs")
    independent = _materialise_pairs(verification_pairs, "verification_pairs")

    def matches(pairs: Sequence[KnownPair]) -> bool:
        return all(
            encrypt_block(pt, key, valid_rounds) == ct
            for pt, ct in pairs
        )

    known_ok = matches(known) if known else False
    independent_ok = matches(independent) if independent else False

    return SimonVerificationResult(
        known_pairs_verified=known_ok,
        independent_pairs_verified=independent_ok,
    )
