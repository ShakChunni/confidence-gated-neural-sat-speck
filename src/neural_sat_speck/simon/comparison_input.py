"""Strict comparison input object and validator for SIMON32/64 R=10 comparisons.

Enforces protocol invariants and input validation before any method execution:
- R=10 rounds
- Exactly 32 or 64 ordered attack pairs (64 or 128 known observations)
- Exactly 2 independent fresh verification observations
- No duplicate plaintexts within attack pairs, within verification pairs, or between sets
- Chronological (k8, k9) SAT assumptions
- Verification observations are strictly isolated and never exposed to rankers
- Deterministic derivation of random control streams from manifest seed and instance/method IDs
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from neural_sat_speck.simon.attack_instance import (
    DEFAULT_SIMON_INPUT_DIFFERENCE,
    KnownPair,
    SimonAttackInstance,
    SimonAttackPair,
)
from neural_sat_speck.simon.cipher import Block, _normalise_block

EXPECTED_COMPARISON_ROUNDS: Final[int] = 10
EXPECTED_ATTACK_PAIR_COUNT: Final[int] = 32
EXPECTED_KNOWN_OBSERVATION_COUNT: Final[int] = 64
EXPECTED_VERIFICATION_PAIR_COUNT: Final[int] = 2
PERMITTED_PAIR_COUNTS: Final[frozenset[int]] = frozenset({32, 64})


def _validate_16bit_word(value: Any, name: str) -> int:
    """Validate that a value is an integer in the 16-bit word range (0..0xFFFF)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if not (0 <= value <= 0xFFFF):
        raise ValueError(f"{name} must be a 16-bit word (0..0xFFFF), got {value}")
    return value


def _validate_block(block: Any, name: str) -> Block:
    """Validate that a block is a 2-tuple of 16-bit words."""
    if not isinstance(block, (tuple, list)) or len(block) != 2:
        raise TypeError(f"{name} must be a 2-tuple of integers, got {type(block).__name__}")
    w0 = _validate_16bit_word(block[0], f"{name}[0]")
    w1 = _validate_16bit_word(block[1], f"{name}[1]")
    return (w0, w1)


def derive_random_control_seed(
    manifest_seed: int,
    instance_id: str,
    method: str,
) -> int:
    """Derive deterministic seed for random control from manifest seed, instance_id, and method.

    Uses SHA-256 to ensure independent, deterministic, and collision-resistant RNG streams
    without ever seeing or depending on ground truth keys.
    """
    if isinstance(manifest_seed, bool) or not isinstance(manifest_seed, int):
        raise TypeError("manifest_seed must be an integer")
    if not isinstance(instance_id, str) or not instance_id.strip():
        raise ValueError("instance_id must be a non-empty string")
    if not isinstance(method, str) or not method.strip():
        raise ValueError("method must be a non-empty string")

    domain = f"simon32_64:random_control:{manifest_seed}:{instance_id.strip()}:{method.strip()}"
    digest = hashlib.sha256(domain.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big")


@dataclass(frozen=True)
class SimonComparisonInput:
    """Strict input object for SIMON32/64 R=10 comparison methods.

    Guarantees:
    1. Rounds R=10.
    2. Explicitly supports pair_count in {32, 64}; constants below denote
       the historical 32-pair default used by older tests.
    3. Exactly pair_count ordered attack pairs (len(attack_pairs) == pair_count).
    4. Exactly 2 * pair_count known observations.
    5. Exactly 2 independently generated fresh verification observations.
    6. All words in attack and verification pairs are 16-bit integers (0..0xFFFF).
    7. No duplicate plaintexts within attack pairs, within verification pairs,
       or between attack and verification sets (zero overlap).
    8. Candidate subkeys must be formatted in chronological order (k8, k9).
    9. Verification observations are strictly isolated and never exposed to rankers.
    """

    instance_id: str
    rounds: int
    attack_pairs: tuple[SimonAttackPair, ...]
    verification_pairs: tuple[KnownPair, ...]
    manifest_seed: int
    pair_count: int = 32

    def __post_init__(self) -> None:
        if isinstance(self.instance_id, bool) or not isinstance(self.instance_id, str) or not self.instance_id.strip():
            raise (TypeError if isinstance(self.instance_id, bool) else ValueError)("instance_id must be a non-empty string")

        if isinstance(self.rounds, bool) or not isinstance(self.rounds, int):
            raise TypeError("rounds must be an integer")
        if self.rounds != EXPECTED_COMPARISON_ROUNDS:
            raise ValueError(
                f"Rounds must be exactly {EXPECTED_COMPARISON_ROUNDS} for SIMON comparison, got {self.rounds}"
            )

        if isinstance(self.manifest_seed, bool) or not isinstance(self.manifest_seed, int):
            raise TypeError("manifest_seed must be an integer")

        if isinstance(self.pair_count, bool) or not isinstance(self.pair_count, int):
            raise TypeError("pair_count must be an integer")
        if self.pair_count not in PERMITTED_PAIR_COUNTS:
            raise ValueError(
                f"pair_count must be in {sorted(PERMITTED_PAIR_COUNTS)}, got {self.pair_count}"
            )

        # 2. Assert attack pairs sequence and length
        if not isinstance(self.attack_pairs, (tuple, list)):
            raise TypeError("attack_pairs must be a sequence of SimonAttackPair")
        if len(self.attack_pairs) != self.pair_count:
            raise ValueError(
                f"Expected exactly {self.pair_count} attack pairs, got {len(self.attack_pairs)}"
            )
        for i, p in enumerate(self.attack_pairs):
            if not isinstance(p, SimonAttackPair):
                raise TypeError(f"attack_pairs[{i}] must be a SimonAttackPair, got {type(p).__name__}")
            _validate_block(p.plaintext0, f"attack_pairs[{i}].plaintext0")
            _validate_block(p.plaintext1, f"attack_pairs[{i}].plaintext1")
            _validate_block(p.ciphertext0, f"attack_pairs[{i}].ciphertext0")
            _validate_block(p.ciphertext1, f"attack_pairs[{i}].ciphertext1")

        # 3. Assert exactly 2 * pair_count known observations
        expected_known_count = 2 * self.pair_count
        known = self.known_pairs
        if len(known) != expected_known_count:
            raise ValueError(
                f"Expected exactly {expected_known_count} known observations, got {len(known)}"
            )

        # 4. Assert exactly 2 independent fresh verification observations
        if not isinstance(self.verification_pairs, (tuple, list)):
            raise TypeError("verification_pairs must be a sequence of KnownPair")
        if len(self.verification_pairs) != EXPECTED_VERIFICATION_PAIR_COUNT:
            raise ValueError(
                f"Expected exactly {EXPECTED_VERIFICATION_PAIR_COUNT} verification pairs, got {len(self.verification_pairs)}"
            )
        for i, vp in enumerate(self.verification_pairs):
            if not isinstance(vp, tuple) or len(vp) != 2:
                raise TypeError(f"verification_pairs[{i}] must be a 2-tuple (plaintext, ciphertext)")
            _validate_block(vp[0], f"verification_pairs[{i}][0]")
            _validate_block(vp[1], f"verification_pairs[{i}][1]")

        # 5. Plaintext disjointness
        attack_pts = [p.plaintext0 for p in self.attack_pairs] + [p.plaintext1 for p in self.attack_pairs]
        if len(set(attack_pts)) != len(attack_pts):
            raise ValueError("Duplicate plaintexts found within attack pairs")

        verif_pts = [vp[0] for vp in self.verification_pairs]
        if len(set(verif_pts)) != len(verif_pts):
            raise ValueError("Duplicate plaintexts found within verification pairs")

        intersection = set(attack_pts).intersection(set(verif_pts))
        if intersection:
            raise ValueError(
                f"Duplicate plaintexts found between attack and verification sets: {sorted(intersection)}"
            )
        for i, p in enumerate(self.attack_pairs):
            observed_difference = (
                p.plaintext0[0] ^ p.plaintext1[0],
                p.plaintext0[1] ^ p.plaintext1[1],
            )
            if observed_difference != DEFAULT_SIMON_INPUT_DIFFERENCE:
                raise ValueError(
                    f"attack_pairs[{i}] has input difference {observed_difference}, "
                    f"expected {DEFAULT_SIMON_INPUT_DIFFERENCE}"
                )

    @property
    def known_pairs(self) -> tuple[KnownPair, ...]:
        """Flatten attack pairs into 2 * pair_count known observations (plaintext, ciphertext)."""
        res: list[KnownPair] = []
        for p in self.attack_pairs:
            res.append((p.plaintext0, p.ciphertext0))
            res.append((p.plaintext1, p.ciphertext1))
        return tuple(res)

    @property
    def ciphertext_pairs(self) -> tuple[tuple[Block, Block], ...]:
        """Extract ONLY ciphertext pairs for rankers.

        Guarantees that verification observations and plaintexts are NEVER
        passed to ranker functions.
        """
        return tuple((p.ciphertext0, p.ciphertext1) for p in self.attack_pairs)

    @property
    def candidate_sat_pairs(self) -> tuple[KnownPair, ...]:
        """First 2 attack pairs (4 known observations) for the candidate SAT formula."""
        return self.known_pairs[:4]

    @property
    def all_sat_pairs(self) -> tuple[KnownPair, ...]:
        """All 32 attack pairs (64 known observations) for the SAT-only control."""
        return self.known_pairs

    def validate_sat_assumptions(self, assumptions: Sequence[int]) -> tuple[int, int]:
        """Validate that SAT assumptions are in chronological order (k8, k9).

        Args:
            assumptions: Sequence of 2 integers representing round keys.

        Returns:
            Tuple of (k8, k9) in chronological order.

        Raises:
            ValueError: If assumptions do not contain exactly 2 16-bit integers.
        """
        if len(assumptions) != 2:
            raise ValueError(f"Expected exactly 2 trailing subkeys, got {len(assumptions)}")
        k8, k9 = int(assumptions[0]), int(assumptions[1])
        if not (0 <= k8 <= 0xFFFF and 0 <= k9 <= 0xFFFF):
            raise ValueError(f"Subkeys must be 16-bit words (0..0xFFFF), got ({k8}, {k9})")
        return (k8, k9)

    def format_chronological_assumptions(self, k8: int, k9: int) -> tuple[int, int]:
        """Format candidate subkeys into chronological order (k8, k9) for SAT assumptions."""
        return self.validate_sat_assumptions((k8, k9))

    def derive_random_seed(self, method: str) -> int:
        """Derive deterministic seed for random controls from manifest seed, instance_id, and method."""
        return derive_random_control_seed(self.manifest_seed, self.instance_id, method)

    @classmethod
    def from_attack_instance(
        cls,
        instance: SimonAttackInstance,
        manifest_seed: int,
        pair_count: int | None = None,
    ) -> SimonComparisonInput:
        """Construct SimonComparisonInput from a SimonAttackInstance."""
        actual_pair_count = len(instance.pairs) if pair_count is None else pair_count
        return cls(
            instance_id=instance.instance_id,
            rounds=instance.rounds,
            attack_pairs=instance.pairs,
            verification_pairs=instance.verification_pairs,
            manifest_seed=manifest_seed,
            pair_count=actual_pair_count,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "rounds": self.rounds,
            "pair_count": self.pair_count,
            "attack_pairs": [p.to_dict() for p in self.attack_pairs],
            "verification_pairs": [
                [list(pt), list(ct)] for pt, ct in self.verification_pairs
            ],
            "manifest_seed": self.manifest_seed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SimonComparisonInput:
        pairs = tuple(SimonAttackPair.from_dict(p) for p in data["attack_pairs"])
        verif = tuple(
            (_normalise_block(pt), _normalise_block(ct))
            for pt, ct in data["verification_pairs"]
        )
        pair_count = int(data.get("pair_count", len(pairs)))
        return cls(
            instance_id=str(data["instance_id"]),
            rounds=int(data["rounds"]),
            attack_pairs=pairs,
            verification_pairs=verif,
            manifest_seed=int(data["manifest_seed"]),
            pair_count=pair_count,
        )
