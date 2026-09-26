"""Deterministic, role-labelled attack instances for reduced-round SIMON32/64.

Generates differential-pair attack instances under strict seed governance,
supporting demo and development roles with fresh-pair verification data.
Ground-truth properties are strictly isolated for post-hoc evaluation only.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias

from neural_sat_speck.data.seed_policy import (
    CLI_DEMO_SEED,
    PROTECTED_RESEARCH_SEEDS,
)
from neural_sat_speck.simon.cipher import (
    Block,
    _normalise_block,
    encrypt_block,
)
from neural_sat_speck.simon.key_schedule import (
    _normalise_master_key,
    _validate_round_count,
    expand_key,
)
from neural_sat_speck.simon.primitives import mask_word

MasterKey: TypeAlias = tuple[int, int, int, int]
KnownPair: TypeAlias = tuple[Block, Block]

# Fixed demo-only input difference: one changed bit in the right word. It has
# not been validated or selected as a SIMON neural-attack difference.
DEFAULT_SIMON_INPUT_DIFFERENCE: Final[Block] = (0x0000, 0x0040)

PERMITTED_SIMON_ROLES: Final[frozenset[str]] = frozenset(
    {"demo", "development", "feasibility"}
)

# Additional frozen seeds from Phase 11 prospective campaign
FROZEN_EVALUATION_SEEDS: Final[frozenset[int]] = frozenset({669981})


@dataclass(frozen=True)
class SimonAttackPair:
    """One related plaintext/ciphertext pair under a shared master key."""

    plaintext0: Block
    plaintext1: Block
    ciphertext0: Block
    ciphertext1: Block

    def to_dict(self) -> dict[str, list[int]]:
        """Convert to JSON-serializable dictionary."""
        return {
            "plaintext0": list(self.plaintext0),
            "plaintext1": list(self.plaintext1),
            "ciphertext0": list(self.ciphertext0),
            "ciphertext1": list(self.ciphertext1),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Sequence[int]]) -> SimonAttackPair:
        """Construct instance from dictionary representation."""
        return cls(
            plaintext0=_normalise_block(data["plaintext0"]),
            plaintext1=_normalise_block(data["plaintext1"]),
            ciphertext0=_normalise_block(data["ciphertext0"]),
            ciphertext1=_normalise_block(data["ciphertext1"]),
        )


@dataclass(frozen=True)
class SimonAttackInstance:
    """One secret-key attack instance with differential pairs and verification data."""

    instance_id: str
    split_role: str
    master_key: MasterKey
    rounds: int
    pairs: tuple[SimonAttackPair, ...]
    verification_pairs: tuple[KnownPair, ...]

    def __post_init__(self) -> None:
        if self.split_role not in PERMITTED_SIMON_ROLES:
            raise ValueError(
                f"split_role {self.split_role!r} not in permitted SIMON roles: "
                f"{sorted(PERMITTED_SIMON_ROLES)}"
            )
        plaintexts = [pt for pair in self.pairs for pt in (pair.plaintext0, pair.plaintext1)]
        plaintexts.extend(pt for pt, _ in self.verification_pairs)
        if len(plaintexts) != len(set(plaintexts)):
            raise ValueError("Attack and verification plaintexts must be distinct")

    @property
    def known_pairs(self) -> tuple[KnownPair, ...]:
        """Flatten attack pairs into individual known (plaintext, ciphertext) pairs."""
        flattened: list[KnownPair] = []
        for pair in self.pairs:
            flattened.append((pair.plaintext0, pair.ciphertext0))
            flattened.append((pair.plaintext1, pair.ciphertext1))
        return tuple(flattened)

    @property
    def expected_final_round_subkey(self) -> int:
        """Ground-truth final round key. FOR POST-HOC EVALUATION ONLY."""
        keys = expand_key(self.master_key, self.rounds)
        return keys[-1]

    def expected_trailing_subkeys(self, count: int) -> tuple[int, ...]:
        """Ground-truth trailing round keys. FOR POST-HOC EVALUATION ONLY."""
        if not 1 <= count <= self.rounds:
            raise ValueError(f"count must be between 1 and {self.rounds}")
        keys = expand_key(self.master_key, self.rounds)
        return keys[-count:]

    def to_dict(self) -> dict[str, object]:
        """Serialize attack instance to dictionary."""
        return {
            "instance_id": self.instance_id,
            "split_role": self.split_role,
            "master_key": list(self.master_key),
            "rounds": self.rounds,
            "pairs": [p.to_dict() for p in self.pairs],
            "verification_pairs": [
                [list(pt), list(ct)] for pt, ct in self.verification_pairs
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> SimonAttackInstance:
        """Deserialize attack instance from dictionary."""
        raw_pairs = data["pairs"]
        pairs = tuple(SimonAttackPair.from_dict(p) for p in raw_pairs)  # type: ignore
        raw_verif = data.get("verification_pairs", [])
        verif = tuple(
            (_normalise_block(pt), _normalise_block(ct))
            for pt, ct in raw_verif  # type: ignore
        )
        return cls(
            instance_id=str(data["instance_id"]),
            split_role=str(data["split_role"]),
            master_key=_normalise_master_key(data["master_key"]),  # type: ignore
            rounds=_validate_round_count(int(data["rounds"])),  # type: ignore
            pairs=pairs,
            verification_pairs=verif,
        )


@dataclass(frozen=True)
class SimonAttackInstanceConfig:
    """Configuration for deterministic SIMON attack instance generation."""

    instance_count: int
    pair_count: int
    rounds: int
    seed: int
    split_role: str = "demo"
    verification_pair_count: int = 2
    input_difference: Block = DEFAULT_SIMON_INPUT_DIFFERENCE

    def __post_init__(self) -> None:
        if isinstance(self.instance_count, bool) or not isinstance(self.instance_count, int):
            raise TypeError("instance_count must be an integer")
        if self.instance_count <= 0:
            raise ValueError("instance_count must be positive")

        if isinstance(self.pair_count, bool) or not isinstance(self.pair_count, int):
            raise TypeError("pair_count must be an integer")
        if self.pair_count <= 0:
            raise ValueError("pair_count must be positive")

        if isinstance(self.verification_pair_count, bool) or not isinstance(
            self.verification_pair_count, int
        ):
            raise TypeError("verification_pair_count must be an integer")
        if self.verification_pair_count <= 0:
            raise ValueError("verification_pair_count must be positive")

        if _normalise_block(self.input_difference) == (0, 0):
            raise ValueError("input_difference must be nonzero for distinct pair plaintexts")
        if 2 * self.pair_count + self.verification_pair_count > 1 << 32:
            raise ValueError("Requested distinct plaintexts exceed the 32-bit block domain")

        _validate_round_count(self.rounds)

        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")

        if self.split_role not in PERMITTED_SIMON_ROLES:
            raise ValueError(
                f"split_role {self.split_role!r} not in permitted roles: "
                f"{sorted(PERMITTED_SIMON_ROLES)}"
            )

        if self.seed in PROTECTED_RESEARCH_SEEDS or self.seed in FROZEN_EVALUATION_SEEDS:
            raise ValueError(
                f"Seed {self.seed} is a protected/quarantined research seed "
                "and cannot be used for SIMON generation."
            )

        if self.split_role == "demo" and self.seed != CLI_DEMO_SEED:
            raise ValueError(
                f"Demo split role requires CLI_DEMO_SEED ({CLI_DEMO_SEED}), got {self.seed}"
            )
        if self.split_role == "development" and self.seed == CLI_DEMO_SEED:
            raise ValueError("Development split role cannot reuse the demo seed")


def generate_simon_attack_instances(
    config: SimonAttackInstanceConfig,
) -> tuple[SimonAttackInstance, ...]:
    """Deterministically generate SIMON32/64 attack instances under config.

    Each instance receives an independent random master key, the requested
    number of differential plaintext pairs under input_difference, and
    fresh verification pairs with plaintexts disjoint from attack pairs.
    """
    rng = random.Random(config.seed)
    diff = _normalise_block(config.input_difference)
    instances: list[SimonAttackInstance] = []

    for index in range(config.instance_count):
        master_key = (
            rng.randrange(1 << 16),
            rng.randrange(1 << 16),
            rng.randrange(1 << 16),
            rng.randrange(1 << 16),
        )

        pairs: list[SimonAttackPair] = []
        used_plaintexts: set[Block] = set()
        for _ in range(config.pair_count):
            while True:
                pt0 = (rng.randrange(1 << 16), rng.randrange(1 << 16))
                pt1 = (mask_word(pt0[0] ^ diff[0]), mask_word(pt0[1] ^ diff[1]))
                if pt0 not in used_plaintexts and pt1 not in used_plaintexts:
                    used_plaintexts.update((pt0, pt1))
                    break
            ct0 = encrypt_block(pt0, master_key, config.rounds)
            ct1 = encrypt_block(pt1, master_key, config.rounds)
            pairs.append(
                SimonAttackPair(
                    plaintext0=pt0,
                    plaintext1=pt1,
                    ciphertext0=ct0,
                    ciphertext1=ct1,
                )
            )

        verif_pairs: list[KnownPair] = []
        for _ in range(config.verification_pair_count):
            while True:
                vpt = (rng.randrange(1 << 16), rng.randrange(1 << 16))
                if vpt not in used_plaintexts:
                    used_plaintexts.add(vpt)
                    break
            vct = encrypt_block(vpt, master_key, config.rounds)
            verif_pairs.append((vpt, vct))

        instance_id = f"simon32_64_{config.split_role}_r{config.rounds}_{index:04d}"
        instances.append(
            SimonAttackInstance(
                instance_id=instance_id,
                split_role=config.split_role,
                master_key=master_key,
                rounds=config.rounds,
                pairs=tuple(pairs),
                verification_pairs=tuple(verif_pairs),
            )
        )

    return tuple(instances)


def save_simon_attack_instances(
    instances: Sequence[SimonAttackInstance],
    path: Path | str,
    *,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Save SIMON attack instances to a JSON file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": dict(metadata) if metadata is not None else {},
        "instance_count": len(instances),
        "instances": [inst.to_dict() for inst in instances],
    }
    with open(target, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_simon_attack_instances(path: Path | str) -> tuple[SimonAttackInstance, ...]:
    """Load SIMON attack instances from a JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return tuple(SimonAttackInstance.from_dict(item) for item in data["instances"])
