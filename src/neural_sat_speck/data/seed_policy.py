"""Centralized seed governance and role validation policy.

This module defines the authoritative lifecycle rules for all random seeds in the
COMP6002 research project, preventing accidental data contamination, quarantine
breaches, and premature exposure of validation or blind confirmation data.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

# Historical seeds from completed phases (strictly open for retrospective diagnostics only)
HISTORICAL_SEEDS: Final[frozenset[int]] = frozenset(
    {
        1001,
        2001,
        3001,
        4001,
        5001,
        6001,
        51001,
        52001,
        53001,
        61001,
        62001,
        63001,
        64001,
        65001,
        72001,
        73001,
        81001,
        82001,
        83001,
        91001,
        92001,
        93001,
        96001,
    }
)

# Historical Phase 7 held-out evaluation seed (opened 2026-08-31; corrective rerun only)
HISTORICAL_PHASE7_SEED: Final[int] = 98001

# Quarantined seed (generated prematurely in Phase 10 on 2026-09-01; debug/smoke only)
QUARANTINED_PHASE10_SEED: Final[int] = 97001

# Retired seed (hardcoded in CLI default and README; burned and disqualified from blind role)
RETIRED_BLIND_SEED: Final[int] = 99001

# Harmless demonstration seed for CLI examples (has no research or experimental role)
CLI_DEMO_SEED: Final[int] = 999999

# All protected research seeds that must never be used as default CLI arguments
PROTECTED_RESEARCH_SEEDS: Final[frozenset[int]] = frozenset(
    HISTORICAL_SEEDS
    | {HISTORICAL_PHASE7_SEED, QUARANTINED_PHASE10_SEED, RETIRED_BLIND_SEED}
)


class SeedRole(StrEnum):
    """Permitted experimental roles for random seeds."""

    PROSPECTIVE_VALIDATION = "prospective_validation"
    BLIND_CONFIRMATION = "blind_confirmation"
    MODEL_SELECTION = "model_selection"
    HISTORICAL_DIAGNOSTIC = "historical_diagnostic"
    DEBUG_SMOKE = "debug_smoke"
    DEMO = "demo"


def validate_seed_for_role(seed: int, role: SeedRole | str) -> None:
    """Enforce strict quarantine and non-contamination rules for experimental seeds.

    Raises
    ------
    ValueError
        If the seed violates research policy for the specified role.
    """
    if isinstance(role, str):
        try:
            role = SeedRole(role)
        except ValueError as err:
            raise ValueError(f"Unknown seed role: {role}") from err

    if role == SeedRole.PROSPECTIVE_VALIDATION:
        if seed == QUARANTINED_PHASE10_SEED:
            raise ValueError(
                f"Seed {seed} is QUARANTINED and prohibited from prospective validation. "
                "Reason: Generated and partially executed prematurely on 2026-09-01. "
                "Permitted use: debug/smoke only."
            )
        if seed == RETIRED_BLIND_SEED:
            raise ValueError(
                f"Seed {seed} is RETIRED and prohibited from research roles. "
                "Reason: Leaked in CLI defaults and documentation."
            )
        if seed == HISTORICAL_PHASE7_SEED:
            raise ValueError(
                f"Seed {seed} is an opened historical evaluation seed and cannot serve as prospective validation."
            )
        if seed in HISTORICAL_SEEDS or seed in PROTECTED_RESEARCH_SEEDS:
            raise ValueError(
                f"Seed {seed} has already been used in historical training/validation and cannot serve as prospective validation."
            )

    elif role == SeedRole.BLIND_CONFIRMATION:
        if seed == RETIRED_BLIND_SEED:
            raise ValueError(
                f"Seed {seed} is RETIRED from the blind confirmation role. "
                "Reason: Published in user-facing tutorials and CLI defaults. "
                "The Phase 12 blind seed remains TBD until final protocol freeze."
            )
        if seed in PROTECTED_RESEARCH_SEEDS or seed == QUARANTINED_PHASE10_SEED:
            raise ValueError(
                f"Seed {seed} is a protected or quarantined research seed and cannot serve as blind confirmation."
            )

    elif role == SeedRole.MODEL_SELECTION:
        if seed == HISTORICAL_PHASE7_SEED:
            raise ValueError(
                f"Seed {seed} is the opened historical Phase 7 benchmark. "
                "It is strictly forbidden to tune, select, or optimize models, features, or thresholds on seed 98001."
            )
        if seed == QUARANTINED_PHASE10_SEED:
            raise ValueError(
                f"Seed {seed} is quarantined and cannot be used for model selection or tuning."
            )

    elif role == SeedRole.DEMO:
        if seed in PROTECTED_RESEARCH_SEEDS:
            raise ValueError(
                f"Seed {seed} is a protected research seed and cannot be used as a CLI demonstration default. "
                f"Use CLI_DEMO_SEED ({CLI_DEMO_SEED}) instead."
            )


def get_cli_default_seed() -> int:
    """Return the safe, non-research default seed for public CLI demonstrations."""
    validate_seed_for_role(CLI_DEMO_SEED, SeedRole.DEMO)
    return CLI_DEMO_SEED
