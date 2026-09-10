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

# Safe development seed for Phase 10 (N=192 keys)
PHASE10_DEVELOPMENT_SEED: Final[int] = 100001

# Safe fresh development seed for Phase 10-G2 (N=512 keys)
PHASE10_G2_DEVELOPMENT_SEED: Final[int] = 200001

# Retired prospective validation seed for Phase 10-G2 (N=256 keys, permanently opened)
PHASE10_G2_PROSPECTIVE_SEED: Final[int] = 201001

# Authoritative Oracle-SAT development/evaluation seed for Phase 11A (N=64 keys, 7 rounds)
PHASE11A_ORACLE_SEED: Final[int] = 300001

# All protected research seeds that must never be used as default CLI arguments
PROTECTED_RESEARCH_SEEDS: Final[frozenset[int]] = frozenset(
    HISTORICAL_SEEDS
    | {
        HISTORICAL_PHASE7_SEED,
        QUARANTINED_PHASE10_SEED,
        RETIRED_BLIND_SEED,
        PHASE10_DEVELOPMENT_SEED,
        101001,
        PHASE10_G2_DEVELOPMENT_SEED,
        PHASE10_G2_PROSPECTIVE_SEED,
        PHASE11A_ORACLE_SEED,
    }
)


class Phase10DataRole(StrEnum):
    """Permitted dataset partition roles for Phase 10 research."""

    DEVELOPMENT_CALIBRATION = "development_calibration"
    LOCKED_VALIDATION = "locked_validation"


def validate_dataset_role_for_training(role: Phase10DataRole | str) -> None:
    """Enforce strict non-contamination of the locked prospective validation set.

    Raises
    ------
    ValueError
        If locked validation data is supplied to a training, fitting, or threshold selection pipeline.
    """
    if isinstance(role, str):
        try:
            role = Phase10DataRole(role)
        except ValueError as err:
            raise ValueError(f"Unknown Phase 10 data role: {role}") from err

    if role == Phase10DataRole.LOCKED_VALIDATION:
        raise ValueError(
            "Locked prospective validation data cannot enter training or calibration pipelines. "
            "Validation data must remain unopened until all hyperparameters, feature sets, "
            "R1 weights, and G1 gating thresholds are frozen."
        )


class SeedRole(StrEnum):
    """Permitted experimental roles for random seeds."""

    PHASE10_DEVELOPMENT = "phase10_development"
    PHASE10_G2_DEVELOPMENT = "phase10_g2_development"
    PROSPECTIVE_VALIDATION = "prospective_validation"
    PHASE10_G2_PROSPECTIVE_VALIDATION = "phase10_g2_prospective_validation"
    PHASE11A_ORACLE_EVALUATION = "phase11a_oracle_evaluation"
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

    if role == SeedRole.PHASE10_DEVELOPMENT:
        if seed in (HISTORICAL_PHASE7_SEED, QUARANTINED_PHASE10_SEED, RETIRED_BLIND_SEED):
            raise ValueError(
                f"Seed {seed} is prohibited from Phase 10 development data generation. "
                "Seeds 98001, 97001, and 99001 cannot be reused for fresh Phase 10 roles."
            )
        if seed in HISTORICAL_SEEDS or seed in PROTECTED_RESEARCH_SEEDS:
            raise ValueError(
                f"Seed {seed} has already been used in historical research and cannot serve as Phase 10 development data."
            )

    elif role == SeedRole.PHASE10_G2_DEVELOPMENT:
        if seed in PROTECTED_RESEARCH_SEEDS or seed in (HISTORICAL_PHASE7_SEED, QUARANTINED_PHASE10_SEED, RETIRED_BLIND_SEED, 100001, 101001):
            raise ValueError(
                f"Seed {seed} is prohibited from Phase 10-G2 development data generation. "
                "Historical, Phase 10 development (100001), and Phase 10 prospective validation (101001) seeds cannot be reused."
            )

    elif role == SeedRole.PHASE10_G2_PROSPECTIVE_VALIDATION:
        if seed in (HISTORICAL_PHASE7_SEED, QUARANTINED_PHASE10_SEED, RETIRED_BLIND_SEED, 100001, 101001, 200001) or (seed in PROTECTED_RESEARCH_SEEDS and seed != PHASE10_G2_PROSPECTIVE_SEED):
            raise ValueError(
                f"Seed {seed} is prohibited from Phase 10-G2 prospective validation. "
                "Cannot reuse past seeds or the G2 development seed (200001)."
            )

    elif role == SeedRole.PHASE11A_ORACLE_EVALUATION:
        prohibited_seeds = (
            HISTORICAL_PHASE7_SEED,    # 98001
            QUARANTINED_PHASE10_SEED,  # 97001
            RETIRED_BLIND_SEED,        # 99001
            PHASE10_DEVELOPMENT_SEED,  # 100001
            101001,                    # Phase 10 prospective
            PHASE10_G2_DEVELOPMENT_SEED,   # 200001
            PHASE10_G2_PROSPECTIVE_SEED,   # 201001
        )
        if seed in prohibited_seeds or seed in HISTORICAL_SEEDS or (seed in PROTECTED_RESEARCH_SEEDS and seed != PHASE11A_ORACLE_SEED):
            raise ValueError(
                f"Seed {seed} is prohibited from Phase 11A Oracle-SAT evaluation. "
                "Historical, quarantined, retired, Phase 10, and Phase 10-G2 seeds cannot be reused."
            )

    elif role == SeedRole.PROSPECTIVE_VALIDATION:
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
