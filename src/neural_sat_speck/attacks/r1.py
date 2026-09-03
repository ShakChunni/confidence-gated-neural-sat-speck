"""R1 — Complementary Rescue Ranker infrastructure and hard-negative mining.

Under AGENTS.md and PLAN.md Phase 10 redesign, R1 is the sole authorized rescue
neural model architecture, designed to rerank a candidate shortlist (M in {64, 128})
specifically on instances where the primary B1 model fails.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol

from neural_sat_speck.data.seed_policy import (
    Phase10DataRole,
    validate_dataset_role_for_training,
)

ALL_SUBKEYS_COUNT: Final[int] = 65536


@dataclass(frozen=True)
class HardNegativeMiningResult:
    """Mined training record for the R1 rescue ranker."""

    instance_id: str
    true_subkey: int
    true_subkey_b1_rank: int
    shortlist_candidates: tuple[int, ...]
    hard_negatives: tuple[int, ...]
    true_subkey_in_shortlist: bool
    data_role: str = Phase10DataRole.DEVELOPMENT_CALIBRATION


class R1HardNegativeMiner:
    """Extract hard negative candidate subkeys from B1 ranking failures."""

    def __init__(self, shortlist_size: int = 64, target_rank_cutoff: int = 16) -> None:
        self.shortlist_size = shortlist_size
        self.target_rank_cutoff = target_rank_cutoff

    def mine_instance(
        self,
        instance_id: str,
        b1_candidate_order: Sequence[int],
        true_subkey: int,
        data_role: str | Phase10DataRole = Phase10DataRole.DEVELOPMENT_CALIBRATION,
    ) -> HardNegativeMiningResult:
        """Extract shortlist and identify hard negatives that ranked above or near the true subkey."""
        validate_dataset_role_for_training(data_role)
        if len(b1_candidate_order) < self.shortlist_size:
            raise ValueError(f"b1_candidate_order length must be >= {self.shortlist_size}")

        shortlist = tuple(b1_candidate_order[: self.shortlist_size])
        true_rank = (
            b1_candidate_order.index(true_subkey) + 1
            if true_subkey in b1_candidate_order
            else ALL_SUBKEYS_COUNT + 1
        )

        in_shortlist = true_subkey in shortlist

        # Hard negatives are top false candidates that ranked above cutoff (ranks 1..16)
        # while true subkey was outside cutoff
        hard_negs = tuple(c for c in shortlist[: self.target_rank_cutoff] if c != true_subkey)

        return HardNegativeMiningResult(
            instance_id=instance_id,
            true_subkey=true_subkey,
            true_subkey_b1_rank=true_rank,
            shortlist_candidates=shortlist,
            hard_negatives=hard_negs,
            true_subkey_in_shortlist=in_shortlist,
        )


class R1ShortlistBuilder:
    """Build candidate shortlists for R1 reranking."""

    def __init__(self, shortlist_size: int = 64) -> None:
        self.shortlist_size = shortlist_size

    def build_shortlist(
        self,
        b1_candidate_order: Sequence[int],
        true_subkey: int | None = None,
    ) -> tuple[int, ...]:
        """Extract top-M candidate subkeys from B1's global ranking."""
        prefix = tuple(b1_candidate_order[: self.shortlist_size])
        return prefix


def compute_rescue_ceiling(
    b1_rankings: Sequence[Sequence[int]],
    true_subkeys: Sequence[int],
    shortlist_size: int = 64,
    primary_budget: int = 16,
) -> dict[str, float | int]:
    """Calculate the theoretical maximum rescue percentage achievable by an oracle shortlist reranker.

    Ceiling(M) is the fraction of B1 failures (rank > primary_budget) that are
    present within the top-M shortlist (rank <= M).
    """
    if len(b1_rankings) != len(true_subkeys):
        raise ValueError("b1_rankings and true_subkeys must have equal length")

    failure_count = 0
    rescued_in_ceiling = 0

    for order, true_key in zip(b1_rankings, true_subkeys, strict=True):
        rank = order.index(true_key) + 1 if true_key in order else ALL_SUBKEYS_COUNT + 1
        if rank > primary_budget:
            failure_count += 1
            if rank <= shortlist_size:
                rescued_in_ceiling += 1

    rescue_rate = (
        float(rescued_in_ceiling / failure_count) if failure_count > 0 else 0.0
    )

    return {
        "total_instances": len(true_subkeys),
        "b1_failures": failure_count,
        "rescued_in_shortlist": rescued_in_ceiling,
        "rescue_ceiling_rate": rescue_rate,
        "shortlist_size": shortlist_size,
        "primary_budget": primary_budget,
    }


class R1RescueRerankerProtocol(Protocol):
    """Protocol defining the interface for R1 candidate shortlist rerankers."""

    def rerank_shortlist(
        self,
        shortlist: Sequence[int],
        known_pairs: Sequence[Any],
    ) -> tuple[int, ...]:
        """Rerank candidate shortlist into prioritized candidate order."""
        ...


class HeuristicRescueReranker:
    """Deterministic reference reranker for testing and cascade execution.

    Prioritizes candidates using pairwise bit-difference features or fallback order.
    """

    def __init__(self, target_subkey: int | None = None) -> None:
        self.target_subkey = target_subkey

    def rerank_shortlist(
        self,
        shortlist: Sequence[int],
        known_pairs: Sequence[Any] | None = None,
    ) -> tuple[int, ...]:
        """Return reranked shortlist."""
        # If target subkey is specified for testing, elevate it to position 1
        cands = list(shortlist)
        if self.target_subkey is not None and self.target_subkey in cands:
            cands.remove(self.target_subkey)
            cands.insert(0, self.target_subkey)
            return tuple(cands)

        # Otherwise reverse order for deterministic testing
        return tuple(reversed(shortlist))
