"""Attack-stage components that consume the verified cipher and model APIs."""

from .diversity_fallback import (
    PHASE6C_FALLBACK_BUDGET,
    PHASE6C_FALLBACK_HEAD_SIZE,
    PHASE6C_TRUSTED_BUDGET,
    choose_phase6c_route,
    compose_diversity_fallback,
)
from .group_ranking import (
    PAIR_GROUP_SCORE_AGGREGATION,
    PairGroupSubkeyRanking,
    evaluate_pair_group_subkey_ranking,
    rank_pair_group_subkeys,
)
from .hierarchical_ranking import (
    HIERARCHICAL_METHOD_ID,
    HIERARCHICAL_PAIR_COUNT,
    HIERARCHICAL_TIE_RULE,
    HIERARCHICAL_WINDOW_SIZE,
    HierarchicalSubkeyRanking,
    rank_hierarchical_subkeys,
)
from .ranking import (
    ALL_FINAL_ROUND_SUBKEYS,
    PairSubsetStability,
    SubkeyRanking,
    evaluate_subkey_ranking,
    rank_final_round_subkeys,
)
from .recovery import (
    CandidateAttempt,
    RecoveryResult,
    VerificationResult,
    independent_verification_pairs,
    known_pairs_for_attack_instance,
    random_candidate_order,
    run_candidate_recovery,
    run_sat_only_recovery,
    verify_master_key,
)

__all__ = [
    "ALL_FINAL_ROUND_SUBKEYS",
    "PHASE6C_FALLBACK_BUDGET",
    "PHASE6C_FALLBACK_HEAD_SIZE",
    "PHASE6C_TRUSTED_BUDGET",
    "CandidateAttempt",
    "HIERARCHICAL_METHOD_ID",
    "HIERARCHICAL_PAIR_COUNT",
    "HIERARCHICAL_TIE_RULE",
    "HIERARCHICAL_WINDOW_SIZE",
    "HierarchicalSubkeyRanking",
    "PairSubsetStability",
    "RecoveryResult",
    "SubkeyRanking",
    "VerificationResult",
    "evaluate_subkey_ranking",
    "choose_phase6c_route",
    "compose_diversity_fallback",
    "independent_verification_pairs",
    "known_pairs_for_attack_instance",
    "rank_final_round_subkeys",
    "rank_hierarchical_subkeys",
    "PAIR_GROUP_SCORE_AGGREGATION",
    "PairGroupSubkeyRanking",
    "evaluate_pair_group_subkey_ranking",
    "rank_pair_group_subkeys",
    "random_candidate_order",
    "run_candidate_recovery",
    "run_sat_only_recovery",
    "verify_master_key",
]
