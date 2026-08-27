"""Attack-stage components that consume the verified cipher and model APIs."""

from .group_ranking import (
    PAIR_GROUP_SCORE_AGGREGATION,
    PairGroupSubkeyRanking,
    evaluate_pair_group_subkey_ranking,
    rank_pair_group_subkeys,
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
    "CandidateAttempt",
    "PairSubsetStability",
    "RecoveryResult",
    "SubkeyRanking",
    "VerificationResult",
    "evaluate_subkey_ranking",
    "independent_verification_pairs",
    "known_pairs_for_attack_instance",
    "rank_final_round_subkeys",
    "PAIR_GROUP_SCORE_AGGREGATION",
    "PairGroupSubkeyRanking",
    "evaluate_pair_group_subkey_ranking",
    "rank_pair_group_subkeys",
    "random_candidate_order",
    "run_candidate_recovery",
    "run_sat_only_recovery",
    "verify_master_key",
]
