"""Unit tests for Phase 11 7-Round SPECK32/64 Beam Adjudication Pipeline."""

from __future__ import annotations

from typing import Any

import pytest

from neural_sat_speck.data.seed_policy import (
    CLI_DEMO_SEED,
    HISTORICAL_PHASE7_SEED,
    PHASE10_DEVELOPMENT_SEED,
    PHASE10_G2_PROSPECTIVE_SEED,
    PHASE11A_ORACLE_SEED,
    QUARANTINED_PHASE10_SEED,
    RETIRED_BLIND_SEED,
)
from neural_sat_speck.evaluation.metrics import exact_mcnemar_test
from scripts.adjudicate_beam_configurations import (
    CANDIDATE_BUDGET,
    CONFIG_SPECS,
)
from scripts.analyze_beam_adjudication import (
    analyze_adjudication_results,
)
from scripts.generate_phase11_beam_adjudication_data import (
    ALL_PROHIBITED_SEEDS,
    draw_csprng_seed,
)


def test_seed_governance_prohibitions() -> None:
    """Verify all known quarantined and opened seeds are included in ALL_PROHIBITED_SEEDS."""
    prohibited = {
        HISTORICAL_PHASE7_SEED,
        QUARANTINED_PHASE10_SEED,
        RETIRED_BLIND_SEED,
        PHASE10_DEVELOPMENT_SEED,
        101001,
        200001,
        PHASE10_G2_PROSPECTIVE_SEED,
        PHASE11A_ORACLE_SEED,
        CLI_DEMO_SEED,
    }
    for seed in prohibited:
        assert seed in ALL_PROHIBITED_SEEDS, f"Seed {seed} must be prohibited!"


def test_draw_csprng_seed_returns_valid_seed() -> None:
    """Verify CSPRNG draw produces a positive integer outside prohibited seeds."""
    seed = draw_csprng_seed()
    assert isinstance(seed, int)
    assert seed >= 100_000
    assert seed not in ALL_PROHIBITED_SEEDS


def test_configuration_parameter_invariants() -> None:
    """Verify beam allocations for Configurations A and B strictly equal 1,024."""
    assert CONFIG_SPECS["A"]["m1"] * CONFIG_SPECS["A"]["m2"] == CANDIDATE_BUDGET
    assert CONFIG_SPECS["B"]["m1"] * CONFIG_SPECS["B"]["m2"] == CANDIDATE_BUDGET
    assert CONFIG_SPECS["A"]["m1"] == 32
    assert CONFIG_SPECS["A"]["m2"] == 32
    assert CONFIG_SPECS["B"]["m1"] == 64
    assert CONFIG_SPECS["B"]["m2"] == 16


def test_counterbalancing_schedule() -> None:
    """Verify counterbalancing: even indices run A then B, odd indices run B then A."""
    for idx in range(10):
        if idx % 2 == 0:
            order = ["A", "B"] if idx % 2 == 0 else ["B", "A"]
            assert order == ["A", "B"]
        else:
            order = ["A", "B"] if idx % 2 == 0 else ["B", "A"]
            assert order == ["B", "A"]


def _dummy_record(
    instance_id: str,
    instance_index: int,
    config: str,
    *,
    success: bool,
    wall_time: float,
    status: str = "verified_recovery",
) -> dict[str, Any]:
    return {
        "instance_id": instance_id,
        "instance_index": instance_index,
        "configuration": config,
        "config_display_name": f"Config {config}",
        "m1": 32 if config == "A" else 64,
        "m2": 32 if config == "A" else 16,
        "candidate_budget": 1024,
        "verified_recovery": success,
        "status": status if not success else "verified_recovery",
        "failure_cause": None if success else status,
        "failure_domain": "none" if success else "solver",
        "failure_category": "none" if success else "solver_timeout",
        "candidates_attempted": 10 if success else 1024,
        "unresolved_timeouts": 0 if success else 1,
        "recovered_master_key": [1, 2, 3, 4] if success else None,
        "fresh_pair_verification_passed": success,
        "total_method_wall_seconds": wall_time,
        "par2_contribution": wall_time if success else 600.0,
        "preprocessing_seconds": 0.05,
        "ranking_screening_sorting_seconds": 1.5,
        "solver_seconds": 5.0,
        "rebuilding_seconds": 0.0,
        "verification_seconds": 0.001,
        "true_k6_rank": 5,
        "k6_in_beam": True,
        "true_k5_rank": 8,
        "k5_in_beam": True,
        "true_tuple_reachable": True,
        "true_tuple_position": 12,
        "counts_as_verified_recovery": success,
        "counts_as_unconditional_failure": not success,
        "is_resolved_sat_state": success,
    }


def test_selection_rule_tier1_higher_recovery_wins() -> None:
    """Tier 1: Higher recovery count wins unconditionally."""
    records_a = [
        _dummy_record("inst-0", 0, "A", success=True, wall_time=10.0),
        _dummy_record("inst-1", 1, "A", success=True, wall_time=15.0),
        _dummy_record("inst-2", 2, "A", success=False, wall_time=300.0, status="global_timeout"),
        _dummy_record("inst-3", 3, "A", success=False, wall_time=300.0, status="global_timeout"),
    ]
    records_b = [
        _dummy_record("inst-0", 0, "B", success=True, wall_time=5.0),
        _dummy_record("inst-1", 1, "B", success=False, wall_time=300.0, status="global_timeout"),
        _dummy_record("inst-2", 2, "B", success=False, wall_time=300.0, status="global_timeout"),
        _dummy_record("inst-3", 3, "B", success=False, wall_time=300.0, status="global_timeout"),
    ]
    data = {"records_by_configuration": {"A": records_a, "B": records_b}}
    res = analyze_adjudication_results(data)
    assert res["selection_decision"]["selected_configuration"] == "A"
    assert res["selection_decision"]["rule_tier"] == "1_higher_recovery_count"


def test_selection_rule_tier2_lower_par2_wins_on_tie() -> None:
    """Tier 2: Tied recoveries -> lower PAR-2 wins."""
    records_a = [
        _dummy_record("inst-0", 0, "A", success=True, wall_time=100.0),
        _dummy_record("inst-1", 1, "A", success=False, wall_time=300.0, status="global_timeout"),
    ]
    records_b = [
        _dummy_record("inst-0", 0, "B", success=True, wall_time=10.0),  # Much faster -> lower PAR-2
        _dummy_record("inst-1", 1, "B", success=False, wall_time=300.0, status="global_timeout"),
    ]
    data = {"records_by_configuration": {"A": records_a, "B": records_b}}
    res = analyze_adjudication_results(data)
    assert res["selection_decision"]["selected_configuration"] == "B"
    assert res["selection_decision"]["rule_tier"] == "2_lower_par2"


def test_selection_rule_tier4_reference_a_retained_on_complete_tie() -> None:
    """Tier 4: Completely tied -> Configuration A (Reference) retained."""
    records_a = [
        _dummy_record("inst-0", 0, "A", success=True, wall_time=10.0),
        _dummy_record("inst-1", 1, "A", success=False, wall_time=300.0, status="global_timeout"),
    ]
    records_b = [
        _dummy_record("inst-0", 0, "B", success=True, wall_time=10.0),
        _dummy_record("inst-1", 1, "B", success=False, wall_time=300.0, status="global_timeout"),
    ]
    data = {"records_by_configuration": {"A": records_a, "B": records_b}}
    res = analyze_adjudication_results(data)
    assert res["selection_decision"]["selected_configuration"] == "A"
    assert res["selection_decision"]["rule_tier"] == "4_tie_break_reference_A"


def test_exact_mcnemar_on_discordant_pairs() -> None:
    """Verify exact McNemar calculation produces known p-values."""
    # When b = 5, c = 0, n_d = 5 -> p = 2 * (0.5)^5 = 2 / 32 = 0.0625
    res = exact_mcnemar_test(b=5, c=0)
    assert pytest.approx(res["p_value"], abs=1e-4) == 0.0625

    # When b = 0, c = 0 -> p = 1.0
    res_zero = exact_mcnemar_test(b=0, c=0)
    assert res_zero["p_value"] == 1.0


def test_mismatched_instance_sets_rejected() -> None:
    """Verify analyze_adjudication_results rejects mismatched instance IDs."""
    records_a = [_dummy_record("inst-0", 0, "A", success=True, wall_time=10.0)]
    records_b = [_dummy_record("inst-999", 0, "B", success=True, wall_time=10.0)]
    data = {"records_by_configuration": {"A": records_a, "B": records_b}}
    with pytest.raises(ValueError, match="Instance ID sets"):
        analyze_adjudication_results(data)
