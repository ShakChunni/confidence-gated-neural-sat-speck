"""Unit and regression tests for solver isolation, method independence, and seed governance.

Guarantees:
1. Every compared method receives a distinct solver/model object identity.
2. Reordering method execution does not change results through shared solver state.
3. Candidate assumptions remain incremental only within one method.
4. Solver state cannot be passed from one method result into another.
5. Atomic/resume logic recreates method-local solver state correctly.
6. Seed-role validation rejects 97001 for prospective validation.
7. Seed-role validation rejects 99001 for blind confirmation.
8. CLI default contains no protected research seed.
9. Historical seed 98001 is rejected by any new model-selection runner.
"""

from __future__ import annotations

import pytest

from neural_sat_speck.attacks.recovery import (
    independent_verification_pairs,
    known_pairs_for_attack_instance,
    run_candidate_recovery,
    run_sat_only_recovery,
)
from neural_sat_speck.cli import get_cli_default_seed
from neural_sat_speck.data.attack import AttackInstanceConfig, generate_attack_instances
from neural_sat_speck.data.seed_policy import (
    CLI_DEMO_SEED,
    HISTORICAL_PHASE7_SEED,
    PHASE10_DEVELOPMENT_SEED,
    PHASE10_G2_DEVELOPMENT_SEED,
    PHASE10_G2_PROSPECTIVE_SEED,
    PHASE10_PROSPECTIVE_SEED,
    PROTECTED_RESEARCH_SEEDS,
    QUARANTINED_PHASE10_SEED,
    RETIRED_BLIND_SEED,
    SeedRole,
    validate_seed_for_role,
)
from neural_sat_speck.sat.isolation import (
    IsolatedSolverFactory,
    create_isolated_sat_model,
)


@pytest.fixture
def sample_attack_instance():
    """Generate a single deterministic attack instance for isolation testing."""
    config = AttackInstanceConfig(
        rounds=6,
        pair_count=4,
        instance_count=1,
        seed=12345,
        split_role="test",
    )
    instances = generate_attack_instances(config)
    return instances.instances[0]


def test_every_compared_method_receives_distinct_object_identity(sample_attack_instance):
    """Test 1: IsolatedSolverFactory issues distinct model and solver instances per method."""
    pairs = known_pairs_for_attack_instance(sample_attack_instance)[:4]
    factory = IsolatedSolverFactory.from_pairs(pairs, rounds=6, instance_id="test-iso-1")

    methods = ["sat_only", "random_sat", "b0_sat", "b1_sat", "gated_fallback_sat"]
    models = {m: factory.build_isolated_model(m) for m in methods}

    # Verify all models and underlying solvers have unique identities
    model_ids = [id(m) for m in models.values()]
    solver_ids = [id(m.solver) for m in models.values()]

    assert len(set(model_ids)) == len(methods), "Duplicate SpeckSatModel instance detected across methods"
    assert len(set(solver_ids)) == len(methods), "Duplicate pycryptosat.Solver instance detected across methods"

    # factory.verify_isolation should succeed
    factory.verify_isolation(models)

    # Calling with duplicate method should raise ValueError
    with pytest.raises(ValueError, match="already been issued"):
        factory.build_isolated_model("sat_only")


def test_reordering_method_execution_does_not_alter_results(sample_attack_instance):
    """Test 2: Method-order invariance under strict solver isolation."""
    pairs = known_pairs_for_attack_instance(sample_attack_instance)[:4]
    verif_pairs = independent_verification_pairs(
        sample_attack_instance.master_key,
        rounds=6,
        pair_count=2,
        seed=54321,
    )

    # Order A: SAT-only then Candidate Recovery
    factory_a = IsolatedSolverFactory.from_pairs(pairs, rounds=6)
    model_sat_a = factory_a.build_isolated_model("sat_only")
    res_sat_a = run_sat_only_recovery(model_sat_a, pairs, verif_pairs, solver_timeout=0.1)

    model_cand_a = factory_a.build_isolated_model("cand")
    res_cand_a = run_candidate_recovery(
        model_cand_a,
        [sample_attack_instance.final_round_subkey],
        pairs,
        verif_pairs,
        candidate_budget=1,
        solver_timeout=5.0,
    )

    # Order B: Candidate Recovery first (solves key), then SAT-only on fresh model
    factory_b = IsolatedSolverFactory.from_pairs(pairs, rounds=6)
    model_cand_b = factory_b.build_isolated_model("cand")
    res_cand_b = run_candidate_recovery(
        model_cand_b,
        [sample_attack_instance.final_round_subkey],
        pairs,
        verif_pairs,
        candidate_budget=1,
        solver_timeout=5.0,
    )

    model_sat_b = factory_b.build_isolated_model("sat_only")
    res_sat_b = run_sat_only_recovery(model_sat_b, pairs, verif_pairs, solver_timeout=0.1)

    # Both candidate runs must succeed identically
    assert res_cand_a.success is True
    assert res_cand_b.success is True
    assert res_cand_a.winning_candidate == res_cand_b.winning_candidate == sample_attack_instance.final_round_subkey

    # Under isolation, order B SAT-only must NOT be polluted by cand_b solve
    # Both SAT-only runs must report identical status (timeout / unsat at 0.1s)
    assert res_sat_a.status == res_sat_b.status
    assert res_sat_a.success == res_sat_b.success


def test_candidate_assumptions_remain_incremental_within_one_method(sample_attack_instance):
    """Test 3: Within one method, temporary assumptions do not contaminate next assumption."""
    pairs = known_pairs_for_attack_instance(sample_attack_instance)[:4]
    model = create_isolated_sat_model(pairs, rounds=6)

    # Assume a wrong key that is unsatisfiable/inconsistent
    wrong_candidate = (sample_attack_instance.final_round_subkey ^ 0x1234) & 0xFFFF
    res_wrong = model.solve_final_round_subkey(wrong_candidate, time_limit=1.0)
    assert not res_wrong.satisfiable, "Wrong candidate unexpectedly satisfied"

    # Next, assume the true key on the SAME model incrementally
    res_true = model.solve_final_round_subkey(sample_attack_instance.final_round_subkey, time_limit=5.0)
    assert res_true.satisfiable, "True candidate failed after wrong assumption backtrack"
    recovered = model.master_key_from_solution(res_true.solution)
    assert recovered == sample_attack_instance.master_key


def test_solver_state_cannot_be_passed_across_methods():
    """Test 4: IsolatedSolverFactory rejects aliased or shared models."""
    pairs = [((0x1234, 0x5678), (0x9ABC, 0xDEF0))]
    factory = IsolatedSolverFactory.from_pairs(pairs, rounds=6)

    m1 = factory.build_isolated_model("m1")
    # Passing m1 as both "m1" and "m2" must trigger error
    with pytest.raises(ValueError, match="Cross-method model reuse detected"):
        factory.verify_isolation({"m1": m1, "m2": m1})


def test_resume_logic_recreates_method_local_solver_state(sample_attack_instance):
    """Test 5: Re-instantiating a fresh model from pairs faithfully recreates state."""
    pairs = known_pairs_for_attack_instance(sample_attack_instance)[:4]
    verif_pairs = independent_verification_pairs(
        sample_attack_instance.master_key,
        rounds=6,
        pair_count=2,
        seed=999,
    )

    # First run
    m1 = create_isolated_sat_model(pairs, rounds=6)
    res1 = run_candidate_recovery(
        m1,
        [sample_attack_instance.final_round_subkey],
        pairs,
        verif_pairs,
        candidate_budget=1,
    )

    # Simulated resume: reconstruct fresh model
    m2 = create_isolated_sat_model(pairs, rounds=6)
    assert id(m1) != id(m2)
    assert id(m1.solver) != id(m2.solver)

    res2 = run_candidate_recovery(
        m2,
        [sample_attack_instance.final_round_subkey],
        pairs,
        verif_pairs,
        candidate_budget=1,
    )

    assert res1.success is True
    assert res2.success is True
    assert res1.recovered_master_key == res2.recovered_master_key == sample_attack_instance.master_key


def test_seed_role_validation_rejects_quarantined_seed_97001():
    """Test 6: Seed 97001 is rejected for prospective validation."""
    with pytest.raises(ValueError, match="(?i)quarantined"):
        validate_seed_for_role(QUARANTINED_PHASE10_SEED, SeedRole.PROSPECTIVE_VALIDATION)

    with pytest.raises(ValueError, match="(?i)quarantined"):
        validate_seed_for_role(QUARANTINED_PHASE10_SEED, SeedRole.BLIND_CONFIRMATION)


def test_seed_role_validation_rejects_retired_seed_99001():
    """Test 7: Seed 99001 is retired and rejected for blind confirmation."""
    with pytest.raises(ValueError, match="RETIRED"):
        validate_seed_for_role(RETIRED_BLIND_SEED, SeedRole.BLIND_CONFIRMATION)

    with pytest.raises(ValueError, match="RETIRED"):
        validate_seed_for_role(RETIRED_BLIND_SEED, SeedRole.PROSPECTIVE_VALIDATION)


def test_cli_default_contains_no_protected_research_seed():
    """Test 8: CLI default seed is a safe demo seed and not in protected research seeds."""
    default_seed = get_cli_default_seed()
    assert default_seed == CLI_DEMO_SEED == 999999
    assert default_seed not in PROTECTED_RESEARCH_SEEDS
    assert default_seed != HISTORICAL_PHASE7_SEED
    assert default_seed != QUARANTINED_PHASE10_SEED
    assert default_seed != RETIRED_BLIND_SEED


def test_historical_seed_98001_rejected_for_model_selection():
    """Test 9: Historical seed 98001 is strictly rejected for model selection."""
    with pytest.raises(ValueError, match="strictly forbidden"):
        validate_seed_for_role(HISTORICAL_PHASE7_SEED, SeedRole.MODEL_SELECTION)

    with pytest.raises(ValueError, match="opened historical evaluation seed"):
        validate_seed_for_role(HISTORICAL_PHASE7_SEED, SeedRole.PROSPECTIVE_VALIDATION)


def test_seed_role_validation_accepts_owner_seeds_for_their_roles():
    """Test 10: Roles accept their own designated seeds without self-blocking."""
    # Phase 10 development accepts 100001
    validate_seed_for_role(PHASE10_DEVELOPMENT_SEED, SeedRole.PHASE10_DEVELOPMENT)

    # Phase 10 prospective validation accepts 101001
    validate_seed_for_role(PHASE10_PROSPECTIVE_SEED, SeedRole.PROSPECTIVE_VALIDATION)

    # Phase 10-G2 development accepts 200001
    validate_seed_for_role(PHASE10_G2_DEVELOPMENT_SEED, SeedRole.PHASE10_G2_DEVELOPMENT)

    # Phase 10-G2 prospective validation accepts 201001
    validate_seed_for_role(PHASE10_G2_PROSPECTIVE_SEED, SeedRole.PHASE10_G2_PROSPECTIVE_VALIDATION)


def test_seed_role_validation_rejects_cross_role_contamination():
    """Test 11: Designated seeds are blocked from cross-role contamination."""
    # 100001 cannot serve as G2 development or prospective validation
    with pytest.raises(ValueError, match="prohibited"):
        validate_seed_for_role(PHASE10_DEVELOPMENT_SEED, SeedRole.PHASE10_G2_DEVELOPMENT)

    with pytest.raises(ValueError, match="already been used"):
        validate_seed_for_role(PHASE10_DEVELOPMENT_SEED, SeedRole.PROSPECTIVE_VALIDATION)

    # 200001 cannot serve as Phase 10 development
    with pytest.raises(ValueError, match="already been used"):
        validate_seed_for_role(PHASE10_G2_DEVELOPMENT_SEED, SeedRole.PHASE10_DEVELOPMENT)

    # 101001 cannot serve as G2 development
    with pytest.raises(ValueError, match="prohibited"):
        validate_seed_for_role(PHASE10_PROSPECTIVE_SEED, SeedRole.PHASE10_G2_DEVELOPMENT)
