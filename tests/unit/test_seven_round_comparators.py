"""Unit tests for seven-round comparator development and evaluation readiness.

This module tests:
1. Classical distribution ranking integration within HierarchicalStage and
   run_hierarchical_multi_round_attack.
2. The truth-free invariant (corrupting master-key fields in AttackInstance
   does not alter search ordering or beam survivors).
3. Deterministic candidate ordering, tie-breaking, and deduplication.
4. Monotonic global wall-clock deadline enforcement across ranking and solving.
5. Random multi-round candidate generation and recovery control.
6. Multi-round SAT-only recovery control.
7. Strict per-method solver isolation across comparators.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from neural_sat_speck.attacks.classical_ranking import ClassicalDistributionTable
from neural_sat_speck.attacks.hierarchical_multi_round import (
    FAILURE_CAUSE_GLOBAL_TIMEOUT,
    STATUS_GLOBAL_TIMEOUT,
    HierarchicalStage,
    run_hierarchical_multi_round_attack,
)
from neural_sat_speck.attacks.multi_round_recovery import (
    STATUS_EXHAUSTED_WITHOUT_UNRESOLVED,
    STATUS_PROVEN_UNSAT,
    STATUS_SOLVER_TIMEOUT,
    STATUS_UNSUCCESSFUL_WITH_UNRESOLVED_TIMEOUTS,
    STATUS_VERIFICATION_FAILURE,
    STATUS_VERIFIED_RECOVERY,
    random_multi_round_candidates,
    run_multi_round_key_recovery,
    run_multi_round_sat_only_recovery,
    run_random_multi_round_recovery,
)
from neural_sat_speck.attacks.recovery import (
    independent_verification_pairs,
    known_pairs_for_attack_instance,
)
from neural_sat_speck.data.attack import (
    AttackInstance,
    AttackInstanceConfig,
    generate_attack_instances,
)
from neural_sat_speck.neural.group_model import PairGroupRanker
from neural_sat_speck.sat.isolation import IsolatedSolverFactory

ROUNDS = 4
SEED = 4001
KNOWN_PAIRS = 4
VERIFICATION_PAIRS = 2
VERIFICATION_SEED = 999999


class _FixedOrderStubRanker(PairGroupRanker):
    """Deterministic test double that orders candidates by a precomputed distance."""

    def __init__(self, pivot_features: np.ndarray) -> None:
        super().__init__(variant="b1_mean")
        self._pivot = torch.from_numpy(
            np.asarray(pivot_features, dtype=np.float32)
        ).reshape(1, -1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        rows = features.reshape(features.shape[0], -1)
        return -((rows - self._pivot) ** 2).sum(dim=1)


class _DummyGroupRanker(PairGroupRanker):
    """Deterministic stub returning arbitrary scores."""

    def __init__(self) -> None:
        super().__init__(variant="b1_mean")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features.mean(dim=(1, 2, 3))


def _build_instance(rounds: int = ROUNDS, pair_count: int = 64) -> AttackInstance:
    config = AttackInstanceConfig(
        instance_count=1,
        pair_count=pair_count,
        rounds=rounds,
        seed=SEED,
        split_role="train",
    )
    return generate_attack_instances(config).instances[0]


def _build_dummy_classical_table(rounds: int = 3) -> ClassicalDistributionTable:
    """Build a minimal valid ClassicalDistributionTable for testing."""
    return ClassicalDistributionTable(
        rounds=rounds,
        weights={0x00400000: 5.0, 0x00000000: 2.0},
        sample_count=1000,
        min_count=1,
        seed=999999,
        seed_role="demo",
    )


def test_stage_model_validation() -> None:
    """HierarchicalStage must accept ClassicalDistributionTable and reject invalid configurations."""
    dummy_table = _build_dummy_classical_table()
    stage = HierarchicalStage(round_index=2, model=dummy_table, beam_width=16)
    assert stage.model is dummy_table

    # Screened ranking is invalid for ClassicalDistributionTable
    with pytest.raises(TypeError, match="screened stage requires a PairGroupRanker model"):
        HierarchicalStage(round_index=2, model=dummy_table, beam_width=16, screened=True)

    # Unsupported model type must be rejected
    with pytest.raises(TypeError, match="must be a PairGroupRanker, NumpyInferenceModule, or ClassicalDistributionTable"):
        HierarchicalStage(round_index=2, model={"unsupported": True}, beam_width=16)


def test_classical_stage_conditional_scoring() -> None:
    """run_hierarchical_multi_round_attack supports a classical ranker in stage 1."""
    instance = _build_instance(rounds=3, pair_count=64)
    dummy_model = _DummyGroupRanker()
    dummy_table = _build_dummy_classical_table(rounds=2)

    stages = (
        HierarchicalStage(round_index=2, model=dummy_model, beam_width=2),
        HierarchicalStage(round_index=1, model=dummy_table, beam_width=2),
    )
    known = known_pairs_for_attack_instance(instance)[:KNOWN_PAIRS]
    verification = independent_verification_pairs(
        instance.master_key,
        rounds=3,
        pair_count=VERIFICATION_PAIRS,
        seed=VERIFICATION_SEED,
    )

    result = run_hierarchical_multi_round_attack(
        instance,
        rounds=3,
        stages=stages,
        known_pairs=known,
        verification_pairs=verification,
        solver_timeout=5.0,
    )

    assert len(result.stages) == 2
    assert result.stages[0].round_index == 2
    assert result.stages[1].round_index == 1
    assert result.stages[0].candidates_ranked == 65536
    assert result.stages[1].candidates_ranked == 65536 * 2  # 2 survivor prefixes from stage 0


def test_truth_free_invariant() -> None:
    """Corrupting master_key in AttackInstance must not alter ranking or candidate order."""
    instance = _build_instance(rounds=3, pair_count=64)
    dummy_model = _DummyGroupRanker()
    dummy_table = _build_dummy_classical_table(rounds=2)

    stages = (
        HierarchicalStage(round_index=2, model=dummy_model, beam_width=2),
        HierarchicalStage(round_index=1, model=dummy_table, beam_width=2),
    )
    known = known_pairs_for_attack_instance(instance)[:KNOWN_PAIRS]
    verification = independent_verification_pairs(
        instance.master_key,
        rounds=3,
        pair_count=VERIFICATION_PAIRS,
        seed=VERIFICATION_SEED,
    )

    res_clean = run_hierarchical_multi_round_attack(
        instance,
        rounds=3,
        stages=stages,
        known_pairs=known,
        verification_pairs=verification,
        solver_timeout=1.0,
    )

    # Corrupt ground truth in instance
    corrupted_instance = AttackInstance(
        instance_id=instance.instance_id,
        master_key=(0x0000, 0x0000, 0x0000, 0x0000),
        final_round_subkey=0x0000,
        pairs=instance.pairs,
    )

    res_corrupted = run_hierarchical_multi_round_attack(
        corrupted_instance,
        rounds=3,
        stages=stages,
        known_pairs=known,
        verification_pairs=verification,
        solver_timeout=1.0,
    )

    # Candidate paths and attempts must be identical
    assert res_clean.attempted_paths == res_corrupted.attempted_paths
    assert res_clean.candidates_attempted == res_corrupted.candidates_attempted
    for s_clean, s_corr in zip(res_clean.stages, res_corrupted.stages, strict=True):
        assert s_clean.top_candidate == s_corr.top_candidate
        assert s_clean.candidates_ranked == s_corr.candidates_ranked


def test_global_timeout_during_ranking() -> None:
    """An expired global wall-clock deadline during ranking halts with global_timeout."""
    instance = _build_instance(rounds=4, pair_count=64)
    dummy_model = _DummyGroupRanker()

    stages = (
        HierarchicalStage(round_index=3, model=dummy_model, beam_width=2),
        HierarchicalStage(round_index=2, model=dummy_model, beam_width=2),
    )
    known = known_pairs_for_attack_instance(instance)[:KNOWN_PAIRS]
    verification = independent_verification_pairs(
        instance.master_key,
        rounds=4,
        pair_count=VERIFICATION_PAIRS,
        seed=VERIFICATION_SEED,
    )

    # Extremely small global timeout so stage 1 or solver loop detects expiration
    result = run_hierarchical_multi_round_attack(
        instance,
        rounds=4,
        stages=stages,
        known_pairs=known,
        verification_pairs=verification,
        solver_timeout=5.0,
        global_timeout=1e-5,
    )

    assert result.success is False
    assert result.status == STATUS_GLOBAL_TIMEOUT
    assert result.failure_cause == FAILURE_CAUSE_GLOBAL_TIMEOUT


def test_global_timeout_in_multi_round_recovery() -> None:
    """run_multi_round_key_recovery halts with global_timeout when deadline expires."""
    instance = _build_instance(rounds=4, pair_count=8)
    known = known_pairs_for_attack_instance(instance)[:KNOWN_PAIRS]
    verification = independent_verification_pairs(
        instance.master_key,
        rounds=4,
        pair_count=VERIFICATION_PAIRS,
        seed=VERIFICATION_SEED,
    )
    factory = IsolatedSolverFactory.from_pairs(known, rounds=4, instance_id=instance.instance_id)
    model = factory.build_isolated_model("timeout_test")

    # Pass candidates that won't solve immediately
    candidates = [{2: 0x1234, 3: 0x5678}, {2: 0x9ABC, 3: 0xDEF0}]
    result = run_multi_round_key_recovery(
        model,
        candidates,
        known,
        verification,
        rounds=4,
        solver_timeout=10.0,
        global_timeout=1e-6,
    )

    assert result.success is False
    assert result.status == STATUS_GLOBAL_TIMEOUT


def test_random_multi_round_candidates() -> None:
    """random_multi_round_candidates generates deterministic, distinct candidate tuples."""
    cand1 = random_multi_round_candidates(count=64, round_indices=(6, 5), seed=123)
    cand2 = random_multi_round_candidates(count=64, round_indices=(6, 5), seed=123)
    cand3 = random_multi_round_candidates(count=64, round_indices=(6, 5), seed=456)

    assert len(cand1) == 64
    assert cand1 == cand2
    assert cand1 != cand3

    # Check distinctness
    tuples = [tuple(c[r] for r in (6, 5)) for c in cand1]
    assert len(set(tuples)) == 64

    # Validation errors
    with pytest.raises(ValueError):
        random_multi_round_candidates(count=0, round_indices=(6, 5), seed=1)
    with pytest.raises(TypeError):
        random_multi_round_candidates(count=10, round_indices="invalid", seed=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        random_multi_round_candidates(count=10, round_indices=(), seed=1)
    with pytest.raises(ValueError):
        random_multi_round_candidates(count=10, round_indices=(6, 6), seed=1)


def test_run_random_multi_round_recovery() -> None:
    """run_random_multi_round_recovery runs against an isolated model under a budget."""
    instance = _build_instance(rounds=3, pair_count=8)
    known = known_pairs_for_attack_instance(instance)[:KNOWN_PAIRS]
    verification = independent_verification_pairs(
        instance.master_key,
        rounds=3,
        pair_count=VERIFICATION_PAIRS,
        seed=VERIFICATION_SEED,
    )
    factory = IsolatedSolverFactory.from_pairs(known, rounds=3, instance_id=instance.instance_id)
    model = factory.build_isolated_model("random_test")

    result = run_random_multi_round_recovery(
        model,
        known,
        verification,
        rounds=3,
        round_indices=(2, 1),
        candidate_budget=5,
        seed=42,
        solver_timeout=1.0,
    )

    assert result.strategy == "random_multi_round_sat"
    assert result.candidates_attempted <= 5
    assert result.rounds == 3
    assert result.assumed_round_keys == 2


def test_run_multi_round_sat_only_recovery() -> None:
    """run_multi_round_sat_only_recovery solves without round-key assumptions."""
    instance = _build_instance(rounds=4, pair_count=8)
    known = known_pairs_for_attack_instance(instance)[:KNOWN_PAIRS]
    verification = independent_verification_pairs(
        instance.master_key,
        rounds=4,
        pair_count=VERIFICATION_PAIRS,
        seed=VERIFICATION_SEED,
    )
    factory = IsolatedSolverFactory.from_pairs(known, rounds=4, instance_id=instance.instance_id)
    model = factory.build_isolated_model("sat_only_test")

    result = run_multi_round_sat_only_recovery(
        model,
        known,
        verification,
        rounds=4,
        solver_timeout=10.0,
    )

    assert result.strategy == "sat_only"
    assert result.assumed_round_keys == 0
    assert result.residual_rounds == 4
    assert result.solver_calls == 1
    # On 4 rounds, exact SAT recovers the complete master key
    assert result.success is True
    assert result.status == STATUS_VERIFIED_RECOVERY
    assert result.recovered_master_key == tuple(instance.master_key)


def test_solver_isolation_across_comparators() -> None:
    """Each comparator must instantiate its own model via IsolatedSolverFactory."""
    instance = _build_instance(rounds=3, pair_count=8)
    known = known_pairs_for_attack_instance(instance)[:KNOWN_PAIRS]

    factory = IsolatedSolverFactory.from_pairs(known, rounds=3, instance_id=instance.instance_id)
    model1 = factory.build_isolated_model("comp1")
    model2 = factory.build_isolated_model("comp2")

    # Separate model instances and underlying solver instances
    assert model1 is not model2
    assert model1.solver is not model2.solver
    factory.verify_isolation({"comp1": model1, "comp2": model2})

    # Duplicate issuance must be rejected
    with pytest.raises(ValueError, match="already been issued a solver model"):
        factory.build_isolated_model("comp1")


def test_resume_preserves_unselected_methods_behavioral_main(tmp_path: Path) -> None:
    """Running main() with methods A and B, then resuming with method A, preserves method B records."""
    from scripts.evaluate_seven_round_comparators import main

    out_file = tmp_path / "behavioral_resume_run.json"

    # Step 1: Run with sat_only_4obs and random_sat for 1 instance
    ret1 = main([
        "--methods", "sat_only_4obs", "random_sat",
        "--instances", "1",
        "--global-timeout", "1.0",
        "--output", str(out_file),
    ])
    assert ret1 == 0

    with out_file.open("r", encoding="utf-8") as f:
        data1 = json.load(f)
    assert set(data1["records_by_method"].keys()) == {"sat_only_4obs", "random_sat"}
    assert len(data1["records_by_method"]["sat_only_4obs"]) == 1
    assert len(data1["records_by_method"]["random_sat"]) == 1

    # Step 2: Resume with only sat_only_4obs for 2 instances
    ret2 = main([
        "--methods", "sat_only_4obs",
        "--instances", "2",
        "--global-timeout", "1.0",
        "--resume",
        "--output", str(out_file),
    ])
    assert ret2 == 0

    with out_file.open("r", encoding="utf-8") as f:
        data2 = json.load(f)
    assert set(data2["records_by_method"].keys()) == {"sat_only_4obs", "random_sat"}
    assert len(data2["records_by_method"]["random_sat"]) == 1
    assert len(data2["records_by_method"]["sat_only_4obs"]) == 2
    assert "resume_events" in data2["metadata"]
    assert len(data2["metadata"]["resume_events"]) == 1
    assert data2["metadata"]["resume_events"][0]["resumed_methods"] == ["sat_only_4obs"]


def test_resume_without_output_fails_loudly() -> None:
    """Passing --resume without an explicit --output path must raise SystemExit."""
    from scripts.evaluate_seven_round_comparators import parse_args
    with pytest.raises(SystemExit):
        parse_args(["--resume"])


def test_method_alias_normalization() -> None:
    """Passing hierarchical_classical must be normalized to canonical hierarchical_neural_classical."""
    from scripts.evaluate_seven_round_comparators import parse_args
    args = parse_args(["--methods", "hierarchical_classical", "random_sat"])
    assert args.methods == ["hierarchical_neural_classical", "random_sat"]


def test_candidate_budget_defaults_to_m1_times_m2() -> None:
    """When --candidate-budget is None, effective budget equals m1 * m2."""
    from scripts.evaluate_seven_round_comparators import parse_args
    args = parse_args(["--m1", "64", "--m2", "32"])
    assert args.candidate_budget is None
    effective_budget = args.candidate_budget if args.candidate_budget is not None else (args.m1 * args.m2)
    assert effective_budget == 2048


def test_sat_only_respects_global_timeout() -> None:
    """run_multi_round_sat_only_recovery halts with STATUS_GLOBAL_TIMEOUT if global deadline is exceeded."""
    instance = _build_instance(rounds=4, pair_count=8)
    known = known_pairs_for_attack_instance(instance)[:KNOWN_PAIRS]
    verification = independent_verification_pairs(
        instance.master_key,
        rounds=4,
        pair_count=VERIFICATION_PAIRS,
        seed=VERIFICATION_SEED,
    )
    factory = IsolatedSolverFactory.from_pairs(known, rounds=4, instance_id=instance.instance_id)
    model = factory.build_isolated_model("sat_only_timeout_test")

    result = run_multi_round_sat_only_recovery(
        model,
        known,
        verification,
        rounds=4,
        solver_timeout=10.0,
        global_timeout=1e-6,
    )

    assert result.success is False
    assert result.status == STATUS_GLOBAL_TIMEOUT


def test_recovery_continues_after_single_candidate_timeout() -> None:
    """Candidate search must continue if a candidate times out but global deadline remains."""
    instance = _build_instance(rounds=4, pair_count=8)
    known = known_pairs_for_attack_instance(instance)[:KNOWN_PAIRS]
    verification = independent_verification_pairs(
        instance.master_key,
        rounds=4,
        pair_count=VERIFICATION_PAIRS,
        seed=VERIFICATION_SEED,
    )
    factory = IsolatedSolverFactory.from_pairs(known, rounds=4, instance_id=instance.instance_id)
    model = factory.build_isolated_model("continue_test")

    # True trailing subkey is schedule[-1]
    from neural_sat_speck.speck.key_schedule import expand_key
    schedule = expand_key(instance.master_key, rounds=4)
    true_k3 = schedule[-1]
    wrong_k3 = (true_k3 + 1) & 0xFFFF

    # Candidate 0 is wrong, Candidate 1 is true
    candidates = [{3: wrong_k3}, {3: true_k3}]

    # With generous global timeout, even if candidate 0 has solver_timeout, search reaches candidate 1
    result = run_multi_round_key_recovery(
        model,
        candidates,
        known,
        verification,
        rounds=4,
        solver_timeout=10.0,
        global_timeout=30.0,
    )

    assert result.success is True
    assert result.status == STATUS_VERIFIED_RECOVERY
    assert result.candidates_attempted == 2
    assert result.recovered_master_key == tuple(instance.master_key)


def test_solver_rebuilt_on_candidate_timeout() -> None:
    """When a candidate times out, the solver must be rebuilt via the factory and rebuild time charged."""
    instance = _build_instance(rounds=4, pair_count=8)
    known = known_pairs_for_attack_instance(instance)[:KNOWN_PAIRS]
    verification = independent_verification_pairs(
        instance.master_key,
        rounds=4,
        pair_count=VERIFICATION_PAIRS,
        seed=VERIFICATION_SEED,
    )
    factory = IsolatedSolverFactory.from_pairs(known, rounds=4, instance_id="spy_test")
    model = factory.build_isolated_model("spy_strategy")

    from neural_sat_speck.speck.key_schedule import expand_key
    schedule = expand_key(instance.master_key, rounds=4)
    true_k3 = schedule[-1]
    wrong_k3 = (true_k3 + 1) & 0xFFFF

    # Spy wrapping factory
    rebuild_calls: list[str] = []

    class _SpyFactory:
        def __init__(self, inner: IsolatedSolverFactory) -> None:
            self._inner = inner

        def rebuild_isolated_model(self, strategy: str):
            rebuild_calls.append(strategy)
            return self._inner.rebuild_isolated_model(strategy)

    spy_factory = _SpyFactory(factory)

    # Candidate 0 times out under 1e-6 timeout; candidate 1 is also a wrong candidate
    wrong_k3_b = (wrong_k3 + 1) & 0xFFFF
    candidates = [{3: wrong_k3}, {3: wrong_k3_b}]
    result = run_multi_round_key_recovery(
        model,
        candidates,
        known,
        verification,
        rounds=4,
        solver_timeout=1e-6,
        global_timeout=30.0,
        strategy="spy_strategy",
        solver_factory=spy_factory,
    )

    # Candidate 0 timed out, triggering solver rebuild via spy_factory
    assert "spy_strategy" in rebuild_calls
    assert len(rebuild_calls) >= 1
    assert result.status == STATUS_UNSUCCESSFUL_WITH_UNRESOLVED_TIMEOUTS
    assert result.solver_seconds > 0.0


def test_distinct_outcome_categories() -> None:
    """Outcome status distinctly differentiates proven UNSAT from unresolving timeouts."""
    instance = _build_instance(rounds=4, pair_count=8)
    known = known_pairs_for_attack_instance(instance)[:KNOWN_PAIRS]
    verification = independent_verification_pairs(
        instance.master_key,
        rounds=4,
        pair_count=VERIFICATION_PAIRS,
        seed=VERIFICATION_SEED,
    )
    factory = IsolatedSolverFactory.from_pairs(known, rounds=4, instance_id="outcome_test")

    from neural_sat_speck.speck.key_schedule import expand_key
    schedule = expand_key(instance.master_key, rounds=4)
    wrong_k3_a = (schedule[-1] + 1) & 0xFFFF
    wrong_k3_b = (schedule[-1] + 2) & 0xFFFF

    # 1. All candidates tested and proven UNSAT (no timeouts)
    model1 = factory.build_isolated_model("strategy_unsat")
    res1 = run_multi_round_key_recovery(
        model1,
        [{3: wrong_k3_a}, {3: wrong_k3_b}],
        known,
        verification,
        rounds=4,
        solver_timeout=10.0,
        global_timeout=30.0,
    )
    assert res1.success is False
    assert res1.status == STATUS_EXHAUSTED_WITHOUT_UNRESOLVED

    # 2. Candidate times out (unresolved state)
    model2 = factory.build_isolated_model("strategy_timeout")
    res2 = run_multi_round_key_recovery(
        model2,
        [{3: wrong_k3_a}],
        known,
        verification,
        rounds=4,
        solver_timeout=1e-6,
        global_timeout=30.0,
    )
    assert res2.success is False
    assert res2.status == STATUS_UNSUCCESSFUL_WITH_UNRESOLVED_TIMEOUTS

    # 3. SAT-only timeout
    model3 = factory.build_isolated_model("strategy_sat_only_timeout")
    res3 = run_multi_round_sat_only_recovery(
        model3,
        known,
        verification,
        rounds=4,
        solver_timeout=1e-6,
        global_timeout=30.0,
    )
    assert res3.success is False
    assert res3.status == STATUS_SOLVER_TIMEOUT

    # 4. Status constant definitions
    assert STATUS_PROVEN_UNSAT == "proven_unsat"
    assert STATUS_VERIFICATION_FAILURE == "verification_failure"


def test_resume_fail_closed_validation(tmp_path: Path) -> None:
    """Resume mode must fail closed on missing files, corrupt JSON, or metadata/record anomalies."""
    from scripts.evaluate_seven_round_comparators import (
        DEFAULT_CLASSICAL_PATH,
        DEFAULT_STAGE1_PATH,
        DEFAULT_STAGE2_PATH,
        SCHEMA_VERSION,
        _sha256_file,
        main,
        parse_args,
    )

    non_existent = tmp_path / "missing.json"
    corrupt_file = tmp_path / "corrupt.json"
    corrupt_file.write_text("{ incomplete json", encoding="utf-8")

    dataset_path = Path("datasets/phase11a_oracle_r7/oracle_evaluation.json")
    dataset_hash = _sha256_file(dataset_path)
    stage1_hash = _sha256_file(DEFAULT_STAGE1_PATH)
    stage2_hash = _sha256_file(DEFAULT_STAGE2_PATH)
    classical_hash = _sha256_file(DEFAULT_CLASSICAL_PATH)

    valid_metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "dataset_id": "speck32_64_attack_phase11a_oracle_r7",
        "stage1_model_path": str(DEFAULT_STAGE1_PATH),
        "stage1_model_sha256": stage1_hash,
        "stage2_model_path": str(DEFAULT_STAGE2_PATH),
        "stage2_model_sha256": stage2_hash,
        "classical_dist_path": str(DEFAULT_CLASSICAL_PATH),
        "classical_dist_sha256": classical_hash,
        "rounds": 7,
        "ranking_differential_pairs": 64,
        "ranking_total_observations": 128,
        "sat_observations_4obs": 4,
        "sat_observations_all": 128,
        "verification_pairs": 2,
        "verification_seed": 999999,
        "m1": 32,
        "m2": 32,
        "candidate_budget": 1024,
        "solver_timeout": 10.0,
        "global_timeout": 300.0,
        "device": "cpu",
        "methods": ["sat_only_4obs"],
        "prospective": False,
        "protocol_manifest_path": None,
    }

    # 1. Missing file fails closed
    with pytest.raises(SystemExit, match="does not exist"):
        parse_args(["--resume", "--output", str(non_existent)])
        main(["--resume", "--output", str(non_existent), "--instances", "1"])

    # 2. Corrupt JSON fails closed
    with pytest.raises(SystemExit, match="Failed to parse JSON"):
        main(["--resume", "--output", str(corrupt_file), "--instances", "1"])

    # 3. Non-object top-level fails closed
    non_obj_file = tmp_path / "non_obj.json"
    non_obj_file.write_text("[]", encoding="utf-8")
    with pytest.raises(SystemExit, match="must be a JSON object"):
        main(["--resume", "--output", str(non_obj_file), "--instances", "1"])

    # 4. Missing required metadata field fails closed
    for req_field in ["schema_version", "dataset_sha256", "sat_observations_4obs"]:
        bad_meta = dict(valid_metadata)
        bad_meta.pop(req_field)
        bad_file = tmp_path / f"missing_{req_field}.json"
        bad_file.write_text(
            json.dumps({"metadata": bad_meta, "records_by_method": {}}),
            encoding="utf-8",
        )
        with pytest.raises(SystemExit, match="metadata missing required fields"):
            main(["--resume", "--output", str(bad_file), "--instances", "1"])

    # 5. Schema version mismatch fails closed
    schema_mismatch_meta = dict(valid_metadata, schema_version="1.0.0")
    schema_mismatch_file = tmp_path / "schema_mismatch.json"
    schema_mismatch_file.write_text(
        json.dumps({"metadata": schema_mismatch_meta, "records_by_method": {}}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="schema_version mismatch"):
        main(["--resume", "--output", str(schema_mismatch_file), "--instances", "1"])

    # 6. Configuration parameter mismatch fails closed
    param_mismatch_meta = dict(valid_metadata, global_timeout=150.0)
    param_mismatch_file = tmp_path / "param_mismatch.json"
    param_mismatch_file.write_text(
        json.dumps({"metadata": param_mismatch_meta, "records_by_method": {}}),
        encoding="utf-8",
    )
    with pytest.raises(
        SystemExit, match="configuration mismatch for 'global_timeout'"
    ):
        main(["--resume", "--output", str(param_mismatch_file), "--instances", "1"])

    # 7. Unrecognized method in records_by_method fails closed
    unknown_method_file = tmp_path / "unknown_method.json"
    unknown_method_file.write_text(
        json.dumps(
            {
                "metadata": valid_metadata,
                "records_by_method": {"invalid_method_xyz": []},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="unrecognized method"):
        main(["--resume", "--output", str(unknown_method_file), "--instances", "1"])

    # 8. Incomplete record in records_by_method fails closed
    incomplete_rec = {
        "instance_id": "test-0000",
        "instance_index": 0,
        "method": "sat_only_4obs",
    }
    incomplete_file = tmp_path / "incomplete_record.json"
    incomplete_file.write_text(
        json.dumps(
            {
                "metadata": valid_metadata,
                "records_by_method": {"sat_only_4obs": [incomplete_rec]},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="incomplete record for instance"):
        main(["--resume", "--output", str(incomplete_file), "--instances", "1"])

    # 9. Instance ID not in dataset fails closed
    valid_record = {
        "instance_id": "train-0000",
        "instance_index": 0,
        "method": "sat_only_4obs",
        "method_display_name": "Exact SAT Only",
        "strategy": "sat_only_4obs",
        "success": False,
        "status": "solver_timeout",
        "failure_cause": "solver_timeout",
        "candidates_attempted": 0,
        "recovered_master_key": None,
        "timing_seconds": {
            "ranking": 0.0,
            "solver": 10.0,
            "verification": 0.0,
            "total": 10.0,
        },
        "wall_seconds": 10.0,
        "method_wall_seconds": 10.05,
        "cutoff_seconds": 300.0,
        "cutoff_overshoot_seconds": 0.0,
        "preprocessing_seconds": 0.05,
        "ranking_screening_sorting_seconds": 0.0,
        "solver_seconds": 10.0,
        "rebuilding_seconds": 0.0,
        "verification_seconds": 0.0,
        "counts_as_verified_recovery": False,
        "counts_as_unconditional_failure": True,
        "is_resolved_sat_state": False,
        "failure_domain": "solver",
        "late_verified_recovery": False,
    }
    unknown_inst_file = tmp_path / "unknown_inst.json"
    unknown_inst_file.write_text(
        json.dumps(
            {
                "metadata": valid_metadata,
                "records_by_method": {
                    "sat_only_4obs": [dict(valid_record, instance_id="unknown-9999")]
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="not found in dataset"):
        main(["--resume", "--output", str(unknown_inst_file), "--instances", "1"])

    # 10. Duplicate instance ID within method fails closed
    dup_file = tmp_path / "duplicate_record.json"
    dup_file.write_text(
        json.dumps(
            {
                "metadata": valid_metadata,
                "records_by_method": {"sat_only_4obs": [valid_record, valid_record]},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        SystemExit, match="duplicate record for instance 'train-0000'"
    ):
        main(["--resume", "--output", str(dup_file), "--instances", "1"])


def test_sat_only_observation_counts_and_formula_sizes() -> None:
    """sat_only_4obs uses 4 observations while sat_only_all_observations uses 128 observations."""
    from neural_sat_speck.data.attack import load_attack_shard
    from neural_sat_speck.sat.dimacs import export_speck_to_dimacs

    dataset_path = Path("datasets/phase11a_oracle_r7/oracle_evaluation.json")
    shard = load_attack_shard(
        dataset_path, expected_dataset_id="speck32_64_attack_phase11a_oracle_r7"
    )
    inst = shard.instances[0]

    obs_all = known_pairs_for_attack_instance(inst)
    obs_4 = obs_all[:4]

    assert len(obs_all) == 128
    assert len(obs_4) == 4

    m_4 = export_speck_to_dimacs(obs_4, rounds=7)
    m_all = export_speck_to_dimacs(obs_all, rounds=7)

    # 4 observations formula matches authoritative 7-round base counts
    assert m_4.num_variables == 2368
    assert m_4.num_cnf_clauses == 3350
    assert m_4.num_xor_clauses == 1632

    # 128 observations formula matches 64-pair information-parity model
    assert m_all.num_variables == 61888
    assert m_all.num_cnf_clauses == 90274
    assert m_all.num_xor_clauses == 43296


def test_model_construction_consumes_method_budget_and_clamps_solver_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Model construction and preprocessing must consume the method budget before solver execution."""
    import scripts.evaluate_seven_round_comparators as runner_module
    from neural_sat_speck.attacks.multi_round_recovery import (
        STATUS_SOLVER_TIMEOUT,
        MultiRoundRecoveryResult,
    )

    observed_solver_timeouts: list[float] = []
    observed_global_timeouts: list[float] = []

    def mock_sat_only_recovery(
        model,
        known_pairs,
        verification_pairs,
        *,
        rounds,
        solver_timeout,
        global_timeout,
        strategy,
    ):
        observed_solver_timeouts.append(solver_timeout)
        observed_global_timeouts.append(global_timeout)
        return MultiRoundRecoveryResult(
            strategy=strategy,
            status=STATUS_SOLVER_TIMEOUT,
            success=False,
            rounds=rounds,
            assumed_round_keys=0,
            residual_rounds=rounds,
            candidates_attempted=0,
            solver_calls=1,
            attempts=(),
            recovered_master_key=None,
            solver_seconds=0.01,
            verification_seconds=0.0,
            total_seconds=0.01,
        )

    monkeypatch.setattr(
        runner_module, "run_multi_round_sat_only_recovery", mock_sat_only_recovery
    )

    out_file = tmp_path / "test_budget.json"
    ret = runner_module.main(
        [
            "--methods",
            "sat_only_4obs",
            "--instances",
            "1",
            "--global-timeout",
            "10.0",
            "--output",
            str(out_file),
        ]
    )
    assert ret == 0

    assert len(observed_solver_timeouts) == 1
    # Solver timeout must be strictly less than 10.0 because factory construction and model building consumed time
    assert observed_solver_timeouts[0] < 10.0
    assert observed_global_timeouts[0] < 10.0
    assert observed_solver_timeouts[0] == observed_global_timeouts[0]

    with out_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rec = data["records_by_method"]["sat_only_4obs"][0]
    assert rec["preprocessing_seconds"] > 0.0
    assert rec["method_wall_seconds"] >= rec["preprocessing_seconds"]
    assert rec["cutoff_seconds"] == 10.0
    assert rec["cutoff_overshoot_seconds"] >= 0.0


def test_method_budget_isolation_and_order_invariance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One method cannot consume another method's budget, and reversing method order preserves per-method allocation."""
    import scripts.evaluate_seven_round_comparators as runner_module

    # Order 1: sat_only_4obs then random_sat
    out1 = tmp_path / "order1.json"
    ret1 = runner_module.main(
        [
            "--methods",
            "sat_only_4obs",
            "random_sat",
            "--instances",
            "1",
            "--global-timeout",
            "2.0",
            "--timeout",
            "1.0",
            "--output",
            str(out1),
        ]
    )
    assert ret1 == 0
    with out1.open("r", encoding="utf-8") as f:
        data1 = json.load(f)

    # Order 2: random_sat then sat_only_4obs
    out2 = tmp_path / "order2.json"
    ret2 = runner_module.main(
        [
            "--methods",
            "random_sat",
            "sat_only_4obs",
            "--instances",
            "1",
            "--global-timeout",
            "2.0",
            "--timeout",
            "1.0",
            "--output",
            str(out2),
        ]
    )
    assert ret2 == 0
    with out2.open("r", encoding="utf-8") as f:
        data2 = json.load(f)

    # Both runs allocate independent 2.0s cutoff to every method regardless of position
    for data in (data1, data2):
        for method in ("sat_only_4obs", "random_sat"):
            rec = data["records_by_method"][method][0]
            assert rec["cutoff_seconds"] == 2.0
            assert rec["method_wall_seconds"] > 0.0
            # Ensure each method had its own fresh budget and did not carry over time from the other
            assert rec["cutoff_overshoot_seconds"] <= 1.0


def test_solver_timeout_leaves_instance_unverified_with_cause() -> None:
    """A solver timeout must leave the instance unverified with failure cause explicitly recorded."""
    instance = _build_instance(rounds=4, pair_count=8)
    known = known_pairs_for_attack_instance(instance)[:KNOWN_PAIRS]
    verification = independent_verification_pairs(
        instance.master_key,
        rounds=4,
        pair_count=VERIFICATION_PAIRS,
        seed=VERIFICATION_SEED,
    )
    factory = IsolatedSolverFactory.from_pairs(
        known, rounds=4, instance_id="timeout_cause_test"
    )
    model = factory.build_isolated_model("sat_timeout")

    # Run with an impossibly small timeout
    res = run_multi_round_sat_only_recovery(
        model,
        known,
        verification,
        rounds=4,
        solver_timeout=1e-6,
        global_timeout=1e-6,
    )

    assert res.success is False
    assert res.recovered_master_key is None
    assert res.status in (STATUS_SOLVER_TIMEOUT, STATUS_GLOBAL_TIMEOUT)


def test_runner_prospective_gate_fails_closed_on_proposed_manifest() -> None:
    """The comparator runner must fail closed when --prospective is invoked on a proposed manifest."""
    from scripts.evaluate_seven_round_comparators import main

    manifest_path = Path("docs/seven_round_protocol_manifest.json")
    dataset_path = Path("datasets/phase11a_oracle_r7/oracle_evaluation.json")
    assert manifest_path.exists()
    assert dataset_path.exists()

    with pytest.raises(SystemExit, match="PROSPECTIVE EXECUTION REFUSED.*PROPOSED_NOT_AUTHORIZED"):
        main([
            "--prospective",
            "--protocol-manifest", str(manifest_path),
            "--dataset", str(dataset_path),
            "--instances", "1",
        ])


def test_runner_flags_conflict_rejection() -> None:
    """Specifying both --prospective and --development-mode must be rejected."""
    from scripts.evaluate_seven_round_comparators import parse_args

    with pytest.raises(SystemExit, match="Cannot specify both --prospective and --development-mode"):
        parse_args(["--prospective", "--development-mode"])

    with pytest.raises(SystemExit, match="--prospective requires an explicit --protocol-manifest"):
        parse_args(["--prospective"])

    with pytest.raises(SystemExit, match="--prospective requires an explicit --dataset path"):
        parse_args([
            "--prospective",
            "--protocol-manifest", "docs/seven_round_protocol_manifest.json",
        ])


def test_runner_rebuilding_seconds_and_summary(tmp_path: Path) -> None:
    """Records must include rebuilding_seconds and method summary must include total_rebuilding_seconds."""
    import scripts.evaluate_seven_round_comparators as runner_module

    out_file = tmp_path / "test_rebuilding.json"
    ret = runner_module.main(
        [
            "--methods",
            "sat_only_4obs",
            "--instances",
            "1",
            "--global-timeout",
            "1.0",
            "--output",
            str(out_file),
        ]
    )
    assert ret == 0
    with out_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rec = data["records_by_method"]["sat_only_4obs"][0]
    assert "rebuilding_seconds" in rec
    assert rec["rebuilding_seconds"] >= 0.0

    sm = data["summary_by_method"]["sat_only_4obs"]
    assert "total_rebuilding_seconds" in sm
    assert sm["total_rebuilding_seconds"] >= 0.0


def test_runner_non_interruptible_overshoot_reclassification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Non-interruptible methods that exceed the cutoff without recovering must be classified as global_timeout."""
    import scripts.evaluate_seven_round_comparators as runner_module
    from neural_sat_speck.attacks.multi_round_recovery import MultiRoundRecoveryResult

    def mock_overshoot_sat_only(
        model,
        known_pairs,
        verification_pairs,
        *,
        rounds,
        solver_timeout,
        global_timeout,
        strategy,
    ):
        import time
        # Simulate a non-interruptible operation that overshoots the cutoff
        time.sleep(0.1)
        return MultiRoundRecoveryResult(
            strategy=strategy,
            status="candidates_exhausted_without_unresolved_attempts",
            success=False,
            rounds=rounds,
            assumed_round_keys=0,
            residual_rounds=rounds,
            candidates_attempted=1,
            solver_calls=1,
            attempts=(),
            recovered_master_key=None,
            solver_seconds=0.1,
            verification_seconds=0.0,
            total_seconds=0.1,
        )

    monkeypatch.setattr(
        runner_module, "run_multi_round_sat_only_recovery", mock_overshoot_sat_only
    )

    out_file = tmp_path / "test_overshoot.json"
    ret = runner_module.main(
        [
            "--methods",
            "sat_only_4obs",
            "--instances",
            "1",
            "--global-timeout",
            "0.05",
            "--output",
            str(out_file),
        ]
    )
    assert ret == 0
    with out_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rec = data["records_by_method"]["sat_only_4obs"][0]
    assert rec["status"] == "global_timeout"
    assert rec["failure_cause"] == "global_timeout"
    assert rec["failure_domain"] == "envelope"


def test_runner_resume_validation_fail_closed(tmp_path: Path) -> None:
    """Resume must fail closed when prospective flags, manifest path, or methods mismatch."""
    import scripts.evaluate_seven_round_comparators as runner_module

    base_out = tmp_path / "base_run.json"
    ret = runner_module.main(
        [
            "--methods",
            "sat_only_4obs",
            "--instances",
            "1",
            "--global-timeout",
            "1.0",
            "--output",
            str(base_out),
        ]
    )
    assert ret == 0

    # 1. Prospective flag mismatch (saved run had prospective=True, resume attempted with prospective=False)
    prosp_out = tmp_path / "prospective_run.json"
    with base_out.open("r", encoding="utf-8") as f:
        prosp_data = json.load(f)
    prosp_data["metadata"]["prospective"] = True
    with prosp_out.open("w", encoding="utf-8") as f:
        json.dump(prosp_data, f, indent=2)

    with pytest.raises(SystemExit, match="--resume prospective flag mismatch"):
        runner_module.main(
            [
                "--methods",
                "sat_only_4obs",
                "--instances",
                "1",
                "--resume",
                "--output",
                str(prosp_out),
            ]
        )

    # 2. Methods mismatch (requested method not in resumed file)
    with pytest.raises(SystemExit, match="must be a subset of resumed file methods"):
        runner_module.main(
            [
                "--methods",
                "random_sat",
                "--instances",
                "1",
                "--global-timeout",
                "1.0",
                "--resume",
                "--output",
                str(base_out),
            ]
        )


def test_runner_late_verified_recovery_reclassification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A method that succeeds after cutoff wall time must be reclassified as global_timeout with late_verified_recovery=True."""
    import scripts.evaluate_seven_round_comparators as runner_module
    from neural_sat_speck.attacks.multi_round_recovery import MultiRoundRecoveryResult

    def mock_late_success_sat_only(
        model,
        known_pairs,
        verification_pairs,
        *,
        rounds,
        solver_timeout,
        global_timeout,
        strategy,
    ):
        import time

        # Simulate finding a verified key only after exceeding the cutoff
        time.sleep(0.1)
        return MultiRoundRecoveryResult(
            strategy=strategy,
            status="sat_and_verified",
            success=True,
            rounds=rounds,
            assumed_round_keys=0,
            residual_rounds=rounds,
            candidates_attempted=1,
            solver_calls=1,
            attempts=(),
            recovered_master_key=(0x1111, 0x2222, 0x3333, 0x4444),
            solver_seconds=0.1,
            rebuilding_seconds=0.0,
            verification_seconds=0.0,
            total_seconds=0.1,
        )

    monkeypatch.setattr(
        runner_module, "run_multi_round_sat_only_recovery", mock_late_success_sat_only
    )

    out_file = tmp_path / "test_late_verified.json"
    ret = runner_module.main(
        [
            "--methods",
            "sat_only_4obs",
            "--instances",
            "1",
            "--global-timeout",
            "0.05",
            "--output",
            str(out_file),
        ]
    )
    assert ret == 0
    with out_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rec = data["records_by_method"]["sat_only_4obs"][0]
    assert rec["late_verified_recovery"] is True
    assert rec["success"] is False
    assert rec["recovered_master_key"] is None
    assert rec["status"] == "global_timeout"
    assert rec["failure_cause"] == "global_timeout"
    assert rec["counts_as_verified_recovery"] is False
    assert rec["counts_as_unconditional_failure"] is True
    assert rec["failure_domain"] == "envelope"


def test_zero_remaining_time_clean_exit() -> None:
    """Non-positive global_timeout (e.g. 0.0) must cleanly exit with global_timeout status without raising ValueError."""
    instance = _build_instance(rounds=4, pair_count=8)
    known = known_pairs_for_attack_instance(instance)[:KNOWN_PAIRS]
    verification = independent_verification_pairs(
        instance.master_key,
        rounds=4,
        pair_count=VERIFICATION_PAIRS,
        seed=VERIFICATION_SEED,
    )
    factory = IsolatedSolverFactory.from_pairs(known, rounds=4, instance_id=instance.instance_id)
    model = factory.build_isolated_model("zero_timeout_test")

    # 1. MultiRoundRecovery key recovery
    candidates = [{2: 0x1234, 3: 0x5678}]
    res_key = run_multi_round_key_recovery(
        model,
        candidates,
        known,
        verification,
        rounds=4,
        solver_timeout=10.0,
        global_timeout=0.0,
    )
    assert res_key.success is False
    assert res_key.status == STATUS_GLOBAL_TIMEOUT
    assert res_key.candidates_attempted == 0

    # 2. MultiRoundRecovery sat-only recovery
    res_sat = run_multi_round_sat_only_recovery(
        model,
        known,
        verification,
        rounds=4,
        solver_timeout=10.0,
        global_timeout=0.0,
    )
    assert res_sat.success is False
    assert res_sat.status == STATUS_GLOBAL_TIMEOUT
    assert res_sat.solver_calls == 0

    # 3. Hierarchical multi-round attack
    instance_64 = _build_instance(rounds=4, pair_count=64)
    known_64 = known_pairs_for_attack_instance(instance_64)[:KNOWN_PAIRS]
    verification_64 = independent_verification_pairs(
        instance_64.master_key,
        rounds=4,
        pair_count=VERIFICATION_PAIRS,
        seed=VERIFICATION_SEED,
    )
    dummy_model = _DummyGroupRanker()
    stages = (
        HierarchicalStage(round_index=3, model=dummy_model, beam_width=2),
        HierarchicalStage(round_index=2, model=dummy_model, beam_width=2),
    )
    res_hier = run_hierarchical_multi_round_attack(
        instance_64,
        rounds=4,
        stages=stages,
        known_pairs=known_64,
        verification_pairs=verification_64,
        solver_timeout=5.0,
        global_timeout=0.0,
    )
    assert res_hier.success is False
    assert res_hier.status == STATUS_GLOBAL_TIMEOUT
    assert res_hier.failure_cause == FAILURE_CAUSE_GLOBAL_TIMEOUT
    assert res_hier.candidates_attempted == 0


def test_verify_record_integrity_contradictions() -> None:
    """verify_record_integrity must catch contradictions in success, wall time, and late_verified_recovery."""
    from scripts.analyze_seven_round_results import verify_record_integrity

    # Base valid success record
    valid_success = {
        "instance_id": "test-0001",
        "success": True,
        "status": "sat_and_verified",
        "counts_as_verified_recovery": True,
        "counts_as_unconditional_failure": False,
        "failure_domain": "none",
        "recovered_master_key": [0x1, 0x2, 0x3, 0x4],
        "late_verified_recovery": False,
        "cutoff_seconds": 300.0,
        "method_wall_seconds": 15.0,
    }
    assert verify_record_integrity(valid_success) is True

    # 1. Success with wall_sec >= cutoff
    bad_wall = dict(valid_success, method_wall_seconds=300.5)
    with pytest.raises(ValueError, match="success=True but method_wall_seconds .* >= cutoff"):
        verify_record_integrity(bad_wall)

    # 2. Success with late_verified_recovery=True
    bad_late_success = dict(valid_success, late_verified_recovery=True)
    with pytest.raises(ValueError, match="success=True but late_verified_recovery=True"):
        verify_record_integrity(bad_late_success)

    # Base valid failure record with late_verified_recovery=True
    valid_late_failure = {
        "instance_id": "test-0002",
        "success": False,
        "status": "global_timeout",
        "counts_as_verified_recovery": False,
        "counts_as_unconditional_failure": True,
        "failure_domain": "envelope",
        "recovered_master_key": None,
        "late_verified_recovery": True,
        "cutoff_seconds": 300.0,
        "method_wall_seconds": 305.0,
    }
    assert verify_record_integrity(valid_late_failure) is False

    # 3. late_verified_recovery=True but status != 'global_timeout'
    bad_status = dict(valid_late_failure, status="solver_timeout")
    with pytest.raises(ValueError, match="late_verified_recovery=True but status='solver_timeout' != 'global_timeout'"):
        verify_record_integrity(bad_status)

    # 4. late_verified_recovery=True but counts_as_unconditional_failure=False
    bad_fail_count = dict(valid_late_failure, counts_as_unconditional_failure=False)
    with pytest.raises(ValueError, match="late_verified_recovery=True but counts_as_unconditional_failure=False"):
        verify_record_integrity(bad_fail_count)

    # 5. late_verified_recovery=True but failure_domain != 'envelope'
    bad_domain = dict(valid_late_failure, failure_domain="timeout")
    with pytest.raises(ValueError, match="late_verified_recovery=True but failure_domain='timeout' != 'envelope'"):
        verify_record_integrity(bad_domain)

