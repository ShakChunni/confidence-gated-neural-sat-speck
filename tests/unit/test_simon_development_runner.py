"""Unit tests for the SIMON32/64 R=10 matched development comparison runner.

Verifies:
1. Counterbalanced method order across instances (Latin-square cyclic shift with diagnostic separation).
2. Strict input validation (64 pairs, 2 verification pairs, zero plaintext overlap, input difference).
3. Atomic persistence and durable resume (interruption simulation, corrupt resume rejection, sidecar checks).
4. Tool provenance and table hash verification (source-binary matching, executable check, schema signature).
5. Diagnostic oracle separation from primary comparators (truth isolation, role tagging, summary separation).
"""

from __future__ import annotations

import json
import stat

import pytest

import scripts.evaluate_simon_development_comparators as runner
from neural_sat_speck.simon.attack_instance import (
    SimonAttackInstance,
    SimonAttackInstanceConfig,
    SimonAttackPair,
    generate_simon_attack_instances,
)
from neural_sat_speck.simon.comparison_input import SimonComparisonInput
from neural_sat_speck.simon.method_worker import (
    MethodWorkerOutcome,
    run_method_worker,
)


@pytest.mark.parametrize(
    ("status", "domain"),
    [
        ("global_deadline", "solving"),
        ("candidate_unresolved", "solving"),
        ("ranking_timeout", "ranking"),
        ("exception", "engineering"),
    ],
)
def test_unresolved_simon_outcomes_are_not_resolved_sat_states(status, domain):
    outcome = MethodWorkerOutcome(
        method="sat_only_n128",
        outcome=status,
        complete_online_wall_seconds=300.0,
        ranking_wall_seconds=0.0,
        solving_wall_seconds=0.0,
        verification_wall_seconds=0.0,
        candidates_tested_count=0,
        candidates_remaining_count=0,
        recovered_master_key=None,
        verified_on_fresh_pairs=False,
        candidate_attempts=(),
        peak_rss_gib=0.0,
    )
    classified = runner.classify_simon_outcome(outcome, success=False)
    assert classified["counts_as_unconditional_failure"] is True
    assert classified["is_resolved_sat_state"] is False
    assert classified["failure_domain"] == domain


def test_watchdog_during_ranking_is_a_ranking_failure():
    outcome = MethodWorkerOutcome(
        method="fully_neural_1parent",
        outcome="global_deadline",
        complete_online_wall_seconds=300.0,
        ranking_wall_seconds=0.0,
        solving_wall_seconds=0.0,
        verification_wall_seconds=0.0,
        candidates_tested_count=0,
        candidates_remaining_count=0,
        recovered_master_key=None,
        verified_on_fresh_pairs=False,
        candidate_attempts=(),
        peak_rss_gib=0.0,
        error_message="Terminated by parent watchdog (interrupted during ranking)",
    )
    classified = runner.classify_simon_outcome(outcome, success=False)
    assert classified["is_resolved_sat_state"] is False
    assert classified["failure_domain"] == "ranking"


@pytest.fixture
def sample_64pair_instance() -> SimonAttackInstance:
    """Deterministic 64-pair R=10 attack instance for unit tests."""
    cfg = SimonAttackInstanceConfig(
        instance_count=1,
        pair_count=64,
        verification_pair_count=2,
        rounds=10,
        seed=999999,
        split_role="demo",
    )
    return generate_simon_attack_instances(cfg)[0]


# ==============================================================================
# 1. Counterbalanced Method Order Across Instances
# ==============================================================================


def test_counterbalanced_method_order_cyclic_shift():
    """Verify primary comparators rotate cyclically by shift = instance_index % 8."""
    comparators = runner.CANONICAL_COMPARATOR_ORDER
    m_count = len(comparators)
    assert m_count == 8

    for idx in range(16):
        scheduled = runner.get_counterbalanced_method_order(idx, comparators)
        expected_shift = idx % m_count
        expected_order = [
            comparators[(expected_shift + j) % m_count] for j in range(m_count)
        ]
        assert scheduled == expected_order


def test_counterbalanced_method_order_latin_square_uniformity():
    """Verify that across 32 instances, each comparator appears in each position exactly 4 times."""
    comparators = runner.CANONICAL_COMPARATOR_ORDER
    m_count = len(comparators)
    total_instances = 32

    # position_counts[method][pos] = number of times method appears at pos
    position_counts = {m: [0] * m_count for m in comparators}

    for idx in range(total_instances):
        order = runner.get_counterbalanced_method_order(idx, comparators)
        for pos, method in enumerate(order):
            position_counts[method][pos] += 1

    expected_per_slot = total_instances // m_count  # 32 // 8 = 4
    for method, counts in position_counts.items():
        assert counts == [expected_per_slot] * m_count, (
            f"Method {method} does not have uniform Latin-square position counts: {counts}"
        )


def test_counterbalanced_method_order_diagnostic_oracle_appended():
    """Verify diagnostic oracle is kept separate and placed at the end, preserving comparator slots."""
    methods = list(runner.CANONICAL_COMPARATOR_ORDER) + ["oracle_2key_n4"]
    for idx in range(16):
        order = runner.get_counterbalanced_method_order(idx, methods)
        assert len(order) == 9
        # First 8 must be comparators
        assert order[-1] == "oracle_2key_n4"
        assert set(order[:8]) == set(runner.CANONICAL_COMPARATOR_ORDER)
        # Shift must still match idx % 8 for the first 8 items
        shift = idx % 8
        assert order[0] == runner.CANONICAL_COMPARATOR_ORDER[shift]


def test_counterbalanced_method_order_subset():
    """Verify cyclic rotation functions properly on arbitrary subsets of methods."""
    subset = ["random_control_128", "sat_only_n4", "fully_classical_8parent"]
    for idx in range(6):
        order = runner.get_counterbalanced_method_order(idx, subset)
        expected_shift = idx % 3
        expected = [subset[(expected_shift + j) % 3] for j in range(3)]
        assert order == expected


def test_counterbalanced_method_order_alias_normalization():
    """Verify method aliases are mapped to canonical names during scheduling."""
    methods_with_aliases = ["hybrid", "sat_only", "oracle_a2_n4"]
    order = runner.get_counterbalanced_method_order(0, methods_with_aliases)
    assert order == [
        "neural_classical_hybrid_8parent",
        "sat_only_n4",
        "oracle_2key_n4",
    ]


# ==============================================================================
# 2. Strict Input Validation (64 pairs, 2 verification pairs, zero overlap)
# ==============================================================================


def test_input_validation_valid_instance(sample_64pair_instance):
    """Valid 64-pair instance passes validation cleanly."""
    runner.validate_attack_instance_for_comparison(sample_64pair_instance)


def test_input_validation_rejects_wrong_pair_count(sample_64pair_instance):
    """Reject instances with != 64 pairs."""
    # 63 pairs
    truncated = SimonAttackInstance(
        instance_id=sample_64pair_instance.instance_id,
        split_role=sample_64pair_instance.split_role,
        master_key=sample_64pair_instance.master_key,
        rounds=sample_64pair_instance.rounds,
        pairs=sample_64pair_instance.pairs[:63],
        verification_pairs=sample_64pair_instance.verification_pairs,
    )
    with pytest.raises(ValueError, match="pair count mismatch"):
        runner.validate_attack_instance_for_comparison(truncated, expected_pair_count=64)


def test_input_validation_rejects_wrong_verification_count(sample_64pair_instance):
    """Reject instances with != 2 verification pairs."""
    # 1 verification pair
    truncated_verif = SimonAttackInstance(
        instance_id=sample_64pair_instance.instance_id,
        split_role=sample_64pair_instance.split_role,
        master_key=sample_64pair_instance.master_key,
        rounds=sample_64pair_instance.rounds,
        pairs=sample_64pair_instance.pairs,
        verification_pairs=sample_64pair_instance.verification_pairs[:1],
    )
    with pytest.raises(ValueError, match="verification pair count mismatch"):
        runner.validate_attack_instance_for_comparison(truncated_verif)


def test_input_validation_rejects_duplicate_attack_plaintexts(sample_64pair_instance):
    """Reject duplicate plaintexts within attack pairs."""
    bad_pairs = list(sample_64pair_instance.pairs)
    # duplicate pair 0 into pair 1
    bad_pairs[1] = bad_pairs[0]
    with pytest.raises(ValueError, match="Attack and verification plaintexts must be distinct"):
        SimonAttackInstance(
            instance_id="bad_inst",
            split_role="demo",
            master_key=sample_64pair_instance.master_key,
            rounds=10,
            pairs=tuple(bad_pairs),
            verification_pairs=sample_64pair_instance.verification_pairs,
        )


def test_input_validation_rejects_duplicate_verification_plaintexts(sample_64pair_instance):
    """Reject duplicate plaintexts within verification pairs."""
    bad_verif = (
        sample_64pair_instance.verification_pairs[0],
        sample_64pair_instance.verification_pairs[0],
    )
    with pytest.raises(ValueError, match="Attack and verification plaintexts must be distinct"):
        SimonAttackInstance(
            instance_id="bad_inst_verif",
            split_role="demo",
            master_key=sample_64pair_instance.master_key,
            rounds=10,
            pairs=sample_64pair_instance.pairs,
            verification_pairs=bad_verif,
        )


def test_input_validation_rejects_overlap_between_attack_and_verification(sample_64pair_instance):
    """Reject any plaintext overlap between attack pairs and verification pairs."""
    # Leaked plaintext: verif[0] has same plaintext as pair[0].plaintext0
    leaked_pt = sample_64pair_instance.pairs[0].plaintext0
    leaked_verif = (
        (leaked_pt, (0x1234, 0x5678)),
        sample_64pair_instance.verification_pairs[1],
    )
    with pytest.raises(ValueError, match="Attack and verification plaintexts must be distinct"):
        SimonAttackInstance(
            instance_id="bad_inst_leak",
            split_role="demo",
            master_key=sample_64pair_instance.master_key,
            rounds=10,
            pairs=sample_64pair_instance.pairs,
            verification_pairs=leaked_verif,
        )


def test_input_validation_rejects_wrong_input_difference(sample_64pair_instance):
    """Reject attack pairs that violate the canonical input difference."""
    bad_pairs = list(sample_64pair_instance.pairs)
    # pair 0 with different difference
    p0 = bad_pairs[0]
    tampered_pair = SimonAttackPair(
        plaintext0=p0.plaintext0,
        plaintext1=(p0.plaintext0[0] ^ 0x0001, p0.plaintext0[1]),  # wrong diff
        ciphertext0=p0.ciphertext0,
        ciphertext1=p0.ciphertext1,
    )
    bad_pairs[0] = tampered_pair
    tampered_inst = SimonAttackInstance(
        instance_id="tampered_diff",
        split_role="demo",
        master_key=sample_64pair_instance.master_key,
        rounds=10,
        pairs=tuple(bad_pairs),
        verification_pairs=sample_64pair_instance.verification_pairs,
    )
    with pytest.raises(ValueError, match="input difference"):
        runner.validate_attack_instance_for_comparison(tampered_inst)


def test_comparison_input_isolation_rankers_never_see_plaintexts_or_verif(sample_64pair_instance):
    """Verify SimonComparisonInput strips plaintexts and verification pairs for rankers."""
    comp_input = SimonComparisonInput.from_attack_instance(
        instance=sample_64pair_instance,
        manifest_seed=12345,
        pair_count=64,
    )
    assert len(comp_input.ciphertext_pairs) == 64
    for ct0, ct1 in comp_input.ciphertext_pairs:
        assert isinstance(ct0, tuple) and len(ct0) == 2
        assert isinstance(ct1, tuple) and len(ct1) == 2

    # Verification plaintexts/ciphertexts must never be in ciphertext_pairs
    verif_pts = {vp[0] for vp in comp_input.verification_pairs}
    verif_cts = {vp[1] for vp in comp_input.verification_pairs}
    for ct0, ct1 in comp_input.ciphertext_pairs:
        assert ct0 not in verif_pts
        assert ct1 not in verif_pts
        assert ct0 not in verif_cts
        assert ct1 not in verif_cts


# ==============================================================================
# 3. Atomic Persistence and Durable Resume
# ==============================================================================


def test_save_atomic_creates_json_and_matching_sidecar(tmp_path):
    """Verify save_atomic_with_sidecar writes both JSON and matching .sha256 sidecar."""
    target = tmp_path / "results" / "test_out.json"
    sidecar = tmp_path / "results" / "test_out.json.sha256"

    test_data = {"status": "ok", "count": 42}
    computed_hash = runner.save_atomic_with_sidecar(test_data, target)

    assert target.is_file()
    assert sidecar.is_file()
    assert runner._sha256_file(target) == computed_hash

    sidecar_lines = sidecar.read_text(encoding="utf-8").strip().split()
    assert sidecar_lines[0] == computed_hash
    assert sidecar_lines[1] == "test_out.json"


def test_resume_rejects_missing_output_file(tmp_path):
    """Resume fails closed if the output file is missing."""
    target = tmp_path / "missing.json"
    with pytest.raises(SystemExit, match="does not exist"):
        runner.load_and_validate_resume_file(target, {})


def test_resume_rejects_missing_sidecar(tmp_path):
    """Resume fails closed if the .sha256 sidecar is missing."""
    target = tmp_path / "test.json"
    target.write_text(json.dumps({"metadata": {}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="missing required sidecar"):
        runner.load_and_validate_resume_file(target, {})


def test_resume_rejects_corrupted_sidecar_checksum(tmp_path):
    """Resume fails closed if the sidecar checksum does not match the file."""
    target = tmp_path / "test.json"
    runner.save_atomic_with_sidecar({"metadata": {}}, target)

    # Tamper with sidecar
    sidecar = tmp_path / "test.json.sha256"
    sidecar.write_text("bad_checksum_hash  test.json\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="sidecar verification failed"):
        runner.load_and_validate_resume_file(target, {})


def test_resume_rejects_config_mismatch(tmp_path):
    """Resume fails closed if metadata parameters do not match current configuration."""
    target = tmp_path / "test.json"
    meta = {
        "schema_version": runner.SCHEMA_VERSION,
        "rounds": 9,  # Mismatched rounds (expected 10)
        "pair_count": 64,
        "dataset_sha256": "abc",
        "global_deadline_seconds": 300.0,
        "candidate_cutoff_seconds": 45.0,
        "dataset_path": "path",
        "known_observations": 128,
        "candidate_formula_observations": 4,
        "verification_pairs": 2,
        "instances": 32,
        "start": 0,
        "methods": [],
        "method_order_policy": "latin_square_cyclic_shift_with_diagnostic_separation",
        "candidate_cutoff_soft_seconds": 45.0,
        "ranking_time_limit": 120.0,
        "git_commit": "unknown",
        "environment_provenance": {},
        "tool_provenance": {},
        "code_provenance": {},
        "resume_events": [],
    }
    data = {
        "metadata": meta,
        "records_by_method": {},
    }
    runner.save_atomic_with_sidecar(data, target)

    expected_cfg = {
        "rounds": 10,
        "pair_count": 64,
        "dataset_sha256": "abc",
        "global_deadline_seconds": 300.0,
        "candidate_cutoff_seconds": 45.0,
    }
    with pytest.raises(SystemExit, match="configuration mismatch for 'rounds'"):
        runner.load_and_validate_resume_file(target, expected_cfg)


def test_resume_rejects_duplicate_instance_in_method(tmp_path):
    """Resume fails closed if duplicate instance records exist within a method."""
    target = tmp_path / "test.json"
    meta = {
        "schema_version": runner.SCHEMA_VERSION,
        "rounds": 10,
        "pair_count": 64,
        "dataset_sha256": "abc",
        "global_deadline_seconds": 300.0,
        "candidate_cutoff_seconds": 45.0,
        "dataset_path": "path",
        "known_observations": 128,
        "candidate_formula_observations": 4,
        "verification_pairs": 2,
        "instances": 32,
        "start": 0,
        "methods": ["sat_only_n4"],
        "method_order_policy": "latin_square_cyclic_shift_with_diagnostic_separation",
        "candidate_cutoff_soft_seconds": 45.0,
        "ranking_time_limit": 120.0,
        "git_commit": "unknown",
        "environment_provenance": {},
        "tool_provenance": {},
        "code_provenance": {},
        "resume_events": [],
    }
    rec1 = {
        "instance_id": "inst_0001",
        "instance_index": 0,
        "method": "sat_only_n4",
        "method_display_name": "SAT Only",
        "method_role": "comparator",
        "execution_order": ["sat_only_n4"],
        "execution_order_index": 0,
        "outcome": "candidate_timeout",
        "success": False,
        "status": "candidate_timeout",
        "failure_cause": "candidate_timeout",
        "recovered_master_key": None,
        "verified_on_fresh_pairs": False,
        "candidates_tested_count": 1,
        "candidates_remaining_count": 0,
        "complete_online_wall_seconds": 1.0,
        "ranking_wall_seconds": 0.0,
        "solving_wall_seconds": 1.0,
        "verification_wall_seconds": 0.0,
        "peak_rss_gib": 0.1,
        "cutoff_seconds": 300.0,
        "cutoff_overshoot_seconds": 0.0,
        "candidate_attempts": [],
        "in_flight_attempt": None,
        "queue_sample": [],
        "full_pairs_used": None,
        "error_message": None,
        "counts_as_verified_recovery": False,
        "counts_as_unconditional_failure": True,
        "is_resolved_sat_state": True,
        "failure_domain": "ranking",
    }
    # Duplicate instance record
    data = {
        "metadata": meta,
        "records_by_method": {"sat_only_n4": [rec1, dict(rec1)]},
    }
    runner.save_atomic_with_sidecar(data, target)

    expected_cfg = {
        "rounds": 10,
        "pair_count": 64,
        "dataset_sha256": "abc",
        "global_deadline_seconds": 300.0,
        "candidate_cutoff_seconds": 45.0,
    }
    with pytest.raises(SystemExit, match="duplicate record for instance"):
        runner.load_and_validate_resume_file(target, expected_cfg)


# ==============================================================================
# 4. Tool Provenance and Table Hash Verification
# ==============================================================================


def test_tool_provenance_rejects_missing_binary(tmp_path):
    """Tool provenance fails closed when tool binary does not exist."""
    missing_bin = tmp_path / "missing_tool"
    ddt_dummy = tmp_path / "ddt.bin"
    ddt_dummy.write_bytes(b"dummy")
    src_dummy = tmp_path / "source.cpp"
    src_dummy.write_bytes(b"dummy")

    with pytest.raises(FileNotFoundError, match="tool binary not found"):
        runner.verify_simon_tool_provenance(
            ddt_7r_path=ddt_dummy,
            tool_8r_path=missing_bin,
            tool_8r_source=src_dummy,
            verify_hashes=False,
        )


def test_tool_provenance_rejects_non_executable_binary(tmp_path):
    """Tool provenance fails closed when tool binary is not executable."""
    tool_bin = tmp_path / "non_exec_tool"
    tool_bin.write_bytes(b"simon_queried_row_8r.v1")
    # Make non-executable
    tool_bin.chmod(stat.S_IRUSR | stat.S_IWUSR)
    ddt_dummy = tmp_path / "ddt.bin"
    ddt_dummy.write_bytes(b"dummy")
    src_dummy = tmp_path / "source.cpp"
    src_dummy.write_bytes(b"dummy")

    with pytest.raises(PermissionError, match="not executable"):
        runner.verify_simon_tool_provenance(
            ddt_7r_path=ddt_dummy,
            tool_8r_path=tool_bin,
            tool_8r_source=src_dummy,
            verify_hashes=False,
        )


def test_tool_provenance_rejects_binary_missing_schema_signature(tmp_path):
    """Tool provenance fails closed when binary lacks 'simon_queried_row_8r.v1'."""
    tool_bin = tmp_path / "tool"
    tool_bin.write_bytes(b"wrong_binary_signature")
    tool_bin.chmod(stat.S_IRWXU)
    ddt_dummy = tmp_path / "ddt.bin"
    ddt_dummy.write_bytes(b"dummy")
    src_dummy = tmp_path / "source.cpp"
    src_dummy.write_bytes(b"dummy")

    with pytest.raises(ValueError, match="does not contain expected schema signature"):
        runner.verify_simon_tool_provenance(
            ddt_7r_path=ddt_dummy,
            tool_8r_path=tool_bin,
            tool_8r_source=src_dummy,
            verify_hashes=False,
        )


def test_tool_provenance_real_environment():
    """Verify tool provenance succeeds on current repository and environment files."""
    ddt_path = runner.DEFAULT_DDT_7R_PATH
    tool_bin = runner.DEFAULT_TOOL_8R_PATH
    tool_src = runner.DEFAULT_TOOL_8R_SOURCE
    vv_dir = runner.DEFAULT_VV_DIR

    if not ddt_path.is_file() or not tool_bin.is_file():
        pytest.skip("Local /private/tmp artifacts not present in test environment.")

    info = runner.verify_simon_tool_provenance(
        ddt_7r_path=ddt_path,
        tool_8r_path=tool_bin,
        tool_8r_source=tool_src,
        vv_dir=vv_dir,
        verify_hashes=True,
    )
    assert info["tool_8r_verified_matches_source"] is True
    assert info["ddt_7r_sha256"].startswith("52b3f3d4")
    assert len(info["tool_8r_binary_sha256"]) == 64
    assert len(info["tool_8r_source_sha256"]) == 64


# ==============================================================================
# 5. Diagnostic Oracle Separation from Primary Comparators
# ==============================================================================


def test_comparators_cannot_receive_oracle_trailing_keys(sample_64pair_instance):
    """Comparators MUST reject oracle_trailing_keys with ValueError."""
    comp_input = SimonComparisonInput.from_attack_instance(
        instance=sample_64pair_instance,
        manifest_seed=999999,
        pair_count=64,
    )
    oracle_keys = sample_64pair_instance.expected_trailing_subkeys(2)

    for comparator in runner.CANONICAL_COMPARATOR_ORDER:
        with pytest.raises(ValueError, match="cannot receive oracle trailing keys"):
            run_method_worker(
                method=comparator,
                rounds=10,
                comparison_input=comp_input,
                oracle_trailing_keys=oracle_keys,
                use_parent_watchdog=False,
            )


def test_diagnostic_oracle_requires_oracle_trailing_keys(sample_64pair_instance):
    """Diagnostic oracle MUST reject execution if oracle_trailing_keys is None."""
    comp_input = SimonComparisonInput.from_attack_instance(
        instance=sample_64pair_instance,
        manifest_seed=999999,
        pair_count=64,
    )

    with pytest.raises(ValueError, match="Diagnostic oracle .* requires oracle_trailing_keys"):
        run_method_worker(
            method="oracle_2key_n4",
            rounds=10,
            comparison_input=comp_input,
            oracle_trailing_keys=None,
            use_parent_watchdog=False,
        )


def test_summary_and_reporting_separates_comparators_and_diagnostic():
    """Verify that summary computation separates comparators from diagnostic oracle."""
    records_comp = [
        {
            "success": True,
            "complete_online_wall_seconds": 15.0,
            "cutoff_seconds": 300.0,
            "cutoff_overshoot_seconds": 0.0,
            "candidates_tested_count": 2,
            "ranking_wall_seconds": 5.0,
            "solving_wall_seconds": 10.0,
            "verification_wall_seconds": 0.001,
            "peak_rss_gib": 0.5,
            "outcome": "verified_recovery",
            "failure_domain": "none",
        }
    ]
    records_oracle = [
        {
            "success": True,
            "complete_online_wall_seconds": 22.0,
            "cutoff_seconds": 300.0,
            "cutoff_overshoot_seconds": 0.0,
            "candidates_tested_count": 1,
            "ranking_wall_seconds": 0.0,
            "solving_wall_seconds": 22.0,
            "verification_wall_seconds": 0.001,
            "peak_rss_gib": 0.05,
            "outcome": "verified_recovery",
            "failure_domain": "none",
        }
    ]

    sm_comp = runner.compute_method_summary(records_comp)
    sm_oracle = runner.compute_method_summary(records_oracle)

    assert sm_comp["verified_recoveries"] == 1
    assert sm_comp["verified_recovery_rate"] == 1.0
    assert sm_oracle["verified_recoveries"] == 1
    assert sm_oracle["mean_candidates_tested"] == 1.0


# ==============================================================================
# 6. End-to-End Interruption and Resume Simulation
# ==============================================================================


def test_interruption_simulation_and_full_resumption(tmp_path, sample_64pair_instance):
    """Simulate partial execution, interruption, and durable resumption on remaining methods."""
    results_file = tmp_path / "simon_resumable_test.json"
    sidecar_file = tmp_path / "simon_resumable_test.json.sha256"

    # Step 1: Write partial record (instance 0, sat_only_n4 completed)
    meta = {
        "schema_version": runner.SCHEMA_VERSION,
        "dataset_path": "test_dataset",
        "dataset_sha256": "fake_hash_123",
        "rounds": 10,
        "pair_count": 64,
        "known_observations": 128,
        "candidate_formula_observations": 4,
        "verification_pairs": 2,
        "instances": 1,
        "start": 0,
        "methods": ["sat_only_n4", "random_control_128"],
        "method_order_policy": "latin_square_cyclic_shift_with_diagnostic_separation",
        "global_deadline_seconds": 300.0,
        "candidate_cutoff_seconds": 45.0,
        "candidate_cutoff_soft_seconds": 45.0,
        "ranking_time_limit": 120.0,
        "git_commit": "test_commit",
        "environment_provenance": {},
        "tool_provenance": {},
        "code_provenance": {},
        "resume_events": [],
    }

    rec1 = {
        "instance_id": sample_64pair_instance.instance_id,
        "instance_index": 0,
        "method": "sat_only_n4",
        "method_display_name": "SAT Only (4 obs)",
        "method_role": "comparator",
        "execution_order": ["sat_only_n4", "random_control_128"],
        "execution_order_index": 0,
        "outcome": "candidate_timeout",
        "success": False,
        "status": "candidate_timeout",
        "failure_cause": "candidate_timeout",
        "recovered_master_key": None,
        "verified_on_fresh_pairs": False,
        "candidates_tested_count": 1,
        "candidates_remaining_count": 0,
        "complete_online_wall_seconds": 45.1,
        "ranking_wall_seconds": 0.0,
        "solving_wall_seconds": 45.1,
        "verification_wall_seconds": 0.0,
        "peak_rss_gib": 0.1,
        "cutoff_seconds": 300.0,
        "cutoff_overshoot_seconds": 0.0,
        "candidate_attempts": [],
        "in_flight_attempt": None,
        "queue_sample": [],
        "full_pairs_used": None,
        "error_message": None,
        "counts_as_verified_recovery": False,
        "counts_as_unconditional_failure": True,
        "is_resolved_sat_state": True,
        "failure_domain": "ranking",
    }

    initial_payload = {
        "metadata": meta,
        "evidence_class": "DEVELOPMENT_DIAGNOSTIC",
        "generated_at": "2026-09-29T00:00:00Z",
        "summary_by_method": {"sat_only_n4": runner.compute_method_summary([rec1])},
        "diagnostic_summary": {},
        "records_by_method": {
            "sat_only_n4": [rec1],
            "random_control_128": [],
        },
    }

    runner.save_atomic_with_sidecar(initial_payload, results_file)
    assert results_file.is_file()
    assert sidecar_file.is_file()

    # Step 2: Validate resume
    expected_cfg = {
        "rounds": 10,
        "pair_count": 64,
        "dataset_sha256": "fake_hash_123",
        "global_deadline_seconds": 300.0,
        "candidate_cutoff_seconds": 45.0,
    }
    loaded = runner.load_and_validate_resume_file(results_file, expected_cfg)
    assert len(loaded["records_by_method"]["sat_only_n4"]) == 1
    assert len(loaded["records_by_method"]["random_control_128"]) == 0

    # Step 3: Simulate runner completing random_control_128 and saving
    rec2 = dict(rec1)
    rec2["method"] = "random_control_128"
    rec2["method_display_name"] = "Random Control 128"
    rec2["execution_order_index"] = 1
    loaded["records_by_method"]["random_control_128"].append(rec2)
    loaded["summary_by_method"]["random_control_128"] = runner.compute_method_summary([rec2])

    runner.save_atomic_with_sidecar(loaded, results_file)

    # Step 4: Re-load to verify full resume state
    resumed = runner.load_and_validate_resume_file(results_file, expected_cfg)
    assert len(resumed["records_by_method"]["sat_only_n4"]) == 1
    assert len(resumed["records_by_method"]["random_control_128"]) == 1
