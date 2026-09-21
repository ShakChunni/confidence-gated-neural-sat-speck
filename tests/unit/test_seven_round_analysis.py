"""Unit tests for seven-round statistical analysis and Holm multiplicity control."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from analyze_seven_round_results import (  # noqa: E402
    analyze_records,
    compute_paired_contingency,
    format_text_report,
    holm_bonferroni_correction,
    paired_risk_difference_ci,
)


def test_holm_bonferroni_correction() -> None:
    """Verify Holm-Bonferroni step-down adjustments and monotonicity."""
    # Test 1: Classical 3-hypothesis test
    raw_p = [0.01, 0.04, 0.02]
    # Sorted order: idx 0 (0.01), idx 2 (0.02), idx 1 (0.04)
    # rank 0 (p=0.01): mult=3 -> 0.03
    # rank 1 (p=0.02): mult=2 -> 0.04, max(0.03, 0.04) = 0.04
    # rank 2 (p=0.04): mult=1 -> 0.04, max(0.04, 0.04) = 0.04
    res = holm_bonferroni_correction(raw_p, alpha=0.05)
    assert len(res) == 3

    assert math.isclose(res[0]["adjusted_p"], 0.03, abs_tol=1e-5)
    assert res[0]["rejected"] is True

    assert math.isclose(res[1]["adjusted_p"], 0.04, abs_tol=1e-5)
    assert res[1]["rejected"] is True

    assert math.isclose(res[2]["adjusted_p"], 0.04, abs_tol=1e-5)
    assert res[2]["rejected"] is True

    # Test 2: Monotonicity enforcement when raw p-values would invert
    raw_p2 = [0.02, 0.015]  # p[1] * 2 = 0.03, p[0] * 1 = 0.02 < 0.03 -> must be clamped to 0.03
    res2 = holm_bonferroni_correction(raw_p2, alpha=0.05)
    assert math.isclose(res2[1]["adjusted_p"], 0.03, abs_tol=1e-5)
    assert math.isclose(res2[0]["adjusted_p"], 0.03, abs_tol=1e-5)


def test_compute_paired_contingency() -> None:
    """Verify 2x2 contingency table generation."""
    succ_a = [True, True, False, False, True]
    succ_b = [True, False, True, False, True]

    table = compute_paired_contingency(succ_a, succ_b)
    assert table["n11"] == 2  # idx 0, 4
    assert table["n10"] == 1  # idx 1
    assert table["n01"] == 1  # idx 2
    assert table["n00"] == 1  # idx 3
    assert table["total"] == 5


def test_paired_risk_difference_ci() -> None:
    """Verify paired risk difference and confidence interval."""
    # When n10 == 10, n01 == 0 out of 20: diff = +0.50
    diff, lower, upper = paired_risk_difference_ci(n10=10, n01=0, total=20)
    assert math.isclose(diff, 0.50, abs_tol=1e-5)
    assert lower < diff < upper


def _build_synthetic_record(
    instance_id: str,
    success: bool,
    wall_seconds: float,
    solver_seconds: float = 50.0,
    candidates: int = 64,
    failure_domain: str | None = None,
    failure_cause: str | None = None,
) -> dict:
    if success:
        return {
            "instance_id": instance_id,
            "success": True,
            "status": "sat_and_verified",
            "counts_as_verified_recovery": True,
            "counts_as_unconditional_failure": False,
            "is_resolved_sat_state": True,
            "failure_domain": "none",
            "failure_cause": None,
            "recovered_master_key": [0x1234, 0x5678, 0x9ABC, 0xDEF0],
            "method_wall_seconds": wall_seconds,
            "solver_seconds": solver_seconds,
            "candidates_attempted": candidates,
        }
    else:
        return {
            "instance_id": instance_id,
            "success": False,
            "status": failure_cause or "candidates_exhausted",
            "counts_as_verified_recovery": False,
            "counts_as_unconditional_failure": True,
            "is_resolved_sat_state": True if failure_domain == "ranking" else False,
            "failure_domain": failure_domain or "ranking",
            "failure_cause": failure_cause or "candidates_exhausted_without_unresolved",
            "recovered_master_key": None,
            "method_wall_seconds": wall_seconds,
            "solver_seconds": solver_seconds,
            "candidates_attempted": candidates,
        }


def test_analyze_records_end_to_end() -> None:
    """Verify full end-to-end analysis across synthetic 6-method results."""
    methods = [
        "hierarchical_neural_screened",
        "hierarchical_neural_full",
        "hierarchical_neural_classical",
        "random_sat",
        "sat_only_4obs",
        "sat_only_all_observations",
    ]

    records_by_method: dict[str, list[dict]] = {m: [] for m in methods}
    n = 16

    for i in range(n):
        inst_id = f"inst_{i:04d}"
        # Screened succeeds 13/16 (81.25%)
        succ_scr = i < 13
        records_by_method["hierarchical_neural_screened"].append(
            _build_synthetic_record(
                inst_id,
                success=succ_scr,
                wall_seconds=100.0 if succ_scr else 300.0,
                candidates=64 if succ_scr else 1024,
                failure_domain=None if succ_scr else "ranking",
            )
        )

        # Full succeeds 13/16 (81.25%) but takes longer
        records_by_method["hierarchical_neural_full"].append(
            _build_synthetic_record(
                inst_id,
                success=succ_scr,
                wall_seconds=180.0 if succ_scr else 300.0,
                candidates=64 if succ_scr else 1024,
                failure_domain=None if succ_scr else "ranking",
            )
        )

        # Hybrid succeeds 6/16
        succ_hyb = i < 6
        records_by_method["hierarchical_neural_classical"].append(
            _build_synthetic_record(
                inst_id,
                success=succ_hyb,
                wall_seconds=120.0 if succ_hyb else 300.0,
                candidates=128,
                failure_domain=None if succ_hyb else "ranking",
            )
        )

        # Random succeeds 1/16
        succ_rnd = i < 1
        records_by_method["random_sat"].append(
            _build_synthetic_record(
                inst_id,
                success=succ_rnd,
                wall_seconds=50.0 if succ_rnd else 300.0,
                solver_seconds=200.0,
                candidates=1024,
                failure_domain=None if succ_rnd else "ranking",
            )
        )

        # SAT 4obs succeeds 0/16
        records_by_method["sat_only_4obs"].append(
            _build_synthetic_record(
                inst_id,
                success=False,
                wall_seconds=300.0,
                solver_seconds=300.0,
                candidates=1,
                failure_domain="solver",
                failure_cause="solver_timeout",
            )
        )

        # SAT 128obs succeeds 0/16
        records_by_method["sat_only_all_observations"].append(
            _build_synthetic_record(
                inst_id,
                success=False,
                wall_seconds=300.0,
                solver_seconds=300.0,
                candidates=1,
                failure_domain="solver",
                failure_cause="solver_timeout",
            )
        )

    analysis = analyze_records(records_by_method, global_cutoff=300.0, alpha=0.05)

    # Verify primary contrast (13/16 vs 0/16)
    pc = analysis["primary_paired_contrast"]
    assert pc is not None
    assert pc["contingency_table"]["n10"] == 13
    assert pc["contingency_table"]["n01"] == 0
    assert pc["risk_difference"] == 13 / 16
    assert pc["exact_mcnemar"]["p_value"] < 0.001
    assert pc["significant_at_alpha"] is True

    # Verify secondary contrasts
    sc = analysis["multiplicity_controlled_secondary_contrasts"]
    assert len(sc) == 3
    # Neural vs Hybrid: 13 vs 6 (discordant b=7, c=0)
    assert sc[0]["mcnemar"]["b"] == 7
    assert sc[0]["mcnemar"]["c"] == 0
    assert sc[0]["rejected_h0"] is True

    # Verify text report generation
    report = format_text_report(analysis)
    assert "STATISTICAL ANALYSIS" in report
    assert "PRIMARY HYPOTHESIS CONTRAST" in report
    assert "SIGNIFICANT (H0 rejected at alpha=0.05)" in report


def test_shuffled_instance_order_alignment() -> None:
    """Methods with shuffled instance order must be aligned by instance_id for paired testing."""
    import random
    methods = [
        "hierarchical_neural_screened",
        "hierarchical_neural_full",
        "hierarchical_neural_classical",
        "random_sat",
        "sat_only_4obs",
        "sat_only_all_observations",
    ]
    records_by_method: dict[str, list[dict]] = {m: [] for m in methods}
    n = 10
    for i in range(n):
        inst_id = f"inst_{i:04d}"
        for m in methods:
            succ = (i % 2 == 0) if m == "hierarchical_neural_screened" else (i % 3 == 0)
            records_by_method[m].append(
                _build_synthetic_record(inst_id, success=succ, wall_seconds=50.0 + i)
            )

    # Shuffle the record list order in method_b
    rng = random.Random(42)
    rng.shuffle(records_by_method["sat_only_all_observations"])

    analysis = analyze_records(records_by_method, global_cutoff=300.0)
    pc = analysis["primary_paired_contrast"]
    assert pc is not None
    # Method A: even indices {0, 2, 4, 6, 8} (5)
    # Method B: multiples of 3 {0, 3, 6, 9} (4)
    # Both: {0, 6} (2) -> n11 = 2
    # A only: {2, 4, 8} (3) -> n10 = 3
    # B only: {3, 9} (2) -> n01 = 2
    # Neither: {1, 5, 7} (3) -> n00 = 3
    assert pc["contingency_table"]["n11"] == 2
    assert pc["contingency_table"]["n10"] == 3
    assert pc["contingency_table"]["n01"] == 2
    assert pc["contingency_table"]["n00"] == 3


def test_duplicate_instance_id_rejection() -> None:
    """Duplicate instance IDs within any comparator method must be rejected."""
    import pytest
    methods = [
        "hierarchical_neural_screened",
        "hierarchical_neural_full",
        "hierarchical_neural_classical",
        "random_sat",
        "sat_only_4obs",
        "sat_only_all_observations",
    ]
    records_by_method: dict[str, list[dict]] = {
        m: [_build_synthetic_record("inst_0001", success=False, wall_seconds=10.0)]
        for m in methods
    }
    # Add duplicate to first method
    records_by_method["hierarchical_neural_screened"].append(
        _build_synthetic_record("inst_0001", success=False, wall_seconds=20.0)
    )
    with pytest.raises(ValueError, match="Duplicate instance_id 'inst_0001'"):
        analyze_records(records_by_method)


def test_missing_method_rejection() -> None:
    """Omitting any of the 6 required primary methods must be rejected."""
    import pytest
    records_by_method = {
        "hierarchical_neural_screened": [_build_synthetic_record("inst_0001", success=True, wall_seconds=10.0)]
    }
    with pytest.raises(ValueError, match="Comparator methods mismatch"):
        analyze_records(records_by_method)


def test_mismatched_instance_sets_rejection() -> None:
    """Mismatched instance ID sets across methods must be rejected."""
    import pytest
    methods = [
        "hierarchical_neural_screened",
        "hierarchical_neural_full",
        "hierarchical_neural_classical",
        "random_sat",
        "sat_only_4obs",
        "sat_only_all_observations",
    ]
    records_by_method = {
        m: [_build_synthetic_record("inst_0001", success=True, wall_seconds=10.0)]
        for m in methods
    }
    # Mismatch method B instance ID
    records_by_method["sat_only_all_observations"] = [
        _build_synthetic_record("inst_9999", success=True, wall_seconds=10.0)
    ]
    with pytest.raises(ValueError, match="Instance ID set mismatch"):
        analyze_records(records_by_method)


def test_contradictory_record_integrity() -> None:
    """Records with contradictory recovery flags or missing master keys must fail validation."""
    import pytest
    methods = [
        "hierarchical_neural_screened",
        "hierarchical_neural_full",
        "hierarchical_neural_classical",
        "random_sat",
        "sat_only_4obs",
        "sat_only_all_observations",
    ]

    # Test 1: success=True but counts_as_verified_recovery=False
    bad_rec1 = _build_synthetic_record("inst_0001", success=True, wall_seconds=10.0)
    bad_rec1["counts_as_verified_recovery"] = False
    records1 = {m: [bad_rec1 if m == "hierarchical_neural_screened" else _build_synthetic_record("inst_0001", success=False, wall_seconds=10.0)] for m in methods}
    with pytest.raises(ValueError, match="Contradictory record: success=True but counts_as_verified_recovery=False"):
        analyze_records(records1)

    # Test 2: success=True but recovered_master_key is None
    bad_rec2 = _build_synthetic_record("inst_0001", success=True, wall_seconds=10.0)
    bad_rec2["recovered_master_key"] = None
    records2 = {m: [bad_rec2 if m == "hierarchical_neural_screened" else _build_synthetic_record("inst_0001", success=False, wall_seconds=10.0)] for m in methods}
    with pytest.raises(ValueError, match="Contradictory record: success=True but invalid recovered_master_key"):
        analyze_records(records2)


def test_prospective_requires_n_128() -> None:
    """Prospective evaluation must strictly enforce N=128 sample size."""
    import pytest
    methods = [
        "hierarchical_neural_screened",
        "hierarchical_neural_full",
        "hierarchical_neural_classical",
        "random_sat",
        "sat_only_4obs",
        "sat_only_all_observations",
    ]
    # Test N=16 in prospective mode -> must fail
    records_16 = {
        m: [_build_synthetic_record(f"inst_{i:04d}", success=False, wall_seconds=10.0) for i in range(16)]
        for m in methods
    }
    with pytest.raises(ValueError, match="Prospective evaluation requires exactly N=128 instances; found 16"):
        analyze_records(records_16, prospective=True)

    # Test N=128 in prospective mode -> must succeed
    records_128 = {
        m: [_build_synthetic_record(f"inst_{i:04d}", success=(i == 0), wall_seconds=10.0) for i in range(128)]
        for m in methods
    }
    res = analyze_records(records_128, prospective=True)
    assert res["instances_evaluated"] == 128


def test_even_n_median_correctness() -> None:
    """Wall-clock median for even N must use the standard arithmetic mean of middle elements."""
    import statistics
    methods = [
        "hierarchical_neural_screened",
        "hierarchical_neural_full",
        "hierarchical_neural_classical",
        "random_sat",
        "sat_only_4obs",
        "sat_only_all_observations",
    ]
    # Even N = 4 with wall times: [10.0, 20.0, 30.0, 40.0] -> standard median is 25.0
    times = [10.0, 20.0, 30.0, 40.0]
    records = {
        m: [_build_synthetic_record(f"inst_{i}", success=False, wall_seconds=times[i]) for i in range(4)]
        for m in methods
    }
    res = analyze_records(records)
    screened_stats = res["method_metrics"]["hierarchical_neural_screened"]["wall_seconds"]
    assert screened_stats["median"] == statistics.median(times)
    assert screened_stats["median"] == 25.0


def test_newcombe_method_10_edge_cases() -> None:
    """Newcombe Method 10 paired score CI must correctly handle boundary edge cases."""
    from neural_sat_speck.evaluation.metrics import paired_risk_difference_newcombe

    # 1. Zero discordant pairs (identical outcomes across all instances, some successes)
    diff, low, high = paired_risk_difference_newcombe(n11=20, n10=0, n01=0, n00=20, total=40, confidence=0.95)
    assert diff == 0.0
    assert low < 0.0
    assert high > 0.0
    assert abs(low + high) < 1e-6  # Symmetric around 0

    # 2. All concordant successes (n11=50, all others 0)
    diff, low, high = paired_risk_difference_newcombe(n11=50, n10=0, n01=0, n00=0, total=50, confidence=0.95)
    assert diff == 0.0
    assert low < 0.0
    assert high > 0.0
    assert abs(low + high) < 1e-6

    # 3. All concordant failures (n00=50, all others 0)
    diff, low, high = paired_risk_difference_newcombe(n11=0, n10=0, n01=0, n00=50, total=50, confidence=0.95)
    assert diff == 0.0
    assert low < 0.0
    assert high > 0.0
    assert abs(low + high) < 1e-6

    # 4. Asymmetric discordance (e.g. n10=15, n01=1, N=64)
    diff, low, high = paired_risk_difference_newcombe(n11=10, n10=15, n01=1, n00=38, total=64, confidence=0.95)
    expected_diff = (15 - 1) / 64
    assert abs(diff - expected_diff) < 1e-6
    assert low > 0.0  # Significantly positive
    assert high > diff
    assert low < diff


def test_analyzer_prospective_validation_gate(tmp_path: Path) -> None:
    """Analyzer main() must fail closed in prospective mode on unapproved manifest or parameter drift."""
    from scripts.analyze_seven_round_results import main as analyzer_main

    # Create dummy results file
    results_file = tmp_path / "results.json"
    dummy_data = {
        "metadata": {
            "prospective": True,
            "dataset_sha256": "fake_dataset_hash",
            "m1": 32,
            "m2": 32,
            "candidate_budget": 1024,
            "solver_timeout": 10.0,
            "global_timeout": 300.0,
            "threads": 1,
            "device": "mps",
        },
        "records_by_method": {
            m: [_build_synthetic_record(f"inst_{i}", success=False, wall_seconds=1.0) for i in range(128)]
            for m in [
                "hierarchical_neural_screened",
                "hierarchical_neural_full",
                "hierarchical_neural_classical",
                "random_sat",
                "sat_only_4obs",
                "sat_only_all_observations",
            ]
        },
    }
    results_file.write_text(json.dumps(dummy_data), encoding="utf-8")

    # Calling with --prospective against a proposed (unauthorized) manifest must raise SystemExit
    manifest_path = Path("docs/seven_round_protocol_manifest.json")
    proposed_manifest_path = tmp_path / "proposed_manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["protocol_status"] = "PROPOSED_NOT_AUTHORIZED"
    proposed_manifest_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(SystemExit, match="(ManifestValidationError|Refusing prospective analysis|PROPOSED_NOT_AUTHORIZED)"):
        analyzer_main([
            str(results_file),
            "--prospective",
            "--protocol-manifest", str(proposed_manifest_path),
        ])
