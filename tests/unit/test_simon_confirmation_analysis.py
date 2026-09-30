"""Unit tests for SIMON confirmation and development ledger analysis runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_simon_confirmation import (
    DIAGNOSTIC_ORACLE,
    PRIMARY_METHODS,
    analyze_all_comparators,
    compute_method_summary,
    extract_record_success,
    generate_markdown_audit_report,
    load_ledger_records,
)


@pytest.fixture
def synthetic_records_by_method() -> dict[str, list[dict]]:
    """Generate minimal synthetic records across 4 instances for all primary methods."""
    n = 4
    methods = list(PRIMARY_METHODS) + [DIAGNOSTIC_ORACLE]
    out: dict[str, list[dict]] = {m: [] for m in methods}

    for i in range(n):
        iid = f"test_inst_{i:03d}"
        for m in methods:
            # Deterministic success pattern:
            # fully_classical succeeds on 3/4
            # neural_hybrid succeeds on 2/4
            # faithful_b1 succeeds on 0/4
            # sat_only_n128 succeeds on 0/4
            if m == "fully_classical_8parent":
                succ = i < 3
            elif m == "neural_classical_hybrid_8parent":
                succ = i < 2
            elif m == DIAGNOSTIC_ORACLE:
                succ = True
            else:
                succ = False

            rec = {
                "instance_id": iid,
                "instance_index": i,
                "method": m,
                "outcome": "verified_recovery" if succ else "queue_exhausted",
                "counts_as_verified_recovery": succ,
                "verified_on_fresh_pairs": succ,
                "complete_online_wall_seconds": 10.0 + i if succ else 100.0,
                "ranking_wall_seconds": 2.0,
                "solving_wall_seconds": 8.0 if succ else 98.0,
                "verification_wall_seconds": 0.001,
                "candidates_tested_count": i + 1 if succ else 50,
                "peak_rss_gib": 0.5,
            }
            out[m].append(rec)
    return out


def test_extract_record_success() -> None:
    rec_true = {
        "counts_as_verified_recovery": True,
        "outcome": "verified_recovery",
        "verified_on_fresh_pairs": True,
        "complete_online_wall_seconds": 12.5,
    }
    assert extract_record_success(rec_true) is True

    rec_false = {
        "counts_as_verified_recovery": False,
        "outcome": "queue_exhausted",
        "verified_on_fresh_pairs": False,
        "complete_online_wall_seconds": 100.0,
    }
    assert extract_record_success(rec_false) is False

    # Flat confirmation record without counts_as_verified_recovery
    rec_conf_succ = {
        "outcome": "verified_recovery",
        "verified_on_fresh_pairs": True,
        "complete_online_wall_seconds": 25.0,
    }
    assert extract_record_success(rec_conf_succ) is True

    # Contradictory record: verified_recovery with verified_on_fresh_pairs=False must be rejected
    rec_conf_contradict = {
        "outcome": "verified_recovery",
        "verified_on_fresh_pairs": False,
        "complete_online_wall_seconds": 25.0,
    }
    with pytest.raises(ValueError, match="Contradictory record"):
        extract_record_success(rec_conf_contradict)

    # Success-on-failure flag must be rejected
    rec_succ_on_fail = {
        "outcome": "queue_exhausted",
        "counts_as_verified_recovery": True,
        "complete_online_wall_seconds": 50.0,
    }
    with pytest.raises(ValueError, match="success-on-failure flag"):
        extract_record_success(rec_succ_on_fail)

    # String booleans (e.g. 'false') must be rejected
    rec_str_bool = {
        "outcome": "queue_exhausted",
        "counts_as_verified_recovery": "false",
        "complete_online_wall_seconds": 50.0,
    }
    with pytest.raises(ValueError, match="must be a genuine boolean"):
        extract_record_success(rec_str_bool)

    # Fabricated zero time on recovery must be rejected
    rec_zero_time = {
        "outcome": "verified_recovery",
        "verified_on_fresh_pairs": True,
        "complete_online_wall_seconds": 0.0,
    }
    with pytest.raises(ValueError, match="Fabricated zero time"):
        extract_record_success(rec_zero_time)


def test_compute_method_summary(synthetic_records_by_method: dict[str, list[dict]]) -> None:
    recs = synthetic_records_by_method["fully_classical_8parent"]
    summary = compute_method_summary("fully_classical_8parent", recs, global_cutoff=300.0)

    assert summary["instances_evaluated"] == 4
    assert summary["verified_recoveries"] == 3
    assert summary["verified_recovery_rate"] == 0.75
    assert len(summary["wilson_95_ci"]) == 2
    assert summary["unconditional_wall_seconds"]["median"] > 0
    assert summary["success_conditional_wall_seconds"]["count"] == 3
    assert summary["candidates_tested"]["mean"] > 0


def test_analyze_all_comparators_complete(synthetic_records_by_method: dict[str, list[dict]]) -> None:
    analysis = analyze_all_comparators(
        records_by_method=synthetic_records_by_method,
        global_cutoff=300.0,
        alpha=0.05,
    )

    assert analysis["instances_analyzed"] == 4
    assert len(analysis["primary_contrasts_holm"]) == 2

    # Check Primary Contrast 1 (Classical vs SAT-128): Classical=3, SAT-128=0
    p1 = analysis["primary_contrasts_holm"][0]
    assert p1["method_a"] == "fully_classical_8parent"
    assert p1["method_b"] == "sat_only_n128"
    assert p1["contingency"]["n10"] == 3
    assert p1["contingency"]["n01"] == 0
    assert p1["contingency"]["n11"] == 0
    assert p1["risk_difference"] == 0.75

    # Check Primary Contrast 2 (Hybrid vs SAT-128): Hybrid=2, SAT-128=0
    p2 = analysis["primary_contrasts_holm"][1]
    assert p2["method_a"] == "neural_classical_hybrid_8parent"
    assert p2["contingency"]["n10"] == 2
    assert p2["risk_difference"] == 0.50

    # Diagnostic oracle separated
    assert analysis["diagnostic_oracle"] is not None
    assert analysis["diagnostic_oracle"]["verified_recoveries"] == 4


def test_analyze_instance_mismatch_fails_closed(
    synthetic_records_by_method: dict[str, list[dict]],
) -> None:
    # Remove one record from sat_only_n128
    synthetic_records_by_method["sat_only_n128"].pop()
    with pytest.raises(ValueError, match="Instance ID set mismatch"):
        analyze_all_comparators(synthetic_records_by_method)


def test_load_ledger_records_formats(tmp_path: Path, synthetic_records_by_method: dict[str, list[dict]]) -> None:
    # 1. Test records_by_method format
    p_by_method = tmp_path / "by_method.json"
    p_by_method.write_text(json.dumps({
        "metadata": {"schema": "test_v1"},
        "records_by_method": synthetic_records_by_method,
    }))
    ctx1, recs1 = load_ledger_records(p_by_method)
    assert len(recs1) == len(synthetic_records_by_method)
    assert ctx1["ledger_sha256"] != ""

    # 2. Test flat records format
    flat_list = []
    for rec_list in synthetic_records_by_method.values():
        flat_list.extend(rec_list)
    p_flat = tmp_path / "flat.json"
    p_flat.write_text(json.dumps({
        "execution_identity": {"schema": "test_v2"},
        "records": flat_list,
    }))
    ctx2, recs2 = load_ledger_records(p_flat)
    assert len(recs2) == len(synthetic_records_by_method)

    # 3. Test missing primary method raises ValueError
    bad_recs = dict(synthetic_records_by_method)
    del bad_recs["sat_only_n128"]
    p_bad = tmp_path / "bad.json"
    p_bad.write_text(json.dumps({"records_by_method": bad_recs}))
    with pytest.raises(ValueError, match="missing required arms"):
        load_ledger_records(p_bad)


def test_generate_markdown_audit_report(synthetic_records_by_method: dict[str, list[dict]]) -> None:
    analysis = analyze_all_comparators(synthetic_records_by_method)
    context = {
        "evidence_class": "TEST_SYNTHETIC",
        "ledger_path": "/tmp/test.json",
        "ledger_sha256": "abcdef123456",
    }
    md = generate_markdown_audit_report(context, analysis)
    assert "# SIMON32/64 R=10 Statistical Analysis Audit Report" in md
    assert "fully_classical_8parent" in md
    assert "neural_classical_hybrid_8parent" in md
    assert "faithful_b1_hybrid" in md
    assert "Primary Paired Contrasts" in md
    assert "success-conditional" in md.lower()


def test_missing_oracle_fails_closed(synthetic_records_by_method: dict[str, list[dict]], tmp_path: Path) -> None:
    bad_recs = dict(synthetic_records_by_method)
    del bad_recs[DIAGNOSTIC_ORACLE]
    p_bad = tmp_path / "missing_oracle.json"
    p_bad.write_text(json.dumps({"records_by_method": bad_recs}))
    with pytest.raises(ValueError, match="missing required arms"):
        load_ledger_records(p_bad)


def test_corrupt_sidecar_fails_closed(synthetic_records_by_method: dict[str, list[dict]], tmp_path: Path) -> None:
    p_ledger = tmp_path / "ledger.json"
    p_ledger.write_text(json.dumps({"records_by_method": synthetic_records_by_method}))
    p_sidecar = tmp_path / "ledger.json.sha256"
    p_sidecar.write_text("0000000000000000000000000000000000000000000000000000000000000000  ledger.json\n")
    with pytest.raises(ValueError, match="sidecar mismatch"):
        load_ledger_records(p_ledger)


def test_drift_rejected_in_main(synthetic_records_by_method: dict[str, list[dict]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p_ledger = tmp_path / "ledger.json"
    p_ledger.write_text(json.dumps({"records_by_method": synthetic_records_by_method}))

    from scripts.analyze_simon_confirmation import main
    # Drift cutoff from 300.0 to 100.0 without --allow-drift
    monkeypatch.setattr("sys.argv", ["analyze_simon_confirmation.py", "--ledger", str(p_ledger), "--global-cutoff", "100.0"])
    with pytest.raises(ValueError, match="Cutoff drift is prohibited"):
        main()

    # Drift alpha from 0.05 to 0.01 without --allow-drift
    monkeypatch.setattr("sys.argv", ["analyze_simon_confirmation.py", "--ledger", str(p_ledger), "--alpha", "0.01"])
    with pytest.raises(ValueError, match="Alpha drift is prohibited"):
        main()
