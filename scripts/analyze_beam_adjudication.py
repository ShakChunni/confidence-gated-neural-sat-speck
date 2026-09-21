#!/usr/bin/env python3
"""Statistical Analysis and Predeclared Selection for Phase 11 Beam Adjudication.

Evidence Class: DEVELOPMENT (Decision D-030).

Implements:
- Outcome validation and alignment between Configuration A and B.
- Unconditional verified-recovery count, rate, and Wilson 95% confidence intervals.
- Paired 2x2 contingency table (n11, n10, n01, n00).
- Exact two-sided McNemar hypothesis test on discordant pairs.
- PAR-2 evaluation at 300.0s cutoff.
- Median and IQR method wall-clock time.
- Detailed component timing decomposition (preprocessing, ranking, solving, rebuilding, verification).
- Structural failure decomposition (Stage 1 failure, Stage 2 failure, reachability failure, solver timeout).
- Predeclared prospective configuration selection rule (recovery -> PAR-2 -> median time -> reference A).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from neural_sat_speck.evaluation.metrics import (  # noqa: E402
    calculate_global_par2,
    exact_mcnemar_test,
    wilson_score_interval,
)

DEFAULT_INPUT: Final[Path] = (
    REPO_ROOT
    / "results"
    / "phase11_scaling_diagnostics"
    / "beam_adjudication_results.json"
)
DEFAULT_OUTPUT: Final[Path] = (
    REPO_ROOT
    / "results"
    / "phase11_scaling_diagnostics"
    / "beam_adjudication_analysis.json"
)
GLOBAL_CUTOFF_SECONDS: Final[float] = 300.0


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_config_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        raise ValueError("Cannot compute metrics for empty records list.")

    recoveries = sum(1 for r in records if r["verified_recovery"])
    rate = recoveries / n
    ci_low, ci_high = wilson_score_interval(recoveries, n, confidence=0.95)

    wall_times = [float(r["total_method_wall_seconds"]) for r in records]
    success_bools = [bool(r["verified_recovery"]) for r in records]
    par2 = calculate_global_par2(wall_times, success_bools, global_cutoff=GLOBAL_CUTOFF_SECONDS)

    med_wall = float(statistics.median(wall_times))
    if n >= 4:
        quantiles = statistics.quantiles(wall_times, n=4, method="inclusive")
        iqr_wall = float(quantiles[2] - quantiles[0])
        q25 = float(quantiles[0])
        q75 = float(quantiles[2])
    else:
        iqr_wall = 0.0
        q25 = med_wall
        q75 = med_wall

    prep_times = [float(r["preprocessing_seconds"]) for r in records]
    rank_times = [float(r["ranking_screening_sorting_seconds"]) for r in records]
    solver_times = [float(r["solver_seconds"]) for r in records]
    rebuilding_times = [float(r["rebuilding_seconds"]) for r in records]
    verify_times = [float(r["verification_seconds"]) for r in records]
    tuples_attempted = [int(r["candidates_attempted"]) for r in records]

    # Stage containment metrics
    s1_containment = sum(1 for r in records if r["k6_in_beam"])
    s2_conditional = sum(
        1 for r in records if r["k6_in_beam"] and r["k5_in_beam"]
    )
    s2_cond_rate = (s2_conditional / s1_containment) if s1_containment > 0 else 0.0
    tuple_reachable = sum(1 for r in records if r["true_tuple_reachable"])

    # Failure cause breakdown
    failure_causes: dict[str, int] = {}
    failure_categories: dict[str, int] = {}
    for r in records:
        cause = str(r["failure_cause"]) if r["failure_cause"] is not None else "none"
        cat = str(r["failure_category"])
        failure_causes[cause] = failure_causes.get(cause, 0) + 1
        failure_categories[cat] = failure_categories.get(cat, 0) + 1

    return {
        "n_instances": n,
        "verified_recoveries": recoveries,
        "recovery_rate": round(rate, 4),
        "wilson_95_ci": [round(ci_low, 4), round(ci_high, 4)],
        "par2_seconds": round(par2, 3),
        "median_wall_seconds": round(med_wall, 3),
        "iqr_wall_seconds": round(iqr_wall, 3),
        "q25_wall_seconds": round(q25, 3),
        "q75_wall_seconds": round(q75, 3),
        "mean_wall_seconds": round(float(statistics.mean(wall_times)), 3),
        "timing_components": {
            "mean_preprocessing_seconds": round(float(statistics.mean(prep_times)), 4),
            "median_preprocessing_seconds": round(float(statistics.median(prep_times)), 4),
            "mean_ranking_screening_seconds": round(float(statistics.mean(rank_times)), 3),
            "median_ranking_screening_seconds": round(float(statistics.median(rank_times)), 3),
            "mean_solver_seconds": round(float(statistics.mean(solver_times)), 3),
            "median_solver_seconds": round(float(statistics.median(solver_times)), 3),
            "mean_rebuilding_seconds": round(float(statistics.mean(rebuilding_times)), 4),
            "mean_verification_seconds": round(float(statistics.mean(verify_times)), 4),
        },
        "search_diagnostics": {
            "stage1_containment_count": s1_containment,
            "stage1_containment_rate": round(s1_containment / n, 4),
            "stage2_conditional_containment_count": s2_conditional,
            "stage2_conditional_containment_rate": round(s2_cond_rate, 4),
            "joint_containment_count": s2_conditional,
            "joint_containment_rate": round(s2_conditional / n, 4),
            "true_tuple_reachable_count": tuple_reachable,
            "true_tuple_reachable_rate": round(tuple_reachable / n, 4),
            "mean_candidates_attempted": round(float(statistics.mean(tuples_attempted)), 2),
            "median_candidates_attempted": round(float(statistics.median(tuples_attempted)), 1),
        },
        "failure_causes": failure_causes,
        "failure_categories": failure_categories,
    }


def analyze_adjudication_results(data: dict[str, Any]) -> dict[str, Any]:
    records_by_config = data.get("records_by_configuration", {})
    if "A" not in records_by_config or "B" not in records_by_config:
        raise ValueError("Results missing 'A' or 'B' configurations.")

    records_a = records_by_config["A"]
    records_b = records_by_config["B"]

    if len(records_a) != len(records_b):
        raise ValueError(
            f"Mismatched record counts: A has {len(records_a)}, B has {len(records_b)}"
        )

    # Index and align by instance_id
    a_by_id = {r["instance_id"]: r for r in records_a}
    b_by_id = {r["instance_id"]: r for r in records_b}

    if set(a_by_id.keys()) != set(b_by_id.keys()):
        raise ValueError("Instance ID sets between A and B do not match!")

    instance_ids = sorted(a_by_id.keys(), key=lambda x: a_by_id[x]["instance_index"])
    aligned_a = [a_by_id[iid] for iid in instance_ids]
    aligned_b = [b_by_id[iid] for iid in instance_ids]

    metrics_a = compute_config_metrics(aligned_a)
    metrics_b = compute_config_metrics(aligned_b)

    # Paired 2x2 contingency table
    # n11: both succeed
    # n10: A succeeds, B fails
    # n01: A fails, B succeeds
    # n00: both fail
    n11 = n10 = n01 = n00 = 0
    paired_wall_diffs: list[float] = []

    for ra, rb in zip(aligned_a, aligned_b, strict=True):
        succ_a = bool(ra["verified_recovery"])
        succ_b = bool(rb["verified_recovery"])
        if succ_a and succ_b:
            n11 += 1
        elif succ_a and not succ_b:
            n10 += 1
        elif not succ_a and succ_b:
            n01 += 1
        else:
            n00 += 1
        paired_wall_diffs.append(
            float(rb["total_method_wall_seconds"]) - float(ra["total_method_wall_seconds"])
        )

    # Exact McNemar test
    mcnemar_res = exact_mcnemar_test(b=n10, c=n01)

    # Apply Predeclared Selection Hierarchy
    # 1. Higher unconditional verified-recovery count within 300 seconds
    # 2. If tied, lower PAR-2
    # 3. If still tied, lower median total method wall time
    # 4. If still tied, retain A (M1=32, M2=32) as simpler reference
    rec_a = metrics_a["verified_recoveries"]
    rec_b = metrics_b["verified_recoveries"]
    par2_a = metrics_a["par2_seconds"]
    par2_b = metrics_b["par2_seconds"]
    med_a = metrics_a["median_wall_seconds"]
    med_b = metrics_b["median_wall_seconds"]

    if rec_a > rec_b:
        selected = "A"
        reason = f"Configuration A achieved higher verified recovery count ({rec_a} vs {rec_b})."
        rule_tier = "1_higher_recovery_count"
    elif rec_b > rec_a:
        selected = "B"
        reason = f"Configuration B achieved higher verified recovery count ({rec_b} vs {rec_a})."
        rule_tier = "1_higher_recovery_count"
    else:
        # Tied recoveries
        if par2_a < par2_b:
            selected = "A"
            reason = f"Recoveries tied ({rec_a}/{rec_b}); Configuration A achieved lower PAR-2 ({par2_a:.2f}s vs {par2_b:.2f}s)."
            rule_tier = "2_lower_par2"
        elif par2_b < par2_a:
            selected = "B"
            reason = f"Recoveries tied ({rec_a}/{rec_b}); Configuration B achieved lower PAR-2 ({par2_b:.2f}s vs {par2_a:.2f}s)."
            rule_tier = "2_lower_par2"
        else:
            # Tied PAR-2
            if med_a < med_b:
                selected = "A"
                reason = f"Recoveries and PAR-2 tied; Configuration A achieved lower median wall time ({med_a:.2f}s vs {med_b:.2f}s)."
                rule_tier = "3_lower_median_wall_time"
            elif med_b < med_a:
                selected = "B"
                reason = f"Recoveries and PAR-2 tied; Configuration B achieved lower median wall time ({med_b:.2f}s vs {med_a:.2f}s)."
                rule_tier = "3_lower_median_wall_time"
            else:
                selected = "A"
                reason = "All performance metrics tied; Configuration A retained as the simpler predeclared reference."
                rule_tier = "4_tie_break_reference_A"

    return {
        "analysis_timestamp": datetime.now(UTC).isoformat(),
        "evidence_class": "DEVELOPMENT",
        "sample_size": len(aligned_a),
        "configuration_A": metrics_a,
        "configuration_B": metrics_b,
        "paired_analysis": {
            "contingency_table": {
                "both_succeed_n11": n11,
                "A_only_succeeds_n10": n10,
                "B_only_succeeds_n01": n01,
                "both_fail_n00": n00,
            },
            "exact_mcnemar": mcnemar_res,
            "paired_wall_time_difference_B_minus_A": {
                "median_diff_seconds": round(float(statistics.median(paired_wall_diffs)), 3),
                "mean_diff_seconds": round(float(statistics.mean(paired_wall_diffs)), 3),
            },
        },
        "selection_decision": {
            "selected_configuration": selected,
            "rule_tier": rule_tier,
            "rationale": reason,
            "status": "DEVELOPMENT_METHOD_SELECTION_COMPLETE",
        },
    }


def print_summary_table(analysis: dict[str, Any]) -> None:
    ma = analysis["configuration_A"]
    mb = analysis["configuration_B"]
    pt = analysis["paired_analysis"]["contingency_table"]
    mc = analysis["paired_analysis"]["exact_mcnemar"]
    sel = analysis["selection_decision"]

    print("\n" + "=" * 90)
    print(f"{'PHASE 11 BEAM CONFIGURATION ADJUDICATION SUMMARY (N=64)':^90}")
    print("=" * 90)
    print(f"{'Metric':<36} | {'Config A (M1=32, M2=32)':<24} | {'Config B (M1=64, M2=16)':<24}")
    print("-" * 90)
    ci_a_str = f"[{ma['wilson_95_ci'][0]:.3f}, {ma['wilson_95_ci'][1]:.3f}]"
    ci_b_str = f"[{mb['wilson_95_ci'][0]:.3f}, {mb['wilson_95_ci'][1]:.3f}]"
    print(f"{'Verified Recoveries':<36} | {ma['verified_recoveries']:>2}/{ma['n_instances']} ({ma['recovery_rate']*100:5.1f}%)        | {mb['verified_recoveries']:>2}/{mb['n_instances']} ({mb['recovery_rate']*100:5.1f}%)")
    print(f"{'Wilson 95% CI':<36} | {ci_a_str:<24} | {ci_b_str:<24}")
    print(f"{'PAR-2 (seconds)':<36} | {ma['par2_seconds']:>10.2f}s              | {mb['par2_seconds']:>10.2f}s")
    print(f"{'Median Wall Time (IQR)':<36} | {ma['median_wall_seconds']:>6.2f}s (IQR {ma['iqr_wall_seconds']:5.2f}s)    | {mb['median_wall_seconds']:>6.2f}s (IQR {mb['iqr_wall_seconds']:5.2f}s)")
    print(f"{'Mean Wall Time':<36} | {ma['mean_wall_seconds']:>10.2f}s              | {mb['mean_wall_seconds']:>10.2f}s")
    print("-" * 90)
    print(f"{'Stage 1 Containment (k6 in M1)':<36} | {ma['search_diagnostics']['stage1_containment_count']:>2}/{ma['n_instances']} ({ma['search_diagnostics']['stage1_containment_rate']*100:5.1f}%)        | {mb['search_diagnostics']['stage1_containment_count']:>2}/{mb['n_instances']} ({mb['search_diagnostics']['stage1_containment_rate']*100:5.1f}%)")
    print(f"{'Stage 2 Cond Containment (k5 in M2)':<36} | {ma['search_diagnostics']['stage2_conditional_containment_count']:>2}/{ma['search_diagnostics']['stage1_containment_count']} ({ma['search_diagnostics']['stage2_conditional_containment_rate']*100:5.1f}%)        | {mb['search_diagnostics']['stage2_conditional_containment_count']:>2}/{mb['search_diagnostics']['stage1_containment_count']} ({mb['search_diagnostics']['stage2_conditional_containment_rate']*100:5.1f}%)")
    print(f"{'True Tuple Reachable (in 1,024)':<36} | {ma['search_diagnostics']['true_tuple_reachable_count']:>2}/{ma['n_instances']} ({ma['search_diagnostics']['true_tuple_reachable_rate']*100:5.1f}%)        | {mb['search_diagnostics']['true_tuple_reachable_count']:>2}/{mb['n_instances']} ({mb['search_diagnostics']['true_tuple_reachable_rate']*100:5.1f}%)")
    print(f"{'Mean Tuples Attempted':<36} | {ma['search_diagnostics']['mean_candidates_attempted']:>10.1f}               | {mb['search_diagnostics']['mean_candidates_attempted']:>10.1f}")
    print("-" * 90)
    print(f"{'Mean Preprocessing Time':<36} | {ma['timing_components']['mean_preprocessing_seconds']:>10.4f}s              | {mb['timing_components']['mean_preprocessing_seconds']:>10.4f}s")
    print(f"{'Mean Ranking & Screening Time':<36} | {ma['timing_components']['mean_ranking_screening_seconds']:>10.3f}s              | {mb['timing_components']['mean_ranking_screening_seconds']:>10.3f}s")
    print(f"{'Mean Solver Time':<36} | {ma['timing_components']['mean_solver_seconds']:>10.3f}s              | {mb['timing_components']['mean_solver_seconds']:>10.3f}s")
    print(f"{'Mean Rebuilding Time':<36} | {ma['timing_components']['mean_rebuilding_seconds']:>10.4f}s              | {mb['timing_components']['mean_rebuilding_seconds']:>10.4f}s")
    print(f"{'Mean Verification Time':<36} | {ma['timing_components']['mean_verification_seconds']:>10.4f}s              | {mb['timing_components']['mean_verification_seconds']:>10.4f}s")
    print("=" * 90)

    print("\n" + "-" * 45)
    print("PAIRED 2x2 CONTINGENCY TABLE (A vs B):")
    print("-" * 45)
    print(f"  Both Recovered (n11):      {pt['both_succeed_n11']:>2}")
    print(f"  A Recovered, B Failed (n10): {pt['A_only_succeeds_n10']:>2}")
    print(f"  B Recovered, A Failed (n01): {pt['B_only_succeeds_n01']:>2}")
    print(f"  Both Failed (n00):         {pt['both_fail_n00']:>2}")
    print(f"  Exact McNemar p-value:     {mc['p_value']:.4f} (discordant = {mc['n_discordant']})")
    print("-" * 45)

    print("\n" + "=" * 90)
    print(f"SELECTION RULE VERDICT: CONFIGURATION {sel['selected_configuration']}")
    print(f"Tier: {sel['rule_tier']}")
    print(f"Rationale: {sel['rationale']}")
    print("=" * 90 + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to beam adjudication results JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to output analysis JSON",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    data = json.loads(args.input.read_text(encoding="utf-8"))
    analysis = analyze_adjudication_results(data)
    analysis["input_results_file"] = str(args.input)
    analysis["input_results_sha256"] = _sha256_file(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")

    print_summary_table(analysis)
    print(f"Detailed analysis saved to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
