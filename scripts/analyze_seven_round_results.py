#!/usr/bin/env python3
"""Statistical analysis of seven-round SPECK32/64 recovery comparators.

Performs the complete frozen statistical plan specified in
docs/seven_round_protocol_manifest.json:
1. Primary Estimand: Unconditional fresh-pair-verified recovery probability within 300.0s cutoff,
   with Wilson 95% score confidence intervals.
2. Primary Paired Contrast: Exact two-sided McNemar test comparing
   hierarchical_neural_screened vs sat_only_all_observations under strict information parity (128 observations).
3. Secondary Contrasts with Multiplicity Control:
   Holm-Bonferroni step-down procedure controlling FWER at alpha=0.05 across 3 predeclared contrasts:
   - Contrast 1: hierarchical_neural_screened vs hierarchical_neural_classical
   - Contrast 2: hierarchical_neural_screened vs random_sat
   - Contrast 3: hierarchical_neural_screened vs sat_only_4obs
4. Resource Accounting & Endpoints:
   - Penalized Average Runtime (PAR-2) with 300.0s cutoff (600.0s failure penalty).
   - Wall-clock seconds: median, IQR, mean, min, max.
   - Screening ablation: time savings and candidate efficiency.
   - Orthogonal failure domain distribution (ranking, solver, envelope, system, verification).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from neural_sat_speck.evaluation.metrics import (  # noqa: E402
    calculate_cactus_curve,
    calculate_global_par2,
    exact_mcnemar_test,
    paired_risk_difference_newcombe,
    wilson_score_interval,
)
from neural_sat_speck.protocol.manifest_validator import (  # noqa: E402
    ManifestValidationError,
    validate_manifest_for_prospective_execution,
    verify_record_integrity,
)

EXPECTED_METHODS: frozenset[str] = frozenset({
    "hierarchical_neural_screened",
    "hierarchical_neural_full",
    "hierarchical_neural_classical",
    "random_sat",
    "sat_only_4obs",
    "sat_only_all_observations",
})

PRIMARY_METHOD_A = "hierarchical_neural_screened"
PRIMARY_METHOD_B = "sat_only_all_observations"

SECONDARY_CONTRASTS = [
    {
        "contrast_id": "secondary_1_neural_vs_hybrid",
        "name": "Neural Screened vs Neural-Classical Hybrid",
        "method_a": "hierarchical_neural_screened",
        "method_b": "hierarchical_neural_classical",
    },
    {
        "contrast_id": "secondary_2_neural_vs_random",
        "name": "Neural Screened vs Random Candidate SAT",
        "method_a": "hierarchical_neural_screened",
        "method_b": "random_sat",
    },
    {
        "contrast_id": "secondary_3_neural_vs_sat4obs",
        "name": "Neural Screened vs Formula-Matched SAT (4 obs)",
        "method_a": "hierarchical_neural_screened",
        "method_b": "sat_only_4obs",
    },
]


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def holm_bonferroni_correction(
    p_values: list[float],
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """Apply Holm-Bonferroni step-down procedure to control family-wise error rate.

    Parameters
    ----------
    p_values : list[float]
        Raw p-values for m hypothesis tests.
    alpha : float
        Family-wise significance level (default: 0.05).

    Returns
    -------
    list[dict[str, Any]]
        List of dicts containing original index, raw p-value, adjusted p-value,
        and boolean indicator whether H0 is rejected (adjusted_p <= alpha).
    """
    m = len(p_values)
    if m == 0:
        return []

    # Sort indices by p-value ascending
    sorted_indices = sorted(range(m), key=lambda i: p_values[i])

    adj_p: list[float] = [0.0] * m
    cum_max = 0.0

    for rank, idx in enumerate(sorted_indices):
        multiplier = m - rank
        raw_p = p_values[idx]
        val = min(1.0, multiplier * raw_p)
        cum_max = max(cum_max, val)
        adj_p[idx] = min(1.0, cum_max)

    results = []
    for i in range(m):
        results.append({
            "index": i,
            "raw_p": p_values[i],
            "adjusted_p": adj_p[i],
            "rejected": adj_p[i] <= alpha,
        })
    return results


def compute_paired_contingency(
    successes_a: Sequence[bool],
    successes_b: Sequence[bool],
) -> dict[str, int]:
    """Compute 2x2 contingency table for paired binary outcomes."""
    if len(successes_a) != len(successes_b):
        raise ValueError("Length mismatch between paired outcomes")

    n11 = 0
    n10 = 0
    n01 = 0
    n00 = 0

    for a, b in zip(successes_a, successes_b, strict=True):
        if a and b:
            n11 += 1
        elif a and not b:
            n10 += 1
        elif not a and b:
            n01 += 1
        else:
            n00 += 1

    return {
        "n11": n11,
        "n10": n10,
        "n01": n01,
        "n00": n00,
        "total": len(successes_a),
    }


def paired_risk_difference_ci(
    n10: int,
    n01: int,
    total: int,
    confidence: float = 0.95,
    n11: int = 0,
    n00: int = 0,
) -> tuple[float, float, float]:
    """Compute risk difference (p_A - p_B) and paired score confidence interval (Newcombe Method 10)."""
    if total <= 0:
        return 0.0, 0.0, 0.0
    if n11 + n00 == 0:
        rem = max(0, total - n10 - n01)
        n11 = rem // 2
        n00 = rem - n11
    return paired_risk_difference_newcombe(
        n11=n11, n10=n10, n01=n01, n00=n00, total=total, confidence=confidence
    )




def analyze_records(
    records_by_method: dict[str, list[dict[str, Any]]],
    global_cutoff: float = 300.0,
    alpha: float = 0.05,
    *,
    prospective: bool = False,
) -> dict[str, Any]:
    """Execute complete statistical analysis across comparator methods.

    Enforces:
    1. Exactly the 6 declared primary methods in EXPECTED_METHODS.
    2. Indexing by instance_id with no duplicate instance IDs within any method.
    3. Identical instance_id sets across all 6 methods.
    4. Explicit recovery cross-checking (verify_record_integrity).
    5. Prospective sample size requirement (strictly N=128 if prospective=True).
    6. Exact standard median (statistics.median) and quartiles (inclusive).
    7. Consistent <= alpha hypothesis testing.
    """
    found_methods = set(records_by_method.keys())
    if found_methods != EXPECTED_METHODS:
        diff = EXPECTED_METHODS.symmetric_difference(found_methods)
        raise ValueError(
            f"Comparator methods mismatch. Expected exact primary methods {sorted(EXPECTED_METHODS)}; "
            f"difference: {sorted(diff)}"
        )

    by_method_id: dict[str, dict[str, dict[str, Any]]] = {}
    for m in EXPECTED_METHODS:
        recs = records_by_method[m]
        if not recs:
            raise ValueError(f"Method {m} has 0 records (empty or partial checkpoint)")
        by_id: dict[str, dict[str, Any]] = {}
        for r in recs:
            iid = r.get("instance_id")
            if not iid:
                raise ValueError(f"Method {m} record missing 'instance_id'")
            if iid in by_id:
                raise ValueError(f"Duplicate instance_id {iid!r} in method {m}")
            verify_record_integrity(r)
            by_id[iid] = r
        by_method_id[m] = by_id

    # Verify identical instance_id sets across all methods
    first_method = next(iter(EXPECTED_METHODS))
    common_instance_ids = set(by_method_id[first_method].keys())
    for m in EXPECTED_METHODS:
        m_ids = set(by_method_id[m].keys())
        if m_ids != common_instance_ids:
            diff = common_instance_ids.symmetric_difference(m_ids)
            raise ValueError(
                f"Instance ID set mismatch for method {m}. Difference: {sorted(diff)}"
            )

    sorted_instance_ids = sorted(common_instance_ids)
    n_instances = len(sorted_instance_ids)

    if prospective and n_instances != 128:
        raise ValueError(
            f"Prospective evaluation requires exactly N=128 instances; found {n_instances}"
        )
    if n_instances == 0:
        raise ValueError("No instances found to analyze")

    # 1. Per-Method Descriptive & Recovery Statistics
    method_metrics: dict[str, Any] = {}
    for m in sorted(EXPECTED_METHODS):
        recs = [by_method_id[m][iid] for iid in sorted_instance_ids]
        successes = [verify_record_integrity(r) for r in recs]
        wall_times = [float(r.get("method_wall_seconds", r.get("wall_seconds", 0.0))) for r in recs]
        solver_times = [float(r.get("solver_seconds", 0.0)) for r in recs]
        tuples_attempted = [int(r.get("candidates_attempted", 0)) for r in recs]

        k = sum(successes)
        ci_lower, ci_upper = wilson_score_interval(k, n_instances, confidence=1.0 - alpha)
        par2 = calculate_global_par2(wall_times, successes, global_cutoff=global_cutoff)

        med = statistics.median(wall_times)
        if len(wall_times) >= 2:
            q_vals = statistics.quantiles(wall_times, n=4, method="inclusive")
            q1 = q_vals[0]
            q3 = q_vals[2]
        else:
            q1 = wall_times[0]
            q3 = wall_times[0]
        iqr = q3 - q1

        # Failure domain distribution
        domain_counts: dict[str, int] = {}
        cause_counts: dict[str, int] = {}
        for r in recs:
            dom = r.get("failure_domain", "unknown" if not r.get("success") else "none")
            domain_counts[dom] = domain_counts.get(dom, 0) + 1
            cause = r.get("failure_cause") or r.get("status", "none")
            cause_counts[cause] = cause_counts.get(cause, 0) + 1

        stage_containment: dict[str, Any] | None = None
        if m in ("hierarchical_neural_screened", "hierarchical_neural_full", "hierarchical_neural_classical"):
            s1_count = 0
            s2_cond_count = 0
            has_stage_data = True
            for r in recs:
                st_list = r.get("stages", [])
                if len(st_list) >= 2:
                    s1 = st_list[0]
                    s2 = st_list[1]
                    s1_rank = s1.get("true_round_key_rank")
                    s1_bw = s1.get("beam_width", 32)
                    if s1_rank is not None and s1_rank <= s1_bw:
                        s1_count += 1
                        s2_rank = s2.get("true_round_key_rank")
                        s2_bw = s2.get("beam_width", 32)
                        if s2_rank is not None and s2_rank <= s2_bw:
                            s2_cond_count += 1
                else:
                    has_stage_data = False
                    break
            if has_stage_data:
                stage_containment = {
                    "stage1_top32_containment_count": s1_count,
                    "stage1_top32_containment_rate": round(s1_count / n_instances, 4),
                    "stage2_top32_conditional_containment_count": s2_cond_count,
                    "stage2_top32_conditional_containment_rate": (
                        round(s2_cond_count / s1_count, 4) if s1_count > 0 else 0.0
                    ),
                }

        cactus_data = calculate_cactus_curve(wall_times, successes, max_time=global_cutoff)

        method_metrics[m] = {
            "instances": n_instances,
            "verified_recoveries": k,
            "recovery_rate": round(k / n_instances, 4),
            "wilson_95_ci": [round(ci_lower, 4), round(ci_upper, 4)],
            "par2_seconds": round(par2, 3),
            "wall_seconds": {
                "median": round(med, 3),
                "q1": round(q1, 3),
                "q3": round(q3, 3),
                "iqr": round(iqr, 3),
                "mean": round(sum(wall_times) / n_instances, 3),
                "min": round(min(wall_times), 3),
                "max": round(max(wall_times), 3),
            },
            "solver_seconds_mean": round(sum(solver_times) / n_instances, 3),
            "mean_candidates_attempted": round(sum(tuples_attempted) / n_instances, 1),
            "failure_domain_distribution": domain_counts,
            "failure_cause_distribution": cause_counts,
            "stage_containment": stage_containment,
            "ecdf_data": cactus_data,
        }

    # 2. Primary Paired Contrast (aligned by instance_id)
    succ_a = [verify_record_integrity(by_method_id[PRIMARY_METHOD_A][iid]) for iid in sorted_instance_ids]
    succ_b = [verify_record_integrity(by_method_id[PRIMARY_METHOD_B][iid]) for iid in sorted_instance_ids]
    table = compute_paired_contingency(succ_a, succ_b)
    mcnemar = exact_mcnemar_test(b=table["n10"], c=table["n01"])
    diff, diff_l, diff_u = paired_risk_difference_newcombe(
        table["n11"], table["n10"], table["n01"], table["n00"], n_instances, confidence=1.0 - alpha
    )

    primary_contrast_result = {
        "contrast_id": "primary_contrast_neural_screened_vs_sat_only_all_obs",
        "method_a": PRIMARY_METHOD_A,
        "method_b": PRIMARY_METHOD_B,
        "role": "CONFIRMATORY_PRIMARY",
        "hypothesis": "H0: P(Success_A) == P(Success_B) vs H1: P(Success_A) != P(Success_B)",
        "contingency_table": table,
        "risk_difference": round(diff, 4),
        "risk_difference_95_ci": [round(diff_l, 4), round(diff_u, 4)],
        "exact_mcnemar": mcnemar,
        "significant_at_alpha": bool(mcnemar["p_value"] <= alpha),
    }

    # 3. Secondary Paired Contrasts with Holm-Bonferroni Multiplicity Adjustment
    secondary_results = []
    raw_p_values = []
    applicable_contrasts = []

    for sc in SECONDARY_CONTRASTS:
        ma = sc["method_a"]
        mb = sc["method_b"]
        succ_a_sc = [verify_record_integrity(by_method_id[ma][iid]) for iid in sorted_instance_ids]
        succ_b_sc = [verify_record_integrity(by_method_id[mb][iid]) for iid in sorted_instance_ids]
        tbl = compute_paired_contingency(succ_a_sc, succ_b_sc)
        mc = exact_mcnemar_test(b=tbl["n10"], c=tbl["n01"])
        d, dl, du = paired_risk_difference_newcombe(
            tbl["n11"], tbl["n10"], tbl["n01"], tbl["n00"], n_instances, confidence=1.0 - alpha
        )
        applicable_contrasts.append({
            "contrast_id": sc["contrast_id"],
            "name": sc["name"],
            "role": "EXPLORATORY_SECONDARY",
            "method_a": ma,
            "method_b": mb,
            "contingency_table": tbl,
            "risk_difference": round(d, 4),
            "risk_difference_95_ci": [round(dl, 4), round(du, 4)],
            "mcnemar": mc,
        })
        raw_p_values.append(mc["p_value"])


    if raw_p_values:
        holm_adj = holm_bonferroni_correction(raw_p_values, alpha=alpha)
        for i, c_data in enumerate(applicable_contrasts):
            c_data["raw_p_value"] = round(raw_p_values[i], 6)
            c_data["holm_adjusted_p_value"] = round(holm_adj[i]["adjusted_p"], 6)
            c_data["rejected_h0"] = holm_adj[i]["rejected"]
            secondary_results.append(c_data)

    # 4. Screening Efficiency Ablation (Screened vs Full)
    scr_sm = method_metrics["hierarchical_neural_screened"]
    ful_sm = method_metrics["hierarchical_neural_full"]
    time_diff = ful_sm["wall_seconds"]["mean"] - scr_sm["wall_seconds"]["mean"]
    time_speedup = ful_sm["wall_seconds"]["mean"] / max(1e-4, scr_sm["wall_seconds"]["mean"])
    ablation_result = {
        "screened_recovery_rate": scr_sm["recovery_rate"],
        "full_recovery_rate": ful_sm["recovery_rate"],
        "rate_difference": round(scr_sm["recovery_rate"] - ful_sm["recovery_rate"], 4),
        "screened_mean_wall_seconds": scr_sm["wall_seconds"]["mean"],
        "full_mean_wall_seconds": ful_sm["wall_seconds"]["mean"],
        "mean_time_saved_seconds": round(time_diff, 3),
        "speedup_factor": round(time_speedup, 2),
    }

    return {
        "instances_evaluated": n_instances,
        "global_cutoff_seconds": global_cutoff,
        "family_wise_alpha": alpha,
        "method_metrics": method_metrics,
        "primary_paired_contrast": primary_contrast_result,
        "multiplicity_controlled_secondary_contrasts": secondary_results,
        "screening_efficiency_ablation": ablation_result,
    }


def format_text_report(analysis: dict[str, Any]) -> str:
    """Generate human-readable text summary of analysis results."""
    lines: list[str] = []
    lines.append("=" * 115)
    lines.append(f"{'STATISTICAL ANALYSIS: 7-ROUND SPECK32/64 COMPARATIVE EVALUATION':^115}")
    lines.append("=" * 115)
    lines.append(f"Instances: {analysis['instances_evaluated']} | Global Cutoff: {analysis['global_cutoff_seconds']}s | FWER Alpha: {analysis['family_wise_alpha']}")
    lines.append("-" * 115)

    # Method Performance Table
    lines.append(
        f"{'Method':<32} | {'Success':<8} | {'Rate':<8} | {'95% CI':<14} | {'PAR-2':<9} | {'Med Wall':<9} | {'Mean Tuples':<11}"
    )
    lines.append("-" * 115)
    for m, mm in analysis["method_metrics"].items():
        ci_str = f"[{mm['wilson_95_ci'][0]:.3f}, {mm['wilson_95_ci'][1]:.3f}]"
        lines.append(
            f"{m:<32} | "
            f"{mm['verified_recoveries']:>2}/{mm['instances']:<2}   | "
            f"{mm['recovery_rate']*100:>5.1f}%  | "
            f"{ci_str:<14} | "
            f"{mm['par2_seconds']:>7.1f}s | "
            f"{mm['wall_seconds']['median']:>7.2f}s | "
            f"{mm['mean_candidates_attempted']:>11.1f}"
        )
        if mm.get("stage_containment"):
            sc = mm["stage_containment"]
            s1_str = f"S1 Containment (top-32): {sc['stage1_top32_containment_count']}/{mm['instances']} ({sc['stage1_top32_containment_rate']*100:.1f}%)"
            s2_str = f"S2 Conditional: {sc['stage2_top32_conditional_containment_count']}/{sc['stage1_top32_containment_count']} ({sc['stage2_top32_conditional_containment_rate']*100:.1f}%)"
            lines.append(f"  └─ {s1_str} | {s2_str}")
    lines.append("=" * 115)

    # Primary Contrast
    pc = analysis.get("primary_paired_contrast")
    if pc:
        lines.append("\n--- PRIMARY HYPOTHESIS CONTRAST (Information Parity: 128 Observations) ---")
        lines.append("Confirmatory Status: SOLE CONFIRMATORY PRIMARY CONTRAST (alpha = 0.05)")
        lines.append(f"Method A: {pc['method_a']} vs Method B: {pc['method_b']}")
        ct = pc["contingency_table"]
        lines.append(f"Contingency: N11={ct['n11']}, N10={ct['n10']} (A+ B-), N01={ct['n01']} (A- B+), N00={ct['n00']}")
        diff_str = f"{pc['risk_difference']*100:+.1f}% [{pc['risk_difference_95_ci'][0]*100:.1f}%, {pc['risk_difference_95_ci'][1]*100:.1f}%]"
        lines.append(f"Risk Difference (A - B): {diff_str}")
        lines.append(f"Exact McNemar p-value: {pc['exact_mcnemar']['p_value']:.6f}")
        verdict = "SIGNIFICANT (H0 rejected at alpha=0.05)" if pc['significant_at_alpha'] else "NOT SIGNIFICANT (H0 not rejected)"
        lines.append(f"Statistical Verdict: {verdict}")

    # Secondary Contrasts with Holm Adjustment
    sc_list = analysis.get("multiplicity_controlled_secondary_contrasts", [])
    if sc_list:
        lines.append("\n--- SECONDARY CONTRASTS (Holm-Bonferroni Adjusted FWER alpha=0.05) ---")
        lines.append("Confirmatory Status: EXPLORATORY SECONDARY CONTRASTS (Holm-Bonferroni adjusted FWER alpha = 0.05)")
        lines.append("Note: Study-wide confirmatory significance is not claimed for secondary contrasts.")
        lines.append(
            f"{'Contrast':<42} | {'Risk Diff':<12} | {'Raw p':<10} | {'Holm p':<10} | {'Rejected H0':<11}"
        )
        lines.append("-" * 95)
        for sc in sc_list:
            rd_str = f"{sc['risk_difference']*100:+.1f}%"
            rej_str = "YES (rej)" if sc["rejected_h0"] else "NO"
            lines.append(
                f"{sc['name']:<42} | {rd_str:<12} | {sc['raw_p_value']:<10.6f} | {sc['holm_adjusted_p_value']:<10.6f} | {rej_str:<11}"
            )

    # Screening Ablation
    abl = analysis.get("screening_efficiency_ablation")
    if abl:
        lines.append("\n--- SCREENING EFFICIENCY ABLATION ---")
        lines.append(f"Screened Rate: {abl['screened_recovery_rate']*100:.1f}% vs Full Rate: {abl['full_recovery_rate']*100:.1f}% (Diff: {abl['rate_difference']*100:+.1f}%)")
        lines.append(f"Mean Wall: Screened {abl['screened_mean_wall_seconds']:.2f}s vs Full {abl['full_mean_wall_seconds']:.2f}s (Speedup: {abl['speedup_factor']}x, Saved: {abl['mean_time_saved_seconds']:.2f}s)")

    lines.append("\n--- RESOURCE ACCOUNTING & TIMING NOTE ---")
    lines.append(
        "Resource Accounting Convention: Execution runtime on Apple Silicon (MPS) is evaluated via wall-clock "
        "duration under strict solver isolation and fixed single-threaded execution (torch_num_threads=1, threads=1). "
        "Because GPU kernels execute asynchronously on MPS, CPU process_time does not capture neural inference time; "
        "isolated per-method wall-clock time is the authoritative and standard resource metric."
    )
    lines.append("=" * 115)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_json",
        type=Path,
        help="Input evaluation JSON file path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output analysis JSON file path",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=300.0,
        help="Global cutoff in seconds for PAR-2 calculation (default: 300.0)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level alpha (default: 0.05)",
    )
    parser.add_argument(
        "--prospective",
        action="store_true",
        help="Enforce prospective criteria (strictly N=128 instances)",
    )
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        default=None,
        help="Path to protocol manifest JSON (embeds manifest digest into provenance)",
    )
    args = parser.parse_args(argv)

    if not args.input_json.exists():
        raise SystemExit(f"Input file not found: {args.input_json}")

    with args.input_json.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    records_by_method = data.get("records_by_method")
    if not isinstance(records_by_method, dict):
        raise SystemExit("Input JSON missing valid 'records_by_method' dictionary")

    manifest_path = args.protocol_manifest
    if manifest_path is None:
        default_manifest = REPOSITORY_ROOT / "docs" / "seven_round_protocol_manifest.json"
        if default_manifest.exists():
            manifest_path = default_manifest

    manifest_data = None
    if args.prospective:
        if manifest_path is None or not manifest_path.exists():
            raise SystemExit("Error: Prospective analysis requires an explicit valid --protocol-manifest.")
        try:
            manifest_data = validate_manifest_for_prospective_execution(
                manifest_path,
                base_dir=REPOSITORY_ROOT,
            )
        except ManifestValidationError as err:
            raise SystemExit(f"PROSPECTIVE ANALYSIS REFUSED: {err}") from err

        # Derive cutoff and alpha directly from manifest
        envelope = manifest_data.get("search_and_resource_envelope", {})
        manifest_cutoff = float(envelope.get("global_method_cutoff_seconds", 300.0))
        stat_plan = manifest_data.get("statistical_plan", {})
        manifest_alpha = float(stat_plan.get("primary_paired_contrast", {}).get("alpha", 0.05))

        if not math.isclose(args.cutoff, manifest_cutoff, abs_tol=1e-4):
            raise SystemExit(
                f"Error: CLI --cutoff {args.cutoff} does not match frozen manifest cutoff {manifest_cutoff}"
            )
        if not math.isclose(args.alpha, manifest_alpha, abs_tol=1e-4):
            raise SystemExit(
                f"Error: CLI --alpha {args.alpha} does not match frozen manifest alpha {manifest_alpha}"
            )
        args.cutoff = manifest_cutoff
        args.alpha = manifest_alpha

        meta = data.get("metadata", {})
        if not meta.get("prospective"):
            raise SystemExit(
                "Error: Prospective analysis requires input results with metadata.prospective=True"
            )
        if data.get("evidence_class") != "PROSPECTIVE_EVALUATION":
            raise SystemExit(
                f"Error: Prospective analysis requires evidence_class='PROSPECTIVE_EVALUATION'; got {data.get('evidence_class')!r}"
            )
        if meta.get("device") != "mps":
            raise SystemExit(
                f"Error: Prospective results must use device='mps'; found {meta.get('device')!r}"
            )
        expected_dataset_sha256 = manifest_data.get("protected_dataset", {}).get("sha256")
        if meta.get("dataset_sha256") != expected_dataset_sha256:
            raise SystemExit(
                f"Error: Results dataset_sha256 {meta.get('dataset_sha256')!r} does not match manifest {expected_dataset_sha256!r}"
            )

    try:
        analysis = analyze_records(
            records_by_method,
            global_cutoff=args.cutoff,
            alpha=args.alpha,
            prospective=args.prospective,
        )
    except ValueError as err:
        raise SystemExit(f"Analysis failed validation: {err}") from err

    # Provenance tracking
    input_results_sha256 = _sha256_file(args.input_json)
    manifest_sha256 = _sha256_file(manifest_path) if manifest_path and manifest_path.exists() else None

    analysis["provenance"] = {
        "input_results_path": str(args.input_json),
        "input_results_sha256": input_results_sha256,
        "protocol_manifest_path": str(manifest_path) if manifest_path else None,
        "protocol_manifest_sha256": manifest_sha256,
        "prospective": args.prospective,
    }

    text_report = format_text_report(analysis)
    print(text_report)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(analysis, handle, indent=2)
            handle.write("\n")
        print(f"\nAnalysis saved to: {args.output}")

    return 0



if __name__ == "__main__":
    raise SystemExit(main())
