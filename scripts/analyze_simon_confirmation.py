#!/usr/bin/env python3
"""Statistical analysis of SIMON32/64 R=10 recovery comparators.

Executes the complete frozen statistical plan specified in
docs/SIMON_UNSEEN_CONFIRMATION_PROTOCOL_PROPOSED.md and audits development records:
1. Primary Estimand: Unconditional fresh-pair-verified recovery probability within 300.0s cutoff,
   with Wilson 95% score confidence intervals.
2. Two Primary Paired Contrasts with Family-Wise Multiplicity Control:
   Holm-Bonferroni step-down procedure controlling FWER at alpha=0.05 across:
   - Primary Contrast 1: fully_classical_8parent vs sat_only_n128
   - Primary Contrast 2: neural_classical_hybrid_8parent vs sat_only_n128
   Reporting exact paired contingency tables, signed recovery differences,
   pointwise Newcombe Method 10 95% paired confidence intervals, raw and Holm-adjusted p-values.
3. Secondary Exploratory Contrasts:
   Unadjusted paired contrasts comparing hybrid vs classical, hybrid vs 1-parent neural,
   faithful B1 vs baselines, and width-matched random controls.
4. Resource Accounting & Endpoints:
   - Penalized Average Runtime (PAR-2) with 300.0s cutoff (600.0s penalty for all failures).
   - Unconditional online wall seconds: mean, median, IQR, min, max.
   - Success-conditional online wall seconds explicitly labelled.
   - Component wall times: ranking, solving, verification.
   - Failure taxonomy and candidate attempt statistics.
   - Diagnostic oracle separated from primary comparators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from neural_sat_speck.analysis.paired_stats import (  # noqa: E402
    compute_2x2_contingency,
    exact_mcnemar_test,
    holm_bonferroni_adjustment,
    newcombe_method10_paired_ci,
)
from neural_sat_speck.evaluation.metrics import (  # noqa: E402
    calculate_global_par2,
    wilson_score_interval,
)

PRIMARY_METHODS: tuple[str, ...] = (
    "neural_classical_hybrid_8parent",
    "fully_classical_8parent",
    "fully_neural_1parent",
    "faithful_b1_hybrid",
    "random_control_128",
    "random_control_1024",
    "sat_only_n4",
    "sat_only_n128",
)

DIAGNOSTIC_ORACLE: str = "oracle_2key_n4"
REQUIRED_ARMS: tuple[str, ...] = PRIMARY_METHODS + (DIAGNOSTIC_ORACLE,)

PRIMARY_CONTRASTS: list[dict[str, str]] = [
    {
        "id": "primary_1_classical_vs_sat128",
        "name": "Fully Classical (8-parent) vs SAT-only (128 obs)",
        "method_a": "fully_classical_8parent",
        "method_b": "sat_only_n128",
    },
    {
        "id": "primary_2_hybrid_vs_sat128",
        "name": "Neural-Classical Hybrid (8-parent) vs SAT-only (128 obs)",
        "method_a": "neural_classical_hybrid_8parent",
        "method_b": "sat_only_n128",
    },
]

SECONDARY_CONTRASTS: list[dict[str, str]] = [
    {
        "id": "secondary_1_classical_vs_hybrid",
        "name": "Fully Classical vs Neural-Classical Hybrid",
        "method_a": "fully_classical_8parent",
        "method_b": "neural_classical_hybrid_8parent",
    },
    {
        "id": "secondary_2_hybrid_vs_neural1parent",
        "name": "Neural-Classical Hybrid vs Fully Neural (1-parent)",
        "method_a": "neural_classical_hybrid_8parent",
        "method_b": "fully_neural_1parent",
    },
    {
        "id": "secondary_3_b1_vs_sat128",
        "name": "Faithful B1 Hybrid vs SAT-only (128 obs)",
        "method_a": "faithful_b1_hybrid",
        "method_b": "sat_only_n128",
    },
    {
        "id": "secondary_4_b1_vs_classical",
        "name": "Faithful B1 Hybrid vs Fully Classical",
        "method_a": "faithful_b1_hybrid",
        "method_b": "fully_classical_8parent",
    },
    {
        "id": "secondary_5_b1_vs_hybrid",
        "name": "Faithful B1 Hybrid vs Neural-Classical Hybrid",
        "method_a": "faithful_b1_hybrid",
        "method_b": "neural_classical_hybrid_8parent",
    },
    {
        "id": "secondary_6_hybrid_vs_random1024",
        "name": "Neural-Classical Hybrid vs Random Control (1,024 tuples)",
        "method_a": "neural_classical_hybrid_8parent",
        "method_b": "random_control_1024",
    },
    {
        "id": "secondary_7_classical_vs_random1024",
        "name": "Fully Classical vs Random Control (1,024 tuples)",
        "method_a": "fully_classical_8parent",
        "method_b": "random_control_1024",
    },
    {
        "id": "secondary_8_neural1parent_vs_random128",
        "name": "Fully Neural (1-parent) vs Random Control (128 tuples)",
        "method_a": "fully_neural_1parent",
        "method_b": "random_control_128",
    },
    {
        "id": "secondary_9_sat4obs_vs_sat128obs",
        "name": "SAT-only (4 obs) vs SAT-only (128 obs)",
        "method_a": "sat_only_n4",
        "method_b": "sat_only_n128",
    },
]


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_record_invariants(record: dict[str, Any]) -> None:
    """Validate invariants for a single ledger record.

    Fails closed on:
    - Non-boolean flags (rejecting 'false', 'true', int, etc.)
    - Contradictory flags (success-on-failure or failure-on-success)
    - Missing or non-finite wall time
    - Fabricated zero time on verified recovery
    """
    iid = record.get("instance_id")
    m = record.get("method")
    outcome = record.get("outcome")
    if not outcome:
        raise ValueError(f"Record {iid}:{m} missing 'outcome'")

    # Genuine boolean flag validation
    for flag in ("counts_as_verified_recovery", "verified_on_fresh_pairs"):
        if flag in record:
            val = record[flag]
            if not isinstance(val, bool):
                raise ValueError(
                    f"Record {iid}:{m} field '{flag}' must be a genuine boolean, got {type(val).__name__}: {val!r}"
                )

    counts_as = record.get("counts_as_verified_recovery")
    verified_fresh = record.get("verified_on_fresh_pairs")

    if outcome == "verified_recovery":
        if verified_fresh is not True:
            raise ValueError(
                f"Contradictory record {iid}:{m}: outcome is 'verified_recovery' but verified_on_fresh_pairs is missing or False"
            )
        if counts_as is False:
            raise ValueError(
                f"Contradictory record {iid}:{m}: outcome is 'verified_recovery' but counts_as_verified_recovery is False"
            )
    else:
        # Failure outcome
        if counts_as is True:
            raise ValueError(
                f"Contradictory record {iid}:{m}: success-on-failure flag! Outcome is {outcome!r} but counts_as_verified_recovery is True"
            )
        if verified_fresh is True:
            raise ValueError(
                f"Contradictory record {iid}:{m}: outcome is {outcome!r} but verified_on_fresh_pairs is True"
            )

    # Time checks
    wall_raw = record.get("complete_online_wall_seconds", record.get("online_wall_seconds"))
    if wall_raw is None:
        raise ValueError(f"Record {iid}:{m} missing online wall time")
    try:
        wall_f = float(wall_raw)
    except (TypeError, ValueError):
        raise ValueError(f"Record {iid}:{m} invalid online wall time: {wall_raw!r}")

    import math
    if not math.isfinite(wall_f) or wall_f < 0.0:
        raise ValueError(f"Record {iid}:{m} wall time must be non-negative and finite, got {wall_f}")

    if outcome == "verified_recovery" and wall_f == 0.0:
        raise ValueError(f"Fabricated zero time: verified_recovery with 0.0s wall time for {iid}:{m}")


def load_ledger_records(ledger_path: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Load and index ledger records from JSON.

    Supports both development schema (records_by_method) and confirmation schema (flat records).
    Validates SHA-256 sidecar if present and requires all 9 arms (8 primary methods + diagnostic oracle).
    """
    if not ledger_path.exists():
        raise FileNotFoundError(f"Ledger file not found: {ledger_path}")

    # Validate sidecar if present
    sidecar_path = ledger_path.with_name(f"{ledger_path.name}.sha256")
    if sidecar_path.exists():
        expected_hash = sidecar_path.read_text(encoding="utf-8").strip().split()[0]
        actual_hash = _sha256_file(ledger_path)
        if actual_hash != expected_hash:
            raise ValueError(f"Ledger SHA-256 sidecar mismatch: expected {expected_hash}, got {actual_hash}")

    with ledger_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("metadata") or data.get("execution_identity") or {}
    evidence_class = data.get("evidence_class", "UNKNOWN")

    records_by_method: dict[str, list[dict[str, Any]]] = {}

    if "records_by_method" in data:
        # Development ledger format
        records_by_method = data["records_by_method"]
    elif "records" in data:
        # Flat records format
        for r in data["records"]:
            m = r.get("method")
            if not m:
                continue
            records_by_method.setdefault(m, []).append(r)
    else:
        raise ValueError("Ledger contains neither 'records_by_method' nor 'records'")

    # Validate that all required arms exist (8 primary + 1 diagnostic oracle)
    missing_arms = [m for m in REQUIRED_ARMS if m not in records_by_method]
    if missing_arms:
        raise ValueError(f"Ledger missing required arms: {missing_arms}")

    # Validate every record
    for m in REQUIRED_ARMS:
        for r in records_by_method[m]:
            validate_record_invariants(r)

    context = {
        "metadata": meta,
        "evidence_class": evidence_class,
        "ledger_sha256": _sha256_file(ledger_path),
        "ledger_path": str(ledger_path.resolve()),
    }
    return context, records_by_method


def extract_record_success(record: dict[str, Any]) -> bool:
    """Determine verified recovery status from record.

    Must count as recovery iff recovery succeeded and passed fresh-block verification.
    """
    validate_record_invariants(record)
    if "counts_as_verified_recovery" in record:
        return record["counts_as_verified_recovery"]
    return (
        record.get("outcome") == "verified_recovery"
        and record.get("verified_on_fresh_pairs") is True
    )


def compute_method_summary(
    method: str,
    records: list[dict[str, Any]],
    global_cutoff: float = 300.0,
) -> dict[str, Any]:
    """Compute comprehensive performance and timing summary for one method."""
    n_instances = len(records)
    if n_instances == 0:
        raise ValueError(f"No records for method {method}")

    successes = [extract_record_success(r) for r in records]
    wall_times = [
        float(r.get("complete_online_wall_seconds", r.get("online_wall_seconds", 0.0)))
        for r in records
    ]
    def components(field):
        values = [r.get(field) for r in records]
        # Unknown legacy telemetry must never be rendered as measured zero work.
        if any(v is None for v in values):
            return None, None
        return statistics.mean(values), statistics.median(values)
    rank_mean, rank_median = components("ranking_wall_seconds")
    solve_mean, solve_median = components("solving_wall_seconds")
    verif_mean, verif_median = components("verification_wall_seconds")
    candidates_tested = [int(r.get("candidates_tested_count", 0)) for r in records]

    k = sum(successes)
    rate = k / float(n_instances)
    ci_lower, ci_upper = wilson_score_interval(k, n_instances, confidence=0.95)
    par2 = calculate_global_par2(wall_times, successes, global_cutoff=global_cutoff)

    # Unconditional wall time statistics
    med_time = statistics.median(wall_times)
    mean_time = statistics.mean(wall_times)
    if len(wall_times) >= 2:
        q_vals = statistics.quantiles(wall_times, n=4, method="inclusive")
        iqr_time = q_vals[2] - q_vals[0]
    else:
        iqr_time = 0.0

    # Success-conditional wall time statistics
    succ_times = [t for t, succ in zip(wall_times, successes, strict=True) if succ]
    if succ_times:
        succ_mean = statistics.mean(succ_times)
        succ_median = statistics.median(succ_times)
        succ_min = min(succ_times)
        succ_max = max(succ_times)
    else:
        succ_mean = None
        succ_median = None
        succ_min = None
        succ_max = None

    # Outcome breakdown / failure taxonomy
    outcomes: dict[str, int] = {}
    for r in records:
        out = r.get("outcome", "unknown")
        outcomes[out] = outcomes.get(out, 0) + 1

    rss_vals = [float(r.get("peak_rss_gib", 0.0)) for r in records if "peak_rss_gib" in r]
    max_rss = max(rss_vals) if rss_vals else 0.0
    mean_rss = statistics.mean(rss_vals) if rss_vals else 0.0

    return {
        "method": method,
        "instances_evaluated": n_instances,
        "verified_recoveries": k,
        "verified_recovery_rate": rate,
        "wilson_95_ci": [float(ci_lower), float(ci_upper)],
        "par2_score": par2,
        "unconditional_wall_seconds": {
            "mean": mean_time,
            "median": med_time,
            "iqr": iqr_time,
            "min": min(wall_times),
            "max": max(wall_times),
        },
        "success_conditional_wall_seconds": {
            "count": len(succ_times),
            "mean": succ_mean,
            "median": succ_median,
            "min": succ_min,
            "max": succ_max,
            "label": "[SUCCESS-CONDITIONAL: DO NOT USE FOR UNCONDITIONAL SPEED CLAIMS]",
        },
        "mean_component_seconds": {
            "ranking": rank_mean,
            "solving": solve_mean,
            "verification": verif_mean,
        },
        "median_component_seconds": {
            "ranking": rank_median,
            "solving": solve_median,
            "verification": verif_median,
        },
        "pair_usage_counts": {str(value): sum(r.get("full_pairs_used") == value for r in records) for value in {r.get("full_pairs_used") for r in records}},
        "success_candidate_indices": [next((i + 1 for i, a in enumerate(r.get("candidate_attempts", [])) if a.get("status") == "sat"), None) for r, success in zip(records, successes, strict=True) if success],
        "true_tuple_ranks": [{"instance_id": r["instance_id"], "rank": r.get("audit_true_tuple_rank"), "queue_recorded": bool(r.get("full_queue"))} for r in records],
        "candidates_tested": {
            "mean": statistics.mean(candidates_tested),
            "median": statistics.median(candidates_tested),
            "min": min(candidates_tested),
            "max": max(candidates_tested),
        },
        "peak_rss_gib": {
            "max": max_rss,
            "mean": mean_rss,
        },
        "outcome_breakdown": outcomes,
    }


def analyze_all_comparators(
    records_by_method: dict[str, list[dict[str, Any]]],
    global_cutoff: float = 300.0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Execute complete audited statistical analysis on indexed records."""
    # 1. Align instances across all required arms (8 primary + 1 diagnostic oracle)
    by_method_instance: dict[str, dict[str, dict[str, Any]]] = {}
    for m in REQUIRED_ARMS:
        if m not in records_by_method:
            raise ValueError(f"Ledger missing required arm: {m}")
        recs = records_by_method[m]
        m_dict: dict[str, dict[str, Any]] = {}
        for r in recs:
            iid = r.get("instance_id")
            if not iid:
                raise ValueError(f"Record in method {m} missing instance_id")
            if iid in m_dict:
                raise ValueError(f"Duplicate instance_id {iid} in method {m}")
            m_dict[iid] = r
        by_method_instance[m] = m_dict

    # Check identical instance ID sets across all 9 arms (no partial ID sets)
    ref_method = REQUIRED_ARMS[0]
    common_instance_ids = sorted(by_method_instance[ref_method].keys())
    for m in REQUIRED_ARMS[1:]:
        m_ids = sorted(by_method_instance[m].keys())
        if m_ids != common_instance_ids:
            diff = set(common_instance_ids).symmetric_difference(set(m_ids))
            raise ValueError(
                f"Instance ID set mismatch between {ref_method} and {m}. Difference: {diff}"
            )

    n_instances = len(common_instance_ids)

    # 2. Per-method performance summaries
    method_summaries: dict[str, Any] = {}
    for m in PRIMARY_METHODS:
        sorted_recs = [by_method_instance[m][iid] for iid in common_instance_ids]
        method_summaries[m] = compute_method_summary(m, sorted_recs, global_cutoff=global_cutoff)

    # 3. Diagnostic Oracle (required arm)
    sorted_oracle = [by_method_instance[DIAGNOSTIC_ORACLE][iid] for iid in common_instance_ids]
    oracle_summary = compute_method_summary(
        DIAGNOSTIC_ORACLE, sorted_oracle, global_cutoff=global_cutoff
    )
    oracle_summary["role"] = "DIAGNOSTIC_ORACLE (NOT A PRIMARY COMPARATOR)"

    # 4. Primary Contrasts with Holm-Bonferroni FWER Control
    primary_results: list[dict[str, Any]] = []
    primary_raw_p: list[float] = []

    for contrast in PRIMARY_CONTRASTS:
        m_a = contrast["method_a"]
        m_b = contrast["method_b"]
        succ_a = [extract_record_success(by_method_instance[m_a][iid]) for iid in common_instance_ids]
        succ_b = [extract_record_success(by_method_instance[m_b][iid]) for iid in common_instance_ids]

        ctg = compute_2x2_contingency(succ_a, succ_b)
        mcnemar_res = exact_mcnemar_test(ctg.n10, ctg.n01)
        d, lower, upper = newcombe_method10_paired_ci(
            ctg.n11, ctg.n10, ctg.n01, ctg.n00, ctg.total, confidence=0.95
        )

        primary_raw_p.append(mcnemar_res["p_value"])
        primary_results.append({
            "contrast_id": contrast["id"],
            "contrast_name": contrast["name"],
            "method_a": m_a,
            "method_b": m_b,
            "contingency": ctg.to_dict(),
            "risk_difference": float(d),
            "newcombe_95_ci": [float(lower), float(upper)],
            "mcnemar_raw_p": float(mcnemar_res["p_value"]),
        })

    # Apply Holm-Bonferroni correction to primary family
    holm_adj = holm_bonferroni_adjustment(primary_raw_p, alpha=alpha)
    for res, adj in zip(primary_results, holm_adj, strict=True):
        res["rank"] = adj["rank"]
        res["step_down_threshold"] = adj["threshold"]
        res["mcnemar_adjusted_p"] = adj["adjusted_p"]
        res["rejected_at_alpha_05"] = adj["rejected"]

    # 5. Secondary Exploratory Contrasts (Unadjusted)
    secondary_results: list[dict[str, Any]] = []
    for contrast in SECONDARY_CONTRASTS:
        m_a = contrast["method_a"]
        m_b = contrast["method_b"]
        succ_a = [extract_record_success(by_method_instance[m_a][iid]) for iid in common_instance_ids]
        succ_b = [extract_record_success(by_method_instance[m_b][iid]) for iid in common_instance_ids]

        ctg = compute_2x2_contingency(succ_a, succ_b)
        mcnemar_res = exact_mcnemar_test(ctg.n10, ctg.n01)
        d, lower, upper = newcombe_method10_paired_ci(
            ctg.n11, ctg.n10, ctg.n01, ctg.n00, ctg.total, confidence=0.95
        )

        secondary_results.append({
            "contrast_id": contrast["id"],
            "contrast_name": contrast["name"],
            "method_a": m_a,
            "method_b": m_b,
            "role": "EXPLORATORY_DESCRIPTIVE (NO PROSPECTIVE CONFIRMATION)",
            "contingency": ctg.to_dict(),
            "risk_difference": float(d),
            "newcombe_95_ci": [float(lower), float(upper)],
            "mcnemar_raw_p": float(mcnemar_res["p_value"]),
            "significant_at_unadjusted_05": mcnemar_res["significant_at_05"],
        })

    return {
        "instances_analyzed": n_instances,
        "instance_ids": common_instance_ids,
        "global_cutoff_seconds": global_cutoff,
        "nominal_alpha": alpha,
        "method_summaries": method_summaries,
        "diagnostic_oracle": oracle_summary,
        "primary_contrasts_holm": primary_results,
        "secondary_contrasts_exploratory": secondary_results,
    }


def generate_markdown_audit_report(
    context: dict[str, Any],
    analysis: dict[str, Any],
) -> str:
    """Generate comprehensive GitHub-style markdown audit document."""
    def seconds(value):
        return "Unknown" if value is None else f"{value:.2f}s"
    lines: list[str] = []
    n = analysis["instances_analyzed"]
    lines.append(f"# SIMON32/64 R=10 Statistical Analysis Audit Report (N={n})")
    lines.append("")
    lines.append(f"**Evidence Class:** `{context.get('evidence_class', 'UNKNOWN')}`")
    lines.append(f"**Source Ledger:** `{context.get('ledger_path', 'N/A')}`")
    lines.append(f"**Ledger SHA-256:** `{context.get('ledger_sha256', 'N/A')}`")
    lines.append(f"**Global Cutoff:** `{analysis['global_cutoff_seconds']:.1f}s` | **Family-Wise Alpha:** `{analysis['nominal_alpha']}`")
    lines.append("")

    if "DEVELOPMENT" in context.get("evidence_class", ""):
        lines.append("> [!NOTE]")
        lines.append("> **Provenance and Supersession:**")
        lines.append("> This versioned derived audit supplements the original independent audit and V1/V2 statistical audits; earlier documents and raw ledgers remain unchanged.")
        lines.append("> - The raw development ledger (`results/simon_development/simon_development_comparators_32keys.json`) and previous audit files are preserved as immutable history.")
        lines.append("> - Pointwise paired intervals are updated with the published Newcombe (1998) Method 10 continuity-corrected association coefficient (phi).")
        lines.append("> - The paired classical vs Bao hybrid interval becomes `[-0.0335, 0.2806]` (exact `[-0.03353845, 0.28064412]`, p=0.21875); primary contrasts against SAT-only remain unchanged.")
        lines.append("> - The two ranking timeouts in `fully_neural_1parent` are explicitly reported in the failure breakdown so that every row sums to N=32.")
        lines.append("> - Diagnostic oracle is labeled `oracle_2key_n4` (true-subkey-conditioned diagnostic; not an upper bound).")
        lines.append("> - Legacy component telemetry is unavailable for valid aggregate cost claims; raw recorded values are preserved unchanged. Complete online wall times remain usable.")
        lines.append("> - No causal ARX/AND-RX or mandatory-guidance claim is justified by these development observations.")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Section 1: Per-Method Summary Table
    lines.append("## 1. Per-Method Recovery and Resource Endpoints")
    lines.append("")
    lines.append("| Method | Role | Verified | Recovery Rate | Wilson 95% CI | PAR-2 (300s) | Median Wall | Mean Wall | Mean Ranking Time | Mean Solve Time |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    role_labels: dict[str, str] = {
        "fully_classical_8parent": "Classical",
        "neural_classical_hybrid_8parent": "Neural Hybrid",
        "fully_neural_1parent": "2-Stage Neural",
        "faithful_b1_hybrid": "B1 Hybrid",
        "random_control_128": "Random (128)",
        "random_control_1024": "Random (1024)",
        "sat_only_n4": "SAT-only (4 obs)",
        "sat_only_n128": "SAT-only (128 obs)",
    }

    for m in PRIMARY_METHODS:
        s = analysis["method_summaries"][m]
        ci = s["wilson_95_ci"]
        lines.append(
            f"| `{m}` | {role_labels.get(m, 'Comparator')} | "
            f"**{s['verified_recoveries']}/{n}** | {s['verified_recovery_rate']:.1%} | "
            f"[{ci[0]:.4f}, {ci[1]:.4f}] | {s['par2_score']:.2f}s | "
            f"{s['unconditional_wall_seconds']['median']:.2f}s | "
            f"{s['unconditional_wall_seconds']['mean']:.2f}s | "
            f"{seconds(s['mean_component_seconds']['ranking'])} | "
            f"{seconds(s['mean_component_seconds']['solving'])} |"
        )

    if analysis.get("diagnostic_oracle"):
        so = analysis["diagnostic_oracle"]
        cio = so["wilson_95_ci"]
        lines.append(
            f"| `{DIAGNOSTIC_ORACLE}` | *Diagnostic Oracle* | "
            f"*{so['verified_recoveries']}/{n}* | *{so['verified_recovery_rate']:.1%}* | "
            f"*[{cio[0]:.4f}, {cio[1]:.4f}]* | *{so['par2_score']:.2f}s* | "
            f"*{so['unconditional_wall_seconds']['median']:.2f}s* | "
            f"*{so['unconditional_wall_seconds']['mean']:.2f}s* | "
            f"*{seconds(so['mean_component_seconds']['ranking'])}* | "
            f"*{seconds(so['mean_component_seconds']['solving'])}* |"
        )
    lines.append("")
    lines.append("> [!NOTE]")
    lines.append("> All B1 outcomes are preserved regardless of their direction. The denominator for all rates is the total instance count.")
    if "DEVELOPMENT" in context.get("evidence_class", ""):
        lines.append(">")
        lines.append("> **Retrospective Development Telemetry Note:**")
        lines.append("> In the historical development ledger, SAT-only runs killed at the 300s global deadline recorded 0.0s solving time due to interrupted component accounting; full online wall time (300.01s) is preserved.")
    lines.append("")

    # Section 2: Primary Contrasts Table
    lines.append("## 2. Primary Paired Contrasts (FWER Controlled at Alpha = 0.05)")
    lines.append("")
    lines.append("Multiplicity adjustment: Holm-Bonferroni step-down across the two predeclared primary tests.")
    lines.append("Pointwise intervals: Newcombe Method 10 paired difference 95% confidence intervals.")
    lines.append("")
    lines.append("| Contrast | Method A | Method B | (1,1) | (1,0) | (0,1) | (0,0) | Risk Diff | Pointwise 95% CI | Raw p | Holm Adj p | Decision |")
    lines.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for p in analysis["primary_contrasts_holm"]:
        c = p["contingency"]
        ci = p["newcombe_95_ci"]
        dec = "**REJECT H0**" if p["rejected_at_alpha_05"] else "FAIL TO REJECT"
        lines.append(
            f"| {p['contrast_name']} | `{p['method_a']}` | `{p['method_b']}` | "
            f"{c['n11']} | {c['n10']} | {c['n01']} | {c['n00']} | "
            f"{p['risk_difference']:+.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | "
            f"{p['mcnemar_raw_p']:.4e} | {p['mcnemar_adjusted_p']:.4e} | {dec} |"
        )
    lines.append("")

    # Section 3: Secondary Contrasts Table
    lines.append("## 3. Secondary Paired Contrasts (Exploratory / Descriptive)")
    lines.append("")
    lines.append("> [!WARNING]")
    lines.append("> Secondary contrasts are descriptive complete-policy comparisons. They do not constitute prospective confirmatory claims.")
    lines.append("")
    lines.append("| Contrast | Method A | Method B | (1,1) | (1,0) | (0,1) | (0,0) | Risk Diff | Pointwise 95% CI | Raw McNemar p | Unadj p < 0.05 |")
    lines.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for s in analysis["secondary_contrasts_exploratory"]:
        c = s["contingency"]
        ci = s["newcombe_95_ci"]
        sig = "Yes" if s["significant_at_unadjusted_05"] else "No"
        lines.append(
            f"| {s['contrast_name']} | `{s['method_a']}` | `{s['method_b']}` | "
            f"{c['n11']} | {c['n10']} | {c['n01']} | {c['n00']} | "
            f"{s['risk_difference']:+.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | "
            f"{s['mcnemar_raw_p']:.5f} | {sig} |"
        )
    lines.append("")

    # Section 4: Detailed Failure Taxonomy (every row sums to N)
    lines.append("## 4. Failure Breakdown by Method")
    lines.append("")
    lines.append("| Method | Verified Success | Queue Exhausted | Unresolved SAT | Global Deadline | Ranking Timeout | Other / Integrity | Total |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for m in list(PRIMARY_METHODS) + [DIAGNOSTIC_ORACLE]:
        if m in PRIMARY_METHODS:
            s = analysis["method_summaries"][m]
        else:
            s = analysis["diagnostic_oracle"]
        ob = s["outcome_breakdown"]
        v_succ = ob.get("verified_recovery", 0)
        q_exh = ob.get("queue_exhausted", 0)
        unres = ob.get("candidate_unresolved", 0) + ob.get("solver_timeout", 0)
        g_dead = ob.get("global_deadline", 0)
        r_time = ob.get("ranking_timeout", 0)
        other = ob.get("integrity_error", 0) + ob.get("error", 0) + ob.get("unknown", 0)
        tot = v_succ + q_exh + unres + g_dead + r_time + other
        if tot < n:
            other += (n - tot)
            tot = n
        lines.append(
            f"| `{m}` | {v_succ} | {q_exh} | {unres} | {g_dead} | {r_time} | {other} | {tot} |"
        )
    lines.append("")

    # Section 5: Success-Conditional Timing (Labelled)
    lines.append("## 5. Success-Conditional Wall Time (Descriptive)")
    lines.append("")
    lines.append("> [!CAUTION]")
    lines.append("> Success-conditional timings select only instances where keys were recovered. They cannot be used for unconditional speed or efficiency claims.")
    lines.append("")
    lines.append("| Method | Recovered Count | Mean Success Time | Median Success Time | Min Success Time | Max Success Time |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for m in PRIMARY_METHODS:
        sc = analysis["method_summaries"][m]["success_conditional_wall_seconds"]
        cnt = sc["count"]
        if cnt > 0:
            lines.append(
                f"| `{m}` | {cnt} | {sc['mean']:.2f}s | {sc['median']:.2f}s | {sc['min']:.2f}s | {sc['max']:.2f}s |"
            )
        else:
            lines.append(f"| `{m}` | 0 | N/A | N/A | N/A | N/A |")
    lines.append("")
    return "\n".join(lines)



def validate_confirmation_analysis_inputs(args, context, records_by_method):
    from neural_sat_speck.simon.attack_instance import load_simon_attack_instances
    from neural_sat_speck.simon.confirmation_guard import (
        ALL_CONFIRMATION_METHODS,
        compute_execution_identity,
        file_sha256,
        validate_and_filter_resume,
        validate_confirmation_manifest,
        verify_disjoint_family_certificate,
    )
    from neural_sat_speck.simon.confirmation_profile import code_identity
    if args.allow_drift or args.global_cutoff != 300.0 or args.alpha != 0.05:
        raise ValueError("Confirmation analysis cannot override the frozen statistical plan")
    if not args.manifest or not args.dataset or not args.certificate:
        raise ValueError("Confirmation analysis requires the frozen manifest, dataset and certificate")
    manifest = validate_confirmation_manifest(args.manifest)
    if manifest.get("execution_pins", {}).get("source_sha256") != code_identity():
        raise ValueError("Confirmation analysis source has drifted")
    instances = load_simon_attack_instances(args.dataset)
    verify_disjoint_family_certificate(manifest, instances, args.certificate)
    identity = compute_execution_identity(manifest, file_sha256(args.dataset))
    cells, pending = validate_and_filter_resume(args.ledger, identity, instances, ALL_CONFIRMATION_METHODS, resume_flag=True)
    if pending or len(cells) != manifest["sample_size_specification"]["selected_instances"] * 9:
        raise ValueError("Confirmation ledger is incomplete: the full selected N x 9 grid is required")
    from neural_sat_speck.simon.key_schedule import expand_key
    for inst in instances:
        truth = list(expand_key(inst.master_key, rounds=10)[-2:])
        for record in (cells[(inst.instance_id, m)] for m in ALL_CONFIRMATION_METHODS):
            queue = record.get("full_queue", [])
            rank = next((i+1 for i, item in enumerate(queue) if item == truth), None)
            # Keep absent true tuples distinct from unavailable queues.
            record["audit_true_tuple_rank"] = rank if queue else None
            target = next(r for r in records_by_method[record["method"]] if r["instance_id"] == inst.instance_id)
            target["audit_true_tuple_rank"] = record["audit_true_tuple_rank"]
    context.update(manifest_sha256=file_sha256(args.manifest), dataset_sha256=file_sha256(args.dataset), certificate_sha256=file_sha256(args.certificate))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run reproducible statistical analysis and audit on SIMON recovery ledgers."
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        required=True,
        help="Path to input ledger JSON (development or confirmation).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path to manifest JSON to validate parameters against ledger.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional path to dataset JSON to validate instance IDs against ledger.",
    )
    parser.add_argument("--certificate", type=Path, default=None, help="Required disjoint-family certificate for confirmation analysis.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to output full analysis as JSON.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Optional path to output markdown audit document.",
    )
    parser.add_argument(
        "--global-cutoff",
        type=float,
        default=300.0,
        help="Global wall-clock deadline in seconds (default: 300.0).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Nominal family-wise significance level (default: 0.05).",
    )
    parser.add_argument(
        "--allow-drift",
        action="store_true",
        help="Explicitly permit non-frozen alpha or cutoff parameters.",
    )
    parser.add_argument(
        "--audit-development",
        action="store_true",
        help="Run development regression audit and write to docs/SIMON_DEVELOPMENT_RECOMPUTED_AUDIT_V3.md.",
    )

    args = parser.parse_args()

    # Enforce frozen protocol parameters against silent drift
    if not args.allow_drift:
        if args.global_cutoff != 300.0:
            raise ValueError(
                f"Frozen global cutoff is 300.0s; got {args.global_cutoff}s. "
                "Cutoff drift is prohibited without --allow-drift."
            )
        if args.alpha != 0.05:
            raise ValueError(
                f"Frozen family-wise alpha is 0.05; got {args.alpha}. "
                "Alpha drift is prohibited without --allow-drift."
            )

    print(f"[*] Loading ledger from: {args.ledger}")
    context, records_by_method = load_ledger_records(args.ledger)

    if context["evidence_class"] == "CONFIRMATION":
        validate_confirmation_analysis_inputs(args, context, records_by_method)
    print(f"[*] Analyzing {len(records_by_method)} methods...")
    analysis = analyze_all_comparators(
        records_by_method=records_by_method,
        global_cutoff=args.global_cutoff,
        alpha=args.alpha,
    )

    # Validate against manifest if provided
    if args.manifest:
        if not args.manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {args.manifest}")
        manifest_data = json.loads(args.manifest.read_text(encoding="utf-8"))
        if "rounds" in manifest_data and manifest_data["rounds"] != 10:
            raise ValueError(f"Manifest rounds must be 10, got {manifest_data['rounds']}")
        if "pair_count" in manifest_data and manifest_data["pair_count"] != 64:
            raise ValueError(f"Manifest pair_count must be 64, got {manifest_data['pair_count']}")

    # Validate against dataset if provided
    if args.dataset:
        if not args.dataset.exists():
            raise FileNotFoundError(f"Dataset not found: {args.dataset}")
        dataset_data = json.loads(args.dataset.read_text(encoding="utf-8"))
        dataset_instances = dataset_data if isinstance(dataset_data, list) else dataset_data.get("instances", [])
        dataset_iids = {inst["instance_id"] for inst in dataset_instances if "instance_id" in inst}
        ledger_iids = set(analysis["instance_ids"])
        if dataset_iids != ledger_iids:
            diff = dataset_iids.symmetric_difference(ledger_iids)
            raise ValueError(f"Dataset instance IDs do not match ledger instance IDs: {diff}")

    if "DEVELOPMENT" in context["evidence_class"]:
        # Development was run with the earlier overlapping/incomplete components.
        for group in records_by_method.values():
            for record in group:
                for field in ("ranking_wall_seconds", "solving_wall_seconds", "verification_wall_seconds"):
                    record[field] = None
        analysis = analyze_all_comparators(records_by_method, global_cutoff=args.global_cutoff, alpha=args.alpha)
    md_report = generate_markdown_audit_report(context, analysis)

    if args.audit_development:
        audit_md_path = REPOSITORY_ROOT / "docs" / "SIMON_DEVELOPMENT_RECOMPUTED_AUDIT_V3.md"
        audit_json_path = REPOSITORY_ROOT / "results" / "simon_development" / "simon_development_recomputed_audit_v3.json"
        if audit_md_path.exists() or audit_json_path.exists():
            raise ValueError("Refusing to overwrite a versioned development audit")
        audit_md_path.write_text(md_report, encoding="utf-8")
        audit_json_path.parent.mkdir(parents=True, exist_ok=True)
        audit_json_path.write_text(json.dumps({"context": context, "analysis": analysis}, indent=2), encoding="utf-8")
        print(f"[+] Recomputed development audit V3 written to {audit_md_path} and {audit_json_path}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({"context": context, "analysis": analysis}, indent=2), encoding="utf-8")
        print(f"[+] Analysis JSON persisted to: {args.output_json}")

    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(md_report, encoding="utf-8")
        print(f"[+] Analysis Markdown persisted to: {args.output_md}")

    # Print summary to stdout
    print("\n" + "=" * 78)
    print("SIMON32/64 R=10 STATISTICAL ANALYSIS SUMMARY")
    print("=" * 78)
    for m in PRIMARY_METHODS:
        s = analysis["method_summaries"][m]
        ci = s["wilson_95_ci"]
        print(
            f"{m:32s}: {s['verified_recoveries']:2d}/{analysis['instances_analyzed']:2d} "
            f"({s['verified_recovery_rate']:5.1%}) [95% CI: {ci[0]:.3f}..{ci[1]:.3f}] | "
            f"PAR-2: {s['par2_score']:6.2f}s | Med: {s['unconditional_wall_seconds']['median']:6.2f}s"
        )
    if analysis.get("diagnostic_oracle"):
        so = analysis["diagnostic_oracle"]
        cio = so["wilson_95_ci"]
        print(
            f"{DIAGNOSTIC_ORACLE:32s}: {so['verified_recoveries']:2d}/{analysis['instances_analyzed']:2d} "
            f"({so['verified_recovery_rate']:5.1%}) [95% CI: {cio[0]:.3f}..{cio[1]:.3f}] | "
            f"PAR-2: {so['par2_score']:6.2f}s | Med: {so['unconditional_wall_seconds']['median']:6.2f}s (DIAGNOSTIC)"
        )
    print("-" * 78)
    print("PRIMARY CONTRASTS (Holm-Bonferroni FWER alpha=0.05):")
    for p in analysis["primary_contrasts_holm"]:
        ci = p["newcombe_95_ci"]
        dec = "REJECT H0" if p["rejected_at_alpha_05"] else "FAIL TO REJECT"
        print(
            f"- {p['contrast_name']}:\n"
            f"  Diff = {p['risk_difference']:+.4f} (95% CI [{ci[0]:.4f}, {ci[1]:.4f}])\n"
            f"  Raw p = {p['mcnemar_raw_p']:.4e} | Holm Adj p = {p['mcnemar_adjusted_p']:.4e} -> {dec}"
        )
    print("=" * 78 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
