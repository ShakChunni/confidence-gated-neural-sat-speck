"""Audit and verification functions for experimental evaluation records.

This module provides fail-closed verification of Phase 7 evaluation records:
1. Validates record schemas, integrity, timing components, and verification status;
2. Reproduces summary metrics and exact success rates from atomic JSON records;
3. Computes global PAR-2 under explicit wall-clock cutoffs;
4. Computes paired McNemar contingency tables and Wilson score intervals.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .metrics import (
    calculate_global_par2,
    exact_mcnemar_test,
    wilson_score_interval,
)

METHODS: tuple[str, ...] = (
    "sat_only",
    "random_sat",
    "b0_sat",
    "b1_sat",
    "gated_b1_fallback_sat",
)

EXPECTED_SCHEMA = "neural_sat_speck.phase7_final_evaluation_result.v1"


@dataclass(frozen=True)
class AuditedMethodSummary:
    """Audited performance metrics for one evaluation method."""

    method: str
    instance_count: int
    success_count: int
    success_rate: float
    wilson_95_ci: tuple[float, float]
    mean_candidates_attempted: float
    median_candidates_attempted: float
    mean_online_seconds: float
    median_online_seconds: float
    mean_rank_seconds: float
    mean_sat_seconds: float
    mean_ver_seconds: float
    mean_conf_seconds: float
    timeout_count: int
    budget_exhausted_count: int
    par2_sensitivity: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "instance_count": self.instance_count,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "wilson_95_ci": list(self.wilson_95_ci),
            "mean_candidates_attempted": self.mean_candidates_attempted,
            "median_candidates_attempted": self.median_candidates_attempted,
            "mean_online_seconds": self.mean_online_seconds,
            "median_online_seconds": self.median_online_seconds,
            "mean_rank_seconds": self.mean_rank_seconds,
            "mean_sat_seconds": self.mean_sat_seconds,
            "mean_ver_seconds": self.mean_ver_seconds,
            "mean_conf_seconds": self.mean_conf_seconds,
            "timeout_count": self.timeout_count,
            "budget_exhausted_count": self.budget_exhausted_count,
            "par2_sensitivity": self.par2_sensitivity,
        }


def audit_phase7_records(
    records_dir: Path,
    expected_count: int = 64,
    cutoffs: tuple[float, ...] = (80.0, 100.0, 160.0, 200.0, 300.0),
) -> dict[str, Any]:
    """Audit and reproduce Phase 7 experimental evaluation records with fail-closed checks.

    Args:
        records_dir: Path to directory containing instance_*.json files.
        expected_count: Expected number of instance files (default 64).
        cutoffs: Global wall-clock cutoffs for PAR-2 sensitivity analysis.

    Returns:
        Dictionary containing audited method summaries and paired McNemar statistics.

    Raises:
        ValueError: If any integrity constraint or invariant fails.
    """
    if not records_dir.is_dir():
        raise ValueError(f"records directory does not exist: {records_dir}")

    record_files = sorted(records_dir.glob("instance_*.json"))
    if len(record_files) != expected_count:
        raise ValueError(
            f"expected {expected_count} instance files, found {len(record_files)} in {records_dir}"
        )

    records: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    seen_ids: set[str] = set()

    for path in record_files:
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)

        # Validate schema
        if rec.get("schema") != EXPECTED_SCHEMA:
            raise ValueError(f"invalid schema {rec.get('schema')} in {path.name}")

        idx = rec.get("instance_index")
        if idx is None or idx in seen_indices or not (0 <= idx < expected_count):
            raise ValueError(f"duplicate or invalid instance_index {idx} in {path.name}")
        seen_indices.add(idx)

        inst_id = rec.get("instance_id")
        if not inst_id or inst_id in seen_ids:
            raise ValueError(f"duplicate or missing instance_id {inst_id} in {path.name}")
        seen_ids.add(inst_id)

        true_key = rec.get("true_final_round_subkey")
        if true_key is None or not (0 <= true_key <= 0xFFFF):
            raise ValueError(f"invalid true_final_round_subkey {true_key} in {path.name}")

        methods_dict = rec.get("methods", {})
        for m in METHODS:
            if m not in methods_dict:
                raise ValueError(f"method {m} missing in {path.name}")
            m_res = methods_dict[m]

            # Check timing summation invariant: t_online ≈ t_rank + t_conf + t_sat + t_ver
            t_online = m_res.get("t_online", 0.0)
            t_rank = m_res.get("t_rank", 0.0)
            t_conf = m_res.get("t_conf", 0.0)
            t_sat = m_res.get("t_sat", 0.0)
            t_ver = m_res.get("t_ver", 0.0)

            sum_times = t_rank + t_conf + t_sat + t_ver
            if not math.isclose(t_online, sum_times, abs_tol=1e-4):
                raise ValueError(
                    f"timing mismatch in {path.name} for {m}: t_online={t_online} vs sum={sum_times}"
                )

            # Check verification consistency
            succ = m_res.get("success", False)
            status = m_res.get("status")
            winning = m_res.get("winning_candidate")

            if succ:
                if status != "recovered":
                    raise ValueError(f"success True but status {status} in {path.name} for {m}")
                if winning != true_key:
                    raise ValueError(
                        f"success True but winning {winning} != true_key {true_key} in {path.name}"
                    )
            else:
                if status not in ("solver_timeout", "budget_exhausted", "inconclusive"):
                    raise ValueError(f"success False with invalid status {status} in {path.name}")

        records.append(rec)

    # Sort records by instance_index
    records.sort(key=lambda r: r["instance_index"])

    # Compute audited summaries for each method
    audited_methods: dict[str, Any] = {}
    for m in METHODS:
        successes = [r["methods"][m]["success"] for r in records]
        runtimes = [r["methods"][m]["t_online"] for r in records]
        attempts = [r["methods"][m]["candidates_attempted"] for r in records]
        t_ranks = [r["methods"][m]["t_rank"] for r in records]
        t_sats = [r["methods"][m]["t_sat"] for r in records]
        t_vers = [r["methods"][m]["t_ver"] for r in records]
        t_confs = [r["methods"][m]["t_conf"] for r in records]

        succ_count = sum(1 for s in successes if s)
        succ_rate = succ_count / expected_count
        wilson_ci = wilson_score_interval(succ_count, expected_count, confidence=0.95)

        timeouts = sum(1 for r in records if r["methods"][m]["status"] == "solver_timeout")
        exhausted = sum(1 for r in records if r["methods"][m]["status"] == "budget_exhausted")

        # Compute PAR-2 sensitivity table
        par2_dict: dict[str, float] = {}
        for c in cutoffs:
            c_key = f"par2_at_{int(c)}s"
            par2_val = calculate_global_par2(runtimes, successes, global_cutoff=c)
            par2_dict[c_key] = par2_val

        # Mean and median calculations
        med_online = float(sorted(runtimes)[len(runtimes) // 2]) if runtimes else 0.0
        sorted_attempts = sorted(attempts)
        med_attempts = (
            float(sorted_attempts[len(sorted_attempts) // 2])
            if len(sorted_attempts) % 2 == 1
            else float(sorted_attempts[len(sorted_attempts) // 2 - 1] + sorted_attempts[len(sorted_attempts) // 2]) / 2.0
        )

        summary_obj = AuditedMethodSummary(
            method=m,
            instance_count=expected_count,
            success_count=succ_count,
            success_rate=succ_rate,
            wilson_95_ci=wilson_ci,
            mean_candidates_attempted=float(sum(attempts) / len(attempts)),
            median_candidates_attempted=med_attempts,
            mean_online_seconds=float(sum(runtimes) / len(runtimes)),
            median_online_seconds=med_online,
            mean_rank_seconds=float(sum(t_ranks) / len(t_ranks)),
            mean_sat_seconds=float(sum(t_sats) / len(t_sats)),
            mean_ver_seconds=float(sum(t_vers) / len(t_vers)),
            mean_conf_seconds=float(sum(t_confs) / len(t_confs)),
            timeout_count=timeouts,
            budget_exhausted_count=exhausted,
            par2_sensitivity=par2_dict,
        )
        audited_methods[m] = summary_obj.to_dict()

    # Paired McNemar comparisons
    paired_comparisons: dict[str, Any] = {}
    m1 = "b1_sat"
    for m2 in ("b0_sat", "sat_only", "random_sat", "gated_b1_fallback_sat"):
        b = sum(1 for r in records if r["methods"][m1]["success"] and not r["methods"][m2]["success"])
        c = sum(1 for r in records if not r["methods"][m1]["success"] and r["methods"][m2]["success"])
        paired_comparisons[f"{m1}_vs_{m2}"] = exact_mcnemar_test(b, c)

    return {
        "schema": "neural_sat_speck.phase7_audited_summary.v1",
        "phase": 9,
        "study": "phase9_phase7_record_audit",
        "instance_count": expected_count,
        "cutoffs_evaluated": list(cutoffs),
        "methods": audited_methods,
        "paired_comparisons": paired_comparisons,
    }
