#!/usr/bin/env python3
"""Generate final publication figures, traceable CSV tables, and metadata.

Strictly follows:
- docs/AGY_FINAL_FIGURES_AND_CSV_TASK_20261004.md
- docs/FINAL_FIGURES_AND_CSV_IMPLEMENTATION_NOTE_20261004.md
- docs/PUBLICATION_CLAIM_TO_EXPERIMENT_TABLE_V2_20261004.md

Styling & Palette:
- Exact scientific palette sampled from /Users/ashfaq/Downloads/pqc_ieee_t14.pdf:
    * Primary Navy Blue:       #2f5597
    * Burnt Terracotta / Rust: #c55a11
    * Olive / Forest Green:    #548235
    * Slate Purple:            #8064a2
    * Neutral Gray:            #666666
    * Dark Charcoal:           #222222
    * Dotted Grid Gray:        #d9d9d9
    * Light Blue Tint:         #d9e2f3
    * Light Peach Tint:        #fce4d6
    * Light Ice Blue:          #eaf2f8
- Typography: Times New Roman / Computer Modern serif with STIX math symbols.
- Framing: Despined axes (top/right hidden), 0.8pt spines, outward ticks.
- Outputs: Both vector PDF (Type 42 TrueType) and 600 DPI PNG.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import zipfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Repository Root
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neural_sat_speck.analysis.paired_stats import (  # noqa: E402
    exact_mcnemar_test,
    holm_bonferroni_adjustment,
    newcombe_method10_paired_ci,
)
from neural_sat_speck.evaluation.metrics import (  # noqa: E402
    wilson_score_interval,
)

# =============================================================================
# FROZEN INPUT SOURCE CATALOG & EXPECTED SHA-256 HASHES
# =============================================================================
EXPECTED_HASHES: dict[str, str] = {
    "speck_raw": "b8d06d276d18a5d50df96cbe5d599c830fa6c46c2a35083dc35265a6772ff606",
    "speck_analysis": "bd51138aab74aae5f2b3cc196716b2ed1c2ac79d71e57ec742897bc3fd0f2fbf",
    "simon_raw": "8cc9da7fbdf5207d94e0b108452819244a9dd4aee3dd28de14419af05e397b61",
    "simon_analysis": "2ccbae6c0bdcbfb9110bd21c29c45b758424914b45d7de3c58c787d950b315cb",
    "depth_raw": "bf5b05d89a7b03095298b4ad964d220178f37d584a3f3c166551a39e5b3810a2",
    "depth_analysis_v2": "cfceb5a96c72ea8082c73e4909c0fa1b473146bd2760d44a2a24da033c6ae3bb",
    "backend_raw": "30552bcfdc7a10e505c14ebc5a946cd9325cdd715f8e5af027e3596010d118aa",
    "backend_analysis": "a7eb470e1e4af396129ac8ac49154bbb48c1762a3c42b9781023df2a2faa0b94",
}

SOURCE_PATHS: dict[str, Path] = {
    "speck_raw": REPO_ROOT
    / "results/phase11_prospective/seven_round_prospective_results.json",
    "speck_analysis": REPO_ROOT
    / "results/phase11_prospective/seven_round_prospective_analysis.json",
    "simon_raw": REPO_ROOT
    / "results/simon_confirmation_replacement/simon_confirmation_comparators.json",
    "simon_analysis": REPO_ROOT
    / "results/simon_confirmation_replacement/simon_confirmation_analysis.json",
    "depth_raw": REPO_ROOT
    / "results/phase11_scaling_diagnostics/cross_cipher_residual_depth_map.json",
    "depth_analysis_v2": REPO_ROOT
    / "results/phase11_scaling_diagnostics/cross_cipher_residual_depth_analysis_reviewed_v2.json",
    "backend_raw": REPO_ROOT
    / "results/backend_comparison/backend_comparison_20261004.json",
    "backend_analysis": REPO_ROOT
    / "results/backend_comparison/backend_analysis_20261004.json",
}

# =============================================================================
# SCIENTIFIC PALETTE (Direct match with pqc_ieee_t14.pdf)
# =============================================================================
C_NAVY = "#2f5597"  # Steel / Navy Blue (Classical / Reference / O4)
C_ORANGE = "#c55a11"  # Burnt Terracotta / Rust (Hybrid / Guided / O128)
C_GREEN = "#548235"  # Olive / Forest Green (Pure Neural / Fast / Alternative)
C_PURPLE = "#8064a2"  # Slate Purple (Two-stage variant / Pure CNF)
C_GREY = "#666666"  # Neutral Medium Grey (Controls / SAT-only)
C_DARK = "#222222"  # Spine & Border Black
C_GRID = "#d9d9d9"  # Subtle Dotted Grid Grey

# Tints
BG_BLUE = "#d9e2f3"
BG_ORANGE = "#fce4d6"
BG_ICE = "#eaf2f8"

# IEEE Plotting Standards
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.8,
        "figure.titlesize": 10.0,
        "lines.linewidth": 1.2,
        "lines.markersize": 5.0,
        "axes.linewidth": 0.8,
        "axes.edgecolor": C_DARK,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def format_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    """Apply strict IEEE formatting: despine top/right, outward ticks, dotted grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C_DARK)
    ax.spines["bottom"].set_color(C_DARK)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    if grid_axis:
        ax.grid(
            axis=grid_axis,
            linestyle=":",
            linewidth=0.6,
            color=C_GRID,
            alpha=0.9,
            zorder=0,
        )
    ax.set_axisbelow(True)


def sha256_file(path: Path) -> str:
    """Calculate SHA-256 hex digest of file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_source_hashes() -> None:
    """Verify all 8 source files match expected SHA-256 digests bit-for-bit."""
    for key, path in SOURCE_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required source file for {key}: {path}")
        computed = sha256_file(path)
        expected = EXPECTED_HASHES[key]
        if computed != expected:
            raise ValueError(
                f"Source hash mismatch for {key} ({path})!\n"
                f"  Expected: {expected}\n"
                f"  Computed: {computed}\n"
                f"Execution terminated to preserve scientific integrity."
            )


def newcombe_method10_interval(
    n11: int, n10: int, n01: int, n00: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Use the audited implementation, including corrected association."""
    _, lower, upper = newcombe_method10_paired_ci(
        n11, n10, n01, n00, n11 + n10 + n01 + n00, confidence
    )
    return lower, upper


def exact_mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar binomial p-value."""
    return exact_mcnemar_test(b, c)["p_value"]


def _strict_bool(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be a genuine boolean")
    return value


def _seconds(value: Any, field: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid {field}: expected recorded seconds")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"Invalid {field}: {value}")
    return value


def validate_attack_records(raw: dict, cipher: str, methods: list[str], n: int) -> None:
    """Reject ambiguous success flags and incomplete/duplicate matched grids."""
    if cipher == "speck32_64":
        if set(raw["records_by_method"]) != set(methods):
            raise ValueError("SPECK method matrix drift")
        records = []
        for method, rows in raw["records_by_method"].items():
            if any(r["method"] != method for r in rows):
                raise ValueError("SPECK method/bucket disagreement")
            records.extend(rows)
    else:
        records = raw["records"]
    seen = set()
    for r in records:
        method, kid = r["method"], r["instance_index"]
        if type(kid) is not int or not 0 <= kid < n or method not in methods:
            raise ValueError("Attack cell outside declared grid")
        identity = (method, kid)
        if identity in seen:
            raise ValueError(f"Duplicate attack cell: {identity}")
        seen.add(identity)
        expected_id = (
            f"validation-{kid:04d}"
            if cipher == "speck32_64"
            else f"simon32_64_confirmation_r10_{kid:04d}"
        )
        if r["instance_id"] != expected_id:
            raise ValueError("Attack instance identity disagreement")
        if cipher == "speck32_64":
            verified = _strict_bool(
                r["counts_as_verified_recovery"], "verified recovery"
            )
            success = _strict_bool(r["success"], "success")
            if success != verified or (r["status"] == "verified_recovery") != verified:
                raise ValueError("Contradictory SPECK recovery flags")
            if _strict_bool(r["late_verified_recovery"], "late recovery"):
                raise ValueError("Late SPECK recovery cannot enter publication counts")
            wall = _seconds(r["online_wall_seconds"], "online wall", positive=True)
        else:
            if r["outcome"] not in {
                "verified_recovery",
                "candidate_unresolved",
                "queue_exhausted",
                "ranking_timeout",
                "global_deadline",
            }:
                raise ValueError("Unknown SIMON outcome")
            verified = r["outcome"] == "verified_recovery"
            if (
                _strict_bool(r["verified_on_fresh_pairs"], "fresh verification")
                != verified
            ):
                raise ValueError("Contradictory SIMON recovery/fresh verification")
            wall = _seconds(
                r["complete_online_wall_seconds"], "online wall", positive=True
            )
        if verified and wall > 300:
            raise ValueError("Late recovery cannot enter publication counts")
    expected = {(m, k) for m in methods for k in range(n)}
    if seen != expected:
        raise ValueError("Missing attack cells in declared matched grid")


def _write_provenance(path: Path, sources: list[str]) -> None:
    """Attach stable repository paths and pinned hashes to every table row."""
    rows = _read_csv(path)
    fields = list(rows[0])
    for index, source in enumerate(sources, 1):
        path_field, hash_field = f"source_{index}_path", f"source_{index}_sha256"
        fields.extend([path_field, hash_field])
        for row in rows:
            row[path_field] = str(SOURCE_PATHS[source].relative_to(REPO_ROOT))
            row[hash_field] = EXPECTED_HASHES[source]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_diagnostic_records(records: list[dict], kind: str) -> None:
    """Validate the fixed depth/backend grids and success evidence flags."""
    grids = {
        "speck32_64": [(6, 0), (6, 1), (7, 1), (7, 2), (8, 2), (8, 3)],
        "simon32_64": [(9, 0), (9, 1), (10, 1), (10, 2), (11, 2), (11, 3)],
    }
    full_profiles = {p for p, _ in _BACKEND_PROFILES}
    unguided_profiles = {
        "cms_wrapper",
        "cms_native_default",
        "cms_native_gauss",
        "cadical_cnf",
    }
    if kind == "depth":
        expected = {
            (cipher, r, m, o, k)
            for cipher, points in grids.items()
            for r, m in points
            for o in (4, 128)
            for k in range(8)
        }
    else:
        expected = {
            (cipher, r, m, o, k, p)
            for cipher, r in (("speck32_64", 7), ("simon32_64", 10))
            for m, o in ((0, 4), (0, 128), (1, 4), (2, 4))
            for k in range(4)
            for p in (unguided_profiles if m == 0 else full_profiles)
        }
    seen = set()
    for rec in records:
        r, o = rec["rounds"], rec["observations"]
        m = rec["assumed_keys_count"] if kind == "depth" else rec["trailing_keys"]
        k = rec["instance_index"] if kind == "depth" else rec["key_index"]
        if any(type(x) is not int for x in (r, m, o, k)):
            raise ValueError("Diagnostic indices must be integers")
        identity = (rec["cipher"], r, m, o, k)
        if kind == "backend":
            identity += (rec["profile"],)
        if identity in seen or identity not in expected:
            raise ValueError("Duplicate or undeclared diagnostic cell")
        seen.add(identity)
        ok = rec["outcome"] == ("verified_recovery" if kind == "depth" else "recovered")
        if kind == "depth":
            if rec["outcome"] not in ("verified_recovery", "parent_watchdog_timeout"):
                raise ValueError("Unknown depth outcome")
            if _strict_bool(rec["verified_recovery"], "depth verification") != ok:
                raise ValueError("Contradictory depth recovery flag")
            wanted_id = f"{rec['cipher']}_r{r}_m{m}_o{o}_k{k}"
            wall = _seconds(rec["parent_wall_seconds"], "parent wall", positive=True)
            if rec["nominal_residual_depth"] != r - m:
                raise ValueError("Nominal residual depth drift")
        else:
            if rec["outcome"] not in (
                "recovered",
                "global_deadline",
                "environment_rejected",
            ):
                raise ValueError("Unknown backend outcome")
            for field in ("verified_fresh", "verified_known", "assigned_key_match"):
                if _strict_bool(rec[field], field) != ok:
                    raise ValueError("Contradictory backend recovery verification")
            for field in ("guardian_healthy", "process_group_absent"):
                _strict_bool(rec[field], field)
            wanted_id = f"{rec['cipher']}_k{k}_o{o}_m{m}__{rec['profile']}"
            wall = _seconds(rec["online_seconds"], "backend online", positive=True)
        if rec["cell_id"] != wanted_id:
            raise ValueError("Diagnostic cell identity disagreement")
        if ok and wall > 300:
            raise ValueError("Late diagnostic recovery")
    if seen != expected:
        raise ValueError("Missing diagnostic cells in declared grid")


def save_figure_dual(
    fig: plt.Figure, output_dir: Path, base_name: str
) -> tuple[Path, Path]:
    """Save figure in both vector PDF (embedded fonts) and 600 DPI PNG."""
    pdf_path = output_dir / f"{base_name}.pdf"
    png_path = output_dir / f"{base_name}.png"
    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        dpi=600,
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(png_path, format="png", bbox_inches="tight", dpi=600)
    plt.close(fig)
    return pdf_path, png_path


# =============================================================================
# DATA EXTRACTION & CSV WRITERS
# =============================================================================


def build_attack_tables(
    speck_raw_path: Path, simon_raw_path: Path, output_tables_dir: Path
) -> tuple[Path, Path, Path, Path]:
    """Generate attack_cells.csv, attack_summary.csv, attack_paired_contrasts.csv, attack_failures.csv."""
    with open(speck_raw_path) as f:
        speck_raw = json.load(f)
    with open(simon_raw_path) as f:
        simon_raw = json.load(f)

    # 1. attack_cells.csv
    attack_cells_path = output_tables_dir / "attack_cells.csv"
    cells_rows: list[dict[str, Any]] = []

    # Process SPECK (768 cells: 6 methods x 128 instances)
    speck_method_order = [
        "hierarchical_neural_classical",
        "hierarchical_neural_screened",
        "hierarchical_neural_full",
        "random_sat",
        "sat_only_4obs",
        "sat_only_all_observations",
    ]
    validate_attack_records(speck_raw, "speck32_64", speck_method_order, 128)
    for method in speck_method_order:
        recs = speck_raw["records_by_method"][method]
        if len(recs) != 128:
            raise ValueError(
                f"SPECK method {method} has {len(recs)} records; expected 128."
            )
        for r in recs:
            verified = r["counts_as_verified_recovery"]
            wall_sec = r["online_wall_seconds"]
            outcome = (
                "verified_recovery"
                if verified
                else (
                    r.get("failure_cause") or r.get("status") or "unconditional_failure"
                )
            )
            cand_attempts = int(r.get("candidates_attempted", 0))
            ranking_sec = _seconds(
                r["ranking_screening_sorting_seconds"], "ranking seconds"
            )
            solver_sec = _seconds(r["solver_seconds"], "solver seconds")
            solver_status = (
                "not_entered" if method == "hierarchical_neural_full" else "recorded"
            )
            key_idx = int(
                r.get(
                    "instance_index",
                    r.get("instance_id", 0).split("-")[-1]
                    if isinstance(r.get("instance_id"), str)
                    and "-" in r.get("instance_id")
                    else 0,
                )
            )

            cells_rows.append(
                {
                    "study": "speck_seven_round_prospective",
                    "cipher": "speck32_64",
                    "instance_id": str(
                        r.get("instance_id", f"validation-{key_idx:04d}")
                    ),
                    "key_id": key_idx,
                    "method": method,
                    "method_name": METHOD_NAMES[method],
                    "cell_id": f"speck:{r['instance_id']}:{method}",
                    "role": "autonomous_attack",
                    "verified_recovery": verified,
                    "wall_seconds": round(wall_sec, 6),
                    "outcome": outcome,
                    "candidate_attempts": cand_attempts,
                    "ranking_seconds": round(ranking_sec, 6),
                    "solver_seconds": round(solver_sec, 6) if solver_sec != "" else "",
                    "solver_seconds_available": (solver_sec != ""),
                    "solver_seconds_status": solver_status,
                    "late_recovery_flag": bool(r.get("late_verified_recovery", False)),
                }
            )

    # Process SIMON (576 cells: 9 methods x 64 instances)
    simon_method_order = [
        "fully_classical_8parent",
        "neural_classical_hybrid_8parent",
        "fully_neural_1parent",
        "faithful_b1_hybrid",
        "random_control_128",
        "random_control_1024",
        "sat_only_n4",
        "sat_only_n128",
        "oracle_2key_n4",
    ]
    simon_by_method: dict[str, list[dict[str, Any]]] = {
        m: [] for m in simon_method_order
    }
    validate_attack_records(simon_raw, "simon32_64", simon_method_order, 64)
    for r in simon_raw["records"]:
        m = r["method"]
        if m in simon_by_method:
            simon_by_method[m].append(r)

    for method in simon_method_order:
        recs = simon_by_method[method]
        if len(recs) != 64:
            raise ValueError(
                f"SIMON method {method} has {len(recs)} records; expected 64."
            )
        for r in recs:
            verified = r.get("outcome") == "verified_recovery"
            wall_sec = float(r.get("complete_online_wall_seconds", 0.0))
            outcome = r.get("outcome", "unknown")
            cand_attempts = int(r.get("candidates_tested_count", 0))
            ranking_sec = _seconds(r["ranking_wall_seconds"], "ranking seconds")
            solving_sec = _seconds(r["solving_wall_seconds"], "solver seconds")
            solver_status = (
                "not_entered"
                if outcome == "ranking_timeout"
                else "recorded"
                if solving_sec > 0
                else "unavailable"
            )
            if solver_status == "unavailable":
                solving_sec = ""
            role = (
                "diagnostic_control"
                if method == "oracle_2key_n4"
                else "autonomous_attack"
            )
            key_idx = int(
                r.get(
                    "instance_index",
                    r.get("instance_id", 0).split("_")[-1]
                    if isinstance(r.get("instance_id"), str)
                    and "_" in r.get("instance_id")
                    else 0,
                )
            )

            cells_rows.append(
                {
                    "study": "simon_confirmation_replacement",
                    "cipher": "simon32_64",
                    "instance_id": str(
                        r.get(
                            "instance_id", f"simon32_64_confirmation_r10_{key_idx:04d}"
                        )
                    ),
                    "key_id": key_idx,
                    "method": method,
                    "method_name": METHOD_NAMES[method],
                    "cell_id": f"simon:{r['instance_id']}:{method}",
                    "role": role,
                    "verified_recovery": verified,
                    "wall_seconds": round(wall_sec, 6),
                    "outcome": outcome,
                    "candidate_attempts": cand_attempts,
                    "ranking_seconds": round(ranking_sec, 6),
                    "solver_seconds": round(solving_sec, 6)
                    if solving_sec != ""
                    else "",
                    "solver_seconds_available": (solving_sec != ""),
                    "solver_seconds_status": solver_status,
                    "late_recovery_flag": False,
                }
            )

    if len(cells_rows) != 1344:
        raise ValueError(
            f"Total attack cells count is {len(cells_rows)}; expected exactly 1344."
        )

    fieldnames = [
        "study",
        "cipher",
        "instance_id",
        "key_id",
        "method",
        "method_name",
        "cell_id",
        "role",
        "verified_recovery",
        "wall_seconds",
        "outcome",
        "candidate_attempts",
        "ranking_seconds",
        "solver_seconds",
        "solver_seconds_available",
        "solver_seconds_status",
        "late_recovery_flag",
    ]
    with open(attack_cells_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cells_rows)

    # 2. attack_summary.csv
    attack_summary_path = output_tables_dir / "attack_summary.csv"
    summary_rows: list[dict[str, Any]] = []

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in cells_rows:
        key = (row["study"], row["cipher"], row["method"], row["role"])
        grouped.setdefault(key, []).append(row)

    for (study, cipher, method, role), group in grouped.items():
        n = len(group)
        rec = sum(1 for r in group if r["verified_recovery"])
        rate = rec / n
        ci_low, ci_high = wilson_score_interval(rec, n, 0.95)
        walls = [r["wall_seconds"] for r in group]
        med_wall = float(np.median(walls))
        mean_wall = float(np.mean(walls))

        # PAR-2: success -> wall_seconds, failure -> 600.0s
        par2_times = [
            r["wall_seconds"] if r["verified_recovery"] else 600.0 for r in group
        ]
        par2 = float(np.mean(par2_times))

        cands = [r["candidate_attempts"] for r in group]
        mean_cands = float(np.mean(cands))
        med_cands = float(np.median(cands))

        summary_rows.append(
            {
                "study": study,
                "cipher": cipher,
                "method": method,
                "method_name": METHOD_NAMES[method],
                "role": role,
                "n_assigned": n,
                "n_recovered": rec,
                "recovery_rate": round(rate, 6),
                "wilson_95_lower": round(ci_low, 6),
                "wilson_95_upper": round(ci_high, 6),
                "median_wall_seconds": round(med_wall, 6),
                "mean_wall_seconds": round(mean_wall, 6),
                "par2_seconds": round(par2, 6),
                "mean_candidate_attempts": round(mean_cands, 4),
                "median_candidate_attempts": round(med_cands, 4),
            }
        )

    summary_fields = [
        "study",
        "cipher",
        "method",
        "method_name",
        "role",
        "n_assigned",
        "n_recovered",
        "recovery_rate",
        "wilson_95_lower",
        "wilson_95_upper",
        "median_wall_seconds",
        "mean_wall_seconds",
        "par2_seconds",
        "mean_candidate_attempts",
        "median_candidate_attempts",
    ]
    with open(attack_summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    # 3. attack_failures.csv
    attack_failures_path = output_tables_dir / "attack_failures.csv"
    failure_rows: list[dict[str, Any]] = []

    for (study, cipher, method, role), group in grouped.items():
        n = len(group)
        counts: dict[str, int] = {}
        for r in group:
            out = r["outcome"]
            counts[out] = counts.get(out, 0) + 1

        if sum(counts.values()) != n:
            raise ValueError(
                f"Failure counts for {method} sum to {sum(counts.values())} != {n}"
            )

        for out, count in sorted(counts.items()):
            failure_rows.append(
                {
                    "study": study,
                    "cipher": cipher,
                    "method": method,
                    "method_name": METHOD_NAMES[method],
                    "role": role,
                    "outcome_category": out,
                    "count": count,
                    "n_assigned": n,
                    "fraction_of_n": round(count / n, 6),
                }
            )

    failure_fields = [
        "study",
        "cipher",
        "method",
        "method_name",
        "role",
        "outcome_category",
        "count",
        "n_assigned",
        "fraction_of_n",
    ]
    with open(attack_failures_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=failure_fields)
        writer.writeheader()
        writer.writerows(failure_rows)

    # 4. attack_paired_contrasts.csv
    attack_contrasts_path = output_tables_dir / "attack_paired_contrasts.csv"
    contrast_rows: list[dict[str, Any]] = []

    speck_key_map: dict[tuple[str, int], bool] = {}
    for r in cells_rows:
        if r["study"] == "speck_seven_round_prospective":
            speck_key_map[(r["method"], r["key_id"])] = r["verified_recovery"]

    simon_key_map: dict[tuple[str, int], bool] = {}
    for r in cells_rows:
        if r["study"] == "simon_confirmation_replacement":
            simon_key_map[(r["method"], r["key_id"])] = r["verified_recovery"]

    declared_contrasts = [
        (
            "speck_seven_round_prospective",
            "speck32_64",
            "hierarchical_neural_screened",
            "sat_only_all_observations",
            "primary_declared",
            "None",
        ),
        (
            "speck_seven_round_prospective",
            "speck32_64",
            "hierarchical_neural_screened",
            "sat_only_4obs",
            "secondary_declared",
            "speck_secondary",
        ),
        (
            "speck_seven_round_prospective",
            "speck32_64",
            "hierarchical_neural_screened",
            "random_sat",
            "secondary_declared",
            "speck_secondary",
        ),
        (
            "speck_seven_round_prospective",
            "speck32_64",
            "hierarchical_neural_screened",
            "hierarchical_neural_classical",
            "secondary_declared",
            "speck_secondary",
        ),
        (
            "speck_seven_round_prospective",
            "speck32_64",
            "hierarchical_neural_classical",
            "sat_only_all_observations",
            "exploratory_secondary",
            "None",
        ),
        (
            "simon_confirmation_replacement",
            "simon32_64",
            "fully_classical_8parent",
            "sat_only_n128",
            "primary_declared",
            "simon_primary",
        ),
        (
            "simon_confirmation_replacement",
            "simon32_64",
            "neural_classical_hybrid_8parent",
            "sat_only_n128",
            "primary_declared",
            "simon_primary",
        ),
        (
            "simon_confirmation_replacement",
            "simon32_64",
            "fully_classical_8parent",
            "neural_classical_hybrid_8parent",
            "exploratory_descriptive",
            "None",
        ),
        (
            "simon_confirmation_replacement",
            "simon32_64",
            "fully_classical_8parent",
            "fully_neural_1parent",
            "exploratory_descriptive",
            "None",
        ),
        (
            "simon_confirmation_replacement",
            "simon32_64",
            "neural_classical_hybrid_8parent",
            "fully_neural_1parent",
            "exploratory_descriptive",
            "None",
        ),
    ]
    for meth_a, meth_b in (
        ("faithful_b1_hybrid", "sat_only_n128"),
        ("faithful_b1_hybrid", "fully_classical_8parent"),
        ("faithful_b1_hybrid", "neural_classical_hybrid_8parent"),
        ("neural_classical_hybrid_8parent", "random_control_1024"),
        ("fully_classical_8parent", "random_control_1024"),
        ("fully_neural_1parent", "random_control_128"),
        ("sat_only_n4", "sat_only_n128"),
    ):
        declared_contrasts.append(
            (
                "simon_confirmation_replacement",
                "simon32_64",
                meth_a,
                meth_b,
                "exploratory_descriptive",
                "None",
            )
        )

    for study, cipher, meth_a, meth_b, role, adj in declared_contrasts:
        kmap = (
            speck_key_map if study == "speck_seven_round_prospective" else simon_key_map
        )
        keys = range(128) if study == "speck_seven_round_prospective" else range(64)
        n11 = n10 = n01 = n00 = 0
        for kid in keys:
            a = kmap[(meth_a, kid)]
            b = kmap[(meth_b, kid)]
            if a and b:
                n11 += 1
            elif a and not b:
                n10 += 1
            elif not a and b:
                n01 += 1
            else:
                n00 += 1
        n_pairs = n11 + n10 + n01 + n00
        p_diff = (n10 - n01) / n_pairs
        ci_l, ci_u = newcombe_method10_interval(n11, n10, n01, n00, 0.95)
        p_val = exact_mcnemar_p(n10, n01)

        contrast_rows.append(
            {
                "study": study,
                "cipher": cipher,
                "method_a": meth_a,
                "method_b": meth_b,
                "method_a_name": METHOD_NAMES[meth_a],
                "method_b_name": METHOD_NAMES[meth_b],
                "role": role,
                "multiplicity_adjustment": "Holm" if adj != "None" else "None",
                "multiplicity_family": adj,
                "holm_adjusted_p": "",
                "interval_coverage": "pointwise_95_percent",
                "n_pairs": n_pairs,
                "n11_both_succeed": n11,
                "n10_a_only": n10,
                "n01_b_only": n01,
                "n00_neither": n00,
                "paired_difference": round(p_diff, 6),
                "newcombe_95_lower": round(ci_l, 6),
                "newcombe_95_upper": round(ci_u, 6),
                "exact_mcnemar_p": p_val,
            }
        )

    for family in ("speck_secondary", "simon_primary"):
        family_rows = [r for r in contrast_rows if r["multiplicity_family"] == family]
        adjustments = holm_bonferroni_adjustment(
            [r["exact_mcnemar_p"] for r in family_rows]
        )
        for row, adjustment in zip(family_rows, adjustments, strict=True):
            row["holm_adjusted_p"] = adjustment["adjusted_p"]

    contrast_fields = [
        "study",
        "cipher",
        "method_a",
        "method_b",
        "method_a_name",
        "method_b_name",
        "role",
        "multiplicity_adjustment",
        "multiplicity_family",
        "holm_adjusted_p",
        "interval_coverage",
        "n_pairs",
        "n11_both_succeed",
        "n10_a_only",
        "n01_b_only",
        "n00_neither",
        "paired_difference",
        "newcombe_95_lower",
        "newcombe_95_upper",
        "exact_mcnemar_p",
    ]
    with open(attack_contrasts_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=contrast_fields)
        writer.writeheader()
        writer.writerows(contrast_rows)

    for path in (
        attack_cells_path,
        attack_summary_path,
        attack_failures_path,
        attack_contrasts_path,
    ):
        _write_provenance(path, ["speck_raw", "simon_raw"])
    return (
        attack_cells_path,
        attack_summary_path,
        attack_failures_path,
        attack_contrasts_path,
    )


def reconcile_attack_summary(path: Path) -> None:
    """Compare raw-derived summaries with audited, frozen analysis outputs."""
    speck = json.loads(SOURCE_PATHS["speck_analysis"].read_text())["method_metrics"]
    simon_an = json.loads(SOURCE_PATHS["simon_analysis"].read_text())["analysis"]
    simon = dict(simon_an["method_summaries"])
    simon["oracle_2key_n4"] = simon_an["diagnostic_oracle"]
    for row in _read_csv(path):
        is_speck = row["cipher"] == "speck32_64"
        ref = (speck if is_speck else simon)[row["method"]]
        n = ref["instances"] if is_speck else ref["instances_evaluated"]
        if (
            int(row["n_assigned"]) != n
            or int(row["n_recovered"]) != ref["verified_recoveries"]
        ):
            raise ValueError(
                "Raw-derived recovery summary disagrees with frozen analysis"
            )
        timings = ref["wall_seconds"] if is_speck else ref["unconditional_wall_seconds"]
        for field, val, tolerance in (
            (
                "median_wall_seconds",
                timings["median"],
                0.00051 if is_speck else 0.000002,
            ),
            ("mean_wall_seconds", timings["mean"], 0.00051 if is_speck else 0.000002),
            (
                "par2_seconds",
                ref["par2_seconds"] if is_speck else ref["par2_score"],
                0.00051 if is_speck else 0.000002,
            ),
            (
                "wilson_95_lower",
                ref["wilson_95_ci"][0],
                0.000051 if is_speck else 0.000002,
            ),
            (
                "wilson_95_upper",
                ref["wilson_95_ci"][1],
                0.000051 if is_speck else 0.000002,
            ),
        ):
            if abs(float(row[field]) - val) > tolerance:
                raise ValueError(
                    f"Raw-derived {field} disagrees with frozen analysis: {row['method']}"
                )


def reconcile_attack_contrasts(path: Path) -> None:
    """Check paired identities/counts/p-values; preserve historical SPECK rounding."""
    speck = json.loads(SOURCE_PATHS["speck_analysis"].read_text())
    simon = json.loads(SOURCE_PATHS["simon_analysis"].read_text())["analysis"]
    rows = {(r["cipher"], r["method_a"], r["method_b"]): r for r in _read_csv(path)}
    for ref in [speck["primary_paired_contrast"]] + speck[
        "multiplicity_controlled_secondary_contrasts"
    ]:
        row = rows[("speck32_64", ref["method_a"], ref["method_b"])]
        _reconcile_pair_counts(row, ref["contingency_table"])
        exact = ref.get("exact_mcnemar", ref.get("mcnemar"))["p_value"]
        if not math.isclose(
            float(row["exact_mcnemar_p"]), exact, rel_tol=1e-12, abs_tol=0
        ):
            raise ValueError("SPECK paired p-value drift")
    for ref in (
        simon["primary_contrasts_holm"] + simon["secondary_contrasts_exploratory"]
    ):
        row = rows[("simon32_64", ref["method_a"], ref["method_b"])]
        _reconcile_pair_counts(row, ref["contingency"])
        if not math.isclose(
            float(row["exact_mcnemar_p"]),
            ref["mcnemar_raw_p"],
            rel_tol=1e-12,
            abs_tol=0,
        ):
            raise ValueError("SIMON paired p-value drift")
        for field, value in zip(
            ("newcombe_95_lower", "newcombe_95_upper"),
            ref["newcombe_95_ci"],
            strict=True,
        ):
            if abs(float(row[field]) - value) > 0.000002:
                raise ValueError("SIMON paired interval drift")
        if "mcnemar_adjusted_p" in ref and not math.isclose(
            float(row["holm_adjusted_p"]),
            ref["mcnemar_adjusted_p"],
            rel_tol=1e-12,
            abs_tol=0,
        ):
            raise ValueError("SIMON Holm-adjusted p-value drift")


def _reconcile_pair_counts(row: dict, ref: dict) -> None:
    for field, ref_field in (
        ("n11_both_succeed", "n11"),
        ("n10_a_only", "n10"),
        ("n01_b_only", "n01"),
        ("n00_neither", "n00"),
        ("n_pairs", "total"),
    ):
        if int(row[field]) != ref[ref_field]:
            raise ValueError("Paired contingency disagrees with frozen analysis")


def build_depth_tables(
    depth_raw_path: Path, depth_analysis_path: Path, output_tables_dir: Path
) -> tuple[Path, Path, Path]:
    """Generate depth_cells.csv, depth_summary.csv, depth_paired_contrasts.csv."""
    with open(depth_raw_path) as f:
        depth_raw = json.load(f)
    with open(depth_analysis_path) as f:
        depth_an = json.load(f)

    # 1. depth_cells.csv (192 rows)
    depth_cells_path = output_tables_dir / "depth_cells.csv"
    cells: list[dict[str, Any]] = []

    formula_meta: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for gb in depth_an["grid_breakdown"]:
        key = (gb["cipher"], gb["rounds"], gb["assumed_keys"], gb["observations"])
        if key in formula_meta:
            raise ValueError("Duplicate depth summary group")
        formula_meta[key] = {
            "nominal_residual_depth": gb["nominal_residual_depth"],
            "variables_count": gb["variables_count"],
            "cnf_clauses_count": gb["cnf_clauses_count"],
            "xor_clauses_count": gb["xor_clauses_count"],
        }

    raw_cells = (
        list(depth_raw["cells"].values())
        if isinstance(depth_raw["cells"], dict)
        else depth_raw["cells"]
    )
    if isinstance(depth_raw["cells"], dict) and any(
        k != v["cell_id"] for k, v in depth_raw["cells"].items()
    ):
        raise ValueError("Depth storage key/cell ID disagreement")
    validate_diagnostic_records(raw_cells, "depth")
    if len(raw_cells) != 192:
        raise ValueError(f"Depth raw cells count is {len(raw_cells)}; expected 192.")
    raw_groups = {
        (c["cipher"], c["rounds"], c["assumed_keys_count"], c["observations"])
        for c in raw_cells
    }
    if set(formula_meta) != raw_groups:
        raise ValueError("Depth analysis/raw group identity disagreement")

    for c in raw_cells:
        ciph = c["cipher"]
        r = int(c["rounds"])
        m = int(c.get("assumed_keys_count", c.get("assumed_keys", 0)))
        obs = int(c["observations"])
        meta = formula_meta.get((ciph, r, m, obs), {})
        nom_d = int(
            meta.get("nominal_residual_depth", c.get("nominal_residual_depth", r - m))
        )
        verified = c["outcome"] == "verified_recovery"
        solve_sec = c.get("solve_wall_seconds", c.get("solve_duration_seconds"))
        solve_avail = solve_sec is not None and solve_sec > 0.0
        key_id = c.get("instance_index", c.get("key_index", 0))

        cells.append(
            {
                "cell_id": c["cell_id"],
                "cipher": ciph,
                "rounds": r,
                "assumed_keys_m": m,
                "nominal_residual_depth_d": nom_d,
                "observations_O": obs,
                "key_id": key_id,
                "outcome": c["outcome"],
                "verified_recovery": verified,
                "parent_wall_seconds": round(float(c["parent_wall_seconds"]), 6),
                "solve_duration_seconds": round(float(solve_sec), 6)
                if solve_avail
                else "",
                "solve_duration_available": solve_avail,
                "solve_duration_status": "recorded"
                if solve_avail
                else "unavailable_timeout_telemetry",
                "variables_count": meta.get(
                    "variables_count", c.get("variables_count", "")
                ),
                "cnf_clauses_count": meta.get(
                    "cnf_clauses_count", c.get("cnf_clauses_count", "")
                ),
                "xor_clauses_count": meta.get(
                    "xor_clauses_count", c.get("xor_clauses_count", "")
                ),
            }
        )

    depth_cell_fields = [
        "cell_id",
        "cipher",
        "rounds",
        "assumed_keys_m",
        "nominal_residual_depth_d",
        "observations_O",
        "key_id",
        "outcome",
        "verified_recovery",
        "parent_wall_seconds",
        "solve_duration_seconds",
        "solve_duration_available",
        "solve_duration_status",
        "variables_count",
        "cnf_clauses_count",
        "xor_clauses_count",
    ]
    with open(depth_cells_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=depth_cell_fields)
        writer.writeheader()
        writer.writerows(cells)

    # 2. depth_summary.csv (24 groups)
    depth_summary_path = output_tables_dir / "depth_summary.csv"
    summary_rows: list[dict[str, Any]] = []

    for gb in depth_an["grid_breakdown"]:
        group = [
            c
            for c in raw_cells
            if (c["cipher"], c["rounds"], c["assumed_keys_count"], c["observations"])
            == (gb["cipher"], gb["rounds"], gb["assumed_keys"], gb["observations"])
        ]
        successes = [c for c in group if c["verified_recovery"]]
        walls = [
            _seconds(c["parent_wall_seconds"], "parent wall", positive=True)
            for c in group
        ]
        solves = [
            _seconds(c["solve_wall_seconds"], "completed solve", positive=True)
            for c in successes
        ]
        computed = {
            "n_instances": len(group),
            "verified_recovery_count": len(successes),
            "verified_recovery_rate": len(successes) / len(group),
            "median_parent_seconds": float(np.median(walls)),
            "iqr_parent_seconds": statistics.quantiles(walls, n=4)[2]
            - statistics.quantiles(walls, n=4)[0],
            "median_solve_seconds": float(np.median(solves)) if solves else None,
            "par2_seconds": float(
                np.mean(
                    [
                        c["parent_wall_seconds"] if c["verified_recovery"] else 600
                        for c in group
                    ]
                )
            ),
        }
        for field, value in computed.items():
            if value is None:
                if gb[field] is not None:
                    raise ValueError(
                        "Unavailable depth solve median disagrees with reviewed analysis"
                    )
            elif gb[field] is None or abs(value - gb[field]) > 0.000002:
                raise ValueError(f"Raw depth {field} disagrees with reviewed analysis")
        gb.update(computed)
        summary_rows.append(
            {
                "cipher": gb["cipher"],
                "rounds": gb["rounds"],
                "assumed_keys_m": gb["assumed_keys"],
                "nominal_residual_depth_d": gb["nominal_residual_depth"],
                "observations_O": gb["observations"],
                "n_instances": gb["n_instances"],
                "verified_recovery_count": gb["verified_recovery_count"],
                "verified_recovery_rate": round(gb["verified_recovery_rate"], 6),
                "median_parent_seconds": round(gb["median_parent_seconds"], 6),
                "iqr_parent_seconds": round(gb["iqr_parent_seconds"], 6),
                "median_solve_seconds": round(gb["median_solve_seconds"], 6)
                if gb["median_solve_seconds"] is not None
                else "",
                "solve_median_scope": gb["solve_median_scope"],
                "par2_seconds": round(gb["par2_seconds"], 6),
                "variables_count": gb["variables_count"],
                "cnf_clauses_count": gb["cnf_clauses_count"],
                "xor_clauses_count": gb["xor_clauses_count"],
            }
        )

    depth_summary_fields = [
        "cipher",
        "rounds",
        "assumed_keys_m",
        "nominal_residual_depth_d",
        "observations_O",
        "n_instances",
        "verified_recovery_count",
        "verified_recovery_rate",
        "median_parent_seconds",
        "iqr_parent_seconds",
        "median_solve_seconds",
        "solve_median_scope",
        "par2_seconds",
        "variables_count",
        "cnf_clauses_count",
        "xor_clauses_count",
    ]
    with open(depth_summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=depth_summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    # 3. depth_paired_contrasts.csv (matched-key deltas O128 vs O4 and equal-depth contrasts)
    depth_contrasts_path = output_tables_dir / "depth_paired_contrasts.csv"
    contrast_rows: list[dict[str, Any]] = []

    cell_map: dict[tuple[str, int, int, int, int], dict[str, Any]] = {}
    for c in raw_cells:
        ciph = c["cipher"]
        r = int(c["rounds"])
        m = int(c.get("assumed_keys_count", c.get("assumed_keys", 0)))
        obs = int(c["observations"])
        kid = int(c.get("instance_index", c.get("key_index", 0)))
        cell_map[(ciph, r, m, obs, kid)] = c

    # 3A. Observation scaling contrasts: O=128 minus O=4 for all 12 tested (cipher, R, m) regimes
    obs_conditions = [
        ("speck32_64", 6, 0),
        ("speck32_64", 6, 1),
        ("speck32_64", 7, 1),
        ("speck32_64", 7, 2),
        ("speck32_64", 8, 2),
        ("speck32_64", 8, 3),
        ("simon32_64", 9, 0),
        ("simon32_64", 9, 1),
        ("simon32_64", 10, 1),
        ("simon32_64", 10, 2),
        ("simon32_64", 11, 2),
        ("simon32_64", 11, 3),
    ]

    for ciph, r, m in obs_conditions:
        d = r - m
        deltas = []
        rec_a, rec_b = 0, 0
        for k in range(8):
            ca = cell_map.get((ciph, r, m, 4, k))
            cb = cell_map.get((ciph, r, m, 128, k))
            if ca and cb:
                if ca["outcome"] == "verified_recovery":
                    rec_a += 1
                if cb["outcome"] == "verified_recovery":
                    rec_b += 1
                deltas.append(
                    float(cb["parent_wall_seconds"]) - float(ca["parent_wall_seconds"])
                )
        contrast_rows.append(
            {
                "contrast_family": "observation_scaling_o128_minus_o4",
                "cipher": ciph,
                "nominal_residual_depth_d": d,
                "rounds_a": r,
                "assumed_keys_m_a": m,
                "observations_O_a": 4,
                "rounds_b": r,
                "assumed_keys_m_b": m,
                "observations_O_b": 128,
                "matched_keys_count": 8,
                "recovery_count_a": rec_a,
                "recovery_count_b": rec_b,
                "recovery_delta_b_minus_a": rec_b - rec_a,
                "mean_parent_delta_seconds": round(float(np.mean(deltas)), 6),
                "median_parent_delta_seconds": round(float(np.median(deltas)), 6),
                "note": "Descriptive matched-key delta (O128 minus O4) over 8 demo keys.",
            }
        )

    # 3B. Equal-residual-depth contrasts across rounds at fixed O=4
    depth_pairs = [
        ("speck32_64", 6, (6, 0), (7, 1)),
        ("speck32_64", 6, (7, 1), (8, 2)),
        ("speck32_64", 5, (6, 1), (7, 2)),
        ("speck32_64", 5, (7, 2), (8, 3)),
        ("simon32_64", 9, (9, 0), (10, 1)),
        ("simon32_64", 9, (10, 1), (11, 2)),
        ("simon32_64", 8, (9, 1), (10, 2)),
        ("simon32_64", 8, (10, 2), (11, 3)),
    ]

    for ciph, d, (ra, ma), (rb, mb) in depth_pairs:
        deltas = []
        rec_a, rec_b = 0, 0
        for k in range(8):
            ca = cell_map.get((ciph, ra, ma, 4, k))
            cb = cell_map.get((ciph, rb, mb, 4, k))
            if ca and cb:
                if ca["outcome"] == "verified_recovery":
                    rec_a += 1
                if cb["outcome"] == "verified_recovery":
                    rec_b += 1
                deltas.append(
                    float(cb["parent_wall_seconds"]) - float(ca["parent_wall_seconds"])
                )
        contrast_rows.append(
            {
                "contrast_family": "depth_invariance_fixed_o4",
                "cipher": ciph,
                "nominal_residual_depth_d": d,
                "rounds_a": ra,
                "assumed_keys_m_a": ma,
                "observations_O_a": 4,
                "rounds_b": rb,
                "assumed_keys_m_b": mb,
                "observations_O_b": 4,
                "matched_keys_count": 8,
                "recovery_count_a": rec_a,
                "recovery_count_b": rec_b,
                "recovery_delta_b_minus_a": rec_b - rec_a,
                "mean_parent_delta_seconds": round(float(np.mean(deltas)), 6),
                "median_parent_delta_seconds": round(float(np.median(deltas)), 6),
                "note": f"Descriptive equal-depth contrast (R{rb}m{mb} minus R{ra}m{ma}) at O=4 over 8 demo keys.",
            }
        )

    contrast_fields = [
        "contrast_family",
        "cipher",
        "nominal_residual_depth_d",
        "rounds_a",
        "assumed_keys_m_a",
        "observations_O_a",
        "rounds_b",
        "assumed_keys_m_b",
        "observations_O_b",
        "matched_keys_count",
        "recovery_count_a",
        "recovery_count_b",
        "recovery_delta_b_minus_a",
        "mean_parent_delta_seconds",
        "median_parent_delta_seconds",
        "note",
    ]
    with open(depth_contrasts_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=contrast_fields)
        writer.writeheader()
        writer.writerows(contrast_rows)

    for path in (depth_cells_path, depth_summary_path, depth_contrasts_path):
        _write_provenance(path, ["depth_raw", "depth_analysis_v2"])
    return depth_cells_path, depth_summary_path, depth_contrasts_path


def build_backend_tables(
    backend_raw_path: Path, backend_analysis_path: Path, output_tables_dir: Path
) -> tuple[Path, Path, Path]:
    """Generate backend_cells.csv, backend_summary.csv, backend_paired_contrasts.csv."""
    with open(backend_raw_path) as f:
        backend_raw = json.load(f)
    with open(backend_analysis_path) as f:
        backend_an = json.load(f)

    # 1. backend_cells.csv (160 rows)
    backend_cells_path = output_tables_dir / "backend_cells.csv"
    cells: list[dict[str, Any]] = []

    raw_records = backend_raw["records"]
    if any(k != r["cell_id"] for k, r in raw_records.items()):
        raise ValueError("Backend storage key/cell ID disagreement")
    validate_diagnostic_records(list(raw_records.values()), "backend")
    expected_groups = {
        f"{r['cipher']}_o{r['observations']}_m{r['trailing_keys']}__{r['profile']}"
        for r in raw_records.values()
    }
    if set(backend_an["summaries"]) != expected_groups or len(expected_groups) != 40:
        raise ValueError("Backend summary group matrix drift")
    from itertools import combinations

    expected_pairs = set()
    for condition in {g.split("__")[0] for g in expected_groups}:
        profiles = sorted(
            g.split("__")[1] for g in expected_groups if g.split("__")[0] == condition
        )
        expected_pairs.update((condition, a, b) for a, b in combinations(profiles, 2))
    observed_pairs = [
        (condition, c["first"], c["second"])
        for condition, cs in backend_an["paired_descriptive_contrasts"].items()
        for c in cs
    ]
    if (
        len(observed_pairs) != len(set(observed_pairs))
        or set(observed_pairs) != expected_pairs
    ):
        raise ValueError("Backend paired contrast matrix drift")
    if len(raw_records) != 160:
        raise ValueError(
            f"Backend raw records count is {len(raw_records)}; expected 160."
        )

    for cell_id, r in sorted(raw_records.items()):
        verified = r["outcome"] == "recovered"
        online_sec = r.get("online_seconds")
        interrupted = r["outcome"] == "environment_rejected"

        cells.append(
            {
                "cell_id": cell_id,
                "case_id": r["case_id"],
                "cipher": r["cipher"],
                "rounds": r["rounds"],
                "trailing_keys_m": r["trailing_keys"],
                "observations_O": r["observations"],
                "profile": r["profile"],
                "profile_name": dict(_BACKEND_PROFILES)[r["profile"]],
                "key_id": r["key_index"],
                "outcome": r["outcome"],
                "verified_recovery": verified,
                "online_seconds": round(float(online_sec), 6)
                if not interrupted and online_sec is not None
                else "",
                "online_seconds_available": (
                    not interrupted and online_sec is not None
                ),
                "observed_elapsed_seconds": round(float(online_sec), 6),
                "online_seconds_status": "infrastructure_unavailable"
                if interrupted
                else "recorded",
                "infrastructure_interrupted": interrupted,
                "interruption_reason": r.get("error", "") if interrupted else "",
                "guardian_healthy": bool(r.get("guardian_healthy", False)),
                "process_group_absent": bool(r.get("process_group_absent", False)),
            }
        )

    backend_cell_fields = [
        "cell_id",
        "case_id",
        "cipher",
        "rounds",
        "trailing_keys_m",
        "observations_O",
        "profile",
        "profile_name",
        "key_id",
        "outcome",
        "verified_recovery",
        "online_seconds",
        "observed_elapsed_seconds",
        "online_seconds_status",
        "online_seconds_available",
        "infrastructure_interrupted",
        "interruption_reason",
        "guardian_healthy",
        "process_group_absent",
    ]
    with open(backend_cells_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=backend_cell_fields)
        writer.writeheader()
        writer.writerows(cells)

    # 2. backend_summary.csv (40 groups)
    backend_summary_path = output_tables_dir / "backend_summary.csv"
    summary_rows: list[dict[str, Any]] = []

    for group_id, s in sorted(backend_an["summaries"].items()):
        parts = group_id.split("__")
        cond = parts[0]
        profile = parts[1]
        ciph = "simon32_64" if cond.startswith("simon") else "speck32_64"

        n_assigned = s["n"]
        outcomes = s["outcomes"]
        rec = outcomes.get("recovered", 0)
        deadlines = outcomes.get("global_deadline", 0)
        rejected = outcomes.get("environment_rejected", 0)
        n_completed = rec + deadlines

        completed_rows = [
            r
            for r in cells
            if f"{r['cipher']}_o{r['observations_O']}_m{r['trailing_keys_m']}" == cond
            and r["profile"] == profile
            and not r["infrastructure_interrupted"]
        ]
        if len(completed_rows) != n_completed:
            raise ValueError("Backend completed-trial summary disagrees with raw cells")
        assigned_rows = [
            r
            for r in cells
            if f"{r['cipher']}_o{r['observations_O']}_m{r['trailing_keys_m']}" == cond
            and r["profile"] == profile
        ]
        raw_rec = sum(r["verified_recovery"] for r in assigned_rows)
        raw_deadlines = sum(r["outcome"] == "global_deadline" for r in assigned_rows)
        raw_rejected = sum(r["infrastructure_interrupted"] for r in assigned_rows)
        if (len(assigned_rows), raw_rec, raw_deadlines, raw_rejected) != (
            n_assigned,
            rec,
            deadlines,
            rejected,
        ):
            raise ValueError("Raw backend outcomes disagree with frozen analysis")
        med_online = (
            float(np.median([r["online_seconds"] for r in completed_rows]))
            if completed_rows
            else None
        )
        par2 = s.get("par2_seconds")
        raw_par2 = float(
            np.mean(
                [
                    r["online_seconds"] if r["verified_recovery"] else 600
                    for r in assigned_rows
                ]
            )
        )
        if par2 is None or abs(raw_par2 - par2) > 0.000002:
            raise ValueError(
                "Raw backend scheduled PAR-2 disagrees with frozen analysis"
            )
        par2 = raw_par2

        summary_rows.append(
            {
                "group_id": group_id,
                "condition": cond,
                "cipher": ciph,
                "profile": profile,
                "n_assigned": n_assigned,
                "n_completed_bounded_trials": n_completed,
                "infrastructure_unavailable": rejected,
                "verified_recoveries": rec,
                "solver_deadlines": deadlines,
                "median_available_online_seconds": round(med_online, 6)
                if med_online is not None
                else "",
                "scheduled_cell_par2_seconds": round(par2, 6)
                if par2 is not None
                else "",
            }
        )

    backend_summary_fields = [
        "group_id",
        "condition",
        "cipher",
        "profile",
        "n_assigned",
        "n_completed_bounded_trials",
        "infrastructure_unavailable",
        "verified_recoveries",
        "solver_deadlines",
        "median_available_online_seconds",
        "scheduled_cell_par2_seconds",
    ]
    with open(backend_summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=backend_summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    # 3. backend_paired_contrasts.csv
    backend_contrasts_path = output_tables_dir / "backend_paired_contrasts.csv"
    contrast_rows: list[dict[str, Any]] = []

    for cond, contrasts in backend_an.get("paired_descriptive_contrasts", {}).items():
        ciph = "simon32_64" if cond.startswith("simon") else "speck32_64"
        for c in contrasts:
            paired = []
            for key_id in range(4):
                pair = []
                for profile in (c["first"], c["second"]):
                    matches = [
                        r
                        for r in cells
                        if f"{r['cipher']}_o{r['observations_O']}_m{r['trailing_keys_m']}"
                        == cond
                        and r["profile"] == profile
                        and r["key_id"] == key_id
                    ]
                    if len(matches) != 1:
                        raise ValueError("Missing backend contrast cell")
                    pair.append(matches[0])
                if not any(r["infrastructure_interrupted"] for r in pair):
                    paired.append(pair)
            n11 = sum(
                a["verified_recovery"] and b["verified_recovery"] for a, b in paired
            )
            n10 = sum(
                a["verified_recovery"] and not b["verified_recovery"] for a, b in paired
            )
            n01 = sum(
                not a["verified_recovery"] and b["verified_recovery"] for a, b in paired
            )
            n00 = len(paired) - n11 - n10 - n01
            deltas = [
                (a["online_seconds"] if a["verified_recovery"] else 600)
                - (b["online_seconds"] if b["verified_recovery"] else 600)
                for a, b in paired
            ]
            contrast_rows.append(
                {
                    "condition": cond,
                    "cipher": ciph,
                    "profile_a": c["first"],
                    "profile_b": c["second"],
                    "valid_pair_count": len(paired),
                    "excluded_interrupted_pairs": 4 - len(paired),
                    "n_both_succeed": n11,
                    "n_a_only": n10,
                    "n_b_only": n01,
                    "n_neither": n00,
                    "recovery_difference": round((n10 - n01) / len(paired), 6),
                    "mean_paired_par2_difference_seconds": round(
                        float(np.mean(deltas)), 6
                    ),
                    "note": "Descriptive completed-trial paired contrast; interrupted pairs unavailable; no population significance claim.",
                }
            )

    backend_contrast_fields = [
        "condition",
        "cipher",
        "profile_a",
        "profile_b",
        "valid_pair_count",
        "excluded_interrupted_pairs",
        "n_both_succeed",
        "n_a_only",
        "n_b_only",
        "n_neither",
        "recovery_difference",
        "mean_paired_par2_difference_seconds",
        "note",
    ]
    with open(backend_contrasts_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=backend_contrast_fields)
        writer.writeheader()
        writer.writerows(contrast_rows)

    for path in (backend_cells_path, backend_summary_path, backend_contrasts_path):
        _write_provenance(path, ["backend_raw", "backend_analysis"])
    return backend_cells_path, backend_summary_path, backend_contrasts_path


# =============================================================================
# FIGURE PLOTTERS (revised layout, 2026-10-05: same palette, same data tables)
#
# Design rules shared by every figure:
#   * one idea per panel; one arm per row; arm order identical across F1/F3/F4;
#   * colour = arm family (navy classical, orange hybrid, green neural, grey
#     controls); the SIMON true-tuple diagnostic stays in the appendix table;
#   * every printed number is computed from the CSV tables, never typed in;
#   * counts / PAR-2 are printed in an aligned column outside the data area so
#     they cannot collide with bars, whiskers or markers.
# =============================================================================

# Arm definitions: raw method id -> (row label, colour). Grey = control /
# zero-recovery family; the true-tuple diagnostic is handled separately.
SPECK_ARMS: list[tuple[str, str, str]] = [
    ("hierarchical_neural_classical", "Neural-classical hybrid", C_ORANGE),
    ("hierarchical_neural_screened", "Screened neural", C_GREEN),
    ("hierarchical_neural_full", "Unscreened neural", C_PURPLE),
    ("random_sat", "Random tuples (1,024)", C_GREY),
    ("sat_only_4obs", "Unguided SAT (4 obs.)", C_GREY),
    ("sat_only_all_observations", "Unguided SAT (128 obs.)", C_GREY),
]
SIMON_ARMS: list[tuple[str, str, str]] = [
    ("fully_classical_8parent", "Classical DDT (8 parents)", C_NAVY),
    ("neural_classical_hybrid_8parent", "Bao-DDT hybrid (8 parents)", C_ORANGE),
    ("fully_neural_1parent", "Bao neural (1 parent)", C_GREEN),
    ("faithful_b1_hybrid", "B1-DDT hybrid", C_PURPLE),
    ("random_control_128", "Random tuples (128)", C_GREY),
    ("random_control_1024", "Random tuples (1,024)", C_GREY),
    ("sat_only_n4", "Unguided SAT (4 obs.)", C_GREY),
    ("sat_only_n128", "Unguided SAT (128 obs.)", C_GREY),
]
SIMON_DIAG: tuple[str, str, str] = ("oracle_2key_n4", "True-tuple diagnostic", C_NAVY)
METHOD_NAMES = {m: label for m, label, _ in SPECK_ARMS + SIMON_ARMS + [SIMON_DIAG]}

SPECK_TITLE = "(a) SPECK32/64, 7 rounds, N = 128"
SIMON_TITLE = "(b) SIMON32/64, 10 rounds, N = 64"
DIAG_TITLE = "(c) SIMON32/64 true-tuple diagnostic, N = 64 (not an attack arm)"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _arm_rows(
    rows: list[dict[str, str]],
    cipher: str,
    role: str,
    arm_defs: list[tuple[str, str, str]],
) -> list[tuple[tuple[str, str, str], dict[str, str]]]:
    """Return rows in plotting order; fail closed on method-set drift."""
    sel = {r["method"]: r for r in rows if r["cipher"] == cipher and r["role"] == role}
    expected = {d[0] for d in arm_defs}
    if set(sel) != expected:
        raise ValueError(
            f"Method-set drift for {cipher}/{role}: "
            f"missing={sorted(expected - set(sel))} extra={sorted(set(sel) - expected)}"
        )
    return [(d, sel[d[0]]) for d in arm_defs]


def _text_on(color: str) -> str:
    """Dark or white text, whichever contrasts better with a hex fill."""
    r, g, b = (int(color[i : i + 2], 16) / 255.0 for i in (1, 3, 5))
    return C_DARK if (0.299 * r + 0.587 * g + 0.114 * b) > 0.6 else "white"


def _arm_axes(
    fig: plt.Figure, specs: list[tuple[str, list[str]]], grid_axis: str = "x"
):
    """Stacked panels, one row per arm, rows top-to-bottom in the given order."""
    gs = fig.add_gridspec(
        len(specs), 1, height_ratios=[len(lbls) + 1.3 for _, lbls in specs]
    )
    axes = []
    for k, (title, labels) in enumerate(specs):
        ax = fig.add_subplot(gs[k, 0])
        format_axis(ax, grid_axis=grid_axis)
        ax.set_title(title, loc="left", fontsize=9.5, weight="bold", pad=4)
        ax.set_ylim(len(labels) - 0.45, -0.55)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8.2)
        ax.tick_params(axis="y", length=0)
        axes.append(ax)
    return axes


def _right_labels(ax: plt.Axes, texts: list[str], header: str = "") -> None:
    """Aligned label column to the right of the data area."""
    for i, t in enumerate(texts):
        ax.text(
            1.02,
            i,
            t,
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=8.0,
            color=C_DARK,
        )
    if header:
        ax.text(
            1.02,
            -0.55,
            header,
            transform=ax.get_yaxis_transform(),
            va="bottom",
            ha="left",
            fontsize=7.4,
            color=C_GREY,
            style="italic",
        )


# ----------------------------------------------------------------------------
# F1: verified recovery with marginal Wilson 95% intervals
# ----------------------------------------------------------------------------
def plot_fig1_attack_recovery(
    attack_summary_path: Path, output_figures_dir: Path
) -> tuple[Path, Path]:
    """Figure 1: verified recovery per arm (horizontal bars + Wilson 95% CI)."""
    summaries = _read_csv(attack_summary_path)
    speck = _arm_rows(summaries, "speck32_64", "autonomous_attack", SPECK_ARMS)
    simon = _arm_rows(summaries, "simon32_64", "autonomous_attack", SIMON_ARMS)
    fig = plt.figure(figsize=(7.16, 5.0), layout="constrained")
    axes = _arm_axes(
        fig,
        [
            (SPECK_TITLE, [d[1] for d, _ in speck]),
            (SIMON_TITLE, [d[1] for d, _ in simon]),
        ],
    )

    for ax, rows, is_diag in zip(axes, [speck, simon], [False, False]):
        texts = []
        for i, (d, s) in enumerate(rows):
            rate = float(s["recovery_rate"]) * 100.0
            lo = float(s["wilson_95_lower"]) * 100.0
            hi = float(s["wilson_95_upper"]) * 100.0
            if is_diag:
                ax.barh(
                    i,
                    rate,
                    height=0.58,
                    facecolor=BG_BLUE,
                    edgecolor=C_NAVY,
                    linewidth=1.2,
                    linestyle="--",
                    zorder=3,
                )
            else:
                ax.barh(
                    i,
                    rate,
                    height=0.58,
                    color=d[2],
                    edgecolor=C_DARK,
                    linewidth=0.8,
                    zorder=3,
                )
            ax.errorbar(
                rate,
                i,
                xerr=[[rate - lo], [hi - rate]],
                fmt="o",
                color=C_DARK,
                ms=3.0,
                capsize=2.5,
                elinewidth=1.0,
                capthick=1.0,
                zorder=5,
            )
            texts.append(f"{s['n_recovered']}/{s['n_assigned']}  ({rate:.1f}%)")
        ax.set_xlim(0, 100)
        ax.set_xticks(range(0, 101, 20))
        _right_labels(ax, texts, header="recovered / N")
    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=True)
    axes[-1].set_xlabel("Verified key recovery (%)")
    return save_figure_dual(fig, output_figures_dir, "fig1_attack_recovery")


# ----------------------------------------------------------------------------
# F2: cumulative verified recovery versus complete online seconds
# ----------------------------------------------------------------------------
def _step_xy(
    times: list[float], n: int, xmax: float
) -> tuple[list[float], list[float]]:
    """Exact empirical cumulative curve with the unconditional denominator n."""
    ts = sorted(times)
    xs = [0.0] + ts + [xmax]
    ys = [0.0] + [100.0 * (i + 1) / n for i in range(len(ts))] + [100.0 * len(ts) / n]
    return xs, ys


def plot_fig2_cumulative_recovery_time(
    attack_cells_path: Path, output_figures_dir: Path
) -> tuple[Path, Path]:
    """Figure 2: step curves from successful cell times, denominator fixed at N."""
    from matplotlib.lines import Line2D

    cells = _read_csv(attack_cells_path)
    xmax = 300.0

    def success_times(study: str, role: str, method: str) -> list[float]:
        out = []
        for c in cells:
            if c["study"] == study and c["role"] == role and c["method"] == method:
                if c["verified_recovery"] not in ("True", "False"):
                    raise ValueError(
                        f"Non-boolean verified_recovery: {c['verified_recovery']!r}"
                    )
                if c["verified_recovery"] == "True":
                    t = float(c["wall_seconds"])
                    if t > 300.0:
                        raise ValueError(
                            "Late recovery must not be counted as verified"
                        )
                    out.append(t)
        return out

    panels = [
        (
            "(a) SPECK32/64, 7 rounds, N = 128",
            "speck_seven_round_prospective",
            "autonomous_attack",
            SPECK_ARMS,
            128,
        ),
        (
            "(b) SIMON32/64, 10 rounds, N = 64",
            "simon_confirmation_replacement",
            "autonomous_attack",
            SIMON_ARMS,
            64,
        ),
    ]
    fig, axes = plt.subplots(
        1, 2, figsize=(7.16, 3.6), sharey=True, layout="constrained"
    )
    for ax, (title, study, role, arms, n) in zip(axes, panels):
        format_axis(ax, grid_axis="both")
        ax.set_title(title, loc="left", fontsize=9.0, weight="bold", pad=14)
        ax.axvline(300.0, color=C_DARK, linestyle="--", linewidth=0.9, zorder=2)
        ax.text(
            300.0,
            103.5,
            "300 s cutoff",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=C_DARK,
        )
        ax.set_xlim(0, xmax)
        ax.set_ylim(-3, 103)
        ax.set_xticks(range(0, 301, 100))
        ax.set_xlabel("Complete online time (s)")
        handles = []
        zero_names = []
        for method, label, color in arms:
            ts = success_times(study, role, method)
            if not ts:
                zero_names.append(label)
                continue
            xs, ys = _step_xy(ts, n, xmax)
            ls = "--" if role == "diagnostic_control" else "-"
            ax.step(
                xs, ys, where="post", color=color, linewidth=1.7, linestyle=ls, zorder=4
            )
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linewidth=1.7,
                    linestyle=ls,
                    label=f"{label} ({len(ts)}/{n})",
                )
            )
        if zero_names:
            ax.plot(
                [0, xmax],
                [0, 0],
                color=C_GREY,
                linewidth=1.3,
                linestyle=(0, (4, 2)),
                zorder=3,
            )
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=C_GREY,
                    linewidth=1.3,
                    linestyle=(0, (4, 2)),
                    label=f"No recovery (0/{n}): {len(zero_names)} arms",
                )
            )
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.24),
            frameon=False,
            fontsize=7.4,
            ncol=1,
            handlelength=2.2,
            borderaxespad=0.0,
        )
    axes[0].set_ylabel("Verified recovery, share of all N keys (%)")
    return save_figure_dual(fig, output_figures_dir, "fig2_cumulative_recovery_time")


# ----------------------------------------------------------------------------
# F3: unconditional operational cost (PAR-2 with recovery rate)
# ----------------------------------------------------------------------------
def plot_fig3_operational_cost(
    attack_summary_path: Path, output_figures_dir: Path
) -> tuple[Path, Path]:
    """Figure 3: PAR-2 lollipops per arm; recovery rate printed beside each row."""
    summaries = _read_csv(attack_summary_path)
    speck = _arm_rows(summaries, "speck32_64", "autonomous_attack", SPECK_ARMS)
    simon = _arm_rows(summaries, "simon32_64", "autonomous_attack", SIMON_ARMS)
    fig = plt.figure(figsize=(7.16, 5.0), layout="constrained")
    axes = _arm_axes(
        fig,
        [
            (SPECK_TITLE, [d[1] for d, _ in speck]),
            (SIMON_TITLE, [d[1] for d, _ in simon]),
        ],
    )
    for ax, rows, is_diag in zip(axes, [speck, simon], [False, False]):
        ax.axvline(600, color=C_DARK, linestyle=":", linewidth=1.0, zorder=2)
        texts = []
        for i, (d, s) in enumerate(rows):
            par2 = float(s["par2_seconds"])
            ax.plot(
                [0, par2],
                [i, i],
                color=C_GRID,
                linewidth=2.2,
                zorder=2,
                solid_capstyle="butt",
            )
            if is_diag:
                ax.scatter(
                    [par2],
                    [i],
                    s=66,
                    facecolor="white",
                    edgecolor=C_NAVY,
                    linewidth=1.6,
                    marker="o",
                    zorder=5,
                )
            else:
                ax.scatter(
                    [par2],
                    [i],
                    s=66,
                    color=d[2],
                    edgecolor=C_DARK,
                    linewidth=0.9,
                    marker="o",
                    zorder=5,
                )
            rate = float(s["recovery_rate"]) * 100.0
            texts.append(f"PAR-2 {par2:5.0f} s  |  {rate:5.1f}% recovered")
        ax.set_xlim(0, 640)
        ax.set_xticks(range(0, 601, 100))
        _right_labels(ax, texts)
    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=True)
    axes[-1].set_xlabel("PAR-2 (s)")
    return save_figure_dual(fig, output_figures_dir, "fig3_operational_cost")


# ----------------------------------------------------------------------------
# F4: failure decomposition (every bar sums to the full N)
# ----------------------------------------------------------------------------
_CAT_COLOR = {
    "verified_recovery": C_GREEN,
    "ranking_miss": C_ORANGE,
    "ranking_timeout": "#d08048",
    "budget_exhausted": C_PURPLE,
    "unresolved_solving": "#c0b068",
    "queue_exhausted": "#b0b8d0",
    "global_deadline": C_GREY,
}
_CAT_NAME = {
    "verified_recovery": "Verified recovery",
    "ranking_miss": "Ranking miss (key outside shortlist)",
    "ranking_timeout": "Ranking allowance timeout (no SAT)",
    "budget_exhausted": "Budget exhausted during ranking (no SAT)",
    "unresolved_solving": "Candidate solving unresolved",
    "queue_exhausted": "Candidate queue exhausted",
    "global_deadline": "Global deadline reached",
}
_CAT_ORDER = [
    "verified_recovery",
    "ranking_miss",
    "ranking_timeout",
    "budget_exhausted",
    "unresolved_solving",
    "queue_exhausted",
    "global_deadline",
]
_SPECK_RAW = {
    "verified_recovery": "verified_recovery",
    "ranking": "ranking_miss",
    "candidates_exhausted_without_unresolved_attempts": "queue_exhausted",
}
_SIMON_RAW = {
    "verified_recovery": "verified_recovery",
    "ranking_timeout": "ranking_timeout",
    "candidate_unresolved": "unresolved_solving",
    "queue_exhausted": "queue_exhausted",
    "global_deadline": "global_deadline",
}


def _category_counts(failures, study, role, arms, raw_map) -> dict[str, dict[str, int]]:
    """Map raw outcome labels to plotting categories; fail closed on unknown labels."""
    out: dict[str, dict[str, int]] = {a[0]: {} for a in arms}
    n_of: dict[str, int] = {}
    seen = set()
    for r in failures:
        if r["study"] != study or r["role"] != role:
            continue
        m = r["method"]
        if m not in out:
            raise ValueError(f"Unexpected method in failure table: {m}")
        raw = r["outcome_category"]
        if (m, raw) in seen:
            raise ValueError("Duplicate failure-category row")
        seen.add((m, raw))
        count, n = int(r["count"]), int(r["n_assigned"])
        if count < 0 or n <= 0 or (m in n_of and n_of[m] != n):
            raise ValueError("Invalid outcome counts or inconsistent denominator")
        if (
            raw == "global_timeout"
        ):  # SPECK raw label: ranking-budget failure vs solver deadline
            cat = (
                "budget_exhausted"
                if m == "hierarchical_neural_full"
                else "global_deadline"
            )
        elif raw in raw_map:
            cat = raw_map[raw]
        else:
            raise ValueError(f"Unmapped outcome category {raw!r} for {m}")
        out[m][cat] = out[m].get(cat, 0) + count
        n_of[m] = n
    for m, cats in out.items():
        if sum(cats.values()) != n_of.get(m):
            raise ValueError(f"Failure counts for {m} do not sum to N")
    return out


def plot_fig4_failure_decomposition(
    attack_failures_path: Path, output_figures_dir: Path
) -> tuple[Path, Path]:
    """Figure 4: two count matrices, exposing even single-key outcomes."""
    failures = _read_csv(attack_failures_path)
    speck_map = _category_counts(
        failures,
        "speck_seven_round_prospective",
        "autonomous_attack",
        SPECK_ARMS,
        _SPECK_RAW,
    )
    simon_map = _category_counts(
        failures,
        "simon_confirmation_replacement",
        "autonomous_attack",
        SIMON_ARMS,
        _SIMON_RAW,
    )
    columns = [
        ("Recovered", ["verified_recovery"], C_GREEN),
        ("Shortlist\nmiss", ["ranking_miss"], C_ORANGE),
        ("Ranking\nlimit", ["ranking_timeout", "budget_exhausted"], C_PURPLE),
        ("Unresolved\nSAT", ["unresolved_solving"], "#ab9155"),
        ("Queue\nexhausted", ["queue_exhausted"], C_NAVY),
        ("Global\nlimit", ["global_deadline"], C_GREY),
    ]
    fig = plt.figure(figsize=(7.16, 4.9), layout="constrained")
    axes = _arm_axes(
        fig,
        [
            (SPECK_TITLE, [d[1] for d in SPECK_ARMS]),
            (SIMON_TITLE, [d[1] for d in SIMON_ARMS]),
        ],
        grid_axis="",
    )
    for ax, arms, cmap, n in zip(
        axes, [SPECK_ARMS, SIMON_ARMS], [speck_map, simon_map], [128, 64]
    ):
        for i, arm in enumerate(arms):
            counts = [
                sum(cmap[arm[0]].get(cat, 0) for cat in cats) for _, cats, _ in columns
            ]
            if sum(counts) != n:
                raise ValueError("Displayed outcomes do not sum to the assigned cohort")
            for j, (cnt, (_, _, color)) in enumerate(zip(counts, columns, strict=True)):
                rgb = np.array(matplotlib.colors.to_rgb(color))
                fraction = cnt / n
                fill = (
                    tuple(1 - (1 - rgb) * (0.13 + 0.87 * fraction))
                    if cnt
                    else "#f5f5f5"
                )
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor=fill,
                        edgecolor="white",
                        linewidth=2,
                    )
                )
                ax.text(
                    j,
                    i,
                    str(cnt) if cnt else "-",
                    ha="center",
                    va="center",
                    fontsize=9,
                    weight="bold" if cnt else "normal",
                    color="white" if fraction > 0.65 else C_DARK if cnt else "#aaaaaa",
                )
        ax.set_xlim(-0.5, len(columns) - 0.5)
        ax.set_ylim(len(arms) - 0.5, -0.5)
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels([label for label, _, _ in columns], fontsize=8)
        ax.xaxis.tick_top()
        ax.tick_params(axis="both", length=0)
        ax.set_title(
            ax.get_title(loc="left"), loc="left", fontsize=9.5, weight="bold", pad=29
        )
        for spine in ax.spines.values():
            spine.set_visible(False)
    return save_figure_dual(fig, output_figures_dir, "fig4_failure_decomposition")


# ----------------------------------------------------------------------------
# F5: cross-cipher residual-depth map (counts/8 at every R, m, O)
# ----------------------------------------------------------------------------
_RAMP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "recovery_ramp", ["#f4f6fa", C_NAVY]
)


def _fmt_seconds(t: float) -> str:
    return f"{t:.1f} s" if t < 100 else f"{t:.0f} s"


def plot_fig5_depth_map(
    depth_summary_path: Path, output_figures_dir: Path
) -> tuple[Path, Path]:
    """Figure 5: one heat-table per cipher; rows = (R, m), columns = O."""
    summaries = _read_csv(depth_summary_path)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 4.0), layout="constrained")
    for ax, cipher, title in zip(
        axes,
        ["speck32_64", "simon32_64"],
        [
            "(a) SPECK32/64, 8 demo keys per cell",
            "(b) SIMON32/64, 8 demo keys per cell",
        ],
    ):
        rows = [s for s in summaries if s["cipher"] == cipher]
        cfgs = sorted(
            {
                (
                    int(s["rounds"]),
                    int(s["assumed_keys_m"]),
                    int(s["nominal_residual_depth_d"]),
                )
                for s in rows
            }
        )
        if len(cfgs) != 6 or len(rows) != 12:
            raise ValueError(f"Depth grid incomplete for {cipher}")
        obs = [4, 128]
        ax.set_title(title, loc="left", fontsize=9.5, weight="bold", pad=22)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(len(cfgs) - 0.5, -0.5)
        for ri, (r, m, d) in enumerate(cfgs):
            for ci, o in enumerate(obs):
                s = next(
                    x
                    for x in rows
                    if int(x["rounds"]) == r
                    and int(x["assumed_keys_m"]) == m
                    and int(x["observations_O"]) == o
                )
                k = int(s["verified_recovery_count"])
                n = int(s["n_instances"])
                if n != 8:
                    raise ValueError("Depth cell n != 8")
                ax.add_patch(
                    plt.Rectangle(
                        (ci - 0.5, ri - 0.5),
                        1,
                        1,
                        facecolor=_RAMP(k / n),
                        edgecolor="white",
                        linewidth=2.0,
                        zorder=2,
                    )
                )
                tcol = "white" if k / n > 0.55 else C_DARK
                ax.text(
                    ci,
                    ri - 0.12,
                    f"{k}/{n}",
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    weight="bold",
                    color=tcol,
                    zorder=4,
                )
                ax.text(
                    ci,
                    ri + 0.24,
                    _fmt_seconds(float(s["median_parent_seconds"])),
                    ha="center",
                    va="center",
                    fontsize=7.4,
                    color=tcol,
                    zorder=4,
                )
        for ri in range(1, len(cfgs)):  # separator between round counts
            if cfgs[ri][0] != cfgs[ri - 1][0]:
                ax.axhline(ri - 0.5, color=C_DARK, linewidth=0.9, zorder=5)
        ax.set_yticks(range(len(cfgs)))
        ax.set_yticklabels(
            [f"R = {r}, m = {m}  (d = {d})" for r, m, d in cfgs], fontsize=8.2
        )
        ax.xaxis.tick_top()
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["O = 4", "O = 128"], fontsize=8.2)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    return save_figure_dual(fig, output_figures_dir, "fig5_depth_map")


# ----------------------------------------------------------------------------
# F6: matched G/X contrast per key; F6b: all profiles and controls (appendix)
# ----------------------------------------------------------------------------
def _backend_rounds(cells: list[dict[str, str]], cipher: str) -> int:
    rs = {
        int(c["rounds"])
        for c in cells
        if c["cipher"] == cipher
        and int(c["trailing_keys_m"]) in (1, 2)
        and int(c["observations_O"]) == 4
    }
    if len(rs) != 1:
        raise ValueError(f"Ambiguous backend rounds for {cipher}: {rs}")
    return rs.pop()


def plot_fig6_backend_treatments(
    backend_summary_path: Path, backend_cells_path: Path, output_figures_dir: Path
) -> tuple[Path, Path]:
    """Figure 6: matched per-key native-XOR Gauss (G) versus no-Gauss (X) at m = 1, 2."""
    from matplotlib.lines import Line2D

    cells = _read_csv(backend_cells_path)
    fig, axes = plt.subplots(
        1, 2, figsize=(7.16, 3.9), sharey=True, layout="constrained"
    )
    for ax, cipher, name, tag in zip(
        axes, ["speck32_64", "simon32_64"], ["SPECK32/64", "SIMON32/64"], ["(a)", "(b)"]
    ):
        rounds = _backend_rounds(cells, cipher)
        format_axis(ax, grid_axis="y")
        ax.set_title(
            f"{tag} {name}, {rounds} rounds, O = 4",
            loc="left",
            fontsize=9.5,
            weight="bold",
            pad=4,
        )
        ax.set_yscale("log")
        ax.set_ylim(2.0, 500)
        ax.axhline(300.0, color=C_DARK, linestyle="--", linewidth=0.9, zorder=1)
        ax.text(
            1.55,
            300.0 * 1.12,
            "300 s cutoff",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=C_DARK,
        )
        ticks, ticklabels = [], []
        for gi, m in enumerate([1, 2]):
            xg, xx = gi * 2.2, gi * 2.2 + 1.0
            counts: dict[str, int] = {}
            for kid in range(4):
                pair = {}
                for prof, x in (("cms_native_gauss", xg), ("cms_native_no_gauss", xx)):
                    sel = [
                        c
                        for c in cells
                        if c["cipher"] == cipher
                        and int(c["trailing_keys_m"]) == m
                        and int(c["observations_O"]) == 4
                        and c["profile"] == prof
                        and int(c["key_id"]) == kid
                    ]
                    if len(sel) != 1:
                        raise ValueError("Backend cell missing/duplicated for F6")
                    c = sel[0]
                    if c["verified_recovery"] not in ("True", "False"):
                        raise ValueError("Non-boolean backend verified flag")
                    ok = c["verified_recovery"] == "True"
                    if not ok and c["outcome"] != "global_deadline":
                        raise ValueError("Unexpected non-deadline failure in G/X cell")
                    pair[prof] = (
                        x + (kid - 1.5) * 0.07,
                        float(c["online_seconds"]) if ok else 300.0,
                        ok,
                    )
                    counts[prof] = counts.get(prof, 0) + int(ok)
                (x1, y1, _), (x2, y2, _) = (
                    pair["cms_native_gauss"],
                    pair["cms_native_no_gauss"],
                )
                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color=C_GREY,
                    alpha=0.55,
                    linewidth=0.9,
                    zorder=2,
                )
                for prof, color, mk in (
                    ("cms_native_gauss", C_ORANGE, "o"),
                    ("cms_native_no_gauss", C_GREEN, "s"),
                ):
                    px, py, ok = pair[prof]
                    ax.scatter(
                        [px],
                        [py],
                        color=color,
                        marker=mk if ok else "X",
                        s=42 if ok else 54,
                        edgecolor=C_DARK,
                        linewidth=0.7,
                        zorder=5,
                    )
            ticks += [xg, xx]
            ticklabels += [
                f"G\n{counts['cms_native_gauss']}/4 solved",
                f"X\n{counts['cms_native_no_gauss']}/4 solved",
            ]
            ax.text(
                (xg + xx) / 2,
                -0.17,
                f"m = {m} trailing key{'s' if m > 1 else ''} assumed",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8.2,
                weight="bold",
            )
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels, fontsize=7.8)
        ax.set_xlim(-0.6, 3.8)
    axes[0].set_ylabel("Complete online time (s, log scale)")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=C_ORANGE,
            markeredgecolor=C_DARK,
            markersize=6,
            label="G: native XOR, matrix capacity 5",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=C_GREEN,
            markeredgecolor=C_DARK,
            markersize=6,
            label="X: native XOR, matrix capacity 0",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            color="none",
            markerfacecolor="white",
            markeredgecolor=C_DARK,
            markersize=6,
            label="unsolved at 300 s (plotted at cutoff)",
        ),
        Line2D(
            [0],
            [0],
            color=C_GREY,
            alpha=0.7,
            linewidth=1.0,
            label="line joins the same key",
        ),
    ]
    fig.legend(
        handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=7.8
    )
    return save_figure_dual(fig, output_figures_dir, "fig6_backend_treatments")


_BACKEND_PROFILES = [
    ("cms_native_gauss", "CMS native XOR (G)"),
    ("cms_native_no_gauss", "CMS native XOR (X)"),
    ("cms_native_default", "CMS native defaults (N)"),
    ("cms_wrapper", "CMS Python wrapper (W)"),
    ("cms_cnf_no_gauss", "CMS pure CNF (C)"),
    ("cadical_cnf", "CaDiCaL pure CNF (K)"),
]
# Paper-facing labels for the appendix figure (no internal profile letters).
_BACKEND_PROFILE_PAPER_LABELS = {
    "cms_native_gauss": "CMS native XOR,\nmatrices allowed (5)",
    "cms_native_no_gauss": "CMS native XOR,\nmatrices disabled (0)",
    "cms_native_default": "CMS native XOR,\nsolver defaults",
    "cms_wrapper": "CMS incremental\nPython wrapper",
    "cms_cnf_no_gauss": "CMS, XOR as CNF",
    "cadical_cnf": "CaDiCaL, XOR as CNF",
}
_BACKEND_CONDS = [
    ("o4_m0", 0, 4),
    ("o128_m0", 0, 128),
    ("o4_m1", 1, 4),
    ("o4_m2", 2, 4),
]


def plot_fig6b_backend_profiles(
    backend_summary_path: Path, backend_cells_path: Path, output_figures_dir: Path
) -> tuple[Path, Path]:
    """Appendix F6b: all six profiles and the unguided controls, one cell per group."""
    summ = _read_csv(backend_summary_path)
    cells = _read_csv(backend_cells_path)
    fig, axes = plt.subplots(
        1, 2, figsize=(7.16, 4.1), sharey=True, layout="constrained"
    )
    for ax, cipher, name, tag in zip(
        axes, ["speck32_64", "simon32_64"], ["SPECK32/64", "SIMON32/64"], ["(a)", "(b)"]
    ):
        rounds = {int(c["rounds"]) for c in cells if c["cipher"] == cipher}
        if len(rounds) != 1:
            raise ValueError("Ambiguous rounds in backend cells")
        ax.set_title(
            f"{tag} {name}, {rounds.pop()} rounds",
            loc="left",
            fontsize=9.5,
            weight="bold",
            pad=28,
        )
        ax.set_xlim(-0.5, len(_BACKEND_CONDS) - 0.5)
        ax.set_ylim(len(_BACKEND_PROFILES) - 0.5, -0.5)
        for ri, (prof, _) in enumerate(_BACKEND_PROFILES):
            for ci, (cond, _, _) in enumerate(_BACKEND_CONDS):
                g = [
                    s
                    for s in summ
                    if s["cipher"] == cipher
                    and s["condition"] == f"{cipher}_{cond}"
                    and s["profile"] == prof
                ]
                if len(g) > 1:
                    raise ValueError("Duplicate backend summary group")
                if not g:
                    ax.add_patch(
                        plt.Rectangle(
                            (ci - 0.5, ri - 0.5),
                            1,
                            1,
                            facecolor="white",
                            edgecolor=C_GRID,
                            linewidth=1.0,
                            hatch="////",
                            zorder=2,
                        )
                    )
                    ax.text(
                        ci,
                        ri,
                        "not run",
                        ha="center",
                        va="center",
                        fontsize=6.8,
                        color=C_GREY,
                        style="italic",
                        zorder=4,
                        bbox=dict(facecolor="white", edgecolor="none", pad=1.0),
                    )
                    continue
                s = g[0]
                n_as = int(s["n_assigned"])
                k = int(s["verified_recoveries"])
                n_done = int(s["n_completed_bounded_trials"])
                par2 = float(s["scheduled_cell_par2_seconds"])
                ax.add_patch(
                    plt.Rectangle(
                        (ci - 0.5, ri - 0.5),
                        1,
                        1,
                        facecolor=_RAMP(k / n_as),
                        edgecolor="white",
                        linewidth=2.0,
                        zorder=2,
                    )
                )
                tcol = "white" if k / n_as > 0.55 else C_DARK
                dag = "\u2020" if n_done < n_as else ""
                ax.text(
                    ci,
                    ri - 0.14,
                    f"{k}/{n_done}{dag}",
                    ha="center",
                    va="center",
                    fontsize=9.2,
                    weight="bold",
                    color=tcol,
                    zorder=4,
                )
                ax.text(
                    ci,
                    ri + 0.25,
                    f"PAR-2 {par2:.0f} s",
                    ha="center",
                    va="center",
                    fontsize=7.0,
                    color=tcol,
                    zorder=4,
                )
        ax.set_yticks(range(len(_BACKEND_PROFILES)))
        ax.set_yticklabels(
            [_BACKEND_PROFILE_PAPER_LABELS[p] for p, _ in _BACKEND_PROFILES],
            fontsize=7.6,
        )
        ax.xaxis.tick_top()
        ax.set_xticks(range(len(_BACKEND_CONDS)))
        ax.set_xticklabels(
            [
                ("no key word" if m == 0 else f"{m} true key word{'s' if m > 1 else ''}")
                + f"\n{o} blocks"
                for _, m, o in _BACKEND_CONDS
            ],
            fontsize=7.4,
        )
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    return save_figure_dual(fig, output_figures_dir, "fig6b_backend_profiles_appendix")


# =============================================================================
# METADATA & ARTIFACT PACKAGING
# =============================================================================


def build_metadata(
    output_dir: Path,
    table_paths: list[Path],
    figure_paths: list[Path],
) -> tuple[Path, Path, Path, Path]:
    """Generate CAPTIONS.md, FIGURE_DATA_MAP.md, DATA_DICTIONARY.md, figure_manifest.json."""
    meta_dir = output_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # 1. CAPTIONS.md
    captions_path = meta_dir / "CAPTIONS.md"
    captions_content = """# Figure Captions & Contributing Data Mappings

### Figure 1: Attack Recovery Rates with 95% Wilson Score Intervals
- **File:** `figures/fig1_attack_recovery.pdf` (and `.png`)
- **Contributing CSV:** `tables/attack_summary.csv`
- **Source JSON:** the two raw attack ledgers, independently reconciled against the pinned analyses.
- **Evidence Role:** Confirmatory prospective comparison (SPECK D-031, $N=128$; SIMON D-041, $N=64$).
- **Denominators:** Exact unpooled cohort totals ($N=128$ for SPECK, $N=64$ for SIMON).
- **Uncertainty:** Marginal Wilson 95% score intervals with score inversion.
- **Layout:** (a) SPECK and (b) SIMON autonomous methods. Counts are printed in a column right of each panel. The true-tuple diagnostic stays in the tables and appendix note, outside the main figures.
- **Limitations:** Autonomous attack arms exclude oracle diagnostic. SPECK $111/128$ is a neural Stage1 + classical Stage2 hybrid. SIMON classical ($53/64$) vs hybrid ($49/64$) recovery difference is statistically inconclusive ($p=0.34375$). SIMON B1 zero recovery reflects operational ranking throughput failure under the frozen allowance, not measured inaccuracy.

---

### Figure 2: Cumulative Verified Recovery vs Complete Online Time
- **File:** `figures/fig2_cumulative_recovery_time.pdf` (and `.png`)
- **Contributing CSV:** `tables/attack_cells.csv`
- **Source JSON:** Raw cell execution records (`seven_round_prospective_results.json`, `simon_confirmation_comparators.json`).
- **Evidence Role:** Operational efficiency profile under common resource bounds.
- **Denominators:** Fixed unconditional denominator ($N=128$ for SPECK, $N=64$ for SIMON); unsuccessful instances remain in denominator at all $t$.
- **Layout:** (a) SPECK and (b) SIMON autonomous methods. Arms with zero recoveries share one grey baseline per panel; every autonomous method is listed in Figures 1, 3 and 4.
- **Limitations:** Step curves are empirical realizations, not survival estimators. SPECK records cooperative cutoff overshoot; SIMON uses a hard watchdog. Vertical dashed line indicates nominal 300 s cutoff.

---

### Figure 3: Unconditional Operational Cost (Recovery Rate vs PAR-2)
- **File:** `figures/fig3_operational_cost.pdf` (and `.png`)
- **Contributing CSV:** `tables/attack_summary.csv`
- **Source JSON:** `seven_round_prospective_analysis.json`, `simon_confirmation_analysis.json`.
- **Evidence Role:** Resource-accounting comparison combining success rate and execution cost.
- **Layout:** Dot plot per autonomous method (same row order as Figures 1 and 4); recovery rate printed beside each row. The diagnostic is tabulated separately.
- **Metric Definition:** $\\text{PAR-2} = \\frac{1}{N} [\\sum_{i \\in \\text{recovered}} t_i + \\sum_{i \\notin \\text{recovered}} 600.0]$.
- **Limitations:** Non-recovery penalty is fixed at $2 \\times 300 = 600\\,\\text{s}$. No claim of speed advantage conditional on success only.

---

### Figure 4: Attack Failure Decomposition
- **File:** `figures/fig4_failure_decomposition.pdf` (and `.png`)
- **Contributing CSV:** `tables/attack_failures.csv`
- **Source JSON:** Raw failure causes and domain classifications in prospective result files.
- **Evidence Role:** Recorded outcome decomposition across ranking and constraint solving; not a causal explanation.
- **Layout:** Two outcome-count matrices, (a) SPECK and (b) SIMON. Each nonzero count is printed; a dash means zero. Within each outcome column, darker shading denotes a larger share of that method's assigned cohort. "Ranking limit" combines recorded ranking-allowance timeout and ranking-budget exhaustion for display; the CSV preserves those categories separately.
- **Completeness:** Every matrix row sums exactly to $N=128$ for SPECK or $N=64$ for SIMON. The diagnostic remains in the separate appendix table.
- **Findings:** SIMON B1 failure is entirely ranking timeout (64/64); SPECK full neural failure is entirely budget exhaustion during ranking (128/128); unguided SAT failures are 100% solver deadlines.

---

### Figure 5: Cross-Cipher Residual-Depth Scaling Map
- **File:** `figures/fig5_depth_map.pdf` (and `.png`)
- **Contributing CSV:** `tables/depth_summary.csv`, `tables/depth_cells.csv`
- **Source JSON:** `results/phase11_scaling_diagnostics/cross_cipher_residual_depth_analysis_reviewed_v2.json`.
- **Evidence Role:** Empirical scaling map across nominal residual depths $d = R - m$.
- **Layout:** Heat-table per cipher; rows are tested (R, m) pairs with nominal d = R - m, columns are O = 4 and O = 128; each cell prints recoveries/8 and the median parent time over all 8 keys (timeouts included).
- **Sample Size:** $n=8$ demo keys per cell.
- **Notation:** O counts known plaintext/ciphertext observations (4 = two direct pairs; 128 = 64 pairs). R is total rounds, m the number of true assumed trailing subkeys, and d = R - m is nominal residual depth. Counts and median parent seconds are shown inside cells; shading denotes recovery fraction. Caption prose is provided here, not embedded beneath the image.
- **Limitations:** Descriptive matched-key observations under a 300 s cutoff. No claim of universal phase boundary, asymptotic law, or physical carry/AND mechanisms. Formula clause counts scale linearly with observations $O$.

---

### Figure 6: Controlled Backend Diagnostic (Gauss vs No-Gauss & Profiles)
- **File:** `figures/fig6_backend_treatments.pdf` (and `.png`)
- **Panels:** matched per-key native-XOR Gauss (G) versus no-Gauss (X) at m = 1 and m = 2, O = 4, for each cipher (rounds read from the data: SPECK 7, SIMON 10). Unsolved cells are plotted at the 300 s cutoff with a distinct marker.
- **Contributing CSV:** `tables/backend_summary.csv`, `tables/backend_cells.csv`, `tables/backend_paired_contrasts.csv`
- **Source JSON:** `results/backend_comparison/backend_analysis_20261004.json`.
- **Evidence Role:** Configured matrix-capacity comparison on native-XOR formulas. G and X set maximum matrix counts to 5 and 0, respectively; configured capacity is not proof of actual elimination on every cell.
- **Sample Size:** $n=4$ demo keys per condition ($N=160$ total assigned cells).
- **Findings:** At $m=1$, G recovered 4/4 for both ciphers; X recovered 2/4 for SPECK and 0/4 for SIMON. All selected $m=2$ profiles recovered, with costs differing by profile. Unguided $m=0$ has 0/63 completed trials and one unavailable infrastructure interruption. No universal best-backend claim. True trailing-key units are diagnostic and omit ranking costs.

---

### Figure 6b (appendix): All Backend Profiles and Unguided Controls
- **File:** `figures/fig6b_backend_profiles_appendix.pdf` (and `.png`)
- **Contributing CSV:** `tables/backend_summary.csv`, `tables/backend_cells.csv`
- **Content:** every profile in every tested condition (m = 0 O = 4, m = 0 O = 128, m = 1, m = 2), including the unfavourable CNF/CaDiCaL and m = 2 cost results and the unguided m = 0 controls (four profiles). Cells print verified/completed bounded trials and scheduled-cell PAR-2.
- **Limitations:** 4 keys per cell; the one interrupted SIMON O = 128 / m = 0 wrapper trial is unavailable (0/3 completed), and its elapsed interruption time is not a solver time. Descriptive only; no winner is claimed.
- **Notation:** O counts known observations. The first line in a cell is verified/completed trials; the second is scheduled-cell PAR-2, which retains all four assigned trials and the fixed 600 s non-recovery penalty. A dagger marks the unavailable interrupted trial; hatched cells were not run. G/X configure native matrix capacities 5/0; N uses native defaults; W uses the Python wrapper; C uses pure CNF with CMS; K uses the identical pure CNF with CaDiCaL. These explanations belong in the paper caption.
"""
    with open(captions_path, "w", encoding="utf-8") as f:
        f.write(captions_content)

    # 2. DATA_DICTIONARY.md
    dict_path = meta_dir / "DATA_DICTIONARY.md"
    dict_content = """# Traceable CSV Data Dictionary

### 1. `attack_cells.csv` (1,344 rows)
- `study`: Identifier of prospective study (`speck_seven_round_prospective` or `simon_confirmation_replacement`).
- `cipher`: Cryptographic primitive evaluated (`speck32_64` or `simon32_64`).
- `instance_id`: String instance identifier.
- `key_id`: Integer instance index (0..127 for SPECK, 0..63 for SIMON).
- `method`: Identifier of evaluated pipeline arm.
- `role`: Protocol role (`autonomous_attack` or `diagnostic_control`).
- `verified_recovery`: Boolean flag (true/false) denoting full key recovery passing fresh-pair verification.
- `wall_seconds`: Complete elapsed online wall-clock time in seconds.
- `outcome`: Specific execution outcome (`verified_recovery`, `ranking_timeout`, `queue_exhausted`, `candidate_unresolved`, `global_deadline`, `budget_exhausted`).
- `candidate_attempts`: Number of candidate subkey hypotheses tested against exact solver.
- `ranking_seconds`: Elapsed time in ranking/screening stage.
- `solver_seconds`: Recorded solver/component time, including valid unrecovered trials. Zero with status `not_entered` identifies a ranking failure before SAT; missing interrupted telemetry is empty with status `unavailable`.
- `solver_seconds_available`: Boolean indicating whether solver time is recorded.
- `late_recovery_flag`: Boolean flag indicating whether key was recovered after nominal cutoff.

### 2. `attack_summary.csv` (15 rows)
- `study`, `cipher`, `method`, `role`: Primary grouping keys.
- `n_assigned`: Total instances assigned to method (128 for SPECK, 64 for SIMON).
- `n_recovered`: Count of verified recovered instances.
- `recovery_rate`: Verified recovery proportion $\\hat{p} = \\text{n\\_recovered} / \\text{n\\_assigned}$.
- `wilson_95_lower`, `wilson_95_upper`: Lower and upper bounds of 95% Wilson score interval.
- `median_wall_seconds`, `mean_wall_seconds`: Unconditional summary statistics of online seconds.
- `par2_seconds`: Penalized Average Runtime with 2x penalty (600 s for failures).
- `mean_candidate_attempts`, `median_candidate_attempts`: Summary statistics of candidate queries.

### 3. `attack_paired_contrasts.csv` (17 rows)
- `study`, `cipher`: Context of comparison.
- `method_a`, `method_b`: Compared methods (Method A minus Method B).
- `role`: Frozen primary/secondary status or explicitly exploratory comparison.
- `multiplicity_adjustment`, `multiplicity_family`, `holm_adjusted_p`: Executed Holm correction within the frozen SPECK secondary or SIMON primary family; unadjusted contrasts have an empty adjusted p-value. Marginal and paired intervals are pointwise 95%, not simultaneous.
- `n_pairs`: Number of matched attack instances.
- `n11_both_succeed`: Instances where both Method A and Method B succeeded.
- `n10_a_only`: Discordant instances where only Method A succeeded.
- `n01_b_only`: Discordant instances where only Method B succeeded.
- `n00_neither`: Instances where neither method succeeded.
- `paired_difference`: Difference in proportions $(n_{10} - n_{01}) / n_{\\text{pairs}}$.
- `newcombe_95_lower`, `newcombe_95_upper`: 95% confidence interval using Newcombe Method 10.
- `exact_mcnemar_p`: Exact two-sided binomial McNemar test p-value.

### 4. `attack_failures.csv` (37 rows)
- `study`, `cipher`, `method`, `role`: Primary grouping keys.
- `outcome_category`: Mutually exclusive outcome category.
- `count`: Number of instances in this category.
- `n_assigned`: Total instances assigned to method.
- `fraction_of_n`: Proportion of total instances $\\text{count} / \\text{n\\_assigned}$.

### 5. `depth_cells.csv` (192 rows)
- `cipher`: Primitive evaluated (`speck32_64` or `simon32_64`).
- `rounds`: Total rounds evaluated ($R$).
- `assumed_keys_m`: Assumed trailing subkeys ($m$).
- `nominal_residual_depth_d`: Nominal depth $d = R - m$.
- `observations_O`: Known plaintext/ciphertext observation count ($O \\in \\{4, 128\\}$), equal to two or 64 direct differential pairs.
- `key_id`: Integer key index (0..7).
- `outcome`: Trial outcome (`verified_recovery` or `parent_watchdog_timeout`).
- `verified_recovery`: Boolean flag.
- `parent_wall_seconds`: Parent watchdog wall time in seconds.
- `solve_duration_seconds`: Clean solver execution time (empty string if unavailable on timeout).
- `solve_duration_available`: Boolean indicating availability.
- `variables_count`, `cnf_clauses_count`, `xor_clauses_count`: DIMACS formula metrics.

### 6. `depth_summary.csv` (24 rows)
- `cipher`, `rounds`, `assumed_keys_m`, `nominal_residual_depth_d`, `observations_O`: Group keys.
- `n_instances`: Number of demo instances ($n=8$).
- `verified_recovery_count`, `verified_recovery_rate`: Recovery performance.
- `median_parent_seconds`, `iqr_parent_seconds`: Parent runtime statistics.
- `median_solve_seconds`: Median solve time conditional on verified recovery.
- `solve_median_scope`: Explicit scope of solve time median (`completed_verified_recoveries_only`).
- `par2_seconds`: PAR-2 score with 600 s non-recovery penalty.
- `variables_count`, `cnf_clauses_count`, `xor_clauses_count`: Reconstructed formula metrics.

### 7. `backend_cells.csv` (160 rows)
- `cell_id`, `case_id`: Identifier strings.
- `cipher`, `rounds`, `trailing_keys_m`, `observations_O`, `profile`: Experimental cell attributes.
- `key_id`: Integer key index (0..3).
- `outcome`: Trial outcome (`recovered`, `global_deadline`, `environment_rejected`).
- `verified_recovery`: Boolean flag.
- `online_seconds`: Complete bounded online runtime, including construction/conversion/load/solve/decode/verification (empty if interrupted).
- `observed_elapsed_seconds`: Actual recorded elapsed runtime, including the interrupted observation; this is not a completed solver trial.
- `online_seconds_available`: Availability flag.
- `infrastructure_interrupted`: Boolean flag indicating AC power disconnect.
- `interruption_reason`: Description of infrastructure interruption.
- `guardian_healthy`, `process_group_absent`: Process hygiene verification flags.

### 8. `backend_summary.csv` (40 rows)
- `group_id`, `condition`, `cipher`, `profile`: Grouping identifiers.
- `n_assigned`: Total assigned cells (4).
- `n_completed_bounded_trials`: Number of trials reaching a clean bounded solver outcome (3 or 4).
- `infrastructure_unavailable`: Count of infrastructure rejections (1 for SIMON O128/wrapper, 0 otherwise).
- `verified_recoveries`: Count of verified keys recovered.
- `solver_deadlines`: Count of clean 300 s solver timeouts.
- `median_available_online_seconds`: Median complete online runtime of completed bounded trials, excluding the interrupted trial.
- `scheduled_cell_par2_seconds`: Scheduled-cell PAR-2 score.
"""
    dict_content += """

### Provenance and additional paired fields

Every CSV row includes numbered `source_*_path` and `source_*_sha256` fields.
Cell tables retain a unique `cell_id`; attack tables also include `method_name`
from the paper naming map, while `method` remains the immutable repository ID.
Timing status fields distinguish recorded work, solver not entered, and unavailable
telemetry. `backend_paired_contrasts.csv` is computed from common completed-trial
keys; `valid_pair_count` is three for comparisons involving the interrupted group,
four elsewhere, and `excluded_interrupted_pairs` records that exclusion. Signed
effects are profile A minus profile B; no population confidence intervals or
significance tests are attached to these four-key diagnostics.

The frozen SPECK secondary analysis stores a historical paired interval and rounds
extremely small secondary p-values to zero in its display fields. This versioned
export retains the exact paired counts, recomputes unrounded p-values/Holm within
the frozen family, and uses audited Newcombe Method 10 throughout. For screened
neural minus neural-classical hybrid, its corrected interval is [-0.038714,
0.021310]; the historical analysis remains untouched. This changes no recovery,
timing or significance conclusion.
"""
    with open(dict_path, "w", encoding="utf-8") as f:
        f.write(dict_content)

    # 3. FIGURE_DATA_MAP.md
    map_path = meta_dir / "FIGURE_DATA_MAP.md"
    map_content = """# Figure to Table and Source Data Mapping

| Figure File | Title / Content | Contributing Tables | Primary Source Artifacts |
| :--- | :--- | :--- | :--- |
| `figures/fig1_attack_recovery.pdf` / `.png` | Attack Recovery Rates (Wilson 95% CI) | `tables/attack_summary.csv` | `seven_round_prospective_analysis.json`<br>`simon_confirmation_analysis.json` |
| `figures/fig2_cumulative_recovery_time.pdf` / `.png` | Cumulative Recovery vs Complete Time | `tables/attack_cells.csv` | `seven_round_prospective_results.json`<br>`simon_confirmation_comparators.json` |
| `figures/fig3_operational_cost.pdf` / `.png` | Operational Cost (Recovery vs PAR-2) | `tables/attack_summary.csv` | `seven_round_prospective_analysis.json`<br>`simon_confirmation_analysis.json` |
| `figures/fig4_failure_decomposition.pdf` / `.png` | Attack Failure Mode Breakdown | `tables/attack_failures.csv` | `seven_round_prospective_results.json`<br>`simon_confirmation_comparators.json` |
| `figures/fig5_depth_map.pdf` / `.png` | Cross-Cipher Residual-Depth Scaling Map | `tables/depth_summary.csv`<br>`tables/depth_cells.csv` | `cross_cipher_residual_depth_map.json`<br>`cross_cipher_residual_depth_analysis_reviewed_v2.json` |
| `figures/fig6b_backend_profiles_appendix.pdf` / `.png` | Backend profiles and controls (appendix) | `tables/backend_summary.csv`<br>`tables/backend_cells.csv` | `backend_comparison_20261004.json`<br>`backend_analysis_20261004.json` |
| `figures/fig6_backend_treatments.pdf` / `.png` | Controlled Backend Diagnostic Comparison | `tables/backend_summary.csv`<br>`tables/backend_cells.csv`<br>`tables/backend_paired_contrasts.csv` | `backend_comparison_20261004.json`<br>`backend_analysis_20261004.json` |
"""
    with open(map_path, "w", encoding="utf-8") as f:
        f.write(map_content)

    # Reproduction assets retain the repository layout. Execute in the original
    # checkout with the pinned raw inputs; the zip is not a new attack runtime.
    code_paths = [
        "scripts/generate_final_publication_figures.py",
        "src/neural_sat_speck/analysis/paired_stats.py",
        "src/neural_sat_speck/evaluation/metrics.py",
    ]
    for relative in code_paths:
        target = output_dir / "code" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO_ROOT / relative).read_bytes())
    reproduce = meta_dir / "REPRODUCE.md"
    reproduce.write_text(
        "# Reproduce the figures\n\nThe `code/` folder contains exact generator/statistics sources. Use them in the audited project checkout with the eight frozen inputs listed in `figure_manifest.json`. Their paths and hashes are recorded; public CSVs contain every assigned cell without master keys.\n\n"
        "```bash\nMPLCONFIGDIR=runtime/publication_plot_cache .venv/bin/python scripts/generate_final_publication_figures.py --output-dir output/publication_figures_reproduction\n```\n\n"
        "Python, Matplotlib and NumPy versions are recorded in the manifest. This command reads existing outcomes and renders figures; it performs no attack, training or data draw. CSVs are reconciled against the audited analyses. The formula/evidence archive is a separate delivery; full attack assets are not included in this figure zip.\n",
        encoding="utf-8",
    )

    # 4. figure_manifest.json
    manifest_path = meta_dir / "figure_manifest.json"

    table_hashes = {p.name: sha256_file(p) for p in table_paths}
    fig_hashes = {p.name: sha256_file(p) for p in figure_paths}
    doc_hashes = {
        captions_path.name: sha256_file(captions_path),
        dict_path.name: sha256_file(dict_path),
        map_path.name: sha256_file(map_path),
    }

    manifest_data = {
        "schema": "publication_figures_and_tables_manifest_v1",
        "release_date": "2026-10-05",
        "generator_path": "scripts/generate_final_publication_figures.py",
        "generator_sha256": sha256_file(Path(__file__)),
        "source_paths": {
            k: str(p.relative_to(REPO_ROOT)) for k, p in SOURCE_PATHS.items()
        },
        "statistics_source_hashes": {
            name: sha256_file(REPO_ROOT / name)
            for name in (
                "src/neural_sat_speck/analysis/paired_stats.py",
                "src/neural_sat_speck/evaluation/metrics.py",
            )
        },
        "paper_method_names": METHOD_NAMES,
        "reproduction_command": "python3 scripts/generate_final_publication_figures.py --output-dir <new-output-directory>",
        "render_environment": {
            "python": sys.version.split()[0],
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
        },
        "source_hash_verification": EXPECTED_HASHES,
        "table_artifacts": table_hashes,
        "code_artifacts": {
            f"code/{p}": sha256_file(output_dir / "code" / p) for p in code_paths
        },
        "figure_artifacts": fig_hashes,
        "metadata_artifacts": doc_hashes,
        "styling_specification": {
            "palette": {
                "navy_blue": C_NAVY,
                "burnt_terracotta": C_ORANGE,
                "olive_forest_green": C_GREEN,
                "slate_purple": C_PURPLE,
                "neutral_gray": C_GREY,
                "dark_charcoal": C_DARK,
                "grid_gray": C_GRID,
            },
            "typography": "Times New Roman / STIX math (Type 42 TrueType vector)",
            "dpi": 600,
            "framing": "IEEE single-column / double-column despined Cartesian axes",
        },
        "operating_limits": [
            "SPECK 111/128 method is a neural Stage1 + classical Stage2 hybrid",
            "SIMON classical vs hybrid difference is statistically inconclusive (p=0.34375)",
            "SIMON B1 zero recovery reflects operational ranking timeout under frozen allowance",
            "Nominal cutoff is 300 s; SPECK records cooperative overshoot; SIMON uses parent watchdog",
            "Depth and backend diagnostics are small-n descriptive comparisons on fixed demo keys",
            "The single AC power-stop interruption at cell 80 is preserved transparently as unavailable",
            "No claims of physical carry-chain dynamics or asymptotic universal phase boundaries",
        ],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, sort_keys=True)

    naming = meta_dir / "METHOD_NAMES.md"
    descriptions = {
        "hierarchical_neural_classical": "Neural Stage 1 and classical Stage 2; a hybrid, not a fully classical attack.",
        "hierarchical_neural_screened": "Two neural ranking stages with the frozen screening policy.",
        "hierarchical_neural_full": "Two neural stages without screening; no SAT calls completed within the ranking budget.",
        "fully_classical_8parent": "Classical queried-row 8R Stage 1 and 7R DDT Stage 2; eight parents.",
        "neural_classical_hybrid_8parent": "Bao VV9 Stage 1 and 7R DDT Stage 2; eight parents.",
        "fully_neural_1parent": "Bao VV9 then VV8; one parent. Width differs from eight-parent arms.",
        "faithful_b1_hybrid": "SIMON-trained B1 adaptation then 7R DDT; fixed negative arm. B1 architecture is not novel by itself.",
        "oracle_2key_n4": "True trailing-key tuple supplied as a diagnostic; excluded from autonomous figures, not a recovery upper bound.",
    }
    naming.write_text(
        "# Paper method names\n\nUse these labels consistently in text, figures and tables. Repository IDs remain in CSV provenance.\n\n"
        "| Repository ID | Paper name | Definition |\n|---|---|---|\n"
        + "\n".join(
            f"| `{m}` | {label} | {descriptions.get(m, 'Frozen random queue or unguided control; obs. counts known plaintext/ciphertext observations.')} |"
            for m, label in METHOD_NAMES.items()
        )
        + "\n",
        encoding="utf-8",
    )
    diag = next(
        r
        for r in _read_csv(output_dir / "tables/attack_summary.csv")
        if r["role"] == "diagnostic_control"
    )
    (meta_dir / "DIAGNOSTIC_TABLE.md").write_text(
        "# Appendix diagnostic\n\nThe true-tuple diagnostic uses supplied correct trailing keys and one advisory solver request. It is not an autonomous attack or a guaranteed upper bound.\n\n"
        "| Diagnostic | Verified recovery | Median complete online seconds | PAR-2 seconds |\n|---|---|---|---|\n"
        f"| True-tuple diagnostic | {diag['n_recovered']}/{diag['n_assigned']} | {float(diag['median_wall_seconds']):.2f} | {float(diag['par2_seconds']):.2f} |\n\nTen unresolved outcomes remain in the denominator.\n",
        encoding="utf-8",
    )
    manifest_data["metadata_artifacts"].update(
        {
            p.name: sha256_file(p)
            for p in (naming, meta_dir / "DIAGNOSTIC_TABLE.md", reproduce)
        }
    )
    manifest_path.write_text(
        json.dumps(manifest_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path.with_name(manifest_path.name + ".sha256").write_text(
        f"{sha256_file(manifest_path)}  {manifest_path.name}\n", encoding="utf-8"
    )

    return captions_path, dict_path, map_path, manifest_path


def create_release_zip(output_dir: Path) -> Path:
    """Create a compressed distribution zip containing all figures, tables, and metadata."""
    zip_path = (
        output_dir
        / f"publication_figures_and_tables_{output_dir.name.removeprefix('publication_figures_')}.zip"
    )
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file() and path.suffix != ".zip":
                info = zipfile.ZipInfo(
                    str(path.relative_to(output_dir)), (2026, 10, 5, 0, 0, 0)
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(info, path.read_bytes())

    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate final publication figures and traceable CSVs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output/publication_figures_20261005_final",
        help="Output directory",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing output directory"
    )
    args = parser.parse_args()

    out_dir: Path = args.output_dir
    if out_dir.exists() and not args.force:
        print(
            f"Error: Output directory {out_dir} already exists. Use --force to replace."
        )
        sys.exit(1)

    print("Verifying 8 frozen source file hashes...")
    verify_source_hashes()
    print("All source file hashes verified bit-for-bit.")

    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    meta_dir = out_dir / "metadata"
    for d in [fig_dir, tab_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("Extracting and writing traceable CSV tables...")
    tab_cells, tab_summary, tab_failures, tab_contrasts = build_attack_tables(
        SOURCE_PATHS["speck_raw"], SOURCE_PATHS["simon_raw"], tab_dir
    )
    reconcile_attack_summary(tab_summary)
    reconcile_attack_contrasts(tab_contrasts)
    d_cells, d_summary, d_contrasts = build_depth_tables(
        SOURCE_PATHS["depth_raw"], SOURCE_PATHS["depth_analysis_v2"], tab_dir
    )
    b_cells, b_summary, b_contrasts = build_backend_tables(
        SOURCE_PATHS["backend_raw"], SOURCE_PATHS["backend_analysis"], tab_dir
    )
    table_paths = [
        tab_cells,
        tab_summary,
        tab_failures,
        tab_contrasts,
        d_cells,
        d_summary,
        d_contrasts,
        b_cells,
        b_summary,
        b_contrasts,
    ]
    print(f"Successfully generated {len(table_paths)} CSV tables.")

    print("Rendering publication figures (vector PDF and 600 DPI PNG)...")
    fig_paths: list[Path] = []

    p_pdf, p_png = plot_fig1_attack_recovery(tab_summary, fig_dir)
    fig_paths.extend([p_pdf, p_png])
    print(f"Rendered Figure 1: {p_pdf.name}")

    p_pdf, p_png = plot_fig2_cumulative_recovery_time(tab_cells, fig_dir)
    fig_paths.extend([p_pdf, p_png])
    print(f"Rendered Figure 2: {p_pdf.name}")

    p_pdf, p_png = plot_fig3_operational_cost(tab_summary, fig_dir)
    fig_paths.extend([p_pdf, p_png])
    print(f"Rendered Figure 3: {p_pdf.name}")

    p_pdf, p_png = plot_fig4_failure_decomposition(tab_failures, fig_dir)
    fig_paths.extend([p_pdf, p_png])
    print(f"Rendered Figure 4: {p_pdf.name}")

    p_pdf, p_png = plot_fig5_depth_map(d_summary, fig_dir)
    fig_paths.extend([p_pdf, p_png])
    print(f"Rendered Figure 5: {p_pdf.name}")

    p_pdf, p_png = plot_fig6_backend_treatments(b_summary, b_cells, fig_dir)
    fig_paths.extend([p_pdf, p_png])
    print(f"Rendered Figure 6: {p_pdf.name}")

    p_pdf, p_png = plot_fig6b_backend_profiles(b_summary, b_cells, fig_dir)
    fig_paths.extend([p_pdf, p_png])
    print(f"Rendered Figure 6b (appendix): {p_pdf.name}")

    print("Generating captions, data maps, and cryptographic manifest...")
    c_p, d_p, m_p, mf_p = build_metadata(out_dir, table_paths, fig_paths)

    print("Packaging release zip bundle...")
    zip_p = create_release_zip(out_dir)
    print(f"Release package sealed: {zip_p.name} ({zip_p.stat().st_size} bytes)")

    print(
        f"\nAll publication figures, tables, and metadata generated successfully in {out_dir}."
    )


if __name__ == "__main__":
    main()
