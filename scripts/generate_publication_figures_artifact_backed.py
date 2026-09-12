#!/usr/bin/env python3
"""Generate artifact-backed report and publication figures.

The script intentionally does not read the quarantined figure outputs or copy
paper-facing values from prose. Numerical results are loaded from retained JSON
artifacts and checked against their own summaries before plotting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import lines as mlines
from matplotlib import patches

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "output" / "pdf" / "publication_figures"
IMAGE_DIR = ROOT / "output" / "figures" / "publication"

INPUTS = {
    "phase7_summary": ROOT / "results" / "phase7_final_evaluation" / "summary.json",
    "phase7_records": ROOT / "results" / "phase7_final_evaluation" / "records",
    "classical": ROOT
    / "results"
    / "classical_comparator"
    / "classical_6round_retrospective_evaluation_v3.json",
    "phase10_summary": ROOT
    / "results"
    / "phase10_validation"
    / "prospective_validation_results.json",
    "phase10_records": ROOT
    / "results"
    / "phase10_validation"
    / "validation_atomic_records.json",
    "phase10_g2_summary": ROOT
    / "results"
    / "phase10_g2_prospective"
    / "phase10_g2_prospective_evaluation_report.json",
    "phase10_g2_predictions": ROOT
    / "results"
    / "phase10_g2_prospective"
    / "prospective_predictions.json",
    "screening": ROOT
    / "results"
    / "phase11_scaling_diagnostics"
    / "screening_controlled_audit_v3.json",
    "solver": ROOT
    / "results"
    / "phase11_scaling_diagnostics"
    / "solver_controlled_audit_v3.json",
    "hierarchical": ROOT
    / "results"
    / "phase11_scaling_diagnostics"
    / "hierarchical_eval_16_instances_consolidated.json",
    "phase11a": ROOT / "results" / "phase11a_oracle" / "summary.json",
    "plan": ROOT / "PLAN.md",
    "decisions": ROOT / "docs" / "DECISIONS.md",
}

# Solid series palette sampled from the supplied IEEE reference paper. Large
# tinted fields are avoided: color identifies stages or data on a white field.
BLUE = "#2F5597"
ORANGE = "#C55A11"
GREEN = "#548235"
PURPLE = "#8064A2"
RED = "#A23B3B"
CHARCOAL = "#222222"
GREY = "#666666"
LIGHT_GREY = "#D9D9D9"
PALE_BLUE = "#F2F5F9"
PALE_ORANGE = "#FBF3EE"
PALE_GREEN = "#F3F7F0"
PALE_PURPLE = "#F6F3F8"
PALE_GREY = "#FAFAFA"

EXPECTED_SCHEMAS = {
    "phase7_summary": "neural_sat_speck.phase7_final_evaluation_summary.v1",
    "classical": "neural_sat_speck.classical_comparator_evaluation.v3",
    "screening": "neural_sat_speck.screening_controlled_audit.v3",
    "solver": "neural_sat_speck.solver_controlled_audit.v3",
}

FIGURE_STEMS = (
    "fig01_hierarchical_recovery_stack",
    "fig02_six_round_rank_ecdf",
    "fig03_prospective_reliability_roc",
    "fig04_controlled_engineering_diagnostics",
    "fig05_project_timeline",
    "fig06_seven_round_development_results",
    "fig07_xor_gaussian_solver_ablation",
)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8.2,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.edgecolor": CHARCOAL,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_close(
    actual: float, expected: float, label: str, atol: float = 1e-12
) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=atol):
        raise ValueError(f"{label}: recomputed {actual} != artifact summary {expected}")


def empirical_roc(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return an empirical ROC curve with deterministic handling of tied scores."""
    require(y_true.ndim == 1 and scores.ndim == 1, "ROC inputs must be one-dimensional")
    require(len(y_true) == len(scores), "ROC inputs must have equal lengths")
    require(set(np.unique(y_true)).issubset({0, 1}), "ROC labels must be binary")
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    require(positives > 0 and negatives > 0, "ROC requires both outcome classes")
    require(bool(np.all(np.isfinite(scores))), "ROC scores must be finite")

    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    ordered_labels = y_true[order]
    distinct = np.where(np.diff(ordered_scores))[0]
    threshold_indices = np.r_[distinct, len(ordered_scores) - 1]
    true_positives = np.cumsum(ordered_labels)[threshold_indices]
    false_positives = 1 + threshold_indices - true_positives
    tpr = np.r_[0.0, true_positives / positives]
    fpr = np.r_[0.0, false_positives / negatives]
    return fpr.astype(float), tpr.astype(float)


def empirical_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr = empirical_roc(y_true, scores)
    return float(np.trapezoid(tpr, fpr))


def load_and_verify() -> dict[str, Any]:
    phase7 = load_json(INPUTS["phase7_summary"])
    classical = load_json(INPUTS["classical"])
    phase10 = load_json(INPUTS["phase10_summary"])
    phase10_records = load_json(INPUTS["phase10_records"])
    g2_summary = load_json(INPUTS["phase10_g2_summary"])
    g2_predictions = load_json(INPUTS["phase10_g2_predictions"])
    screening = load_json(INPUTS["screening"])
    solver = load_json(INPUTS["solver"])
    hierarchical = load_json(INPUTS["hierarchical"])
    phase11a = load_json(INPUTS["phase11a"])

    for name, expected in EXPECTED_SCHEMAS.items():
        artifact = {
            "phase7_summary": phase7,
            "classical": classical,
            "screening": screening,
            "solver": solver,
        }[name]
        require(artifact.get("schema") == expected, f"Unexpected schema for {name}")

    phase7_record_paths = sorted(INPUTS["phase7_records"].glob("instance_*.json"))
    require(len(phase7_record_paths) == 64, "Phase 7 must contain exactly 64 records")
    phase7_records = [load_json(path) for path in phase7_record_paths]
    require(phase7.get("instance_count") == 64, "Phase 7 summary denominator changed")
    require(len(phase10_records) == 64, "Phase 10 must contain exactly 64 records")
    require(
        len(g2_predictions.get("y_fail", [])) == 256, "Phase 10-G2 denominator changed"
    )

    classical_records = classical.get("records", [])
    require(len(classical_records) == 64, "Classical evaluation denominator changed")
    require(
        classical.get("method") == "sparse_empirical_monte_carlo_llr_surrogate",
        "Classical method identity changed",
    )

    p7_ids = [record["instance_id"] for record in phase7_records]
    classical_ids = [record["instance_id"] for record in classical_records]
    require(p7_ids == classical_ids, "Phase 7 and classical instance ordering differs")
    for p7_record, classical_record in zip(
        phase7_records, classical_records, strict=True
    ):
        require(
            p7_record["true_final_round_subkey"]
            == classical_record["true_final_round_subkey"],
            f"True-subkey mismatch for {p7_record['instance_id']}",
        )

    b0_ranks = np.asarray(
        [record["methods"]["b0_sat"]["exact_true_rank"] for record in phase7_records],
        dtype=int,
    )
    b1_ranks = np.asarray(
        [record["methods"]["b1_sat"]["exact_true_rank"] for record in phase7_records],
        dtype=int,
    )
    classical_ranks = np.asarray(
        [record["classical_rank"] for record in classical_records], dtype=int
    )
    require(int(np.sum(b0_ranks <= 16)) == 50, "B0 top-16 count changed")
    require(int(np.sum(b1_ranks <= 16)) == 58, "B1 top-16 count changed")
    require(int(np.sum(classical_ranks <= 4)) == 64, "Classical top-4 count changed")
    require(int(np.max(b0_ranks)) <= 32, "Unexpected B0 rank above plotted domain")
    require(int(np.max(b1_ranks)) <= 32, "Unexpected B1 rank above plotted domain")

    y10 = np.asarray([record["y_fail"] for record in phase10_records], dtype=int)
    g0_10 = np.asarray([record["p_g0_fail"] for record in phase10_records], dtype=float)
    g1_10 = np.asarray([record["p_g1_fail"] for record in phase10_records], dtype=float)
    require(int(y10.sum()) == 6, "Phase 10 positive-event count changed")
    auc_g0_10 = empirical_roc_auc(y10, g0_10)
    auc_g1_10 = empirical_roc_auc(y10, g1_10)
    require_close(
        auc_g0_10,
        phase10["reliability_inference"]["g0_scalar"]["roc_auc"],
        "Phase 10 G0 ROC-AUC",
    )
    require_close(
        auc_g1_10,
        phase10["reliability_inference"]["g1_stability"]["roc_auc"],
        "Phase 10 G1 ROC-AUC",
    )

    y_g2 = np.asarray(g2_predictions["y_fail"], dtype=int)
    g1_g2 = np.asarray(g2_predictions["g1_probs"], dtype=float)
    g2_g2 = np.asarray(g2_predictions["g2_probs"], dtype=float)
    require(int(y_g2.sum()) == 21, "Phase 10-G2 positive-event count changed")
    auc_g1_g2 = empirical_roc_auc(y_g2, g1_g2)
    auc_g2_g2 = empirical_roc_auc(y_g2, g2_g2)
    require_close(
        auc_g1_g2,
        g2_summary["statistical_results"]["g1_v1"]["roc_auc"],
        "Phase 10-G2 G1 ROC-AUC",
    )
    require_close(
        auc_g2_g2,
        g2_summary["statistical_results"]["g2_v1"]["roc_auc"],
        "Phase 10-G2 G2 ROC-AUC",
    )

    screen_records = screening["records"]["true_prefix_evaluations"]
    require(len(screen_records) == 4, "Screening true-prefix denominator changed")
    require(
        all(record["rank_matched"] for record in screen_records), "Rank mismatch found"
    )
    require(
        all(record["ordered_agreement_top32"] == 32 for record in screen_records),
        "Top-32 ordered agreement changed",
    )
    observed_speedups = np.asarray(
        [record["matched_speedup"] for record in screen_records], dtype=float
    )
    require_close(
        float(np.mean(observed_speedups)),
        screening["true_prefix_summary"]["mean_matched_speedup"],
        "Screening mean paired speedup",
    )

    runs = solver["runs"]
    expected_solve_counts = {
        "1_pycryptosat_default_policy": 0,
        "2_native_default_matrix_limit": 0,
        "3_native_raised_matrix_limit_autodisable_off": 2,
    }
    require(set(runs) == set(expected_solve_counts), "Solver treatment set changed")
    for treatment, expected_count in expected_solve_counts.items():
        require(
            len(runs[treatment]) == 3, f"Solver denominator changed for {treatment}"
        )
        observed_count = sum(bool(record["verified"]) for record in runs[treatment])
        require(
            observed_count == expected_count,
            f"Solver verified count changed for {treatment}",
        )

    # This field is deliberately checked only to ensure the architecture is tied
    # to the retained development run. It is not presented as prospective evidence.
    require(
        hierarchical.get("evidence_class")
        in {"DEVELOPMENT", "DEVELOPMENT / DIAGNOSTIC", None},
        "Hierarchical artifact is no longer development evidence",
    )
    hierarchical_records = hierarchical.get("records", [])
    require(len(hierarchical_records) == 16, "Hierarchical denominator changed")
    hierarchical_successes = sum(
        bool(record["success"]) for record in hierarchical_records
    )
    require(hierarchical_successes == 13, "Hierarchical recovery count changed")
    hierarchical_ranks = np.asarray(
        [record["stages"][0]["true_round_key_rank"] for record in hierarchical_records],
        dtype=int,
    )
    require(
        int(np.sum(hierarchical_ranks <= 32)) == 13,
        "Stage-1 top-32 containment changed",
    )
    ranking_failures = [
        record
        for record in hierarchical_records
        if not record["success"] and record["failure_cause"] == "ranking"
    ]
    require(len(ranking_failures) == 3, "Hierarchical failure classification changed")
    require(
        all(record["status"] == "exhausted_candidates" for record in ranking_failures),
        "Unexpected hierarchical failure status",
    )
    hierarchical_wall = np.asarray(
        [record["wall_seconds"] for record in hierarchical_records], dtype=float
    )
    require_close(
        float(np.median(hierarchical_wall)),
        hierarchical["median_wall_seconds"],
        "Hierarchical median wall time",
        atol=0.01,
    )

    require(
        phase11a.get("protocol") == "phase11a_oracle_sat_feasibility_gate",
        "Phase 11A protocol changed",
    )
    require(phase11a.get("instance_count") == 64, "Phase 11A denominator changed")
    require(
        phase11a["recovery"]["verified_recovery_count"] == 0,
        "Phase 11A recovery count changed",
    )
    require(
        phase11a["status_breakdown"]["timeout"] == 64, "Phase 11A timeout count changed"
    )
    require_close(float(phase11a["timeout_seconds"]), 10.0, "Phase 11A timeout")
    require(
        phase11a["go_no_go_evaluation"]["final_decision"] == "PHASE11B BLOCKED",
        "Phase 11A gate decision changed",
    )

    return {
        "phase7": phase7,
        "phase7_records": phase7_records,
        "classical": classical,
        "classical_records": classical_records,
        "phase10": phase10,
        "phase10_records": phase10_records,
        "g2_summary": g2_summary,
        "g2_predictions": g2_predictions,
        "screening": screening,
        "solver": solver,
        "hierarchical": hierarchical,
        "phase11a": phase11a,
        "derived": {
            "b0_ranks": b0_ranks,
            "b1_ranks": b1_ranks,
            "classical_ranks": classical_ranks,
            "phase10_y": y10,
            "phase10_g0": g0_10,
            "phase10_g1": g1_10,
            "phase10_g0_auc": auc_g0_10,
            "phase10_g1_auc": auc_g1_10,
            "g2_y": y_g2,
            "g2_g1": g1_g2,
            "g2_g2": g2_g2,
            "g2_g1_auc": auc_g1_g2,
            "g2_g2_auc": auc_g2_g2,
            "hierarchical_stage1_ranks": hierarchical_ranks,
            "hierarchical_wall_seconds": hierarchical_wall,
            "hierarchical_successes": hierarchical_successes,
        },
    }


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        PDF_DIR / f"{stem}.pdf",
        IMAGE_DIR / f"{stem}.svg",
        IMAGE_DIR / f"{stem}.png",
    ]
    metadata = {"Creator": "artifact-backed publication figure generator"}
    fig.savefig(outputs[0], bbox_inches="tight", pad_inches=0.06, metadata=metadata)
    fig.savefig(outputs[1], bbox_inches="tight", pad_inches=0.06, metadata=metadata)
    svg_text = outputs[1].read_text(encoding="utf-8")
    outputs[1].write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(
        outputs[2], dpi=300, bbox_inches="tight", pad_inches=0.06, metadata=metadata
    )
    plt.close(fig)
    return outputs


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
    )


def diagram_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    body: str,
    edge: str,
    face: str,
    title_size: float = 7.5,
    body_size: float = 6.4,
) -> None:
    x, y = xy
    box = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.006,rounding_size=0.008",
        linewidth=0.9,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height * 0.72,
        title,
        ha="center",
        va="center",
        color=edge,
        fontweight="bold",
        fontsize=title_size,
    )
    ax.text(
        x + width / 2,
        y + height * 0.35,
        body,
        ha="center",
        va="center",
        color=CHARCOAL,
        fontsize=body_size,
        linespacing=1.2,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str | None = None,
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": CHARCOAL},
    )
    if label:
        ax.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + 0.018,
            label,
            ha="center",
            va="bottom",
            fontsize=6.9,
            color=GREY,
        )


def _make_hierarchical_stack_single_spine(_: dict[str, Any]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.7, 8.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.985,
        "Hierarchical seven-round recovery stack",
        ha="center",
        va="top",
        fontsize=10.4,
        fontweight="bold",
        color=CHARCOAL,
    )
    ax.text(
        0.5,
        0.955,
        r"Static development configuration: $M_1=32$, $M_2=32$",
        ha="center",
        va="top",
        fontsize=7.6,
        color=ORANGE,
    )

    # Side rails identify the scientific role of the corresponding stack blocks.
    ranking_rail = patches.FancyBboxPatch(
        (0.035, 0.35),
        0.105,
        0.49,
        boxstyle="round,pad=0.005,rounding_size=0.008",
        linewidth=0.8,
        edgecolor=BLUE,
        facecolor=PALE_BLUE,
    )
    exact_rail = patches.FancyBboxPatch(
        (0.86, 0.035),
        0.105,
        0.285,
        boxstyle="round,pad=0.005,rounding_size=0.008",
        linewidth=0.8,
        edgecolor=PURPLE,
        facecolor=PALE_PURPLE,
    )
    ax.add_patch(ranking_rail)
    ax.add_patch(exact_rail)
    ax.text(
        0.0875,
        0.595,
        "STATISTICAL GUIDANCE  -  RANKING ONLY",
        rotation=90,
        ha="center",
        va="center",
        color=BLUE,
        fontweight="bold",
        fontsize=7.0,
    )
    ax.text(
        0.9125,
        0.1775,
        "EXACT RECOVERY  /  VERIFY",
        rotation=90,
        ha="center",
        va="center",
        color=PURPLE,
        fontweight="bold",
        fontsize=6.2,
    )

    stack_x = 0.205
    stack_w = 0.59
    boxes = [
        (
            0.865,
            0.06,
            "Chosen-pair evidence",
            r"64 pairs; $\Delta P=(0x0040,0)$",
            CHARCOAL,
            PALE_GREY,
            7.7,
            6.6,
        ),
        (
            0.765,
            0.073,
            r"Stage 1: complete-domain rank of $k_6$",
            "B1 scores all 65,536 values; deterministic order",
            BLUE,
            "#E8EEF7",
            7.5,
            6.4,
        ),
        (
            0.685,
            0.052,
            r"Prefix beam",
            r"retain top $M_1=32$ values of $k_6$",
            BLUE,
            "#E8EEF7",
            7.3,
            6.3,
        ),
        (
            0.595,
            0.064,
            "Conditional partial decryption",
            r"undo round 6 under each retained $k_6$",
            GREEN,
            "#EAF2E5",
            7.4,
            6.4,
        ),
        (
            0.485,
            0.078,
            r"Stage 2a: screen the $k_5$ domain",
            "16 pairs score all 65,536 values; retain 512",
            GREEN,
            "#EAF2E5",
            7.4,
            6.4,
        ),
        (
            0.375,
            0.078,
            r"Stage 2b: full rescore",
            r"64 pairs rescore 512 survivors; retain top $M_2=32$",
            GREEN,
            "#EAF2E5",
            7.4,
            6.4,
        ),
        (
            0.295,
            0.052,
            "Deterministic tuple queue",
            r"ordered unique $(k_6,k_5)$; at most $32\times32$ tuples",
            PURPLE,
            "#EEE8F2",
            7.3,
            6.2,
        ),
        (
            0.190,
            0.075,
            "Incremental CryptoMiniSat",
            "isolated model per attack/method; tuple assumptions; 10 s per tuple",
            PURPLE,
            "#EEE8F2",
            7.5,
            6.2,
        ),
        (
            0.105,
            0.055,
            "Independent fresh-pair verification",
            "2 fresh pairs; full-key re-encryption",
            GREEN,
            "#EAF2E5",
            7.3,
            6.3,
        ),
        (
            0.030,
            0.047,
            "Outcome record",
            "verified master key or explicit failure class",
            CHARCOAL,
            PALE_GREY,
            7.2,
            6.2,
        ),
    ]

    for y, height, title, body, edge, face, title_size, body_size in boxes:
        diagram_box(
            ax,
            (stack_x, y),
            stack_w,
            height,
            title,
            body,
            edge,
            face,
            title_size=title_size,
            body_size=body_size,
        )

    # A single vertical spine keeps the data flow unambiguous.
    for current, following in zip(boxes[:-1], boxes[1:], strict=True):
        current_y = current[0]
        following_y, following_height = following[0], following[1]
        arrow(
            ax,
            (0.5, current_y - 0.006),
            (0.5, following_y + following_height + 0.006),
        )

    ax.text(
        0.81,
        0.720,
        "complete 16-bit domain",
        rotation=90,
        ha="center",
        va="center",
        color=GREY,
        fontsize=6.2,
    )
    ax.text(
        0.81,
        0.455,
        "per retained prefix",
        rotation=90,
        ha="center",
        va="center",
        color=GREY,
        fontsize=6.2,
    )
    ax.text(
        0.18,
        0.225,
        "pycryptosat path used\nin development",
        rotation=90,
        ha="center",
        va="center",
        color=GREY,
        fontsize=6.1,
    )
    return fig


def _make_hierarchical_stack_swimlanes(_: dict[str, Any]) -> plt.Figure:
    """Draw the method as three compact top-to-bottom scientific stacks."""
    fig, ax = plt.subplots(figsize=(7.16, 4.85))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.985,
        "Hierarchical seven-round recovery architecture",
        ha="center",
        va="top",
        fontsize=10.2,
        fontweight="bold",
        color=CHARCOAL,
    )
    ax.text(
        0.5,
        0.94,
        r"Static development configuration: $M_1=32$, $M_2=32$; not a frozen prospective protocol",
        ha="center",
        va="top",
        fontsize=7.1,
        color=RED,
    )

    column_x = [0.025, 0.35, 0.675]
    column_w = 0.30
    column_y = 0.095
    column_h = 0.775
    column_colors = [BLUE, GREEN, PURPLE]
    column_faces = [PALE_BLUE, PALE_GREEN, PALE_PURPLE]
    column_titles = [
        r"1. Prefix ranking ($k_6$)",
        r"2. Conditional ranking ($k_5$)",
        "3. Exact recovery",
    ]

    for x, edge, face, title in zip(
        column_x, column_colors, column_faces, column_titles, strict=True
    ):
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, column_y),
                column_w,
                column_h,
                boxstyle="round,pad=0.006,rounding_size=0.009",
                linewidth=0.9,
                edgecolor=edge,
                facecolor="white",
            )
        )
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, 0.79),
                column_w,
                0.08,
                boxstyle="round,pad=0.006,rounding_size=0.009",
                linewidth=0.9,
                edgecolor=edge,
                facecolor=face,
            )
        )
        ax.text(
            x + column_w / 2,
            0.83,
            title,
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color=edge,
        )

    # The stage headers communicate the global left-to-right sequence. Each
    # stage itself reads strictly from top to bottom.
    arrow(ax, (0.328, 0.83), (0.344, 0.83))
    arrow(ax, (0.653, 0.83), (0.669, 0.83))

    def stack_box(
        x: float,
        y: float,
        height: float,
        title: str,
        body: str,
        edge: str,
        face: str,
        title_size: float = 7.0,
        body_size: float = 6.1,
    ) -> None:
        width = 0.255
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.005,rounding_size=0.006",
                linewidth=0.75,
                edgecolor=edge,
                facecolor=face,
            )
        )
        ax.text(
            x + width / 2,
            y + height * 0.67,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=edge,
        )
        ax.text(
            x + width / 2,
            y + height * 0.30,
            body,
            ha="center",
            va="center",
            fontsize=body_size,
            color=CHARCOAL,
            linespacing=1.15,
        )

    def down_arrow(x_center: float, y_start: float, y_end: float) -> None:
        arrow(ax, (x_center, y_start), (x_center, y_end))

    # Stack 1: one complete-domain ordering of k6.
    x1 = column_x[0] + 0.0225
    c1 = column_x[0] + column_w / 2
    stack_box(
        x1,
        0.635,
        0.105,
        "Chosen-pair input",
        r"64 ciphertext pairs" "\n" r"$\Delta P=(0x0040,0)$",
        CHARCOAL,
        PALE_GREY,
    )
    stack_box(
        x1,
        0.435,
        0.135,
        r"B1 ranks $k_6$",
        "score all 65,536 values\ncomplete deterministic order",
        BLUE,
        "#E8EEF7",
    )
    stack_box(
        x1,
        0.260,
        0.105,
        "Prefix beam",
        r"retain top $M_1=32$" "\n" r"candidate prefixes",
        BLUE,
        "#E8EEF7",
    )
    down_arrow(c1, 0.63, 0.575)
    down_arrow(c1, 0.43, 0.37)
    ax.text(
        c1,
        0.155,
        "Output: ranked prefixes\nRanking is guidance, not proof.",
        ha="center",
        va="center",
        fontsize=6.1,
        color=GREY,
        linespacing=1.25,
    )

    # Stack 2: repeat the conditional screen/rescore path for each retained k6.
    x2 = column_x[1] + 0.0225
    c2 = column_x[1] + column_w / 2
    stack_box(
        x2,
        0.665,
        0.075,
        r"For each retained $k_6$",
        "conditional branch",
        GREEN,
        "#EAF2E5",
        title_size=6.8,
        body_size=5.9,
    )
    stack_box(
        x2,
        0.535,
        0.085,
        "Partial decrypt",
        r"undo round 6 under $k_6$",
        GREEN,
        "#EAF2E5",
    )
    stack_box(
        x2,
        0.365,
        0.12,
        r"16-pair screen of $k_5$",
        "65,536 values -> 512 survivors",
        GREEN,
        "#EAF2E5",
    )
    stack_box(
        x2,
        0.195,
        0.12,
        "64-pair full rescore",
        r"512 survivors -> top $M_2=32$",
        GREEN,
        "#EAF2E5",
    )
    down_arrow(c2, 0.66, 0.625)
    down_arrow(c2, 0.53, 0.49)
    down_arrow(c2, 0.36, 0.32)
    ax.text(
        c2,
        0.135,
        r"Output: top $k_5$ values per prefix",
        ha="center",
        va="center",
        fontsize=6.1,
        color=GREY,
    )

    # Stack 3: compose tuples, solve exactly, then independently verify.
    x3 = column_x[2] + 0.0225
    c3 = column_x[2] + column_w / 2
    stack_box(
        x3,
        0.635,
        0.105,
        "Tuple queue",
        r"unique ordered $(k_6,k_5)$" "\n" r"at most $32\times32$ tuples",
        PURPLE,
        "#EEE8F2",
    )
    stack_box(
        x3,
        0.445,
        0.125,
        "Incremental CryptoMiniSat",
        "isolated model per attack/method\n10 s per tuple assumption",
        PURPLE,
        "#EEE8F2",
        title_size=6.7,
    )
    stack_box(
        x3,
        0.285,
        0.105,
        "Fresh verification",
        "2 independent pairs\nfull-key re-encryption",
        GREEN,
        "#EAF2E5",
    )
    stack_box(
        x3,
        0.130,
        0.10,
        "Outcome record",
        "verified master key or\nexplicit failure class",
        CHARCOAL,
        PALE_GREY,
    )
    down_arrow(c3, 0.63, 0.575)
    down_arrow(c3, 0.44, 0.395)
    down_arrow(c3, 0.28, 0.235)

    ax.text(
        0.5,
        0.035,
        "Measured development path uses pycryptosat. G2 routing and native-Gauss diagnostics are separate experiments.",
        ha="center",
        va="center",
        fontsize=6.4,
        color=GREY,
    )
    return fig


def make_hierarchical_stack(_: dict[str, Any]) -> plt.Figure:
    """Draw a spacious methodology-only vertical architecture."""
    fig, ax = plt.subplots(figsize=(7.16, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    canvas_x = 0.10
    canvas_w = 0.80
    center_x = 0.50
    border = "#505050"
    paper_blue_fill = "#EAF2F8"
    paper_grey_fill = "#F2F2F2"
    paper_orange_fill = "#FCE9DD"

    ax.text(
        center_x,
        0.982,
        "Hierarchical neural-SAT architecture",
        ha="center",
        va="top",
        fontsize=9.3,
        fontweight="normal",
        color=CHARCOAL,
    )
    ax.text(
        center_x,
        0.950,
        "Seven-round SPECK32/64 key recovery",
        ha="center",
        va="top",
        fontsize=6.8,
        color=GREY,
    )

    def shell(
        y: float,
        height: float,
        accent: str | None = None,
        facecolor: str = "white",
    ) -> None:
        ax.add_patch(
            patches.Rectangle(
                (canvas_x, y),
                canvas_w,
                height,
                linewidth=0.75,
                edgecolor=border,
                facecolor=facecolor,
                zorder=2,
            )
        )
        if accent is not None:
            ax.add_patch(
                patches.Rectangle(
                    (canvas_x, y + height - 0.008),
                    canvas_w,
                    0.008,
                    linewidth=0,
                    facecolor=accent,
                    zorder=3,
                )
            )

    def flow_arrow(y_start: float, y_end: float) -> None:
        ax.add_patch(
            patches.FancyArrowPatch(
                (center_x, y_start),
                (center_x, y_end),
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=0.8,
                color=CHARCOAL,
                shrinkA=0,
                shrinkB=0,
                zorder=5,
            )
        )

    def stage_tag(x: float, y: float, label: str, color: str) -> None:
        ax.text(
            x,
            y,
            label,
            ha="left",
            va="center",
            fontsize=5.7,
            fontweight="normal",
            color=color,
            zorder=4,
        )

    def dimension_badge(x: float, y: float, width: float, text_value: str) -> None:
        ax.add_patch(
            patches.Rectangle(
                (x, y - 0.016),
                width,
                0.032,
                linewidth=0.55,
                edgecolor=LIGHT_GREY,
                facecolor=PALE_GREY,
                zorder=3,
            )
        )
        ax.text(
            x + width / 2,
            y,
            text_value,
            ha="center",
            va="center",
            fontsize=5.8,
            color=CHARCOAL,
            zorder=4,
        )

    # Input evidence
    input_y, input_h = 0.835, 0.075
    shell(input_y, input_h, facecolor=paper_grey_fill)
    stage_tag(0.125, input_y + input_h / 2, "INPUT", GREY)
    ax.text(
        0.34,
        input_y + input_h * 0.62,
        "Ciphertext-pair evidence",
        ha="left",
        va="center",
        fontsize=7.0,
        fontweight="normal",
        color=CHARCOAL,
    )
    ax.text(
        0.34,
        input_y + input_h * 0.30,
        r"64 chosen pairs with $\Delta P=(0x0040,0)$",
        ha="left",
        va="center",
        fontsize=5.8,
        color=GREY,
    )
    flow_arrow(input_y - 0.006, 0.806)

    # Complete-domain last-round ranking
    stage1_y, stage1_h = 0.690, 0.110
    shell(stage1_y, stage1_h, BLUE, paper_blue_fill)
    stage_tag(0.125, stage1_y + 0.071, "NEURAL RANK I", BLUE)
    ax.text(
        0.31,
        stage1_y + 0.071,
        r"Complete-domain ranking of $k_6$",
        ha="left",
        va="center",
        fontsize=7.0,
        fontweight="normal",
        color=CHARCOAL,
    )
    ax.text(
        0.31,
        stage1_y + 0.033,
        "Pair-aware B1 scoring",
        ha="left",
        va="center",
        fontsize=5.8,
        color=GREY,
    )
    dimension_badge(0.715, stage1_y + 0.050, 0.155, r"$2^{16}\ \rightarrow\ M_1=32$")
    flow_arrow(stage1_y - 0.006, 0.656)

    # Conditional penultimate-round ranking
    stage2_y, stage2_h = 0.390, 0.260
    shell(stage2_y, stage2_h, BLUE, paper_blue_fill)
    stage_tag(0.125, stage2_y + 0.215, "NEURAL RANK II", BLUE)
    ax.text(
        0.31,
        stage2_y + 0.215,
        r"Conditional ranking of $k_5$",
        ha="left",
        va="center",
        fontsize=7.0,
        fontweight="normal",
        color=CHARCOAL,
    )
    ax.text(
        0.31,
        stage2_y + 0.180,
        r"One ranking problem for each retained $k_6$",
        ha="left",
        va="center",
        fontsize=5.8,
        color=GREY,
    )

    inner_y, inner_h, inner_w = stage2_y + 0.065, 0.088, 0.205
    inner_xs = [0.145, 0.3975, 0.65]
    inner_content = [
        ("Partial decrypt", r"undo round 6 under $k_6$"),
        ("Low-fidelity screen", r"16 pairs: $2^{16}\rightarrow512$"),
        ("Full rescore", r"64 pairs: $512\rightarrow M_2=32$"),
    ]
    for x, (title, body) in zip(inner_xs, inner_content, strict=True):
        ax.add_patch(
            patches.Rectangle(
                (x, inner_y),
                inner_w,
                inner_h,
                linewidth=0.65,
                edgecolor=BLUE,
                facecolor="white",
                zorder=3,
            )
        )
        ax.text(
            x + inner_w / 2,
            inner_y + inner_h * 0.63,
            title,
            ha="center",
            va="center",
            fontsize=6.0,
            fontweight="normal",
            color=CHARCOAL,
            zorder=4,
        )
        ax.text(
            x + inner_w / 2,
            inner_y + inner_h * 0.31,
            body,
            ha="center",
            va="center",
            fontsize=5.1,
            color=GREY,
            zorder=4,
        )
    for start_x, end_x in [(0.350, 0.395), (0.6025, 0.6475)]:
        ax.add_patch(
            patches.FancyArrowPatch(
                (start_x, inner_y + inner_h / 2),
                (end_x, inner_y + inner_h / 2),
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=0.75,
                color=CHARCOAL,
                shrinkA=0,
                shrinkB=0,
                zorder=5,
            )
        )
    ax.text(
        center_x,
        stage2_y + 0.031,
        r"Per-prefix output: top $M_2=32$ candidates for $k_5$",
        ha="center",
        va="center",
        fontsize=5.7,
        color=GREY,
        zorder=4,
    )
    flow_arrow(stage2_y - 0.006, 0.356)

    # Candidate scheduling
    queue_y, queue_h = 0.295, 0.055
    shell(queue_y, queue_h, facecolor=paper_grey_fill)
    stage_tag(0.125, queue_y + queue_h / 2, "QUEUE", GREY)
    ax.text(
        0.31,
        queue_y + queue_h / 2,
        r"Stable, deduplicated $(k_6,k_5)$ tuple order",
        ha="left",
        va="center",
        fontsize=6.2,
        fontweight="normal",
        color=CHARCOAL,
    )
    dimension_badge(0.735, queue_y + queue_h * 0.44, 0.135, r"$\leq1{,}024$")
    flow_arrow(queue_y - 0.006, 0.266)

    # Exact completion
    sat_y, sat_h = 0.190, 0.070
    shell(sat_y, sat_h, ORANGE, paper_orange_fill)
    stage_tag(0.125, sat_y + sat_h * 0.66, "EXACT SEARCH", ORANGE)
    ax.text(
        0.31,
        sat_y + sat_h * 0.66,
        "Incremental XOR-SAT completion",
        ha="left",
        va="center",
        fontsize=6.8,
        fontweight="normal",
        color=CHARCOAL,
    )
    ax.text(
        0.31,
        sat_y + sat_h * 0.30,
        "Isolated CryptoMiniSat model; tuple assumptions complete the key schedule",
        ha="left",
        va="center",
        fontsize=5.4,
        color=GREY,
    )
    flow_arrow(sat_y - 0.006, 0.156)

    # Independent verification
    verify_y, verify_h = 0.100, 0.050
    shell(verify_y, verify_h, ORANGE, paper_orange_fill)
    stage_tag(0.125, verify_y + verify_h / 2, "VERIFY", ORANGE)
    ax.text(
        0.31,
        verify_y + verify_h / 2,
        "Independent full-key re-encryption on fresh pairs",
        ha="left",
        va="center",
        fontsize=6.1,
        fontweight="normal",
        color=CHARCOAL,
    )
    dimension_badge(0.785, verify_y + verify_h / 2, 0.085, r"$V=2$")
    flow_arrow(verify_y - 0.006, 0.066)

    # Typed output
    output_y, output_h = 0.020, 0.040
    shell(output_y, output_h)
    stage_tag(0.125, output_y + output_h / 2, "OUTPUT", GREY)
    ax.text(
        0.31,
        output_y + output_h / 2,
        "Verified 64-bit master key or explicit failure state",
        ha="left",
        va="center",
        fontsize=6.0,
        fontweight="normal",
        color=CHARCOAL,
    )

    for text_item in fig.findobj(match=matplotlib.text.Text):
        text_item.set_fontfamily("DejaVu Sans")
    return fig


def ecdf_at(ranks: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.asarray([np.mean(ranks <= rank) for rank in x], dtype=float)


def make_rank_ecdf(data: dict[str, Any]) -> plt.Figure:
    ranks = data["derived"]
    x = np.arange(1, 33)
    fig, ax = plt.subplots(figsize=(7.16, 3.75))
    curves = [
        (
            ranks["classical_ranks"],
            "Sparse empirical differential surrogate",
            GREEN,
            "-",
            "o",
        ),
        (ranks["b1_ranks"], "B1 pair-aware neural", BLUE, "-", "s"),
        (ranks["b0_ranks"], "B0 independent-pair neural", ORANGE, "--", "^"),
    ]
    for values, label, color, linestyle, marker in curves:
        y = ecdf_at(values, x)
        ax.step(
            x, y, where="post", color=color, lw=1.7, linestyle=linestyle, label=label
        )
        marker_x = np.asarray([1, 2, 4, 8, 16, 32])
        ax.plot(
            marker_x,
            ecdf_at(values, marker_x),
            linestyle="none",
            marker=marker,
            markersize=3.8,
            color=color,
            markerfacecolor="white",
            markeredgewidth=0.8,
        )

    ax.axvline(16, color=GREY, lw=0.8, linestyle=":")
    ax.axvline(32, color=GREY, lw=0.8, linestyle=":")
    ax.set_xscale("log", base=2)
    ax.set_xlim(1, 32)
    ax.set_ylim(0, 1.03)
    ax.set_xticks([1, 2, 4, 8, 16, 32], labels=["1", "2", "4", "8", "16", "32"])
    ax.set_yticks(np.linspace(0, 1, 6), labels=["0", "20", "40", "60", "80", "100"])
    ax.set_xlabel("Candidate budget / true-subkey rank threshold")
    ax.set_ylabel("Instances contained (%)")
    ax.set_title("Six-round complete-domain rank containment")
    ax.grid(axis="y", color=LIGHT_GREY, lw=0.7, linestyle=":")
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor=LIGHT_GREY)
    ax.text(
        0.015,
        0.97,
        "At rank 16: classical 64/64, B1 58/64, B0 50/64",
        transform=ax.transAxes,
        va="top",
        fontsize=7.5,
        bbox={
            "boxstyle": "square,pad=0.3",
            "facecolor": "white",
            "edgecolor": LIGHT_GREY,
        },
    )
    fig.text(
        0.5,
        0.012,
        "RETROSPECTIVE - opened Seed 98001 - N=64 - ranking containment, not end-to-end recovery",
        ha="center",
        fontsize=7,
        color=RED,
    )
    fig.subplots_adjust(left=0.095, right=0.985, top=0.90, bottom=0.19)
    return fig


def plot_roc(
    ax: plt.Axes,
    y_true: np.ndarray,
    series: list[tuple[np.ndarray, str, str, str]],
) -> None:
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.9, linestyle=":", label="Random ordering")
    for scores, label, color, linestyle in series:
        fpr, tpr = empirical_roc(y_true, scores)
        auc = empirical_roc_auc(y_true, scores)
        ax.step(
            fpr,
            tpr,
            where="post",
            color=color,
            lw=1.7,
            linestyle=linestyle,
            label=f"{label} (AUC={auc:.3f})",
        )
    ax.set_xlim(-0.015, 1.015)
    ax.set_ylim(-0.015, 1.015)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.grid(color=LIGHT_GREY, lw=0.7, linestyle=":")
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor=LIGHT_GREY)


def make_reliability_roc(data: dict[str, Any]) -> plt.Figure:
    derived = data["derived"]
    summary10 = data["phase10"]["reliability_inference"]
    summary_g2 = data["g2_summary"]["statistical_results"]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.45))

    plot_roc(
        axes[0],
        derived["phase10_y"],
        [
            (derived["phase10_g1"], "G1 stability", BLUE, "-"),
            (derived["phase10_g0"], "G0 scalar", ORANGE, "--"),
        ],
    )
    axes[0].set_title(
        "(a) Phase 10 prospective (N=64; 6 failures)", fontsize=8.5, pad=5
    )
    axes[0].text(
        0.03,
        0.96,
        "G1 95% CI: "
        f"[{summary10['g1_stability']['roc_auc_95_ci'][0]:.3f}, "
        f"{summary10['g1_stability']['roc_auc_95_ci'][1]:.3f}]",
        transform=axes[0].transAxes,
        va="top",
        fontsize=6.8,
        color=GREY,
        bbox={
            "boxstyle": "square,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.9,
        },
    )

    plot_roc(
        axes[1],
        derived["g2_y"],
        [
            (derived["g2_g2"], "G2 geometry", PURPLE, "-"),
            (derived["g2_g1"], "Frozen G1", BLUE, "--"),
        ],
    )
    axes[1].set_title(
        "(b) Phase 10-G2 prospective (N=256; 21 failures)", fontsize=8.5, pad=5
    )
    paired = summary_g2["paired_comparison"]
    axes[1].text(
        0.03,
        0.96,
        r"Paired $\Delta$AUC="
        f"{paired['delta_roc_auc']:.3f}; 95% CI "
        f"[{paired['delta_roc_auc_ci'][0]:.3f}, {paired['delta_roc_auc_ci'][1]:.3f}]\n"
        f"score-swap p={paired['p_perm_score_swap']:.5f}",
        transform=axes[1].transAxes,
        va="top",
        fontsize=6.8,
        color=GREY,
        bbox={
            "boxstyle": "square,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.9,
        },
    )

    fig.suptitle(
        "Failure-risk discrimination for B1 top-16 failure", y=0.985, fontsize=9.8
    )
    fig.text(
        0.5,
        0.01,
        "PROSPECTIVE VALIDATION - empirical step curves - panels are separate cohorts",
        ha="center",
        fontsize=7,
        color=RED,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.18, wspace=0.35)
    return fig


def make_engineering_diagnostics(data: dict[str, Any]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.55))
    screening = data["screening"]
    screen_records = screening["records"]["true_prefix_evaluations"]
    summary = screening["true_prefix_summary"]

    ax = axes[0]
    for record in screen_records:
        values = [record["timing_full_seconds"], record["timing_screened_seconds"]]
        ax.plot([0, 1], values, color="#9A9A9A", lw=0.9, zorder=1)
        ax.scatter(0, values[0], color=BLUE, marker="o", s=24, zorder=2)
        ax.scatter(1, values[1], color=GREEN, marker="s", s=24, zorder=2)
    ax.set_yscale("log")
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(1.0, 65)
    ax.set_xticks([0, 1], labels=["Full\n64-pair pass", "Screened\n16 -> 512 -> 64"])
    ax.set_ylabel("Wall time (seconds, log scale)")
    ax.set_title("Matched-device Stage-2 ranking (N=4)")
    ax.grid(axis="y", color=LIGHT_GREY, linestyle=":", lw=0.7)
    ax.text(
        0.5,
        0.96,
        f"mean {summary['mean_full_seconds']:.2f} s -> {summary['mean_screened_seconds']:.2f} s\n"
        f"paired mean speedup {summary['mean_matched_speedup']:.2f}x\n"
        "top-32 ordering identical: 4/4",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7,
        bbox={
            "boxstyle": "square,pad=0.3",
            "facecolor": "white",
            "edgecolor": LIGHT_GREY,
        },
    )
    add_panel_label(ax, "(a)")

    ax = axes[1]
    runs = data["solver"]["runs"]
    treatments = [
        "1_pycryptosat_default_policy",
        "2_native_default_matrix_limit",
        "3_native_raised_matrix_limit_autodisable_off",
    ]
    labels = [
        "pycryptosat\ndefault",
        "native\ndefault limits",
        "native raised limit\nautodisable off",
    ]
    x = np.arange(3)
    for instance_index in range(3):
        times = [
            min(float(runs[treatment][instance_index]["wall_seconds_raw"]), 60.0)
            for treatment in treatments
        ]
        ax.plot(x, times, color="#A8A8A8", lw=0.8, zorder=1)
        for treatment_index, treatment in enumerate(treatments):
            record = runs[treatment][instance_index]
            if record["verified"]:
                ax.scatter(
                    treatment_index,
                    times[treatment_index],
                    marker="o",
                    s=30,
                    color=GREEN,
                    edgecolor=CHARCOAL,
                    linewidth=0.5,
                    zorder=3,
                )
                ax.text(
                    treatment_index + 0.04,
                    times[treatment_index] - 1.6,
                    f"{record['wall_seconds_raw']:.1f}",
                    fontsize=6.4,
                    color=CHARCOAL,
                    va="top",
                )
            else:
                ax.scatter(
                    treatment_index,
                    60.0,
                    marker="^",
                    s=35,
                    facecolor="white",
                    edgecolor=RED,
                    linewidth=1.0,
                    zorder=3,
                )
    ax.axhline(60, color=RED, lw=0.8, linestyle=":")
    ax.text(2.43, 59.1, "60 s cutoff", ha="right", va="top", fontsize=6.7, color=RED)
    ax.set_xlim(-0.35, 2.45)
    ax.set_ylim(30, 63)
    ax.set_xticks(x, labels=labels)
    ax.set_ylabel("Observed wall time (seconds)")
    ax.set_title("Paired solver configurations (N=3)")
    ax.grid(axis="y", color=LIGHT_GREY, linestyle=":", lw=0.7)
    handles = [
        mlines.Line2D(
            [],
            [],
            marker="o",
            color="none",
            markerfacecolor=GREEN,
            markeredgecolor=CHARCOAL,
            markersize=5,
            label="SAT and verified",
        ),
        mlines.Line2D(
            [],
            [],
            marker="^",
            color="none",
            markerfacecolor="white",
            markeredgecolor=RED,
            markersize=5,
            label="timeout (right-censored)",
        ),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, edgecolor=LIGHT_GREY)
    add_panel_label(ax, "(b)")

    fig.suptitle(
        "Controlled engineering diagnostics on opened Seed 300001",
        y=0.995,
        fontsize=10.2,
    )
    fig.text(
        0.5,
        0.008,
        "ENGINEERING / DIAGNOSTIC - small opened cohorts - config. 3 changes a setting bundle; no mechanism claim",
        ha="center",
        fontsize=7,
        color=RED,
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.85, bottom=0.24, wspace=0.37)
    return fig


def make_seven_round_development_results(data: dict[str, Any]) -> plt.Figure:
    """Show the latest opened-development evidence for the hierarchical method."""
    hierarchical = data["hierarchical"]
    records = hierarchical["records"]
    ranks = data["derived"]["hierarchical_stage1_ranks"]
    wall_seconds = data["derived"]["hierarchical_wall_seconds"]
    successes = np.asarray([bool(record["success"]) for record in records])

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.25))

    ax = axes[0, 0]
    outcome_labels = [
        "Verified\nrecovery",
        "Stage-1 ranking\ntruncation",
        "Solver\ntimeout",
    ]
    outcome_counts = [int(successes.sum()), int((~successes).sum()), 0]
    colors = [GREEN, ORANGE, PURPLE]
    bars = ax.bar(
        np.arange(3),
        outcome_counts,
        width=0.62,
        color=colors,
        edgecolor=CHARCOAL,
        linewidth=0.65,
    )
    for bar, count in zip(bars, outcome_counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(count, 0) + 0.35,
            str(count),
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_xticks(np.arange(3), labels=outcome_labels)
    ax.set_ylim(0, 15.5)
    ax.set_ylabel("Attack instances")
    ax.set_title("Verified end-to-end outcomes", pad=5)
    ci_low, ci_high = hierarchical["wilson_95_ci"]
    ax.text(
        0.5,
        0.90,
        f"13/16 verified (81.25%)\nWilson 95% CI [{ci_low * 100:.1f}%, {ci_high * 100:.1f}%]",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.8,
        color=CHARCOAL,
    )
    ax.grid(axis="y", color=LIGHT_GREY, linestyle=":", linewidth=0.65)
    add_panel_label(ax, "(a)")

    ax = axes[0, 1]
    beam_widths = np.asarray([1, 2, 4, 8, 16, 32, 64, 128])
    containment = ecdf_at(ranks, beam_widths)
    ax.step(
        beam_widths,
        containment,
        where="post",
        color=BLUE,
        linewidth=1.7,
    )
    ax.plot(
        beam_widths,
        containment,
        color=BLUE,
        marker="o",
        markersize=4,
        linewidth=0,
    )
    ax.axvline(32, color=ORANGE, linewidth=1.0, linestyle="--")
    ax.scatter([32], [13 / 16], color=ORANGE, edgecolor=CHARCOAL, s=28, zorder=4)
    ax.annotate(
        "measured beam $M_1=32$\n13/16 retained",
        xy=(32, 13 / 16),
        xytext=(8.5, 0.58),
        fontsize=6.8,
        arrowprops={"arrowstyle": "->", "color": CHARCOAL, "lw": 0.7},
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(beam_widths, labels=[str(value) for value in beam_widths])
    ax.set_xlim(1, 128)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.set_xlabel(r"Stage-1 beam width $M_1$")
    ax.set_ylabel(r"Fraction with true $k_6$ retained")
    ax.set_title("Stage-1 containment (N=16)", pad=5)
    ax.grid(color=LIGHT_GREY, linestyle=":", linewidth=0.65)
    add_panel_label(ax, "(b)")

    ax = axes[1, 0]
    instance_indices = np.arange(1, len(records) + 1)
    ax.scatter(
        instance_indices[successes],
        wall_seconds[successes],
        marker="o",
        s=25,
        color=GREEN,
        edgecolor=CHARCOAL,
        linewidth=0.45,
        label="verified recovery",
        zorder=3,
    )
    ax.scatter(
        instance_indices[~successes],
        wall_seconds[~successes],
        marker="s",
        s=27,
        color=ORANGE,
        edgecolor=CHARCOAL,
        linewidth=0.45,
        label="ranking truncation",
        zorder=3,
    )
    median_wall = float(np.median(wall_seconds))
    ax.axhline(median_wall, color=PURPLE, linewidth=1.0, linestyle="--")
    ax.text(
        16.35,
        median_wall + 0.8,
        f"median {median_wall:.2f} s",
        ha="right",
        va="bottom",
        fontsize=6.6,
        color=PURPLE,
    )
    ax.set_xlim(0.5, 16.5)
    ax.set_ylim(90, 145)
    ax.set_xticks([1, 4, 8, 12, 16])
    ax.set_xlabel("Opened development instance")
    ax.set_ylabel("End-to-end wall time (s)")
    ax.set_title("Measured runtime by outcome", pad=5)
    ax.grid(axis="y", color=LIGHT_GREY, linestyle=":", linewidth=0.65)
    ax.legend(loc="upper left", frameon=False, fontsize=6.7)
    add_panel_label(ax, "(c)")

    ax = axes[1, 1]
    ranking_seconds = np.asarray(
        [record["timing_seconds"]["ranking"] for record in records], dtype=float
    )
    solver_seconds = np.asarray(
        [record["timing_seconds"]["solver"] for record in records], dtype=float
    )
    outcome_groups = [successes, ~successes]
    group_labels = ["Verified\nrecovery", "Stage-1 ranking\ntruncation"]
    ranking_means = [float(np.mean(ranking_seconds[group])) for group in outcome_groups]
    solver_means = [float(np.mean(solver_seconds[group])) for group in outcome_groups]
    x = np.arange(2)
    ax.bar(
        x,
        ranking_means,
        width=0.58,
        color=BLUE,
        edgecolor=CHARCOAL,
        linewidth=0.6,
        label="hierarchical ranking",
    )
    ax.bar(
        x,
        solver_means,
        width=0.58,
        bottom=ranking_means,
        color=PURPLE,
        edgecolor=CHARCOAL,
        linewidth=0.6,
        label="SAT attempts",
    )
    for index, (rank_time, solve_time) in enumerate(
        zip(ranking_means, solver_means, strict=True)
    ):
        ax.text(
            index,
            rank_time / 2,
            f"{rank_time:.1f} s",
            ha="center",
            va="center",
            color="white",
            fontsize=6.7,
            fontweight="bold",
        )
        ax.text(
            index,
            rank_time + solve_time / 2,
            f"{solve_time:.1f} s",
            ha="center",
            va="center",
            color="white",
            fontsize=6.7,
            fontweight="bold",
        )
    ax.set_xticks(x, labels=group_labels)
    ax.set_ylabel("Mean measured time (s)")
    ax.set_title("Online time composition by outcome", pad=5)
    ax.grid(axis="y", color=LIGHT_GREY, linestyle=":", linewidth=0.65)
    ax.legend(loc="upper left", frameon=False, fontsize=6.7)
    add_panel_label(ax, "(d)")

    fig.suptitle(
        "Latest hierarchical seven-round recovery results",
        y=0.992,
        fontsize=10.2,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.945,
        "13/16 independently verified master-key recoveries on opened Seed 300001",
        ha="center",
        fontsize=7.3,
        color=GREEN,
    )
    fig.text(
        0.5,
        0.008,
        "DEVELOPMENT / DIAGNOSTIC (N=16). The earlier frozen oracle gate was 0/64 at 10 s under a different protocol and remains recorded separately.",
        ha="center",
        fontsize=6.8,
        color=CHARCOAL,
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.87,
        bottom=0.13,
        hspace=0.48,
        wspace=0.34,
    )
    return fig


def make_xor_gaussian_solver_ablation(data: dict[str, Any]) -> plt.Figure:
    """Show the opened three-way solver configuration audit without overclaiming."""
    runs = data["solver"]["runs"]
    summary = data["solver"]["summary"]
    treatments = [
        "1_pycryptosat_default_policy",
        "2_native_default_matrix_limit",
        "3_native_raised_matrix_limit_autodisable_off",
    ]
    short_labels = [
        "pycryptosat\ndefault policy",
        "native CLI\ndefault limits",
        "native CLI\nXOR-Gauss bundle",
    ]
    colors = [BLUE, ORANGE, GREEN]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.55))

    ax = axes[0]
    solve_counts = [summary[treatment]["verified_solves"] for treatment in treatments]
    bars = ax.bar(
        np.arange(3),
        solve_counts,
        width=0.62,
        color=colors,
        edgecolor=CHARCOAL,
        linewidth=0.65,
    )
    for bar, count in zip(bars, solve_counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count + 0.08,
            f"{count}/3",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    ax.set_xticks(np.arange(3), labels=short_labels)
    ax.set_ylim(0, 3.25)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_ylabel("SAT and independently verified")
    ax.set_title("Verified recovery count", pad=5)
    ax.grid(axis="y", color=LIGHT_GREY, linestyle=":", linewidth=0.65)
    ax.text(
        0.97,
        0.94,
        "single thread\n60 s cutoff\nN=3 opened instances",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.7,
        color=GREY,
    )
    add_panel_label(ax, "(a)")

    ax = axes[1]
    x = np.arange(3)
    for instance_index in range(3):
        displayed_times = []
        for treatment in treatments:
            record = runs[treatment][instance_index]
            displayed_times.append(min(float(record["wall_seconds_raw"]), 60.0))
        ax.plot(x, displayed_times, color="#A0A0A0", linewidth=0.8, zorder=1)
        for treatment_index, treatment in enumerate(treatments):
            record = runs[treatment][instance_index]
            if record["verified"]:
                ax.scatter(
                    treatment_index,
                    displayed_times[treatment_index],
                    marker="o",
                    s=34,
                    color=GREEN,
                    edgecolor=CHARCOAL,
                    linewidth=0.5,
                    zorder=3,
                )
                ax.text(
                    treatment_index + 0.05,
                    displayed_times[treatment_index] - 1.0,
                    f"{record['wall_seconds_raw']:.1f} s",
                    fontsize=6.4,
                    va="top",
                    color=CHARCOAL,
                )
            else:
                ax.scatter(
                    treatment_index,
                    60.0,
                    marker="^",
                    s=38,
                    facecolor="white",
                    edgecolor=ORANGE,
                    linewidth=1.1,
                    zorder=3,
                )
    ax.axhline(60.0, color=ORANGE, linestyle="--", linewidth=0.9)
    ax.set_xticks(x, labels=short_labels)
    ax.set_xlim(-0.35, 2.45)
    ax.set_ylim(34, 62.5)
    ax.set_ylabel("Observed wall time (s)")
    ax.set_title("Paired instance traces", pad=5)
    ax.grid(axis="y", color=LIGHT_GREY, linestyle=":", linewidth=0.65)
    handles = [
        mlines.Line2D(
            [],
            [],
            marker="o",
            color="none",
            markerfacecolor=GREEN,
            markeredgecolor=CHARCOAL,
            markersize=5,
            label="SAT + verified",
        ),
        mlines.Line2D(
            [],
            [],
            marker="^",
            color="none",
            markerfacecolor="white",
            markeredgecolor=ORANGE,
            markersize=5,
            label="timeout (right-censored)",
        ),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=6.7)
    add_panel_label(ax, "(b)")

    fig.suptitle(
        "Seven-round XOR-aware solver configuration audit",
        y=0.995,
        fontsize=10.0,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.012,
        "ENGINEERING / DIAGNOSTIC. The successful treatment jointly raises the matrix-column limit and disables automatic Gauss shutdown.",
        ha="center",
        fontsize=6.7,
        color=CHARCOAL,
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.84, bottom=0.25, wspace=0.37)
    return fig


def make_project_timeline(_: dict[str, Any]) -> plt.Figure:
    rows = [
        ("Literature review and research gap", 1, 4, 4, "planning"),
        ("Research questions and project proposal", 2, 4, 4, "planning"),
        ("SPECK implementation and correctness tests", 2, 5, None, "build"),
        ("Dataset generation and validation", 3, 6, 6, "build"),
        ("Neural ranking models", 4, 8, 8, "build"),
        ("XOR-SAT integration and key verification", 5, 9, 9, "build"),
        ("Six-round experiments and reliability study", 7, 10, 10, "experiment"),
        ("Seven-round hierarchical experiments", 9, 12, 12, "experiment"),
        ("Classical baseline and solver controls", 10, 13, 13, "experiment"),
        ("Statistical analysis and result checking", 10, 13, None, "reporting"),
        ("Figures, tables and reproducibility package", 11, 14, None, "reporting"),
        ("Report drafting, feedback and revision", 8, 14, 14, "reporting"),
        ("Presentation and final demonstration", 13, 14, 14, "reporting"),
        ("Final seven-round validation, if approved", 13, 14, None, "conditional"),
    ]
    fig, ax = plt.subplots(figsize=(7.16, 5.7))
    y = np.arange(len(rows))[::-1]
    ax.set_xlim(0.5, 14.5)
    ax.set_ylim(-0.65, len(rows) - 0.35)
    ax.set_xticks(range(1, 15), labels=[f"W{i}" for i in range(1, 15)])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_yticks(y, labels=[row[0] for row in rows])
    ax.tick_params(axis="x", length=3, width=0.7, pad=4)
    ax.tick_params(axis="y", length=0, pad=7, labelsize=6.2)
    ax.grid(axis="x", color=LIGHT_GREY, lw=0.55, linestyle=":", zorder=0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_color(LIGHT_GREY)
    for row_y in y[::2]:
        ax.axhspan(row_y - 0.42, row_y + 0.42, color="#FAFAFA", zorder=-1)

    colors = {
        "planning": BLUE,
        "build": ORANGE,
        "experiment": GREEN,
        "reporting": PURPLE,
    }
    for row_y, (_, start, finish, milestone, category) in zip(y, rows, strict=True):
        left = start - 0.38
        width = finish - start + 0.76
        if category == "conditional":
            ax.add_patch(
                patches.Rectangle(
                    (left, row_y - 0.19),
                    width,
                    0.38,
                    facecolor="white",
                    edgecolor=ORANGE,
                    linewidth=0.9,
                    linestyle="--",
                    zorder=2,
                )
            )
        else:
            color = colors[category]
            ax.barh(
                row_y,
                width,
                left=left,
                height=0.38,
                color=color,
                edgecolor="white",
                linewidth=0.35,
                zorder=2,
            )
        if milestone is not None:
            ax.scatter(
                milestone,
                row_y,
                marker="D",
                s=18,
                facecolor="white",
                edgecolor=CHARCOAL,
                linewidth=0.7,
                zorder=4,
            )

    legend_handles = [
        patches.Patch(facecolor=BLUE, edgecolor="none", label="planning and review"),
        patches.Patch(facecolor=ORANGE, edgecolor="none", label="implementation"),
        patches.Patch(facecolor=GREEN, edgecolor="none", label="experiments"),
        patches.Patch(
            facecolor=PURPLE, edgecolor="none", label="analysis and reporting"
        ),
        mlines.Line2D(
            [],
            [],
            marker="D",
            color="none",
            markerfacecolor="white",
            markeredgecolor=CHARCOAL,
            markersize=5,
            label="milestone",
        ),
        patches.Patch(
            facecolor="white",
            edgecolor=ORANGE,
            linestyle="--",
            label="subject to approval",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=False,
        fontsize=5.9,
        columnspacing=1.5,
        handlelength=1.4,
    )
    ax.set_title(
        "Project timeline",
        pad=18,
        fontsize=9.2,
        fontweight="normal",
    )
    fig.subplots_adjust(left=0.36, right=0.985, top=0.87, bottom=0.16)
    for text_item in fig.findobj(match=matplotlib.text.Text):
        text_item.set_fontfamily("DejaVu Sans")
    return fig


def write_manifest(output_paths: list[Path], data: dict[str, Any]) -> Path:
    input_paths: list[Path] = []
    for key, path in INPUTS.items():
        if key == "phase7_records":
            input_paths.extend(sorted(path.glob("instance_*.json")))
        else:
            input_paths.append(path)
    script_path = Path(__file__).resolve()
    input_paths.append(script_path)

    manifest = {
        "schema": "neural_sat_speck.publication_figure_manifest.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "generator": str(script_path.relative_to(ROOT)),
        "generator_sha256": sha256(script_path),
        "evidence_boundary": {
            "phase7_and_classical": "RETROSPECTIVE",
            "phase10_and_phase10_g2": "PROSPECTIVE_VALIDATION",
            "screening_and_solver": "ENGINEERING_DIAGNOSTIC",
            "hierarchical_configuration": "DEVELOPMENT_ONLY",
            "future_seven_round_evaluation": "BLOCKED_PENDING_FREEZE_AND_APPROVAL",
        },
        "verification": {
            "phase7_n": len(data["phase7_records"]),
            "classical_n": len(data["classical_records"]),
            "phase10_n": len(data["phase10_records"]),
            "phase10_positive_events": int(data["derived"]["phase10_y"].sum()),
            "phase10_g2_n": len(data["derived"]["g2_y"]),
            "phase10_g2_positive_events": int(data["derived"]["g2_y"].sum()),
            "phase10_g0_auc_recomputed": data["derived"]["phase10_g0_auc"],
            "phase10_g1_auc_recomputed": data["derived"]["phase10_g1_auc"],
            "phase10_g2_cohort_g1_auc_recomputed": data["derived"]["g2_g1_auc"],
            "phase10_g2_cohort_g2_auc_recomputed": data["derived"]["g2_g2_auc"],
            "phase11a_frozen_verified_recoveries": data["phase11a"]["recovery"][
                "verified_recovery_count"
            ],
            "hierarchical_development_n": len(data["hierarchical"]["records"]),
            "hierarchical_development_verified_recoveries": data["derived"][
                "hierarchical_successes"
            ],
        },
        "inputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in input_paths
            if path.is_file()
        },
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in output_paths
            if path.is_file()
        },
    }
    manifest_path = PDF_DIR / "publication_figure_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate source artifacts and derived values without writing figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    data = load_and_verify()
    if args.verify_only:
        print("Artifact verification passed; no figures written.")
        return

    output_paths: list[Path] = []
    figures = [
        make_hierarchical_stack(data),
        make_rank_ecdf(data),
        make_reliability_roc(data),
        make_engineering_diagnostics(data),
        make_project_timeline(data),
        make_seven_round_development_results(data),
        make_xor_gaussian_solver_ablation(data),
    ]
    for stem, figure in zip(FIGURE_STEMS, figures, strict=True):
        output_paths.extend(save_figure(figure, stem))
    manifest_path = write_manifest(output_paths, data)
    print(f"Generated {len(FIGURE_STEMS)} figures in PDF, SVG, and PNG formats.")
    print(f"PDF directory: {PDF_DIR}")
    print(f"Image directory: {IMAGE_DIR}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
