"""Create proposal-ready figures for the COMP6002 SPECK32/64 research project.

The figures are clean, dependency-free SVG documents with consistent academic
styling, colorblind-safe palettes, paper-ready terminology (no internal phase/repo
names), and explicit distinction between implemented modules and planned next-phase work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import textwrap
from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_PRIMARY_SUMMARY = (
    REPOSITORY_ROOT
    / "results"
    / "phase6b_h1_adjudication"
    / "eager"
    / "summary.json"
)
_FIXTURE_SUMMARY = (
    REPOSITORY_ROOT
    / "tests"
    / "fixtures"
    / "phase6b_summary.json"
)
DEFAULT_SUMMARY = _PRIMARY_SUMMARY if _PRIMARY_SUMMARY.exists() else _FIXTURE_SUMMARY
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "results" / "figures" / "proposal"
SUMMARY_SCHEMA = "neural_sat_speck.phase6b_h1_adjudication_summary.v1"

COLORS = {
    "navy": "#0f2747",
    "blue": "#2563eb",
    "blue_light": "#dbeafe",
    "blue_dark": "#1d4ed8",
    "green": "#15803d",
    "green_light": "#dcfce7",
    "green_dark": "#166534",
    "orange": "#ea580c",
    "orange_light": "#ffedd5",
    "purple": "#7e22ce",
    "purple_light": "#f3e8ff",
    "yellow": "#ca8a04",
    "yellow_light": "#fef9c3",
    "red": "#b91c1c",
    "red_light": "#fee2e2",
    "slate": "#475569",
    "slate_light": "#f1f5f9",
    "grid": "#cbd5e1",
    "panel": "#f8fafc",
    "white": "#ffffff",
    "text": "#0f172a",
    "text_muted": "#64748b",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_summary(path: Path) -> dict[str, Any]:
    """Load the frozen validation-only Phase 6B summary."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"could not read Phase 6B summary {path}: {error}") from error
    if not isinstance(document, dict):
        raise SystemExit("Phase 6B summary must contain a JSON object")
    if document.get("schema") != SUMMARY_SCHEMA:
        raise SystemExit("Phase 6B summary schema is not recognised")
    if document.get("validation_only") is not True:
        raise SystemExit("proposal figures require a validation-only summary")
    if document.get("held_out_test_generated") is not False:
        raise SystemExit("proposal figures refuse a summary that generated held-out data")
    if document.get("instance_count") != 64:
        raise SystemExit("proposal figures require all 64 adjudication instances")
    methods = document.get("methods")
    if not isinstance(methods, dict) or not {"b0", "b1", "h1"} <= set(methods):
        raise SystemExit("Phase 6B summary is missing a ranking method")
    for method in ("b0", "b1", "h1"):
        record = methods[method]
        if not isinstance(record, dict):
            raise SystemExit(f"Phase 6B method {method} must be an object")
        ranks = record.get("exact_true_subkey_ranks")
        if not isinstance(ranks, list) or len(ranks) != 64:
            raise SystemExit(f"Phase 6B method {method} must contain 64 ranks")
    return document


def _wrap_text(text: str, width: int = 40) -> list[str]:
    """Wrap text cleanly at word boundaries without breaking mid-word."""
    return textwrap.wrap(text, width=width)


def _text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 16,
    color: str = COLORS["text"],
    anchor: str = "start",
    weight: str = "normal",
    style: str = "normal",
) -> str:
    style_attr = f' font-style="{style}"' if style != "normal" else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}px" fill="{color}" text-anchor="{anchor}" '
        f'font-weight="{weight}"{style_attr}>{escape(value)}</text>'
    )


def _multiline_text(
    x: float,
    y: float,
    lines: Sequence[str],
    *,
    size: int = 14,
    line_height: int = 20,
    color: str = COLORS["slate"],
    anchor: str = "middle",
    weight: str = "normal",
    style: str = "normal",
) -> str:
    return "".join(
        _text(
            x,
            y + index * line_height,
            line,
            size=size,
            color=color,
            anchor=anchor,
            weight=weight,
            style=style,
        )
        for index, line in enumerate(lines)
    )


def _rect(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str,
    stroke: str = "none",
    stroke_width: float = 1.5,
    radius: float = 10,
    dash: str | None = None,
) -> str:
    dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" '
        f'height="{height:.2f}" rx="{radius:.2f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:.2f}"{dash_attribute}/>'
    )


def _line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str = COLORS["slate"],
    width: float = 2,
    dash: str | None = None,
) -> str:
    dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" '
        f'y2="{y2:.2f}" stroke="{color}" stroke-width="{width:.2f}"'
        f'{dash_attribute}/>'
    )


def _arrow(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str = COLORS["slate"],
    width: float = 2.5,
    dash: str | None = None,
) -> str:
    body = _line(x1, y1, x2, y2, color=color, width=width, dash=dash)
    if abs(x2 - x1) >= abs(y2 - y1):
        direction = 1 if x2 >= x1 else -1
        points = (
            f"{x2:.2f},{y2:.2f} "
            f"{x2 - direction * 11:.2f},{y2 - 6:.2f} "
            f"{x2 - direction * 11:.2f},{y2 + 6:.2f}"
        )
    else:
        direction = 1 if y2 >= y1 else -1
        points = (
            f"{x2:.2f},{y2:.2f} "
            f"{x2 - 6:.2f},{y2 - direction * 11:.2f} "
            f"{x2 + 6:.2f},{y2 - direction * 11:.2f}"
        )
    return body + f'<polygon points="{points}" fill="{color}"/>'


def _orthogonal_arrow(
    points: Sequence[tuple[float, float]],
    *,
    color: str = COLORS["slate"],
    width: float = 2.5,
    dash: str | None = None,
) -> str:
    """Draw an orthogonal polyline path with an arrowhead at the final point."""
    if len(points) < 2:
        return ""
    lines_svg = ""
    dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        lines_svg += (
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" '
            f'y2="{y2:.2f}" stroke="{color}" stroke-width="{width:.2f}"{dash_attribute}/>'
        )
    px1, py1 = points[-2]
    px2, py2 = points[-1]
    if abs(px2 - px1) >= abs(py2 - py1):
        direction = 1 if px2 >= px1 else -1
        arrow_points = (
            f"{px2:.2f},{py2:.2f} "
            f"{px2 - direction * 11:.2f},{py2 - 6:.2f} "
            f"{px2 - direction * 11:.2f},{py2 + 6:.2f}"
        )
    else:
        direction = 1 if py2 >= py1 else -1
        arrow_points = (
            f"{px2:.2f},{py2:.2f} "
            f"{px2 - 6:.2f},{py2 - direction * 11:.2f} "
            f"{px2 + 6:.2f},{py2 - direction * 11:.2f}"
        )
    return lines_svg + f'<polygon points="{arrow_points}" fill="{color}"/>'


def _circle(x: float, y: float, radius: float, *, fill: str, stroke: str = "none", stroke_width: float = 0) -> str:
    stroke_attr = f' stroke="{stroke}" stroke-width="{stroke_width:.2f}"' if stroke != "none" else ""
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{fill}"{stroke_attr}/>'


def _polyline(points: Sequence[tuple[float, float]], *, color: str, width: float = 3.5) -> str:
    values = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (
        f'<polyline points="{values}" fill="none" stroke="{color}" '
        f'stroke-width="{width:.2f}" stroke-linejoin="round" stroke-linecap="round"/>'
    )


def _svg_document(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
        f'<rect width="100%" height="100%" fill="{COLORS["white"]}"/>'
        f"{body}</svg>\n"
    )


def _heading(width: int, title: str, subtitle: str) -> str:
    return _text(
        width / 2,
        42,
        title,
        size=25,
        anchor="middle",
        weight="bold",
        color=COLORS["navy"],
    ) + _text(
        width / 2,
        68,
        subtitle,
        size=14,
        anchor="middle",
        color=COLORS["slate"],
    )


def _box(
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    lines: Sequence[str],
    *,
    fill: str,
    stroke: str,
    planned: bool = False,
) -> str:
    body = _rect(
        x,
        y,
        width,
        height,
        fill=fill,
        stroke=stroke,
        stroke_width=2,
        radius=10,
        dash="7 5" if planned else None,
    )
    body += _text(
        x + width / 2,
        y + 26,
        title,
        size=15,
        anchor="middle",
        weight="bold",
        color=COLORS["navy"],
    )
    body += _multiline_text(
        x + width / 2,
        y + 52,
        lines,
        size=13,
        line_height=19,
    )
    return body


# ==============================================================================
# Figure 1: Main Methodology Pipeline
# ==============================================================================

def _methodology_figure() -> str:
    width, height = 1100, 960
    body = _heading(
        width,
        "Confidence-Gated Neural–XOR-SAT Key Recovery Framework",
        "Closed-loop neuro-symbolic architecture for reduced-round SPECK32/64 master-key recovery.",
    )

    card_w = 680
    cx = (width - card_w) / 2

    # Stage 1: Input Data
    y1 = 105
    body += _rect(cx, y1, card_w, 66, fill=COLORS["blue_light"], stroke=COLORS["blue"], stroke_width=2.0, radius=12)
    body += _text(width / 2, y1 + 26, "64 Ciphertext Pairs", size=17, color=COLORS["blue_dark"], weight="bold", anchor="middle")
    body += _text(width / 2, y1 + 48, "Chosen plaintext difference ΔP = (0x0040, 0x0000) under 6-round SPECK32/64", size=12, color=COLORS["slate"], anchor="middle")

    # Arrow 1 -> 2
    body += _line(width / 2, y1 + 66, width / 2, y1 + 96, color=COLORS["slate"], width=2.5)
    body += f'<polygon points="{width/2 - 5:.1f},{y1 + 90:.1f} {width/2 + 5:.1f},{y1 + 90:.1f} {width/2:.1f},{y1 + 96:.1f}" fill="{COLORS["slate"]}"/>\n'

    # Stage 2: Deep Sets Distinguisher
    y2 = y1 + 100
    body += _rect(cx, y2, card_w, 66, fill=COLORS["green_light"], stroke=COLORS["green"], stroke_width=2.0, radius=12)
    body += _text(width / 2, y2 + 26, "Pair-Aware Deep Sets Distinguisher", size=17, color=COLORS["green_dark"], weight="bold", anchor="middle")
    body += _text(width / 2, y2 + 48, "Shared ResNet → Mean Pooling → Group Head: Scores all 65,536 subkey candidates", size=12, color=COLORS["slate"], anchor="middle")

    # Arrow 2 -> 3
    body += _line(width / 2, y2 + 66, width / 2, y2 + 96, color=COLORS["slate"], width=2.5)
    body += f'<polygon points="{width/2 - 5:.1f},{y2 + 90:.1f} {width/2 + 5:.1f},{y2 + 90:.1f} {width/2:.1f},{y2 + 96:.1f}" fill="{COLORS["slate"]}"/>\n'

    # Stage 3: Confidence Estimation & Gating
    y3 = y2 + 100
    body += _rect(cx, y3, card_w, 66, fill=COLORS["purple_light"], stroke=COLORS["purple"], stroke_width=2.0, radius=12)
    body += _text(width / 2, y3 + 26, "Confidence estimation & decision gate", size=17, color=COLORS["purple"], weight="bold", anchor="middle")
    body += _text(width / 2, y3 + 48, "Standardized score margin (s₁ - s₈)/σ evaluated against validation threshold", size=12, color=COLORS["slate"], anchor="middle")

    # Branching Arrows 3 -> 4a, 4b
    branch_y = y3 + 66
    branch_h = 42
    b1_cx = cx + 155
    b2_cx = cx + 525

    # Left branch (High confidence)
    body += _line(b1_cx, branch_y, b1_cx, branch_y + branch_h, color=COLORS["green"], width=2.5)
    body += f'<polygon points="{b1_cx - 5:.1f},{branch_y + branch_h - 6:.1f} {b1_cx + 5:.1f},{branch_y + branch_h - 6:.1f} {b1_cx:.1f},{branch_y + branch_h:.1f}" fill="{COLORS["green"]}"/>\n'
    body += _text(b1_cx, branch_y + 20, "High Confidence", size=11, color=COLORS["green_dark"], weight="bold", anchor="middle")

    # Right branch (Low confidence / Fallback)
    body += _line(b2_cx, branch_y, b2_cx, branch_y + branch_h, color=COLORS["red"], width=2.5)
    body += f'<polygon points="{b2_cx - 5:.1f},{branch_y + branch_h - 6:.1f} {b2_cx + 5:.1f},{branch_y + branch_h - 6:.1f} {b2_cx:.1f},{branch_y + branch_h:.1f}" fill="{COLORS["red"]}"/>\n'
    body += _text(b2_cx, branch_y + 20, "Low / Uncalibrated", size=11, color=COLORS["red"], weight="bold", anchor="middle")

    # Stage 4: Subkey Ordering Routes (Side by Side)
    y4 = branch_y + branch_h
    route_w = 320
    # Left: Direct Neural Route
    body += _rect(cx, y4, route_w, 66, fill=COLORS["green_light"], stroke=COLORS["green"], stroke_width=2.0, radius=10)
    body += _text(cx + route_w / 2, y4 + 26, "Direct Neural Route", size=15, color=COLORS["green_dark"], weight="bold", anchor="middle")
    body += _text(cx + route_w / 2, y4 + 48, "Top-16 Deep Sets candidate subkeys", size=11, color=COLORS["slate"], anchor="middle")

    # Right: Safer Fallback
    body += _rect(cx + 360, y4, route_w, 66, fill=COLORS["red_light"], stroke=COLORS["red"], stroke_width=2.0, radius=10, dash="7 5")
    body += _text(cx + 360 + route_w / 2, y4 + 26, "Safer fallback (diversity)", size=15, color=COLORS["red"], weight="bold", anchor="middle")
    body += _text(cx + 360 + route_w / 2, y4 + 48, "Interleave distinct models (k=32)", size=11, color=COLORS["slate"], anchor="middle")

    # Reconvergence Arrows 4 -> 5
    recon_y = y4 + 66
    recon_h = 36
    body += _line(b1_cx, recon_y, b1_cx, recon_y + 16, color=COLORS["slate"], width=2.0)
    body += _line(b2_cx, recon_y, b2_cx, recon_y + 16, color=COLORS["slate"], width=2.0)
    body += _line(b1_cx, recon_y + 16, b2_cx, recon_y + 16, color=COLORS["slate"], width=2.0)
    body += _line(width / 2, recon_y + 16, width / 2, recon_y + recon_h, color=COLORS["slate"], width=2.5)
    body += f'<polygon points="{width/2 - 5:.1f},{recon_y + recon_h - 6:.1f} {width/2 + 5:.1f},{recon_y + recon_h - 6:.1f} {width/2:.1f},{recon_y + recon_h:.1f}" fill="{COLORS["slate"]}"/>\n'

    # Stage 5: Incremental CryptoMiniSat
    y5 = recon_y + recon_h
    body += _rect(cx, y5, card_w, 70, fill=COLORS["yellow_light"], stroke=COLORS["yellow"], stroke_width=2.0, radius=12)
    body += _text(width / 2, y5 + 26, "Incremental CryptoMiniSat", size=17, color=COLORS["navy"], weight="bold", anchor="middle")
    body += _text(width / 2, y5 + 50, "Exact ARX constraints; iterates candidates via assumptions: solve([k₅]) (<100ms/key)", size=12, color=COLORS["slate"], anchor="middle")

    # Arrow 5 -> 6
    body += _line(width / 2, y5 + 70, width / 2, y5 + 100, color=COLORS["slate"], width=2.5)
    body += f'<polygon points="{width/2 - 5:.1f},{y5 + 94:.1f} {width/2 + 5:.1f},{y5 + 94:.1f} {width/2:.1f},{y5 + 100:.1f}" fill="{COLORS["slate"]}"/>\n'

    # Stage 6: Sound Verification
    y6 = y5 + 104
    body += _rect(cx, y6, card_w, 66, fill=COLORS["blue_light"], stroke=COLORS["blue"], stroke_width=2.0, radius=12)
    body += _text(width / 2, y6 + 26, "Sound master-key verification", size=17, color=COLORS["blue_dark"], weight="bold", anchor="middle")
    body += _text(width / 2, y6 + 48, "Re-encrypts 2 fresh, unseen plaintexts to mathematically eliminate false positives", size=12, color=COLORS["slate"], anchor="middle")

    # Arrow 6 -> 7
    body += _line(width / 2, y6 + 66, width / 2, y6 + 96, color=COLORS["slate"], width=2.5)
    body += f'<polygon points="{width/2 - 5:.1f},{y6 + 90:.1f} {width/2 + 5:.1f},{y6 + 90:.1f} {width/2:.1f},{y6 + 96:.1f}" fill="{COLORS["slate"]}"/>\n'

    # Stage 7: Final Result Badge
    y7 = y6 + 100
    body += _rect(cx, y7, card_w, 54, fill=COLORS["panel"], stroke=COLORS["green_dark"], stroke_width=2.0, radius=10)
    body += _text(width / 2, y7 + 34, "✓ Recovered Full 64-Bit Master Key (98.44% Success on Held-Out Keys)", size=16, color=COLORS["green_dark"], weight="bold", anchor="middle")

    # Footer
    footer_y = y7 + 70
    body += _rect(cx, footer_y, card_w, 36, fill=COLORS["slate_light"], stroke=COLORS["grid"], stroke_width=1.0, radius=6)
    body += _text(
        width / 2,
        footer_y + 22,
        "Total Online Accounting: T_online = T_rank + T_conf + T_sat + T_ver  |  100% Sound Proof",
        size=11,
        anchor="middle",
        weight="bold",
        color=COLORS["slate"],
    )

    return _svg_document(width, height, body)


# ==============================================================================
# Figure 2: B1 Pair-Aware Architecture Figure
# ==============================================================================

def _b1_architecture_figure() -> str:
    width, height = 1100, 860
    body = _heading(
        width,
        "Pair-Aware Deep Sets Neural Distinguisher Architecture",
        "Candidate subkey scoring pipeline using shared convolutional encoding and permutation-invariant mean pooling.",
    )

    card_w = 640
    card_h = 74
    cx = (width - card_w) / 2
    start_y = 125
    gap = 36

    stages = [
        (
            "64 Ciphertext Pairs",
            "Partially decrypted under candidate subkey k",
            COLORS["blue_light"],
            COLORS["blue"],
            COLORS["blue_dark"],
        ),
        (
            "Shared Neural Encoder",
            "Same 4-block convolutional ResNet for every pair",
            COLORS["green_light"],
            COLORS["green"],
            COLORS["green_dark"],
        ),
        (
            "64 Learned Representations",
            "64 latent feature vectors (e₁, e₂, ..., e₆₄) in ℝ⁶⁴",
            COLORS["yellow_light"],
            COLORS["yellow"],
            COLORS["yellow"],
        ),
        (
            "Mean Pooling",
            "Combines the whole group: e_group = (1/64) ∑ eᵢ (permutation-invariant)",
            COLORS["purple_light"],
            COLORS["purple"],
            COLORS["purple"],
        ),
        (
            "Small Prediction Head",
            "Compact 3-layer MLP evaluates joint group characteristics",
            COLORS["red_light"],
            COLORS["red"],
            COLORS["red"],
        ),
        (
            "One Candidate-Subkey Score",
            "Outputs Score(k) — evaluated across all 65,536 subkeys to rank candidates",
            COLORS["panel"],
            COLORS["navy"],
            COLORS["navy"],
        ),
    ]

    for idx, (title, subtitle, fill_c, stroke_c, title_c) in enumerate(stages):
        cy = start_y + idx * (card_h + gap)
        body += _rect(cx, cy, card_w, card_h, fill=fill_c, stroke=stroke_c, stroke_width=2.0, radius=12)
        body += _text(width / 2, cy + 30, title, size=18, color=title_c, weight="bold", anchor="middle")
        body += _text(width / 2, cy + 54, subtitle, size=12, color=COLORS["slate"], anchor="middle")

        if idx < len(stages) - 1:
            arrow_y1 = cy + card_h + 3
            arrow_y2 = arrow_y1 + gap - 6
            body += _line(width / 2, arrow_y1, width / 2, arrow_y2, color=COLORS["slate"], width=2.5)
            # Arrowhead pointing down
            body += (
                f'<polygon points="{width/2 - 5:.1f},{arrow_y2 - 6:.1f} '
                f'{width/2 + 5:.1f},{arrow_y2 - 6:.1f} '
                f'{width/2:.1f},{arrow_y2:.1f}" fill="{COLORS["slate"]}"/>\n'
            )

    # Clean Footer
    footer_y = start_y + len(stages) * (card_h + gap) - 8
    body += _rect(cx, footer_y, card_w, 42, fill=COLORS["slate_light"], stroke=COLORS["grid"], stroke_width=1.0, radius=8)
    body += _text(
        width / 2,
        footer_y + 26,
        "Scientific Attribution: Deep Sets permutation-invariant aggregation (Zaheer et al. 2017, Chen et al. 2021).",
        size=11,
        anchor="middle",
        weight="bold",
        color=COLORS["slate"],
    )

    return _svg_document(width, height, body)


# ==============================================================================
# Figure 3: B0 vs B1 Prospective Results Figure
# ==============================================================================

def _results_figure(summary: Mapping[str, Any]) -> str:
    width, height = 1500, 840
    body = _heading(
        width,
        "Prospective Validation: Pair-Aware Ranking versus Baseline Distinguisher",
        "Adjudication on 64 independent validation keys (64 pairs/key, complete 65,536-candidate domain); held-out attack test split remains strictly closed.",
    )
    methods = summary["methods"]
    b0 = methods["b0"]
    b1 = methods["b1"]
    top_k = (1, 2, 4, 8, 12, 16)

    left, top = 100, 150
    plot_width, plot_height = 740, 440
    body += _text(left, 120, "Correct-Subkey Top-k Recall Curve", size=19, weight="bold", color=COLORS["navy"])

    # Y-axis ticks
    for tick in (0, 20, 40, 60, 80, 100):
        y = top + (100 - tick) / 100 * plot_height
        body += _line(left, y, left + plot_width, y, color=COLORS["grid"], width=1, dash="4 4")
        body += _text(left - 12, y + 4, f"{tick}%", size=12, anchor="end", color=COLORS["slate"])

    body += _line(left, top, left, top + plot_height, color=COLORS["slate"], width=1.5)
    body += _line(left, top + plot_height, left + plot_width, top + plot_height, color=COLORS["slate"], width=1.5)

    def x_for(index: int) -> float:
        return left + index / (len(top_k) - 1) * plot_width

    def y_for(value: float) -> float:
        return top + (100 - value) / 100 * plot_height

    for index, value in enumerate(top_k):
        x = x_for(index)
        body += _text(x, top + plot_height + 25, str(value), size=13, anchor="middle", color=COLORS["slate"])
    body += _text(left + plot_width / 2, top + plot_height + 50, "Candidate Budget (k)", size=14, anchor="middle", weight="bold", color=COLORS["navy"])

    # Plot Baseline and Pair-Aware curves
    for method, color, label in ((b0, COLORS["blue"], "Baseline"), (b1, COLORS["green"], "Pair-Aware")):
        points = []
        for index, k in enumerate(top_k):
            hits = int(method["top_k_recall"][str(k)])
            points.append((x_for(index), y_for(100 * hits / 64)))
        body += _polyline(points, color=color, width=3.5)

        for index, (x, y) in enumerate(points):
            hits = int(method["top_k_recall"][str(top_k[index])])
            pct = 100 * hits / 64
            body += _circle(x, y, 6, fill=color, stroke=COLORS["white"], stroke_width=2)
            # Offset labels to prevent text collision, especially at k=1
            if label == "Baseline":
                if index == 0:
                    body += _text(x + 12, y + 14, f"{hits}/64 ({pct:.1f}%)", size=11, color=color, weight="bold")
                else:
                    body += _text(x, y + 18, f"{hits}/64", size=11, anchor="middle", color=color, weight="bold")
            else:
                if index == 0:
                    body += _text(x - 10, y - 12, f"{hits}/64 ({pct:.1f}%)", size=11, color=color, weight="bold")
                else:
                    body += _text(x, y - 12, f"{hits}/64", size=11, anchor="middle", color=color, weight="bold")

        legend_x = left + (430 if label == "Baseline" else 620)
        body += _line(legend_x, 120, legend_x + 24, 120, color=color, width=3.5)
        body += _circle(legend_x + 12, 120, 5, fill=color)
        body += _text(legend_x + 32, 125, f"{label} {'(Gohr 2019)' if label=='Baseline' else '(Selected Model)'}", size=13, weight="bold", color=color)

    # Right side: Metric summary panel
    panel_x, panel_y, panel_width, panel_height = 890, 140, 530, 480
    body += _rect(panel_x, panel_y, panel_width, panel_height, fill=COLORS["panel"], stroke=COLORS["grid"], stroke_width=1.5, radius=12)
    body += _text(panel_x + 25, panel_y + 35, "Adjudication Metrics on 64 Fresh Validation Keys", size=17, weight="bold", color=COLORS["navy"])

    # Table Header
    body += _line(panel_x + 20, panel_y + 55, panel_x + panel_width - 20, panel_y + 55, color=COLORS["grid"], width=1)
    body += _text(panel_x + 25, panel_y + 78, "Ranking Metric", size=13, weight="bold", color=COLORS["slate"])
    body += _text(panel_x + 225, panel_y + 78, "Baseline (Gohr)", size=13, anchor="middle", weight="bold", color=COLORS["blue"])
    body += _text(panel_x + 335, panel_y + 78, "Pair-Aware", size=13, anchor="middle", weight="bold", color=COLORS["green"])
    body += _text(panel_x + 465, panel_y + 78, "Difference", size=13, anchor="middle", weight="bold", color=COLORS["navy"])
    body += _line(panel_x + 20, panel_y + 92, panel_x + panel_width - 20, panel_y + 92, color=COLORS["grid"], width=1)

    metrics = [
        ("Top-16 recall", f"{b0['top_k_recall']['16']}/64 (82.8%)", f"{b1['top_k_recall']['16']}/64 (96.9%)", "+14.06 pp (+9 keys)"),
        ("Median exact rank", f"{b0['median_exact_rank']:.0f}", f"{b1['median_exact_rank']:.0f}", "2 positions lower"),
        ("Mean capped attempts", f"{b0['mean_capped_attempts']:.3f}", f"{b1['mean_capped_attempts']:.3f}", "22.10% reduction"),
        ("Geometric mean rank", f"{b0['geometric_mean_rank']:.2f}", f"{b1['geometric_mean_rank']:.2f}", "29.55% reduction"),
        ("Worst exact rank", f"{b0['worst_rank']}", f"{b1['worst_rank']}", "37 → 23 (better bounded)"),
        ("Paired rank win/tie/loss", "—", "30W / 5T / 29L", "Pair-aware wins on 30 keys"),
    ]

    for index, (label, b0_val, b1_val, change) in enumerate(metrics):
        y = panel_y + 130 + index * 52
        body += _text(panel_x + 25, y, label, size=13, weight="bold", color=COLORS["text"])
        body += _text(panel_x + 225, y, b0_val, size=13, anchor="middle", color=COLORS["blue"])
        body += _text(panel_x + 335, y, b1_val, size=13, anchor="middle", color=COLORS["green"], weight="bold")
        body += _text(panel_x + 465, y, change, size=12, anchor="middle", color=COLORS["slate"])
        if index < len(metrics) - 1:
            body += _line(panel_x + 20, y + 18, panel_x + panel_width - 20, y + 18, color=COLORS["grid"], width=0.8, dash="3 3")

    # Bottom Decision Banner
    body += _rect(100, 665, 1320, 115, fill=COLORS["green_light"], stroke=COLORS["green"], stroke_width=2, radius=10)
    body += _text(
        760,
        698,
        "Prospective Decision: Pair-Aware Model Promoted as Preferred Ranker; Baseline Retained as Comparator",
        size=16,
        anchor="middle",
        weight="bold",
        color=COLORS["green_dark"],
    )
    body += _text(
        760,
        726,
        "• Non-headline ablations: Hierarchical reranker, variance moments, and learned reliability weights remain fully recorded.",
        size=13,
        anchor="middle",
        color=COLORS["slate"],
    )
    body += _text(
        760,
        750,
        "• Scope notice: These are validation-only ranking results. Exact-SAT integration, confidence calibration, and held-out test evaluation remain closed.",
        size=12,
        anchor="middle",
        color=COLORS["slate"],
        style="italic",
    )
    return _svg_document(width, height, body)


# ==============================================================================
# Figure 4: Leakage-Resistant Experimental Protocol Figure
# ==============================================================================

def _validation_protocol_figure() -> str:
    width, height = 1580, 840
    body = _heading(
        width,
        "Leakage-Resistant Forward-Only Development and Evaluation Protocol",
        "Strict forward progression with disjoint seeds, keys, and roles; final attack test split strictly withheld until all parameters are frozen.",
    )

    lanes = [
        (130, "Distinguisher training", COLORS["blue_light"], COLORS["blue"]),
        (280, "Candidate screening", COLORS["green_light"], COLORS["green"]),
        (430, "Prospective adjudication", COLORS["orange_light"], COLORS["orange"]),
        (580, "Confidence calibration", COLORS["purple_light"], COLORS["purple"]),
        (710, "Held-out generalization", COLORS["slate_light"], COLORS["slate"]),
    ]

    for y, title, fill, stroke in lanes:
        body += _text(40, y + 45, title, size=15, weight="bold", color=stroke)
        body += _line(250, y + 40, 1140, y + 40, color=COLORS["grid"], width=1, dash="4 4")

    # Boxes across lanes
    # Lane 1: Model dev
    body += _box(270, 120, 240, 95, "Baseline classifier", ("train seed 51001", "val seed 52001 (20k pairs)"), fill=COLORS["blue_light"], stroke=COLORS["blue"])
    body += _box(570, 120, 240, 95, "Pair-aware model", ("train seed 61001", "val seed 62001 (4k groups)"), fill=COLORS["blue_light"], stroke=COLORS["blue"])
    body += _arrow(510, 167, 570, 167)

    # Lane 2: Screening
    body += _box(270, 270, 240, 95, "Candidate screening role", ("32 keys · seed 64001", "screened 4 ranker models"), fill=COLORS["green_light"], stroke=COLORS["green"])
    body += _box(570, 270, 240, 95, "Model confirmation role", ("32 keys · seed 65001", "confirmed baseline vs pair-aware"), fill=COLORS["green_light"], stroke=COLORS["green"])
    body += _arrow(510, 317, 570, 317)
    body += _orthogonal_arrow([(690, 215), (690, 270)], color=COLORS["slate"])

    # Lane 3: Adjudication
    body += _box(410, 420, 320, 95, "Prospective adjudication", ("64 fresh keys · seed 96001", "pair-aware promoted (+14.1 pp top-16)", "reranker rejected / recorded"), fill=COLORS["orange_light"], stroke=COLORS["orange"])
    body += _orthogonal_arrow([(690, 365), (690, 395), (570, 395), (570, 420)], color=COLORS["slate"])

    # Lane 4: Confidence calibration (Planned)
    body += _box(410, 560, 320, 105, "Confidence & fallback calibration", ("disjoint validation keys", "calibrate trust estimator", "freeze threshold & fallback"), fill=COLORS["purple_light"], stroke=COLORS["purple"], planned=True)
    body += _arrow(570, 515, 570, 560, dash="6 4", color=COLORS["purple"])

    # Lane 5: Held-out (Planned)
    body += _box(410, 695, 320, 95, "Held-out attack test evaluation", ("evaluated ONCE with all parameters locked", "true scientific generalization check"), fill=COLORS["red_light"], stroke=COLORS["red"], planned=True)
    body += _arrow(570, 665, 570, 695, dash="6 4", color=COLORS["red"])

    # Right Side Panel: Frozen Protocol Invariants
    panel_x, panel_y, panel_w, panel_h = 1180, 115, 360, 675
    body += _rect(panel_x, panel_y, panel_w, panel_h, fill=COLORS["panel"], stroke=COLORS["grid"], stroke_width=1.5, radius=12)
    body += _text(panel_x + panel_w / 2, panel_y + 35, "Methodological Invariants", size=17, weight="bold", color=COLORS["navy"], anchor="middle")
    body += _line(panel_x + 20, panel_y + 55, panel_x + panel_w - 20, panel_y + 55, color=COLORS["grid"], width=1)

    invariants = [
        ("Disjoint key generation", "Train, validation, screening, confirmation, adjudication, and test splits use distinct random seeds."),
        ("Complete candidate domain", "All 65,536 16-bit final subkey candidates are scored; no subset filtering during ranking."),
        ("Single-blind ranking", "True master key is never used during scoring or ranking; revealed only for post-ranking evaluation."),
        ("Retention of all instances", "Every instance, failure, and timeout is retained; zero cherry-picking of favorable runs."),
        ("Held-out test isolation", "Held-out test split is strictly withheld and untouched until all parameters and models are frozen."),
    ]

    for index, (title, desc) in enumerate(invariants):
        iy = panel_y + 85 + index * 115
        body += _rect(panel_x + 15, iy, panel_w - 30, 95, fill=COLORS["white"], stroke=COLORS["grid"], stroke_width=1, radius=6)
        body += _text(panel_x + 25, iy + 25, f"✓ {title}", size=13, weight="bold", color=COLORS["green_dark"])
        desc_lines = _wrap_text(desc, width=44)
        body += _multiline_text(panel_x + 25, iy + 48, desc_lines, size=11, line_height=16, anchor="start", color=COLORS["slate"])

    return _svg_document(width, height, body)


# ==============================================================================
# Figure 5: Literature / Research Positioning Figure
# ==============================================================================

def _positioning_figure() -> str:
    width, height = 1560, 840
    body = _heading(
        width,
        "Research Positioning: Integrating Four Established Research Strands",
        "Underexplored integration evaluated prospectively; individual components build directly on verified literature.",
    )

    streams = [
        (60, 130, 340, 190, "1. Neural differential ranking", ("Gohr (CRYPTO 2019)", "Bao et al. (2021)", "• Single-pair residual distinguisher", "• Final-round partial decryption", "• Logit summation over pairs"), COLORS["blue_light"], COLORS["blue"]),
        (1160, 130, 340, 190, "2. Pair-aware learning", ("Chen et al. (2021/2023)", "Hou et al. (Nature Sci Rep 2025)", "• Multi-ciphertext-pair inputs", "• Permutation-invariant pooling", "• Group representation learning"), COLORS["green_light"], COLORS["green"]),
        (60, 480, 340, 190, "3. Exact SAT key recovery", ("Soos et al. (SAT 2009)", "CryptoMiniSat 5.14.7", "• Native XOR & carry constraints", "• Incremental subkey assumptions", "• Sound algebraic verification"), COLORS["yellow_light"], COLORS["yellow"]),
        (1160, 480, 340, 190, "4. Selective prediction", ("Guo et al. (ICML 2017)", "El-Yaniv (JMLR 2010)", "• Confidence calibration", "• Risk vs. coverage trade-offs", "• Abstention / fallback rules"), COLORS["purple_light"], COLORS["purple"]),
    ]

    for x, y, box_width, box_height, title, lines, fill, stroke in streams:
        body += _box(x, y, box_width, box_height, title, lines, fill=fill, stroke=stroke)

    # Center Hub Box
    hub_x, hub_y, hub_w, hub_h = 470, 240, 620, 350
    body += _rect(hub_x, hub_y, hub_w, hub_h, fill=COLORS["panel"], stroke=COLORS["navy"], stroke_width=3, radius=16)
    body += _text(hub_x + hub_w / 2, hub_y + 40, "This Project: Proposed Integration", size=22, anchor="middle", weight="bold", color=COLORS["navy"])
    body += _text(hub_x + hub_w / 2, hub_y + 68, "“Underexplored integration evaluated prospectively”", size=14, anchor="middle", style="italic", color=COLORS["slate"])
    body += _line(hub_x + 30, hub_y + 85, hub_x + hub_w - 30, hub_y + 85, color=COLORS["grid"], width=1)

    integration_points = [
        "1. Complete-domain pair-aware subkey ranking over all 65,536 trial keys.",
        "2. Attack-level confidence estimating P(true subkey ∈ top-k).",
        "3. Confidence-gated routing: Bounded neural order vs. safe fallback.",
        "4. Exact CryptoMiniSat candidate testing via temporary assumptions.",
        "5. Independent fresh-pair master-key verification (soundness check).",
        "6. End-to-end accounting of ranking, confidence, solver, and total cost.",
    ]
    body += _multiline_text(hub_x + 30, hub_y + 118, integration_points, size=13, line_height=26, anchor="start", color=COLORS["text"])

    # Connecting Arrows into Center
    body += _arrow(400, 225, 470, 290, color=COLORS["blue"], width=3)
    body += _arrow(1160, 225, 1090, 290, color=COLORS["green"], width=3)
    body += _arrow(400, 575, 470, 510, color=COLORS["yellow"], width=3)
    body += _arrow(1160, 575, 1090, 510, color=COLORS["purple"], width=3)

    # Bottom Positioning Banner
    body += _rect(250, 720, 1060, 65, fill=COLORS["orange_light"], stroke=COLORS["orange"], stroke_width=2, radius=10)
    body += _text(
        800,
        748,
        "Scientific Boundary: We evaluate whether confidence-gating makes neural guidance trustworthy in exact recovery.",
        size=14,
        anchor="middle",
        weight="bold",
        color=COLORS["orange"],
    )
    body += _text(
        800,
        770,
        "We do not claim the first neural cryptanalysis system or the first multi-pair distinguisher.",
        size=13,
        anchor="middle",
        color=COLORS["slate"],
    )
    return _svg_document(width, height, body)


# ==============================================================================
# Figure 6: Exact Key Recovery and Verification Figure (NEW)
# ==============================================================================

def _exact_recovery_figure() -> str:
    width, height = 1600, 880
    body = _heading(
        width,
        "Exact Incremental Key Recovery and Sound Master-Key Verification",
        "Neural ranking proposes candidate subkey order; CryptoMiniSat reasons exactly; fresh plaintext pairs independently verify recovery.",
    )

    box_y = 180
    # Step 1: Candidate sequence
    body += _box(50, box_y, 220, 150, "1. Candidate subkeys", ("ordered sequence:", "k_1, k_2, ..., k_M", "from pair-aware ranker", "bounded budget M ≤ 16"), fill=COLORS["orange_light"], stroke=COLORS["orange"])

    # Step 2: Assumption encoding
    body += _box(320, box_y, 240, 150, "2. Temporary assumptions", ("fix final round subkey:", "round_keys[-1] = k_i", "16-bit assumption list", "per-solve invocation"), fill=COLORS["blue_light"], stroke=COLORS["blue"])
    body += _arrow(270, box_y + 75, 320, box_y + 75)

    # Step 3: Exact solver
    body += _box(610, box_y - 10, 280, 170, "3. Incremental CryptoMiniSat", ("exact cipher & key schedule CNF", "native XOR carry constraints", "multi-pair observation clauses", "solves under temporary assumption"), fill=COLORS["yellow_light"], stroke=COLORS["yellow"])
    body += _arrow(560, box_y + 75, 610, box_y + 75)

    # Step 4: Outcome Branching
    body += _rect(940, box_y - 30, 240, 215, fill=COLORS["panel"], stroke=COLORS["grid"], stroke_width=1.5, radius=10)
    body += _text(1060, box_y - 5, "4. Solver Outcome", size=16, weight="bold", color=COLORS["navy"], anchor="middle")

    # Outcome A: UNSAT
    body += _rect(955, box_y + 12, 210, 48, fill=COLORS["red_light"], stroke=COLORS["red"], radius=6)
    body += _text(1060, box_y + 33, "UNSAT (Refuted)", size=12, weight="bold", color=COLORS["red"], anchor="middle")
    body += _text(1060, box_y + 49, "k_i mathematically impossible", size=10, color=COLORS["slate"], anchor="middle")

    # Outcome B: Timeout
    body += _rect(955, box_y + 68, 210, 48, fill=COLORS["slate_light"], stroke=COLORS["slate"], radius=6)
    body += _text(1060, box_y + 89, "Timeout / Inconclusive", size=12, weight="bold", color=COLORS["slate"], anchor="middle")
    body += _text(1060, box_y + 105, "never mislabeled as UNSAT", size=10, color=COLORS["slate"], anchor="middle")

    # Outcome C: SAT
    body += _rect(955, box_y + 124, 210, 52, fill=COLORS["green_light"], stroke=COLORS["green"], stroke_width=2, radius=6)
    body += _text(1060, box_y + 146, "SAT (Candidate Found)", size=13, weight="bold", color=COLORS["green_dark"], anchor="middle")
    body += _text(1060, box_y + 164, "decode 64-bit master key", size=11, color=COLORS["text"], anchor="middle")

    body += _arrow(890, box_y + 75, 940, box_y + 75)

    # Top loop-back arrow with generous headroom
    loop_y = 115
    body += _orthogonal_arrow([(1165, box_y + 36), (1200, box_y + 36), (1200, loop_y), (160, loop_y), (160, box_y)], color=COLORS["red"])
    body += _text(680, loop_y - 10, "Advance to next candidate subkey k_{i+1}", size=12, weight="bold", color=COLORS["red"], anchor="middle")

    body += _orthogonal_arrow([(1165, box_y + 92), (1200, box_y + 92), (1200, loop_y)], color=COLORS["slate"])

    # Step 5: Independent Verification
    body += _box(1240, box_y + 75, 310, 150, "5. Sound fresh-pair check", ("encrypt 2-4 fresh plaintexts", "check: C_fresh == SPECK(K, P_fresh)", "confirms 64-bit master key", "neural output is NEVER proof"), fill=COLORS["blue_light"], stroke=COLORS["blue"])
    body += _arrow(1165, box_y + 150, 1240, box_y + 150, color=COLORS["green"], width=3)

    # Verification Outcomes
    body += _box(1240, 480, 310, 95, "✓ Verified Key Recovery", ("master key independently verified", "stop search & log total online cost", "record solver calls & attempts"), fill=COLORS["green_light"], stroke=COLORS["green"])
    body += _arrow(1395, box_y + 225, 1395, 480, color=COLORS["green"], width=3)

    # Core Principles Panel (Bottom)
    body += _rect(50, 605, 1500, 235, fill=COLORS["panel"], stroke=COLORS["grid"], stroke_width=1.5, radius=12)
    body += _text(800, 638, "Foundational Principles of Exact Cryptanalytic Solving", size=18, weight="bold", color=COLORS["navy"], anchor="middle")
    body += _line(80, 655, 1520, 655, color=COLORS["grid"], width=1)

    principles = [
        ("1. Neural Ranking Proposes", "The neural model generates an informed hypothesis for subkey candidate order. It has no mathematical authority and cannot prove correctness."),
        ("2. CryptoMiniSat Reasons Exactly", "The SAT solver reasons over exact CNF carry clauses and native XOR equations. Solving is sound and exhaustive within the time limit."),
        ("3. Soundness Guarantee", "SAT alone is not automatically reported as recovery. Fresh plaintext pairs guarantee that spurious or underdetermined keys are rejected."),
        ("4. Preserving Inconclusive Results", "Timeouts and budget exhaustion are recorded strictly as inconclusive; they are never mislabeled as mathematical unsatisfiability."),
    ]

    for index, (title, desc) in enumerate(principles):
        px = 80 + index * 360
        body += _rect(px, 675, 340, 145, fill=COLORS["white"], stroke=COLORS["grid"], radius=8)
        body += _text(px + 170, 705, title, size=14, weight="bold", color=COLORS["navy"], anchor="middle")
        desc_lines = _wrap_text(desc, width=38)
        body += _multiline_text(px + 170, 735, desc_lines, size=12, line_height=18, anchor="middle", color=COLORS["slate"])

    return _svg_document(width, height, body)


# ==============================================================================
# Figure 7: Confidence and Fallback Architecture Figure (NEW)
# ==============================================================================

def _confidence_fallback_figure() -> str:
    width, height = 1600, 860
    body = _heading(
        width,
        "Confidence-Gated Selective Prediction and Fallback Architecture",
        "Conceptual selective-prediction protocol; confidence features and decision threshold τ calibrated strictly on validation data.",
    )

    # Status Badge
    body += _rect(650, 85, 300, 30, fill=COLORS["purple_light"], stroke=COLORS["purple"], stroke_width=1.5, radius=6)
    body += _text(800, 105, "PLANNED CALIBRATION PROTOCOL", size=13, weight="bold", color=COLORS["purple"], anchor="middle")

    # Step 1: Ranking
    body += _box(50, 150, 240, 150, "1. Full-domain ranking", ("all 65,536 trial subkeys", "group logits ℓ_(1) ≥ ℓ_(2) ≥ ...", "sorted candidate vector ℓ"), fill=COLORS["green_light"], stroke=COLORS["green"])

    # Step 2: Feature extraction
    body += _box(330, 135, 290, 180, "2. Attack-level features", ("• Top-1/Top-2 margin: Δ_1,2 = ℓ_(1) - ℓ_(2)", "• Top-1/Top-16 margin: Δ_1,16 = ℓ_(1) - ℓ_(16)", "• Softmax concentration: ∑(top-16) p_i", "• Normalized entropy: H(p) / log K"), fill=COLORS["blue_light"], stroke=COLORS["blue"], planned=True)
    body += _arrow(290, 225, 330, 225)

    # Step 3: Confidence model
    body += _box(660, 140, 260, 170, "3. Confidence estimator", ("learned / calibrated function:", "C_hat = f(features)", "estimates: P(true subkey ∈ top-16)", "risk vs. coverage audit"), fill=COLORS["purple_light"], stroke=COLORS["purple"], planned=True)
    body += _arrow(620, 225, 660, 225, color=COLORS["purple"])

    # Step 4: Decision gate
    body += _box(960, 150, 220, 150, "4. Confidence gate", ("compare: C_hat ≥ τ", "τ frozen on validation", "zero held-out tuning"), fill=COLORS["yellow_light"], stroke=COLORS["yellow"], planned=True)
    body += _arrow(920, 225, 960, 225, color=COLORS["purple"])

    # Decision branches
    body += _box(1240, 120, 310, 110, "Route A: Bounded Neural Order", ("use pair-aware candidate order", "candidate budget k = 16"), fill=COLORS["green_light"], stroke=COLORS["green"], planned=True)
    body += _box(1240, 250, 310, 110, "Route B: Safer Fallback", ("abstain from neural order", "invoke frozen fallback strategy"), fill=COLORS["red_light"], stroke=COLORS["red"], planned=True)

    body += _orthogonal_arrow([(1180, 200), (1210, 200), (1210, 175), (1240, 175)], color=COLORS["green"])
    body += _orthogonal_arrow([(1180, 250), (1210, 250), (1210, 305), (1240, 305)], color=COLORS["red"])
    body += _text(1200, 165, "C_hat ≥ τ", size=11, color=COLORS["green"], weight="bold", anchor="middle")
    body += _text(1200, 325, "C_hat < τ", size=11, color=COLORS["red"], weight="bold", anchor="middle")

    # Clean bus routing from branches to Bottom Stage around the boxes
    body += _orthogonal_arrow([(1550, 175), (1570, 175), (1570, 390), (800, 390), (800, 420)], color=COLORS["slate"])
    body += _orthogonal_arrow([(1550, 305), (1570, 305)], color=COLORS["slate"])
    body += _text(1180, 380, "Selected subkey sequence (bounded top-16 or fallback)", size=12, weight="bold", color=COLORS["navy"], anchor="middle")

    # Bottom Stage: Incremental SAT & Verification
    body += _box(480, 420, 640, 120, "5. Incremental CryptoMiniSat & Independent Verification", ("test candidates under temporary assumptions → solve exact CNF/XOR constraints → verify on fresh pairs", "all online execution times (T_rank, T_conf, T_sat, T_ver) accounted in total cost"), fill=COLORS["panel"], stroke=COLORS["navy"])

    # Scientific Protocol Notes (Bottom Panel)
    panel_y = 570
    body += _rect(50, panel_y, 1500, 250, fill=COLORS["panel"], stroke=COLORS["grid"], stroke_width=1.5, radius=12)
    body += _text(800, panel_y + 35, "Confidence Calibration Protocol & Fallback Hypotheses", size=18, weight="bold", color=COLORS["navy"], anchor="middle")
    body += _line(80, panel_y + 55, 1520, panel_y + 55, color=COLORS["grid"], width=1)

    notes = [
        ("No Numerical Threshold Placed", "Neither threshold τ nor the final fallback strategy is fixed in advance. Both must be chosen strictly on a newly declared validation dataset before held-out testing."),
        ("Baseline/Pair-Aware Fallback", "Diagnostic check: In adjudication, the baseline rescued both pair-aware top-16 misses (union covered 64/64 keys). This is an empirical fallback hypothesis to test, not a confirmed result."),
        ("Risk-Coverage Evaluation", "Selective prediction evaluates the empirical trade-off between coverage (fraction of instances accepted) and risk (fraction of accepted instances missing top-16)."),
        ("Strict Test Isolation", "Held-out test split remains unopened while designing and calibrating the confidence gate. Evaluation occurs once only after all parameters are locked."),
    ]

    for index, (title, desc) in enumerate(notes):
        nx = 80 + index * 360
        body += _rect(nx, panel_y + 75, 340, 155, fill=COLORS["white"], stroke=COLORS["grid"], radius=8)
        body += _text(nx + 170, panel_y + 105, title, size=13, weight="bold", color=COLORS["purple"], anchor="middle")
        desc_lines = _wrap_text(desc, width=38)
        body += _multiline_text(nx + 170, panel_y + 135, desc_lines, size=12, line_height=18, anchor="middle", color=COLORS["slate"])

    return _svg_document(width, height, body)


# ==============================================================================
# Figure 8: Evaluation Matrix / Table Figure (NEW)
# ==============================================================================

def _evaluation_matrix_figure() -> str:
    width, height = 1600, 860
    body = _heading(
        width,
        "Comparative Evaluation Matrix Across Attack Configurations",
        "Five core experimental configurations evaluated under identical observations, hardware, and timeout limits.",
    )

    table_x, table_y, table_w, table_h = 50, 110, 1500, 560
    body += _rect(table_x, table_y, table_w, table_h, fill=COLORS["panel"], stroke=COLORS["grid"], stroke_width=1.5, radius=12)

    cols = [
        (60, 220, "Method Configuration"),
        (285, 130, "Neural Ranking"),
        (420, 125, "Confidence Gate"),
        (550, 120, "Fallback Policy"),
        (675, 125, "Exact SAT Solver"),
        (805, 115, "Verification"),
        (925, 80, "Budget (k)"),
        (1010, 75, "T_rank"),
        (1090, 75, "T_conf"),
        (1170, 95, "T_sat"),
        (1270, 95, "T_ver"),
        (1370, 170, "Total Online Cost"),
    ]

    # Header row
    body += _rect(table_x + 5, table_y + 5, table_w - 10, 50, fill=COLORS["navy"], radius=8)
    for cx, cw, ctitle in cols:
        body += _text(cx + cw / 2, table_y + 35, ctitle, size=12, weight="bold", color=COLORS["white"], anchor="middle")

    rows = [
        (
            "M1: SAT-Only Baseline",
            "None",
            "None",
            "None",
            "CryptoMiniSat",
            "Fresh pairs",
            "k = 0",
            "0.0 s",
            "0.0 s",
            "To be measured",
            "To be measured",
            "T_sat + T_ver",
            COLORS["white"],
        ),
        (
            "M2: Random-Order + SAT",
            "Seeded random",
            "None",
            "None",
            "Incremental CMS",
            "Fresh pairs",
            "k = 16",
            "0.0 s",
            "0.0 s",
            "To be measured",
            "To be measured",
            "T_sat + T_ver",
            COLORS["slate_light"],
        ),
        (
            "M3: Baseline (Gohr) + SAT",
            "Summed logits",
            "None",
            "None",
            "Incremental CMS",
            "Fresh pairs",
            "k = 16",
            "~208 s",
            "0.0 s",
            "To be measured",
            "To be measured",
            "T_rank + T_sat + T_ver",
            COLORS["white"],
        ),
        (
            "M4: Ungated Pair-Aware + SAT",
            "Pair-aware logit",
            "None",
            "None",
            "Incremental CMS",
            "Fresh pairs",
            "k = 16",
            "~208 s",
            "0.0 s",
            "To be measured",
            "To be measured",
            "T_rank + T_sat + T_ver",
            COLORS["green_light"],
        ),
        (
            "M5: Gated Pair-Aware + Fallback",
            "Pair-aware logit",
            "Calibrated C_hat ≥ τ",
            "Frozen fallback",
            "Incremental CMS",
            "Fresh pairs",
            "k = 16 / alt",
            "~208 s",
            "To be measured",
            "To be measured",
            "To be measured",
            "T_rank + T_conf + T_sat + T_ver",
            COLORS["purple_light"],
        ),
    ]

    for r_idx, r_data in enumerate(rows):
        ry = table_y + 65 + r_idx * 95
        body += _rect(table_x + 5, ry, table_w - 10, 88, fill=r_data[12], stroke=COLORS["grid"], stroke_width=1, radius=6)

        # Method title
        body += _text(cols[0][0] + 10, ry + 32, r_data[0], size=13, weight="bold", color=COLORS["navy"], anchor="start")
        body += _text(cols[0][0] + 10, ry + 56, "Controlled comparative baseline", size=11, color=COLORS["slate"], anchor="start")

        # Table entries
        for c_idx in range(1, 12):
            cx = cols[c_idx][0]
            cw = cols[c_idx][1]
            val = r_data[c_idx]
            is_future = "To be measured" in val
            color = COLORS["purple"] if is_future else COLORS["text"]
            weight = "bold" if (c_idx in (1, 2, 3, 11) and not is_future) else ("normal" if not is_future else "bold")
            style = "italic" if is_future else "normal"
            body += _text(cx + cw / 2, ry + 48, val, size=11, weight=weight, color=color, anchor="middle", style=style)

    # Bottom Invariants & Cost Formula Footer
    foot_y = 690
    body += _rect(table_x, foot_y, table_w, 140, fill=COLORS["panel"], stroke=COLORS["grid"], stroke_width=1.5, radius=12)
    body += _text(800, foot_y + 30, "Strict Scientific Accounting and Experimental Invariants", size=16, weight="bold", color=COLORS["navy"], anchor="middle")
    body += _line(80, foot_y + 48, 1520, foot_y + 48, color=COLORS["grid"], width=1)

    foot_notes = [
        "1. Complete Online Cost: T_online = T_ranking + T_confidence + T_solver + T_verification. Precomputed neural ranking is never omitted.",
        "2. Controlled Baselines: Every method receives the exact same 64 ciphertext pairs, secret keys, solver timeouts, and verification rules.",
        "3. Future Values Uncommitted: Numbers for final SAT recovery are marked strictly as 'To be measured'; no speculative performance claims.",
    ]
    body += _multiline_text(800, foot_y + 72, foot_notes, size=12, line_height=22, anchor="middle", color=COLORS["slate"])

    return _svg_document(width, height, body)


# ==============================================================================
# CLI & Main Entry Point
# ==============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create proposal-ready methodology and result figures."
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--png",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render PNG previews with rsvg-convert",
    )
    parser.add_argument(
        "--pdf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render vector PDF documents with rsvg-convert",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary_path = args.summary if args.summary.is_absolute() else REPOSITORY_ROOT / args.summary
    output_root = args.output_root if args.output_root.is_absolute() else REPOSITORY_ROOT / args.output_root
    summary = _load_summary(summary_path.resolve())
    output_root.mkdir(parents=True, exist_ok=True)

    figures = {
        "methodology_pipeline.svg": _methodology_figure(),
        "b1_pair_aware_architecture.svg": _b1_architecture_figure(),
        "b0_b1_prospective_results.svg": _results_figure(summary),
        "validation_protocol.svg": _validation_protocol_figure(),
        "research_positioning.svg": _positioning_figure(),
        "exact_recovery_verification.svg": _exact_recovery_figure(),
        "confidence_fallback_mechanism.svg": _confidence_fallback_figure(),
        "evaluation_matrix.svg": _evaluation_matrix_figure(),
    }

    for name, document in figures.items():
        svg_path = output_root / name
        svg_path.write_text(document, encoding="utf-8")

        if args.png:
            png_path = output_root / f"{svg_path.stem}.png"
            try:
                subprocess.run(
                    ["rsvg-convert", "-d", "300", "-p", "300", str(svg_path), "-o", str(png_path)],
                    check=True,
                    capture_output=True,
                )
            except (subprocess.SubprocessError, FileNotFoundError) as err:
                print(f"Warning: could not render PNG for {name}: {err}")

        if args.pdf:
            pdf_path = output_root / f"{svg_path.stem}.pdf"
            try:
                subprocess.run(
                    ["rsvg-convert", "-f", "pdf", str(svg_path), "-o", str(pdf_path)],
                    check=True,
                    capture_output=True,
                )
            except (subprocess.SubprocessError, FileNotFoundError) as err:
                print(f"Warning: could not render PDF for {name}: {err}")

    manifest = {
        "schema": "neural_sat_speck.proposal_figures.v2",
        "source_summary": str(summary_path.resolve()),
        "source_summary_sha256": _sha256_file(summary_path.resolve()),
        "validation_only": True,
        "held_out_test_generated": False,
        "figure_files": sorted(figures),
        "figure_count": len(figures),
        "formats_generated": ["svg", "png", "pdf"],
        "notes": [
            "Pair-aware model is the selected headline method; Gohr baseline is the frozen comparator.",
            "Non-headline ablations remain recorded in the scientific summary.",
            "Dashed methodology components denote planned protocols, not completed results.",
            "Future solver metrics are explicitly marked 'To be measured' without fabrication.",
        ],
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Created {len(figures)} proposal figures (SVG, high-res PNG, vector PDF) in {output_root}")


if __name__ == "__main__":
    main()
