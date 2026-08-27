"""Create a dependency-free Phase 6 risk--coverage figure."""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from typing import Any

from neural_sat_speck.confidence import CONFIDENCE_FEATURE_NAMES

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPOSITORY_ROOT / "results" / "phase6_confidence_validation_r6.json"
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "results"
    / "figures"
    / "phase6"
    / "confidence_risk_coverage.svg"
)
COLORS = ("#2563eb", "#ea580c", "#15803d", "#9333ea")
GRID = "#cbd5e1"
AXIS = "#334155"
TEXT = "#0f172a"
MUTED = "#475569"


def _repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _load_result(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"Could not read confidence result {path}: {error}") from error
    if not isinstance(document, dict):
        raise SystemExit("confidence result must contain a JSON object")
    if document.get("schema") != "neural_sat_speck.phase6_confidence_result.v1":
        raise SystemExit("confidence result schema is not recognised")
    protocol = document.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("split_role") != "validation":
        raise SystemExit("the figure requires a validation-only confidence result")
    curves = document.get("risk_coverage")
    if not isinstance(curves, dict):
        raise SystemExit("confidence result does not contain risk-coverage curves")
    for feature_name in CONFIDENCE_FEATURE_NAMES:
        points = curves.get(feature_name)
        if not isinstance(points, list) or not points:
            raise SystemExit(f"curve is missing for {feature_name}")
    return document


def _text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 16,
    color: str = TEXT,
    anchor: str = "start",
    weight: str = "normal",
) -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, sans-serif" '
        f'font-size="{size}px" fill="{color}" text-anchor="{anchor}" '
        f'font-weight="{weight}">{escape(value)}</text>'
    )


def _line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str = AXIS,
    width: float = 1.0,
    dash: str | None = None,
) -> str:
    dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" '
        f'y2="{y2:.2f}" stroke="{color}" stroke-width="{width:.2f}"'
        f"{dash_attribute}/>"
    )


def _circle(x: float, y: float, radius: float, *, fill: str) -> str:
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{fill}"/>'


def _polyline(points: list[tuple[float, float]], *, color: str) -> str:
    point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (
        f'<polyline points="{point_text}" fill="none" stroke="{color}" '
        'stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
    )


def _figure(document: dict[str, Any]) -> str:
    width, height = 1200, 850
    margin_x, margin_y = 90, 110
    gap_x, gap_y = 70, 90
    panel_width = (width - 2 * margin_x - gap_x) / 2
    panel_height = (height - 2 * margin_y - gap_y) / 2
    body = '<rect width="100%" height="100%" fill="white"/>'
    body += _text(
        width / 2,
        42,
        "Phase 6: validation risk--coverage pilot",
        size=27,
        anchor="middle",
        weight="bold",
    )
    body += _text(
        width / 2,
        69,
        "Top-16 target; lower risk is better; no threshold fitted",
        size=15,
        color=MUTED,
        anchor="middle",
    )

    curves = document["risk_coverage"]
    for feature_index, feature_name in enumerate(CONFIDENCE_FEATURE_NAMES):
        column = feature_index % 2
        row = feature_index // 2
        left = margin_x + column * (panel_width + gap_x)
        top = margin_y + row * (panel_height + gap_y)
        plot_left, plot_top = left + 62, top + 35
        plot_right, plot_bottom = left + panel_width - 18, top + panel_height - 48
        plot_width = plot_right - plot_left
        plot_height = plot_bottom - plot_top
        points_value = curves[feature_name]
        direction = str(points_value[0]["direction"])
        title = feature_name.replace("_", " ")
        body += _text(left, top + 10, title, size=16, weight="bold")
        body += _text(
            left + panel_width,
            top + 10,
            "higher" if direction == "higher" else "lower",
            size=12,
            color=MUTED,
            anchor="end",
        )
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            x = plot_left + tick * plot_width
            y = plot_bottom - tick * plot_height
            body += _line(x, plot_top, x, plot_bottom, color=GRID, dash="3 5")
            body += _line(plot_left, y, plot_right, y, color=GRID, dash="3 5")
            body += _text(x, plot_bottom + 21, f"{tick:.2g}", size=11, anchor="middle")
            body += _text(plot_left - 10, y + 4, f"{tick:.2g}", size=11, anchor="end")
        body += _line(plot_left, plot_top, plot_left, plot_bottom)
        body += _line(plot_left, plot_bottom, plot_right, plot_bottom)
        body += _text(
            plot_left + plot_width / 2,
            top + panel_height - 8,
            "coverage",
            size=12,
            color=MUTED,
            anchor="middle",
        )
        body += _text(
            left + 15,
            plot_top + plot_height / 2,
            "risk",
            size=12,
            color=MUTED,
            anchor="middle",
        )
        curve_points: list[tuple[float, float]] = []
        for point in points_value:
            coverage = float(point["coverage"])
            risk = float(point["risk"])
            curve_points.append(
                (
                    plot_left + coverage * plot_width,
                    plot_bottom - risk * plot_height,
                )
            )
        color = COLORS[feature_index]
        body += _polyline(curve_points, color=color)
        for x, y in curve_points:
            body += _circle(x, y, 4, fill=color)

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">{body}</svg>\n'
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a local validation-only Phase 6 risk-coverage result."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = _repository_path(args.input).resolve()
    output_path = _repository_path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_figure(_load_result(input_path)), encoding="utf-8")
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
