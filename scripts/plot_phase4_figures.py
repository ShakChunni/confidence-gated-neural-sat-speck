"""Create dependency-free Phase 4 figures from local ranking results."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from html import escape
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = REPOSITORY_ROOT / "results"
DEFAULT_OUTPUT_ROOT = DEFAULT_RESULT_ROOT / "figures" / "phase4"

COLORS = {
    "validation": "#2563eb",
    "test": "#ea580c",
    "all": "#15803d",
    "grid": "#cbd5e1",
    "axis": "#334155",
    "text": "#0f172a",
    "muted": "#475569",
    "background": "#ffffff",
}


def _load_results(result_root: Path) -> list[dict[str, Any]]:
    """Load and validate complete-domain Phase 4 result files."""

    records: list[dict[str, Any]] = []
    for path in sorted(result_root.glob("phase4_ranking_r6_*.json")):
        if "_b" in path.stem:
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("phase") != 4:
            continue
        if document.get("candidate_domain", {}).get("count") != 65536:
            continue
        ranking = document["ranking"]
        candidate_order = ranking["candidate_order"]
        if len(candidate_order) != 65536 or len(set(candidate_order)) != 65536:
            raise ValueError(f"ranking order is incomplete in {path}")
        evaluation = document["evaluation"]
        records.append(
            {
                "path": str(path),
                "split": str(document["instance"]["split_role"]),
                "index": int(document["instance"]["index"]),
                "rank": int(evaluation["true_rank"]),
                "seconds": float(ranking["timing_seconds"]["total"]),
            }
        )
    if not records:
        raise ValueError(f"no complete Phase 4 result files found in {result_root}")
    split_order = {"validation": 0, "test": 1}
    records.sort(key=lambda record: (split_order.get(record["split"], 2), record["index"]))
    return records


def _svg_document(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
        f'<rect width="100%" height="100%" fill="{COLORS["background"]}"/>'
        f"{body}</svg>\n"
    )


def _text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 16,
    color: str = COLORS["text"],
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
    color: str = COLORS["axis"],
    width: float = 1.0,
    dash: str | None = None,
) -> str:
    dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" '
        f'y2="{y2:.2f}" stroke="{color}" stroke-width="{width:.2f}"'
        f'{dash_attribute}/>'
    )


def _rect(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str,
    stroke: str = "none",
    radius: float = 0.0,
) -> str:
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" '
        f'height="{height:.2f}" rx="{radius:.2f}" fill="{fill}" '
        f'stroke="{stroke}"/>'
    )


def _circle(x: float, y: float, radius: float, *, fill: str) -> str:
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{fill}"/>'


def _polyline(points: Sequence[tuple[float, float]], *, color: str) -> str:
    point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (
        f'<polyline points="{point_text}" fill="none" stroke="{color}" '
        'stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
    )


def _chart_heading(title: str, subtitle: str) -> str:
    return _text(70, 45, title, size=26, weight="bold") + _text(
        70,
        70,
        subtitle,
        size=14,
        color=COLORS["muted"],
    )


def _rank_figure(records: list[dict[str, Any]]) -> str:
    width, height = 1100, 650
    left, right, top, bottom = 105, 35, 105, 95
    plot_width, plot_height = width - left - right, height - top - bottom
    maximum = max(256, max(record["rank"] for record in records) * 2)

    def y_for(rank: float) -> float:
        return top + math.log10(rank) / math.log10(maximum) * plot_height

    body = _chart_heading(
        "Phase 4: correct final-round subkey rank",
        "All 65,536 candidates; lower rank is better; 32 pairs per instance",
    )
    body += _text(55, top + plot_height / 2, "rank (log scale)", size=15, anchor="middle")
    body += _text(
        left + plot_width / 2,
        height - 25,
        "independent attack instance",
        size=15,
        anchor="middle",
    )

    ticks = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    for tick in ticks:
        if tick > maximum:
            continue
        y = y_for(tick)
        body += _line(left, y, left + plot_width, y, color=COLORS["grid"], dash="4 5")
        body += _text(left - 12, y + 5, str(tick), size=13, anchor="end")
    body += _line(left, top, left, top + plot_height)
    body += _line(left, top + plot_height, left + plot_width, top + plot_height)

    bar_width = min(56, plot_width / max(1, len(records)) * 0.56)
    slot_width = plot_width / len(records)
    for position, record in enumerate(records):
        x = left + slot_width * (position + 0.5)
        rank = record["rank"]
        y = y_for(rank)
        color = COLORS[record["split"]]
        body += _rect(x - bar_width / 2, top, bar_width, y - top, fill=color, radius=4)
        label_y = max(top + 18, y - 8)
        body += _text(x, label_y, str(rank), size=14, anchor="middle", weight="bold")
        body += _text(
            x,
            top + plot_height + 24,
            f"{record['split'][:3]}-{record['index']}",
            size=13,
            anchor="middle",
        )

    for rank, label in ((16, "top-16"), (32, "top-32"), (128, "top-128")):
        if rank <= maximum:
            body += _line(
                left,
                y_for(rank),
                left + plot_width,
                y_for(rank),
                color="#64748b",
                dash="8 5",
            )
            body += _text(left + plot_width - 4, y_for(rank) - 5, label, size=12, anchor="end")

    legend_x = width - 190
    body += _rect(legend_x, 38, 14, 14, fill=COLORS["validation"], radius=2)
    body += _text(legend_x + 21, 50, "validation", size=13)
    body += _rect(legend_x + 92, 38, 14, 14, fill=COLORS["test"], radius=2)
    body += _text(legend_x + 113, 50, "test", size=13)
    return _svg_document(width, height, body)


def _recall_figure(records: list[dict[str, Any]]) -> str:
    width, height = 1100, 650
    left, right, top, bottom = 105, 45, 105, 95
    plot_width, plot_height = width - left - right, height - top - bottom
    top_k_values = (1, 16, 32, 64, 128)
    body = _chart_heading(
        "Phase 4: correct-subkey top-k recall",
        "Measured after ranking; validation and held-out test instances shown separately",
    )
    body += _text(55, top + plot_height / 2, "recall (%)", size=15, anchor="middle")
    body += _text(
        left + plot_width / 2,
        height - 25,
        "candidate budget k",
        size=15,
        anchor="middle",
    )

    def x_for(position: int) -> float:
        return left + position / (len(top_k_values) - 1) * plot_width

    def y_for(percent: float) -> float:
        return top + (100.0 - percent) / 100.0 * plot_height

    for tick in (0, 25, 50, 75, 100):
        y = y_for(tick)
        body += _line(left, y, left + plot_width, y, color=COLORS["grid"], dash="4 5")
        body += _text(left - 12, y + 5, str(tick), size=13, anchor="end")
    for position, value in enumerate(top_k_values):
        x = x_for(position)
        body += _line(x, top, x, top + plot_height, color="#e2e8f0", dash="2 6")
        body += _text(x, top + plot_height + 25, str(value), size=13, anchor="middle")
    body += _line(left, top, left, top + plot_height)
    body += _line(left, top + plot_height, left + plot_width, top + plot_height)

    for split in ("validation", "test", "all"):
        subset = records if split == "all" else [r for r in records if r["split"] == split]
        if not subset:
            continue
        points: list[tuple[float, float]] = []
        for position, k in enumerate(top_k_values):
            recall = 100.0 * sum(record["rank"] <= k for record in subset) / len(subset)
            point = (x_for(position), y_for(recall))
            points.append(point)
        body += _polyline(points, color=COLORS[split])
        for position, (x, y) in enumerate(points):
            body += _circle(x, y, 6, fill=COLORS[split])
            recall = 100.0 * sum(
                record["rank"] <= top_k_values[position] for record in subset
            ) / len(subset)
            body += _text(x, y - 12, f"{recall:.0f}%", size=12, anchor="middle", weight="bold")

    legend_x = width - 285
    for offset, split in enumerate(("validation", "test", "all")):
        x = legend_x + offset * 88
        body += _line(x, 46, x + 18, 46, color=COLORS[split], width=3)
        body += _circle(x + 9, 46, 4, fill=COLORS[split])
        body += _text(x + 24, 50, split, size=13)
    return _svg_document(width, height, body)


def _runtime_figure(records: list[dict[str, Any]]) -> str:
    width, height = 1100, 650
    left, right, top, bottom = 105, 45, 105, 95
    plot_width, plot_height = width - left - right, height - top - bottom
    maximum = max(160, math.ceil(max(record["seconds"] for record in records) / 20) * 20)
    body = _chart_heading(
        "Phase 4: full-domain ranking runtime",
        "CPU wall time for 65,536 candidates and 32 pairs; batch size 256",
    )
    body += _text(55, top + plot_height / 2, "seconds", size=15, anchor="middle")
    body += _text(
        left + plot_width / 2,
        height - 25,
        "independent attack instance",
        size=15,
        anchor="middle",
    )

    def y_for(seconds: float) -> float:
        return top + (maximum - seconds) / maximum * plot_height

    for tick in range(0, maximum + 1, 20):
        y = y_for(tick)
        body += _line(left, y, left + plot_width, y, color=COLORS["grid"], dash="4 5")
        body += _text(left - 12, y + 5, str(tick), size=13, anchor="end")
    body += _line(left, top, left, top + plot_height)
    body += _line(left, top + plot_height, left + plot_width, top + plot_height)

    bar_width = min(56, plot_width / max(1, len(records)) * 0.56)
    slot_width = plot_width / len(records)
    for position, record in enumerate(records):
        x = left + slot_width * (position + 0.5)
        y = y_for(record["seconds"])
        body += _rect(
            x - bar_width / 2,
            y,
            bar_width,
            top + plot_height - y,
            fill=COLORS[record["split"]],
            radius=4,
        )
        body += _text(
            x,
            y - 8,
            f"{record['seconds']:.1f}",
            size=13,
            anchor="middle",
            weight="bold",
        )
        body += _text(
            x,
            top + plot_height + 24,
            f"{record['split'][:3]}-{record['index']}",
            size=13,
            anchor="middle",
        )

    legend_x = width - 190
    body += _rect(legend_x, 38, 14, 14, fill=COLORS["validation"], radius=2)
    body += _text(legend_x + 21, 50, "validation", size=13)
    body += _rect(legend_x + 92, 38, 14, 14, fill=COLORS["test"], radius=2)
    body += _text(legend_x + 113, 50, "test", size=13)
    return _svg_document(width, height, body)


def _workflow_figure() -> str:
    width, height = 1400, 600
    body = _chart_heading(
        "Phase 4 workflow: final-round subkey ranking",
        "Ranking-only baseline; CryptoMiniSat and confidence gating are intentionally outside this phase",
    )
    boxes = [
        (
            60,
            220,
            220,
            130,
            "Related pairs",
            "six-round attack\n32 pairs / instance",
            "#dbeafe",
        ),
        (
            350,
            220,
            220,
            130,
            "Candidate key",
            "each value\n0x0000 ... 0xffff",
            "#ffedd5",
        ),
        (
            640,
            220,
            220,
            130,
            "Inverse one round",
            "partial decrypt both\nciphertexts",
            "#fef3c7",
        ),
        (
            930,
            220,
            220,
            130,
            "Neural scoring",
            "MSB-first features\nsummed raw logits",
            "#dcfce7",
        ),
        (
            1220,
            220,
            120,
            130,
            "Rank",
            "true-key\nposition",
            "#ede9fe",
        ),
    ]
    for x, y, box_width, box_height, title, subtitle, fill in boxes:
        body += _rect(x, y, box_width, box_height, fill=fill, stroke="#64748b", radius=12)
        body += _text(x + box_width / 2, y + 42, title, size=18, anchor="middle", weight="bold")
        for line_index, line in enumerate(subtitle.split("\n")):
            body += _text(
                x + box_width / 2,
                y + 78 + line_index * 23,
                line,
                size=14,
                anchor="middle",
                color=COLORS["muted"],
            )
    for left, right in ((280, 350), (570, 640), (860, 930), (1150, 1220)):
        body += _line(left, 285, right, 285, color="#475569", width=3)
        body += f'<polygon points="{right - 12},277 {right},285 {right - 12},293" fill="#475569"/>'

    body += _rect(350, 425, 800, 70, fill="#f8fafc", stroke="#94a3b8", radius=10)
    body += _text(
        750,
        455,
        "Post hoc evaluation only: compare the ranked position with the generated true final-round subkey",
        size=15,
        anchor="middle",
    )
    body += _line(1040, 350, 1040, 425, color="#64748b", width=2, dash="6 5")
    body += '<polygon points="1032,414 1040,425 1048,414" fill="#64748b"/>'
    return _svg_document(width, height, body)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create local SVG figures from Phase 4 ranking JSON results."
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help="Directory containing phase4_ranking_r6_*.json files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory that will receive SVG figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result_root = args.result_root
    output_root = args.output_root
    records = _load_results(result_root)
    output_root.mkdir(parents=True, exist_ok=True)
    figures = {
        "phase4_rank_by_instance.svg": _rank_figure(records),
        "phase4_top_k_recall.svg": _recall_figure(records),
        "phase4_runtime.svg": _runtime_figure(records),
        "phase4_workflow.svg": _workflow_figure(),
    }
    for name, document in figures.items():
        (output_root / name).write_text(document, encoding="utf-8")
    manifest = {
        "source_results": [record["path"] for record in records],
        "figure_files": sorted(figures),
        "record_count": len(records),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Created {len(figures)} SVG figures in {output_root}")


if __name__ == "__main__":
    main()
