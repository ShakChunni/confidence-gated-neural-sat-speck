#!/usr/bin/env python3
"""Predicted versus observed recovery for the prospective queue-cost test (fig 9).

Inputs: the frozen predictions (docs/queue_cost_model_evidence_20261009/step5_predictions.json)
and the analysis JSON from scripts/analyze_prospective_queue_cost.py. For each of the
eight model-test arms: the frozen 90% interval (widened over solver speed x0.9-1.1),
the expected count at speed x1.0, and the observed count. Nothing is fitted here.

Run as:
    python3 -I scripts/generate_prospective_prediction_figure.py --analysis A.json --out-stem figures/fig9_prospective_predictions
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "docs/queue_cost_model_evidence_20261009/step5_predictions.json"
BLUE, ORANGE, GREY = "#2f5597", "#c55a11", "#888888"

# Arm label -> (prediction key, display label). Order = plotting order.
ARMS = [
    ("S-G", "speck32_64|cms_native_gauss|cap=None", "SPECK, matrices on"),
    ("S-X", "speck32_64|cms_native_no_gauss|cap=None", "SPECK, matrices off"),
    ("S-G-c", "speck32_64|cms_native_gauss|cap=1.5", "SPECK, on, 1.5 s cap"),
    ("S-X-c", "speck32_64|cms_native_no_gauss|cap=1.5", "SPECK, off, 1.5 s cap"),
    ("M-G", "simon32_64|cms_native_gauss|cap=None", "SIMON, matrices on"),
    ("M-X", "simon32_64|cms_native_no_gauss|cap=None", "SIMON, matrices off"),
    ("M-G-c", "simon32_64|cms_native_gauss|cap=10.0", "SIMON, on, 10 s cap"),
    ("M-X-c", "simon32_64|cms_native_no_gauss|cap=10.0", "SIMON, off, 10 s cap"),
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rows(analysis: dict, predictions: dict) -> list[dict]:
    out = []
    for label, key, display in ARMS:
        pred = predictions[key]
        lo = min(pred[s]["pi90"][0] for s in ("0.9", "1.0", "1.1"))
        hi = max(pred[s]["pi90"][1] for s in ("0.9", "1.0", "1.1"))
        frozen = analysis["frozen_interval_check"][label]
        if frozen["frozen_interval"] != [lo, hi]:
            raise ValueError(
                f"{label}: analysis interval differs from predictions file"
            )
        out.append(
            {
                "arm": label,
                "display": display,
                "expected": pred["1.0"]["expected"],
                "lo": lo,
                "hi": hi,
                "observed": frozen["recovered"],
                "inside": frozen["inside"],
            }
        )
    return out


def draw(data: list[dict]):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.48, 3.0), layout="constrained")
    ys = list(range(len(data)))[::-1]
    for y, r in zip(ys, data, strict=True):
        ax.plot(
            [r["lo"], r["hi"]],
            [y, y],
            color=GREY,
            linewidth=4,
            alpha=0.45,
            solid_capstyle="butt",
            zorder=1,
        )
        ax.plot(
            r["expected"],
            y,
            marker="|",
            color="#333333",
            markersize=9,
            markeredgewidth=1.4,
            zorder=2,
        )
        colour = BLUE if r["inside"] else ORANGE
        ax.plot(
            r["observed"],
            y,
            marker="o",
            color=colour,
            markersize=5.5,
            zorder=3,
            markerfacecolor=colour if r["inside"] else "white",
            markeredgewidth=1.2,
        )
    ax.set_yticks(ys)
    ax.set_yticklabels([r["display"] for r in data], fontsize=7.4)
    ax.axhline(3.5, color="#cccccc", linewidth=0.6)
    ax.set_xlim(-0.5, 32.5)
    ax.set_xticks([0, 8, 16, 24, 32])
    ax.set_xlabel("Keys recovered (of 32)")
    ax.grid(True, axis="x", color="#e6e6e6", linewidth=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [
        plt.Line2D(
            [], [], color=GREY, linewidth=4, alpha=0.45, label="Frozen 90% interval"
        ),
        plt.Line2D(
            [],
            [],
            color="#333333",
            marker="|",
            linestyle="",
            markersize=9,
            markeredgewidth=1.4,
            label="Predicted",
        ),
        plt.Line2D(
            [], [], color=BLUE, marker="o", linestyle="", label="Observed, inside"
        ),
        plt.Line2D(
            [],
            [],
            color=ORANGE,
            marker="o",
            markerfacecolor="white",
            linestyle="",
            label="Observed, outside",
        ),
    ]
    fig.legend(
        handles=handles, loc="outside upper center", ncol=2, fontsize=6.8, frameon=False
    )
    return fig


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument("--predictions", type=Path, default=PREDICTIONS)
    p.add_argument("--out-stem", type=Path, required=True)
    args = p.parse_args(argv)
    outputs = {
        ext: args.out_stem.with_suffix(f".{ext}")
        for ext in ("pdf", "png", "csv", "json")
    }
    if any(path.exists() for path in outputs.values()):
        raise SystemExit("Refusing to overwrite existing figure outputs")
    analysis = json.loads(args.analysis.read_text())
    predictions = json.loads(args.predictions.read_text())
    data = rows(analysis, predictions)
    fig = draw(data)
    pdf, png = io.BytesIO(), io.BytesIO()
    fig.savefig(pdf, format="pdf", metadata={"CreationDate": None})
    fig.savefig(png, format="png", dpi=600)
    outputs["pdf"].write_bytes(pdf.getvalue())
    outputs["png"].write_bytes(png.getvalue())
    header = "arm,display,expected,interval_lo,interval_hi,observed,inside\n"
    body = "".join(
        f"{r['arm']},{r['display']},{r['expected']},{r['lo']},{r['hi']},{r['observed']},{r['inside']}\n"
        for r in data
    )
    outputs["csv"].write_text(header + body)
    provenance = {
        "script_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "analysis": str(args.analysis),
        "analysis_sha256": sha256_bytes(args.analysis.read_bytes()),
        "predictions_sha256": sha256_bytes(args.predictions.read_bytes()),
        "command": f"python3 -I scripts/generate_prospective_prediction_figure.py --analysis {args.analysis} --out-stem {args.out_stem}",
    }
    outputs["json"].write_text(json.dumps(provenance, indent=2) + "\n")
    for path in outputs.values():
        Path(str(path) + ".sha256").write_text(
            f"{sha256_bytes(path.read_bytes())}  {path.name}\n"
        )
    print({r["arm"]: (r["observed"], [r["lo"], r["hi"]], r["inside"]) for r in data})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
