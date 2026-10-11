#!/usr/bin/env python3
"""Pipeline schematic: ranking, queue, exact completion and verification under one clock.

Double-column width (7.16 in) for a figure* at the top of a page. Same fonts
and palette as the publication figures.

Run as:
    python3 -I scripts/generate_pipeline_figure.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_final_publication_figures as g  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

STEM = "fig_pipeline"

# (x centre, title, body, edge colour)
BOXES = [
    (0.083, "Data", "64 chosen-plaintext\npairs under one key", g.C_GREY),
    (0.250, "Stage 1", "score all $2^{16}$ values\nof the last round key;\nkeep the top $P$", g.C_NAVY),
    (0.417, "Later stages", "decrypt one round per\nkept word; rank the\nnext key; keep top $C$", g.C_NAVY),
    (0.584, "Queue", "$P \\times C$ chains of\ntrailing round keys,\nbest summed rank first", g.C_NAVY),
    (0.751, "Exact completion", "SAT on 4 known blocks\nwith the chain fixed;\nUNSAT: next chain", g.C_ORANGE),
    (0.918, "Verification", "key must reproduce\n128 known blocks and\n2 fresh blocks", g.C_DARK),
]
W, H, Y = 0.136, 0.46, 0.60


def main() -> None:
    out_dir = ROOT / "figures"
    if any((out_dir / f"{STEM}.{ext}").exists() for ext in ("pdf", "png", "json")):
        raise SystemExit("Refusing to overwrite existing figure outputs")
    fig = g.plt.figure(figsize=(7.16, 1.85))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for x, title, body, colour in BOXES:
        ax.add_patch(FancyBboxPatch((x - W / 2, Y - H / 2), W, H, boxstyle="round,pad=0.004,rounding_size=0.012",
                                    linewidth=1.2, edgecolor=colour, facecolor="white", zorder=2))
        ax.text(x, Y + H / 2 - 0.075, title, ha="center", va="center", fontsize=7.6, weight="bold", color=colour, zorder=3)
        ax.text(x, Y - 0.045, body, ha="center", va="center", fontsize=6.6, linespacing=1.25, color=g.C_DARK, zorder=3)
    for (x0, *_), (x1, *_) in zip(BOXES, BOXES[1:]):
        ax.add_patch(FancyArrowPatch((x0 + W / 2 + 0.003, Y), (x1 - W / 2 - 0.003, Y), arrowstyle="-|>",
                                     mutation_scale=8, linewidth=1.0, color=g.C_DARK, zorder=1))
    # Next-candidate loop on the solver box.
    xs = BOXES[4][0]
    ax.add_patch(FancyArrowPatch((xs + 0.035, Y + H / 2 + 0.005), (xs - 0.035, Y + H / 2 + 0.005),
                                 connectionstyle="arc3,rad=0.55", arrowstyle="-|>", mutation_scale=7,
                                 linewidth=0.9, color=g.C_ORANGE, zorder=1))
    ax.text(xs, Y + H / 2 + 0.125, "wrong chain: reject, try next", ha="center", va="bottom", fontsize=6.3,
            color=g.C_ORANGE, style="italic")
    # Containment and recovery markers.
    ax.text(BOXES[3][0], Y + H / 2 + 0.035, "containment: true chain is queued", ha="center", va="bottom",
            fontsize=6.3, color=g.C_NAVY, style="italic")
    ax.text(BOXES[5][0], Y + H / 2 + 0.035, "recovery", ha="center", va="bottom", fontsize=6.3,
            color=g.C_DARK, style="italic")

    # One clock under everything except the data.
    x_start, x_prep_end, x_end = BOXES[1][0] - W / 2, BOXES[3][0] + W / 2, BOXES[5][0] + W / 2
    yb = 0.19
    ax.plot([x_start, x_end], [yb, yb], color=g.C_DARK, linewidth=0.9)
    for xv in (x_start, x_prep_end, x_end):
        ax.plot([xv, xv], [yb - 0.025, yb + 0.025], color=g.C_DARK, linewidth=0.9)
    ax.text((x_start + x_prep_end) / 2, yb + 0.045, "preparation $P_i$ (loading, ranking, queue)", ha="center",
            va="bottom", fontsize=6.6, color=g.C_NAVY)
    ax.text((x_prep_end + x_end) / 2, yb + 0.045,
            "$\\Sigma_{j \\leq k_i}(R_j + H_j)$ wrong chains $+\\;S_i$ correct solve", ha="center", va="bottom",
            fontsize=6.6, color=g.C_ORANGE)
    ax.text((x_start + x_end) / 2, yb - 0.05,
            "one 300 s clock per key:  $\\widehat{T}_i = P_i + \\Sigma_{j=1}^{k_i}(R_j + H_j) + S_i \\leq 300$ s",
            ha="center", va="top", fontsize=7.2, color=g.C_DARK)

    pdf, png = g.save_figure_dual(fig, out_dir, STEM)
    meta = {
        "script_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
        "command": "python3 -I scripts/generate_pipeline_figure.py",
    }
    (out_dir / f"{STEM}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(pdf, png)


if __name__ == "__main__":
    main()
