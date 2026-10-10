#!/usr/bin/env python3
"""Single-column version of the cumulative-recovery figure (paper Fig. 1).

The original fig2_cumulative_recovery_time is drawn 7.16 in wide, so at
\\columnwidth every label shrinks to about half size. This draws the same
curves from the same attack_cells.csv at 3.5 in wide, panels stacked, legends
inside the empty lower-right area. Data and step construction are reused from
generate_final_publication_figures.py; nothing is recomputed differently.

Run as:
    python3 -I scripts/generate_recovery_time_column_figure.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_final_publication_figures as g  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

CELLS = ROOT / "output/publication_figures_20261005_final/tables/attack_cells.csv"
STEM = "fig2_cumulative_recovery_time_col"


def success_times(cells, study, method):
    out = []
    for c in cells:
        if c["study"] == study and c["role"] == "autonomous_attack" and c["method"] == method:
            if c["verified_recovery"] not in ("True", "False"):
                raise ValueError(f"Non-boolean verified_recovery: {c['verified_recovery']!r}")
            if c["verified_recovery"] == "True":
                t = float(c["wall_seconds"])
                if t > 300.0:
                    raise ValueError("Late recovery must not be counted as verified")
                out.append(t)
    return out


def main() -> None:
    out_dir = ROOT / "figures"
    if any((out_dir / f"{STEM}.{ext}").exists() for ext in ("pdf", "png", "json")):
        raise SystemExit("Refusing to overwrite existing figure outputs")
    cells = g._read_csv(CELLS)
    panels = [
        ("(a) SPECK32/64, 7 rounds, N = 128", "speck_seven_round_prospective", g.SPECK_ARMS, 128),
        ("(b) SIMON32/64, 10 rounds, N = 64", "simon_confirmation_replacement", g.SIMON_ARMS, 64),
    ]
    fig, axes = g.plt.subplots(2, 1, figsize=(3.5, 4.75), sharex=True, layout="constrained")
    counts = {}
    for ax, (title, study, arms, n) in zip(axes, panels):
        g.format_axis(ax, grid_axis="both")
        ax.set_title(title, loc="left", fontsize=8.5, weight="bold", pad=4)
        ax.axvline(300.0, color=g.C_DARK, linestyle="--", linewidth=0.8, zorder=2)
        ax.set_xlim(0, 300)
        ax.set_ylim(-3, 103)
        ax.set_xticks(range(0, 301, 50))
        ax.set_yticks(range(0, 101, 25))
        ax.set_ylabel("Verified keys (% of N)", fontsize=8)
        ax.tick_params(labelsize=7.5)
        handles, zero = [], []
        for method, label, color in arms:
            ts = success_times(cells, study, method)
            counts[f"{study}|{method}"] = len(ts)
            if not ts:
                zero.append(label)
                continue
            xs, ys = g._step_xy(ts, n, 300.0)
            ax.step(xs, ys, where="post", color=color, linewidth=1.5, zorder=4)
            handles.append(Line2D([0], [0], color=color, linewidth=1.5, label=f"{label} ({len(ts)}/{n})"))
        if zero:
            ax.plot([0, 300], [0, 0], color=g.C_GREY, linewidth=1.2, linestyle=(0, (4, 2)), zorder=3)
            handles.append(
                Line2D([0], [0], color=g.C_GREY, linewidth=1.2, linestyle=(0, (4, 2)),
                       label=f"No recovery (0/{n}): {len(zero)} arms")
            )
        if study.startswith("speck"):
            # Empty lower-right area in (a); nothing is hidden.
            ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.985, 0.06), frameon=True,
                      framealpha=0.95, edgecolor="none", fontsize=6.8, handlelength=1.8,
                      borderpad=0.3, labelspacing=0.25)
        else:
            # In (b) the Bao-neural curve crosses the lower right, so the legend goes below.
            ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False,
                      fontsize=6.8, handlelength=1.8, borderpad=0.2, labelspacing=0.25, ncol=2,
                      columnspacing=1.0)
    axes[1].set_xlabel("Complete online time (s); dashed line = 300 s cutoff", fontsize=8)
    pdf, png = g.save_figure_dual(fig, out_dir, STEM)
    script = Path(__file__).resolve()
    meta = {
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "attack_cells": str(CELLS.relative_to(ROOT)),
        "attack_cells_sha256": hashlib.sha256(CELLS.read_bytes()).hexdigest(),
        "recoveries": counts,
        "command": "python3 -I scripts/generate_recovery_time_column_figure.py",
    }
    (out_dir / f"{STEM}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(pdf, png)
    print({k: v for k, v in counts.items() if v})


if __name__ == "__main__":
    main()
