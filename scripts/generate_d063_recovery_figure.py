#!/usr/bin/env python3
"""Cumulative verified recovery for the D-063 seven-/eight-round SPECK campaign.

Single column (3.5 in), panels stacked, same style and step construction as
the paper's Fig. 1 (generate_recovery_time_column_figure.py). Reads the
committed campaign ledger; a cell counts only if its outcome is "recovered"
(all 128 known blocks and 2 fresh blocks reproduced) within 300 s.

Run as:
    python3 -I scripts/generate_d063_recovery_figure.py
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

LEDGER = ROOT / "docs/paper/evidence/d063_campaign_20261011/ledger_k00-31.jsonl"
STEM = "fig_d063_cumulative_recovery_col"
PANELS = [
    ("(a) SPECK32/64, 7 rounds, N = 32", [
        ("R7-G", "Gohr net6 + table", g.C_ORANGE),
        ("R7-C", "Classical table", g.C_NAVY),
        ("R7-B1", "B1 + table", g.C_PURPLE),
    ]),
    ("(b) SPECK32/64, 8 rounds, N = 32", [
        ("R8-G", "Gohr net7 + tables", g.C_ORANGE),
        ("R8-C", "Classical tables", g.C_NAVY),
    ]),
]


def main() -> None:
    out_dir = ROOT / "figures"
    if any((out_dir / f"{STEM}.{ext}").exists() for ext in ("pdf", "png", "json")):
        raise SystemExit("Refusing to overwrite existing figure outputs")
    rows = [r for r in map(json.loads, LEDGER.open()) if r.get("type") != "header"]
    n = len({r["key_index"] for r in rows})
    fig, axes = g.plt.subplots(2, 1, figsize=(3.5, 4.4), sharex=True, layout="constrained")
    counts = {}
    for ax, (title, arms) in zip(axes, PANELS):
        g.format_axis(ax, grid_axis="both")
        ax.set_title(title, loc="left", fontsize=8.5, weight="bold", pad=4)
        ax.axvline(300.0, color=g.C_DARK, linestyle="--", linewidth=0.8, zorder=2)
        ax.set_xlim(0, 300)
        ax.set_ylim(-3, 103)
        ax.set_xticks(range(0, 301, 50))
        ax.set_yticks(range(0, 101, 25))
        ax.set_ylabel("Verified keys (% of N)", fontsize=8)
        ax.tick_params(labelsize=7.5)
        handles = []
        for arm, label, color in arms:
            ts = []
            for r in rows:
                if r["arm"] == arm and r["outcome"] == "recovered":
                    if r["total_s"] > 300.0:
                        raise ValueError("Late recovery must not be counted as verified")
                    ts.append(r["total_s"])
            counts[arm] = len(ts)
            if ts:
                xs, ys = g._step_xy(ts, n, 300.0)
                ax.step(xs, ys, where="post", color=color, linewidth=1.5, zorder=4)
                handles.append(Line2D([0], [0], color=color, linewidth=1.5, label=f"{label} ({len(ts)}/{n})"))
            else:
                ax.plot([0, 300], [0, 0], color=color, linewidth=1.5, linestyle=(0, (4, 2)), zorder=5)
                handles.append(Line2D([0], [0], color=color, linewidth=1.5, linestyle=(0, (4, 2)),
                                      label=f"{label} (0/{n})"))
        ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.985, 0.06), frameon=True,
                  framealpha=0.95, edgecolor="none", fontsize=6.8, handlelength=1.8,
                  borderpad=0.3, labelspacing=0.25)
    axes[1].set_xlabel("Complete online time (s); dashed line = 300 s cutoff", fontsize=8)
    pdf, png = g.save_figure_dual(fig, out_dir, STEM)
    meta = {
        "script_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
        "ledger": str(LEDGER.relative_to(ROOT)),
        "ledger_sha256": hashlib.sha256(LEDGER.read_bytes()).hexdigest(),
        "recoveries": counts,
        "command": "python3 -I scripts/generate_d063_recovery_figure.py",
    }
    (out_dir / f"{STEM}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(pdf, png, counts)


if __name__ == "__main__":
    main()
