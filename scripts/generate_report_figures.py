#!/usr/bin/env python3
"""Redraw the six paper figures for one-column pages (report and LNCS).

The paper figures were drawn for a 3.5 in IEEE column with Times fonts and
stacked panels. Enlarged to a one-column A4 report or an LNCS page they waste a
page each and their fonts no longer match the text. This script redraws the
same figures at their final printed width, so LaTeX never rescales them, in
Computer Modern to match the Latin Modern text of both formats, with panels
side by side.

Nothing is recomputed differently. Every figure reads the same committed or
hash-pinned inputs as its paper version, and the counts drawn are checked
against the published counts before anything is saved.

Profiles (printed width; include each PDF at natural size, no width option):
    report  6.0 in full width, one-column A4 report (text about 9 pt)
    lncs    4.75 in full width, LNCS 12.2 cm text block (text about 8 pt)

Run as:
    python scripts/generate_report_figures.py --profile report
    python scripts/generate_report_figures.py --profile lncs
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_affordable_depth_figure as afd  # noqa: E402
import generate_final_publication_figures as g  # noqa: E402
import generate_recovery_time_column_figure as col  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator  # noqa: E402

plt = g.plt

PROFILES = {
    "report": {"width": 6.0, "single": 4.6, "base": 9.0, "small": 8.0, "tiny": 7.5},
    "lncs": {"width": 4.75, "single": 4.1, "base": 8.0, "small": 7.2, "tiny": 6.8},
}

# Inputs. Untracked inputs are pinned to the hash recorded by the paper figure.
# Tracked byte-identical copy of col.CELLS (that path is git-ignored); the pin check below confirms it.
CELLS = ROOT / "output/writing_handoff_20261006/six_page_writer_bundle_v3/tables/attack_cells.csv"
CELLS_RECEIPT = ROOT / "figures/fig2_cumulative_recovery_time_col.json"
LEDGER = ROOT / "docs/paper/evidence/d063_campaign_20261011/ledger_k00-31.jsonl"
LEDGER_RECEIPT = ROOT / "figures/fig_d063_cumulative_recovery_col.json"
DEPTH_CSV = ROOT / "figures/fig7_backend_ranking_depth_n32.csv"
AFFORD_CSV = ROOT / "figures/fig8_affordable_depth.csv"
PRED_CSV = ROOT / "figures/fig9_prospective_predictions.csv"

# Published counts; a mismatch stops the run before any file is written.
EXPECTED = {
    "fig2": {"speck_seven_round_prospective|hierarchical_neural_classical": 111,
             "speck_seven_round_prospective|hierarchical_neural_screened": 110,
             "simon_confirmation_replacement|fully_classical_8parent": 53,
             "simon_confirmation_replacement|neural_classical_hybrid_8parent": 49,
             "simon_confirmation_replacement|fully_neural_1parent": 26},
    "d063": {"R7-G": 32, "R7-C": 30, "R7-B1": 19, "R8-G": 26, "R8-C": 0},
    "depth": {("speck32_64", "cms_native_gauss", 1): 4, ("speck32_64", "cms_native_gauss", 2): 18,
              ("speck32_64", "cms_native_no_gauss", 1): 2, ("speck32_64", "cms_native_no_gauss", 2): 19,
              ("simon32_64", "cms_native_gauss", 1): 23, ("simon32_64", "cms_native_gauss", 2): 27,
              ("simon32_64", "cms_native_no_gauss", 1): 3, ("simon32_64", "cms_native_no_gauss", 2): 22},
    "pred_inside": 8,
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_pinned(path: Path, receipt: Path, field: str) -> str:
    digest = sha(path)
    want = json.loads(receipt.read_text())[field]
    if digest != want:
        raise ValueError(f"{path.name}: SHA-256 {digest} differs from {receipt.name} pin {want}")
    return digest


def check_sidecar(path: Path) -> str:
    digest = sha(path)
    side = path.with_name(path.name + ".sha256").read_text().split()
    if side != [digest, path.name]:
        raise ValueError(f"{path.name}: SHA-256 sidecar mismatch")
    return digest


def set_style(p: dict) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "font.size": p["base"],
        "axes.titlesize": p["base"],
        "axes.labelsize": p["base"],
        "xtick.labelsize": p["small"],
        "ytick.labelsize": p["small"],
        "legend.fontsize": p["tiny"],
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "lines.linewidth": 1.3,
        "pdf.fonttype": 42,
        "savefig.pad_inches": 0.02,
    })


def tidy(ax, grid="both") -> None:
    g.format_axis(ax, grid_axis=grid)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)


# --------------------------------------------------------------------------- pipeline

def fig_pipeline(p: dict):
    W = p["width"]
    fs_t, fs_b, fs_n = p["small"], p["tiny"], p["tiny"]
    bh = 0.62 if W > 5 else 0.60
    gap_row = 0.50
    y_bot = 0.64
    y2 = y_bot + bh / 2
    y1 = y2 + bh + gap_row
    H = y1 + bh / 2 + 0.36
    m = 0.04
    ncol = 4
    unit = (W - 2 * m) / (ncol - 0.22)
    bw = 0.78 * unit
    xs = [m + bw / 2 + i * unit for i in range(ncol)]

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    def box(x, y, title, body, colour):
        ax.add_patch(FancyBboxPatch((x - bw / 2, y - bh / 2), bw, bh,
                                    boxstyle="round,pad=0.0,rounding_size=0.06",
                                    linewidth=1.0, edgecolor=colour, facecolor="white", zorder=2))
        ax.text(x, y + bh / 2 - 0.11, title, ha="center", va="center", fontsize=fs_t,
                family="cmb10", color=colour, zorder=3)
        ax.text(x, y - 0.07, body, ha="center", va="center", fontsize=fs_b, linespacing=1.2,
                color=g.C_DARK, zorder=3)

    def arrow(a, b, colour=g.C_DARK, **kw):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, linewidth=0.9,
                                     color=colour, zorder=1, shrinkA=0, shrinkB=0, **kw))

    row1 = [
        ("Data", "64 chosen-\nplaintext pairs\nunder one key", g.C_GREY),
        ("Stage 1", "score all $2^{16}$\nlast-key words;\nkeep the top $P$", g.C_NAVY),
        ("Later stages", "decrypt a round;\nrank the next word;\nkeep top $C$ each", g.C_NAVY),
        ("Queue", "$P \\times C$ chains,\nbest summed\nrank first", g.C_NAVY),
    ]
    for x, (t, b, c) in zip(xs, row1, strict=True):
        box(x, y1, t, b, c)
    for x0, x1 in zip(xs, xs[1:]):
        arrow((x0 + bw / 2, y1), (x1 - bw / 2, y1))

    box(xs[1], y2, "Exact completion", "SAT with the\nchain fixed;\nUNSAT: next chain", g.C_ORANGE)
    box(xs[2], y2, "Verification", "reproduce 128\nknown blocks and\n2 fresh blocks", g.C_DARK)
    arrow((xs[1] + bw / 2, y2), (xs[2] - bw / 2, y2))
    arrow((xs[2] + bw / 2, y2), (xs[3] - bw / 2 + 0.02, y2))
    ax.text(xs[3] - bw / 2 + 0.06, y2, "verified\nrecovery", ha="left", va="center", fontsize=fs_t,
            family="cmb10", color=g.C_DARK, linespacing=1.15)

    # Queue -> completion, routed through the gap between rows.
    y_mid = y2 + bh / 2 + gap_row / 2
    ax.plot([xs[3], xs[3], xs[1]], [y1 - bh / 2, y_mid, y_mid], color=g.C_DARK, linewidth=0.9, zorder=1)
    arrow((xs[1], y_mid), (xs[1], y2 + bh / 2))
    ax.text((xs[1] + xs[3]) / 2, y_mid + 0.035, "containment: the true chain is queued", ha="center",
            va="bottom", fontsize=fs_n, family="cmti10", color=g.C_NAVY)

    # Reject-and-retry loop on the solver box, label in the empty first column.
    xl = xs[1] - bw / 2
    ax.add_patch(FancyArrowPatch((xl, y2 - 0.16), (xl, y2 + 0.16), connectionstyle="arc3,rad=-0.9",
                                 arrowstyle="-|>", mutation_scale=7, linewidth=0.9, color=g.C_ORANGE,
                                 zorder=1, shrinkA=0, shrinkB=0))
    ax.text(xl - 0.17, y2, "wrong chain:\nreject and\ntry the next", ha="right", va="center", fontsize=fs_n,
            family="cmti10", color=g.C_ORANGE, linespacing=1.15)

    # Cost brackets.
    def bracket(xa, xb, y, up, text, colour):
        d = 0.05 if up else -0.05
        ax.plot([xa, xa, xb, xb], [y, y + d, y + d, y], color=colour, linewidth=0.8)
        ax.text((xa + xb) / 2, y + d + (0.03 if up else -0.03), text, ha="center",
                va="bottom" if up else "top", fontsize=fs_n, color=colour)

    bracket(xs[1] - bw / 2, xs[3] + bw / 2, y1 + bh / 2 + 0.05, True,
            "preparation $P_i$: loading, ranking, queue", g.C_NAVY)
    bracket(xs[1] - bw / 2, xs[2] + bw / 2, y2 - bh / 2 - 0.05, False,
            "$\\Sigma_{j \\leq k_i}(R_j + H_j)$ wrong chains $+\\ S_i$ correct solve", g.C_ORANGE)
    ax.text(W / 2, 0.05, "one 300 s clock per key:   $\\widehat{T}_i = P_i + \\Sigma_{j=1}^{k_i}(R_j + H_j)"
            " + S_i \\leq 300$ s", ha="center", va="bottom", fontsize=fs_t, color=g.C_DARK)
    return fig, {}


# --------------------------------------------------------------------------- recovery curves

def _curve_panel(ax, p, title, n, series, zero_label=None):
    tidy(ax)
    ax.set_title(title, loc="left", pad=4)
    ax.axvline(300.0, color=g.C_DARK, linestyle="--", linewidth=0.7, zorder=2)
    ax.set_xlim(0, 300)
    ax.set_ylim(-3, 103)
    ax.set_xticks(range(0, 301, 50))
    ax.set_yticks(range(0, 101, 25))
    handles = []
    for label, colour, ts in series:
        if ts:
            xs, ys = g._step_xy(ts, n, 300.0)
            ax.step(xs, ys, where="post", color=colour, linewidth=1.4, zorder=4)
            handles.append(Line2D([0], [0], color=colour, linewidth=1.4, label=f"{label} ({len(ts)}/{n})"))
        else:
            ax.plot([0, 300], [0, 0], color=colour, linewidth=1.4, linestyle=(0, (4, 2)), zorder=5)
            handles.append(Line2D([0], [0], color=colour, linewidth=1.4, linestyle=(0, (4, 2)),
                                  label=f"{label} (0/{n})"))
    if zero_label:
        ax.plot([0, 300], [0, 0], color=g.C_GREY, linewidth=1.1, linestyle=(0, (4, 2)), zorder=3)
        handles.append(Line2D([0], [0], color=g.C_GREY, linewidth=1.1, linestyle=(0, (4, 2)),
                              label=zero_label))
    ax.set_xlabel("Complete online time (s)")
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.24), frameon=False,
              handlelength=1.8, borderaxespad=0.0, labelspacing=0.3)


def fig_recovery_main(p: dict):
    cells = g._read_csv(CELLS)
    panels = [
        ("(a) SPECK32/64, 7 rounds, $N = 128$", "speck_seven_round_prospective", g.SPECK_ARMS, 128),
        ("(b) SIMON32/64, 10 rounds, $N = 64$", "simon_confirmation_replacement", g.SIMON_ARMS, 64),
    ]
    h = 2.95 if p["width"] > 5 else 2.85
    fig, axes = plt.subplots(1, 2, figsize=(p["width"], h), sharey=True, layout="constrained")
    counts = {}
    for ax, (title, study, arms, n) in zip(axes, panels, strict=True):
        series, zero = [], 0
        for method, label, colour in arms:
            ts = col.success_times(cells, study, method)
            counts[f"{study}|{method}"] = len(ts)
            if ts:
                series.append((label, colour, ts))
            else:
                zero += 1
        _curve_panel(ax, p, title, n, series, zero_label=f"No recovery (0/{n}): {zero} arms")
    axes[0].set_ylabel("Verified keys (% of $N$)")
    nonzero = {k: v for k, v in counts.items() if v}
    if nonzero != EXPECTED["fig2"]:
        raise ValueError(f"fig2 counts {nonzero} differ from published {EXPECTED['fig2']}")
    return fig, {"recoveries": nonzero}


def fig_recovery_d063(p: dict):
    rows = [r for r in map(json.loads, LEDGER.open()) if r.get("type") != "header"]
    n = len({r["key_index"] for r in rows})
    panels = [
        ("(a) SPECK32/64, 7 rounds, $N = 32$", [("R7-G", "Gohr net6 + table", g.C_ORANGE),
                                                ("R7-C", "Classical table", g.C_NAVY),
                                                ("R7-B1", "B1 + table", g.C_PURPLE)]),
        ("(b) SPECK32/64, 8 rounds, $N = 32$", [("R8-G", "Gohr net7 + tables", g.C_ORANGE),
                                                ("R8-C", "Classical tables", g.C_NAVY)]),
    ]
    h = 2.75 if p["width"] > 5 else 2.65
    fig, axes = plt.subplots(1, 2, figsize=(p["width"], h), sharey=True, layout="constrained")
    counts = {}
    for ax, (title, arms) in zip(axes, panels, strict=True):
        series = []
        for arm, label, colour in arms:
            ts = []
            for r in rows:
                if r["arm"] == arm and r["outcome"] == "recovered":
                    if r["total_s"] > 300.0:
                        raise ValueError("Late recovery must not be counted as verified")
                    ts.append(r["total_s"])
            counts[arm] = len(ts)
            series.append((label, colour, ts))
        _curve_panel(ax, p, title, n, series)
    axes[0].set_ylabel("Verified keys (% of $N$)")
    if counts != EXPECTED["d063"] or n != 32:
        raise ValueError(f"D-063 counts {counts} (n={n}) differ from published {EXPECTED['d063']}")
    return fig, {"recoveries": counts}


# --------------------------------------------------------------------------- second stage

def fig_second_stage(p: dict):
    rows = list(csv.DictReader(DEPTH_CSV.open()))
    got = {(r["cipher"], r["profile"], int(r["trailing_keys"])): int(r["verified_recoveries"]) for r in rows}
    if got != EXPECTED["depth"]:
        raise ValueError(f"second-stage counts {got} differ from published {EXPECTED['depth']}")
    h = 2.55 if p["width"] > 5 else 2.45
    fig, axes = plt.subplots(1, 2, figsize=(p["width"], h), sharey=True, layout="constrained")
    styles = {"cms_native_gauss": (g.C_NAVY, "s", "Matrices allowed (5)"),
              "cms_native_no_gauss": (g.C_ORANGE, "o", "Matrices disabled (0)")}
    titles = {"speck32_64": "(a) SPECK32/64, 7 rounds", "simon32_64": "(b) SIMON32/64, 10 rounds"}
    for ax, cipher in zip(axes, ("speck32_64", "simon32_64"), strict=True):
        tidy(ax, grid="y")
        ax.axhline(27, color="#888888", linestyle="--", linewidth=0.8, zorder=1)
        pairs = {}
        for prof, (colour, marker, label) in styles.items():
            ys = [got[(cipher, prof, 1)], got[(cipher, prof, 2)]]
            pairs[prof] = ys
            ax.plot((0, 1), ys, color=colour, marker=marker, markersize=6, markerfacecolor="white",
                    markeredgewidth=1.4, linewidth=1.5, label=label, zorder=3)
        # Higher value of each pair labelled above, lower below.
        for i in (0, 1):
            a, b = pairs["cms_native_gauss"][i], pairs["cms_native_no_gauss"][i]
            for v, above in ((a, a >= b), (b, b > a)):
                ax.annotate(f"{v}/32", (i, v), xytext=(0, 5 if above else -6), textcoords="offset points",
                            ha="center", va="bottom" if above else "top", fontsize=p["tiny"])
        ax.set_title(titles[cipher], loc="left", pad=4)
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(-4, 34)
        ax.set_yticks([0, 8, 16, 24, 32])
        ax.set_xticks((0, 1))
        ax.set_xticklabels(("One stage\n1 trailing key", "Two-stage hybrid\n2 trailing keys"))
    axes[0].set_ylabel("Verified recoveries (of 32)")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="#888888", linestyle="--", linewidth=0.8))
    labels.append("Containment (27/32)")
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, handlelength=2.0,
               columnspacing=1.4)
    return fig, {"recoveries": {"|".join(map(str, k)): v for k, v in got.items()}}


# --------------------------------------------------------------------------- affordable depth

def fig_affordable_depth(p: dict):
    rows = list(csv.DictReader(AFFORD_CSV.open()))
    plotted = [r for r in rows if r["plotted"] == "True"]
    w = p["single"]
    fig, ax = plt.subplots(figsize=(w, 0.72 * w), layout="constrained")
    ax.plot([0.8, 500], [0.8, 500], color="#888888", linestyle=":", linewidth=0.9, zorder=1)
    style = {"incremental": (g.C_NAVY, "o"), "fresh_process": (g.C_ORANGE, "s")}
    for r in plotted:
        colour, marker = style[r["regime"]]
        hollow = r["hollow_marker"] == "True"
        x, y = float(r["x_position_plus_1"]), float(r["k_star"])
        ax.plot([x], [y], linestyle="none", marker=marker, markersize=5.2, markeredgewidth=1.1,
                markerfacecolor="white" if hollow else colour, markeredgecolor=colour, zorder=3)
        dx, dy, ha, va = afd.LABEL_OFFSET.get(r["row_id"], (4, 0, "left", "center"))
        ax.annotate(r["recovered_label"], (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va=va, fontsize=p["tiny"], color=g.C_DARK, zorder=4)
    for text, x, y, ha in afd.GROUP_TEXT:
        ax.text(x, y, text.replace("N=32", "$N = 32$"), fontsize=p["tiny"], family="cmti10",
                color="#555555", ha=ha, va="center", zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.8, 500)
    ax.set_ylim(0.25, 5000)
    xt, yt = [1, 3, 10, 30, 100, 300], [0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]
    ax.xaxis.set_major_locator(FixedLocator(xt))
    ax.xaxis.set_major_formatter(FixedFormatter([str(v) for v in xt]))
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.yaxis.set_major_formatter(FixedFormatter([f"{v:g}" for v in yt]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    tidy(ax)
    ax.set_xlabel("Median correct-candidate position + 1")
    ax.set_ylabel("Affordable depth $k^{*}$")
    handles = [
        Line2D([], [], color=g.C_NAVY, marker="o", linestyle="none", markersize=5, label="Incremental (primary)"),
        Line2D([], [], color=g.C_ORANGE, marker="s", linestyle="none", markersize=5,
               label="Fresh process ($N = 32$)"),
        Line2D([], [], color="#444444", marker="o", markerfacecolor="white", linestyle="none", markersize=5,
               label="Fewer than 10 completed rejections"),
        Line2D([], [], color="#888888", linestyle=":", linewidth=0.9, label="$k^{*}$ = position + 1"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, handlelength=1.6, borderaxespad=0.3)
    return fig, {"points": len(plotted)}


# --------------------------------------------------------------------------- prospective test

def read_predictions() -> list[dict]:
    # The display column holds unquoted commas, so split from both ends.
    lines = PRED_CSV.read_text().splitlines()
    if lines[0] != "arm,display,expected,interval_lo,interval_hi,observed,inside":
        raise ValueError("Unexpected prediction CSV header")
    out = []
    for line in lines[1:]:
        arm, rest = line.split(",", 1)
        display, exp, lo, hi, obs, inside = rest.rsplit(",", 5)
        out.append({"arm": arm, "display": display, "expected": float(exp), "lo": int(lo), "hi": int(hi),
                    "observed": int(obs), "inside": inside == "True"})
    return out


def fig_predictions(p: dict):
    data = read_predictions()
    if len(data) != 8 or sum(r["inside"] for r in data) != EXPECTED["pred_inside"]:
        raise ValueError("Prediction rows differ from the published 8/8 inside")
    for r in data:
        if r["inside"] != (r["lo"] <= r["observed"] <= r["hi"]):
            raise ValueError(f"Inside flag inconsistent for {r['arm']}")
    w = p["single"]
    fig, ax = plt.subplots(figsize=(w, 0.62 * w), layout="constrained")
    ys = list(range(len(data)))[::-1]
    for y, r in zip(ys, data, strict=True):
        ax.plot([r["lo"], r["hi"]], [y, y], color="#888888", linewidth=4, alpha=0.45, solid_capstyle="butt",
                zorder=1)
        ax.plot(r["expected"], y, marker="|", color="#333333", markersize=9, markeredgewidth=1.3, zorder=2)
        colour = g.C_NAVY if r["inside"] else g.C_ORANGE
        ax.plot(r["observed"], y, marker="o", color=colour, markersize=5.2, markeredgewidth=1.1,
                markerfacecolor=colour if r["inside"] else "white", zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["display"] for r in data])
    ax.axhline(3.5, color="#cccccc", linewidth=0.6)
    ax.set_xlim(-0.5, 32.5)
    ax.set_xticks([0, 8, 16, 24, 32])
    ax.set_xlabel("Keys recovered (of 32)")
    tidy(ax, grid="x")
    ax.tick_params(axis="y", length=0)
    handles = [
        Line2D([], [], color="#888888", linewidth=4, alpha=0.45, label="90% prediction interval (fixed in advance)"),
        Line2D([], [], color="#333333", marker="|", linestyle="", markersize=9, markeredgewidth=1.3,
               label="Predicted"),
        Line2D([], [], color=g.C_NAVY, marker="o", linestyle="", markersize=5, label="Observed, inside"),
        Line2D([], [], color=g.C_ORANGE, marker="o", markerfacecolor="white", linestyle="", markersize=5,
               label="Observed, outside"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False, columnspacing=1.2)
    return fig, {"inside": sum(r["inside"] for r in data)}


FIGURES = [
    ("fig_pipeline", fig_pipeline),
    ("fig_recovery_main", fig_recovery_main),
    ("fig_recovery_d063", fig_recovery_d063),
    ("fig_second_stage", fig_second_stage),
    ("fig_affordable_depth", fig_affordable_depth),
    ("fig_prospective_predictions", fig_predictions),
]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--profile", choices=sorted(PROFILES), required=True)
    ap.add_argument("--output-root", type=Path, default=ROOT / "figures")
    ap.add_argument("--force", action="store_true", help="replace outputs from an earlier run of this script")
    args = ap.parse_args(argv)
    p = PROFILES[args.profile]
    out = args.output_root / args.profile
    out.mkdir(parents=True, exist_ok=True)
    targets = [out / f"{stem}.{ext}" for stem, _ in FIGURES for ext in ("pdf", "png")] + [out / "manifest.json"]
    if not args.force and any(t.exists() for t in targets):
        raise SystemExit(f"Refusing to overwrite outputs in {out}; pass --force to replace them")

    inputs = {
        str(CELLS.relative_to(ROOT)): check_pinned(CELLS, CELLS_RECEIPT, "attack_cells_sha256"),
        str(LEDGER.relative_to(ROOT)): check_pinned(LEDGER, LEDGER_RECEIPT, "ledger_sha256"),
        str(DEPTH_CSV.relative_to(ROOT)): check_sidecar(DEPTH_CSV),
        str(AFFORD_CSV.relative_to(ROOT)): check_sidecar(AFFORD_CSV),
        str(PRED_CSV.relative_to(ROOT)): check_sidecar(PRED_CSV),
    }
    set_style(p)
    rendered, checks = {}, {}
    for stem, draw in FIGURES:
        fig, info = draw(p)
        rendered[stem] = fig
        checks[stem] = info
    outputs = {}
    for stem, fig in rendered.items():
        pdf, png = out / f"{stem}.pdf", out / f"{stem}.png"
        fig.savefig(pdf, metadata={"CreationDate": None})
        fig.savefig(png, dpi=300)
        plt.close(fig)
        size = fig.get_size_inches()
        outputs[stem] = {"pdf_sha256": sha(pdf), "png_sha256": sha(png),
                         "size_in": [round(float(size[0]), 3), round(float(size[1]), 3)]}
    manifest = {
        "profile": args.profile,
        "printed_width_in": p["width"],
        "script": "scripts/generate_report_figures.py",
        "script_sha256": sha(Path(__file__).resolve()),
        "inputs_sha256": inputs,
        "checks": checks,
        "outputs": outputs,
        "command": f"python scripts/generate_report_figures.py --profile {args.profile}",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for stem, o in outputs.items():
        print(f"{stem}: {o['size_in'][0]} x {o['size_in'][1]} in")


if __name__ == "__main__":
    main()
