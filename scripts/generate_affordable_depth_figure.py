"""Generate the affordable-depth figure (fig8) and its companion evidence.

Derives the rows of the affordable-depth table in
docs/BREAK_EVEN_ANALYSIS_20261009.md from the
evidence files instead of typing values in:

- N=32 arms: docs/queue_cost_model_evidence_20261009/cells.csv and attempts.csv.
- SIMON primary arms: results/simon_confirmation_replacement/
  simon_confirmation_comparators.json.
- SPECK primary arms: results/phase11_prospective/
  seven_round_prospective_results.json.

Affordable depth is k* = (300 - prep - s) / reject, where prep is a median
preparation time, s the Kaplan-Meier median of the true-candidate solve time
(zero for SPECK primary, which keeps no per-attempt times) and reject the
time to reject one wrong candidate (the mean over completed attempts for the
N=32 and SIMON rows; the median over keys of a per-candidate estimate for
SPECK primary, as specified for this figure).

The script checks each computed row against the doc table, then writes the
figure (PDF and 600 dpi PNG), a CSV of all rows and inputs, a provenance JSON,
a caption and SHA-256 sidecars. It never overwrites an existing output.

Run from the repository root:

    python3 -I scripts/generate_affordable_depth_figure.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import platform
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUDGET_SECONDS = 300.0
HOLLOW_BELOW_COMPLETED = 10
N32_RUNS_EXPECTED = 32
DEFAULT_STEM = "fig8_affordable_depth"

EVIDENCE_DIR = REPO / "docs" / "queue_cost_model_evidence_20261009"
CELLS_CSV = EVIDENCE_DIR / "cells.csv"
ATTEMPTS_CSV = EVIDENCE_DIR / "attempts.csv"
KM_SOURCE = REPO / "scripts" / "fit_queue_cost_model.py"
SIMON_JSON = (
    REPO
    / "results"
    / "simon_confirmation_replacement"
    / "simon_confirmation_comparators.json"
)
SPECK_JSON = (
    REPO / "results" / "phase11_prospective" / "seven_round_prospective_results.json"
)
DOC_MD = REPO / "docs" / "BREAK_EVEN_ANALYSIS_20261009.md"

BLUE = "#2f5597"
ORANGE = "#c55a11"
GREY = "#888888"

# (cipher, depth, profile, row id, doc label or None). Profile "cms_native_gauss"
# is the matrices-on arm, "cms_native_no_gauss" the matrices-off arm.
N32_ARMS = (
    (
        "speck32_64",
        1,
        "cms_native_gauss",
        "N32_SPECK_1w_on",
        "N=32 SPECK, 1 word, matrices on",
    ),
    ("speck32_64", 1, "cms_native_no_gauss", "N32_SPECK_1w_off", None),
    (
        "speck32_64",
        2,
        "cms_native_gauss",
        "N32_SPECK_2w_on",
        "N=32 SPECK, 2 words, matrices on",
    ),
    (
        "speck32_64",
        2,
        "cms_native_no_gauss",
        "N32_SPECK_2w_off",
        "N=32 SPECK, 2 words, matrices off",
    ),
    (
        "simon32_64",
        1,
        "cms_native_gauss",
        "N32_SIMON_1w_on",
        "N=32 SIMON, 1 word, matrices on",
    ),
    (
        "simon32_64",
        1,
        "cms_native_no_gauss",
        "N32_SIMON_1w_off",
        "N=32 SIMON, 1 word, matrices off",
    ),
    (
        "simon32_64",
        2,
        "cms_native_gauss",
        "N32_SIMON_2w_on",
        "N=32 SIMON, 2 words, matrices on",
    ),
    (
        "simon32_64",
        2,
        "cms_native_no_gauss",
        "N32_SIMON_2w_off",
        "N=32 SIMON, 2 words, matrices off",
    ),
)
SIMON_ARMS = (
    ("fully_classical_8parent", "SIMON_primary_classical", "SIMON primary, classical"),
    (
        "neural_classical_hybrid_8parent",
        "SIMON_primary_hybrid",
        "SIMON primary, Bao hybrid",
    ),
    (
        "fully_neural_1parent",
        "SIMON_primary_neural",
        "SIMON primary, Bao neural (1 parent)",
    ),
)
SPECK_ARMS = (
    ("hierarchical_neural_classical", "SPECK_primary_hybrid", "SPECK primary, hybrid"),
    (
        "hierarchical_neural_screened",
        "SPECK_primary_screened",
        "SPECK primary, screened neural",
    ),
)
SPECK_STAGE_BLOCK = 32

CSV_FIELDS = [
    "row_id",
    "group",
    "cipher",
    "label",
    "doc_label",
    "regime",
    "plotted",
    "hollow_marker",
    "prep_s",
    "reject_s",
    "reject_basis",
    "n_wrong_completed",
    "n_wrong_total",
    "n_wrong_solver_timeout",
    "mean_reject_excl_timeouts_s",
    "solve_true_km_median_s",
    "n_true_obs",
    "n_true_censored",
    "k_star",
    "k_star_status",
    "median_true_position",
    "x_position_plus_1",
    "n_position_records",
    "queue_length",
    "recovered",
    "runs",
    "recovered_label",
    "per_candidate_median_s",
    "k_star_median_variant",
    "stage_rank_formula_not_used",
    "source",
    "notes",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked_input(path: Path) -> dict:
    """Hash an input and verify its .sha256 sidecar when one exists."""
    if not path.is_file():
        raise FileNotFoundError(f"Input not found: {path}")
    digest = sha256_bytes(path.read_bytes())
    sidecars = [
        path.with_name(path.name + ".sha256"),
        path.with_suffix(".sha256"),
    ]
    sidecar = next((s for s in sidecars if s.is_file()), None)
    verified = False
    if sidecar is not None:
        parts = sidecar.read_text(encoding="utf-8").split()
        if not parts or parts[0] != digest:
            raise ValueError(f"SHA-256 sidecar {sidecar} does not match {path}")
        verified = True
    return {
        "path": str(path.relative_to(REPO)),
        "sha256": digest,
        "sidecar": sidecar.name if sidecar else None,
        "sidecar_verified": verified,
    }


def load_km_class():
    """Import the Kaplan-Meier estimator from scripts/fit_queue_cost_model.py."""
    spec = importlib.util.spec_from_file_location("fit_queue_cost_model", KM_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {KM_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.KM


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def base_row(**fields) -> dict:
    row = {name: "" for name in CSV_FIELDS}
    row.update({k: ("" if v is None else v) for k, v in fields.items()})
    return row


def n32_rows(km_class) -> list[dict]:
    cells = read_rows(CELLS_CSV)
    attempts = read_rows(ATTEMPTS_CSV)
    arm_of_cell = {}
    cells_by_arm = defaultdict(list)
    for cell in cells:
        arm = (cell["cipher"], int(cell["depth"]), cell["profile"])
        arm_of_cell[cell["cell_id"]] = arm
        cells_by_arm[arm].append(cell)
    attempts_by_arm = defaultdict(list)
    for attempt in attempts:
        attempts_by_arm[arm_of_cell[attempt["cell_id"]]].append(attempt)

    rows = []
    for cipher, depth, profile, row_id, doc_label in N32_ARMS:
        arm = (cipher, depth, profile)
        arm_cells = cells_by_arm[arm]
        if len(arm_cells) != N32_RUNS_EXPECTED:
            raise ValueError(f"{row_id}: expected 32 cells, found {len(arm_cells)}")
        arm_attempts = attempts_by_arm[arm]

        prep = statistics.median(float(c["prep_seconds"]) for c in arm_cells)
        wrong_completed = [
            float(a["solve_wall_seconds"])
            for a in arm_attempts
            if a["is_true"] == "False" and a["censored"] == "False"
        ]
        wrong_total = sum(a["is_true"] == "False" for a in arm_attempts)
        true_obs = []
        for a in arm_attempts:
            if a["is_true"] != "True":
                continue
            censored = a["censored"] == "True"
            seconds = (
                float(a["censor_seconds"])
                if censored
                else float(a["solve_wall_seconds"])
            )
            true_obs.append((seconds, censored))
        solve_median = km_class(true_obs).quantile(0.5)

        positions = [
            int(c["true_queue_position"])
            for c in arm_cells
            if c["true_queue_position"] not in ("", "None")
        ]
        queue_lengths = {int(c["queue_length"]) for c in arm_cells}
        if len(queue_lengths) != 1:
            raise ValueError(f"{row_id}: queue lengths differ {queue_lengths}")
        recovered = sum(c["recovered"] == "True" for c in arm_cells)
        mean_reject = statistics.fmean(wrong_completed)

        cipher_name = "SPECK" if cipher.startswith("speck") else "SIMON"
        words = "1 word" if depth == 1 else "2 words"
        matrices = "on" if profile == "cms_native_gauss" else "off"
        label = f"N=32 {cipher_name}, {words}, matrices {matrices}"
        rows.append(
            base_row(
                row_id=row_id,
                group="N32",
                cipher=cipher,
                label=label,
                doc_label=doc_label or "",
                regime="fresh_process",
                prep_s=prep,
                reject_s=mean_reject,
                reject_basis="mean over completed wrong-candidate attempts",
                n_wrong_completed=len(wrong_completed),
                n_wrong_total=wrong_total,
                solve_true_km_median_s=solve_median,
                n_true_obs=len(true_obs),
                n_true_censored=sum(cens for _, cens in true_obs),
                median_true_position=(
                    statistics.median(positions) if positions else None
                ),
                n_position_records=len(positions),
                queue_length=queue_lengths.pop(),
                recovered=recovered,
                runs=len(arm_cells),
                recovered_label=f"{recovered}/{len(arm_cells)}",
                source=f"{CELLS_CSV.name}; {ATTEMPTS_CSV.name}",
                notes="",
            )
        )
    return rows


def simon_rows(km_class) -> list[dict]:
    records = json.loads(SIMON_JSON.read_text(encoding="utf-8"))["records"]
    truth = {
        r["instance_id"]: tuple(r["full_queue"][0])
        for r in records
        if r["method"] == "oracle_2key_n4"
    }
    rows = []
    for method, row_id, doc_label in SIMON_ARMS:
        arm = [r for r in records if r["method"] == method]
        if len(arm) != 64:
            raise ValueError(f"{method}: expected 64 records, found {len(arm)}")
        prep_values = []
        wrong = []
        wrong_timeouts = []
        wrong_other = []
        true_obs = []
        positions = []
        queue_lengths = set()
        empty_queues = 0
        recovered = 0
        in_flight = 0
        for r in arm:
            truth_key = truth[r["instance_id"]]
            queue = [tuple(x) for x in r["full_queue"]]
            if queue:
                queue_lengths.add(len(queue))
            else:
                empty_queues += 1
            if truth_key in queue:
                positions.append(queue.index(truth_key))
            recovered += r["outcome"] == "verified_recovery"
            prep_values.append(
                r["complete_online_wall_seconds"]
                - r["solving_wall_seconds"]
                - r["verification_wall_seconds"]
            )
            if r["in_flight_attempt"]:
                in_flight += 1
            for attempt in r["candidate_attempts"]:
                candidate = (attempt["candidate_k8"], attempt["candidate_k9"])
                seconds = attempt["solve_wall_seconds"]
                if candidate == truth_key:
                    true_obs.append((seconds, False))
                else:
                    wrong.append(seconds)
                    if attempt["status"] == "solver_timeout":
                        wrong_timeouts.append(seconds)
                    else:
                        wrong_other.append(seconds)
        if len(queue_lengths) != 1:
            raise ValueError(f"{method}: queue lengths differ {queue_lengths}")
        rows.append(
            base_row(
                row_id=row_id,
                group="SIMON_primary",
                cipher="simon32_64",
                label=doc_label,
                doc_label=doc_label,
                regime="incremental",
                prep_s=statistics.median(prep_values),
                reject_s=statistics.fmean(wrong),
                reject_basis=(
                    "mean over candidate_attempts whose candidate is not the truth"
                ),
                n_wrong_completed=len(wrong),
                n_wrong_total=len(wrong),
                n_wrong_solver_timeout=len(wrong_timeouts),
                mean_reject_excl_timeouts_s=statistics.fmean(wrong_other),
                solve_true_km_median_s=km_class(true_obs).quantile(0.5),
                n_true_obs=len(true_obs),
                n_true_censored=0,
                median_true_position=statistics.median(positions),
                n_position_records=len(positions),
                queue_length=queue_lengths.pop(),
                recovered=recovered,
                runs=len(arm),
                recovered_label=f"{recovered}/{len(arm)}",
                source=f"{SIMON_JSON.name}: {method}, oracle_2key_n4 truth",
                notes=(
                    f"{in_flight} in-flight attempts excluded (as in step4); "
                    f"{empty_queues} records with empty queue excluded from queue "
                    "length; solver_timeout attempts included in the mean, as in the "
                    "doc table"
                ),
            )
        )
    return rows


def speck_rows() -> list[dict]:
    by_method = json.loads(SPECK_JSON.read_text(encoding="utf-8"))["records_by_method"]
    rows = []
    for method, row_id, doc_label in SPECK_ARMS:
        records = by_method[method]
        if len(records) != 128:
            raise ValueError(f"{method}: expected 128 records, found {len(records)}")
        prep = statistics.median(r["timing_seconds"]["ranking"] for r in records)
        per_candidate = [
            (r["online_seconds"] - r["timing_seconds"]["ranking"])
            / r["candidates_attempted"]
            for r in records
        ]
        per_candidate_mean = statistics.fmean(per_candidate)
        per_candidate_median = statistics.median(per_candidate)

        stage_positions = []
        queue_lengths = set()
        observed_index = []
        for r in records:
            stages = {s["stage_index"]: s for s in r["stages"]}
            ranks = [stages.get(i, {}).get("true_round_key_rank") for i in (0, 1)]
            beams = [stages[i]["beam_width"] for i in (0, 1) if i in stages]
            if len(beams) == 2:
                queue_lengths.add(beams[0] * beams[1])
            if None not in ranks:
                stage_positions.append(
                    (ranks[0] - 1) * SPECK_STAGE_BLOCK + (ranks[1] - 1)
                )
            if r["success"]:
                observed_index.append(r["candidates_attempted"] - 1)
        if len(queue_lengths) != 1:
            raise ValueError(f"{method}: queue lengths differ {queue_lengths}")
        recovered = sum(bool(r["success"]) for r in records)
        if len(observed_index) != recovered:
            raise ValueError(f"{method}: recovered count mismatch")

        rows.append(
            base_row(
                row_id=row_id,
                group="SPECK_primary",
                cipher="speck32_64",
                label=doc_label,
                doc_label=doc_label,
                regime="incremental",
                prep_s=prep,
                reject_s=per_candidate_mean,
                reject_basis=(
                    "mean over keys of (online - ranking) / candidates_attempted; "
                    "includes the correct solve (approximate)"
                ),
                solve_true_km_median_s=0.0,
                median_true_position=statistics.median(observed_index),
                n_position_records=len(observed_index),
                queue_length=queue_lengths.pop(),
                recovered=recovered,
                runs=len(records),
                recovered_label=f"{recovered}/{len(records)}",
                per_candidate_median_s=per_candidate_median,
                k_star_median_variant=(BUDGET_SECONDS - prep) / per_candidate_median,
                stage_rank_formula_not_used=statistics.median(stage_positions),
                source=f"{SPECK_JSON.name}: {method}",
                notes=(
                    "solve term s = 0 (the correct solve is inside the per-candidate "
                    "mean); position = observed attempt index candidates_attempted-1 "
                    "over recovered keys only (assumes the verified candidate is the "
                    f"final attempt); stage_rank_formula_not_used = median of "
                    f"(rank0-1)*32+(rank1-1) over {len(stage_positions)} records, "
                    "not plotted"
                ),
            )
        )
    return rows


def finalize(row: dict) -> dict:
    """Compute k*, its status, the x position and the plotting flags."""
    prep = row["prep_s"]
    reject = row["reject_s"]
    solve = row["solve_true_km_median_s"]
    if row["median_true_position"] != "":
        row["x_position_plus_1"] = row["median_true_position"] + 1
    if not math.isfinite(solve):
        row["solve_true_km_median_s"] = "inf"
        row["k_star"] = ""
        bound = (BUDGET_SECONDS - prep - BUDGET_SECONDS) / reject
        row["k_star_status"] = (
            "not estimable: KM median of true solve not reached by the largest "
            f"observation; k* <= {bound:.2f} (solve at deadline) and not plotted"
        )
    else:
        k_star = (BUDGET_SECONDS - prep - solve) / reject
        row["k_star"] = k_star
        row["k_star_status"] = "computed"
    plotted = (
        row["k_star"] != "" and row["k_star"] > 0 and row["x_position_plus_1"] != ""
    )
    row["plotted"] = plotted
    if row["group"] == "N32":
        row["hollow_marker"] = row["n_wrong_completed"] < HOLLOW_BELOW_COMPLETED
    else:
        row["hollow_marker"] = False
    return row


def parse_doc_table() -> dict:
    rows = {}
    in_table = False
    for line in DOC_MD.read_text(encoding="utf-8").splitlines():
        if line.startswith("| Campaign / arm"):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if re.fullmatch(r"[-:]+", cells[0]):
            continue
        rows[cells[0]] = cells[1:]
    if len(rows) != 12:
        raise ValueError(f"Expected 12 doc table rows, parsed {len(rows)}")
    return rows


def parse_doc_number(text: str):
    """Return ('=', value, tol), ('<', value, None), ('>', value, None), None or 'text'."""
    cleaned = re.sub(r"[≈†‡§,]", "", text).strip()
    if cleaned in ("—", "-", ""):
        return None
    bound = re.fullmatch(r"([<>])\s*([0-9.]+)", cleaned)
    if bound:
        return (bound.group(1), float(bound.group(2)), None)
    if "/" in cleaned:
        return ("text", cleaned, None)
    if re.fullmatch(r"-?[0-9]+(\.[0-9]+)?", cleaned):
        if "." in cleaned:
            tol = 0.5 * 10 ** -len(cleaned.split(".")[1])
        else:
            trailing = len(cleaned) - len(cleaned.rstrip("0")) if cleaned != "0" else 0
            tol = 0.5 * 10**trailing
        return ("=", float(cleaned), tol)
    raise ValueError(f"Cannot parse doc value {text!r}")


def compare_with_doc(rows: list[dict], doc: dict) -> list[dict]:
    checks = []
    for row in rows:
        label = row["doc_label"]
        if not label:
            checks.append(
                dict(
                    row_id=row["row_id"],
                    field="row",
                    doc="",
                    computed="",
                    status="not in doc table",
                )
            )
            continue
        doc_cells = doc[label]
        fields = [
            ("prep_s", doc_cells[0], row["prep_s"]),
            ("reject_s", doc_cells[1], row["reject_s"]),
            ("solve_true_km_median_s", doc_cells[2], row["solve_true_km_median_s"]),
            ("k_star", doc_cells[3], row["k_star"]),
            ("median_true_position", doc_cells[4], row["median_true_position"]),
            ("queue_length", doc_cells[5], row["queue_length"]),
            ("recovered_label", doc_cells[6], row["recovered_label"]),
        ]
        for field, doc_text, computed in fields:
            parsed = parse_doc_number(doc_text)
            if parsed is None:
                checks.append(
                    dict(
                        row_id=row["row_id"],
                        field=field,
                        doc=doc_text,
                        computed=computed,
                        status="info: doc shows no value",
                    )
                )
                continue
            kind, expected, tol = parsed
            if kind == "text":
                ok = expected == computed
            elif kind == "<":
                if computed == "":
                    ok = True
                else:
                    ok = computed < expected
            elif kind == ">":
                ok = computed == "inf" or (computed != "" and computed > expected)
            else:
                if computed in ("", "inf"):
                    ok = False
                else:
                    ok = abs(computed - expected) <= tol + 1e-9
            checks.append(
                dict(
                    row_id=row["row_id"],
                    field=field,
                    doc=doc_text,
                    computed=computed,
                    status="agree" if ok else "DISAGREE",
                )
            )
    return checks


def draw(rows: list[dict]) -> tuple[bytes, bytes]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import (
        FixedFormatter,
        FixedLocator,
        NullFormatter,
        NullLocator,
    )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.48, 3.4), layout="constrained")

    ax.plot(
        [0.8, 500],
        [0.8, 500],
        color=GREY,
        linestyle=":",
        linewidth=0.9,
        zorder=1,
    )
    regime_style = {
        "incremental": {"color": BLUE, "marker": "o"},
        "fresh_process": {"color": ORANGE, "marker": "s"},
    }
    for row in rows:
        if not row["plotted"]:
            continue
        style = regime_style[row["regime"]]
        hollow = row["hollow_marker"]
        ax.plot(
            [row["x_position_plus_1"]],
            [row["k_star"]],
            linestyle="none",
            marker=style["marker"],
            markersize=5.2,
            markerfacecolor="white" if hollow else style["color"],
            markeredgecolor=style["color"],
            markeredgewidth=1.2,
            zorder=3,
        )
        dx, dy, ha, va = LABEL_OFFSET.get(row["row_id"], (4, 0, "left", "center"))
        ax.annotate(
            row["recovered_label"],
            (row["x_position_plus_1"], row["k_star"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=6.5,
            color="#222222",
            zorder=4,
        )

    for text, x, y, ha in GROUP_TEXT:
        ax.text(
            x,
            y,
            text,
            fontsize=6.8,
            color="#555555",
            style="italic",
            ha=ha,
            va="center",
            zorder=2,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.8, 500)
    ax.set_ylim(0.25, 5000)
    ax.xaxis.set_major_locator(FixedLocator([1, 3, 10, 30, 100, 300]))
    ax.xaxis.set_major_formatter(FixedFormatter(["1", "3", "10", "30", "100", "300"]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator([0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]))
    ax.yaxis.set_major_formatter(
        FixedFormatter(["0.3", "1", "3", "10", "30", "100", "300", "1000", "3000"])
    )
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", color="#e0e0e0", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Median correct-candidate position + 1")
    ax.set_ylabel(r"Affordable depth $k^{*}$")

    handles = [
        Line2D(
            [],
            [],
            color=BLUE,
            marker="o",
            linestyle="none",
            markersize=5,
            label="Incremental (primary)",
        ),
        Line2D(
            [],
            [],
            color=ORANGE,
            marker="s",
            linestyle="none",
            markersize=5,
            label="Fresh process (N=32)",
        ),
        Line2D(
            [],
            [],
            color="#444444",
            marker="o",
            markerfacecolor="white",
            linestyle="none",
            markersize=5,
            label="< 10 completed rejections",
        ),
        Line2D(
            [],
            [],
            color=GREY,
            linestyle=":",
            linewidth=0.9,
            label=r"$k^{*}$ = position $+ 1$",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(1.0, 0.0),
        frameon=False,
        fontsize=6.6,
        handlelength=1.6,
        borderaxespad=0.3,
    )

    pdf_buffer = io.BytesIO()
    fig.savefig(pdf_buffer, format="pdf", metadata={"CreationDate": None})
    png_buffer = io.BytesIO()
    fig.savefig(png_buffer, format="png", dpi=600)
    plt.close(fig)
    return pdf_buffer.getvalue(), png_buffer.getvalue()


# Label offsets (points: dx, dy, ha, va) per row id, and group captions
# (text, x, y, ha) in data coordinates. Set after inspecting the rendered layout.
LABEL_OFFSET = {
    "SIMON_primary_hybrid": (5, 2, "left", "bottom"),
    "SIMON_primary_classical": (5, -2, "left", "top"),
    "N32_SPECK_1w_on": (-5, 0, "right", "center"),
    "N32_SPECK_1w_off": (-5, 0, "right", "center"),
}
GROUP_TEXT = [
    ("SIMON primary", 1.15, 1500, "left"),
    ("SPECK primary", 90, 3300, "left"),
    ("SIMON N=32", 3.4, 70, "left"),
    ("SPECK N=32", 45, 380, "left"),
]


def caption_text(n_rows: int, n_plotted: int, unplotted: list[str]) -> str:
    unplotted_text = ", ".join(unplotted) if unplotted else "none"
    return (
        "Affordable depth k* against the median position of the correct candidate, "
        f"for {n_rows} pipelines from the break-even table ({n_plotted} plotted). "
        "k* = (300 s - preparation - time to solve the correct candidate) / time to "
        "reject one wrong candidate. Blue circles: incremental solver, primary "
        "campaigns. For SPECK the time to reject is the mean over keys of (online - "
        "ranking) / candidates attempted, which includes the correct solve, so it is "
        "approximate, and the solve term is zero; SPECK positions are the observed "
        "attempt index of the correct candidate, from recovered keys only. Orange "
        "squares: a fresh solver process per candidate, N=32 study. Hollow markers: "
        "fewer than 10 completed wrong-candidate attempts; cut-off attempts are "
        "excluded from the mean, so k* is an upper bound. Labels give recovered / "
        "runs. Points on or above the dotted line k* = position + 1 can reach the "
        "correct candidate. Not plotted: "
        f"{unplotted_text} (Kaplan-Meier median solve time not reached by the "
        "deadline, so k* is not estimable)."
    )


def write_new(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", type=Path, default=REPO / "figures")
    parser.add_argument("--stem", type=str, default=DEFAULT_STEM)
    args = parser.parse_args(argv)

    stem = args.stem
    out = args.output_dir
    names = {
        "pdf": f"{stem}.pdf",
        "png": f"{stem}.png",
        "csv": f"{stem}.csv",
        "json": f"{stem}.json",
        "caption": f"{stem}.caption.txt",
    }
    targets = [out / name for name in names.values()]
    targets += [p.with_name(p.name + ".sha256") for p in targets]
    existing = [str(p) for p in targets if p.exists()]
    if existing:
        raise SystemExit(f"Refusing to overwrite existing outputs: {existing}")

    km_class = load_km_class()
    ordered = [
        finalize(r) for r in (*simon_rows(km_class), *speck_rows(), *n32_rows(km_class))
    ]

    doc = parse_doc_table()
    checks = compare_with_doc(ordered, doc)
    n_disagree = sum(c["status"] == "DISAGREE" for c in checks)

    plotted = [r for r in ordered if r["plotted"]]
    unplotted = [r["label"] for r in ordered if not r["plotted"]]
    out.mkdir(parents=True, exist_ok=True)

    pdf_bytes, png_bytes = draw(ordered)
    caption = caption_text(len(ordered), len(plotted), unplotted) + "\n"

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in ordered:
        writer.writerow(
            {k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}
        )
    csv_bytes = buffer.getvalue().encode("utf-8")

    inputs = [
        checked_input(CELLS_CSV),
        checked_input(ATTEMPTS_CSV),
        checked_input(KM_SOURCE),
        checked_input(SIMON_JSON),
        checked_input(SPECK_JSON),
        checked_input(DOC_MD),
    ]
    payloads = {
        names["pdf"]: pdf_bytes,
        names["png"]: png_bytes,
        names["csv"]: csv_bytes,
        names["caption"]: caption.encode("utf-8"),
    }
    output_hashes = {name: sha256_bytes(data) for name, data in payloads.items()}

    import matplotlib

    script_bytes = Path(__file__).resolve().read_bytes()
    receipt = {
        "study": "EXPLORATORY/RETROSPECTIVE affordable-depth figure",
        "source_doc": str(DOC_MD.relative_to(REPO)),
        "code_path": str(Path(__file__).resolve().relative_to(REPO)),
        "code_sha256": sha256_bytes(script_bytes),
        "command": " ".join(
            [
                "python3 -I" if sys.flags.isolated else "python3",
                *sys.argv,
            ]
        ),
        "interpreter": sys.executable,
        "python_version": platform.python_version(),
        "matplotlib_version": matplotlib.__version__,
        "inputs": inputs,
        "outputs": output_hashes,
        "row_count": len(ordered),
        "plotted_count": len(plotted),
        "not_plotted": unplotted,
        "doc_comparison": {
            "agree": sum(c["status"] == "agree" for c in checks),
            "disagree": n_disagree,
            "checks": checks,
        },
        "role": "exploratory_descriptive_retrospective",
    }
    json_bytes = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    payloads[names["json"]] = json_bytes

    for name, data in payloads.items():
        write_new(out / name, data)
        digest = sha256_bytes(data)
        write_new(out / f"{name}.sha256", f"{digest}  {name}\n".encode())

    print(f"rows={len(ordered)} plotted={len(plotted)} not_plotted={unplotted}")
    for row in ordered:
        k = row["k_star"]
        k_text = f"{k:.3f}" if isinstance(k, float) else str(k)
        print(
            f"{row['row_id']:<22} prep={row['prep_s']:.2f} "
            f"reject={row['reject_s']:.4f} solve={row['solve_true_km_median_s']} "
            f"pos={row['median_true_position']} queue={row['queue_length']} "
            f"rec={row['recovered_label']} k*={k_text} plotted={row['plotted']}"
        )
    for check in checks:
        if check["status"] != "agree":
            print(
                f"  {check['status']}: {check['row_id']} {check['field']} "
                f"doc={check['doc']!r} computed={check['computed']}"
            )
    print(
        f"doc comparison: agree={receipt['doc_comparison']['agree']} "
        f"disagree={n_disagree}"
    )
    print(f"wrote {out / names['pdf']} and siblings")


if __name__ == "__main__":
    main()
