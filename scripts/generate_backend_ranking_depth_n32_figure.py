"""Plot the audited N=32 factorial backend-by-guidance study (D-056/D-057).

This script reads only the frozen, audited analysis_v2.json evidence from D-056/D-057.
It strictly validates the complete 256-cell factorial grid before rendering the
single-column publication figure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

CANONICAL_SOURCE_SHA256 = (
    "7eb12c76ded7ab078109ef0d387440152d1281175054a55dafc38b5b3644ade8"
)
CIPHERS = ("speck32_64", "simon32_64")
PROFILES = ("cms_native_gauss", "cms_native_no_gauss")
EXPECTED_CELLS_COUNT = 256
EXPECTED_KEYS_PER_CIPHER = 32
EXPECTED_CONTAINMENT_COUNT = 27


def checked_input(path: Path) -> dict:
    """Validate canonical source file, SHA-256 sidecar, and evidence index."""
    if not path.is_file():
        raise FileNotFoundError(f"Canonical analysis file not found: {path}")

    content = path.read_bytes()
    actual_sha = hashlib.sha256(content).hexdigest()

    sidecar_path = path.with_name(path.name + ".sha256")
    if not sidecar_path.is_file():
        raise ValueError(f"Missing SHA-256 sidecar: {sidecar_path}")

    sidecar_parts = sidecar_path.read_text().strip().split()
    if len(sidecar_parts) != 2 or sidecar_parts != [actual_sha, path.name]:
        raise ValueError(
            f"Input SHA-256 sidecar mismatch: expected {actual_sha} for {path.name}"
        )

    if actual_sha != CANONICAL_SOURCE_SHA256:
        raise ValueError(
            f"Source SHA-256 {actual_sha} differs from required pin {CANONICAL_SOURCE_SHA256}"
        )

    evidence_index = path.parent / "evidence_index_v1.json"
    if not evidence_index.is_file():
        raise ValueError(f"Missing evidence index: {evidence_index}")

    data = json.loads(content)
    if data.get("audit", {}).get("passed") is not True:
        raise ValueError("Input audit has not passed")
    if data.get("evidence_role", "").lower() != "development":
        raise ValueError(f"Unexpected evidence role: {data.get('evidence_role')}")
    return data


def summarize_cells(cells: list[dict]) -> tuple[list[dict], dict]:
    """Validate the exact 256-cell factorial grid and recompute recovery and PAR-2 totals."""
    if len(cells) != EXPECTED_CELLS_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CELLS_COUNT} cells, got {len(cells)}"
        )

    expected_grid = {
        (c, k, m, p)
        for c in CIPHERS
        for k in range(EXPECTED_KEYS_PER_CIPHER)
        for m in (1, 2)
        for p in PROFILES
    }

    seen = set()
    groups = defaultdict(list)
    contained_keys_by_arm = defaultdict(set)

    for cell in cells:
        identity = (
            cell.get("cipher"),
            cell.get("key_index"),
            cell.get("trailing_keys"),
            cell.get("profile"),
        )
        if identity not in expected_grid or identity in seen:
            raise ValueError(f"Invalid or duplicate cell: {identity}")
        seen.add(identity)

        success = cell.get("assigned_verified_recovery")
        if type(success) is not bool:
            raise ValueError(f"Non-boolean recovery flag in cell {identity}")

        outcome = cell.get("outcome")
        if success:
            if outcome != "recovered":
                raise ValueError(
                    f"Inconsistent outcome {outcome} for verified cell {identity}"
                )
        else:
            if outcome not in ("global_deadline", "queue_exhausted"):
                raise ValueError(
                    f"Unexpected unrecovered outcome {outcome} in cell {identity}"
                )

        wall = cell.get("parent_wall_seconds")
        if type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0:
            raise ValueError(f"Invalid parent wall time {wall} in cell {identity}")

        if success and wall > 300.0:
            raise ValueError(f"Late positive recovery ({wall} s) in cell {identity}")

        if not success and outcome == "global_deadline" and wall < 299.9:
            raise ValueError(f"Premature deadline ({wall} s) in cell {identity}")

        # Containment check
        if cell.get("queue_containment_measured") is not True:
            raise ValueError(f"Unmeasured containment in cell {identity}")
        if cell.get("true_tuple_queue_index_zero_based") is not None:
            contained_keys_by_arm[(identity[0], identity[2], identity[3])].add(
                identity[1]
            )

        groups[(identity[0], identity[2], identity[3])].append(cell)

    if seen != expected_grid:
        raise ValueError("Incomplete 256-cell factorial grid")

    # Verify identical containment across all arms within each cipher
    for cipher in CIPHERS:
        ref_keys = None
        for profile in PROFILES:
            for depth in (1, 2):
                arm_keys = contained_keys_by_arm[(cipher, depth, profile)]
                if len(arm_keys) != EXPECTED_CONTAINMENT_COUNT:
                    raise ValueError(
                        f"Unexpected containment count {len(arm_keys)} for {cipher} m={depth} {profile}"
                    )
                if ref_keys is None:
                    ref_keys = arm_keys
                elif arm_keys != ref_keys:
                    raise ValueError(
                        f"Containment key mismatch across arms for {cipher}"
                    )

    rows = []
    for cipher in CIPHERS:
        for profile in PROFILES:
            for depth in (1, 2):
                group = groups[(cipher, depth, profile)]
                n = len(group)
                recoveries = sum(
                    1 for c in group if c["assigned_verified_recovery"]
                )
                mean_par2 = (
                    sum(
                        c["parent_wall_seconds"]
                        if c["assigned_verified_recovery"]
                        else 600.0
                        for c in group
                    )
                    / n
                )
                rows.append(
                    {
                        "cipher": cipher,
                        "profile": profile,
                        "trailing_keys": depth,
                        "n": n,
                        "verified_recoveries": recoveries,
                        "mean_par2_seconds": mean_par2,
                    }
                )

    metadata = {
        "containment_count": EXPECTED_CONTAINMENT_COUNT,
        "total_keys_per_cipher": EXPECTED_KEYS_PER_CIPHER,
        "total_cells": EXPECTED_CELLS_COUNT,
    }
    return rows, metadata


def generate(
    source: Path,
    output_dir: Path,
    stem: str = "fig7_backend_ranking_depth_n32",
) -> dict:
    """Generate the publication-ready single-column N=32 figure and artifacts."""
    data = checked_input(source)
    rows, metadata = summarize_cells(data["cells"])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "pdf.fonttype": 42,
        }
    )

    # Native single-column vertically stacked 2-panel layout (~3.48 in wide)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(3.48, 4.3),
        sharex=True,
        sharey=True,
        layout="constrained",
    )

    colors = {
        "cms_native_gauss": "#2f5597",
        "cms_native_no_gauss": "#c55a11",
    }
    markers = {
        "cms_native_gauss": "s",
        "cms_native_no_gauss": "o",
    }
    labels = {
        "cms_native_gauss": "Matrices allowed (5)",
        "cms_native_no_gauss": "Matrices disabled (0)",
    }

    cipher_titles = {
        "speck32_64": "(a) SPECK32/64, 7 rounds",
        "simon32_64": "(b) SIMON32/64, 10 rounds",
    }

    for ax, cipher in zip(axes, CIPHERS, strict=True):
        # Containment reference line
        ax.axhline(
            metadata["containment_count"],
            color="#888888",
            linestyle="--",
            linewidth=0.8,
            alpha=0.7,
            zorder=1,
        )

        for profile in PROFILES:
            ys = [
                r["verified_recoveries"]
                for r in rows
                if r["cipher"] == cipher and r["profile"] == profile
            ]
            ax.plot(
                (0, 1),
                ys,
                color=colors[profile],
                marker=markers[profile],
                markersize=6.5,
                markerfacecolor="white",
                markeredgewidth=1.5,
                linewidth=1.6,
                label=labels[profile],
                zorder=3,
            )

        # Annotations positioned to avoid collisions
        if cipher == "speck32_64":
            # m=1 (4 and 2)
            ax.annotate(
                "4/32",
                (0, 4),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            ax.annotate(
                "2/32",
                (0, 2),
                xytext=(0, -8),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=8,
            )
            # m=2 (19 and 18)
            ax.annotate(
                "19/32",
                (1, 19),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            ax.annotate(
                "18/32",
                (1, 18),
                xytext=(0, -8),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=8,
            )
        else:
            # simon32_64: m=1 (23 and 3)
            ax.annotate(
                "23/32",
                (0, 23),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            ax.annotate(
                "3/32",
                (0, 3),
                xytext=(0, -8),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=8,
            )
            # m=2 (27 and 22)
            ax.annotate(
                "27/32",
                (1, 27),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            ax.annotate(
                "22/32",
                (1, 22),
                xytext=(0, -8),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=8,
            )

        ax.set_title(cipher_titles[cipher], fontsize=9.5, loc="left", pad=4)
        ax.set_xlim(-0.25, 1.25)
        ax.set_ylim(-3.0, 33.5)
        ax.set_yticks([0, 8, 16, 24, 32])
        ax.grid(axis="y", color="#e0e0e0", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel("Verified recoveries / 32 keys", fontsize=8.5)

    axes[1].set_xticks((0, 1))
    axes[1].set_xticklabels(
        ("One stage\n1 trailing key", "Two-stage hybrid\n2 trailing keys"),
        fontsize=8.5,
    )

    # Shared legend placed inside panel (a) in open space below y=27
    containment_line = plt.Line2D(
        [0], [0], color="#888888", linestyle="--", linewidth=0.8
    )
    handles, leg_labels = axes[0].get_legend_handles_labels()
    handles.append(containment_line)
    leg_labels.append("Containment (27/32)")
    axes[0].legend(
        handles,
        leg_labels,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.76),
        frameon=False,
        fontsize=7.5,
    )

    # Export vector PDF and 600 dpi PNG
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)

    # Export companion CSV
    csv_path = output_dir / f"{stem}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    # Write SHA-256 sidecars
    outputs_hashes = {}
    for path in (pdf_path, png_path, csv_path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        outputs_hashes[path.name] = digest
        sidecar = path.with_name(path.name + ".sha256")
        sidecar.write_text(f"{digest}  {path.name}\n")

    # Export provenance JSON
    receipt = {
        "canonical_study": "D-056/D-057",
        "code_path": str(
            Path(__file__).relative_to(Path(__file__).resolve().parents[1])
        ),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "segments": data.get("audit", {}).get("segments", []),
        "matplotlib_version": matplotlib.__version__,
        "outputs": outputs_hashes,
        "role": "audited_development_descriptive_32_keys_per_cipher",
        "source_path": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    json_path = output_dir / f"{stem}.json"
    json_content = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    json_path.write_text(json_content)
    json_digest = hashlib.sha256(json_path.read_bytes()).hexdigest()
    json_sidecar = json_path.with_name(json_path.name + ".sha256")
    json_sidecar.write_text(f"{json_digest}  {json_path.name}\n")

    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis",
        type=Path,
        default=Path(
            "docs/backend_ranking_depth_evidence_20261009/analysis_v2.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument(
        "--stem",
        type=str,
        default="fig7_backend_ranking_depth_n32",
    )
    args = parser.parse_args()
    generate(args.analysis, args.output_dir, stem=args.stem)
