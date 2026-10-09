"""Plot only the audited opened D-055 study; never run an attack or draw data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

CIPHERS = ("speck32_64", "simon32_64")
PROFILES = ("cms_native_gauss", "cms_native_no_gauss")


def checked_input(path: Path) -> dict:
    content = path.read_bytes()
    sidecar = path.with_name(path.name + ".sha256").read_text().split()
    if len(sidecar) != 2 or sidecar != [hashlib.sha256(content).hexdigest(), path.name]:
        raise ValueError("Input SHA-256 sidecar mismatch")
    data = json.loads(content)
    if data.get("audit", {}).get("passed") is not True:
        raise ValueError("Input has no passed audit")
    return data


def summarize_cells(cells: list[dict]) -> list[dict]:
    expected = {
        (c, k, m, p)
        for c in CIPHERS
        for k in range(8)
        for m in (1, 2)
        for p in PROFILES
    }
    seen = set()
    groups = defaultdict(list)
    for cell in cells:
        identity = (
            cell["cipher"],
            cell["key_index"],
            cell["trailing_keys"],
            cell["profile"],
        )
        if identity not in expected or identity in seen:
            raise ValueError("Invalid or duplicate cell")
        seen.add(identity)
        success = cell["assigned_verified_recovery"]
        if type(success) is not bool or cell["outcome"] != (
            "recovered" if success else "global_deadline"
        ):
            raise ValueError("Invalid recovery/outcome flags")
        wall = cell["parent_wall_seconds"]
        if type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0:
            raise ValueError("Invalid online time")
        if success and wall > 300:
            raise ValueError("Late recovery")
        if not success and wall < 300:
            raise ValueError("Premature deadline")
        groups[(identity[0], identity[2], identity[3])].append(cell)
    if seen != expected:
        raise ValueError("Incomplete 64-cell grid")
    rows = []
    for cipher in CIPHERS:
        for profile in PROFILES:
            for depth in (1, 2):
                group = groups[(cipher, depth, profile)]
                rows.append(
                    {
                        "cipher": cipher,
                        "profile": profile,
                        "trailing_keys": depth,
                        "n": len(group),
                        "verified_recoveries": sum(
                            c["assigned_verified_recovery"] for c in group
                        ),
                        "mean_par2_seconds": sum(
                            c["parent_wall_seconds"]
                            if c["assigned_verified_recovery"]
                            else 600
                            for c in group
                        )
                        / len(group),
                    }
                )
    return rows


def generate(source: Path, output_dir: Path) -> None:
    data = checked_input(source)
    rows = summarize_cells(data["cells"])
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(7.16, 2.65), sharey=True, layout="constrained"
    )
    colors = ("#2f5597", "#c55a11")
    for ax, cipher, title in zip(
        axes,
        CIPHERS,
        ("(a) SPECK32/64, 7 rounds", "(b) SIMON32/64, 10 rounds"),
        strict=True,
    ):
        for profile, color, marker, label in zip(
            PROFILES,
            colors,
            ("s", "o"),
            ("G: matrix capacity 5", "X: matrix capacity 0"),
            strict=True,
        ):
            ys = [
                r["verified_recoveries"]
                for r in rows
                if r["cipher"] == cipher and r["profile"] == profile
            ]
            ax.plot(
                (0, 1),
                ys,
                color=color,
                marker=marker,
                markersize=7,
                markerfacecolor="none",
                markeredgewidth=1.5,
                linewidth=1.7,
                label=label,
            )
            for x, y in enumerate(ys):
                # SPECK's two final counts coincide; label that point once.
                if cipher == "speck32_64" and profile == PROFILES[1] and x == 1:
                    continue
                offset = (
                    -14
                    if cipher == "simon32_64" and profile == PROFILES[1] and x == 1
                    else 8
                )
                ax.annotate(
                    f"{y}/8",
                    (x, y),
                    xytext=(0, offset),
                    textcoords="offset points",
                    ha="center",
                    color="#222222",
                    fontsize=10,
                )
        ax.set_title(title, fontsize=11)
        ax.set_xticks(
            (0, 1), ("One stage\n1 trailing key", "Two-stage hybrid\n2 trailing keys")
        )
        ax.set_xlim(-0.25, 1.25)
        ax.set_ylim(-0.35, 8.1)
        ax.set_yticks(range(0, 9, 2))
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Verified recoveries / 8 keys")
    axes[1].legend(loc="upper left", frameon=False, fontsize=8)
    stem = "fig7_backend_ranking_depth"
    for extension in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{extension}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    table = output_dir / f"{stem}.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    receipt = {
        "source_path": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "ledger_sha256": data["audit"]["ledger_sha256"],
        "manifest_sha256": data["audit"]["manifest_sha256"],
        "code_path": str(
            Path(__file__).relative_to(Path(__file__).resolve().parents[1])
        ),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "matplotlib_version": matplotlib.__version__,
        "role": "opened_demo_descriptive_8_keys_per_cipher",
        "outputs": {
            name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
            for name in (f"{stem}.pdf", f"{stem}.png", f"{stem}.csv")
        },
    }
    target = output_dir / f"{stem}.json"
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis",
        type=Path,
        default=Path("docs/backend_ranking_depth_evidence_20261007/analysis_v1.json"),
    )
    parser.add_argument(
        "--study",
        choices=("n8", "n32"),
        default="n8",
        help="Select between legacy D-055 (n8) and audited D-056/D-057 (n32)",
    )
    args = parser.parse_args()
    if args.study == "n32" or "20261009" in str(args.analysis):
        from scripts.generate_backend_ranking_depth_n32_figure import (
            generate as generate_n32,
        )

        generate_n32(args.analysis, args.output_dir)
    else:
        generate(args.analysis, args.output_dir)
