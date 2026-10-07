"""Adversarial checks before scientific data can become a publication figure."""

import copy
import json
from pathlib import Path

import pytest

from scripts.generate_backend_ranking_depth_figure import checked_input, summarize_cells

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "docs/backend_ranking_depth_evidence_20261007/analysis_v1.json"
)


def test_plot_totals_agree_with_independent_raw_audit():
    rows = summarize_cells(checked_input(SOURCE)["cells"])
    assert [r["verified_recoveries"] for r in rows] == [4, 6, 1, 6, 5, 6, 0, 5]
    assert [r["mean_par2_seconds"] for r in rows] == pytest.approx(
        [
            382.13409934,
            230.41131532,
            561.11099582,
            215.26308345,
            326.10592614,
            209.47889375,
            600.0,
            294.53994054,
        ]
    )


@pytest.mark.parametrize(
    "defect",
    [
        "duplicate",
        "missing",
        "string_flag",
        "wrong_outcome",
        "late",
        "nan",
        "premature",
    ],
)
def test_reject_misleading_figure_inputs(defect):
    cells = copy.deepcopy(json.loads(SOURCE.read_text())["cells"])
    if defect == "duplicate":
        cells.append(cells[0])
    elif defect == "missing":
        cells.pop()
    elif defect == "string_flag":
        cells[0]["assigned_verified_recovery"] = "false"
    elif defect == "wrong_outcome":
        cells[0]["outcome"] = "recovered"
    elif defect == "late":
        next(c for c in cells if c["assigned_verified_recovery"])[
            "parent_wall_seconds"
        ] = 300.1
    elif defect == "nan":
        cells[0]["parent_wall_seconds"] = float("nan")
    elif defect == "premature":
        cells[0]["parent_wall_seconds"] = 1
    with pytest.raises(ValueError):
        summarize_cells(cells)


def test_corrupted_sidecar_rejected(tmp_path):
    path = tmp_path / SOURCE.name
    path.write_bytes(SOURCE.read_bytes())
    path.with_name(path.name + ".sha256").write_text("0" * 64 + "  " + path.name)
    with pytest.raises(ValueError, match="sidecar"):
        checked_input(path)
