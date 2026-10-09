"""Adversarial validation and unit tests for the N=32 backend ranking depth publication figure."""

import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.generate_backend_ranking_depth_n32_figure import (
    CANONICAL_SOURCE_SHA256,
    checked_input,
    generate,
    summarize_cells,
)

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "docs/backend_ranking_depth_evidence_20261009/analysis_v2.json"
)


def test_canonical_source_hash_and_sidecar():
    """Verify canonical source matches required pin and sidecar."""
    assert SOURCE.is_file(), f"Missing source: {SOURCE}"
    content = SOURCE.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    assert digest == CANONICAL_SOURCE_SHA256

    sidecar = SOURCE.with_name(SOURCE.name + ".sha256")
    assert sidecar.is_file()
    parts = sidecar.read_text().strip().split()
    assert parts == [CANONICAL_SOURCE_SHA256, SOURCE.name]


def test_n32_plot_totals_and_par2_agree_with_audit():
    """Verify all 8 arms match audited recovery totals and PAR-2 times."""
    data = checked_input(SOURCE)
    rows, metadata = summarize_cells(data["cells"])

    assert metadata["total_cells"] == 256
    assert metadata["total_keys_per_cipher"] == 32
    assert metadata["containment_count"] == 27
    assert len(rows) == 8

    # Recoveries: [speck_g_m1, speck_g_m2, speck_x_m1, speck_x_m2, simon_g_m1, simon_g_m2, simon_x_m1, simon_x_m2]
    recoveries = [r["verified_recoveries"] for r in rows]
    assert recoveries == [4, 18, 2, 19, 23, 27, 3, 22]

    # Exact PAR-2 checks
    par2s = [r["mean_par2_seconds"] for r in rows]
    expected_par2s = [
        543.1347931979378,
        329.3279917604691,
        575.2461305455934,
        307.07477749600025,
        255.09946512512556,
        163.59790535928096,
        554.4998725129999,
        273.62298777478077,
    ]
    assert par2s == pytest.approx(expected_par2s, rel=1e-6)


def test_containment_invariants():
    """Verify containment is 27/32 across all arms on identical keys."""
    data = checked_input(SOURCE)
    cells = data["cells"]

    for cipher in ("speck32_64", "simon32_64"):
        contained_keys_by_arm = []
        for profile in ("cms_native_gauss", "cms_native_no_gauss"):
            for depth in (1, 2):
                arm_cells = [
                    c
                    for c in cells
                    if c["cipher"] == cipher
                    and c["profile"] == profile
                    and c["trailing_keys"] == depth
                ]
                assert len(arm_cells) == 32
                contained = {
                    c["key_index"]
                    for c in arm_cells
                    if c.get("true_tuple_queue_index_zero_based") is not None
                }
                assert len(contained) == 27
                contained_keys_by_arm.append(contained)
        # All 4 arms of the cipher must contain the exact same 27 keys
        for other in contained_keys_by_arm[1:]:
            assert other == contained_keys_by_arm[0]


@pytest.mark.parametrize(
    "defect",
    [
        "duplicate_cell",
        "missing_cell",
        "string_flag",
        "wrong_outcome_recovered_but_false",
        "wrong_outcome_unrecovered_but_true",
        "unexpected_unrecovered_outcome",
        "late_recovery_over_300s",
        "premature_deadline_under_300s",
        "nan_wall_time",
        "negative_wall_time",
        "unmeasured_containment",
        "containment_key_mismatch",
        "wrong_cell_count",
    ],
)
def test_reject_defective_figure_inputs(defect):
    """Ensure malicious or malformed cells fail closed."""
    data = json.loads(SOURCE.read_text())
    cells = copy.deepcopy(data["cells"])

    if defect == "duplicate_cell":
        cells[1] = copy.deepcopy(cells[0])
    elif defect == "missing_cell":
        cells.pop()
    elif defect == "string_flag":
        cells[0]["assigned_verified_recovery"] = "true"
    elif defect == "wrong_outcome_recovered_but_false":
        cells[0]["assigned_verified_recovery"] = False
        cells[0]["outcome"] = "recovered"
    elif defect == "wrong_outcome_unrecovered_but_true":
        # Find a recovered cell and change outcome to global_deadline
        rec_cell = next(c for c in cells if c["assigned_verified_recovery"])
        rec_cell["outcome"] = "global_deadline"
    elif defect == "unexpected_unrecovered_outcome":
        unrec_cell = next(c for c in cells if not c["assigned_verified_recovery"])
        unrec_cell["outcome"] = "internal_solver_crash"
    elif defect == "late_recovery_over_300s":
        rec_cell = next(c for c in cells if c["assigned_verified_recovery"])
        rec_cell["parent_wall_seconds"] = 300.05
    elif defect == "premature_deadline_under_300s":
        deadline_cell = next(
            c for c in cells if c["outcome"] == "global_deadline"
        )
        deadline_cell["parent_wall_seconds"] = 150.0
    elif defect == "nan_wall_time":
        cells[0]["parent_wall_seconds"] = float("nan")
    elif defect == "negative_wall_time":
        cells[0]["parent_wall_seconds"] = -10.0
    elif defect == "unmeasured_containment":
        cells[0]["queue_containment_measured"] = False
    elif defect == "containment_key_mismatch":
        # Break containment parity by removing a key from containment
        cells[0]["true_tuple_queue_index_zero_based"] = None
    elif defect == "wrong_cell_count":
        cells = cells[:64]  # accidental N=8 input

    with pytest.raises(ValueError):
        summarize_cells(cells)


def test_corrupted_sidecar_or_hash_rejected(tmp_path):
    """Ensure missing or corrupted sidecars fail closed."""
    target = tmp_path / SOURCE.name
    target.write_bytes(SOURCE.read_bytes())
    index_target = tmp_path / "evidence_index_v1.json"
    index_target.write_text("{}")

    # Corrupted sidecar
    sidecar = target.with_name(target.name + ".sha256")
    sidecar.write_text("0" * 64 + "  " + target.name + "\n")
    with pytest.raises(ValueError, match="sidecar"):
        checked_input(target)

    # Missing sidecar
    sidecar.unlink()
    with pytest.raises(ValueError, match="Missing SHA-256 sidecar"):
        checked_input(target)


def test_generate_end_to_end_in_tmp_path(tmp_path):
    """Verify end-to-end figure generation, file formats, and sidecars."""
    receipt = generate(SOURCE, tmp_path, stem="test_fig7_n32")
    assert receipt["canonical_study"] == "D-056/D-057"

    pdf_file = tmp_path / "test_fig7_n32.pdf"
    png_file = tmp_path / "test_fig7_n32.png"
    csv_file = tmp_path / "test_fig7_n32.csv"
    json_file = tmp_path / "test_fig7_n32.json"

    for f in (pdf_file, png_file, csv_file, json_file):
        assert f.is_file(), f"Missing output file: {f}"
        sidecar = f.with_name(f.name + ".sha256")
        assert sidecar.is_file(), f"Missing sidecar: {sidecar}"
        actual_digest = hashlib.sha256(f.read_bytes()).hexdigest()
        parts = sidecar.read_text().strip().split()
        assert parts == [actual_digest, f.name]

    # Verify CSV line endings are LF
    raw_csv = csv_file.read_bytes()
    assert b"\r\n" not in raw_csv
    assert b"\n" in raw_csv

    # Verify CSV rows
    with csv_file.open("r", newline="") as h:
        rows = list(csv.DictReader(h))
    assert len(rows) == 8
    assert [int(r["n"]) for r in rows] == [32] * 8
    assert [int(r["verified_recoveries"]) for r in rows] == [
        4,
        18,
        2,
        19,
        23,
        27,
        3,
        22,
    ]
