"""Protect figure exports against false positives, broken pairing and stale stats."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/generate_final_publication_figures.py"
)
SPEC = importlib.util.spec_from_file_location("final_figures", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
FIGURES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIGURES)


def attack_fixture(cipher="simon32_64"):
    rows = []
    for kid in range(2):
        row = {
            "method": "fixture",
            "instance_index": kid,
            "instance_id": f"simon32_64_confirmation_r10_{kid:04d}",
            "outcome": "verified_recovery" if kid == 0 else "global_deadline",
            "verified_on_fresh_pairs": kid == 0,
            "complete_online_wall_seconds": 1.0 if kid == 0 else 300.01,
        }
        if cipher == "speck32_64":
            row.update(
                instance_id=f"validation-{kid:04d}",
                success=kid == 0,
                counts_as_verified_recovery=kid == 0,
                late_verified_recovery=False,
                status="verified_recovery" if kid == 0 else "global_timeout",
                online_wall_seconds=1.0 if kid == 0 else 302.0,
            )
        rows.append(row)
    return (
        {"records_by_method": {"fixture": rows}}
        if cipher == "speck32_64"
        else {"records": rows}
    )


def test_source_hash_mismatch_fails_closed(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    source.write_text("{}")
    monkeypatch.setattr(FIGURES, "SOURCE_PATHS", {"fixture": source})
    monkeypatch.setattr(FIGURES, "EXPECTED_HASHES", {"fixture": "0" * 64})
    with pytest.raises(ValueError, match="hash mismatch"):
        FIGURES.verify_source_hashes()


@pytest.mark.parametrize("cipher", ["speck32_64", "simon32_64"])
def test_attack_duplicate_or_missing_pair_rejected(cipher):
    raw = attack_fixture(cipher)
    rows = (
        raw["records"]
        if cipher == "simon32_64"
        else raw["records_by_method"]["fixture"]
    )
    rows[1] = copy.deepcopy(rows[0])
    with pytest.raises(ValueError, match="Duplicate"):
        FIGURES.validate_attack_records(raw, cipher, ["fixture"], 2)
    rows.pop()
    with pytest.raises(ValueError, match="Missing"):
        FIGURES.validate_attack_records(raw, cipher, ["fixture"], 2)


@pytest.mark.parametrize(
    "field,value",
    [
        ("verified_on_fresh_pairs", "false"),
        ("verified_on_fresh_pairs", False),
        ("complete_online_wall_seconds", 0),
        ("complete_online_wall_seconds", float("nan")),
        ("complete_online_wall_seconds", 301),
        ("instance_id", "another_key"),
        ("outcome", "fabricated"),
    ],
)
def test_attack_contradictions_rejected(field, value):
    raw = attack_fixture()
    raw["records"][0][field] = value
    with pytest.raises(ValueError):
        FIGURES.validate_attack_records(raw, "simon32_64", ["fixture"], 2)


@pytest.mark.parametrize(
    "field,value",
    [
        ("counts_as_verified_recovery", "false"),
        ("success", False),
        ("late_verified_recovery", True),
    ],
)
def test_speck_boolean_and_late_flags_rejected(field, value):
    raw = attack_fixture("speck32_64")
    raw["records_by_method"]["fixture"][0][field] = value
    with pytest.raises(ValueError):
        FIGURES.validate_attack_records(raw, "speck32_64", ["fixture"], 2)


@pytest.mark.parametrize(
    "counts,expected",
    [
        ((14, 5, 0, 22), (0.011483, 0.226493)),
        ((20, 12, 2, 16), (0.0562, 0.3292)),
    ],
)
def test_paired_intervals_match_literature_fixtures(counts, expected):
    assert FIGURES.newcombe_method10_interval(*counts) == pytest.approx(
        expected, abs=5e-5
    )


def test_failure_matrix_rejects_wrong_total_and_duplicate():
    arm = [("fixture", "Fixture", "#666666")]
    row = {
        "study": "fixture",
        "role": "autonomous_attack",
        "method": "fixture",
        "outcome_category": "verified_recovery",
        "count": "1",
        "n_assigned": "2",
    }
    with pytest.raises(ValueError, match="sum to N"):
        FIGURES._category_counts(
            [row], "fixture", "autonomous_attack", arm, FIGURES._SPECK_RAW
        )
    row["count"] = "2"
    with pytest.raises(ValueError, match="Duplicate"):
        FIGURES._category_counts(
            [row, row], "fixture", "autonomous_attack", arm, FIGURES._SPECK_RAW
        )


@pytest.fixture(scope="module")
def actual_tables(tmp_path_factory):
    FIGURES.verify_source_hashes()
    outputs = []
    for index in range(2):
        out = tmp_path_factory.mktemp(f"figure_tables_{index}")
        FIGURES.build_attack_tables(
            FIGURES.SOURCE_PATHS["speck_raw"], FIGURES.SOURCE_PATHS["simon_raw"], out
        )
        FIGURES.reconcile_attack_summary(out / "attack_summary.csv")
        FIGURES.reconcile_attack_contrasts(out / "attack_paired_contrasts.csv")
        FIGURES.build_depth_tables(
            FIGURES.SOURCE_PATHS["depth_raw"],
            FIGURES.SOURCE_PATHS["depth_analysis_v2"],
            out,
        )
        FIGURES.build_backend_tables(
            FIGURES.SOURCE_PATHS["backend_raw"],
            FIGURES.SOURCE_PATHS["backend_analysis"],
            out,
        )
        outputs.append(out)
    return outputs


def test_actual_table_totals_provenance_and_replay(actual_tables):
    first, second = actual_tables
    counts = {
        "attack_cells.csv": 1344,
        "attack_summary.csv": 15,
        "depth_cells.csv": 192,
        "depth_summary.csv": 24,
        "backend_cells.csv": 160,
        "backend_summary.csv": 40,
    }
    for path in first.glob("*.csv"):
        assert path.read_bytes() == (second / path.name).read_bytes()
        rows = FIGURES._read_csv(path)
        assert all(len(r["source_1_sha256"]) == 64 and r["source_1_path"] for r in rows)
        if path.name in counts:
            assert len(rows) == counts[path.name]


def test_recorded_no_entry_is_zero_but_depth_timeout_is_unavailable(actual_tables):
    path = actual_tables[0]
    rows = FIGURES._read_csv(path / "attack_cells.csv")
    no_entry = [r for r in rows if r["solver_seconds_status"] == "not_entered"]
    assert len(no_entry) == 128 + 64 + 7
    assert all(
        float(r["solver_seconds"]) == 0 and int(r["candidate_attempts"]) == 0
        for r in no_entry
    )
    assert any(
        not r["verified_recovery"] == "True" and float(r["solver_seconds"]) > 0
        for r in rows
        if r["solver_seconds"]
    )
    depth = FIGURES._read_csv(path / "depth_cells.csv")
    unavailable = [r for r in depth if r["solve_duration_available"] == "False"]
    assert len(unavailable) == 41
    assert all(r["solve_duration_seconds"] == "" for r in unavailable)


def test_corrected_paired_ci_and_frozen_holm_families(actual_tables):
    rows = FIGURES._read_csv(actual_tables[0] / "attack_paired_contrasts.csv")
    row = next(
        r
        for r in rows
        if r["method_a"] == "fully_classical_8parent"
        and r["method_b"] == "neural_classical_hybrid_8parent"
    )
    assert (
        float(row["newcombe_95_lower"]),
        float(row["newcombe_95_upper"]),
    ) == pytest.approx((-0.039898181, 0.166395798), abs=1e-6)
    primaries = [r for r in rows if r["multiplicity_family"] == "simon_primary"]
    assert len(primaries) == 2
    assert float(primaries[0]["holm_adjusted_p"]) == pytest.approx(
        4.440892098500626e-16, rel=1e-10, abs=0
    )
    assert len([r for r in rows if r["multiplicity_family"] == "speck_secondary"]) == 3


def test_backend_interruption_excluded_only_from_affected_pairs(actual_tables):
    out = actual_tables[0]
    pairs = FIGURES._read_csv(out / "backend_paired_contrasts.csv")
    for row in pairs:
        affected = row["condition"] == "simon32_64_o128_m0" and "cms_wrapper" in (
            row["profile_a"],
            row["profile_b"],
        )
        assert int(row["valid_pair_count"]) == (3 if affected else 4)
        assert sum(
            int(row[f]) for f in ("n_both_succeed", "n_a_only", "n_b_only", "n_neither")
        ) == int(row["valid_pair_count"])
    rows = FIGURES._read_csv(out / "backend_cells.csv")
    interrupted = [r for r in rows if r["infrastructure_interrupted"] == "True"]
    assert len(interrupted) == 1
    assert interrupted[0]["online_seconds"] == ""
    assert float(interrupted[0]["observed_elapsed_seconds"]) == pytest.approx(90.566676)
    group = next(
        r
        for r in FIGURES._read_csv(out / "backend_summary.csv")
        if r["group_id"] == "simon32_64_o128_m0__cms_wrapper"
    )
    assert int(group["n_completed_bounded_trials"]) == 3
    assert float(group["median_available_online_seconds"]) > 300


@pytest.mark.parametrize(
    "kind,source", [("depth", "depth_raw"), ("backend", "backend_raw")]
)
def test_diagnostic_grids_reject_duplicates_and_bad_flags(kind, source):
    data = json.loads(FIGURES.SOURCE_PATHS[source].read_text())
    rows = list((data["cells"] if kind == "depth" else data["records"]).values())
    rows[1] = copy.deepcopy(rows[0])
    with pytest.raises(ValueError, match="Duplicate"):
        FIGURES.validate_diagnostic_records(rows, kind)
    rows.pop(1)
    rows[0]["verified_recovery" if kind == "depth" else "verified_fresh"] = "false"
    with pytest.raises(ValueError, match="boolean"):
        FIGURES.validate_diagnostic_records(rows, kind)


def test_main_figures_only_two_panels_and_paper_names(
    actual_tables, tmp_path, monkeypatch
):
    captured = []

    def capture(fig, out, name):
        captured.append(fig)
        return out / f"{name}.pdf", out / f"{name}.png"

    monkeypatch.setattr(FIGURES, "save_figure_dual", capture)
    for fn, table in (
        (FIGURES.plot_fig1_attack_recovery, "attack_summary.csv"),
        (FIGURES.plot_fig4_failure_decomposition, "attack_failures.csv"),
    ):
        fn(actual_tables[0] / table, tmp_path)
        fig = captured[-1]
        assert len(fig.axes) == 2
        assert fig.axes[0].get_title(loc="left").startswith("(a) SPECK")
        assert fig.axes[1].get_title(loc="left").startswith("(b) SIMON")
        labels = [t.get_text() for ax in fig.axes for t in ax.get_yticklabels()]
        assert "True-tuple diagnostic" not in labels
        assert all("_" not in label and "pairs" not in label for label in labels)
        FIGURES.plt.close(fig)
