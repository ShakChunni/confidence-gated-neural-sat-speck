"""Analysis of a synthetic fake campaign ledger (protocol sections 7-8)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from neural_sat_speck.backend.artifacts import cipher_functions
from neural_sat_speck.backend.guard import atomic_json
from neural_sat_speck.followups import ranking_depth_control as control

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "analyze_prospective_queue_cost.py"
EVIDENCE = REPO / "docs" / "queue_cost_model_evidence_20261009"


def _analysis():
    spec = importlib.util.spec_from_file_location(
        "analyze_prospective_queue_cost", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


an = _analysis()

# Recovery iff key_index < count: a prefix pattern makes every statistic hand-checkable.
PATTERN = {
    "S-G": 20,
    "S-X": 22,
    "S-G-c": 3,
    "S-X-c": 19,
    "S-C-G": 10,
    "S-C-X": 12,
    "M-G": 25,
    "M-X": 20,
    "M-G-c": 21,
    "M-X-c": 9,
}
PREP_SECONDS = 40.0


def _write_campaign(root: Path) -> Path:
    cells = control.prospective_cells("full")
    inputs = root / "inputs"
    cases = {}
    for n, case in enumerate(sorted({c["case_id"] for c in cells})):
        master = [n + 1, 2 * n + 3, 5, 7]
        truth = inputs / f"{case}_truth.json"
        atomic_json(truth, {"master_key": master, "verification_pairs": []})
        cases[case] = {"truth": {"path": str(truth)}}
    manifest = inputs / "manifest.json"
    atomic_json(manifest, {"mode": "full", "cells": cells, "cases": cases})
    records = {}
    for index, cell in enumerate(cells):
        label = an.ARM_LABELS[an.arm_key(cell)]
        recovered = cell["key_index"] < PATTERN[label]
        master = cases[cell["case_id"]]
        _, expand = cipher_functions(cell["cipher"])
        key = json.loads(Path(master["truth"]["path"]).read_text())["master_key"]
        truth = list(expand(tuple(key), cell["rounds"])[-2:])
        decoys = [[(7 * j + 1) % 65536, (11 * j + 3) % 65536] for j in range(3)]
        position = cell["key_index"] % 4
        queue = decoys[:position] + [truth] + decoys[position:]
        parents = [truth[-1]] + [w for w in range(65536) if w != truth[-1]][:31]
        directory = root / "cells" / cell["cell_id"] / f"{index:032x}"
        atomic_json(directory / "stage1.json", {"stage1_parents": parents})
        atomic_json(directory / "queue.json", {"queue": queue})
        start = 100.0 + index
        attempt = directory / "attempts" / "0000"
        atomic_json(
            attempt / "native_process.json",
            {
                "pid": 1000 + index,
                "process_group": 1,
                "argv": [],
                "started_monotonic": start + PREP_SECONDS,
            },
        )
        atomic_json(
            attempt / "attempt.json",
            {
                "status": "capped"
                if cell["candidate_cap_seconds"] is not None
                else "unsat"
            },
        )
        records[cell["cell_id"]] = {
            "cell": cell,
            "directory": str(directory),
            "outcome": "recovered" if recovered else "queue_exhausted",
            "started_monotonic": start,
            "parent_wall_seconds": 10.0 if recovered else 300.0,
        }
    atomic_json(
        root / "ledger.json",
        {"mode": "full", "manifest_path": str(manifest), "records": records},
    )
    return root


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    return _write_campaign(tmp_path_factory.mktemp("fake_campaign"))


@pytest.fixture(scope="module")
def evidence():
    return an.load_dataset(EVIDENCE)


@pytest.fixture(scope="module")
def result(campaign, evidence):
    return an.analyze(an.load_campaign_rows(campaign), evidence)


def test_counts_frozen_intervals_and_directional_checks_follow_the_pattern(result):
    counts = result["counts"]
    for label, expected in PATTERN.items():
        assert counts[label]["recovered"] == expected
        assert counts[label]["of"] == 32
        assert (
            counts[label]["contained"] == 32
        )  # the true final subkey is always a parent here
        par2 = (expected * 10.0 + (32 - expected) * 600.0) / 32
        assert counts[label]["par2_mean_seconds"] == pytest.approx(par2)
        assert counts[label]["median_prep_seconds"] == PREP_SECONDS
    assert counts["S-X-c"]["capped_attempts"] == 32
    assert counts["S-C-G"]["capped_attempts"] == 0
    assert all(v["inside"] for v in result["frozen_interval_check"].values())
    assert (
        result["directional_checks"]["speck_capped_S-X-c_minus_S-G-c"]["holds"] is True
    )
    assert (
        result["directional_checks"]["simon_capped_M-G-c_minus_M-X-c"]["holds"] is True
    )
    assert result["primary_model_test"]["verdict"] == "PASS"


def test_classical_contrasts_use_exact_mcnemar_holm_and_newcombe(result):
    tests = {(t["a"], t["b"]): t for t in result["classical_comparison_D061"]["tests"]}
    assert result["classical_comparison_D061"]["holm_family_size"] == 2
    first = tests[("S-C-G", "S-G")]
    assert (first["n10"], first["n01"]) == (0, 10)
    assert first["p_raw"] == pytest.approx(2**-9)
    assert first["p_holm"] == pytest.approx(2**-8)
    assert first["difference_a_minus_b"] == pytest.approx(-10 / 32)
    lower, upper = first["newcombe_95"]
    assert lower <= first["difference_a_minus_b"] <= upper
    assert tests[("S-C-X", "S-X")]["p_holm"] == pytest.approx(2**-8)


def test_model_contrasts_family_has_six_tests_with_holm(result):
    family = result["model_contrasts_D060"]
    assert family["holm_family_size"] == 6
    tests = {(t["a"], t["b"]): t for t in family["tests"]}
    speck = tests[("S-X-c", "S-G-c")]
    assert (speck["n10"], speck["n01"]) == (16, 0)
    assert speck["p_raw"] == pytest.approx(2**-15)
    assert all(0 <= t["p_holm"] <= 1 for t in family["tests"])
    assert speck["p_holm"] >= speck["p_raw"]


def test_d056_replication_is_descriptive_with_exact_intervals(result):
    for label, row in result["d056_replication_descriptive"].items():
        lower, upper = row["exact_95"]
        assert row["of"] == 32
        assert lower <= row["recovered"] / 32 <= upper


def test_conditional_model_test_scores_all_256_neural_cells(result):
    conditional = result["conditional_model_test"]
    assert conditional["cells"] == 256
    for key in (
        "accuracy_model",
        "accuracy_recovered_if_contained",
        "accuracy_arm_majority",
    ):
        assert 0.0 <= conditional[key] <= 1.0
    assert conditional["verdict"] in {"PASS", "FAIL"}


@pytest.mark.parametrize(
    ("inside", "holds", "verdict"),
    [
        (8, (True, True), "PASS"),
        (7, (True, True), "PASS"),
        (8, (True, False), "PARTIAL"),
        (6, (True, True), "PARTIAL"),
        (5, (False, False), "PARTIAL"),
        (4, (True, True), "FAIL"),
        (8, (False, False), "FAIL"),
    ],
)
def test_primary_verdict_follows_the_frozen_rules(inside, holds, verdict):
    frozen = {f"a{i}": {"inside": i < inside} for i in range(8)}
    directional = {f"d{i}": {"holds": h} for i, h in enumerate(holds)}
    assert an.primary_verdict(frozen, directional) == verdict


def test_holm_and_exact_mcnemar_hand_values():
    assert an.mcnemar_exact(0, 10) == pytest.approx(2**-9)
    assert an.mcnemar_exact(3, 3) == 1.0
    assert an.mcnemar_exact(0, 0) == 1.0
    assert an.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_newcombe_is_zero_width_under_perfect_concordance():
    lower, upper = an.newcombe_paired(10, 0, 0, 10)
    assert lower == pytest.approx(0.0, abs=1e-12)
    assert upper == pytest.approx(0.0, abs=1e-12)


def test_refuses_fewer_than_320_cells(campaign, evidence):
    rows = an.load_campaign_rows(campaign)
    with pytest.raises(ValueError, match="expected 320"):
        an.analyze(rows[:-1], evidence)


def test_refuses_a_non_terminal_cell(campaign, evidence):
    rows = an.load_campaign_rows(campaign)
    rows[5] = {**rows[5], "outcome": "unavailable"}
    with pytest.raises(ValueError, match="not terminal"):
        an.analyze(rows, evidence)


def test_refuses_a_cell_outside_the_design(campaign, evidence):
    rows = an.load_campaign_rows(campaign)
    rows[0] = {**rows[0], "profile": "cms_native_default"}
    with pytest.raises(ValueError, match="differs from the prospective design"):
        an.analyze(rows, evidence)


def test_main_writes_once_and_refuses_overwrite(campaign, tmp_path):
    output = tmp_path / "analysis.json"
    an.main(["--campaign-dir", str(campaign), "--output", str(output)])
    written = json.loads(output.read_text())
    assert written["primary_model_test"]["verdict"] == "PASS"
    with pytest.raises(SystemExit):
        an.main(["--campaign-dir", str(campaign), "--output", str(output)])


def test_main_refuses_a_campaign_that_is_not_complete(campaign, tmp_path):
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    ledger = json.loads((campaign / "ledger.json").read_text())
    first = next(iter(ledger["records"]))
    ledger["records"][first]["outcome"] = "worker_exception"
    atomic_json(incomplete / "ledger.json", ledger)
    with pytest.raises(SystemExit, match="not terminal"):
        an.main(["--campaign-dir", str(incomplete)])
