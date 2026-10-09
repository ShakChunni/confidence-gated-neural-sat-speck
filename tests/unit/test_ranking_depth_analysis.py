"""Independent arithmetic fixtures and pairing failures for cohort analysis."""

import copy
import math

import pytest

from scripts.analyze_backend_ranking_depth import summarize


def fixture(n=32):
    cells = []
    # Hand-specified nested recoveries make the gains +2/N versus +1/N.
    for cipher in ("speck32_64", "simon32_64"):
        for depth in (1, 2):
            for backend in ("cms_native_gauss", "cms_native_no_gauss"):
                count = 1 if depth == 1 else (3 if backend == "cms_native_gauss" else 2)
                for key in range(n):
                    recovered = key < count
                    cells.append(
                        {
                            "cipher": cipher,
                            "key_index": key,
                            "trailing_keys": depth,
                            "profile": backend,
                            "assigned_verified_recovery": recovered,
                            "outcome": "recovered" if recovered else "global_deadline",
                            "parent_wall_seconds": 100.0 if recovered else 300.01,
                            "par2_seconds": 100.0 if recovered else 600.0,
                            "queue_containment_measured": True,
                            "true_tuple_queue_index_zero_based": 0,
                            "true_tuple_dispatched": True,
                        }
                    )
    return cells


@pytest.mark.parametrize("n", [8, 32])
def test_counts_and_interaction_follow_actual_paired_cohort(n):
    summaries, interactions = summarize(fixture(n))
    assert all(s["n"] == n for s in summaries)
    assert [s["recovered"] for s in summaries] == [1, 1, 3, 2, 1, 1, 3, 2]
    for row in interactions:
        assert row["n"] == n
        assert row["recovery_interaction_G_minus_X"] == pytest.approx(1 / n)
        assert row["par2_interaction_G_minus_X_seconds"] == pytest.approx(-500 / n)
        assert row["within_backend"]["cms_native_gauss"]["two_only"] == 2
        assert row["within_backend"]["cms_native_no_gauss"]["two_only"] == 1


def test_input_order_does_not_change_pairing():
    cells = fixture()
    assert summarize(list(reversed(cells))) == summarize(cells)


def test_retry_sensitivity_removes_whole_key_from_every_arm():
    cells = fixture()
    retained = [c for c in cells if (c["cipher"], c["key_index"]) != ("speck32_64", 17)]
    summaries, interactions = summarize(retained)
    assert [r["n"] for r in interactions] == [31, 32]
    assert interactions[0]["recovery_interaction_G_minus_X"] == pytest.approx(1 / 31)
    assert [s["n"] for s in summaries] == [32] * 4 + [31] * 4


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "duplicate",
        "different_key",
        "string_flag",
        "nan",
        "late",
        "premature",
        "par2",
        "unknown_group",
    ],
)
def test_misleading_cells_are_rejected(defect):
    cells = copy.deepcopy(fixture())
    if defect == "missing":
        cells.pop()
    elif defect == "duplicate":
        cells.append(cells[0])
    elif defect == "different_key":
        cells[0]["key_index"] = 999
    elif defect == "string_flag":
        cells[0]["assigned_verified_recovery"] = "true"
    elif defect == "nan":
        cells[0]["parent_wall_seconds"] = math.nan
    elif defect == "late":
        cells[0]["parent_wall_seconds"] = cells[0]["par2_seconds"] = 300.1
    elif defect == "premature":
        cells[-1]["parent_wall_seconds"] = 100
    elif defect == "par2":
        cells[0]["par2_seconds"] = 600
    elif defect == "unknown_group":
        cells[0]["profile"] = "unknown"
    with pytest.raises(ValueError):
        summarize(cells)
