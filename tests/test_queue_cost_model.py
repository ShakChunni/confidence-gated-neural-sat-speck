"""Focused tests for the N=32 queue-cost model (scripts/fit_queue_cost_model.py)."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIT_SCRIPT = REPO_ROOT / "scripts" / "fit_queue_cost_model.py"
EVIDENCE_DIR = REPO_ROOT / "docs" / "queue_cost_model_evidence_20261009"


def _load_fit_module():
    spec = importlib.util.spec_from_file_location("fit_queue_cost_model", FIT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve string annotations through sys.modules, so register first.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fq = _load_fit_module()


class _FixedUniforms:
    """Stand-in for a Generator whose random() returns preset values."""

    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    def random(self, size):
        return self._values


def _degenerate(value, n=3, censored=False):
    """Kaplan-Meier with n identical observations, all events unless censored."""
    return fq.KM([(value, censored)] * n)


# ---- Kaplan-Meier ---------------------------------------------------------


def test_km_without_censoring_matches_hand_computation():
    km = fq.KM([(1.0, False), (2.0, False), (3.0, False), (4.0, False)])
    assert km.n == 4
    assert km.ncens == 0
    assert list(km.t) == [1.0, 2.0, 3.0, 4.0]
    assert km.F == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert km.quantile(0.4) == 2.0
    assert km.quantile(0.25) == 1.0
    assert km.surv_at(2.5) == pytest.approx(0.5)


def test_km_right_censoring_shrinks_the_risk_set():
    # At t=2 the censored attempt leaves the risk set without an event.
    km = fq.KM([(1.0, False), (2.0, True), (3.0, False), (4.0, False)])
    assert km.ncens == 1
    assert list(km.t) == [1.0, 3.0, 4.0]
    assert km.F == pytest.approx([0.25, 0.625, 1.0])
    assert km.quantile(0.5) == 3.0
    assert km.surv_at(2.5) == pytest.approx(0.75)


def test_km_censored_tie_with_event_stays_at_risk():
    # Two observations at t=2 (one event, one censored): risk set is 3, so S=2/3.
    km = fq.KM([(2.0, False), (2.0, True), (5.0, False)])
    assert list(km.t) == [2.0, 5.0]
    assert km.F == pytest.approx([1 / 3, 1.0])


def test_km_all_censored_samples_infinity():
    km = fq.KM([(1.0, True), (2.0, True)])
    assert km.t.size == 0
    assert km.quantile(0.5) == math.inf
    assert km.surv_at(1.5) == 1.0
    draws = km.sample(np.random.default_rng(0), 50)
    assert draws.shape == (50,)
    assert np.all(np.isinf(draws))


def test_km_quantile_is_infinite_when_median_is_not_reached():
    km = fq.KM([(1.0, False), (2.0, True), (3.0, True), (4.0, True)])
    assert km.F == pytest.approx([0.25])
    assert km.quantile(0.5) == math.inf
    assert km.quantile(0.25) == 1.0


def test_km_sample_maps_uniforms_through_the_cdf():
    km = fq.KM([(1.0, False), (2.0, True), (3.0, True), (4.0, True)])
    draws = km.sample(_FixedUniforms([0.1, 0.9]), 2)
    assert draws[0] == 1.0
    assert math.isinf(draws[1])


# ---- p_recover and the cap ------------------------------------------------


def test_p_recover_is_zero_for_absent_candidate():
    rng = np.random.default_rng(1)
    got = fq.p_recover(
        0.0, None, _degenerate(1.0), _degenerate(1.0), np.array([0.0]), rng
    )
    assert got == 0.0


def test_p_recover_is_one_when_clearly_within_budget():
    rng = np.random.default_rng(1)
    rkm, skm, oarr = _degenerate(1.0), _degenerate(5.0), np.array([0.0])
    # 10 + 3 * 1 + 5 = 18 s, far below the 300 s budget.
    assert fq.p_recover(10.0, 3, rkm, skm, oarr, rng) == 1.0
    assert fq.p_recover(10.0, 3, rkm, skm, oarr, rng, tau=10.0) == 1.0


def test_p_recover_is_zero_when_preparation_alone_exceeds_budget():
    rng = np.random.default_rng(1)
    got = fq.p_recover(
        400.0, 0, _degenerate(1.0), _degenerate(5.0), np.array([0.0]), rng
    )
    assert got == 0.0


def test_true_candidate_slower_than_cap_is_lost():
    rng = np.random.default_rng(2)
    rkm, skm, oarr = _degenerate(1.0), _degenerate(20.0), np.array([0.0])
    assert fq.p_recover(0.0, 0, rkm, skm, oarr, rng) == 1.0
    assert fq.p_recover(0.0, 0, rkm, skm, oarr, rng, tau=10.0) == 0.0


def test_wrong_candidate_costs_are_truncated_at_tau():
    rng = np.random.default_rng(3)
    rkm, skm, oarr = _degenerate(1000.0), _degenerate(1.0), np.array([0.0])
    assert fq.p_recover(0.0, 1, rkm, skm, oarr, rng) == 0.0
    # Truncated wrong cost is 5 s, so 5 + 1 = 6 s is within budget.
    assert fq.p_recover(0.0, 1, rkm, skm, oarr, rng, tau=5.0) == 1.0


# ---- leave-one-key-out ----------------------------------------------------


def _two_key_dataset():
    arm = ("speck32_64", 1, "cms_native_gauss")
    cells = [
        dict(
            cell_id="a",
            cipher="speck32_64",
            key_index="00",
            depth="1",
            profile="cms_native_gauss",
            prep_seconds="0.0",
            true_queue_position="3",
            recovered="True",
        ),
        dict(
            cell_id="b",
            cipher="speck32_64",
            key_index="01",
            depth="1",
            profile="cms_native_gauss",
            prep_seconds="0.0",
            true_queue_position="3",
            recovered="True",
        ),
    ]
    obs = {
        "a": dict(r=[(1.0, False)], s=[(2.0, False)], o=[]),
        "b": dict(r=[(9.0, False)], s=[(8.0, False)], o=[0.5]),
    }
    ds = fq.Dataset(cells=cells, by_cell={}, obs=obs, arms=[arm])
    return ds, arm


def test_leave_one_key_out_excludes_that_key():
    ds, arm = _two_key_dataset()
    assert fq.pooled(ds, arm) == [(1.0, False), (9.0, False)]
    assert fq.pooled(ds, arm, exclude_key="00") == [(9.0, False)]

    rkm, skm, oarr = fq.dists(ds, arm, exclude_key="00")
    assert list(rkm.t) == [9.0]
    assert list(skm.t) == [8.0]
    assert list(oarr) == [0.5]


def test_leave_one_key_out_falls_back_to_zero_overhead_when_none_left():
    ds, arm = _two_key_dataset()
    rkm, _, oarr = fq.dists(ds, arm, exclude_key="01")
    assert list(rkm.t) == [1.0]
    assert list(oarr) == [0.0]


# ---- evidence (skipped when the evidence directory is absent) -------------


@pytest.mark.skipif(not EVIDENCE_DIR.is_dir(), reason="evidence directory not present")
def test_evidence_predictions_and_summary_match_audit_counts():
    with open(EVIDENCE_DIR / "predictions.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 256
    summary = json.loads((EVIDENCE_DIR / "summary.json").read_text(encoding="utf-8"))
    assert summary["overall"]["pred_primary"]["correct"] == 242
