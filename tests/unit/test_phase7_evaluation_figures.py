"""Tests for Phase 7 final comparative evaluation figures and LaTeX tables."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts/plot_phase7_evaluation_figures.py"
_PRIMARY_SUMMARY = REPOSITORY_ROOT / "results/phase7_final_evaluation/summary.json"
_FIXTURE_SUMMARY = REPOSITORY_ROOT / "tests/fixtures/phase7_summary.json"
SUMMARY_PATH = _PRIMARY_SUMMARY if _PRIMARY_SUMMARY.exists() else _FIXTURE_SUMMARY


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("phase7_plots", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load Phase 7 plot generator")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_phase7_summary_is_complete_and_has_all_five_methods() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    assert summary["instance_count"] == 64
    assert summary["phase"] == 7
    methods = summary["methods"]
    assert set(methods.keys()) == {
        "sat_only",
        "random_sat",
        "b0_sat",
        "b1_sat",
        "gated_b1_fallback_sat",
    }
    assert methods["b1_sat"]["success_count"] == 58
    assert methods["gated_b1_fallback_sat"]["success_count"] == 63


def test_phase7_figures_render_valid_svg_strings() -> None:
    mod = _module()
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    svg1 = mod._figure_success_rates(summary)
    assert svg1.startswith("<svg")
    assert "Comparative Key Recovery Success Rate" in svg1
    assert "90.6%" in svg1 or "90.62%" in svg1
    assert "98.4%" in svg1 or "98.44%" in svg1

    svg2 = mod._figure_ecdf_attempts(summary, [])
    assert svg2.startswith("<svg")
    assert "Cumulative Master Keys Solved" in svg2

    svg3 = mod._figure_runtime_par2(summary)
    assert svg3.startswith("<svg")
    assert "Median Online Runtime and PAR-2" in svg3


def test_phase7_latex_and_markdown_tables_are_valid() -> None:
    mod = _module()
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    latex = mod._generate_latex_table(summary)
    assert r"\begin{table}" in latex
    assert r"\end{table}" in latex
    assert "58/64" in latex
    assert "63/64" in latex

    md = mod._generate_markdown_table(summary)
    assert "# Phase 7 Comparative Evaluation Results" in md
    assert "90.62%" in md
    assert "98.44%" in md
