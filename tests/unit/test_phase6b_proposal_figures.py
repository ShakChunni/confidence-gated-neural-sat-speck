"""Tests for proposal figures derived from the frozen Phase 6B summary."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts/plot_phase6b_proposal_figures.py"
SUMMARY_PATH = (
    REPOSITORY_ROOT
    / "results/phase6b_h1_adjudication/eager/summary.json"
)
FIGURES_DIR = REPOSITORY_ROOT / "results/figures/proposal"


def _figures() -> ModuleType:
    spec = importlib.util.spec_from_file_location("phase6b_proposal_figures", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load the Phase 6B figure generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase6b_summary_is_validation_only_and_complete() -> None:
    module = _figures()
    summary = module._load_summary(SUMMARY_PATH)

    assert summary["instance_count"] == 64
    assert summary["validation_only"] is True
    assert summary["held_out_test_generated"] is False
    assert summary["methods"]["b1"]["top_k_recall"]["16"] == 62


def test_phase6b_result_figure_uses_selected_b1_metrics() -> None:
    module = _figures()
    summary = module._load_summary(SUMMARY_PATH)
    document = module._results_figure(summary)

    assert document.startswith("<svg")
    assert "Pair-Aware Ranking versus Baseline Distinguisher" in document
    assert "62/64" in document
    assert "22.10% reduction" in document
    assert "held-out attack test split remains strictly closed" in document


def test_phase6b_methodology_marks_planned_components() -> None:
    module = _figures()
    document = module._methodology_figure()

    assert "Confidence estimation" in document
    assert "Safer fallback" in document
    assert "Incremental CryptoMiniSat" in document
    assert "Sound master-key verification" in document
    assert 'stroke-dasharray="7 5"' in document


def test_all_eight_proposal_figures_generate_valid_svg() -> None:
    module = _figures()
    summary = module._load_summary(SUMMARY_PATH)

    figures = {
        "methodology_pipeline.svg": module._methodology_figure(),
        "b1_pair_aware_architecture.svg": module._b1_architecture_figure(),
        "b0_b1_prospective_results.svg": module._results_figure(summary),
        "validation_protocol.svg": module._validation_protocol_figure(),
        "research_positioning.svg": module._positioning_figure(),
        "exact_recovery_verification.svg": module._exact_recovery_figure(),
        "confidence_fallback_mechanism.svg": module._confidence_fallback_figure(),
        "evaluation_matrix.svg": module._evaluation_matrix_figure(),
    }

    assert len(figures) == 8
    for name, content in figures.items():
        assert content.startswith("<svg"), f"{name} should start with <svg"
        assert content.endswith("</svg>\n"), f"{name} should end with </svg>"
        assert "Phase 6" not in content, f"{name} should not contain internal phase names"


def test_proposal_figures_exist_in_all_three_formats() -> None:
    expected_stems = [
        "methodology_pipeline",
        "b1_pair_aware_architecture",
        "b0_b1_prospective_results",
        "validation_protocol",
        "research_positioning",
        "exact_recovery_verification",
        "confidence_fallback_mechanism",
        "evaluation_matrix",
    ]

    for stem in expected_stems:
        svg_file = FIGURES_DIR / f"{stem}.svg"
        png_file = FIGURES_DIR / f"{stem}.png"
        pdf_file = FIGURES_DIR / f"{stem}.pdf"

        assert svg_file.exists(), f"Missing {svg_file}"
        assert png_file.exists(), f"Missing {png_file}"
        assert pdf_file.exists(), f"Missing {pdf_file}"

        assert svg_file.stat().st_size > 0
        assert png_file.stat().st_size > 0
        assert pdf_file.stat().st_size > 0


def test_phase6b_figure_manifest_is_reproducible(tmp_path: Path) -> None:
    module = _figures()
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    first = module._results_figure(summary)
    second = module._results_figure(summary)
    path = tmp_path / "figure.svg"
    path.write_text(first, encoding="utf-8")

    assert first == second
    assert path.read_text(encoding="utf-8") == first
