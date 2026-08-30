"""Protocol and resumability checks for the Phase 6C-A runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from neural_sat_speck.attacks.ranking import ALL_FINAL_ROUND_SUBKEYS

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs/experiments/phase6c_confidence_fallback_r6.json"
RUNNER_PATH = REPOSITORY_ROOT / "scripts/run_phase6c_confidence_fallback.py"


def _runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("phase6c_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load Phase 6C runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_phase6c_config_is_frozen_validation_only_and_seed_disjoint() -> None:
    runner = _runner()
    config = _config()
    runner._validate_config(config)
    dataset = config["dataset"]
    assert dataset["seed"] == 97001
    assert dataset["seed"] not in config["seed_exclusions"]
    assert dataset["seed"] not in runner.EXISTING_DATA_SEEDS
    assert config["validation_only"] is True
    assert config["held_out_test_generated"] is False
    assert "test" not in dataset
    assert config["exact_recovery"]["sat_executed"] is False
    assert config["exact_recovery"]["verification_executed"] is False


def test_phase6c_feature_gate_and_fallback_budgets_are_frozen() -> None:
    config = _config()
    calibration = config["calibration"]
    fallback = config["fallback"]
    assert calibration["top_k"] == 8
    assert calibration["feature_formula"].endswith("max(score_std_over_all_65536_candidates, 1e-12)")
    assert calibration["threshold"] == {
        "risk_limit": 0.1,
        "minimum_accepted": 16,
        "accept_rule": "estimated_probability >= selected_threshold",
        "candidate_thresholds": "unique_calibrated_probabilities_descending",
        "tie_break": "maximum_accepted_count_then_lower_risk_then_higher_threshold",
    }
    assert fallback["trusted_budget"] == 8
    assert fallback["head_size"] == 16
    assert fallback["budget"] == 32
    assert config["ranking"]["b0_candidate_batch_size"] == 4096
    assert config["ranking"]["b1_candidate_batch_size"] == 4096


def test_phase6c_counterbalance_and_no_heldout_manifest_policy() -> None:
    runner = _runner()
    assert runner._counterbalanced_execution_order(0) == ("b0", "b1")
    assert runner._counterbalanced_execution_order(1) == ("b1", "b0")
    assert runner._counterbalanced_execution_order(63) == ("b1", "b0")
    with pytest.raises(SystemExit):
        altered = _config()
        altered["held_out_test_generated"] = True
        runner._validate_config(altered)


def test_phase6c_calibration_dry_run_uses_only_authoritative_64_records() -> None:
    runner = _runner()
    config = _config()
    calibration = config.get("calibration", {})
    manifest_path = runner._repository_path(
        str(calibration.get("source_manifest_path", ""))
    ).resolve()
    source_glob = runner._repository_path(
        str(calibration.get("source_result_glob", ""))
    ).resolve()
    records = list(source_glob.parent.glob(source_glob.name))
    if not manifest_path.is_file() or len(records) != int(calibration.get("source_record_count", 64)):
        pytest.skip("Phase 6B source manifest and records not present in clean CI environment")
    result, calibrator, threshold, observations = runner._fit_calibration(
        config,
        config_path=CONFIG_PATH,
    )
    assert len(observations) == 64
    assert sum(item.target_top8 for item in observations) == 41
    assert calibrator.fit_count == 64
    assert result["source"]["record_count"] == 64
    assert result["provenance"]["held_out_accessed"] is False
    assert result["metrics"]["sample_count"] == 64
    assert result["metrics"]["prevalence"] == pytest.approx(41 / 64)
    assert not threshold.feasible
    assert result["threshold_selection"]["threshold"] is None


def test_phase6c_atomic_write_and_resume_validation(tmp_path: Path) -> None:
    runner = _runner()
    config_path = CONFIG_PATH
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("placeholder\n", encoding="utf-8")
    zero_scores = [0.0] * len(ALL_FINAL_ROUND_SUBKEYS)
    ranking = {
        "candidate_values": list(ALL_FINAL_ROUND_SUBKEYS),
        "candidate_scores": zero_scores,
        "candidate_order": list(ALL_FINAL_ROUND_SUBKEYS),
    }
    record = {
        "schema": runner.RESULT_SCHEMA,
        "validation_only": True,
        "held_out_test_generated": False,
        "instance": {
            "index": 0,
            "split_role": "validation",
            "execution_order": ["b0", "b1"],
        },
        "dataset": {"manifest_sha256": runner._sha256_file(manifest_path)},
        "provenance": {
            "config_sha256": runner._sha256_file(config_path),
            "inference_backend": "eager_pytorch",
            "held_out_accessed": False,
            "peak_rss_bytes": 1,
        },
        "methods": {
            "b0": {"ranking": ranking},
            "b1": {"ranking": ranking},
        },
        "confidence": {"estimated_probability": 0.5},
        "route": {"route_id": "trusted_top8", "candidate_order": list(range(8))},
    }
    result_path = tmp_path / "ranking_0000.json"
    runner._write_atomic(result_path, record)
    restored = runner._load_json(result_path)
    runner._validate_saved_result(
        restored,
        index=0,
        config_path=config_path,
        manifest_path=manifest_path,
    )
