"""Tests for the frozen, validation-only Phase 6B H1 protocol."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs/experiments/phase6b_h1_adjudication.json"
RUNNER_PATH = REPOSITORY_ROOT / "scripts/run_phase6b_h1_adjudication.py"


def _runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("phase6b_h1_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load the Phase 6B H1 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_phase6b_config_passes_runner_validation() -> None:
    runner = _runner()
    runner._validate_config(_config())


def test_phase6b_execution_order_is_counterbalanced_by_index() -> None:
    runner = _runner()
    assert runner._counterbalanced_execution_order(0) == ("b0", "b1")
    assert runner._counterbalanced_execution_order(1) == ("b1", "b0")
    assert runner._counterbalanced_execution_order(63) == ("b1", "b0")


def test_phase6b_protocol_is_validation_only_and_seed_disjoint() -> None:
    config = _config()
    dataset = config["dataset"]
    assert isinstance(dataset, dict)
    assert dataset["split_role"] == "validation"
    assert dataset["seed"] == 96001
    assert dataset["seed"] not in config["seed_exclusions"]
    assert config["validation_only"] is True
    assert config["held_out_test_generated"] is False
    assert "test" not in config
    assert config["smoke"]["instance_count"] == 1
    assert config["smoke"]["b0_window_call_count_required"] == 16


def test_phase6b_method_and_budget_are_frozen() -> None:
    config = _config()
    methods = config["methods"]
    ranking = config["ranking"]
    assert methods["h1"]["method_id"] == "h1_b1_top16_b0_rerank"
    assert methods["h1"]["window_size"] == 16
    assert methods["h1"]["candidate_budget"] == 16
    assert methods["b0"]["score_aggregation"] == "sum_logits"
    assert methods["b1"]["score_aggregation"] == "group_logit"
    assert ranking["b1_order_is_input"] is True
    assert ranking["h1_tail_policy"] == "unchanged_b1_positions_17_through_65536"
    assert config["metrics"]["primary"] == "capped_attempts"
    assert config["metrics"]["miss_value"] == 17
    timing = config["timing_policy"]
    assert timing["single_process_sequential"] is True
    assert timing["concurrent_ranking_jobs"] is False
    assert timing["peak_rss_bytes_recorded"] is True
    assert config["runtime"] == {
        "device": "cpu",
        "torch_num_threads": 4,
        "torch_num_interop_threads": 10,
    }
