"""Protocol and schema tests for Phase 7 held-out comparative evaluation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs/experiments/phase7_final_evaluation.json"
RUNNER_PATH = REPOSITORY_ROOT / "scripts/run_phase7_final_evaluation.py"


def _runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("phase7_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load Phase 7 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_phase7_config_is_held_out_test_and_seed_disjoint() -> None:
    runner = _runner()
    config = _config()
    runner._validate_config(config)

    assert config["phase"] == 7
    assert config["validation_only"] is False
    assert config["held_out_test_generated"] is True

    dataset = config["dataset"]
    assert dataset["split_role"] == "test"
    assert dataset["seed"] == 98001
    assert dataset["seed"] not in config["seed_exclusions"]


def test_phase7_methods_include_all_five_configurations() -> None:
    config = _config()
    methods = [m["method_id"] for m in config["evaluation_methods"]]
    assert methods == [
        "sat_only",
        "random_sat",
        "b0_sat",
        "b1_sat",
        "gated_b1_fallback_sat",
    ]


def test_phase7_counterbalanced_order_is_cyclic_permutation() -> None:
    runner = _runner()
    order0 = runner._counterbalanced_methods_order(0)
    order1 = runner._counterbalanced_methods_order(1)
    order5 = runner._counterbalanced_methods_order(5)

    assert len(order0) == 5
    assert set(order0) == set(runner.METHODS)
    assert order0[0] == "sat_only"
    assert order1[0] == "random_sat"
    assert order5 == order0


def test_phase7_atomic_write_and_summary_aggregation(tmp_path: Path) -> None:
    runner = _runner()
    records_dir = tmp_path / "records"
    records_dir.mkdir(parents=True)

    dummy_record = {
        "schema": runner.RESULT_SCHEMA,
        "instance_id": "speck32_64_test_0000",
        "instance_index": 0,
        "master_key": [0x1918, 0x1110, 0x0908, 0x0100],
        "true_final_round_subkey": 0x1234,
        "execution_order": list(runner.METHODS),
        "provenance": {
            "config_sha256": "dummy_config",
            "manifest_sha256": "dummy_manifest",
            "git_commit": "dummy_commit",
        },
        "methods": {
            "sat_only": {"success": False, "t_online": 10.0, "candidates_attempted": 0},
            "random_sat": {"success": False, "t_online": 20.0, "candidates_attempted": 16},
            "b0_sat": {"success": True, "t_online": 8.0, "candidates_attempted": 4},
            "b1_sat": {"success": True, "t_online": 5.0, "candidates_attempted": 2},
            "gated_b1_fallback_sat": {"success": True, "t_online": 5.2, "candidates_attempted": 2},
        },
    }
    runner._write_atomic(records_dir / "instance_0000.json", dummy_record)

    summary = runner.summarize_records(
        records_dir,
        _config(),
        manifest_sha256="dummy_manifest",
        config_sha256="dummy_config",
    )
    assert summary["instance_count"] == 1
    assert summary["methods"]["b1_sat"]["success_rate"] == 1.0
    assert summary["methods"]["sat_only"]["success_rate"] == 0.0
    assert summary["methods"]["b1_sat"]["mean_online_seconds"] == 5.0
