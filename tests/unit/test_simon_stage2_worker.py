from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from neural_sat_speck.followups.stage2_control import validate_order

ROOT = Path(__file__).resolve().parents[2]
WORKER_PATH = ROOT / "scripts" / "run_simon_matched_stage2_worker.py"


def _worker_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_simon_stage2_worker_test", WORKER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rank_result(words: list[int] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        order=list(range(512)) if words is None else words,
        feature_seconds=0.1,
        inference_seconds=0.2,
        aggregation_seconds=0.03,
        screen_seconds=0.4,
        row_load_seconds=0.02,
        rescore_seconds=0.05,
        sort_seconds=0.01,
        total_seconds=0.81,
    )


def _control() -> SimpleNamespace:
    return SimpleNamespace(
        validate_order=_strict_order,
        validate_payload=lambda payload: None,
        fixed_jobs=lambda: [],
    )


def _strict_order(words: Any) -> tuple[int, ...]:
    result = tuple(words)
    if len(result) != 512 or len(set(result)) != 512:
        raise ValueError("expected 512 distinct words")
    if any(type(word) is not int or not 0 <= word <= 0xFFFF for word in result):
        raise ValueError("invalid word")
    return result


def test_call_schedule_preserves_all_eight_parent_ordinals() -> None:
    worker = _worker_module()
    parents = [101, 202, 303, 404, 505, 606, 707, 808]

    assert worker._call_schedule("stage1", "vv9", []) == ((0, None),)
    assert worker._call_schedule("stage2", "vv8", parents) == tuple(enumerate(parents))
    assert worker._call_schedule("stage2", "ddt7", parents) == tuple(enumerate(parents))
    with pytest.raises(ValueError):
        worker._call_schedule("stage2", "vv8", parents[:7])
    with pytest.raises(ValueError):
        worker._call_schedule("stage1", "vv9", [parents[0]])


def test_rank_arguments_keep_the_frozen_screen_rescore_and_depths() -> None:
    worker = _worker_module()
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
    pairs = [[(1, 2), (3, 4)] for _ in range(64)]
    model = object()
    table = object()

    def neural_ranker(*args: Any, **kwargs: Any) -> SimpleNamespace:
        calls.append(("neural", args, kwargs))
        return _rank_result()

    def classical_ranker(*args: Any, **kwargs: Any) -> SimpleNamespace:
        calls.append(("classical", args, kwargs))
        return _rank_result()

    modules = {
        "control": _control(),
        "published_vv_ranking": SimpleNamespace(rank_neural_domain=neural_ranker),
        "classical_ranking": SimpleNamespace(rank_classical_domain=classical_ranker),
    }
    for scorer, parent, scorer_object in (("vv9", None, model), ("vv8", 0x1234, model)):
        words, timings = worker._rank_one(scorer, pairs, parent, scorer_object, modules)
        assert len(words) == 512 and len(set(words)) == 512
        assert timings["total_seconds"] == 0.81
    words, _ = worker._rank_one("ddt7", pairs, 0xBEEF, table, modules)
    assert len(words) == 512

    assert len(calls) == 3
    for _, args, kwargs in calls[:2]:
        assert args[0] is pairs
        assert args[1] is model
        assert kwargs["screen_pairs"] == 16
        assert kwargs["full_pairs"] == 64
        assert kwargs["rescore_top"] == 512
        assert kwargs["feature_batch_size"] == 1024
        assert kwargs["inference_batch_size"] == 16384
    assert calls[0][2]["final_key_candidate"] is None
    assert calls[1][2]["final_key_candidate"] == 0x1234
    assert calls[2][0] == "classical"
    assert calls[2][1][0] is pairs
    assert calls[2][1][1] is table
    assert calls[2][2] == {
        "final_key_candidate": 0xBEEF,
        "expected_rounds": 7,
        "screen_pairs": 16,
        "full_pairs": 64,
        "rescore_top": 512,
        "batch_size": 1024,
    }


def test_checkpoint_loader_selects_vv9_vv8_and_ddt7_depths(tmp_path: Path) -> None:
    worker = _worker_module()
    calls: list[tuple[Path, int]] = []

    def load_checkpoint(path: Path, rounds: int) -> tuple[object, dict[str, Any]]:
        calls.append((path, rounds))
        name = "vv9" if rounds == 9 else "vv8"
        return object(), {
            "rounds": rounds,
            "sha256": assets[name]["sha256"],
            "loaded_layers": 616,
            "weight_arrays_exactly_matched": 1080,
            "scalar_parameters_exactly_matched": 449457,
        }

    class FakeDdt:
        def __init__(self, path: Path, expected_rounds: int) -> None:
            calls.append((path, expected_rounds))
            self.rounds = expected_rounds

    paths = {name: tmp_path / name for name in ("vv9", "vv8", "ddt")}
    for path in paths.values():
        path.write_bytes(b"synthetic asset identity")
    assets = {name: {"sha256": f"{index:064x}"} for index, name in enumerate(paths, 1)}
    modules = {
        "published_vv": SimpleNamespace(load_published_vv_checkpoint=load_checkpoint),
        "classical_ranking": SimpleNamespace(FullPredictedDdt=FakeDdt),
    }

    worker._load_scorer("vv9", paths, assets, modules)
    worker._load_scorer("vv8", paths, assets, modules)
    table, ddt_metadata = worker._load_scorer("ddt7", paths, assets, modules)

    assert calls == [(paths["vv9"], 9), (paths["vv8"], 8), (paths["ddt"], 7)]
    assert table.rounds == 7
    assert ddt_metadata["expected_rounds"] == 7


def test_rank_result_rejects_duplicate_and_wrong_width_orders() -> None:
    worker = _worker_module()
    pairs = [[(1, 2), (3, 4)] for _ in range(64)]
    modules = {
        "control": _control(),
        "published_vv_ranking": SimpleNamespace(
            rank_neural_domain=lambda *args, **kwargs: _rank_result([0] * 512)
        ),
        "classical_ranking": SimpleNamespace(),
    }

    with pytest.raises(ValueError):
        worker._rank_one("vv9", pairs, None, object(), modules)
    modules["published_vv_ranking"].rank_neural_domain = lambda *a, **k: _rank_result(
        list(range(511))
    )
    with pytest.raises(ValueError):
        worker._rank_one("vv9", pairs, None, object(), modules)


@pytest.mark.parametrize("scorer,parent", [("vv9", None), ("vv8", 123), ("ddt7", 123)])
def test_real_ranker_uint16_array_crosses_strict_json_boundary(scorer, parent):
    worker = _worker_module()
    result = _rank_result()
    result.order = np.arange(512, dtype=np.uint16)
    modules = {
        "control": SimpleNamespace(validate_order=validate_order),
        "published_vv_ranking": SimpleNamespace(
            rank_neural_domain=lambda *a, **k: result
        ),
        "classical_ranking": SimpleNamespace(
            rank_classical_domain=lambda *a, **k: result
        ),
    }
    words, _ = worker._rank_one(
        scorer, [[[0, 1], [2, 3]]] * 64, parent, object(), modules
    )
    assert words == tuple(range(512))
    assert all(type(word) is int for word in words)


def test_float_rank_array_is_not_silently_truncated_to_words():
    worker = _worker_module()
    result = _rank_result()
    result.order = np.arange(512, dtype=np.float64)
    modules = {
        "control": SimpleNamespace(validate_order=validate_order),
        "published_vv_ranking": SimpleNamespace(
            rank_neural_domain=lambda *a, **k: result
        ),
    }
    with pytest.raises(ValueError):
        worker._rank_one("vv9", [[[0, 1], [2, 3]]] * 64, None, object(), modules)


def test_worker_loads_one_checkpoint_and_passes_ciphertext_only_to_eight_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _worker_module()
    job = {
        "job_id": "simon_r10_k00_stage2_vv8",
        "key_index": 0,
        "operation": "stage2",
        "scorer": "vv8",
    }
    parents = [101, 202, 303, 404, 505, 606, 707, 808]
    ciphertext_pairs = [[(i, i + 1), (i + 2, i + 3)] for i in range(64)]
    public_case = {
        "cipher": "SIMON32/64",
        "rounds": 10,
        "key_index": 0,
        "instance_id": "demo999999_r10_k00",
        "ciphertext_pairs": ciphertext_pairs,
    }
    directory = tmp_path / "worker"
    directory.mkdir()
    payload_path = directory / "payload.json"
    payload = {
        "schema": "stage2_rank_payload_v1",
        "job": job,
        "public_case": public_case,
        "parents": parents,
        "invocation": "invocation-001",
        "directory": str(directory),
        "source_root": str(tmp_path),
        "source_commit": "commit-fixed",
        "source_files": {},
        "assets": {},
        "runtime": {"python": {}, "tensorflow": {}},
        "profile": {},
    }
    control = _control()
    control.fixed_jobs = lambda: [job]
    control.validate_payload = lambda observed: None
    control.validate_order = _strict_order
    asset_paths = {}
    expected_stats = {}
    assets = {}
    for name in ("vv9", "vv8", "ddt"):
        asset_path = tmp_path / f"{name}.asset"
        asset_path.write_bytes(name.encode())
        asset_paths[name] = asset_path
        expected_stats[name] = worker._identity(asset_path)
        assets[name] = {"sha256": "a" * 64}
    payload["assets"] = assets
    worker._write_checked_json(payload_path, payload)

    model = object()
    model_loads: list[tuple[Path, int]] = []
    neural_calls: list[tuple[int | None, list[Any], Any]] = []

    def loader(path: Path, rounds: int) -> tuple[object, dict[str, Any]]:
        model_loads.append((path, rounds))
        return model, {
            "rounds": rounds,
            "sha256": assets["vv8"]["sha256"],
            "loaded_layers": 616,
            "weight_arrays_exactly_matched": 1080,
            "scalar_parameters_exactly_matched": 449457,
        }

    def neural_ranker(
        pairs: list[Any], loaded_model: Any, **kwargs: Any
    ) -> SimpleNamespace:
        neural_calls.append((kwargs["final_key_candidate"], pairs, loaded_model))
        return _rank_result()

    modules = {
        "control": control,
        "published_vv": SimpleNamespace(load_published_vv_checkpoint=loader),
        "published_vv_ranking": SimpleNamespace(rank_neural_domain=neural_ranker),
        "classical_ranking": SimpleNamespace(),
        "import_blocker": worker._ImportBlocker(),
    }
    telemetry = SimpleNamespace(
        git_commit="commit-fixed",
        to_dict=lambda: {"is_certified": True, "git_commit": "commit-fixed"},
    )
    telemetry_module = SimpleNamespace(certify_worker_or_raise=lambda root: telemetry)

    monkeypatch.setattr(worker, "_validate_source_files", lambda p: (tmp_path, {}))
    monkeypatch.setattr(
        worker, "_validate_assets", lambda p: (asset_paths, expected_stats)
    )
    monkeypatch.setattr(worker, "_load_control_module", lambda root, files: control)
    monkeypatch.setattr(
        worker,
        "_check_runtime",
        lambda p, s, root, **kwargs: ({"runtime": "pinned"}, modules),
    )
    monkeypatch.setattr(
        worker, "_load_telemetry_module", lambda files: telemetry_module
    )
    monkeypatch.setattr(worker, "_assert_import_boundary", lambda imported: None)
    result = worker._run_worker(payload_path)

    assert result["completed_call_count"] == 8
    assert model_loads == [(asset_paths["vv8"], 8)]
    assert [call[0] for call in neural_calls] == parents
    payload_pairs = json.loads(json.dumps(ciphertext_pairs))
    assert all(call[1] == payload_pairs for call in neural_calls)
    assert all(call[2] is model for call in neural_calls)
    assert set(public_case) == {
        "cipher",
        "rounds",
        "key_index",
        "instance_id",
        "ciphertext_pairs",
    }

    result_records = [
        json.loads((directory / "calls" / f"{ordinal:04d}_result.json").read_text())
        for ordinal in range(8)
    ]
    assert [record["ordinal"] for record in result_records] == list(range(8))
    assert [record["parent"] for record in result_records] == parents
    assert all(record["ordered_words"] == list(range(512)) for record in result_records)
    events = sorted((directory / "events").glob("*.json"))
    assert [path.name for path in events] == [f"{index:04d}.json" for index in range(9)]
    final = json.loads((directory / "worker_result.json").read_text())
    assert final["completed_call_count"] == 8
    assert [call["ordinal"] for call in final["calls"]] == list(range(8))


def test_namespace_bootstrap_skips_both_package_initializers(tmp_path: Path) -> None:
    package = tmp_path / "src" / "neural_sat_speck"
    simon = package / "simon"
    simon.mkdir(parents=True)
    marker = tmp_path / "initializer_ran"
    (package / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
    )
    (simon / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
    )

    # The test process may have imported repository packages already, so verify
    # the bootstrap in an isolated interpreter with an adversarial initializer.
    import subprocess

    program = (
        "import importlib, runpy, sys; "
        f"m=runpy.run_path({str(WORKER_PATH)!r}); "
        f"m['_bootstrap_minimal_namespace'](__import__('pathlib').Path({str(tmp_path)!r})); "
        "importlib.import_module('neural_sat_speck.simon'); "
        f"assert not __import__('pathlib').Path({str(marker)!r}).exists(); "
        "assert getattr(sys.modules['neural_sat_speck.simon'],'__file__',None) is None"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert not marker.exists()


def test_import_blocker_rejects_solver_and_verifier_modules() -> None:
    worker = _worker_module()
    blocker = worker._ImportBlocker()
    for module_name in (
        "pycryptosat",
        "neural_sat_speck.simon.sat_model",
        "neural_sat_speck.simon.verification",
        "neural_sat_speck.protocol.worker_telemetry",
    ):
        with pytest.raises(ImportError):
            blocker.find_spec(module_name)
    assert blocker.attempts == [
        "pycryptosat",
        "neural_sat_speck.simon.sat_model",
        "neural_sat_speck.simon.verification",
        "neural_sat_speck.protocol.worker_telemetry",
    ]


def test_importing_worker_module_does_not_import_project_or_solver_modules() -> None:
    before = set(sys.modules)
    _worker_module()
    introduced = set(sys.modules) - before
    assert not any(
        name == "neural_sat_speck"
        or name.startswith("neural_sat_speck.")
        or name == "pycryptosat"
        for name in introduced
    )
