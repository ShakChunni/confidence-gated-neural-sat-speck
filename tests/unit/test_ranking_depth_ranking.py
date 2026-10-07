"""Synthetic contract tests for the D-055 queue adapter; no model/solver runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from neural_sat_speck.backend.guard import read_checked_json
from neural_sat_speck.followups import ranking_depth_ranking as adapter
from neural_sat_speck.followups.ranking import compose_simon_ranked_queue


def _pairs() -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    return tuple(((index, index + 1), (index + 2, index + 3)) for index in range(64))


def _paths(tmp_path: Path, cipher: str, trailing_keys: int) -> dict[str, Path]:
    names = (
        ("speck_stage1", "speck_classical")
        if cipher == "speck32_64"
        else ("bao_vv9", "tf_python", "simon_ddt")
    )
    required = (
        names
        if trailing_keys == 2
        else names[:2]
        if cipher == "simon32_64"
        else names[:1]
    )
    result = {}
    for name in required:
        path = tmp_path / (
            "ND_VV_Simon32_9R.h5" if name == "bao_vv9" else f"{name}.asset"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic-placeholder")
        if name == "tf_python":
            path.chmod(0o755)
        result[name] = path
    return result


def _full_domain_order(start: int = 0) -> tuple[int, ...]:
    return tuple((start + offset) & 0xFFFF for offset in range(1 << 16))


def test_speck_stage1_spy_receives_only_ciphertext_and_fixed_mps_batch(
    tmp_path, monkeypatch
):
    stage1_path = tmp_path / "stage1.pt"
    stage1_path.write_bytes(b"checkpoint")
    calls: dict[str, Any] = {"rank": 0, "sync": 0, "threads": []}

    class FakeTorch:
        backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True))

        class mps:
            @staticmethod
            def synchronize():
                calls["sync"] += 1

        @staticmethod
        def device(name: str) -> str:
            return name

        @staticmethod
        def set_num_threads(count: int) -> None:
            calls["threads"].append(("intra", count))

        @staticmethod
        def set_num_interop_threads(count: int) -> None:
            calls["threads"].append(("interop", count))

    @dataclass(frozen=True)
    class FakeAttackPair:
        plaintext0: tuple[int, int]
        plaintext1: tuple[int, int]
        ciphertext0: tuple[int, int]
        ciphertext1: tuple[int, int]

    @dataclass(frozen=True)
    class FakeAttackInstance:
        instance_id: str
        master_key: tuple[int, int, int, int]
        final_round_subkey: int
        pairs: tuple[FakeAttackPair, ...]

    class FakeModel:
        pass

    def load_model(torch, model_type, path, device):
        assert torch is FakeTorch
        assert model_type is FakeModel
        assert path == stage1_path
        assert device == "mps"
        return object()

    def rank(instance, model, *, candidate_batch_size, device):
        calls["rank"] += 1
        assert isinstance(instance, FakeAttackInstance)
        assert instance.master_key == (0, 0, 0, 0)
        assert instance.final_round_subkey == 0
        assert len(instance.pairs) == 64
        assert all(
            pair.plaintext0 == pair.plaintext1 == (0, 0) for pair in instance.pairs
        )
        assert (
            tuple((pair.ciphertext0, pair.ciphertext1) for pair in instance.pairs)
            == _pairs()
        )
        assert candidate_batch_size == 1024
        assert device == "mps"
        return SimpleNamespace(
            candidate_order=np.arange(1 << 16, dtype=np.uint16),
            pair_count=64,
            group_size=64,
            candidate_batch_size=1024,
            score_aggregation="group_logit",
            total_seconds=0.125,
            partial_decryption_seconds=0.01,
            inference_seconds=0.1,
            sorting_seconds=0.015,
        )

    monkeypatch.setattr(
        adapter,
        "_load_speck_dependencies",
        lambda: (
            FakeTorch,
            rank,
            None,
            FakeAttackInstance,
            FakeAttackPair,
            FakeModel,
            load_model,
        ),
    )
    order, detail, _ = adapter._rank_speck_stage1(
        _pairs(), {"speck_stage1": stage1_path}
    )
    assert order == tuple(range(1 << 16))
    assert detail["device"] == "mps"
    assert detail["candidate_batch_size"] == 1024
    assert calls == {"rank": 1, "sync": 1, "threads": [("intra", 1), ("interop", 1)]}


def test_speck_sparse_r5_stage2_uses_all_fixed_parents_and_existing_queue_order(
    tmp_path, monkeypatch
):
    table_path = tmp_path / "sparse_r5.json"
    table_path.write_text("{}", encoding="utf-8")
    calls: dict[str, Any] = {"descent": [], "rank": [], "sync": 0}
    parents = tuple(range(0x3100, 0x3120))
    full_order = _full_domain_order()

    class FakeTorch:
        backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True))

        class mps:
            @staticmethod
            def synchronize():
                calls["sync"] += 1

        @staticmethod
        def device(name: str) -> str:
            return name

        @staticmethod
        def set_num_threads(_count: int) -> None:
            pass

        @staticmethod
        def set_num_interop_threads(_count: int) -> None:
            pass

    class FakeTable:
        @staticmethod
        def load(path):
            assert path == table_path
            calls["table_loads"] = calls.get("table_loads", 0) + 1
            return object()

    @dataclass(frozen=True)
    class FakeAttackPair:
        plaintext0: tuple[int, int]
        plaintext1: tuple[int, int]
        ciphertext0: tuple[int, int]
        ciphertext1: tuple[int, int]

    @dataclass(frozen=True)
    class FakeAttackInstance:
        instance_id: str
        master_key: tuple[int, int, int, int]
        final_round_subkey: int
        pairs: tuple[FakeAttackPair, ...]

    def partial_decrypt(instance, parent, *, expected_next_subkey, label):
        assert instance.master_key == (0, 0, 0, 0)
        assert expected_next_subkey == 0
        assert label.endswith(f"{parent:04x}")
        calls["descent"].append(parent)
        return instance

    def rank(instance, table):
        assert table is not None
        assert instance.master_key == (0, 0, 0, 0)
        calls["rank"].append(instance.instance_id)
        return SimpleNamespace(candidate_order=full_order, total_seconds=0.01)

    def load_speck_stage2():
        return (
            FakeTorch,
            FakeTable,
            rank,
            partial_decrypt,
            FakeAttackInstance,
            FakeAttackPair,
            compose_speck_ranked_queue,
        )

    # Exercise the established composer rather than reproducing its tie rule here.
    from neural_sat_speck.followups.ranking import compose_speck_ranked_queue

    monkeypatch.setattr(adapter, "_load_speck_stage2_dependencies", load_speck_stage2)
    queue, detail, _ = adapter._rank_speck_stage2(
        _pairs(), parents, {"speck_classical": table_path}
    )
    expected = compose_speck_ranked_queue(
        parents,
        {parent: full_order for parent in parents},
        parent_width=32,
        child_width=32,
    )
    assert queue == expected
    assert len(queue) == 1024
    assert all(child_parent[1] in parents for child_parent in queue)
    assert calls["descent"] == list(parents)
    assert len(calls["rank"]) == 32
    assert calls["table_loads"] == 1
    assert calls["sync"] == 1
    assert detail["table"] == "sparseR5"
    assert detail["per_parent"][0]["top_children"] == list(range(32))


def test_one_and_two_key_simon_queues_share_persisted_stage1_prefix(
    tmp_path, monkeypatch
):
    stage1_order = tuple(range(0x1000, 0x1000 + 512))
    stage2_order_by_parent = {parent: tuple(range(128)) for parent in stage1_order[:8]}
    observed: dict[str, Any] = {"stage1_calls": 0, "stage2_calls": 0}

    def fake_stage1(pairs, paths, directory):
        observed["stage1_calls"] += 1
        assert pairs == _pairs()
        return stage1_order, {"ranker": "synthetic-vv9"}, {"vv.py": "a" * 64}

    def fake_stage2(pairs, order, parents, paths):
        observed["stage2_calls"] += 1
        assert pairs == _pairs()
        assert order == stage1_order
        assert parents == stage1_order[:8]
        stage1_receipt = read_checked_json(observed["active_dir"] / "stage1.json")
        assert tuple(stage1_receipt["stage1_parents"]) == parents
        assert [name for name, _ in observed["events"]] == [
            "stage1_start",
            "stage1_complete",
            "stage2_start",
        ]
        return (
            compose_simon_ranked_queue(
                order, stage2_order_by_parent, parent_width=8, child_width=128
            ),
            {"ranker": "synthetic-ddt"},
            {"ddt.py": "b" * 64},
        )

    monkeypatch.setattr(adapter, "_rank_simon_stage1", fake_stage1)
    monkeypatch.setattr(adapter, "_rank_simon_stage2", fake_stage2)
    paths_1 = _paths(tmp_path / "m1", "simon32_64", 1)
    paths_2 = _paths(tmp_path / "m2", "simon32_64", 2)
    case_1, case_2 = tmp_path / "m1_case", tmp_path / "m2_case"
    case_1.mkdir()
    case_2.mkdir()

    observed["active_dir"] = case_1
    observed["events"] = []
    one = adapter.rank_queue(
        "simon32_64",
        _pairs(),
        1,
        paths_1,
        case_1,
        lambda name, detail: observed["events"].append((name, detail)),
    )
    assert one["queue"] == [[parent] for parent in stage1_order[:8]]
    assert one["stage2_wall_seconds"] == 0.0
    one_persisted = read_checked_json(case_1 / "queue.json")
    assert one_persisted["queue"] == one["queue"]
    assert (
        one_persisted["stage1_sha256"]
        == one_persisted["ranking_receipts"]["stage1"]["stage1_prefix_sha256"]
    )
    assert read_checked_json(case_1 / "stage1.json")["stage1_parents"] == list(
        stage1_order[:8]
    )

    observed["active_dir"] = case_2
    observed["events"] = []
    two = adapter.rank_queue(
        "simon32_64",
        _pairs(),
        2,
        paths_2,
        case_2,
        lambda name, detail: observed["events"].append((name, detail)),
    )
    assert two["stage1_parents"] == one["stage1_parents"]
    assert two["queue"] == [
        list(candidate)
        for candidate in compose_simon_ranked_queue(
            stage1_order,
            stage2_order_by_parent,
            parent_width=8,
            child_width=128,
        )
    ]
    assert [name for name, _ in observed["events"]] == [
        "stage1_start",
        "stage1_complete",
        "stage2_start",
    ]
    assert observed["stage1_calls"] == 2
    assert observed["stage2_calls"] == 1
    two_persisted = read_checked_json(case_2 / "queue.json")
    assert two_persisted["queue_sha256"] == two["queue_sha256"]
    assert (
        two_persisted["stage1_sha256"]
        == two_persisted["ranking_receipts"]["stage1"]["stage1_prefix_sha256"]
    )


@pytest.mark.parametrize("invalid", [0, 3, True, 1.0])
def test_rejects_nonfrozen_trailing_key_counts(invalid, tmp_path):
    with pytest.raises(ValueError):
        adapter.rank_queue(
            "simon32_64",
            _pairs(),
            invalid,
            {},
            tmp_path / "case",
        )


@pytest.mark.parametrize(
    "bad_pairs",
    [
        (),
        (("plaintext", (0, 0)),) * 64,
        (((0, 0), (0, True)),) * 64,
        (((0, 0), (0, 65536)),) * 64,
    ],
)
def test_accepts_only_exact_public_ciphertext_pair_shape(bad_pairs):
    with pytest.raises(ValueError):
        adapter._validate_ciphertext_pairs(bad_pairs)


def test_rejects_bool_and_duplicate_rank_words():
    with pytest.raises(ValueError):
        adapter._validate_word_order((False, 1), 2, "synthetic")
    with pytest.raises(ValueError):
        adapter._validate_word_order((1, 1), 2, "synthetic")


def test_tf_child_profile_is_stage1_only_and_has_no_truth_inputs():
    import inspect

    signature = inspect.signature(adapter.rank_queue)
    assert set(signature.parameters) == {
        "cipher",
        "ciphertext_pairs",
        "trailing_keys",
        "paths",
        "directory",
        "phase_callback",
    }
    source = inspect.getsource(adapter._tf_stage1_main)
    assert "load_published_vv_checkpoint" in source
    assert "rank_neural_domain" in source
    assert "feature_batch_size=_BATCH_SIZE" in source
    assert "set_intra_op_parallelism_threads(_TF_INTRA_OP_THREADS)" in source
    assert "set_inter_op_parallelism_threads(_TF_INTER_OP_THREADS)" in source
    assert "master_key" not in source
    assert 'payload["pairs"]' in source
    assert "_install_tf_rank_import_boundary(root)" in source
    assert "_loaded_forbidden_modules()" in source


def test_tf_import_boundary_bypasses_simon_init_and_blocks_sat(monkeypatch, tmp_path):
    import importlib
    import sys

    for name in tuple(sys.modules):
        if any(
            name == forbidden or name.startswith(forbidden + ".")
            for forbidden in adapter._FORBIDDEN_TF_IMPORTS
        ):
            monkeypatch.delitem(sys.modules, name, raising=False)
    package_name = "neural_sat_speck.simon"
    monkeypatch.delitem(sys.modules, package_name, raising=False)
    root = tmp_path / "checkout"
    (root / "src/neural_sat_speck/simon").mkdir(parents=True)
    blocker = adapter._install_tf_rank_import_boundary(root)
    try:
        simon = sys.modules[package_name]
        assert getattr(simon, "__file__", None) is None
        assert simon.__path__ == [str(root / "src/neural_sat_speck/simon")]
        with pytest.raises(ImportError, match="rank import boundary blocked"):
            importlib.import_module("pycryptosat")
        with pytest.raises(ImportError, match="rank import boundary blocked"):
            importlib.import_module("neural_sat_speck.simon.sat_model")
        assert blocker.attempts == [
            "pycryptosat",
            "neural_sat_speck.simon.sat_model",
        ]
    finally:
        sys.meta_path.remove(blocker)


def test_source_receipts_hash_project_package_code_not_runtime_packages(
    monkeypatch, tmp_path
):
    import sys

    repo = tmp_path / "repo"
    project_source = repo / "src/neural_sat_speck/followups/ranker.py"
    runtime_source = repo / "results/tensorflow/site-packages/tensorflow/__init__.py"
    project_source.parent.mkdir(parents=True)
    runtime_source.parent.mkdir(parents=True)
    project_source.write_text("PROJECT_SOURCE = True\n", encoding="utf-8")
    runtime_source.write_text("INSTALLED_RUNTIME = True\n", encoding="utf-8")
    monkeypatch.setattr(adapter, "_REPO_ROOT", repo)

    project_module = "d055_test_project_ranker"
    runtime_module = "d055_test_tensorflow_runtime"
    sys.modules[project_module] = SimpleNamespace(__file__=str(project_source))
    sys.modules[runtime_module] = SimpleNamespace(__file__=str(runtime_source))
    try:
        receipts = adapter._source_imports(set())
    finally:
        del sys.modules[project_module]
        del sys.modules[runtime_module]

    assert receipts == {
        "src/neural_sat_speck/followups/ranker.py": adapter._sha256_file(project_source)
    }
