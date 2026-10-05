"""Queue-order ranking tests without checkpoints, data generation, or SAT calls."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from neural_sat_speck.data.attack import AttackInstance, AttackPair
from neural_sat_speck.followups import ranking as queue_ranking


def test_speck_composition_matches_original_one_based_stage_ties() -> None:
    stage1 = (91, 8, 44)
    stage2 = {
        91: (17, 2, 53),
        8: (5, 1, 33),
        44: (40, 7, 9),
    }

    expected_paths = []
    for parent_rank, parent in enumerate(stage1, start=1):
        for child_rank, child in enumerate(stage2[parent], start=1):
            expected_paths.append((parent_rank + child_rank, (parent, child)))
    expected_paths.sort(key=lambda item: (item[0], item[1]))
    expected = tuple((child, parent) for _, (parent, child) in expected_paths)

    assert (
        queue_ranking.compose_speck_ranked_queue(
            stage1, stage2, parent_width=3, child_width=3
        )
        == expected
    )
    assert len(expected) == len(set(expected)) == 9


def test_simon_composition_matches_original_zero_based_numpy_orders() -> None:
    stage1 = np.array([91, 8, 44], dtype=np.uint16)
    stage2 = {
        91: np.array([17, 2, 53, 6], dtype=np.uint16),
        8: np.array([5, 1, 33, 7], dtype=np.uint16),
        44: np.array([40, 7, 9, 3], dtype=np.uint16),
    }

    tuples = []
    for parent_rank, parent in enumerate(stage1):
        for child_rank, child in enumerate(stage2[int(parent)]):
            tuples.append(
                (
                    parent_rank + child_rank,
                    parent_rank,
                    child_rank,
                    int(parent),
                    int(child),
                )
            )
    tuples.sort()
    expected = tuple((child, parent) for _, _, _, parent, child in tuples)

    assert (
        queue_ranking.compose_simon_ranked_queue(
            stage1, stage2, parent_width=3, child_width=4
        )
        == expected
    )
    assert len(expected) == len(set(expected)) == 12


@pytest.mark.parametrize(
    ("stage1", "stage2"),
    [
        ((1, 1), {1: (3, 4)}),
        ((1, 2), {1: (3, 4)}),
        ((1, 2), {1: (3, 3), 2: (4, 5)}),
        ((1.0, 2), {1: (3, 4), 2: (5, 6)}),
    ],
)
def test_composers_reject_malformed_prefixes(stage1: Any, stage2: Any) -> None:
    with pytest.raises(ValueError):
        queue_ranking.compose_speck_ranked_queue(
            stage1, stage2, parent_width=2, child_width=2
        )


def _public_case(cipher: str, rounds: int) -> dict[str, Any]:
    pairs = []
    observations = []
    for index in range(64):
        p0 = [index, index + 1]
        p1 = [index + 2, index + 3]
        c0 = [index + 4, index + 5]
        c1 = [index + 6, index + 7]
        pairs.append([c0, c1])
        observations.extend([[p0, c0], [p1, c1]])
    return {
        "cipher": cipher,
        "rounds": rounds,
        "instance_id": f"{cipher}-opened-demo-0",
        "ciphertext_pairs": pairs,
        "known_observations": observations,
    }


def _rank_job(cipher: str, policy: str, rounds: int, request: float) -> dict[str, Any]:
    case_id = f"{cipher}_k0_{policy}"
    return {
        "case_id": case_id,
        "cipher": cipher,
        "policy": policy,
        "rounds": rounds,
        "key_index": 0,
        "candidate_request_seconds": request,
        "job_id": case_id + "__prepare",
        "operation": "rank",
    }


def _ranking(order: Any, *, pair_count: int = 64) -> SimpleNamespace:
    return SimpleNamespace(
        candidate_order=order,
        pair_count=pair_count,
        group_size=pair_count,
        score_aggregation="group_logit",
        total_seconds=0.01,
        partial_decryption_seconds=0.002,
        inference_seconds=0.006,
        sorting_seconds=0.002,
    )


class _FakeTorch:
    def __init__(self, *, mps_available: bool = True) -> None:
        self.thread_calls: list[tuple[str, int]] = []
        self.backends = SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps_available)
        )

    @staticmethod
    def device(name: str) -> str:
        return name

    def set_num_threads(self, value: int) -> None:
        self.thread_calls.append(("intra", value))

    def set_num_interop_threads(self, value: int) -> None:
        self.thread_calls.append(("interop", value))


def test_speck_rank_queue_uses_frozen_stage_calls_and_public_only_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_torch = _FakeTorch()
    calls: dict[str, Any] = {"parents": [], "screened": [], "loaded": []}
    paths = {}
    for name in ("speck_stage1", "speck_stage2"):
        path = tmp_path / name
        path.write_bytes(b"pinned-placeholder")
        paths[name] = path

    def partial_decrypt(instance, parent, *, expected_next_subkey, label):
        assert isinstance(instance, AttackInstance)
        assert expected_next_subkey == 0
        calls["parents"].append((parent, label))
        return instance

    def rank_group(instance, model, *, candidate_batch_size, device):
        assert isinstance(instance, AttackInstance)
        assert instance.master_key == (0, 0, 0, 0)
        assert instance.final_round_subkey == 0
        assert len(instance.pairs) == 64
        assert candidate_batch_size == 1024
        assert device == "mps"
        return _ranking(tuple(range(1 << 16)))

    def rank_screened(
        instance,
        model,
        *,
        screen_pairs,
        screen_top_k,
        candidate_batch_size,
        device,
    ):
        assert isinstance(instance, AttackInstance)
        assert screen_pairs == 16
        assert screen_top_k == 512
        assert candidate_batch_size == 1024
        assert device == "mps"
        calls["screened"].append((screen_pairs, screen_top_k))
        return _ranking(tuple(range(1 << 16)))

    dependencies = (
        fake_torch,
        object,
        rank_group,
        rank_screened,
        partial_decrypt,
        AttackInstance,
        AttackPair,
    )
    monkeypatch.setattr(queue_ranking, "_load_speck_dependencies", lambda: dependencies)

    def load_model(torch, model_type, path, device):
        calls["loaded"].append((path, device))
        return object()

    monkeypatch.setattr(queue_ranking, "_load_pair_group_ranker", load_model)
    public = _public_case("speck32_64", 7)
    job = _rank_job("speck32_64", "screened_neural", 7, 10.0)

    result = queue_ranking.rank_queue(public, job, paths)

    assert len(result["ranked"]) == 1024
    assert calls["loaded"] == [
        (paths["speck_stage1"], "mps"),
        (paths["speck_stage2"], "mps"),
    ]
    assert len(calls["parents"]) == len(calls["screened"]) == 32
    assert calls["parents"][0][0] == 0 and calls["parents"][-1][0] == 31
    assert fake_torch.thread_calls == [("intra", 1), ("interop", 1)]
    expected = queue_ranking.compose_speck_ranked_queue(
        tuple(range(32)), {parent: tuple(range(32)) for parent in range(32)}
    )
    assert result["ranked"] == [list(candidate) for candidate in expected]
    assert result["component_diagnostics"]["ordering"]["stored_tuple_order"] == (
        "chronological_child_parent"
    )


def test_speck_ranking_fails_closed_without_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorch(mps_available=False)
    monkeypatch.setattr(
        queue_ranking,
        "_load_speck_dependencies",
        lambda: (fake_torch, object, None, None, None, AttackInstance, AttackPair),
    )
    with pytest.raises(RuntimeError, match="requires MPS"):
        queue_ranking._rank_speck(
            _public_case("speck32_64", 7),
            tuple((((0, 0), (0, 0))) for _ in range(64)),
            tuple((((0, 0), (0, 0))) for _ in range(128)),
            "screened_neural",
            {},
        )


def _simon_ranking(order: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        order=order,
        screen_pairs_used=16,
        full_pairs_used=64,
        total_seconds=0.01,
        row_load_seconds=0.001,
        screen_seconds=0.004,
        rescore_seconds=0.003,
        sort_seconds=0.002,
        unique_left_rows=64,
        screen_tie_count=1,
        full_tie_count=1,
        screen_collision_count=0,
        full_collision_count=0,
    )


def test_simon_classical_uses_fixed_public_ranking_settings_and_numpy_orders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, Any] = {"stage2": []}
    ddt_path = tmp_path / "ddt.bin"
    ddt_path.write_bytes(b"pinned-placeholder")
    tool_path = tmp_path / "queried-rows"
    tool_path.write_bytes(b"pinned-placeholder")
    tool_path.chmod(0o755)

    class FakeDdt:
        def __init__(self, path, rounds):
            assert path == ddt_path
            assert rounds == 7

    def rank_stage1(pairs, **kwargs):
        assert len(pairs) == 64
        assert kwargs == {
            "tool_path": tool_path,
            "ddt_7r_path": ddt_path,
            "screen_pairs": 16,
            "full_pairs": 64,
            "rescore_top": 512,
            "batch_size": 1024,
        }
        return _simon_ranking(np.arange(512, dtype=np.uint16)), {
            "tool_wall_seconds": 0.02
        }

    def rank_classical(pairs, table, **kwargs):
        assert len(pairs) == 64
        calls["stage2"].append(kwargs)
        return _simon_ranking(np.arange(512, dtype=np.uint16))

    monkeypatch.setattr(
        queue_ranking,
        "_load_simon_classical_dependencies",
        lambda: (FakeDdt, rank_classical, rank_stage1),
    )
    result = queue_ranking.rank_queue(
        _public_case("simon32_64", 10),
        _rank_job("simon32_64", "classical_8parent", 10, 45.0),
        {"simon_ddt": ddt_path, "simon_tool": tool_path},
    )

    assert len(result["ranked"]) == 1024
    assert len(calls["stage2"]) == 8
    assert [entry["final_key_candidate"] for entry in calls["stage2"]] == list(range(8))
    assert all(
        entry
        == {
            "final_key_candidate": entry["final_key_candidate"],
            "expected_rounds": 7,
            "screen_pairs": 16,
            "full_pairs": 64,
            "rescore_top": 512,
            "batch_size": 1024,
        }
        for entry in calls["stage2"]
    )


def test_simon_bao_path_uses_existing_pinned_subprocess_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ddt = tmp_path / "ddt.bin"
    ddt.write_bytes(b"ddt")
    tf = tmp_path / "tf-python"
    tf.write_bytes(b"python")
    tf.chmod(0o755)
    model = tmp_path / "ND_VV_Simon32_9R.h5"
    model.write_bytes(b"checkpoint")
    worker = tmp_path / "run_queue_order_neural_ranking.py"
    worker.write_bytes(b"worker")
    calls: dict[str, Any] = {}
    monkeypatch.setattr(queue_ranking.importlib.util, "find_spec", lambda name: None)

    def generate(pairs, **kwargs):
        calls["pairs"] = pairs
        calls["kwargs"] = kwargs
        candidates = [(child, parent) for parent in range(8) for child in range(128)]
        return candidates, {
            "ranking_wall_seconds": 12.0,
            "stage1_seconds": 3.0,
            "stage2_seconds": 8.0,
            "peak_rss_gib": 1.5,
            "full_pairs_used": 64,
            "candidate_tuples_produced": 1024,
        }

    monkeypatch.setattr(
        queue_ranking, "_load_simon_hybrid_dependency", lambda: (None, generate)
    )
    result = queue_ranking.rank_queue(
        _public_case("simon32_64", 10),
        _rank_job("simon32_64", "bao_ddt_hybrid_8parent", 10, 45.0),
        {
            "simon_ddt": ddt,
            "tf_python": tf,
            "bao_vv9": model,
            "source:scripts/run_queue_order_neural_ranking.py": worker,
        },
    )

    assert len(calls["pairs"]) == 64
    assert calls["kwargs"] == {
        "vv9_model": None,
        "ddt_7r_path": ddt,
        "tf_python_path": tf,
        "checkpoint_dir": tmp_path,
        "worker_script_path": worker,
        "ranking_time_limit": 120.0,
        "return_diagnostics": True,
    }
    assert result["candidate_tuple_count"] == 1024


def test_rank_queue_rejects_truth_fields_and_policy_drift_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatched = False

    def forbidden(*_args, **_kwargs):
        nonlocal dispatched
        dispatched = True
        raise AssertionError("ranking dispatch should not run")

    monkeypatch.setattr(queue_ranking, "_rank_speck", forbidden)
    public = _public_case("speck32_64", 7)
    public["master_key"] = [0, 0, 0, 0]
    with pytest.raises(ValueError, match="stripped public fields"):
        queue_ranking.rank_queue(
            public,
            _rank_job("speck32_64", "screened_neural", 7, 10.0),
            {},
        )
    assert dispatched is False

    with pytest.raises(ValueError, match="candidate request"):
        queue_ranking.rank_queue(
            _public_case("speck32_64", 7),
            _rank_job("speck32_64", "screened_neural", 7, 45.0),
            {},
        )
    assert dispatched is False

    job = _rank_job("speck32_64", "screened_neural", 7, 10.0)
    job["assigned_key"] = [0, 0, 0, 0]
    with pytest.raises(ValueError, match="fixed public identity"):
        queue_ranking.rank_queue(_public_case("speck32_64", 7), job, {})
    assert dispatched is False
