"""Truth-blind candidate-queue construction for the opened-demo order control.

This module only creates the ranked candidate set. It does not construct SAT
models or inspect assigned keys, true round keys, or fresh verification data.
The independent queue-order coordinator imports it only after manifest and
environment preflight has passed.
"""

from __future__ import annotations

import importlib.util
import math
import os
import time
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from neural_sat_speck.followups.queue_order import POLICIES
from neural_sat_speck.followups.queue_order import queue as validate_queue

_PUBLIC_FIELDS = frozenset(
    {"cipher", "rounds", "instance_id", "ciphertext_pairs", "known_observations"}
)
_RANK_JOB_FIELDS = frozenset(
    {
        "case_id",
        "cipher",
        "policy",
        "rounds",
        "key_index",
        "candidate_request_seconds",
        "job_id",
        "operation",
    }
)
_WORD_MAX = 0xFFFF
_DOMAIN_SIZE = 1 << 16
_QUEUE_SIZE = 1024
_RANK_BATCH_SIZE = 1024


def compose_speck_ranked_queue(
    stage1_order: Sequence[int],
    stage2_orders: Mapping[int, Sequence[int]],
    *,
    parent_width: int = 32,
    child_width: int = 32,
) -> tuple[tuple[int, int], ...]:
    """Compose the frozen SPECK hierarchy order into chronological key tuples.

    The SPECK evaluator orders paths by cumulative one-based stage rank and
    then by the stage-order prefix ``(parent, child)``. Candidate tuples passed
    to SAT are converted to chronological ``(child, parent)`` order only after
    this tie break has been applied.
    """

    parents = _rank_prefix(stage1_order, parent_width, "SPECK Stage 1")
    if set(stage2_orders) != set(parents):
        raise ValueError("SPECK Stage 2 orders must match the retained parents")

    ordered_paths: list[tuple[int, tuple[int, int]]] = []
    for parent_rank, parent in enumerate(parents, start=1):
        children = _rank_prefix(
            stage2_orders[parent], child_width, f"SPECK Stage 2 parent {parent}"
        )
        for child_rank, child in enumerate(children, start=1):
            ordered_paths.append((parent_rank + child_rank, (parent, child)))

    ordered_paths.sort(key=lambda item: (item[0], item[1]))
    chronological = tuple((child, parent) for _, (parent, child) in ordered_paths)
    _validate_unique_queue(chronological, parent_width * child_width)
    return chronological


def compose_simon_ranked_queue(
    stage1_order: Sequence[int],
    stage2_orders: Mapping[int, Sequence[int]],
    *,
    parent_width: int = 8,
    child_width: int = 128,
) -> tuple[tuple[int, int], ...]:
    """Compose the frozen SIMON 8-parent queue and its original rank tie rule.

    This reproduces the SIMON builder's lexicographic
    ``(r1+r2, r1, r2, parent, child)`` ordering, followed by conversion to
    chronological ``(k8, k9)`` tuples and stable duplicate suppression.
    """

    parents = _rank_prefix(stage1_order, parent_width, "SIMON Stage 1")
    if set(stage2_orders) != set(parents):
        raise ValueError("SIMON Stage 2 orders must match the retained parents")

    raw_tuples: list[tuple[int, int, int, int, int]] = []
    for parent_rank, parent in enumerate(parents):
        children = _rank_prefix(
            stage2_orders[parent], child_width, f"SIMON Stage 2 parent {parent}"
        )
        for child_rank, child in enumerate(children):
            raw_tuples.append(
                (parent_rank + child_rank, parent_rank, child_rank, parent, child)
            )

    raw_tuples.sort()
    result: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for _, _, _, parent, child in raw_tuples:
        candidate = (child, parent)
        if candidate not in seen:
            seen.add(candidate)
            result.append(candidate)
    _validate_unique_queue(tuple(result), parent_width * child_width)
    return tuple(result)


def rank_queue(
    public: dict[str, Any],
    job: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, Any]:
    """Build one frozen policy's 1,024-tuple queue from public observations.

    ``public`` must be the stripped object produced by ``public_case``. This
    function rejects extra fields so secret keys, round keys, and verification
    observations cannot accidentally enter the ranking boundary.
    """

    cipher, policy, rounds, pairs, observations = _validate_public_job(public, job)
    started = time.perf_counter()

    if cipher == "speck32_64":
        ranked, component_diagnostics, phase_seconds = _rank_speck(
            public, pairs, observations, policy, paths
        )
    elif policy == "classical_8parent":
        ranked, component_diagnostics, phase_seconds = _rank_simon_classical(
            pairs, paths
        )
    elif policy == "bao_ddt_hybrid_8parent":
        ranked, component_diagnostics, phase_seconds = _rank_simon_bao_hybrid(
            pairs, paths
        )
    else:  # _validate_public_job rejects this; keep dispatch fail-closed.
        raise ValueError(f"Unsupported queue policy: {cipher}/{policy}")

    checked = validate_queue([list(pair) for pair in ranked])
    if len(checked) != _QUEUE_SIZE:
        raise ValueError("Frozen ranking policy did not produce exactly 1,024 tuples")

    phase_seconds["rank_queue_internal_seconds"] = _elapsed(started)
    return {
        "ranked": [list(candidate) for candidate in checked],
        "candidate_tuple_count": len(checked),
        "cipher": cipher,
        "policy": policy,
        "rounds": rounds,
        "component_diagnostics": component_diagnostics,
        "phase_seconds": phase_seconds,
    }


def _rank_speck(
    public: dict[str, Any],
    pairs: tuple[tuple[tuple[int, int], tuple[int, int]], ...],
    observations: tuple[tuple[tuple[int, int], tuple[int, int]], ...],
    policy: str,
    paths: dict[str, Path],
) -> tuple[tuple[tuple[int, int], ...], dict[str, Any], dict[str, float]]:
    (
        torch,
        PairGroupRanker,
        rank_group,
        rank_screened,
        partial_decrypt,
        AttackInstance,
        AttackPair,
    ) = _load_speck_dependencies()
    _configure_torch_threads(torch)
    device = torch.device("mps")
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "Frozen SPECK queue policy requires MPS; CPU fallback is forbidden"
        )

    stage1_path = _require_file(paths, "speck_stage1")
    stage2_path = _require_file(paths, "speck_stage2")
    stage1_classical_path = (
        _require_file(paths, "speck_classical")
        if policy == "neural_classical_hybrid"
        else None
    )

    load_started = time.perf_counter()
    stage1_model = _load_pair_group_ranker(torch, PairGroupRanker, stage1_path, device)
    if policy == "screened_neural":
        stage2_model = _load_pair_group_ranker(
            torch, PairGroupRanker, stage2_path, device
        )
        classical_table = None
    else:
        stage2_model = None
        from neural_sat_speck.attacks.classical_ranking import (
            ClassicalDistributionTable,
        )

        assert stage1_classical_path is not None
        classical_table = ClassicalDistributionTable.load(stage1_classical_path)
    load_seconds = _elapsed(load_started)

    instance = _speck_rank_only_instance(
        public, pairs, observations, AttackInstance, AttackPair
    )
    ranking_started = time.perf_counter()
    first = rank_group(
        instance,
        stage1_model,
        candidate_batch_size=_RANK_BATCH_SIZE,
        device=device,
    )
    _validate_full_domain_order(first.candidate_order, "SPECK Stage 1")
    parents = tuple(int(v) for v in first.candidate_order[:32])

    second_orders: dict[int, Sequence[int]] = {}
    second_details: list[dict[str, Any]] = []
    descent_seconds = 0.0
    stage2_seconds = 0.0
    for parent_rank, parent in enumerate(parents, start=1):
        descent_started = time.perf_counter()
        descended = partial_decrypt(
            instance,
            parent,
            expected_next_subkey=0,
            label=f"{public['instance_id']}-rank-only-r5-{parent:04x}",
        )
        descent_seconds += _elapsed(descent_started)

        rank_started = time.perf_counter()
        if policy == "screened_neural":
            assert stage2_model is not None
            second = rank_screened(
                descended,
                stage2_model,
                screen_pairs=16,
                screen_top_k=512,
                candidate_batch_size=_RANK_BATCH_SIZE,
                device=device,
            )
        else:
            assert classical_table is not None
            from neural_sat_speck.attacks.classical_ranking import (
                rank_classical_subkeys,
            )

            second = rank_classical_subkeys(descended, classical_table)
        measured_rank_seconds = _elapsed(rank_started)
        stage2_seconds += measured_rank_seconds
        _validate_full_domain_order(
            second.candidate_order, f"SPECK Stage 2 parent {parent}"
        )
        second_orders[parent] = second.candidate_order
        second_details.append(
            {
                "parent_key": parent,
                "parent_rank": parent_rank,
                "candidate_domain_size": len(second.candidate_order),
                "candidate_batch_size": _RANK_BATCH_SIZE
                if policy == "screened_neural"
                else None,
                "screen_pairs": 16 if policy == "screened_neural" else None,
                "screen_top_k": 512 if policy == "screened_neural" else None,
                "ranker_total_seconds": _metric(second.total_seconds),
                "invocation_wall_seconds": measured_rank_seconds,
                "partial_decryption_seconds": _metric(second.partial_decryption_seconds)
                if hasattr(second, "partial_decryption_seconds")
                else None,
                "inference_seconds": _metric(second.inference_seconds)
                if hasattr(second, "inference_seconds")
                else None,
                "sorting_seconds": _metric(second.sorting_seconds)
                if hasattr(second, "sorting_seconds")
                else None,
                "top_children": [int(v) for v in second.candidate_order[:32]],
            }
        )

    compose_started = time.perf_counter()
    ranked = compose_speck_ranked_queue(
        first.candidate_order,
        second_orders,
        parent_width=32,
        child_width=32,
    )
    compose_seconds = _elapsed(compose_started)
    ranking_wall = _elapsed(ranking_started)
    diagnostics = {
        "stage1": {
            "candidate_domain_size": len(first.candidate_order),
            "pair_count": first.pair_count,
            "score_aggregation": first.score_aggregation,
            "candidate_batch_size": _RANK_BATCH_SIZE,
            "ranker_total_seconds": _metric(first.total_seconds),
            "partial_decryption_seconds": _metric(first.partial_decryption_seconds),
            "inference_seconds": _metric(first.inference_seconds),
            "sorting_seconds": _metric(first.sorting_seconds),
            "retained": [
                {"key": int(v), "rank": i} for i, v in enumerate(parents, start=1)
            ],
        },
        "stage2_by_parent": second_details,
        "ordering": {
            "rule": "sum_of_one_based_stage_ranks_then_stage_order_parent_child_tuple",
            "stored_tuple_order": "chronological_child_parent",
            "queue_size": len(ranked),
        },
    }
    phases = {
        "model_or_table_load_seconds": load_seconds,
        "ranking_invocations_seconds": ranking_wall,
        "stage1_ranker_seconds": _metric(first.total_seconds),
        "stage2_ranker_seconds": stage2_seconds,
        "inter_stage_descent_seconds": descent_seconds,
        "queue_compose_seconds": compose_seconds,
    }
    return ranked, diagnostics, phases


def _rank_simon_classical(
    pairs: tuple[tuple[tuple[int, int], tuple[int, int]], ...],
    paths: dict[str, Path],
) -> tuple[tuple[tuple[int, int], ...], dict[str, Any], dict[str, float]]:
    FullPredictedDdt, rank_classical_domain, rank_stage1_ondemand = (
        _load_simon_classical_dependencies()
    )
    ddt_path = _require_file(paths, "simon_ddt")
    tool_path = _require_executable(paths, "simon_tool")

    load_started = time.perf_counter()
    table = FullPredictedDdt(ddt_path, 7)
    table_load_seconds = _elapsed(load_started)

    ranking_started = time.perf_counter()
    first, first_detail = rank_stage1_ondemand(
        pairs,
        tool_path=tool_path,
        ddt_7r_path=ddt_path,
        screen_pairs=16,
        full_pairs=64,
        rescore_top=512,
        batch_size=_RANK_BATCH_SIZE,
    )
    _validate_prefix_order(first.order, 512, "SIMON classical Stage 1")
    parents = tuple(int(v) for v in first.order[:8])
    second_rankings: dict[int, Any] = {}
    second_seconds = 0.0
    for parent in parents:
        second = rank_classical_domain(
            pairs,
            table,
            final_key_candidate=parent,
            expected_rounds=7,
            screen_pairs=16,
            full_pairs=64,
            rescore_top=512,
            batch_size=_RANK_BATCH_SIZE,
        )
        _validate_prefix_order(
            second.order, 512, f"SIMON classical Stage 2 parent {parent}"
        )
        second_rankings[parent] = second
        second_seconds += _metric(second.total_seconds)

    stage2_orders = {
        parent: ranking.order for parent, ranking in second_rankings.items()
    }
    compose_started = time.perf_counter()
    ranked = compose_simon_ranked_queue(
        first.order, stage2_orders, parent_width=8, child_width=128
    )
    compose_seconds = _elapsed(compose_started)
    ranking_wall = _elapsed(ranking_started)

    diagnostics = {
        "stage1": _simon_ranking_summary(first, first_detail, 8),
        "stage2_by_parent": [
            {
                "parent_key": parent,
                "parent_rank": rank,
                **_simon_ranking_summary(second_rankings[parent], None, 128),
            }
            for rank, parent in enumerate(parents, start=1)
        ],
        "ordering": {
            "rule": "sum_of_zero_based_stage_ranks_then_parent_rank_child_rank_parent_child",
            "stored_tuple_order": "chronological_k8_k9",
            "queue_size": len(ranked),
        },
    }
    phases = {
        "ddt_load_seconds": table_load_seconds,
        "ranking_invocations_seconds": ranking_wall,
        "stage1_ranker_seconds": _metric(first.total_seconds),
        "stage1_tool_wall_seconds": _metric(first_detail.get("tool_wall_seconds", 0.0)),
        "stage2_ranker_seconds": second_seconds,
        "queue_compose_seconds": compose_seconds,
    }
    return ranked, diagnostics, phases


def _rank_simon_bao_hybrid(
    pairs: tuple[tuple[tuple[int, int], tuple[int, int]], ...],
    paths: dict[str, Path],
) -> tuple[tuple[tuple[int, int], ...], dict[str, Any], dict[str, float]]:
    if importlib.util.find_spec("tensorflow") is not None:
        raise RuntimeError(
            "SIMON Bao ranking must use the pinned TensorFlow subprocess; "
            "the current ranking interpreter exposes TensorFlow"
        )

    _, generate_hybrid_queue = _load_simon_hybrid_dependency()
    ddt_path = _require_file(paths, "simon_ddt")
    tf_python = _require_executable(paths, "tf_python")
    vv9_path = _require_file(paths, "bao_vv9")
    worker_script = _require_file(
        paths, "source:scripts/run_queue_order_neural_ranking.py"
    )
    if vv9_path.name != "ND_VV_Simon32_9R.h5":
        raise ValueError("Pinned Bao checkpoint must be the frozen VV9 file")

    started = time.perf_counter()
    result = generate_hybrid_queue(
        pairs,
        vv9_model=None,
        ddt_7r_path=ddt_path,
        tf_python_path=tf_python,
        checkpoint_dir=vv9_path.parent,
        worker_script_path=worker_script,
        ranking_time_limit=120.0,
        return_diagnostics=True,
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("SIMON Bao ranker did not return queue diagnostics")
    queue_raw, raw_diagnostics = result
    ranked = _validate_ranked_tuples(queue_raw, "SIMON Bao-DDT hybrid")
    if len(ranked) != _QUEUE_SIZE:
        raise ValueError("SIMON Bao-DDT hybrid did not produce 1,024 tuples")

    diag = {
        key: _metric(raw_diagnostics[key])
        for key in (
            "ranking_wall_seconds",
            "stage1_seconds",
            "stage2_seconds",
            "peak_rss_gib",
        )
        if key in raw_diagnostics
    }
    diag["full_pairs_used"] = int(raw_diagnostics.get("full_pairs_used", len(pairs)))
    diag["candidate_tuples_produced"] = int(
        raw_diagnostics.get("candidate_tuples_produced", len(ranked))
    )
    if (
        diag["full_pairs_used"] != 64
        or diag["candidate_tuples_produced"] != _QUEUE_SIZE
    ):
        raise ValueError("SIMON Bao worker diagnostics disagree with frozen profile")

    wall = _elapsed(started)
    components = {
        "stage1": {"candidate_domain_size": _DOMAIN_SIZE, "retained_parent_count": 8},
        "stage2": {
            "candidate_domain_size_per_parent": _DOMAIN_SIZE,
            "retained_child_count_per_parent": 128,
        },
        "worker": diag,
        "ordering": {
            "rule": "sum_of_zero_based_stage_ranks_then_parent_rank_child_rank_parent_child",
            "stored_tuple_order": "chronological_k8_k9",
            "queue_size": len(ranked),
        },
    }
    phases = {
        "pinned_tf_ranker_process_seconds": _metric(
            raw_diagnostics.get("ranking_wall_seconds", wall)
        ),
        "rank_queue_internal_seconds": wall,
    }
    return ranked, components, phases


def _load_speck_dependencies() -> tuple[Any, ...]:
    import torch

    from neural_sat_speck.attacks.group_ranking import rank_pair_group_subkeys
    from neural_sat_speck.attacks.multi_round_instance import partial_decrypt_instance
    from neural_sat_speck.attacks.screened_group_ranking import (
        rank_pair_group_subkeys_screened,
    )
    from neural_sat_speck.data.attack import AttackInstance, AttackPair
    from neural_sat_speck.neural.group_model import PairGroupRanker

    return (
        torch,
        PairGroupRanker,
        rank_pair_group_subkeys,
        rank_pair_group_subkeys_screened,
        partial_decrypt_instance,
        AttackInstance,
        AttackPair,
    )


def _load_simon_classical_dependencies() -> tuple[Any, ...]:
    from neural_sat_speck.simon.classical_ranking import (
        FullPredictedDdt,
        rank_classical_domain,
        rank_stage1_ondemand,
    )

    return FullPredictedDdt, rank_classical_domain, rank_stage1_ondemand


def _load_simon_hybrid_dependency() -> tuple[Any, Any]:
    # This existing builder delegates Bao inference to the frozen TensorFlow
    # subprocess when TensorFlow is absent from the ranking interpreter.
    from neural_sat_speck.simon.method_worker import generate_hybrid_8parent_queue

    return None, generate_hybrid_8parent_queue


def _configure_torch_threads(torch: Any) -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise


def _load_pair_group_ranker(
    torch: Any, model_type: Any, path: Path, device: Any
) -> Any:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Frozen SPECK checkpoint must contain a mapping: {path}")
    variant = payload.get("variant", "b1_mean")
    if variant != "b1_mean":
        raise ValueError(f"Frozen SPECK queue policy requires b1_mean, got {variant!r}")
    model = model_type(variant=variant)
    state = payload["state_dict"] if "state_dict" in payload else payload
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _speck_rank_only_instance(
    public: dict[str, Any],
    pairs: tuple[tuple[tuple[int, int], tuple[int, int]], ...],
    observations: tuple[tuple[tuple[int, int], tuple[int, int]], ...],
    AttackInstance: Any,
    AttackPair: Any,
) -> Any:
    attack_pairs = tuple(
        AttackPair(
            plaintext0=observations[2 * i][0],
            plaintext1=observations[2 * i + 1][0],
            ciphertext0=pair[0],
            ciphertext1=pair[1],
        )
        for i, pair in enumerate(pairs)
    )
    # The ranking API uses AttackInstance as a data container. These fixed
    # sentinels are deliberately unrelated to the assigned key and are never
    # used in score calculation or queue ordering.
    return AttackInstance(
        instance_id=str(public["instance_id"]),
        master_key=(0, 0, 0, 0),
        final_round_subkey=0,
        pairs=attack_pairs,
    )


def _validate_public_job(
    public: dict[str, Any], job: dict[str, Any]
) -> tuple[
    str,
    str,
    int,
    tuple[tuple[tuple[int, int], tuple[int, int]], ...],
    tuple[tuple[tuple[int, int], tuple[int, int]], ...],
]:
    if not isinstance(public, dict) or set(public) != _PUBLIC_FIELDS:
        raise ValueError("Ranking input must contain only stripped public fields")
    if not isinstance(job, dict):
        raise TypeError("job must be a mapping")
    if set(job) != _RANK_JOB_FIELDS:
        raise ValueError(
            "Ranking job must contain only its fixed public identity fields"
        )
    cipher, policy = job.get("cipher"), job.get("policy")
    rounds = job.get("rounds")
    if not isinstance(cipher, str) or not isinstance(policy, str):
        raise ValueError("job cipher and policy must be strings")
    if type(rounds) is not int:
        raise ValueError("job rounds must be an integer")
    allowed = {(p[0], p[1]): p[2] for p in POLICIES}
    if allowed.get((cipher, policy)) != rounds:
        raise ValueError("job does not match the fixed queue-order policy registry")
    requests = {(p[0], p[1]): p[3] for p in POLICIES}
    request = job.get("candidate_request_seconds")
    if type(request) is not float or request != requests[(cipher, policy)]:
        raise ValueError("job candidate request differs from the fixed policy registry")
    if public.get("cipher") != cipher or public.get("rounds") != rounds:
        raise ValueError("public attack observations do not match the job")
    if not isinstance(public.get("instance_id"), str) or not public["instance_id"]:
        raise ValueError("public instance_id must be a nonempty string")
    if job.get("operation") != "rank":
        raise ValueError("rank_queue accepts preparation jobs only")
    key_index = job.get("key_index")
    if type(key_index) is not int or not 0 <= key_index < 8:
        raise ValueError("job key_index must be in the opened eight-key cohort")
    expected_case_id = f"{cipher}_k{key_index}_{policy}"
    if (
        job.get("case_id") != expected_case_id
        or job.get("job_id") != expected_case_id + "__prepare"
    ):
        raise ValueError("Ranking job identity differs from the fixed cohort schedule")

    raw_pairs = public.get("ciphertext_pairs")
    raw_observations = public.get("known_observations")
    if not isinstance(raw_pairs, (list, tuple)) or len(raw_pairs) != 64:
        raise ValueError("Ranking requires exactly64 public ciphertext pairs")
    if not isinstance(raw_observations, (list, tuple)) or len(raw_observations) != 128:
        raise ValueError("Ranking requires exactly128 public known observations")

    pair_values: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for pair in raw_pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Ciphertext pair must contain two blocks")
        pair_values.append((_block(pair[0]), _block(pair[1])))
    pairs = tuple(pair_values)

    observation_values: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for item in raw_observations:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("Known observation must contain plaintext and ciphertext")
        observation_values.append((_block(item[0]), _block(item[1])))
    observations = tuple(observation_values)
    for index, (ct0, ct1) in enumerate(pairs):
        if observations[2 * index][1] != ct0 or observations[2 * index + 1][1] != ct1:
            raise ValueError("ciphertext pair order disagrees with public observations")
    return cipher, policy, rounds, pairs, observations


def _rank_prefix(order: Sequence[int], width: int, label: str) -> tuple[int, ...]:
    if type(width) is not int or width <= 0:
        raise ValueError("rank prefix width must be positive")
    if (
        isinstance(order, (str, bytes, bytearray))
        or not hasattr(order, "__len__")
        or not hasattr(order, "__getitem__")
    ):
        raise ValueError(f"{label} must return an indexable ranked sequence")
    if len(order) < width:
        raise ValueError(f"{label} returned fewer than {width} ranked candidates")
    prefix = tuple(_word(value, f"{label} candidate") for value in order[:width])
    if len(set(prefix)) != width:
        raise ValueError(f"{label} prefix contains duplicate candidates")
    return prefix


def _validate_full_domain_order(order: Sequence[int], label: str) -> None:
    if (
        isinstance(order, (str, bytes, bytearray))
        or not hasattr(order, "__len__")
        or not hasattr(order, "__getitem__")
        or len(order) != _DOMAIN_SIZE
    ):
        raise ValueError(f"{label} must rank the full 16-bit domain")
    values = tuple(_word(value, f"{label} candidate") for value in order)
    if set(values) != set(range(_DOMAIN_SIZE)):
        raise ValueError(f"{label} is not a permutation of the 16-bit domain")


def _validate_prefix_order(order: Sequence[int], count: int, label: str) -> None:
    if len(order) != count:
        raise ValueError(f"{label} must contain its frozen top-{count} shortlist")
    values = tuple(_word(value, f"{label} candidate") for value in order)
    if len(set(values)) != count:
        raise ValueError(f"{label} shortlist contains duplicate candidates")


def _validate_ranked_tuples(
    raw_queue: Sequence[Sequence[int]], label: str
) -> tuple[tuple[int, int], ...]:
    if not isinstance(raw_queue, (list, tuple)):
        raise TypeError(f"{label} queue must be a sequence")
    result = tuple(
        (
            _word(candidate[0], f"{label} tuple[0]"),
            _word(candidate[1], f"{label} tuple[1]"),
        )
        if isinstance(candidate, (list, tuple)) and len(candidate) == 2
        else (_raise_tuple(label))
        for candidate in raw_queue
    )
    _validate_unique_queue(result, len(result))
    return result


def _validate_unique_queue(
    candidates: tuple[tuple[int, int], ...], expected_count: int
) -> None:
    if len(candidates) != expected_count:
        raise ValueError(
            f"Candidate queue length {len(candidates)} differs from expected {expected_count}"
        )
    for candidate in candidates:
        if len(candidate) != 2:
            raise ValueError(
                "Candidate must contain exactly two chronological key words"
            )
        _word(candidate[0], "candidate key word")
        _word(candidate[1], "candidate key word")
    if len(set(candidates)) != len(candidates):
        raise ValueError("Candidate queue contains duplicates")


def _simon_ranking_summary(ranking: Any, details: Any, keep: int) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "candidate_domain_size": _DOMAIN_SIZE,
        "returned_order_size": len(ranking.order),
        "screen_pairs_used": int(ranking.screen_pairs_used),
        "full_pairs_used": int(ranking.full_pairs_used),
        "total_seconds": _metric(ranking.total_seconds),
        "row_load_seconds": _metric(ranking.row_load_seconds),
        "screen_seconds": _metric(ranking.screen_seconds),
        "rescore_seconds": _metric(ranking.rescore_seconds),
        "sort_seconds": _metric(ranking.sort_seconds),
        "unique_left_rows": int(ranking.unique_left_rows),
        "screen_tie_count": int(ranking.screen_tie_count),
        "full_tie_count": int(ranking.full_tie_count),
        "screen_collision_count": int(ranking.screen_collision_count),
        "full_collision_count": int(ranking.full_collision_count),
        "retained": [
            {"key": int(value), "rank": index}
            for index, value in enumerate(ranking.order[:keep], start=1)
        ],
    }
    if details is not None:
        summary["stage1_detail"] = {
            key: _metric(value)
            for key, value in details.items()
            if key.endswith("_seconds") or key.endswith("_rss_gib")
        }
    return summary


def _require_file(paths: dict[str, Path], name: str) -> Path:
    path = paths.get(name)
    if not isinstance(path, Path) or not path.is_file():
        raise FileNotFoundError(f"Validated pinned file path is missing: {name}")
    return path


def _require_executable(paths: dict[str, Path], name: str) -> Path:
    path = _require_file(paths, name)
    if not os.access(path, os.X_OK):
        raise PermissionError(f"Pinned tool is not executable: {name}")
    return path


def _word(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer")
    result = int(value)
    if not 0 <= result <= _WORD_MAX:
        raise ValueError(f"{label} must be a 16-bit integer")
    return result


def _block(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Observation block must contain two words")
    return _word(value[0], "block word"), _word(value[1], "block word")


def _raise_tuple(label: str) -> Any:
    raise ValueError(f"{label} candidate must contain two key words")


def _metric(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Ranking telemetry must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("Ranking telemetry must be finite and nonnegative")
    return result


def _elapsed(started: float) -> float:
    return _metric(time.perf_counter() - started)
