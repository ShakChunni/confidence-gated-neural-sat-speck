"""Truth-blind Stage 1 / Stage 2 ranking adapter for D-055.

The public API accepts only known ciphertext pairs.  It does not accept a key,
plaintext, fresh verification blocks, solver configuration, or solver handle.
All loaded rankers/tables belong to one call (and one study cell); the caller's
300-second parent clock owns every child process and every ranking phase.
"""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.metadata
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import types
from collections.abc import Callable, Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

from neural_sat_speck.backend.guard import (
    atomic_json,
    canonical_bytes,
    read_checked_json,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CIPHERS = {
    "speck32_64": {"rounds": 7, "parents": 32, "children": 32},
    "simon32_64": {"rounds": 10, "parents": 8, "children": 128},
}
_DOMAIN_SIZE = 1 << 16
_SIMON_RESCORING_SIZE = 512
_BATCH_SIZE = 1024
_TF_INTRA_OP_THREADS = 8
_TF_INTER_OP_THREADS = 1
_NUMERIC_THREADS = 1
_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "8",
    "TF_NUM_INTEROP_THREADS": "1",
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "CUDA_VISIBLE_DEVICES": "-1",
}
_FORBIDDEN_TF_IMPORTS = (
    "pycryptosat",
    "neural_sat_speck.simon.sat_model",
    "neural_sat_speck.simon.verification",
    "neural_sat_speck.protocol",
)

PhaseCallback = Callable[[str, Mapping[str, Any]], None]


def rank_queue(
    cipher: str,
    ciphertext_pairs: Sequence[Sequence[Sequence[int]]],
    trailing_keys: int,
    paths: Mapping[str, Path | str],
    directory: Path | str,
    phase_callback: PhaseCallback | None = None,
) -> dict[str, Any]:
    """Rank the fixed D-055 one-key or two-key candidate queue.

    ``ciphertext_pairs`` has 64 entries, each ``(ciphertext0, ciphertext1)``;
    plaintexts are intentionally absent.  The output queue stores the
    chronological word tuple expected by the solver: a singleton final key in
    the one-stage arm, or ``(penultimate_key, final_key)`` in the two-stage
    arm.  The first ``32`` SPECK or ``8`` SIMON final-key candidates are
    returned as ``stage1_parents`` and persisted before Stage 2 starts.

    The supplied callback receives ``stage1_start``, ``stage1_complete``, and
    (for two-key cells) ``stage2_start``.  A caller can persist the completed
    Stage 1 receipt synchronously before the callback for Stage 2 is emitted.
    This function never launches a solver and never assigns its own deadline.
    """

    profile = _CIPHERS.get(cipher)
    if profile is None:
        raise ValueError(f"Unsupported ranking-depth cipher: {cipher!r}")
    if type(trailing_keys) is not int or trailing_keys not in (1, 2):
        raise ValueError("trailing_keys must be exactly 1 or 2")
    if phase_callback is not None and not callable(phase_callback):
        raise TypeError("phase_callback must be callable or None")
    checked_pairs = _validate_ciphertext_pairs(ciphertext_pairs)
    checked_paths = _check_paths(paths, cipher, trailing_keys)
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage1_path = output_dir / "stage1.json"
    queue_path = output_dir / "queue.json"
    if stage1_path.exists() or queue_path.exists():
        raise FileExistsError("Ranking output already exists; cells cannot resume")

    stage1_started = time.perf_counter()
    _notify(phase_callback, "stage1_start", {})
    if cipher == "speck32_64":
        stage1_order, stage1_detail, stage1_imports = _rank_speck_stage1(
            checked_pairs, checked_paths
        )
    else:
        stage1_order, stage1_detail, stage1_imports = _rank_simon_stage1(
            checked_pairs, checked_paths, output_dir
        )
    stage1_wall = _elapsed(stage1_started)
    parents = tuple(int(word) for word in stage1_order[: profile["parents"]])
    _validate_word_prefix(parents, profile["parents"], "Stage 1 parent prefix")
    parents_sha256 = _sha256_json(list(parents))

    stage1_receipt = {
        "schema": "backend_ranking_depth_stage1_v1",
        "cipher": cipher,
        "rounds": profile["rounds"],
        "trailing_keys": trailing_keys,
        "screened_domain_size": _DOMAIN_SIZE,
        "stage1_order_size": len(stage1_order),
        "stage1_parent_count": len(parents),
        "stage1_parents": list(parents),
        "stage1_prefix_sha256": parents_sha256,
        "stage1_order_prefix": [
            int(word) for word in stage1_order[: max(512, len(parents))]
        ],
        "stage1_wall_seconds": stage1_wall,
        "ranker": stage1_detail,
        "code_dependencies": stage1_imports,
        "assets": _assets_for_stage(checked_paths, "stage1"),
        "adapter_sha256": _sha256_file(Path(__file__)),
    }
    atomic_json(stage1_path, stage1_receipt)
    _notify(phase_callback, "stage1_complete", stage1_receipt)

    stage2_wall = 0.0
    stage2_receipt: dict[str, Any] | None = None
    if trailing_keys == 1:
        ordered_queue = tuple((parent,) for parent in parents)
    else:
        stage2_started = time.perf_counter()
        _notify(phase_callback, "stage2_start", {})
        if cipher == "speck32_64":
            ordered_queue, stage2_detail, stage2_imports = _rank_speck_stage2(
                checked_pairs, parents, checked_paths
            )
        else:
            ordered_queue, stage2_detail, stage2_imports = _rank_simon_stage2(
                checked_pairs, stage1_order, parents, checked_paths
            )
        stage2_wall = _elapsed(stage2_started)
        stage2_receipt = {
            "candidate_parent_count": len(parents),
            "candidate_child_count_per_parent": profile["children"],
            "queue_size": len(ordered_queue),
            "stage2_wall_seconds": stage2_wall,
            "ranker": stage2_detail,
            "code_dependencies": stage2_imports,
            "assets": _assets_for_stage(checked_paths, "stage2"),
        }

    _validate_queue(ordered_queue, trailing_keys, profile)
    _validate_ranked_queue_parents(ordered_queue, parents)
    receipt = {
        "schema": "backend_ranking_depth_ranking_receipt_v1",
        "cipher": cipher,
        "rounds": profile["rounds"],
        "trailing_keys": trailing_keys,
        "candidate_batch_size": _BATCH_SIZE,
        "stage1_parents": list(parents),
        "stage1_wall_seconds": stage1_wall,
        "stage2_wall_seconds": stage2_wall,
        "queue": [list(candidate) for candidate in ordered_queue],
        "queue_sha256": _sha256_json([list(candidate) for candidate in ordered_queue]),
        # The worker may be interrupted after this atomic queue write but
        # before it enriches the record; retain the prefix binding here too.
        "stage1_sha256": parents_sha256,
        "ranking_receipts": {
            "stage1": stage1_receipt,
            "stage2": stage2_receipt,
            "tensorflow_worker": _read_optional_json(
                output_dir / "tensorflow_stage1_worker.json"
            ),
        },
    }
    atomic_json(queue_path, receipt)
    return receipt


def _rank_speck_stage1(
    ciphertext_pairs: tuple[tuple[tuple[int, int], tuple[int, int]], ...],
    paths: Mapping[str, Path],
) -> tuple[tuple[int, ...], dict[str, Any], dict[str, str]]:
    imports_before = set(sys.modules)
    (
        torch,
        rank_pair_group_subkeys,
        _partial_decrypt_instance,
        AttackInstance,
        AttackPair,
        PairGroupRanker,
        load_pair_group_ranker,
    ) = _load_speck_dependencies()

    _configure_torch(torch)
    device = torch.device("mps")
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "D-055 SPECK ranking requires MPS; CPU fallback is forbidden"
        )
    if not hasattr(torch, "mps") or not callable(
        getattr(torch.mps, "synchronize", None)
    ):
        raise RuntimeError("D-055 SPECK ranking requires MPS boundary synchronization")

    model_path = paths["speck_stage1"]
    model_load_started = time.perf_counter()
    model = load_pair_group_ranker(torch, PairGroupRanker, model_path, device)
    model_load_seconds = _elapsed(model_load_started)
    instance = _rank_only_speck_instance(ciphertext_pairs, AttackInstance, AttackPair)
    ranking = rank_pair_group_subkeys(
        instance,
        model,
        candidate_batch_size=_BATCH_SIZE,
        device=device,
    )
    torch.mps.synchronize()
    order = _validate_word_order(
        ranking.candidate_order, _DOMAIN_SIZE, "SPECK B1 Stage 1"
    )
    detail = {
        "ranker": "production_rank_pair_group_subkeys",
        "model_variant": "b1_mean",
        "candidate_domain_size": len(order),
        "pair_count": ranking.pair_count,
        "group_size": ranking.group_size,
        "candidate_batch_size": ranking.candidate_batch_size,
        "score_aggregation": ranking.score_aggregation,
        "model_load_seconds": model_load_seconds,
        "ranker_total_seconds": _finite_seconds(ranking.total_seconds),
        "partial_decryption_seconds": _finite_seconds(
            ranking.partial_decryption_seconds
        ),
        "inference_seconds": _finite_seconds(ranking.inference_seconds),
        "sorting_seconds": _finite_seconds(ranking.sorting_seconds),
        "device": str(device),
        "torch_intra_op_threads": 1,
        "torch_inter_op_threads": 1,
    }
    return order, detail, _source_imports(imports_before)


def _rank_speck_stage2(
    ciphertext_pairs: tuple[tuple[tuple[int, int], tuple[int, int]], ...],
    parents: tuple[int, ...],
    paths: Mapping[str, Path],
) -> tuple[tuple[tuple[int, int], ...], dict[str, Any], dict[str, str]]:
    imports_before = set(sys.modules)
    (
        torch,
        ClassicalDistributionTable,
        rank_classical_subkeys,
        partial_decrypt_instance,
        AttackInstance,
        AttackPair,
        compose_speck_ranked_queue,
    ) = _load_speck_stage2_dependencies()

    _configure_torch(torch)
    device = torch.device("mps")
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "D-055 SPECK Stage 2 requires MPS; CPU fallback is forbidden"
        )
    if not hasattr(torch, "mps") or not callable(
        getattr(torch.mps, "synchronize", None)
    ):
        raise RuntimeError("D-055 SPECK ranking requires MPS boundary synchronization")

    table_load_started = time.perf_counter()
    table = ClassicalDistributionTable.load(paths["speck_classical"])
    table_load_seconds = _elapsed(table_load_started)
    instance = _rank_only_speck_instance(ciphertext_pairs, AttackInstance, AttackPair)

    orders: dict[int, tuple[int, ...]] = {}
    per_parent: list[dict[str, Any]] = []
    descent_seconds = 0.0
    ranker_seconds = 0.0
    for parent_rank, parent in enumerate(parents, start=1):
        descent_started = time.perf_counter()
        descended = partial_decrypt_instance(
            instance,
            parent,
            expected_next_subkey=0,
            label=f"d055-rank-only-r5-{parent:04x}",
        )
        parent_descent_seconds = _elapsed(descent_started)
        descent_seconds += parent_descent_seconds

        rank_started = time.perf_counter()
        ranking = rank_classical_subkeys(descended, table)
        parent_ranker_seconds = _elapsed(rank_started)
        ranker_seconds += parent_ranker_seconds
        order = _validate_word_order(
            ranking.candidate_order,
            _DOMAIN_SIZE,
            f"SPECK sparse-R5 Stage 2 parent {parent:04x}",
        )
        orders[parent] = order
        per_parent.append(
            {
                "parent_key": parent,
                "parent_rank": parent_rank,
                "candidate_domain_size": len(order),
                "partial_decryption_seconds": parent_descent_seconds,
                "ranker_total_seconds": _finite_seconds(ranking.total_seconds),
                "invocation_wall_seconds": parent_ranker_seconds,
                "top_children": list(order[:32]),
            }
        )

    torch.mps.synchronize()

    queue = compose_speck_ranked_queue(
        parents,
        orders,
        parent_width=32,
        child_width=32,
    )
    queue = tuple((int(child), int(parent)) for child, parent in queue)
    detail = {
        "ranker": "production_rank_classical_subkeys",
        "table": "sparseR5",
        "table_load_seconds": table_load_seconds,
        "inter_stage_descent_seconds": descent_seconds,
        "stage2_ranker_seconds": ranker_seconds,
        "candidate_batch_size": None,
        "device": str(device),
        "torch_intra_op_threads": 1,
        "torch_inter_op_threads": 1,
        "per_parent": per_parent,
        "ordering": "existing_speck_one_based_sum_rank_then_parent_child",
    }
    return queue, detail, _source_imports(imports_before)


def _rank_simon_stage1(
    ciphertext_pairs: tuple[tuple[tuple[int, int], tuple[int, int]], ...],
    paths: Mapping[str, Path],
    directory: Path,
) -> tuple[tuple[int, ...], dict[str, Any], dict[str, str]]:
    """Run only published Bao VV9 Stage 1 in the pinned TensorFlow child."""
    tf_python = paths["tf_python"]
    checkpoint = paths["bao_vv9"]
    if checkpoint.name != "ND_VV_Simon32_9R.h5":
        raise ValueError("D-055 requires the frozen Bao VV9 checkpoint")
    if not tf_python.is_file() or not os.access(tf_python, os.X_OK):
        raise FileNotFoundError(
            f"Pinned TensorFlow interpreter is unavailable: {tf_python}"
        )

    pairs_payload = [
        [[int(block[0]), int(block[1])] for block in pair] for pair in ciphertext_pairs
    ]
    child_payload = {
        "repo_root": str(_REPO_ROOT),
        "checkpoint": str(checkpoint),
        "invocation_path": str(directory.resolve()),
        "pairs": pairs_payload,
    }
    env = dict(os.environ)
    env.update(_THREAD_ENVIRONMENT)
    repo_src = str(_REPO_ROOT / "src")
    env["PYTHONPATH"] = (
        f"{repo_src}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else repo_src
    )

    invocation_path = str(directory.resolve())
    command = [
        str(tf_python),
        "-m",
        "neural_sat_speck.followups.ranking_depth_ranking",
        "--tf-stage1",
        "--invocation-path",
        invocation_path,
    ]
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=_REPO_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        # Deliberately inherit the timed parent's process group so its existing
        # watchdog can terminate the TensorFlow child together with the cell.
    )
    stdout, stderr = process.communicate(
        input=json.dumps(child_payload, separators=(",", ":"))
    )
    child_wall_seconds = _elapsed(started)
    if process.returncode != 0:
        raise RuntimeError(
            "Pinned Bao VV9 Stage 1 child failed "
            f"(exit {process.returncode}): {stderr[-2000:]}"
        )
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Bao VV9 Stage 1 child returned malformed JSON: {stderr[-1000:]}"
        ) from error
    if not isinstance(result, dict) or result.get("status") != "ok":
        raise RuntimeError(f"Bao VV9 Stage 1 child failed: {result!r}")

    order = _validate_word_order(
        result.get("order"),
        _SIMON_RESCORING_SIZE,
        "Bao VV9 Stage 1 rescored prefix",
    )
    telemetry = result.get("worker_telemetry")
    if (
        not isinstance(telemetry, dict)
        or telemetry.get("is_certified") is not True
        or telemetry.get("pid") != process.pid
        or telemetry.get("parent_pid") != os.getpid()
        or result.get("process_group") != os.getpgrp()
        or telemetry.get("thread_environment", {}).get("TF_NUM_INTRAOP_THREADS") != "8"
        or telemetry.get("thread_environment", {}).get("TF_NUM_INTEROP_THREADS") != "1"
        or result.get("forbidden_import_attempts") != []
        or result.get("loaded_forbidden_modules") != []
        or result.get("invocation_path") != str(directory.resolve())
        or result.get("numeric_environment") != _THREAD_ENVIRONMENT
    ):
        raise RuntimeError(
            "Bao VV9 child runtime/process-group receipt failed validation"
        )

    runtime_path = directory / "tensorflow_stage1_worker.json"
    atomic_json(runtime_path, telemetry)
    detail = {
        "ranker": "production_rank_neural_domain",
        "model": "Bao published VV9",
        "candidate_domain_size": _DOMAIN_SIZE,
        "rescored_order_size": len(order),
        "screen_pairs": 16,
        "full_pairs": len(ciphertext_pairs),
        "rescore_top": _SIMON_RESCORING_SIZE,
        "feature_batch_size": _BATCH_SIZE,
        "inference_batch_size": 16384,
        "tensorflow_intra_op_threads": _TF_INTRA_OP_THREADS,
        "tensorflow_inter_op_threads": _TF_INTER_OP_THREADS,
        "numeric_threads": _NUMERIC_THREADS,
        "child_wall_seconds": child_wall_seconds,
        "child_phase_seconds": result.get("phase_seconds"),
        "checkpoint_receipt": result.get("checkpoint_receipt"),
        "runtime_receipt_path": str(runtime_path),
        "runtime_receipt_sha256": _sha256_file(runtime_path),
        "tensorflow_child_pid": process.pid,
        "tensorflow_child_process_group": result["process_group"],
        "tensorflow_command": command,
        "runtime": result.get("runtime"),
        "external_bao_python_sources": [],
    }
    parent_receipt = _read_optional_json(directory / "python_worker_receipt.json")
    if not isinstance(parent_receipt, dict):
        raise RuntimeError("Pinned Bao ranking requires the parent worker receipt")
    if (
        telemetry.get("git_commit") != parent_receipt.get("git_commit")
        or telemetry.get("git_commit") == "UNKNOWN_COMMIT"
        or parent_receipt.get("pid") != os.getpid()
        or not isinstance(detail["runtime"], dict)
    ):
        raise RuntimeError(
            "Bao VV9 child source/runtime identity differs from its parent"
        )
    runtime_executable = detail["runtime"].get("python_executable")
    if not isinstance(runtime_executable, str) or not Path(runtime_executable).samefile(
        tf_python
    ):
        raise RuntimeError("Bao VV9 child used a different Python executable")
    expected_argv_suffix = ["--invocation-path", str(directory.resolve())]
    command_line = telemetry.get("command_line")
    if (
        not isinstance(command_line, list)
        or command_line[-2:] != expected_argv_suffix
        or result.get("invocation_path") != str(directory.resolve())
    ):
        raise RuntimeError("Bao VV9 child command omitted the absolute cell token")
    package_records = detail["runtime"].get("packages", {})
    expected_packages = {"numpy", "tensorflow", "keras", "h5py"}
    if set(package_records) != expected_packages or any(
        not isinstance(package_records[name], dict)
        or not isinstance(package_records[name].get("version"), str)
        or not isinstance(package_records[name].get("origin"), str)
        for name in expected_packages
    ):
        raise RuntimeError(
            "Bao VV9 child omitted pinned package version/origin metadata"
        )
    imported_package_versions = result.get("runtime_import_smoke")
    if not isinstance(imported_package_versions, dict) or any(
        imported_package_versions.get(name) != package_records[name]["version"]
        for name in expected_packages
    ):
        raise RuntimeError(
            "Bao VV9 imported package versions differ from runtime metadata"
        )
    imports = result.get("code_dependencies")
    if not isinstance(imports, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in imports.items()
    ):
        raise RuntimeError(
            "Bao VV9 child omitted its imported source dependency hashes"
        )
    return order, detail, imports


def _rank_simon_stage2(
    ciphertext_pairs: tuple[tuple[tuple[int, int], tuple[int, int]], ...],
    stage1_order: tuple[int, ...],
    parents: tuple[int, ...],
    paths: Mapping[str, Path],
) -> tuple[tuple[tuple[int, int], ...], dict[str, Any], dict[str, str]]:
    imports_before = set(sys.modules)
    from neural_sat_speck.followups.ranking import compose_simon_ranked_queue
    from neural_sat_speck.simon.classical_ranking import (
        FullPredictedDdt,
        rank_classical_domain,
    )

    table_load_started = time.perf_counter()
    table = FullPredictedDdt(paths["simon_ddt"], 7)
    table_load_seconds = _elapsed(table_load_started)
    orders: dict[int, tuple[int, ...]] = {}
    per_parent: list[dict[str, Any]] = []
    ranker_seconds = 0.0
    for parent_rank, parent in enumerate(parents, start=1):
        rank_started = time.perf_counter()
        ranking = rank_classical_domain(
            ciphertext_pairs,
            table,
            final_key_candidate=parent,
            expected_rounds=7,
            screen_pairs=16,
            full_pairs=64,
            rescore_top=512,
            batch_size=_BATCH_SIZE,
        )
        parent_wall_seconds = _elapsed(rank_started)
        ranker_seconds += parent_wall_seconds
        order = _validate_word_order(
            ranking.order,
            512,
            f"SIMON 7R DDT Stage 2 parent {parent:04x}",
        )
        orders[parent] = order
        per_parent.append(
            {
                "parent_key": parent,
                "parent_rank": parent_rank,
                "candidate_domain_size": _DOMAIN_SIZE,
                "rescored_order_size": len(order),
                "ranker_total_seconds": _finite_seconds(ranking.total_seconds),
                "invocation_wall_seconds": parent_wall_seconds,
                "screen_pairs": ranking.screen_pairs_used,
                "full_pairs": ranking.full_pairs_used,
                "batch_size": _BATCH_SIZE,
                "top_children": list(order[:128]),
            }
        )

    queue = compose_simon_ranked_queue(
        stage1_order,
        orders,
        parent_width=8,
        child_width=128,
    )
    queue = tuple((int(child), int(parent)) for child, parent in queue)
    detail = {
        "ranker": "production_rank_classical_domain",
        "table": "predicted_SIMON_7R_DDT",
        "table_load_seconds": table_load_seconds,
        "stage2_ranker_seconds": ranker_seconds,
        "candidate_batch_size": _BATCH_SIZE,
        "per_parent": per_parent,
        "ordering": "existing_simon_zero_based_sum_rank_then_parent_child",
    }
    return queue, detail, _source_imports(imports_before)


def _rank_only_speck_instance(
    ciphertext_pairs: tuple, AttackInstance: Any, AttackPair: Any
) -> Any:
    """Create the established ranker container with no assigned plaintext/key.

    The neural and classical rankers consume ciphertext fields only.  The
    AttackInstance type requires key-shaped fields for historical API reasons;
    fixed zero sentinels are never derived from the assigned instance and are
    not read by either ranking function.
    """
    pairs = tuple(
        AttackPair(
            plaintext0=(0, 0),
            plaintext1=(0, 0),
            ciphertext0=pair[0],
            ciphertext1=pair[1],
        )
        for pair in ciphertext_pairs
    )
    return AttackInstance(
        instance_id="d055-public-ciphertext-only",
        master_key=(0, 0, 0, 0),
        final_round_subkey=0,
        pairs=pairs,
    )


def _load_speck_dependencies() -> tuple[Any, ...]:
    """Lazy-load fixed production SPECK ranking dependencies per cell."""
    import torch

    from neural_sat_speck.attacks.group_ranking import rank_pair_group_subkeys
    from neural_sat_speck.attacks.multi_round_instance import partial_decrypt_instance
    from neural_sat_speck.data.attack import AttackInstance, AttackPair
    from neural_sat_speck.followups.ranking import _load_pair_group_ranker
    from neural_sat_speck.neural.group_model import PairGroupRanker

    return (
        torch,
        rank_pair_group_subkeys,
        partial_decrypt_instance,
        AttackInstance,
        AttackPair,
        PairGroupRanker,
        _load_pair_group_ranker,
    )


def _load_speck_stage2_dependencies() -> tuple[Any, ...]:
    """Lazy-load the production sparse-R5 second-stage functions per cell."""
    import torch

    from neural_sat_speck.attacks.classical_ranking import (
        ClassicalDistributionTable,
        rank_classical_subkeys,
    )
    from neural_sat_speck.attacks.multi_round_instance import partial_decrypt_instance
    from neural_sat_speck.data.attack import AttackInstance, AttackPair
    from neural_sat_speck.followups.ranking import compose_speck_ranked_queue

    return (
        torch,
        ClassicalDistributionTable,
        rank_classical_subkeys,
        partial_decrypt_instance,
        AttackInstance,
        AttackPair,
        compose_speck_ranked_queue,
    )


def _configure_torch(torch: Any) -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise


def _tf_stage1_main() -> None:
    """Pinned-vv9 child entrypoint; there is deliberately no solver mode."""
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tf-stage1", action="store_true", required=True)
    parser.add_argument("--invocation-path", required=True)
    args = parser.parse_args(sys.argv[1:])
    invocation_path = Path(args.invocation_path)
    if not invocation_path.is_absolute():
        raise ValueError("TensorFlow child invocation path must be absolute")
    payload = json.loads(sys.stdin.read())
    root = Path(payload["repo_root"]).resolve()
    if Path(payload["invocation_path"]).resolve() != invocation_path.resolve():
        raise ValueError(
            "TensorFlow child invocation token differs from input receipt path"
        )
    imports_before = set(sys.modules)
    telemetry = _tf_worker_telemetry(root).certify_worker_or_raise(root)
    blocker = _install_tf_rank_import_boundary(root)
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(_TF_INTRA_OP_THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(_TF_INTER_OP_THREADS)
    import h5py
    import keras
    import numpy as np

    from neural_sat_speck.simon.published_vv import load_published_vv_checkpoint
    from neural_sat_speck.simon.published_vv_ranking import rank_neural_domain

    pairs = tuple(
        ((int(pair[0][0]), int(pair[0][1])), (int(pair[1][0]), int(pair[1][1])))
        for pair in payload["pairs"]
    )
    load_started = time.perf_counter()
    model, checkpoint_receipt = load_published_vv_checkpoint(
        Path(payload["checkpoint"]), 9
    )
    model_load_seconds = _elapsed(load_started)
    rank_started = time.perf_counter()
    ranking = rank_neural_domain(
        pairs,
        model,
        screen_pairs=16,
        full_pairs=64,
        rescore_top=_SIMON_RESCORING_SIZE,
        feature_batch_size=_BATCH_SIZE,
        inference_batch_size=16384,
    )
    rank_wall_seconds = _elapsed(rank_started)
    order = _validate_word_order(
        ranking.order,
        _SIMON_RESCORING_SIZE,
        "Bao VV9 Stage 1 rescored prefix",
    )
    result = {
        "status": "ok",
        "order": list(order),
        "worker_telemetry": telemetry.to_dict(),
        "process_group": os.getpgrp(),
        "runtime": _tf_runtime_identity(),
        "checkpoint_receipt": checkpoint_receipt,
        "phase_seconds": {
            "model_load": model_load_seconds,
            "ranker_total": _finite_seconds(ranking.total_seconds),
            "rank_invocation_wall": rank_wall_seconds,
        },
        "numeric_environment": {
            name: os.environ.get(name) for name in _THREAD_ENVIRONMENT
        },
        "forbidden_import_attempts": list(blocker.attempts),
        "loaded_forbidden_modules": _loaded_forbidden_modules(),
        "invocation_path": str(invocation_path.resolve()),
        "runtime_import_smoke": {
            "numpy": np.__version__,
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "h5py": h5py.__version__,
        },
        "code_dependencies": _source_imports(imports_before),
    }
    if blocker.attempts or result["loaded_forbidden_modules"]:
        raise RuntimeError(
            "TensorFlow rank child crossed the SAT/verification import boundary"
        )
    sys.stdout.write(json.dumps(result, allow_nan=False, separators=(",", ":")))
    sys.stdout.flush()


def _tf_import_smoke_main() -> None:
    """Import-only pinned-runtime probe; no keys, data, models, or ranking calls."""
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tf-import-smoke", action="store_true", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--invocation-path", required=True)
    args = parser.parse_args(sys.argv[1:])
    root = Path(args.repo_root).resolve()
    invocation_path = Path(args.invocation_path)
    if not invocation_path.is_absolute():
        raise ValueError("TensorFlow smoke invocation path must be absolute")
    imports_before = set(sys.modules)
    telemetry = _tf_worker_telemetry(root).certify_worker_or_raise(root)
    blocker = _install_tf_rank_import_boundary(root)
    import h5py
    import keras
    import numpy as np
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(_TF_INTRA_OP_THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(_TF_INTER_OP_THREADS)
    ranker_modules = [
        importlib.import_module(name)
        for name in (
            "neural_sat_speck.simon.cipher",
            "neural_sat_speck.simon.ranking_features",
            "neural_sat_speck.simon.published_vv",
            "neural_sat_speck.simon.published_vv_ranking",
        )
    ]

    result = {
        "status": "ok",
        "worker_telemetry": telemetry.to_dict(),
        "process_group": os.getpgrp(),
        "runtime": _tf_runtime_identity(),
        "numeric_environment": {
            name: os.environ.get(name) for name in _THREAD_ENVIRONMENT
        },
        "forbidden_import_attempts": list(blocker.attempts),
        "loaded_forbidden_modules": _loaded_forbidden_modules(),
        "invocation_path": str(invocation_path.resolve()),
        "code_dependencies": _source_imports(imports_before),
        "ranking_or_model_execution": False,
        "dataset_or_key_access": False,
    }
    if blocker.attempts or result["loaded_forbidden_modules"]:
        raise RuntimeError(
            "TensorFlow import smoke crossed the SAT/verification boundary"
        )
    if any(module is None for module in (np, tf, h5py, keras, *ranker_modules)):
        raise RuntimeError("TensorFlow import smoke missed a required package")
    sys.stdout.write(json.dumps(result, allow_nan=False, separators=(",", ":")))
    sys.stdout.flush()


def _tf_worker_telemetry(root: Path) -> Any:
    telemetry_module_path = root / "src/neural_sat_speck/protocol/worker_telemetry.py"
    telemetry_spec = importlib.util.spec_from_file_location(
        "ranking_depth_tf_worker_telemetry", telemetry_module_path
    )
    if telemetry_spec is None or telemetry_spec.loader is None:
        raise RuntimeError("Could not load the stdlib-only TensorFlow worker telemetry")
    telemetry_module = importlib.util.module_from_spec(telemetry_spec)
    sys.modules[telemetry_spec.name] = telemetry_module
    telemetry_spec.loader.exec_module(telemetry_module)
    return telemetry_module


class _TfRankImportBlocker(importlib.abc.MetaPathFinder):
    """Reject SAT, truth verification and the eager protocol package."""

    def __init__(self) -> None:
        self.attempts: list[str] = []

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        del path, target
        if any(
            fullname == forbidden or fullname.startswith(forbidden + ".")
            for forbidden in _FORBIDDEN_TF_IMPORTS
        ):
            self.attempts.append(fullname)
            raise ImportError(f"TensorFlow rank import boundary blocked {fullname}")
        return None


def _install_tf_rank_import_boundary(root: Path) -> _TfRankImportBlocker:
    """Install a SIMON namespace package before imports would run its SAT initializer."""
    simon_path = root / "src/neural_sat_speck/simon"
    existing = sys.modules.get("neural_sat_speck.simon")
    if existing is not None and getattr(existing, "__file__", None) is not None:
        raise RuntimeError("Eager neural_sat_speck.simon initializer already ran")
    if existing is None:
        simon = types.ModuleType("neural_sat_speck.simon")
        simon.__package__ = "neural_sat_speck.simon"
        simon.__path__ = [str(simon_path)]  # type: ignore[attr-defined]
        simon.__spec__ = importlib.util.spec_from_loader(
            "neural_sat_speck.simon", loader=None, is_package=True
        )
        sys.modules["neural_sat_speck.simon"] = simon
    blocker = _TfRankImportBlocker()
    sys.meta_path.insert(0, blocker)
    return blocker


def _loaded_forbidden_modules() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if any(
            name == forbidden or name.startswith(forbidden + ".")
            for forbidden in _FORBIDDEN_TF_IMPORTS
        )
    )


def _check_paths(
    paths: Mapping[str, Path | str], cipher: str, trailing_keys: int
) -> dict[str, Path]:
    if not isinstance(paths, Mapping):
        raise TypeError("paths must be a path mapping")
    required = {"speck_stage1"} if cipher == "speck32_64" else {"bao_vv9", "tf_python"}
    if trailing_keys == 2:
        required.add("speck_classical" if cipher == "speck32_64" else "simon_ddt")
    checked: dict[str, Path] = {}
    for name in required:
        raw = paths.get(name)
        if not isinstance(raw, (str, Path)) or not str(raw):
            raise ValueError(f"Missing required ranking asset path: {name}")
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(f"Required ranking asset is unavailable: {path}")
        if name == "tf_python" and not os.access(path, os.X_OK):
            raise PermissionError(f"Pinned TensorFlow Python is not executable: {path}")
        checked[name] = path
    return checked


def _validate_ciphertext_pairs(
    value: object,
) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 64
    ):
        raise ValueError("ranking requires exactly 64 public ciphertext pairs")
    result = []
    for pair_index, pair in enumerate(value):
        if (
            not isinstance(pair, Sequence)
            or isinstance(pair, (str, bytes))
            or len(pair) != 2
        ):
            raise ValueError(f"ciphertext pair {pair_index} must contain two blocks")
        blocks = []
        for block_index, block in enumerate(pair):
            if (
                not isinstance(block, Sequence)
                or isinstance(block, (str, bytes))
                or len(block) != 2
            ):
                raise ValueError(
                    f"ciphertext pair {pair_index} block {block_index} must be two words"
                )
            words = []
            for word in block:
                if (
                    isinstance(word, bool)
                    or not isinstance(word, Integral)
                    or not 0 <= int(word) <= 0xFFFF
                ):
                    raise ValueError(
                        "ciphertext words must be unsigned 16-bit integers"
                    )
                words.append(int(word))
            blocks.append((words[0], words[1]))
        result.append((blocks[0], blocks[1]))
    return tuple(result)


def _validate_word_order(
    value: object, expected_size: int, label: str
) -> tuple[int, ...]:
    if not isinstance(value, Sequence) and not hasattr(value, "__iter__"):
        raise ValueError(f"{label} is not an integer word order")
    raw_order = tuple(value)  # type: ignore[arg-type]
    if any(
        isinstance(word, bool)
        or not isinstance(word, Integral)
        or not 0 <= int(word) <= 0xFFFF
        for word in raw_order
    ):
        raise ValueError(f"{label} contains a non-16-bit integer word")
    order = tuple(int(word) for word in raw_order)
    if len(order) != expected_size:
        raise ValueError(f"{label} has {len(order)} words, expected {expected_size}")
    if len(set(order)) != len(order):
        raise ValueError(f"{label} contains duplicate candidate words")
    return order


def _validate_word_prefix(
    value: tuple[int, ...], expected_size: int, label: str
) -> None:
    if len(value) != expected_size or len(set(value)) != expected_size:
        raise ValueError(f"{label} has invalid width or duplicates")
    if any(type(word) is not int or not 0 <= word <= 0xFFFF for word in value):
        raise ValueError(f"{label} contains an invalid 16-bit word")


def _validate_queue(
    queue: tuple[tuple[int, ...], ...], trailing_keys: int, profile: Mapping[str, int]
) -> None:
    expected_length = (
        profile["parents"]
        if trailing_keys == 1
        else profile["parents"] * profile["children"]
    )
    if len(queue) != expected_length or len(set(queue)) != len(queue):
        raise ValueError("Ranker output has the wrong queue size or duplicate tuples")
    if any(len(candidate) != trailing_keys for candidate in queue):
        raise ValueError("Ranker output tuple width differs from trailing_keys")
    # In chronological tuples, the final word is always the Stage 1 parent.
    if any(
        type(candidate[-1]) is not int or not 0 <= candidate[-1] <= 0xFFFF
        for candidate in queue
    ):
        raise ValueError("Ranker output contains an invalid parent word")
    if trailing_keys == 1 and any(type(candidate[0]) is not int for candidate in queue):
        raise ValueError("One-key queue contains a non-integer word")
    if trailing_keys == 2 and any(
        type(word) is not int or not 0 <= word <= 0xFFFF
        for candidate in queue
        for word in candidate
    ):
        raise ValueError("Two-key queue contains an invalid word")


def _validate_ranked_queue_parents(
    queue: tuple[tuple[int, ...], ...], parents: tuple[int, ...]
) -> None:
    allowed = set(parents)
    if any(candidate[-1] not in allowed for candidate in queue):
        raise ValueError("Ranker output contains a parent outside the Stage 1 prefix")


def _assets_for_stage(
    paths: Mapping[str, Path], stage: str
) -> dict[str, dict[str, Any]]:
    names = (
        ("speck_stage1", "bao_vv9", "tf_python")
        if stage == "stage1"
        else ("speck_classical", "simon_ddt")
    )
    result = {}
    for name in names:
        path = paths.get(name)
        if path is not None:
            stat = path.stat()
            result[name] = {
                "path": str(path.absolute()),
                # Large DDT files are hashed by the out-of-clock manifest
                # preflight.  The ranking call records identity/path metadata
                # without adding a second full-file hash to its online cost.
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
    return result


def _source_imports(before: set[str]) -> dict[str, str]:
    """Hash repository source files actually imported during a rank phase."""
    dependencies: dict[str, str] = {}
    package_root = (_REPO_ROOT / "src/neural_sat_speck").resolve()
    for name in sorted(set(sys.modules) - before):
        module = sys.modules.get(name)
        source = getattr(module, "__file__", None)
        if not isinstance(source, str):
            continue
        path = Path(source).resolve()
        try:
            relative = path.relative_to(package_root)
        except ValueError:
            continue
        if path.is_file() and path.suffix == ".py":
            dependencies[str(Path("src/neural_sat_speck") / relative)] = _sha256_file(
                path
            )
    return dependencies


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return read_checked_json(path)


def _tf_runtime_identity() -> dict[str, Any]:
    packages: dict[str, dict[str, str | None]] = {}
    for distribution, module_name in (
        ("numpy", "numpy"),
        ("tensorflow", "tensorflow"),
        ("keras", "keras"),
        ("h5py", "h5py"),
    ):
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = None
        spec = importlib.util.find_spec(module_name)
        packages[distribution] = {
            "version": version,
            "origin": None if spec is None else spec.origin,
        }
    installed = {
        str(distribution.metadata["Name"]): str(distribution.version)
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    return {
        "python_executable": sys.executable,
        "packages": packages,
        "installed_packages": dict(sorted(installed.items())),
    }


def _notify(
    callback: PhaseCallback | None, name: str, detail: Mapping[str, Any]
) -> None:
    if callback is not None:
        callback(name, detail)


def _finite_seconds(value: object) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Ranking timer must be finite and non-negative")
    return seconds


def _elapsed(started: float) -> float:
    return _finite_seconds(time.perf_counter() - started)


if __name__ == "__main__":
    if "--tf-stage1" in sys.argv[1:]:
        _tf_stage1_main()
    elif "--tf-import-smoke" in sys.argv[1:]:
        _tf_import_smoke_main()
