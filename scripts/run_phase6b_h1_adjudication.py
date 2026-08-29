"""Run the prospective validation-only B0/B1/H1 adjudication.

The protocol is frozen in ``configs/experiments/phase6b_h1_adjudication.json``.
This runner creates only a validation shard, verifies it with the existing
attack checks, ranks the same 64 instances with B0, B1, and H1, and writes
resumable per-instance records plus a summary.  It never reads or creates a
test split and never invokes CryptoMiniSat.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import resource
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from neural_sat_speck.attacks.group_ranking import (
    PairGroupSubkeyRanking,
    evaluate_pair_group_subkey_ranking,
    rank_pair_group_subkeys,
)
from neural_sat_speck.attacks.hierarchical_ranking import (
    HIERARCHICAL_METHOD_ID,
    HIERARCHICAL_WINDOW_SIZE,
    HierarchicalSubkeyRanking,
    rank_hierarchical_subkeys,
)
from neural_sat_speck.attacks.ranking import (
    ALL_FINAL_ROUND_SUBKEYS,
    SubkeyRanking,
    evaluate_subkey_ranking,
    rank_final_round_subkeys,
)
from neural_sat_speck.data.attack import (
    ATTACK_KEY_GENERATION_POLICY,
    ATTACK_TASK,
    AttackInstance,
    AttackInstanceConfig,
    generate_attack_instances,
    load_attack_shard,
    verify_attack_shard,
    write_attack_shard,
)
from neural_sat_speck.data.format import (
    CIPHER_NAME,
    DATASET_FORMAT_VERSION,
    DATASET_LICENSE,
    PLAINTEXT_DIFFERENCE_POLICY,
    canonical_json,
)
from neural_sat_speck.data.generator import DEFAULT_INPUT_DIFFERENCE
from neural_sat_speck.data.metadata import BIT_ORDER, GENERATOR_VERSION, WORD_ORDER
from neural_sat_speck.neural.group_train import load_pair_group_checkpoint
from neural_sat_speck.neural.numpy_inference import (
    NumpyInferenceModule,
    NumpyRankerInference,
)
from neural_sat_speck.neural.train import load_checkpoint

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT / "configs" / "experiments" / "phase6b_h1_adjudication.json"
)
MANIFEST_SCHEMA = "neural_sat_speck.phase6b_h1_adjudication_manifest.v1"
RESULT_SCHEMA = "neural_sat_speck.phase6b_h1_adjudication_result.v1"
SUMMARY_SCHEMA = "neural_sat_speck.phase6b_h1_adjudication_summary.v1"
METHODS = ("b0", "b1", "h1")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"could not read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"JSON document {path} must contain an object")
    return value


def _repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _peak_rss_bytes() -> int:
    """Return this process's peak resident set size in bytes."""

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux and most other Unix systems report KiB.
    return value if sys.platform == "darwin" else value * 1024


def _repository_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SystemExit(f"configuration field {name!r} must be an object")
    return value


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f"configuration field {name!r} must be an integer")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SystemExit(f"configuration field {name!r} must be non-empty text")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    """Reject protocol drift before any adjudication result is read."""

    if config.get("schema") != "neural_sat_speck.phase6b_h1_adjudication_config.v1":
        raise SystemExit("Phase 6B H1 configuration schema is not recognised")
    if _require_int(config.get("phase"), "phase") != 6:
        raise SystemExit("Phase 6B H1 configuration must have phase=6")
    if config.get("validation_only") is not True:
        raise SystemExit("Phase 6B H1 configuration must be validation-only")
    if config.get("held_out_test_generated") is not False:
        raise SystemExit("Phase 6B H1 configuration must forbid test generation")

    dataset = _require_mapping(config.get("dataset"), "dataset")
    if _require_int(dataset.get("rounds"), "dataset.rounds") != 6:
        raise SystemExit("Phase 6B H1 requires exactly six cipher rounds")
    if _require_int(dataset.get("pair_count"), "dataset.pair_count") != 64:
        raise SystemExit("Phase 6B H1 requires exactly 64 related pairs")
    if _require_int(dataset.get("instance_count"), "dataset.instance_count") != 64:
        raise SystemExit("Phase 6B H1 requires exactly 64 validation instances")
    if _require_int(dataset.get("seed"), "dataset.seed") <= 0:
        raise SystemExit("dataset.seed must be positive")
    if dataset.get("split_role") != "validation":
        raise SystemExit("Phase 6B H1 material must have validation role")
    if dataset.get("input_difference") != list(DEFAULT_INPUT_DIFFERENCE):
        raise SystemExit("Phase 6B H1 input difference is not the frozen value")
    if dataset.get("pair_construction") != PLAINTEXT_DIFFERENCE_POLICY:
        raise SystemExit("Phase 6B H1 pair construction is not frozen")
    if dataset.get("key_generation_policy") != ATTACK_KEY_GENERATION_POLICY:
        raise SystemExit("Phase 6B H1 key-generation policy is not frozen")
    dataset_id = _require_string(dataset.get("dataset_id"), "dataset.dataset_id")
    if not dataset_id.startswith("speck32_64_attack_phase6b_h1_adjudication"):
        raise SystemExit("Phase 6B H1 dataset_id is not a new adjudication identifier")

    domain = _require_mapping(config.get("candidate_domain"), "candidate_domain")
    if (
        _require_int(domain.get("first"), "candidate_domain.first") != 0
        or _require_int(domain.get("last"), "candidate_domain.last") != 0xFFFF
        or _require_int(domain.get("count"), "candidate_domain.count") != 1 << 16
        or domain.get("complete_16_bit_domain") is not True
    ):
        raise SystemExit("Phase 6B H1 candidate domain must be complete 16-bit")

    methods = _require_mapping(config.get("methods"), "methods")
    b0 = _require_mapping(methods.get("b0"), "methods.b0")
    b1 = _require_mapping(methods.get("b1"), "methods.b1")
    h1 = _require_mapping(methods.get("h1"), "methods.h1")
    if b0.get("method_id") != "b0_sum_logits" or b0.get("score_aggregation") != "sum_logits":
        raise SystemExit("B0 method identity or score convention is not frozen")
    if b1.get("method_id") != "b1_group_logit" or b1.get("score_aggregation") != "group_logit":
        raise SystemExit("B1 method identity or score convention is not frozen")
    if h1.get("method_id") != HIERARCHICAL_METHOD_ID:
        raise SystemExit("H1 method identity is not frozen")
    if _require_int(h1.get("window_size"), "methods.h1.window_size") != HIERARCHICAL_WINDOW_SIZE:
        raise SystemExit("H1 window size must be exactly 16")
    if _require_int(h1.get("candidate_budget"), "methods.h1.candidate_budget") != HIERARCHICAL_WINDOW_SIZE:
        raise SystemExit("H1 candidate budget must be exactly 16")

    backend_policy = _require_mapping(config.get("backend_policy"), "backend_policy")
    if backend_policy.get("reference") != "pytorch":
        raise SystemExit("PyTorch must remain the reference backend")
    if backend_policy.get("accelerated_candidate") != "numpy":
        raise SystemExit("NumPy must remain the only accelerated candidate")
    tolerance = backend_policy.get("score_abs_tolerance")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or float(tolerance) <= 0.0
    ):
        raise SystemExit("backend score_abs_tolerance must be positive and finite")

    runtime = _require_mapping(config.get("runtime"), "runtime")
    if runtime.get("device") != "cpu":
        raise SystemExit("Phase 6B H1 adjudication is CPU-only")
    if _require_int(runtime.get("torch_num_threads"), "runtime.torch_num_threads") <= 0:
        raise SystemExit("runtime.torch_num_threads must be positive")
    if _require_int(
        runtime.get("torch_num_interop_threads"), "runtime.torch_num_interop_threads"
    ) <= 0:
        raise SystemExit("runtime.torch_num_interop_threads must be positive")

    ranking = _require_mapping(config.get("ranking"), "ranking")
    if _require_int(ranking.get("pair_count"), "ranking.pair_count") != 64:
        raise SystemExit("ranking.pair_count must be exactly 64")
    for name in ("b0_candidate_batch_size", "b1_candidate_batch_size", "h1_b0_window_batch_size"):
        if _require_int(ranking.get(name), f"ranking.{name}") <= 0:
            raise SystemExit(f"ranking.{name} must be positive")
    if ranking.get("candidate_domain") != "complete_16_bit_domain":
        raise SystemExit("ranking candidate domain is not frozen")
    if ranking.get("b1_order_is_input") is not True:
        raise SystemExit("H1 must consume the complete B1 order as input")

    metrics = _require_mapping(config.get("metrics"), "metrics")
    if metrics.get("primary") != "capped_attempts":
        raise SystemExit("primary metric must be capped_attempts")
    if metrics.get("capped_attempts_definition") != "rank if rank <= 16 else 17":
        raise SystemExit("capped-attempt metric is not frozen")
    if _require_int(metrics.get("miss_value"), "metrics.miss_value") != 17:
        raise SystemExit("capped-attempt miss value must be 17")
    top_k_values = metrics.get("top_k_values")
    if top_k_values != [1, 2, 4, 8, 12, 16]:
        raise SystemExit("top-k reporting values are not frozen")
    bootstrap_seed = _require_int(metrics.get("bootstrap_seed"), "metrics.bootstrap_seed")
    if bootstrap_seed <= 0:
        raise SystemExit("metrics.bootstrap_seed must be positive")
    if _require_int(metrics.get("bootstrap_replicates"), "metrics.bootstrap_replicates") <= 0:
        raise SystemExit("metrics.bootstrap_replicates must be positive")

    gate = _require_mapping(config.get("acceptance_gate"), "acceptance_gate")
    if gate.get("h1_top16_set_equals_b1_on_every_instance") is not True:
        raise SystemExit("H1 top-16 set identity is not frozen")
    if gate.get("h1_top16_recall_equals_b1_by_construction") is not True:
        raise SystemExit("H1 top-16 recall condition is not frozen")
    overhead_limit = gate.get("h1_b0_window_overhead_max_fraction_of_b1")
    if isinstance(overhead_limit, bool) or not isinstance(overhead_limit, (int, float)) or overhead_limit != 0.05:
        raise SystemExit("H1 overhead limit must be exactly 0.05")

    timing = _require_mapping(config.get("timing_policy"), "timing_policy")
    if timing.get("execution_order") != (
        "b0_then_b1_for_even_validation_index_b1_then_b0_for_odd"
    ):
        raise SystemExit("counterbalanced execution order is not frozen")
    if timing.get("single_process_sequential") is not True:
        raise SystemExit("adjudication must run in one sequential process")
    if timing.get("concurrent_ranking_jobs") is not False:
        raise SystemExit("concurrent ranking jobs are forbidden")
    if timing.get("peak_rss_bytes_recorded") is not True:
        raise SystemExit("peak RSS recording is required")

    exclusions = config.get("seed_exclusions")
    if not isinstance(exclusions, Sequence) or isinstance(exclusions, (str, bytes)):
        raise SystemExit("seed_exclusions must be a sequence")
    exclusion_values = tuple(_require_int(value, "seed_exclusions item") for value in exclusions)
    if len(set(exclusion_values)) != len(exclusion_values):
        raise SystemExit("seed_exclusions must not contain duplicates")
    if _require_int(dataset.get("seed"), "dataset.seed") in set(exclusion_values):
        raise SystemExit("new adjudication seed appears in the exclusion list")
    if bootstrap_seed in set(exclusion_values):
        raise SystemExit("bootstrap seed appears in the key-seed exclusion list")


def _manifest_document(
    instance_set: Any,
    *,
    dataset_id: str,
    repository_commit: str | None,
    generated_at: str,
) -> dict[str, Any]:
    """Build a validation-only manifest without a train or test section."""

    shard_payload = write_attack_shard  # keep the schema construction explicit below
    del shard_payload
    return {
        "schema": MANIFEST_SCHEMA,
        "phase": 6,
        "study": "phase6b_h1_b0_b1_adjudication",
        "validation_only": True,
        "held_out_test_generated": False,
        "bit_order": BIT_ORDER,
        "cipher": CIPHER_NAME,
        "dataset_format_version": DATASET_FORMAT_VERSION,
        "dataset_id": dataset_id,
        "generated_at": generated_at,
        "generator_version": GENERATOR_VERSION,
        "input_difference": list(DEFAULT_INPUT_DIFFERENCE),
        "key_generation_policy": ATTACK_KEY_GENERATION_POLICY,
        "license": DATASET_LICENSE,
        "pair_construction": PLAINTEXT_DIFFERENCE_POLICY,
        "repository_commit": repository_commit,
        "rounds": 6,
        "splits": {
            "validation": {
                "data_checksum": instance_set.metadata["data_checksum"],
                "file_sha256": None,
                "instance_count": instance_set.metadata["instance_count"],
                "pair_count": instance_set.metadata["pair_count"],
                "path": "validation.json",
                "seed": instance_set.metadata["seed"],
                "split_role": "validation",
            }
        },
        "task": ATTACK_TASK,
        "word_order": WORD_ORDER,
    }


def _write_or_verify_dataset(config: Mapping[str, Any], *, generate: bool) -> Path:
    """Create or verify exactly one validation shard and no test material."""

    dataset = _require_mapping(config["dataset"], "dataset")
    dataset_id = str(dataset["dataset_id"])
    output_dir = _repository_path(str(dataset["output_dir"])).resolve()
    manifest_path = _repository_path(str(dataset["manifest_path"])).resolve()
    if manifest_path.parent != output_dir:
        raise SystemExit("dataset manifest_path must be inside dataset.output_dir")
    shard_path = output_dir / "validation.json"
    test_path = output_dir / "test.json"
    if test_path.exists():
        raise SystemExit("refusing to operate: adjudication output contains test.json")

    if generate:
        if manifest_path.exists() or shard_path.exists():
            raise SystemExit(
                "refusing to overwrite an existing adjudication dataset; "
                "verify it without --generate or move it aside explicitly"
            )
        instance_set = generate_attack_instances(
            AttackInstanceConfig(
                instance_count=int(dataset["instance_count"]),
                pair_count=int(dataset["pair_count"]),
                rounds=int(dataset["rounds"]),
                seed=int(dataset["seed"]),
                split_role=str(dataset["split_role"]),
            ),
            generated_at=str(dataset["generated_at"]),
            repository_commit=_repository_commit(),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        record = write_attack_shard(instance_set, shard_path, dataset_id=dataset_id)
        manifest = _manifest_document(
            instance_set,
            dataset_id=dataset_id,
            repository_commit=instance_set.metadata["repository_commit"],
            generated_at=str(dataset["generated_at"]),
        )
        manifest["splits"]["validation"]["file_sha256"] = record["file_sha256"]
        manifest_path.write_bytes(canonical_json(manifest).encode("utf-8"))

    if not manifest_path.is_file() or not shard_path.is_file():
        raise SystemExit(
            "adjudication dataset is missing; run this command once with --generate"
        )
    manifest_payload = manifest_path.read_bytes()
    manifest = _load_json(manifest_path)
    if manifest_payload != canonical_json(manifest).encode("utf-8"):
        raise SystemExit("adjudication manifest is not canonical JSON")
    _verify_manifest(manifest, config, manifest_path)
    verify_attack_shard(
        shard_path,
        expected_dataset_id=dataset_id,
        expected_manifest_record=manifest["splits"]["validation"],
    )
    return manifest_path


def _verify_manifest(
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    manifest_path: Path,
) -> None:
    """Verify the new validation-only manifest and its frozen contract."""

    dataset = _require_mapping(config["dataset"], "dataset")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise SystemExit("adjudication manifest schema is not recognised")
    if manifest.get("dataset_id") != dataset.get("dataset_id"):
        raise SystemExit("adjudication manifest dataset_id does not match config")
    for field, expected in (
        ("phase", 6),
        ("study", "phase6b_h1_b0_b1_adjudication"),
        ("validation_only", True),
        ("held_out_test_generated", False),
        ("cipher", CIPHER_NAME),
        ("dataset_format_version", DATASET_FORMAT_VERSION),
        ("generator_version", GENERATOR_VERSION),
        ("input_difference", list(DEFAULT_INPUT_DIFFERENCE)),
        ("key_generation_policy", ATTACK_KEY_GENERATION_POLICY),
        ("license", DATASET_LICENSE),
        ("pair_construction", PLAINTEXT_DIFFERENCE_POLICY),
        ("rounds", 6),
        ("task", ATTACK_TASK),
        ("word_order", WORD_ORDER),
    ):
        if manifest.get(field) != expected:
            raise SystemExit(f"adjudication manifest field {field!r} is not frozen")
    if manifest.get("bit_order") != BIT_ORDER:
        raise SystemExit("adjudication manifest bit order is not frozen")
    if manifest.get("generated_at") != dataset.get("generated_at"):
        raise SystemExit("adjudication manifest timestamp does not match config")
    if manifest.get("repository_commit") is not None and not isinstance(
        manifest.get("repository_commit"), str
    ):
        raise SystemExit("adjudication manifest repository_commit is invalid")
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {"validation"}:
        raise SystemExit("adjudication manifest must contain validation only")
    record = splits["validation"]
    if not isinstance(record, Mapping):
        raise SystemExit("adjudication validation record is invalid")
    if record.get("instance_count") != dataset.get("instance_count"):
        raise SystemExit("adjudication instance count does not match config")
    if record.get("pair_count") != dataset.get("pair_count"):
        raise SystemExit("adjudication pair count does not match config")
    if record.get("seed") != dataset.get("seed"):
        raise SystemExit("adjudication seed does not match config")
    if record.get("split_role") != "validation" or record.get("path") != "validation.json":
        raise SystemExit("adjudication validation record has the wrong role/path")
    if not manifest_path.is_file():
        raise SystemExit("adjudication manifest disappeared during verification")


def _load_models(
    config: Mapping[str, Any],
) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    """Load and hash the frozen B0 and B1 checkpoints exactly once."""

    methods = _require_mapping(config["methods"], "methods")
    b0_info = _require_mapping(methods["b0"], "methods.b0")
    b1_info = _require_mapping(methods["b1"], "methods.b1")
    b0_path = _repository_path(str(b0_info["checkpoint_path"])).resolve()
    b1_path = _repository_path(str(b1_info["checkpoint_path"])).resolve()
    if not b0_path.is_file() or not b1_path.is_file():
        raise SystemExit("one or more frozen checkpoints are missing")
    b0_hash = _sha256_file(b0_path)
    b1_hash = _sha256_file(b1_path)
    if b0_hash != b0_info.get("checkpoint_sha256"):
        raise SystemExit("B0 checkpoint SHA-256 does not match the frozen config")
    if b1_hash != b1_info.get("checkpoint_sha256"):
        raise SystemExit("B1 checkpoint SHA-256 does not match the frozen config")
    b0_model, b0_metadata = load_checkpoint(b0_path, device="cpu")
    b1_model, b1_metadata = load_pair_group_checkpoint(b1_path, device="cpu")
    if b0_metadata.get("phase") != 3:
        raise SystemExit("B0 checkpoint is not the frozen Phase 3 model")
    if b1_metadata.get("model_id") != b1_info.get("model_id"):
        raise SystemExit("B1 checkpoint model identity does not match config")
    if getattr(b1_model, "model_id", None) != b1_info.get("model_id"):
        raise SystemExit("B1 model identity does not match config")
    if getattr(b1_model, "group_size", None) != 64:
        raise SystemExit("B1 checkpoint group size is not 64")
    b0_model.eval()
    b1_model.eval()
    metadata = {
        "b0": {
            "checkpoint_path": _relative_or_absolute(b0_path),
            "checkpoint_sha256": b0_hash,
            "model_id": b0_info["model_id"],
            "metadata": b0_metadata,
        },
        "b1": {
            "checkpoint_path": _relative_or_absolute(b1_path),
            "checkpoint_sha256": b1_hash,
            "model_id": b1_info["model_id"],
            "metadata": b1_metadata,
        },
    }
    return b0_model, b1_model, metadata


def _execution_models(
    b0_model: torch.nn.Module,
    b1_model: torch.nn.Module,
    backend: str,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    if backend == "eager":
        return b0_model, b1_model
    if backend == "numpy":
        return (
            NumpyInferenceModule(NumpyRankerInference(b0_model)),
            NumpyInferenceModule(NumpyRankerInference(b1_model)),
        )
    raise SystemExit(f"unsupported inference backend {backend!r}")


def _evaluate_order(
    candidate_order: Sequence[int],
    true_subkey: int,
    top_k_values: Sequence[int],
) -> dict[str, Any]:
    """Evaluate an order after construction without feeding the true key back."""

    order = tuple(candidate_order)
    rank = order.index(true_subkey) + 1
    return {
        "true_final_round_subkey": true_subkey,
        "true_rank": rank,
        "true_rank_interval": None,
        "true_subkey_in_candidate_set": True,
        "top_k_membership": {
            str(k): rank <= k for k in top_k_values
        },
        "capped_attempts": min(rank, HIERARCHICAL_WINDOW_SIZE + 1),
    }


def _counterbalanced_execution_order(index: int) -> tuple[str, str]:
    """Return the frozen sequential B0/B1 order for one validation index."""

    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("validation index must be a non-negative integer")
    return ("b0", "b1") if index % 2 == 0 else ("b1", "b0")


def _rank_complete_methods(
    instance: AttackInstance,
    b0_model: torch.nn.Module,
    b1_model: torch.nn.Module,
    *,
    config: Mapping[str, Any],
    index: int,
) -> tuple[SubkeyRanking, PairGroupSubkeyRanking, tuple[str, str]]:
    """Run B0 and authoritative B1 sequentially in a counterbalanced order."""

    ranking_config = _require_mapping(config["ranking"], "ranking")
    execution_order = _counterbalanced_execution_order(index)
    rankings: dict[str, SubkeyRanking | PairGroupSubkeyRanking] = {}
    for method in execution_order:
        if method == "b0":
            rankings[method] = rank_final_round_subkeys(
                instance,
                b0_model,
                candidate_values=ALL_FINAL_ROUND_SUBKEYS,
                candidate_batch_size=int(ranking_config["b0_candidate_batch_size"]),
                device="cpu",
            )
        else:
            rankings[method] = rank_pair_group_subkeys(
                instance,
                b1_model,  # type: ignore[arg-type]
                candidate_values=ALL_FINAL_ROUND_SUBKEYS,
                candidate_batch_size=int(ranking_config["b1_candidate_batch_size"]),
                device="cpu",
            )
    b0_ranking = rankings["b0"]
    b1_ranking = rankings["b1"]
    if not isinstance(b0_ranking, SubkeyRanking) or not isinstance(
        b1_ranking, PairGroupSubkeyRanking
    ):
        raise RuntimeError("complete ranking methods returned unexpected types")
    return b0_ranking, b1_ranking, execution_order


def _ranking_record(
    instance: AttackInstance,
    b0_ranking: SubkeyRanking,
    b1_ranking: PairGroupSubkeyRanking,
    h1_ranking: Any,
    *,
    config: Mapping[str, Any],
    manifest_path: Path,
    config_path: Path,
    model_metadata: Mapping[str, Any],
    backend: str,
    index: int,
    execution_order: Sequence[str],
) -> dict[str, Any]:
    metrics = _require_mapping(config["metrics"], "metrics")
    top_k_values = tuple(int(value) for value in metrics["top_k_values"])
    b0_eval = evaluate_subkey_ranking(
        b0_ranking,
        instance.final_round_subkey,
        top_k_values=top_k_values,
    )
    b1_eval = evaluate_pair_group_subkey_ranking(
        b1_ranking,
        instance.final_round_subkey,
        top_k_values=top_k_values,
    )
    b0_eval["capped_attempts"] = min(
        int(b0_eval["true_rank"]), HIERARCHICAL_WINDOW_SIZE + 1
    )
    b1_eval["capped_attempts"] = min(
        int(b1_eval["true_rank"]), HIERARCHICAL_WINDOW_SIZE + 1
    )
    h1_eval = _evaluate_order(
        h1_ranking.candidate_order,
        instance.final_round_subkey,
        top_k_values,
    )
    h1_payload = h1_ranking.to_dict()
    if HierarchicalSubkeyRanking.from_dict(h1_payload) != h1_ranking:
        raise RuntimeError("H1 serialization round-trip changed the ranking")
    h1_top16_set_equals_b1 = set(h1_ranking.top_k(16)) == set(b1_ranking.top_k(16))
    h1_tail_equals_b1 = (
        h1_ranking.candidate_order[16:] == b1_ranking.candidate_order[16:]
    )
    if not h1_top16_set_equals_b1 or not h1_tail_equals_b1:
        raise RuntimeError("H1 composition invariant failed during adjudication")
    b1_total = float(b1_ranking.total_seconds)
    overhead_fraction = (
        h1_ranking.b0_window_seconds / b1_total if b1_total > 0.0 else math.inf
    )
    methods = _require_mapping(config["methods"], "methods")
    return {
        "schema": RESULT_SCHEMA,
        "phase": 6,
        "study": "phase6b_h1_b0_b1_adjudication",
        "method_order": ["b0", "b1", "h1"],
        "dataset": {
            "dataset_id": _require_mapping(config["dataset"], "dataset")["dataset_id"],
            "manifest_path": _relative_or_absolute(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "validation_only": True,
            "held_out_test_generated": False,
        },
        "instance": {
            "split_role": "validation",
            "index": index,
            "instance_id": instance.instance_id,
            "pair_count": len(instance.pairs),
            "execution_order": list(execution_order),
        },
        "methods": {
            "b0": {
                "method_id": methods["b0"]["method_id"],
                "ranking": b0_ranking.to_dict(),
                "evaluation": b0_eval,
            },
            "b1": {
                "method_id": methods["b1"]["method_id"],
                "ranking": b1_ranking.to_dict(),
                "evaluation": b1_eval,
            },
            "h1": {
                "method_id": methods["h1"]["method_id"],
                "ranking": h1_payload,
                "evaluation": h1_eval,
                "invariants": {
                    "top16_set_equals_b1": h1_top16_set_equals_b1,
                    "tail_equals_b1": h1_tail_equals_b1,
                    "b0_window_candidate_count": len(
                        h1_ranking.b0_window_candidate_values
                    ),
                    "b0_window_overhead_fraction_of_b1": overhead_fraction,
                },
            },
        },
        "provenance": {
            "config_path": _relative_or_absolute(config_path),
            "config_sha256": _sha256_file(config_path),
            "repository_commit": _repository_commit(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch_version": str(torch.__version__),
            "device": "cpu",
            "inference_backend": backend,
            "b0_inference_backend": backend,
            "b1_inference_backend": "pytorch",
            "checkpoints": model_metadata,
        },
    }


def _write_record(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_result(path: Path) -> dict[str, Any]:
    record = _load_json(path)
    if record.get("schema") != RESULT_SCHEMA:
        raise SystemExit(f"result {path} has an unrecognised schema")
    return record


def _validate_saved_result(
    record: Mapping[str, Any],
    *,
    index: int,
    backend: str,
    config_path: Path,
    manifest_path: Path,
) -> None:
    """Reject stale or pre-counterbalance records during resume."""

    instance = _require_mapping(record.get("instance"), "result.instance")
    if instance.get("index") != index or instance.get("split_role") != "validation":
        raise SystemExit("saved result has the wrong validation instance")
    expected_order = ["b0", "b1"] if index % 2 == 0 else ["b1", "b0"]
    if instance.get("execution_order") != expected_order:
        raise SystemExit("saved result does not use the frozen execution order")
    dataset = _require_mapping(record.get("dataset"), "result.dataset")
    if dataset.get("validation_only") is not True or dataset.get(
        "held_out_test_generated"
    ) is not False:
        raise SystemExit("saved result is not validation-only")
    if dataset.get("manifest_sha256") != _sha256_file(manifest_path):
        raise SystemExit("saved result uses another validation manifest")
    provenance = _require_mapping(record.get("provenance"), "result.provenance")
    if provenance.get("config_sha256") != _sha256_file(config_path):
        raise SystemExit("saved result uses another adjudication config")
    if provenance.get("inference_backend") != backend:
        raise SystemExit("saved result uses another inference backend")
    rss = provenance.get("peak_rss_bytes")
    if isinstance(rss, bool) or not isinstance(rss, int) or rss <= 0:
        raise SystemExit("saved result does not contain peak RSS")
    methods = _require_mapping(record.get("methods"), "result.methods")
    for method in METHODS:
        method_record = _require_mapping(methods.get(method), f"result.methods.{method}")
        evaluation = _require_mapping(method_record.get("evaluation"), "evaluation")
        if not isinstance(evaluation.get("capped_attempts"), int):
            raise SystemExit("saved result predates capped-attempt recording")
    HierarchicalSubkeyRanking.from_dict(
        _require_mapping(
            _require_mapping(methods["h1"], "result.methods.h1").get("ranking"),
            "result.methods.h1.ranking",
        )
    )


def _bootstrap_interval(
    differences: Sequence[float],
    *,
    replicates: int,
    seed: int,
) -> list[float]:
    rng = random.Random(seed)
    values = tuple(float(value) for value in differences)
    if not values:
        return [0.0, 0.0]
    means: list[float] = []
    for _ in range(replicates):
        means.append(
            sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        )
    means.sort()
    lower_index = max(0, int(0.025 * len(means)) - 1)
    upper_index = min(len(means) - 1, int(0.975 * len(means)) - 1)
    return [means[lower_index], means[upper_index]]


def _summary(
    records: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    config_path: Path,
    manifest_path: Path,
    backend: str,
) -> dict[str, Any]:
    """Aggregate frozen primary/secondary metrics and evaluate the ranking gate."""

    if len(records) != 64:
        raise ValueError("Phase 6B summary requires all 64 adjudication records")
    metrics = _require_mapping(config["metrics"], "metrics")
    gate = _require_mapping(config["acceptance_gate"], "acceptance_gate")
    top_k_values = tuple(int(value) for value in metrics["top_k_values"])
    ranks: dict[str, list[int]] = {method: [] for method in METHODS}
    capped: dict[str, list[int]] = {method: [] for method in METHODS}
    overheads: list[float] = []
    top16_set_identity: list[bool] = []
    indices: list[int] = []
    execution_orders: list[tuple[str, ...]] = []
    peak_rss_bytes: list[int] = []
    for record in records:
        methods = _require_mapping(record["methods"], "result.methods")
        instance_metadata = _require_mapping(record["instance"], "result.instance")
        if instance_metadata.get("split_role") != "validation":
            raise ValueError("summary input contains a non-validation instance")
        index = instance_metadata.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("summary input contains an invalid instance index")
        indices.append(index)
        execution_order = instance_metadata.get("execution_order")
        if execution_order not in (["b0", "b1"], ["b1", "b0"]):
            raise ValueError("summary input contains an invalid execution order")
        expected_execution_order = ("b0", "b1") if index % 2 == 0 else ("b1", "b0")
        if tuple(execution_order) != expected_execution_order:
            raise ValueError("summary input violates the frozen counterbalance")
        execution_orders.append(tuple(execution_order))
        provenance = _require_mapping(record["provenance"], "result.provenance")
        rss = provenance.get("peak_rss_bytes")
        if isinstance(rss, bool) or not isinstance(rss, int) or rss <= 0:
            raise ValueError("summary input contains invalid peak RSS")
        peak_rss_bytes.append(rss)
        for method in METHODS:
            evaluation = _require_mapping(methods[method], f"result.methods.{method}")["evaluation"]
            ranks[method].append(int(evaluation["true_rank"]))
            capped[method].append(int(evaluation["capped_attempts"]))
        h1_invariants = _require_mapping(methods["h1"], "result.methods.h1")["invariants"]
        top16_set_identity.append(bool(h1_invariants["top16_set_equals_b1"]))
        overheads.append(float(h1_invariants["b0_window_overhead_fraction_of_b1"]))

    if sorted(indices) != list(range(64)):
        raise ValueError("summary requires exactly one record for each validation index")

    def _stats(method: str) -> dict[str, Any]:
        method_ranks = ranks[method]
        method_capped = capped[method]
        return {
            "exact_true_subkey_ranks": method_ranks,
            "capped_attempts": method_capped,
            "top_k_recall": {
                str(k): sum(rank <= k for rank in method_ranks) for k in top_k_values
            },
            "median_exact_rank": statistics.median(method_ranks),
            "median_capped_attempts": statistics.median(method_capped),
            "mean_capped_attempts": sum(method_capped) / len(method_capped),
            "geometric_mean_rank": math.exp(
                sum(math.log(rank) for rank in method_ranks) / len(method_ranks)
            ),
            "mean_log2_rank": sum(math.log2(rank) for rank in method_ranks) / len(method_ranks),
            "worst_rank": max(method_ranks),
            "solved_instance_curve_positions_1_to_16": {
                str(k): sum(rank <= k for rank in method_ranks) for k in range(1, 17)
            },
        }

    def _paired(first: str, second: str) -> dict[str, int]:
        return {
            "wins": sum(a < b for a, b in zip(ranks[first], ranks[second], strict=True)),
            "ties": sum(a == b for a, b in zip(ranks[first], ranks[second], strict=True)),
            "losses": sum(a > b for a, b in zip(ranks[first], ranks[second], strict=True)),
        }

    b0_capped = capped["b0"]
    b1_capped = capped["b1"]
    h1_capped = capped["h1"]
    b0_difference = [h1 - b0 for h1, b0 in zip(h1_capped, b0_capped, strict=True)]
    b1_difference = [h1 - b1 for h1, b1 in zip(h1_capped, b1_capped, strict=True)]
    median_values = {
        method: statistics.median(capped[method]) for method in METHODS
    }
    h1_median_better = all(
        median_values["h1"] < median_values[baseline] for baseline in ("b0", "b1")
    )
    h1_mean_better = all(
        sum(h1_capped) / len(h1_capped) < sum(capped[baseline]) / len(capped[baseline])
        for baseline in ("b0", "b1")
    )
    max_overhead = max(overheads)
    gate_conditions = {
        "h1_top16_set_equals_b1_on_every_instance": all(top16_set_identity),
        "h1_top16_recall_equals_b1_by_construction": all(top16_set_identity)
        and sum(rank <= 16 for rank in ranks["h1"])
        == sum(rank <= 16 for rank in ranks["b1"]),
        "h1_median_capped_attempts_lower_than_b0_and_b1": h1_median_better,
        "h1_mean_capped_attempts_lower_than_b0_and_b1": h1_mean_better,
        "h1_no_material_b0_window_overhead": max_overhead
        <= float(gate["h1_b0_window_overhead_max_fraction_of_b1"]),
    }
    return {
        "schema": SUMMARY_SCHEMA,
        "phase": 6,
        "study": "phase6b_h1_b0_b1_adjudication",
        "validation_only": True,
        "held_out_test_generated": False,
        "instance_count": len(records),
        "backend": backend,
        "methods": {method: _stats(method) for method in METHODS},
        "paired_rank_win_tie_loss": {
            "h1_vs_b0": _paired("h1", "b0"),
            "h1_vs_b1": _paired("h1", "b1"),
        },
        "execution_order_counts": {
            "b0_then_b1": execution_orders.count(("b0", "b1")),
            "b1_then_b0": execution_orders.count(("b1", "b0")),
        },
        "paired_capped_attempt_mean_difference_h1_minus_baseline": {
            "b0": sum(b0_difference) / len(b0_difference),
            "b1": sum(b1_difference) / len(b1_difference),
        },
        "paired_bootstrap_95_percent_intervals_h1_minus_baseline": {
            "b0": _bootstrap_interval(
                b0_difference,
                replicates=int(metrics["bootstrap_replicates"]),
                seed=int(metrics["bootstrap_seed"]),
            ),
            "b1": _bootstrap_interval(
                b1_difference,
                replicates=int(metrics["bootstrap_replicates"]),
                seed=int(metrics["bootstrap_seed"]) + 1,
            ),
        },
        "timing_seconds": {
            method: {
                "mean_complete_ranking": sum(
                    float(_require_mapping(_require_mapping(record["methods"], "methods")[method], method)["ranking"]["timing_seconds"]["total"])
                    for record in records
                )
                / len(records)
                for method in ("b0", "b1")
            },
            "h1_b0_window_overhead_mean": sum(overheads) / len(overheads),
            "h1_b0_window_overhead_max": max_overhead,
            "peak_rss_bytes_mean": sum(peak_rss_bytes) / len(peak_rss_bytes),
            "peak_rss_bytes_max": max(peak_rss_bytes),
        },
        "gate_conditions": gate_conditions,
        "ranking_gate_passed": all(gate_conditions.values()),
        "config": {
            "path": _relative_or_absolute(config_path),
            "sha256": _sha256_file(config_path),
        },
        "dataset": {
            "manifest_path": _relative_or_absolute(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
        },
        "provenance": {
            "repository_commit": _repository_commit(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch_version": str(torch.__version__),
        },
    }


def _backend_equivalence(
    instance: AttackInstance,
    b0_model: torch.nn.Module,
    b1_model: torch.nn.Module,
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare eager and NumPy on the smoke instance's budget-relevant work."""

    ranking_config = _require_mapping(config["ranking"], "ranking")
    backend_policy = _require_mapping(config["backend_policy"], "backend_policy")
    tolerance = float(backend_policy["score_abs_tolerance"])
    eager_b0, eager_b1 = _execution_models(b0_model, b1_model, "eager")
    numpy_b0, numpy_b1 = _execution_models(b0_model, b1_model, "numpy")

    def _rank_b0(model: torch.nn.Module) -> SubkeyRanking:
        return rank_final_round_subkeys(
            instance,
            model,
            candidate_values=ALL_FINAL_ROUND_SUBKEYS,
            candidate_batch_size=int(ranking_config["b0_candidate_batch_size"]),
            device="cpu",
        )

    def _rank_b1(model: torch.nn.Module) -> PairGroupSubkeyRanking:
        return rank_pair_group_subkeys(
            instance,
            model,  # type: ignore[arg-type]
            candidate_values=ALL_FINAL_ROUND_SUBKEYS,
            candidate_batch_size=int(ranking_config["b1_candidate_batch_size"]),
            device="cpu",
        )

    eager_b0_ranking = _rank_b0(eager_b0)
    numpy_b0_ranking = _rank_b0(numpy_b0)
    eager_b1_ranking = _rank_b1(eager_b1)
    numpy_b1_ranking = _rank_b1(numpy_b1)

    def _max_difference(first: Sequence[float], second: Sequence[float]) -> float:
        return max(abs(a - b) for a, b in zip(first, second, strict=True))

    b1_window_set_equal = set(eager_b1_ranking.top_k(16)) == set(
        numpy_b1_ranking.top_k(16)
    )
    eager_h1 = rank_hierarchical_subkeys(
        instance,
        eager_b1_ranking,
        eager_b0,
        b0_candidate_batch_size=int(ranking_config["h1_b0_window_batch_size"]),
        device="cpu",
        b1_device="cpu",
        b1_inference_backend="pytorch",
    )
    numpy_h1 = rank_hierarchical_subkeys(
        instance,
        numpy_b1_ranking,
        numpy_b0,
        b0_candidate_batch_size=int(ranking_config["h1_b0_window_batch_size"]),
        device="cpu",
        b1_device="cpu",
        b1_inference_backend="numpy",
    )
    eager_window_scores = eager_h1.b0_scores
    numpy_window_scores = numpy_h1.b0_scores
    common_window = sorted(
        set(eager_window_scores).intersection(numpy_window_scores)
    )
    window_score_differences = [
        abs(eager_window_scores[candidate] - numpy_window_scores[candidate])
        for candidate in common_window
    ]
    b0_score_max = _max_difference(
        eager_b0_ranking.candidate_scores,
        numpy_b0_ranking.candidate_scores,
    )
    b1_score_max = _max_difference(
        eager_b1_ranking.candidate_scores,
        numpy_b1_ranking.candidate_scores,
    )
    window_score_max = max(window_score_differences, default=math.inf)
    budget_relevant = {
        "b0_score_within_tolerance": b0_score_max <= tolerance,
        "b0_top16_order_equal": eager_b0_ranking.top_k(16)
        == numpy_b0_ranking.top_k(16),
        "b1_score_within_tolerance": b1_score_max <= tolerance,
        "b1_top16_set_equal": b1_window_set_equal,
        "b1_top16_order_equal": eager_b1_ranking.top_k(16)
        == numpy_b1_ranking.top_k(16),
        "h1_b0_window_set_equal": set(eager_h1.b0_scores)
        == set(numpy_h1.b0_scores),
        "h1_b0_window_scores_within_tolerance": window_score_max <= tolerance,
        "h1_top16_order_equal": eager_h1.top_k(16) == numpy_h1.top_k(16),
    }
    return {
        "schema": "neural_sat_speck.phase6b_backend_equivalence.v1",
        "instance_id": instance.instance_id,
        "candidate_count": len(ALL_FINAL_ROUND_SUBKEYS),
        "score_abs_tolerance": tolerance,
        "b0": {
            "max_abs_score_difference": b0_score_max,
            "top16_order_equal": budget_relevant["b0_top16_order_equal"],
            "complete_order_equal": eager_b0_ranking.candidate_order
            == numpy_b0_ranking.candidate_order,
        },
        "b1": {
            "max_abs_score_difference": b1_score_max,
            "top16_set_equal": b1_window_set_equal,
            "top16_order_equal": budget_relevant["b1_top16_order_equal"],
            "complete_order_equal": eager_b1_ranking.candidate_order
            == numpy_b1_ranking.candidate_order,
        },
        "h1": {
            "b0_window_count": len(eager_h1.b0_scores),
            "b0_window_max_abs_score_difference": window_score_max,
            "top16_set_equal": set(eager_h1.top_k(16))
            == set(numpy_h1.top_k(16)),
            "top16_order_equal": budget_relevant["h1_top16_order_equal"],
            "complete_order_equal": eager_h1.candidate_order
            == numpy_h1.candidate_order,
        },
        "budget_relevant": budget_relevant,
        "passed": all(budget_relevant.values()),
    }


def _require_saved_equivalence(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    manifest_path: Path,
    instance_id: str,
) -> None:
    """Require a current passing smoke equivalence record before full NumPy."""

    equivalence_path = (
        _repository_path(str(config["output_root"])).resolve()
        / "equivalence"
        / f"equivalence_{instance_id}.json"
    )
    if not equivalence_path.is_file():
        raise SystemExit(
            "full NumPy adjudication requires a passing smoke equivalence audit; "
            "run --smoke --backend numpy --equivalence-check first"
        )
    equivalence = _load_json(equivalence_path)
    provenance = _require_mapping(
        equivalence.get("provenance"), "equivalence.provenance"
    )
    if equivalence.get("instance_id") != instance_id:
        raise SystemExit("saved NumPy equivalence audit targets another instance")
    if equivalence.get("passed") is not True:
        raise SystemExit("saved NumPy equivalence audit did not pass")
    if provenance.get("config_sha256") != _sha256_file(config_path):
        raise SystemExit("saved NumPy equivalence audit uses another config")
    if provenance.get("manifest_sha256") != _sha256_file(manifest_path):
        raise SystemExit("saved NumPy equivalence audit uses another manifest")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the validation-only Phase 6B B0/B1/H1 adjudication."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--generate",
        action="store_true",
        help="generate the new validation-only shard; refuses to overwrite it",
    )
    parser.add_argument("--backend", choices=("eager", "numpy"), default="eager")
    parser.add_argument("--smoke", action="store_true", help="rank one instance only")
    parser.add_argument(
        "--determinism-check",
        action="store_true",
        help="rerun the smoke instance and compare scores/orders",
    )
    parser.add_argument(
        "--equivalence-check",
        action="store_true",
        help="compare eager and NumPy on the smoke instance before acceleration",
    )
    parser.add_argument("--first-instance", type=int, default=0)
    parser.add_argument("--last-instance", type=int, default=None)
    parser.add_argument("--rerun", action="store_true", help="ignore matching saved records")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = _repository_path(args.config).resolve()
    config = _load_json(config_path)
    _validate_config(config)
    runtime = _require_mapping(config["runtime"], "runtime")
    torch.set_num_threads(int(runtime["torch_num_threads"]))
    torch.set_num_interop_threads(int(runtime["torch_num_interop_threads"]))
    manifest_path = _write_or_verify_dataset(config, generate=args.generate)
    dataset = _require_mapping(config["dataset"], "dataset")
    attack_set = load_attack_shard(
        manifest_path.parent / "validation.json",
        expected_dataset_id=str(dataset["dataset_id"]),
    )
    if len(attack_set.instances) != 64:
        raise SystemExit("adjudication shard does not contain exactly 64 instances")
    if any(len(instance.pairs) != 64 for instance in attack_set.instances):
        raise SystemExit("adjudication shard contains an instance without 64 pairs")

    b0_model, b1_model, model_metadata = _load_models(config)
    if args.equivalence_check:
        if not args.smoke:
            raise SystemExit("--equivalence-check requires --smoke")
        if args.backend != "numpy":
            raise SystemExit("--equivalence-check requires --backend numpy")
        print("Equivalence audit is intentionally run before any accelerated full run.", flush=True)
        instance = attack_set.instances[args.first_instance]
        equivalence = _backend_equivalence(
            instance,
            b0_model,
            b1_model,
            config=config,
        )
        equivalence["provenance"] = {
            "config_path": _relative_or_absolute(config_path),
            "config_sha256": _sha256_file(config_path),
            "manifest_path": _relative_or_absolute(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "repository_commit": _repository_commit(),
        }
        equivalence_path = (
            _repository_path(str(config["output_root"])).resolve()
            / "equivalence"
            / f"equivalence_{instance.instance_id}.json"
        )
        _write_record(equivalence_path, equivalence)
        print(json.dumps(equivalence, indent=2, sort_keys=True), flush=True)
        if not equivalence["passed"]:
            raise SystemExit(
                "NumPy backend failed the frozen budget-relevant equivalence audit"
            )

    first_instance = args.first_instance
    last_instance = first_instance if args.smoke else (
        63 if args.last_instance is None else args.last_instance
    )
    if not 0 <= first_instance <= last_instance < 64:
        raise SystemExit("instance range is outside the 64-instance adjudication set")
    if args.backend == "numpy" and not args.smoke:
        _require_saved_equivalence(
            config,
            config_path=config_path,
            manifest_path=manifest_path,
            instance_id=attack_set.instances[first_instance].instance_id,
        )
    output_root = _repository_path(str(config["output_root"])).resolve()
    if args.smoke:
        output_root = output_root / "smoke" / args.backend
    else:
        output_root = output_root / args.backend
    execution_b0, _ = _execution_models(b0_model, b1_model, args.backend)
    # The complete B1 order is the authoritative reference input to H1.  The
    # accelerated backend is used for B0's full baseline and the H1 window
    # only; the NumPy B1 result remains an equivalence-audit artefact.
    _, execution_b1 = _execution_models(b0_model, b1_model, "eager")
    records: list[dict[str, Any]] = []
    for index in range(first_instance, last_instance + 1):
        output_path = output_root / f"ranking_{index:04d}.json"
        if output_path.is_file() and not args.rerun and not args.determinism_check:
            record = _load_result(output_path)
            _validate_saved_result(
                record,
                index=index,
                backend=args.backend,
                config_path=config_path,
                manifest_path=manifest_path,
            )
            records.append(record)
            print(f"skip [{index + 1}/{last_instance + 1}] {output_path}", flush=True)
            continue
        instance = attack_set.instances[index]
        started = time.perf_counter()
        b0_ranking, b1_ranking, execution_order = _rank_complete_methods(
            instance,
            execution_b0,
            execution_b1,  # type: ignore[arg-type]
            config=config,
            index=index,
        )
        h1_ranking = rank_hierarchical_subkeys(
            instance,
            b1_ranking,
            execution_b0,
            window_size=HIERARCHICAL_WINDOW_SIZE,
            b0_candidate_batch_size=int(config["ranking"]["h1_b0_window_batch_size"]),
            device="cpu",
            b1_device="cpu",
            b1_inference_backend="pytorch",
            b1_checkpoint_sha256=str(model_metadata["b1"]["checkpoint_sha256"]),
            b0_checkpoint_sha256=str(model_metadata["b0"]["checkpoint_sha256"]),
            b1_model_id=str(model_metadata["b1"]["model_id"]),
            b0_model_id=str(model_metadata["b0"]["model_id"]),
        )
        record = _ranking_record(
            instance,
            b0_ranking,
            b1_ranking,
            h1_ranking,
            config=config,
            manifest_path=manifest_path,
            config_path=config_path,
            model_metadata=model_metadata,
            backend=args.backend,
            index=index,
            execution_order=execution_order,
        )
        record["provenance"]["runner_elapsed_seconds"] = time.perf_counter() - started
        record["provenance"]["peak_rss_bytes"] = _peak_rss_bytes()
        record["provenance"]["torch_num_threads"] = torch.get_num_threads()
        record["provenance"]["torch_num_interop_threads"] = (
            torch.get_num_interop_threads()
        )
        _write_record(output_path, record)
        records.append(record)
        print(
            f"ranked [{index + 1}/{last_instance + 1}] {instance.instance_id} "
            f"B0={record['methods']['b0']['evaluation']['true_rank']} "
            f"B1={record['methods']['b1']['evaluation']['true_rank']} "
            f"H1={record['methods']['h1']['evaluation']['true_rank']} "
            f"elapsed={record['provenance']['runner_elapsed_seconds']:.3f}s",
            flush=True,
        )
        if args.determinism_check:
            second_b0, second_b1, second_execution_order = _rank_complete_methods(
                instance,
                execution_b0,
                execution_b1,  # type: ignore[arg-type]
                config=config,
                index=index,
            )
            if (
                b0_ranking.candidate_scores != second_b0.candidate_scores
                or b0_ranking.candidate_order != second_b0.candidate_order
                or b1_ranking.candidate_scores != second_b1.candidate_scores
                or b1_ranking.candidate_order != second_b1.candidate_order
                or execution_order != second_execution_order
            ):
                raise SystemExit("determinism check failed for complete ranking")
            second_h1 = rank_hierarchical_subkeys(
                instance,
                second_b1,
                execution_b0,
                window_size=HIERARCHICAL_WINDOW_SIZE,
                b0_candidate_batch_size=int(config["ranking"]["h1_b0_window_batch_size"]),
                device="cpu",
                b1_device="cpu",
                b1_inference_backend="pytorch",
                b1_checkpoint_sha256=str(model_metadata["b1"]["checkpoint_sha256"]),
                b0_checkpoint_sha256=str(model_metadata["b0"]["checkpoint_sha256"]),
                b1_model_id=str(model_metadata["b1"]["model_id"]),
                b0_model_id=str(model_metadata["b0"]["model_id"]),
            )
            if (
                h1_ranking.b0_window_scores != second_h1.b0_window_scores
                or h1_ranking.candidate_order != second_h1.candidate_order
            ):
                raise SystemExit("determinism check failed for H1 composition")
            print(f"determinism check passed for {instance.instance_id}", flush=True)

    if not args.smoke:
        summary = _summary(
            records,
            config=config,
            config_path=config_path,
            manifest_path=manifest_path,
            backend=args.backend,
        )
        summary_path = output_root / "summary.json"
        _write_record(summary_path, summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        print(f"Summary: {summary_path}", flush=True)
    else:
        print(f"Smoke output: {output_root}", flush=True)


if __name__ == "__main__":
    main()
