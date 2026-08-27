"""Run the Phase 4 full final-round subkey ranking baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from neural_sat_speck.attacks.ranking import (
    ALL_FINAL_ROUND_SUBKEYS,
    evaluate_subkey_ranking,
    rank_final_round_subkeys,
)
from neural_sat_speck.data.attack import (
    ATTACK_SPLIT_ROLES,
    load_attack_shard,
    verify_attack_benchmark,
)
from neural_sat_speck.neural.train import load_checkpoint

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT / "configs" / "experiments" / "phase4_ranking_r6.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"Could not read JSON configuration {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"Configuration {path} must contain a JSON object")
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


def _require_int(config: dict[str, Any], field: str) -> int:
    value = config.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f"configuration field {field!r} must be an integer")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank complete 16-bit SPECK32/64 final-round subkey candidates "
            "with the verified Phase 3 distinguisher."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=(
            "Phase 4 JSON configuration "
            f"(default: {DEFAULT_CONFIG.relative_to(REPOSITORY_ROOT)})"
        ),
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=None,
        help="Pilot override; use the first ascending candidates only.",
    )
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=None,
        help="Override the configured candidate batch size for a pilot.",
    )
    parser.add_argument(
        "--instance-index",
        type=int,
        default=None,
        help="Override the zero-based instance index in the configured split.",
    )
    parser.add_argument(
        "--pair-count",
        type=int,
        default=None,
        help="Pilot override; use the prefix of the instance's related pairs.",
    )
    parser.add_argument(
        "--split-role",
        choices=ATTACK_SPLIT_ROLES,
        default=None,
        help="Override the configured attack split role.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda"),
        default=None,
        help="Override the configured inference device.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the JSON result path from the configuration.",
    )
    return parser.parse_args()


def _validate_model_metadata(
    metadata: dict[str, Any],
    *,
    expected_dataset_id: str,
    expected_rounds: int,
) -> None:
    if metadata.get("phase") != 3:
        raise SystemExit("the selected checkpoint is not marked as a Phase 3 model")
    if metadata.get("dataset_id") != expected_dataset_id:
        raise SystemExit(
            "checkpoint dataset_id does not match the Phase 4 configuration"
        )
    dataset = metadata.get("dataset")
    if not isinstance(dataset, dict):
        raise SystemExit("checkpoint metadata does not contain split provenance")
    for role in ("train", "validation", "test"):
        split_metadata = dataset.get(role)
        if not isinstance(split_metadata, dict):
            raise SystemExit(f"checkpoint metadata is missing {role} provenance")
        if split_metadata.get("rounds") != expected_rounds:
            raise SystemExit(
                f"checkpoint {role} rounds do not match {expected_rounds}"
            )


def main() -> None:
    args = _parse_args()
    config_path = _repository_path(args.config).resolve()
    config = _load_json(config_path)
    if config.get("schema") != "neural_sat_speck.phase4_ranking_config.v1":
        raise SystemExit("ranking config schema is not recognised")
    if config.get("phase") != 4:
        raise SystemExit("the selected configuration is not a Phase 4 config")

    split_role = args.split_role or config.get("split_role")
    if split_role not in ATTACK_SPLIT_ROLES:
        raise SystemExit("split_role must be train, validation, or test")
    instance_index = (
        args.instance_index
        if args.instance_index is not None
        else _require_int(config, "instance_index")
    )
    if instance_index < 0:
        raise SystemExit("instance_index must be non-negative")

    candidate_count = (
        args.candidate_count
        if args.candidate_count is not None
        else _require_int(config, "candidate_count")
    )
    if not 1 <= candidate_count <= len(ALL_FINAL_ROUND_SUBKEYS):
        raise SystemExit("candidate_count must be between 1 and 65536")
    candidate_batch_size = (
        args.candidate_batch_size
        if args.candidate_batch_size is not None
        else _require_int(config, "candidate_batch_size")
    )
    if candidate_batch_size <= 0:
        raise SystemExit("candidate_batch_size must be positive")
    if config.get("score_aggregation") != "sum_logits":
        raise SystemExit("Phase 4 currently requires score_aggregation=sum_logits")
    stability_pair_subsets = config.get("stability_pair_subsets")
    stability_top_k = config.get("stability_top_k", 16)
    if isinstance(stability_top_k, bool) or not isinstance(stability_top_k, int):
        raise SystemExit("stability_top_k must be an integer")

    expected_difference_value = config.get("expected_input_difference")
    if (
        not isinstance(expected_difference_value, list)
        or len(expected_difference_value) != 2
    ):
        raise SystemExit("expected_input_difference must contain two words")
    if any(
        isinstance(word, bool) or not isinstance(word, int)
        for word in expected_difference_value
    ):
        raise SystemExit("expected_input_difference words must be integers")
    expected_difference = tuple(expected_difference_value)

    checkpoint_path = _repository_path(str(config["checkpoint"])).resolve()
    attack_root = _repository_path(str(config["attack_dataset_root"])).resolve()
    manifest_path = attack_root / "manifest.json"
    output_path = _repository_path(
        args.output if args.output is not None else str(config["output"])
    ).resolve()
    device = args.device or str(config.get("device", "cpu"))

    print("=" * 70)
    print("SPECK32/64 FINAL-ROUND SUBKEY RANKING (PHASE 4)")
    print(f"Config: {config_path}")
    print(f"Attack manifest: {manifest_path}")
    print(f"Model checkpoint: {checkpoint_path}")
    print(f"Split and instance: {split_role}[{instance_index}]")
    print(f"Candidate count: {candidate_count}")
    print(f"Candidate batch size: {candidate_batch_size}")
    print(f"Device: {device}")
    if stability_pair_subsets is not None:
        print(
            f"Pair-subset stability: {len(stability_pair_subsets)} subsets, "
            f"top-{stability_top_k}"
        )
    print("=" * 70)

    expected_attack_dataset_id = str(config["expected_attack_dataset_id"])
    attack_manifest = verify_attack_benchmark(
        manifest_path,
        expected_dataset_id=expected_attack_dataset_id,
        expected_rounds=_require_int(config, "expected_rounds"),
        expected_input_difference=expected_difference,
    )
    split_record = attack_manifest["splits"][split_role]
    attack_set = load_attack_shard(
        attack_root / str(split_record["path"]),
        expected_dataset_id=expected_attack_dataset_id,
    )
    if instance_index >= len(attack_set.instances):
        raise SystemExit(
            f"instance_index {instance_index} is outside the {split_role} split"
        )
    instance = attack_set.instances[instance_index]
    if args.pair_count is not None:
        if args.pair_count <= 0 or args.pair_count > len(instance.pairs):
            raise SystemExit(
                "pair-count must be between one and the instance pair count"
            )
        instance = replace(instance, pairs=instance.pairs[: args.pair_count])
    print(
        f"Verified {split_role} shard: {len(attack_set.instances)} instances, "
        f"{len(instance.pairs)} pairs for {instance.instance_id}"
    )

    if not checkpoint_path.is_file():
        raise SystemExit(f"checkpoint does not exist: {checkpoint_path}")
    model, model_metadata = load_checkpoint(checkpoint_path, device=device)
    _validate_model_metadata(
        model_metadata,
        expected_dataset_id=str(config["expected_model_dataset_id"]),
        expected_rounds=_require_int(config, "expected_model_rounds"),
    )
    print(f"Loaded model with {model.parameter_count:,} trainable parameters")

    ranking = rank_final_round_subkeys(
        instance,
        model,
        candidate_values=range(candidate_count),
        candidate_batch_size=candidate_batch_size,
        device=device,
        stability_pair_subsets=stability_pair_subsets,
        stability_top_k=stability_top_k,
    )
    configured_top_k_values = tuple(config.get("top_k_values", ()))
    for top_k in configured_top_k_values:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise SystemExit("top_k_values must contain non-negative integers")
    usable_top_k_values = tuple(
        top_k for top_k in configured_top_k_values if top_k <= candidate_count
    )
    evaluation = evaluate_subkey_ranking(
        ranking,
        instance.final_round_subkey,
        top_k_values=usable_top_k_values,
    )
    evaluation["requested_top_k_values"] = list(configured_top_k_values)
    evaluation["omitted_top_k_values"] = [
        top_k
        for top_k in configured_top_k_values
        if top_k > candidate_count
    ]
    result: dict[str, Any] = {
        "schema": "neural_sat_speck.phase4_ranking_result.v1",
        "phase": 4,
        "protocol": (
            "six-round plaintext-difference attack instances scored by the "
            "five-round Phase 3 raw-logit classifier"
        ),
        "cipher": attack_manifest["cipher"],
        "task": attack_manifest["task"],
        "dataset": {
            "dataset_id": attack_manifest["dataset_id"],
            "manifest_path": _relative_or_absolute(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "repository_commit": attack_manifest["repository_commit"],
            "generated_at": attack_manifest["generated_at"],
            "rounds": attack_manifest["rounds"],
            "input_difference": attack_manifest["input_difference"],
            "word_order": attack_manifest["word_order"],
        },
        "model": {
            "dataset_id": model_metadata["dataset_id"],
            "checkpoint_path": _relative_or_absolute(checkpoint_path),
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "model_config": model.config.to_dict(),
            "parameter_count": model.parameter_count,
        },
        "instance": {
            "split_role": split_role,
            "index": instance_index,
            "instance_id": instance.instance_id,
            "pair_count": len(instance.pairs),
        },
        "candidate_domain": {
            "first": 0,
            "last": candidate_count - 1,
            "count": candidate_count,
            "complete_16_bit_domain": candidate_count == len(ALL_FINAL_ROUND_SUBKEYS),
        },
        "score_aggregation": ranking.score_aggregation,
        "ranking": ranking.to_dict(),
        "evaluation": evaluation,
        "provenance": {
            "repository_commit": _repository_commit(),
            "config_path": _relative_or_absolute(config_path),
            "config_sha256": _sha256_file(config_path),
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch_version": str(torch.__version__),
            "device": device,
            "command": [sys.executable, *sys.argv],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"True final-round subkey: {instance.final_round_subkey:#06x}; "
        f"deterministic rank: {evaluation['true_rank']}"
    )
    print(f"Top-k membership: {evaluation['top_k_membership']}")
    print(f"Ranking time: {ranking.total_seconds:.3f}s")
    print(f"Result saved to: {output_path}")


if __name__ == "__main__":
    main()
