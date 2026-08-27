"""Run one complete-domain B0/B1/B2/B3 Phase 6A ranking job."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch

from neural_sat_speck.attacks.group_ranking import (
    evaluate_pair_group_subkey_ranking,
    rank_pair_group_subkeys,
)
from neural_sat_speck.attacks.ranking import (
    ALL_FINAL_ROUND_SUBKEYS,
    evaluate_subkey_ranking,
    rank_final_round_subkeys,
)
from neural_sat_speck.data.attack import load_attack_shard
from neural_sat_speck.data.phase6a import verify_phase6a_validation_attack
from neural_sat_speck.neural.group_model import PAIR_GROUP_VARIANTS
from neural_sat_speck.neural.group_train import load_pair_group_checkpoint
from neural_sat_speck.neural.numpy_inference import (
    NumpyInferenceModule,
    NumpyRankerInference,
)
from neural_sat_speck.neural.train import load_checkpoint

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT / "configs" / "experiments" / "phase6a_ranking.json"
)
METHODS = ("b0", *PAIR_GROUP_VARIANTS)
ROLES = ("selection", "confirmation")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"could not read configuration {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit("configuration must contain a JSON object")
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank the full final-round subkey domain with one Phase 6 method."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--instance-index", type=int, required=True)
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=None,
        help="pilot-only ascending prefix; omit for the frozen complete domain",
    )
    parser.add_argument("--candidate-batch-size", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default=None)
    parser.add_argument(
        "--inference-backend",
        choices=("eager", "numpy"),
        default="eager",
        help="reference PyTorch inference or the validated CPU NumPy path",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _attack_record(config: dict[str, Any], role: str) -> dict[str, Any]:
    records = config.get("attack_manifests")
    if not isinstance(records, dict) or role not in records:
        raise SystemExit(f"configuration is missing the {role} attack manifest")
    record = records[role]
    if not isinstance(record, dict):
        raise SystemExit(f"attack manifest record for {role} must be an object")
    return record


def _load_model(
    method: str,
    config: dict[str, Any],
    *,
    device: str,
) -> tuple[torch.nn.Module, dict[str, Any], Path]:
    if method == "b0":
        checkpoint_path = _repository_path(str(config["baseline_checkpoint"])).resolve()
        model, metadata = load_checkpoint(checkpoint_path, device=device)
        return model, metadata, checkpoint_path
    checkpoints = config.get("candidate_checkpoints")
    if not isinstance(checkpoints, dict) or method not in checkpoints:
        raise SystemExit(f"configuration is missing the {method} checkpoint")
    checkpoint_path = _repository_path(str(checkpoints[method])).resolve()
    model, metadata = load_pair_group_checkpoint(checkpoint_path, device=device)
    if metadata.get("model_id") != model.model_id:
        raise SystemExit(f"{method} checkpoint metadata has the wrong model id")
    return model, metadata, checkpoint_path


def main() -> None:
    args = _parse_args()
    config_path = _repository_path(args.config).resolve()
    config = _load_json(config_path)
    if config.get("schema") != "neural_sat_speck.phase6a_ranking_config.v1":
        raise SystemExit("Phase 6A ranking configuration schema is not recognised")
    if args.instance_index < 0:
        raise SystemExit("instance-index must be non-negative")
    candidate_count = int(
        config["candidate_count"] if args.candidate_count is None else args.candidate_count
    )
    if not 1 <= candidate_count <= len(ALL_FINAL_ROUND_SUBKEYS):
        raise SystemExit("candidate-count must be between 1 and 65536")
    candidate_batch_size = int(
        config["candidate_batch_size"]
        if args.candidate_batch_size is None
        else args.candidate_batch_size
    )
    if candidate_batch_size <= 0:
        raise SystemExit("candidate-batch-size must be positive")
    device = args.device or str(config.get("device", "cpu"))

    attack_record = _attack_record(config, args.role)
    manifest_path = _repository_path(str(attack_record["manifest"])).resolve()
    dataset_id = str(attack_record["dataset_id"])
    manifest = verify_phase6a_validation_attack(
        manifest_path,
        expected_dataset_id=dataset_id,
    )
    shard_record = manifest["splits"]["validation"]
    attack_set = load_attack_shard(
        manifest_path.parent / str(shard_record["path"]),
        expected_dataset_id=dataset_id,
    )
    if args.instance_index >= len(attack_set.instances):
        raise SystemExit(
            f"instance-index {args.instance_index} exceeds the validation shard"
        )
    instance = attack_set.instances[args.instance_index]
    if len(instance.pairs) != 64:
        raise SystemExit("Phase 6A ranking requires exactly 64 attack pairs")

    model, model_metadata, checkpoint_path = _load_model(
        args.method,
        config,
        device=device,
    )
    ranking_model: torch.nn.Module = model
    if args.inference_backend == "numpy":
        if device != "cpu":
            raise SystemExit("the NumPy inference backend only supports CPU")
        ranking_model = NumpyInferenceModule(NumpyRankerInference(model))
    candidates = range(candidate_count)
    configured_top_k_values = tuple(
        config.get("top_k_values", (1, 16, 32, 64))
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in configured_top_k_values
    ):
        raise SystemExit("top_k_values must contain non-negative integers")
    usable_top_k_values = tuple(
        value for value in configured_top_k_values if value <= candidate_count
    )
    if args.method == "b0":
        ranking = rank_final_round_subkeys(
            instance,
            ranking_model,
            candidate_values=candidates,
            candidate_batch_size=candidate_batch_size,
            device=device,
        )
        evaluation = evaluate_subkey_ranking(
            ranking,
            instance.final_round_subkey,
            top_k_values=usable_top_k_values,
        )
    else:
        ranking = rank_pair_group_subkeys(
            instance,
            ranking_model,  # type: ignore[arg-type]
            candidate_values=candidates,
            candidate_batch_size=candidate_batch_size,
            device=device,
        )
        evaluation = evaluate_pair_group_subkey_ranking(
            ranking,
            instance.final_round_subkey,
            top_k_values=usable_top_k_values,
        )
    evaluation["requested_top_k_values"] = list(configured_top_k_values)
    evaluation["omitted_top_k_values"] = [
        value for value in configured_top_k_values if value > candidate_count
    ]

    output_path = (
        _repository_path(args.output).resolve()
        if args.output is not None
        else _repository_path(str(config["output_root"])).resolve()
        / args.role
        / args.method
        / f"ranking_{args.instance_index:04d}.json"
    )
    result: dict[str, Any] = {
        "schema": "neural_sat_speck.phase6a_ranking_result.v1",
        "phase": 6,
        "study": "phase6a_b1_b2_b3_pair_aware_rankers",
        "method": args.method,
        "protocol": (
            "six-round attack instance, one inverse final round, and 64-pair "
            "five-round candidate scoring"
        ),
        "dataset": {
            "dataset_id": manifest["dataset_id"],
            "manifest_path": _relative_or_absolute(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "generated_at": manifest["generated_at"],
            "rounds": manifest["rounds"],
            "input_difference": manifest["input_difference"],
            "word_order": manifest["word_order"],
            "validation_only": manifest["validation_only"],
        },
        "model": {
            "model_id": (
                "speck32_64_b0_gohr_baseline"
                if args.method == "b0"
                else model.model_id  # type: ignore[attr-defined]
            ),
            "checkpoint_path": _relative_or_absolute(checkpoint_path),
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "metadata": model_metadata,
            "parameter_count": int(
                model.parameter_count  # type: ignore[attr-defined]
            ),
            "inference_backend": args.inference_backend,
        },
        "instance": {
            "split_role": "validation",
            "validation_role": args.role,
            "index": args.instance_index,
            "instance_id": instance.instance_id,
            "pair_count": len(instance.pairs),
        },
        "candidate_domain": {
            "first": 0,
            "last": candidate_count - 1,
            "count": candidate_count,
            "complete_16_bit_domain": candidate_count == len(ALL_FINAL_ROUND_SUBKEYS),
        },
        "ranking": ranking.to_dict(),
        "evaluation": evaluation,
        "provenance": {
            "config_path": _relative_or_absolute(config_path),
            "config_sha256": _sha256_file(config_path),
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch_version": str(torch.__version__),
            "device": device,
            "inference_backend": args.inference_backend,
            "command": [sys.executable, *sys.argv],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"{args.method} {args.role}[{args.instance_index}] rank="
        f"{evaluation['true_rank']} top16="
        f"{evaluation['top_k_membership'].get('16')} time="
        f"{ranking.total_seconds:.3f}s"
    )
    print(f"Result: {output_path}")


if __name__ == "__main__":
    main()
