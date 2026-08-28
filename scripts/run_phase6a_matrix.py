"""Run the resumable Phase 6A complete-domain ranking matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from neural_sat_speck.neural.group_model import PAIR_GROUP_VARIANTS

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "experiments" / "phase6a_ranking.json"
METHODS = ("b0", *PAIR_GROUP_VARIANTS)
ROLES = ("selection", "confirmation")


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or resume the Phase 6A complete-domain ranking matrix."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--backend",
        choices=("eager", "numpy", "mixed"),
        default="eager",
        help="inference backend for all methods, or eager B0/NumPy pair rankers",
    )
    parser.add_argument("--candidate-batch-size", type=int, default=4096)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--roles", nargs="+", choices=ROLES, default=ROLES)
    parser.add_argument("--first-instance", type=int, default=0)
    parser.add_argument("--last-instance", type=int, default=None)
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="rerun matching result files instead of resuming them",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _backend_for(method: str, selected: str) -> str:
    if selected == "mixed":
        return "eager" if method == "b0" else "numpy"
    return selected


def _result_path(output_root: Path, role: str, method: str, index: int) -> Path:
    return output_root / role / method / f"ranking_{index:04d}.json"


def _matches_run(
    path: Path,
    *,
    method: str,
    role: str,
    candidate_batch_size: int,
    backend: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        record = _load_json(path)
    except SystemExit:
        return False
    ranking = record.get("ranking")
    model = record.get("model")
    instance = record.get("instance")
    candidate_domain = record.get("candidate_domain")
    return bool(
        record.get("method") == method
        and isinstance(instance, dict)
        and instance.get("validation_role") == role
        and isinstance(candidate_domain, dict)
        and candidate_domain.get("complete_16_bit_domain") is True
        and isinstance(ranking, dict)
        and ranking.get("candidate_batch_size") == candidate_batch_size
        and isinstance(model, dict)
        and model.get("inference_backend") == backend
    )


def main() -> None:
    args = _parse_args()
    if args.candidate_batch_size <= 0:
        raise SystemExit("candidate-batch-size must be positive")
    if args.first_instance < 0:
        raise SystemExit("first-instance must be non-negative")
    config_path = _repository_path(args.config).resolve()
    config = _load_json(config_path)
    if config.get("schema") != "neural_sat_speck.phase6a_ranking_config.v1":
        raise SystemExit("Phase 6A ranking configuration schema is not recognised")
    instance_count = int(config.get("validation_instance_count", 32))
    last_instance = (
        instance_count - 1 if args.last_instance is None else args.last_instance
    )
    if not 0 <= args.first_instance <= last_instance < instance_count:
        raise SystemExit("instance range is outside the configured validation set")
    output_root = _repository_path(str(config["output_root"])).resolve()
    rank_script = REPOSITORY_ROOT / "scripts" / "rank_phase6a_candidates.py"
    total_jobs = len(args.methods) * len(args.roles) * (
        last_instance - args.first_instance + 1
    )
    completed = 0
    skipped = 0
    for role in args.roles:
        for method in args.methods:
            backend = _backend_for(method, args.backend)
            for index in range(args.first_instance, last_instance + 1):
                output_path = _result_path(output_root, role, method, index)
                if not args.rerun and _matches_run(
                    output_path,
                    method=method,
                    role=role,
                    candidate_batch_size=args.candidate_batch_size,
                    backend=backend,
                ):
                    skipped += 1
                    print(
                        f"[{skipped + completed}/{total_jobs}] skip {method} "
                        f"{role}[{index}] ({backend})",
                        flush=True,
                    )
                    continue
                command = [
                    sys.executable,
                    str(rank_script),
                    "--config",
                    str(config_path),
                    "--method",
                    method,
                    "--role",
                    role,
                    "--instance-index",
                    str(index),
                    "--candidate-batch-size",
                    str(args.candidate_batch_size),
                    "--inference-backend",
                    backend,
                ]
                print(
                    f"[{skipped + completed + 1}/{total_jobs}] run {method} "
                    f"{role}[{index}] ({backend})",
                    flush=True,
                )
                if not args.dry_run:
                    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
                completed += 1
    print(
        f"Matrix jobs selected={total_jobs} completed={completed} skipped={skipped}",
        flush=True,
    )


if __name__ == "__main__":
    main()
