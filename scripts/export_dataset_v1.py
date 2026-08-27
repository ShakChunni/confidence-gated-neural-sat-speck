"""Export the inspectable Dataset v1 classifier and attack fixtures."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from neural_sat_speck.data.attack import (
    AttackInstanceConfig,
    export_attack_benchmark,
)
from neural_sat_speck.data.export import export_classifier_benchmark
from neural_sat_speck.data.generator import DatasetConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_CONFIG_PATH = (
    REPOSITORY_ROOT / "configs" / "data" / "dataset_v1_classifier.json"
)
ATTACK_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "data" / "dataset_v1_attack.json"


def _repository_commit() -> str | None:
    """Return the current Git commit when the repository is available."""

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
    return result.stdout.strip()


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the tiny Dataset v1 classifier and attack fixtures. "
            "This script is for inspectable regeneration, not large-scale "
            "dataset publication."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "tests" / "fixtures" / "dataset_v1",
        help="Directory that will receive classifier/ and attack/ subtrees.",
    )
    parser.add_argument(
        "--include-commit",
        action="store_true",
        help="Record the current Git commit in exported metadata.",
    )
    parser.add_argument(
        "--classifier-config",
        type=Path,
        default=CLASSIFIER_CONFIG_PATH,
        help="JSON configuration for the classifier export.",
    )
    parser.add_argument(
        "--attack-config",
        type=Path,
        default=ATTACK_CONFIG_PATH,
        help="JSON configuration for the attack export.",
    )
    parser.add_argument(
        "--skip-classifier",
        action="store_true",
        help="Do not generate the classifier subtree.",
    )
    parser.add_argument(
        "--skip-attack",
        action="store_true",
        help="Do not generate the attack subtree.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    commit = _repository_commit() if args.include_commit else None

    output: dict[str, object] = {"output_dir": str(args.output_dir)}

    if not args.skip_classifier:
        classifier_config = _load_json(args.classifier_config)
        classifier_splits = classifier_config["splits"]
        classifier_configs = tuple(
            DatasetConfig(
                sample_count=int(classifier_splits[role]["sample_count"]),
                rounds=int(classifier_config["rounds"]),
                seed=int(classifier_splits[role]["seed"]),
                split_role=role,
                input_difference=tuple(classifier_config["input_difference"]),
            )
            for role in ("train", "validation", "test")
        )
        output["classifier_manifest"] = export_classifier_benchmark(
            classifier_configs,
            args.output_dir / "classifier",
            dataset_id=str(classifier_config["dataset_id"]),
            generated_at=str(classifier_config["generated_at"]),
            repository_commit=commit,
        )
    if not args.skip_attack:
        attack_config = _load_json(args.attack_config)
        attack_splits = attack_config["splits"]
        attack_configs = tuple(
            AttackInstanceConfig(
                instance_count=int(attack_splits[role]["instance_count"]),
                pair_count=int(attack_splits[role]["pair_count"]),
                rounds=int(attack_config["rounds"]),
                seed=int(attack_splits[role]["seed"]),
                split_role=role,
                input_difference=tuple(attack_config["input_difference"]),
            )
            for role in ("train", "validation", "test")
        )
        output["attack_manifest"] = export_attack_benchmark(
            attack_configs,
            args.output_dir / "attack",
            dataset_id=str(attack_config["dataset_id"]),
            generated_at=str(attack_config["generated_at"]),
            repository_commit=commit,
        )

    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
