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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    commit = _repository_commit() if args.include_commit else None

    classifier_config = _load_json(CLASSIFIER_CONFIG_PATH)
    classifier_splits = classifier_config["splits"]
    classifier_configs = tuple(
        DatasetConfig(
            sample_count=int(classifier_splits[role]["sample_count"]),
            rounds=int(classifier_config["rounds"]),
            seed=int(classifier_splits[role]["seed"]),
            split_role=role,
        )
        for role in ("train", "validation", "test")
    )
    classifier_manifest = export_classifier_benchmark(
        classifier_configs,
        args.output_dir / "classifier",
        generated_at=str(classifier_config["generated_at"]),
        repository_commit=commit,
    )

    attack_config = _load_json(ATTACK_CONFIG_PATH)
    attack_splits = attack_config["splits"]
    attack_configs = tuple(
        AttackInstanceConfig(
            instance_count=int(attack_splits[role]["instance_count"]),
            pair_count=int(attack_splits[role]["pair_count"]),
            rounds=int(attack_config["rounds"]),
            seed=int(attack_splits[role]["seed"]),
            split_role=role,
        )
        for role in ("train", "validation", "test")
    )
    attack_manifest = export_attack_benchmark(
        attack_configs,
        args.output_dir / "attack",
        generated_at=str(attack_config["generated_at"]),
        repository_commit=commit,
    )

    print(
        json.dumps(
            {
                "classifier_manifest": classifier_manifest,
                "attack_manifest": attack_manifest,
                "output_dir": str(args.output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
