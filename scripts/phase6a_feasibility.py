"""Run the tiny Phase 6A/6B group-model feasibility checks."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from neural_sat_speck.attacks.group_ranking import rank_pair_group_subkeys
from neural_sat_speck.data.attack import AttackInstanceConfig, generate_attack_instances
from neural_sat_speck.data.pair_groups import PairGroupConfig, generate_pair_groups
from neural_sat_speck.neural.group_dataset import (
    PairGroupDataset,
    create_pair_group_dataloader,
)
from neural_sat_speck.neural.group_model import build_pair_group_model
from neural_sat_speck.neural.group_train import (
    PairGroupTrainingConfig,
    evaluate_pair_group,
    train_pair_group_ranker,
)
from neural_sat_speck.neural.train import load_checkpoint

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPOSITORY_ROOT / "models" / "speck32_64_phase3_r5_distinguisher.pt"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tiny pair-group feasibility checks.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checkpoint = args.checkpoint
    if not checkpoint.is_absolute():
        checkpoint = REPOSITORY_ROOT / checkpoint
    baseline, _ = load_checkpoint(checkpoint.resolve(), device="cpu")
    baseline.eval()

    generated = generate_pair_groups(
        PairGroupConfig(8, 64, 5, 61001, "train"),
        generated_at="2026-08-27T00:00:00+08:00",
        include_debug_pairs=True,
    )
    repeated = generate_pair_groups(
        PairGroupConfig(8, 64, 5, 61001, "train"),
        generated_at="2026-08-27T00:00:00+08:00",
    )
    changed = generate_pair_groups(
        PairGroupConfig(8, 64, 5, 61002, "train"),
        generated_at="2026-08-27T00:00:00+08:00",
    )
    assert np.array_equal(generated.features, repeated.features)
    assert np.array_equal(generated.labels, repeated.labels)
    assert not np.array_equal(generated.features, changed.features)
    print(
        f"data: features={generated.features.shape} {generated.features.dtype}, "
        f"labels={generated.labels.shape} {generated.labels.dtype}, "
        f"balance={generated.labels.tolist().count(0)}/"
        f"{generated.labels.tolist().count(1)}"
    )
    print(f"metadata: {generated.metadata}")

    features = torch.from_numpy(generated.features).to(torch.float32)
    for variant in ("b1_mean", "b2_moments", "b3_weighted"):
        model = build_pair_group_model(baseline, variant=variant, seed=63001)
        model.eval()
        permutation = torch.tensor(list(reversed(range(64))))
        with torch.no_grad():
            original = model(features[:2])
            permuted = model(features[:2, permutation])
        difference = float(torch.max(torch.abs(original - permuted)))
        print(f"{variant}: permutation max_abs_difference={difference:.3e}")
        if difference > 1e-6:
            raise AssertionError(f"{variant} is not permutation invariant")

    train_dataset = PairGroupDataset.from_generated_dataset(generated)
    train_model = build_pair_group_model(baseline, variant="b1_mean", seed=63001)
    loader = create_pair_group_dataloader(
        train_dataset,
        batch_size=4,
        shuffle=False,
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    before = evaluate_pair_group(train_model, loader, criterion, torch.device("cpu"))
    train_result = train_pair_group_ranker(
        train_model,
        train_dataset,
        train_dataset,
        PairGroupTrainingConfig(epochs=5, batch_size=4, device="cpu"),
    )
    after = evaluate_pair_group(train_model, loader, criterion, torch.device("cpu"))
    print(
        f"overfit fixture: BCE {before['loss']:.6f} -> {after['loss']:.6f}; "
        f"best_epoch={train_result['best_epoch']}"
    )
    if not after["loss"] < before["loss"]:
        raise AssertionError("B1 did not improve the tiny train-only BCE")

    attack_instance = generate_attack_instances(
        AttackInstanceConfig(1, 64, 6, 64001, "validation"),
        generated_at="2026-08-27T00:00:00+08:00",
    ).instances[0]
    rank_model = build_pair_group_model(baseline, variant="b1_mean", seed=63001)
    rank_model.eval()
    started = time.perf_counter()
    first = rank_pair_group_subkeys(
        attack_instance,
        rank_model,
        candidate_values=range(16),
        candidate_batch_size=1,
    )
    elapsed = time.perf_counter() - started
    second = rank_pair_group_subkeys(
        attack_instance,
        rank_model,
        candidate_values=range(16),
        candidate_batch_size=8,
    )
    assert torch.allclose(
        torch.tensor(first.candidate_scores),
        torch.tensor(second.candidate_scores),
        atol=1e-6,
        rtol=0.0,
    )
    print(
        f"ranking subset: 16 candidates, batch sizes 1/8 agree within 1e-6; "
        f"elapsed={elapsed:.3f}s, peak_features={first.peak_feature_tensor_bytes} bytes"
    )
    print(f"torch threads={torch.get_num_threads()}, device=cpu")


if __name__ == "__main__":
    main()
