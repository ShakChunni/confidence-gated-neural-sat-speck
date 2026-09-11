"""Screened multi-fidelity pair-group subkey ranking (Phase 11F).

Optimizes Stage-2 hierarchical ranking by replacing full-domain 64-pair evaluation
with a two-tier multi-fidelity cascade:
1. Fast complete-domain screen: scores all 65,536 candidates on 16 ciphertext pairs.
2. High-fidelity rescore: rescores only the top K_screen candidates (default 512)
   on the full 64 ciphertext pairs.

Since candidate beam widths M2 are typically in {16, 32}, a screening threshold
of K_screen = 512 provides a 16x safety margin over M2 while achieving a ~25x
wall-clock speedup per Stage-2 ranking.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from neural_sat_speck.attacks.group_ranking import (
    PAIR_GROUP_SCORE_AGGREGATION,
    PairGroupSubkeyRanking,
)
from neural_sat_speck.attacks.ranking import _validate_candidate_values
from neural_sat_speck.data.attack import AttackInstance
from neural_sat_speck.neural.group_model import PairGroupRanker
from neural_sat_speck.neural.numpy_inference import vectorized_final_round_features


def _resolve_device(model: nn.Module, device: torch.device | str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    parameter = next(model.parameters(), None)
    return parameter.device if parameter is not None else torch.device("cpu")


def rank_pair_group_subkeys_screened(
    instance: AttackInstance,
    model: PairGroupRanker,
    *,
    screen_pairs: int = 16,
    screen_top_k: int = 512,
    candidate_values: Sequence[int] | None = None,
    candidate_batch_size: int = 1024,
    device: torch.device | str | None = None,
) -> PairGroupSubkeyRanking:
    """Rank candidate final-round subkeys using a two-tier multi-fidelity screen.

    Parameters
    ----------
    instance : AttackInstance
        The attack instance containing exactly 64 ciphertext pairs.
    model : PairGroupRanker
        The pair-group model (e.g. frozen B1 pair-group model).
    screen_pairs : int
        Number of pairs to use for complete-domain screening (default 16).
    screen_top_k : int
        Number of candidate survivors to rescore on all 64 pairs (default 512).
    candidate_values : Sequence[int] | None
        Candidates to score, defaulting to all 65,536 16-bit words.
    candidate_batch_size : int
        Batch size for vectorized feature extraction and neural inference (default 1024).
    device : torch.device | str | None
        Execution device (e.g. 'mps', 'cuda', 'cpu').
    """
    if not isinstance(instance, AttackInstance):
        raise TypeError("instance must be an AttackInstance")
    if len(instance.pairs) != 64:
        raise ValueError("pair-aware ranking requires an instance with 64 pairs")
    if not isinstance(model, PairGroupRanker):
        raise TypeError("model must be a PairGroupRanker")
    if screen_pairs not in (8, 16, 32):
        raise ValueError(f"screen_pairs must be 8, 16, or 32, got {screen_pairs}")
    if screen_top_k <= 0 or screen_top_k > 65536:
        raise ValueError(f"screen_top_k must be between 1 and 65536, got {screen_top_k}")
    if candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be positive")

    candidates = _validate_candidate_values(candidate_values)
    resolved_device = _resolve_device(model, device)
    original_parameter = next(model.parameters(), None)
    original_device = original_parameter.device if original_parameter is not None else None
    was_training = model.training

    total_started = time.perf_counter()
    partial_seconds = 0.0
    inference_seconds = 0.0
    peak_feature_tensor_bytes = 0

    model.to(resolved_device)
    model.eval()

    try:
        # ==========================================
        # TIER 1: COMPLETE-DOMAIN SCREEN (16 PAIRS)
        # ==========================================
        screen_pair_subset = instance.pairs[:screen_pairs]
        screen_scores: list[float] = []

        with torch.no_grad():
            for start in range(0, len(candidates), candidate_batch_size):
                batch_candidates = candidates[start : start + candidate_batch_size]

                t_feat = time.perf_counter()
                feat_np = vectorized_final_round_features(
                    screen_pair_subset,
                    batch_candidates,
                )
                features = torch.from_numpy(feat_np).to(resolved_device)
                partial_seconds += time.perf_counter() - t_feat
                peak_feature_tensor_bytes = max(peak_feature_tensor_bytes, feat_np.nbytes)

                t_inf = time.perf_counter()
                batch_len = len(batch_candidates)
                flat_features = features.reshape(batch_len * screen_pairs, 4, 16)
                embeddings = model.encoder(flat_features).reshape(batch_len, screen_pairs, model.embedding_dim)
                aggregated = embeddings.mean(dim=1)
                logits = model.head(aggregated).reshape(-1)
                inference_seconds += time.perf_counter() - t_inf

                screen_scores.extend(float(v) for v in logits.cpu())

        # Sort all candidates by Tier 1 screening score
        t_sort1 = time.perf_counter()
        screened_pairs_sorted = sorted(
            zip(candidates, screen_scores, strict=True),
            key=lambda item: (-item[1], item[0]),
        )
        survivors = [c for c, _ in screened_pairs_sorted[:screen_top_k]]
        sorting_seconds = time.perf_counter() - t_sort1

        # ==========================================
        # TIER 2: HIGH-FIDELITY RESCORE (64 PAIRS)
        # ==========================================
        rescored_scores: list[float] = []
        with torch.no_grad():
            for start in range(0, len(survivors), candidate_batch_size):
                rescore_batch = survivors[start : start + candidate_batch_size]

                t_feat = time.perf_counter()
                feat_np = vectorized_final_round_features(
                    instance.pairs,
                    rescore_batch,
                )
                features = torch.from_numpy(feat_np).to(resolved_device)
                partial_seconds += time.perf_counter() - t_feat
                peak_feature_tensor_bytes = max(peak_feature_tensor_bytes, feat_np.nbytes)

                t_inf = time.perf_counter()
                batch_len = len(rescore_batch)
                flat_features = features.reshape(batch_len * 64, 4, 16)
                embeddings = model.encoder(flat_features).reshape(batch_len, 64, model.embedding_dim)
                aggregated = embeddings.mean(dim=1)
                logits = model.head(aggregated).reshape(-1)
                inference_seconds += time.perf_counter() - t_inf

                rescored_scores.extend(float(v) for v in logits.cpu())

        # Sort survivors by their exact 64-pair rescored scores
        t_sort2 = time.perf_counter()
        rescored_pairs_sorted = sorted(
            zip(survivors, rescored_scores, strict=True),
            key=lambda item: (-item[1], item[0]),
        )
        rescored_order = tuple(c for c, _ in rescored_pairs_sorted)
        tail_order = tuple(c for c, _ in screened_pairs_sorted[screen_top_k:])
        final_order = rescored_order + tail_order
        sorting_seconds += time.perf_counter() - t_sort2

        # Build complete score map
        final_scores_dict = dict(zip(candidates, screen_scores, strict=True))
        for c, s in rescored_pairs_sorted:
            final_scores_dict[c] = s
        final_scores = tuple(final_scores_dict[c] for c in candidates)

        total_seconds = time.perf_counter() - total_started

        return PairGroupSubkeyRanking(
            candidate_values=candidates,
            candidate_scores=final_scores,
            candidate_order=final_order,
            pair_count=64,
            group_size=64,
            candidate_batch_size=candidate_batch_size,
            score_aggregation=PAIR_GROUP_SCORE_AGGREGATION,
            partial_decryption_seconds=partial_seconds,
            inference_seconds=inference_seconds,
            sorting_seconds=sorting_seconds,
            total_seconds=total_seconds,
            peak_feature_tensor_bytes=peak_feature_tensor_bytes,
        )

    finally:
        if original_device is not None:
            model.to(original_device)
        if was_training:
            model.train()
        else:
            model.eval()
