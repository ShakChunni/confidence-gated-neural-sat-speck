"""Truth-blind candidate ranking using audited published SIMON VV checkpoints."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .cipher import Block
from .ranking_features import bao_full_value_bits, candidate_value_words


@dataclass(frozen=True)
class NeuralDomainRanking:
    order: np.ndarray
    feature_seconds: float
    inference_seconds: float
    aggregation_seconds: float
    sort_seconds: float
    total_seconds: float
    screen_tie_count: int
    full_tie_count: int


def _log_odds_sum(probabilities: np.ndarray, pair_count: int) -> np.ndarray:
    if pair_count <= 0:
        raise ValueError("pair count must be positive")
    if probabilities.ndim != 2 or probabilities.shape[1] != 1:
        raise ValueError("VV model must return one probability per feature row")
    if probabilities.shape[0] % pair_count:
        raise ValueError("VV output count does not divide into candidate pairs")
    p = probabilities.astype(np.float64, copy=False)
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("VV output is nonfinite or outside [0,1]")
    odds = p / (1 - p + 1e-64)
    scores = np.log2(odds + 1e-64).reshape(-1, pair_count).sum(axis=1, dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise ValueError("VV aggregate score is nonfinite")
    return scores


def _score_candidates(
    pairs: Sequence[tuple[Block, Block]],
    candidates: Sequence[int],
    model: Any,
    *,
    final_key_candidate: int | None,
    feature_batch_size: int,
    inference_batch_size: int,
) -> tuple[np.ndarray, float, float, float]:
    started = time.perf_counter()
    features = np.empty((len(candidates) * len(pairs), 64), dtype=np.float32)
    for start in range(0, len(candidates), feature_batch_size):
        candidate_batch = candidates[start : start + feature_batch_size]
        states = candidate_value_words(
            pairs, candidate_batch, final_key_candidate=final_key_candidate
        )
        bits = bao_full_value_bits(states).reshape(-1, 64)
        position = start * len(pairs)
        features[position : position + len(bits)] = bits
    feature_seconds = time.perf_counter() - started
    t0 = time.perf_counter()
    predictions = np.asarray(model.predict(features, batch_size=inference_batch_size, verbose=0))
    inference_seconds = time.perf_counter() - t0
    del features
    t0 = time.perf_counter()
    scores = _log_odds_sum(predictions, len(pairs))
    aggregation_seconds = time.perf_counter() - t0
    return scores, feature_seconds, inference_seconds, aggregation_seconds


def rank_neural_domain(
    ciphertext_pairs: Sequence[tuple[Block, Block]],
    model: Any,
    *,
    final_key_candidate: int | None = None,
    screen_pairs: int = 16,
    full_pairs: int | None = None,
    rescore_top: int = 512,
    feature_batch_size: int = 1024,
    inference_batch_size: int = 16384,
) -> NeuralDomainRanking:
    """Screen all 16-bit keys then fully rescore a fixed shortlist.

    The loaded model must be VV8 for Stage 1 or VV7 for conditional Stage 2;
    that provenance is checked by :func:`load_published_vv_checkpoint` before
    this function receives it. No truth label or verification pair is accepted.
    """
    actual_full_pairs = len(ciphertext_pairs) if full_pairs is None else full_pairs
    if not 1 <= screen_pairs <= actual_full_pairs <= len(ciphertext_pairs):
        raise ValueError("invalid pair counts")
    if not 1 <= rescore_top <= 1 << 16:
        raise ValueError("invalid rescore width")
    if not 1 <= feature_batch_size <= 1 << 16 or inference_batch_size <= 0:
        raise ValueError("invalid feature/inference batch size")
    started = time.perf_counter()
    candidates = range(1 << 16)
    screen_scores, feature_seconds, inference_seconds, aggregation_seconds = _score_candidates(
        ciphertext_pairs[:screen_pairs], candidates, model,
        final_key_candidate=final_key_candidate,
        feature_batch_size=feature_batch_size,
        inference_batch_size=inference_batch_size,
    )
    t0 = time.perf_counter()
    candidate_words = np.arange(1 << 16, dtype=np.uint16)
    screen_order = np.lexsort((candidate_words, -screen_scores))
    finalists = candidate_words[screen_order[:rescore_top]]
    screen_tie_count = int(np.count_nonzero(screen_scores == screen_scores[screen_order[0]]))
    sort_seconds = time.perf_counter() - t0
    full_scores, feature_time, inference_time, aggregation_time = _score_candidates(
        ciphertext_pairs[:actual_full_pairs], finalists, model,
        final_key_candidate=final_key_candidate,
        feature_batch_size=feature_batch_size,
        inference_batch_size=inference_batch_size,
    )
    feature_seconds += feature_time
    inference_seconds += inference_time
    aggregation_seconds += aggregation_time
    t0 = time.perf_counter()
    full_order = np.lexsort((finalists, -full_scores))
    full_tie_count = int(np.count_nonzero(full_scores == full_scores[full_order[0]]))
    sort_seconds += time.perf_counter() - t0
    return NeuralDomainRanking(
        order=finalists[full_order],
        feature_seconds=feature_seconds,
        inference_seconds=inference_seconds,
        aggregation_seconds=aggregation_seconds,
        sort_seconds=sort_seconds,
        total_seconds=time.perf_counter() - started,
        screen_tie_count=screen_tie_count,
        full_tie_count=full_tie_count,
    )
