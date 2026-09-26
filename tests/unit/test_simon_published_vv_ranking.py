"""Score polarity, finite-output, and deterministic-tie checks for VV ranking."""

from __future__ import annotations

import math

import numpy as np
import pytest

from neural_sat_speck.simon.published_vv_ranking import (
    _log_odds_sum,
    _score_candidates,
    rank_neural_domain,
)
from tests.fixtures.simon_reference_oracle import oracle_simon32_64_decrypt


def test_published_log_odds_polarity_matches_source_formula() -> None:
    probabilities = np.array([[0.25], [0.75], [0.5], [0.5]], dtype=np.float32)
    scores = _log_odds_sum(probabilities, 2)
    expected = [
        sum(math.log2(p / (1 - p + 1e-64) + 1e-64) for p in (0.25, 0.75)),
        0.0,
    ]
    np.testing.assert_allclose(scores, expected, atol=1e-12)
    assert _log_odds_sum(np.array([[0.75], [0.75]]), 1)[0] > _log_odds_sum(
        np.array([[0.25], [0.25]]), 1
    )[0]
    assert np.all(np.isfinite(_log_odds_sum(np.array([[0.0], [1.0]]), 1)))


@pytest.mark.parametrize(
    "malformed",
    [
        np.array([[np.nan]]),
        np.array([[np.inf]]),
        np.array([[-0.1]]),
        np.array([[1.1]]),
        np.array([[0.5, 0.5]]),
    ],
)
def test_published_log_odds_rejects_malformed_outputs(malformed: np.ndarray) -> None:
    with pytest.raises(ValueError):
        _log_odds_sum(malformed, 1)


@pytest.mark.parametrize("final_key_candidate", [None, 7])
def test_complete_domain_uses_candidate_word_ties(final_key_candidate: int | None) -> None:
    class ConstantModel:
        def predict(self, features: np.ndarray, *, batch_size: int, verbose: int) -> np.ndarray:
            assert features.ndim == 2 and features.shape[1] == 64
            assert np.all((features == 0) | (features == 1))
            assert batch_size == 16384 and verbose == 0
            return np.full((len(features), 1), 0.5, dtype=np.float32)

    pairs = (((0x1234, 0x5678), (0xABCD, 0xEF01)),) * 2
    result = rank_neural_domain(
        pairs, ConstantModel(), final_key_candidate=final_key_candidate,
        screen_pairs=1, full_pairs=2, rescore_top=4,
    )
    np.testing.assert_array_equal(result.order, np.arange(4, dtype=np.uint16))
    assert result.screen_tie_count == 1 << 16
    assert result.full_tie_count == 4


def test_neural_ranker_rejects_invalid_pair_budget() -> None:
    with pytest.raises(ValueError, match="pair counts"):
        rank_neural_domain((), object())


@pytest.mark.parametrize("final_key_candidate", [None, 0x5678])
def test_neural_candidate_pair_order_matches_independent_scalar_oracle(
    final_key_candidate: int | None,
) -> None:
    class BitModel:
        def predict(self, features: np.ndarray, *, batch_size: int, verbose: int) -> np.ndarray:
            assert features.shape[1] == 64 and batch_size == 16384 and verbose == 0
            return (0.2 + 0.6 * (features[:, 0] + features[:, 17]) / 2).reshape(-1, 1)

    pairs = (
        ((0x1234, 0x5678), (0x9ABC, 0xDEF0)),
        ((0x5555, 0x7777), (0xAAAA, 0x3333)),
    )
    candidates = (0, 1, 0x1234)
    actual, *_ = _score_candidates(
        pairs, candidates, BitModel(), final_key_candidate=final_key_candidate,
        feature_batch_size=2, inference_batch_size=16384,
    )
    expected = []
    for candidate in candidates:
        score = 0.0
        for pair in pairs:
            states = []
            for block in pair:
                if final_key_candidate is not None:
                    block = oracle_simon32_64_decrypt(
                        block, (0, 0, 0, final_key_candidate), 1
                    )
                states.append(oracle_simon32_64_decrypt(block, (0, 0, 0, candidate), 1))
            words = (*states[0], *states[1])
            bits = "".join(f"{word:016b}" for word in words)
            probability = 0.2 + 0.6 * (int(bits[0]) + int(bits[17])) / 2
            score += math.log2(probability / (1 - probability + 1e-64) + 1e-64)
        expected.append(score)
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)
