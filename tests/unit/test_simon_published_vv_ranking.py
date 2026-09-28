"""Score polarity, finite-output, and deterministic-tie checks for VV ranking."""

from __future__ import annotations

import math

import numpy as np
import pytest

from neural_sat_speck.simon.cipher import decrypt_round
from neural_sat_speck.simon.published_vv_ranking import (
    _log_odds_sum,
    _score_candidates,
    rank_neural_domain,
)
from neural_sat_speck.simon.ranking_features import (
    bao_full_value_bits,
    candidate_value_words,
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


def test_vv9_and_vv8_feature_and_score_aggregation_pinned_reference() -> None:
    """Check VV9 and VV8 feature extraction and log-odds score aggregation against pinned reference."""
    # Pinned synthetic ciphertext pairs: 2 pairs of (block_a, block_b)
    pairs = (
        ((0x1234, 0x5678), (0x9ABC, 0xDEF0)),
        ((0x2345, 0x6789), (0xABCD, 0xEF01)),
    )
    candidates = (0x0000, 0x0040)

    # 1. VV9 (Stage 1): final_key_candidate is None
    words_vv9 = candidate_value_words(pairs, candidates, final_key_candidate=None)
    assert words_vv9.shape == (2, 2, 4)
    assert words_vv9.dtype == np.uint16

    # Verify candidate 0, pair 0, blocks a and b match manual single-round Feistel decryption with key 0x0000
    dec_a_c0 = decrypt_round((0x1234, 0x5678), 0x0000)
    dec_b_c0 = decrypt_round((0x9ABC, 0xDEF0), 0x0000)
    np.testing.assert_array_equal(words_vv9[0, 0], (dec_a_c0[0], dec_a_c0[1], dec_b_c0[0], dec_b_c0[1]))

    # Verify candidate 1, pair 1, blocks a and b match manual single-round Feistel decryption with key 0x0040
    dec_a_c1 = decrypt_round((0x2345, 0x6789), 0x0040)
    dec_b_c1 = decrypt_round((0xABCD, 0xEF01), 0x0040)
    np.testing.assert_array_equal(words_vv9[1, 1], (dec_a_c1[0], dec_a_c1[1], dec_b_c1[0], dec_b_c1[1]))

    # Binary bit conversion: (2 keys * 2 pairs = 4 rows, 64 bits each)
    bits_vv9 = bao_full_value_bits(words_vv9).reshape(4, 64)
    assert bits_vv9.shape == (4, 64)
    assert bits_vv9.dtype == np.uint8
    assert np.all((bits_vv9 == 0) | (bits_vv9 == 1))

    # Bit-level verification for row 0 (cand 0, pair 0)
    row0_words = (dec_a_c0[0], dec_a_c0[1], dec_b_c0[0], dec_b_c0[1])
    expected_bits_row0 = np.array([int(b) for word in row0_words for b in f"{word:016b}"], dtype=np.uint8)
    np.testing.assert_array_equal(bits_vv9[0], expected_bits_row0)

    # 2. VV8 (Stage 2): final_key_candidate = 0x4321
    parent_k9 = 0x4321
    words_vv8 = candidate_value_words(pairs, candidates, final_key_candidate=parent_k9)
    assert words_vv8.shape == (2, 2, 4)

    # Verify candidate 0, pair 0: decrypt first with parent_k9, then with child key 0x0000
    dec_p0_a = decrypt_round(decrypt_round((0x1234, 0x5678), parent_k9), 0x0000)
    dec_p0_b = decrypt_round(decrypt_round((0x9ABC, 0xDEF0), parent_k9), 0x0000)
    np.testing.assert_array_equal(words_vv8[0, 0], (dec_p0_a[0], dec_p0_a[1], dec_p0_b[0], dec_p0_b[1]))

    bits_vv8 = bao_full_value_bits(words_vv8).reshape(4, 64)
    assert bits_vv8.shape == (4, 64)

    # 3. Log-odds score aggregation check
    # Pinned probability outputs for 2 candidates x 2 pairs
    pinned_probs = np.array([[0.80], [0.60], [0.30], [0.70]], dtype=np.float32)
    scores = _log_odds_sum(pinned_probs, pair_count=2)
    expected_score_c0 = math.log2(0.80 / (1 - 0.80)) + math.log2(0.60 / (1 - 0.60))
    expected_score_c1 = math.log2(0.30 / (1 - 0.30)) + math.log2(0.70 / (1 - 0.70))
    np.testing.assert_allclose(scores, [expected_score_c0, expected_score_c1], atol=1e-12)
    assert scores[0] > scores[1]  # candidate 0 strictly outranks candidate 1


def test_rank_neural_domain_rescores_all_64_pairs() -> None:
    """Verify rank_neural_domain does NOT silently default to 32 pairs when given 64."""
    call_log = []

    class MockModel:
        def predict(self, features: np.ndarray, *, batch_size: int, verbose: int) -> np.ndarray:
            call_log.append(len(features))
            return np.full((len(features), 1), 0.5, dtype=np.float32)

    pairs_64 = tuple(
        (((i, i), (i, i))) for i in range(64)
    )
    rank_neural_domain(pairs_64, MockModel(), screen_pairs=16, rescore_top=512)
    # Screen call: 65536 candidates * 16 pairs = 1,048,576 feature rows
    assert call_log[0] == 65536 * 16
    # Rescore call: 512 candidates * 64 pairs = 32,768 feature rows (NOT 512 * 32 = 16,384!)
    assert call_log[1] == 512 * 64
    assert call_log[1] != 512 * 32, "Ranker must NOT silently default to 32 pairs during rescore!"
