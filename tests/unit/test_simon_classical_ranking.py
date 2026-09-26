"""Source-index, independent-oracle, zero-bin, and tie checks for DDT ranking."""

from __future__ import annotations

import random
import struct
from pathlib import Path

import numpy as np
import pytest

from neural_sat_speck.simon.classical_ranking import (
    DENOMINATOR,
    FullPredictedDdt,
    _score_candidates,
    rank_classical_domain,
)
from tests.fixtures.simon_reference_oracle import oracle_simon32_64_decrypt


def _pairs(count: int) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    rng = random.Random(260926)
    return tuple(
        (
            (rng.randrange(1 << 16), rng.randrange(1 << 16)),
            (rng.randrange(1 << 16), rng.randrange(1 << 16)),
        )
        for _ in range(count)
    )


def test_sparse_full_distribution_row_has_exact_zero_bins(tmp_path: Path) -> None:
    path = tmp_path / "sparse.bin"
    records = (
        ((0x1234 << 16) | 7, DENOMINATOR // 2),
        ((0xFFFF << 16) | 0xFFFF, DENOMINATOR // 2),
    )
    header = b"SMDT0001" + struct.pack("<IIIIQ", 16, 6, 0x40, 60, len(records))
    path.write_bytes(header + b"".join(struct.pack("<IQ", *record) for record in records))
    table = FullPredictedDdt(path, 6)
    assert table.mass_row(0).sum() == 0
    assert table.mass_row(0x1234)[7] == DENOMINATOR // 2
    assert table.mass_row(0x1234)[8] == 0
    assert table.mass_row(0xFFFF)[0xFFFF] == DENOMINATOR // 2
    assert table.log_score_row(0x1234)[7] == pytest.approx(-1)
    assert table.log_score_row(0x1234)[8] == pytest.approx(np.log2(1e-64))
    with pytest.raises(ValueError, match="length"):
        path.write_bytes(header)
        FullPredictedDdt(path, 6)


@pytest.mark.parametrize("final_key_candidate", [None, 0, 0x1234])
def test_row_gather_score_matches_independent_scalar_source_index(
    final_key_candidate: int | None,
) -> None:
    pairs = _pairs(4)
    candidates = (0, 1, 0x1234, 0xFFFF)
    rng = np.random.default_rng(26)
    log_rows = rng.integers(-20, 1, size=(len(pairs), 1 << 16)).astype(np.float64)
    expected_left = []
    expected_scores = []
    for candidate in candidates:
        score = 0.0
        for pair_index, pair in enumerate(pairs):
            states = []
            for block in pair:
                if final_key_candidate is not None:
                    block = oracle_simon32_64_decrypt(
                        block, (0, 0, 0, final_key_candidate), 1
                    )
                block = oracle_simon32_64_decrypt(block, (0, 0, 0, candidate), 1)
                states.append(oracle_simon32_64_decrypt(block, (0, 0, 0, 0), 1))
            left = states[0][0] ^ states[1][0]
            right = states[0][1] ^ states[1][1]
            if candidate == candidates[0]:
                expected_left.append(left)
            else:
                assert left == expected_left[pair_index]
            score += log_rows[pair_index, right]
        expected_scores.append(score)
    np.testing.assert_array_equal(
        _score_candidates(
            pairs, candidates, log_rows, np.asarray(expected_left, dtype=np.uint16),
            final_key_candidate=final_key_candidate,
        ),
        expected_scores,
    )


@pytest.mark.parametrize("final_key_candidate,depth", [(None, 7), (7, 6)])
def test_complete_screen_and_rescore_use_deterministic_ascending_ties(
    final_key_candidate: int | None, depth: int
) -> None:
    class FlatTable:
        rounds = depth

        def log_score_row(self, left: int) -> np.ndarray:
            assert 0 <= left <= 0xFFFF
            return np.zeros(1 << 16, dtype=np.float64)

    result = rank_classical_domain(
        _pairs(32), FlatTable(), final_key_candidate=final_key_candidate
    )
    np.testing.assert_array_equal(result.order, np.arange(512, dtype=np.uint16))
    assert result.screen_tie_count == 1 << 16
    assert result.full_tie_count == 512
    assert 1 <= result.unique_left_rows <= 32


def test_nan_row_and_wrong_stage_fail_explicitly() -> None:
    class InvalidTable:
        rounds = 7

        def log_score_row(self, left: int) -> np.ndarray:
            result = np.zeros(1 << 16, dtype=np.float64)
            result[0] = np.nan
            return result

    with pytest.raises(ValueError, match="nonfinite"):
        rank_classical_domain(_pairs(32), InvalidTable())
    with pytest.raises(ValueError, match="depth"):
        rank_classical_domain(_pairs(32), InvalidTable(), final_key_candidate=0)
