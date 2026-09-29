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
    rank_stage1_ondemand,
)
from tests.fixtures.simon_reference_oracle import oracle_simon32_64_decrypt


def test_ondemand_stage1_removes_temporary_output_on_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import neural_sat_speck.simon.classical_ranking as classical_module

    monkeypatch.setattr(classical_module.tempfile, "tempdir", str(tmp_path))

    def fake_tool(cmd, **_kwargs):
        rows = len(cmd) - 3
        np.ones((rows, 1 << 16), dtype=np.float64).tofile(cmd[2])
        return type("Result", (), {"stdout": "{}"})()

    monkeypatch.setattr(classical_module.subprocess, "run", fake_tool)
    pair = (((0, 0), (0, 0x40)),)
    ranking, _ = rank_stage1_ondemand(pair, screen_pairs=1, rescore_top=4)
    assert ranking.order.tolist() == [0, 1, 2, 3]
    assert list(tmp_path.iterdir()) == []

    def fail_tool(_cmd, **_kwargs):
        raise RuntimeError("forced tool failure")

    monkeypatch.setattr(classical_module.subprocess, "run", fail_tool)
    with pytest.raises(RuntimeError, match="forced tool failure"):
        rank_stage1_ondemand(pair, screen_pairs=1, rescore_top=4)
    assert list(tmp_path.iterdir()) == []


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


def test_tie_count_and_collision_count_semantics() -> None:
    """Verify distinct semantics of top-1 candidate tie count vs all-candidate score collisions."""
    # Case A: Flat table - all scores identical
    class FlatTable:
        rounds = 7

        def log_score_row(self, left: int) -> np.ndarray:
            return np.zeros(1 << 16, dtype=np.float64)

    res_flat = rank_classical_domain(_pairs(32), FlatTable())
    assert res_flat.screen_tie_count == 1 << 16  # All candidates tied for top-1
    assert res_flat.screen_collision_count == (1 << 16) - 1  # 65535 adjacent collisions
    assert res_flat.full_tie_count == 512
    assert res_flat.full_collision_count == 511

    # Case B: Unique top candidate, collisions only among lower-ranked candidates
    class UniqueTopTable:
        rounds = 7

        def log_score_row(self, left: int) -> np.ndarray:
            # Anchor left difference for pair 0 with candidate 0 is anchor[0]
            # Set right difference 0 to score 10.0, rest to 0.0
            row = np.zeros(1 << 16, dtype=np.float64)
            row[0] = 10.0
            return row

    res_unique = rank_classical_domain(_pairs(32), UniqueTopTable())
    # 8 candidates match diff=0 in Feistel inversion and tie for top-1
    assert res_unique.screen_tie_count == 8
    # The remaining 65,528 candidates all have score 0.0, yielding 65,534 collisions
    assert res_unique.screen_collision_count == 65534
    assert res_unique.screen_tie_count != res_unique.screen_collision_count


def test_rank_classical_domain_rescores_all_64_pairs() -> None:
    """CRITICAL REQUIREMENT: verify rank_classical_domain does NOT default to 32 when given 64 pairs."""
    accessed_pairs = []

    class CountingTable:
        rounds = 7

        def log_score_row(self, left: int) -> np.ndarray:
            accessed_pairs.append(left)
            return np.zeros(1 << 16, dtype=np.float64)

    pairs_64 = _pairs(64)
    rank_classical_domain(pairs_64, CountingTable())
    # All 64 pairs should have their left difference rows accessed for rescoring
    assert len(accessed_pairs) >= 64, "All 64 pairs must be processed, not truncated to 32!"
