"""Truth-blind SIMON candidate ranking from complete Markov-predicted DDTs.

The DDTs are offline public distributions. This module accepts ciphertext
pairs and explicit candidates only; it has no master-key or truth-label input.
"""

from __future__ import annotations

import struct
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cipher import Block
from .ranking_features import candidate_difference_words, candidate_transformed_words

DENOMINATOR = 1 << 60
LOG_ZERO = 1e-64
SPARSE_RECORD = np.dtype([("index", "<u4"), ("mass", "<u8")])


class FullPredictedDdt:
    """Memory-mapped, complete 6R sparse or 7R dense public distribution."""

    def __init__(self, path: Path | str, expected_rounds: int) -> None:
        if expected_rounds not in (6, 7):
            raise ValueError("expected rounds must be six or seven")
        self.path = Path(path)
        with self.path.open("rb") as handle:
            header = handle.read(32)
        if len(header) != 32 or header[:8] not in (b"SMDT0001", b"SMDT0002"):
            raise ValueError("invalid predicted-DDT header")
        bits, rounds, initial, exponent, count = struct.unpack("<IIIIQ", header[8:])
        if (bits, rounds, initial, exponent) != (16, expected_rounds, 0x40, 60):
            raise ValueError("unexpected predicted-DDT metadata")
        if (rounds == 6 and header[:8] != b"SMDT0001") or (
            rounds == 7 and (header[:8] != b"SMDT0002" or count != 1 << 32)
        ):
            raise ValueError("predicted-DDT storage does not match depth")
        expected_bytes = 32 + count * (12 if rounds == 6 else 8)
        if self.path.stat().st_size != expected_bytes:
            raise ValueError("predicted-DDT file length mismatch")
        self.rounds = rounds
        self.count = count
        if rounds == 6:
            self._records = np.memmap(self.path, dtype=SPARSE_RECORD, mode="r", offset=32, shape=(count,))
            self._indices = self._records["index"]
            self._dense = None
        else:
            self._dense = np.memmap(self.path, dtype="<u8", mode="r", offset=32, shape=(1 << 16, 1 << 16))
            self._records = None
            self._indices = None

    def mass_row(self, left: int) -> np.ndarray:
        if not 0 <= left <= 0xFFFF:
            raise ValueError("left difference must be a 16-bit word")
        if self._dense is not None:
            row = np.asarray(self._dense[left], dtype=np.uint64)
        else:
            assert self._indices is not None and self._records is not None
            start = int(np.searchsorted(self._indices, left << 16, side="left"))
            end = (
                self.count
                if left == 0xFFFF
                else int(np.searchsorted(self._indices, (left + 1) << 16, side="left"))
            )
            records = self._records[start:end]
            row = np.zeros(1 << 16, dtype=np.uint64)
            row[records["index"] & np.uint32(0xFFFF)] = records["mass"]
        if np.any(row > DENOMINATOR):
            raise ValueError("predicted-DDT probability exceeds one")
        return row

    def log_score_row(self, left: int) -> np.ndarray:
        masses = self.mass_row(left)
        scores = np.log2(masses.astype(np.float64) / DENOMINATOR + LOG_ZERO)
        if not np.all(np.isfinite(scores)):
            raise ValueError("nonfinite predicted-DDT score")
        return scores


@dataclass(frozen=True)
class DomainRanking:
    """Screened-domain result; ``order`` contains 512 full-rescored words."""

    order: np.ndarray
    screen_seconds: float
    row_load_seconds: float
    rescore_seconds: float
    sort_seconds: float
    total_seconds: float
    unique_left_rows: int
    screen_tie_count: int
    full_tie_count: int


def _score_candidates(
    pairs: Sequence[tuple[Block, Block]],
    candidates: Sequence[int],
    log_rows: np.ndarray,
    expected_left: np.ndarray,
    *,
    final_key_candidate: int | None,
) -> np.ndarray:
    transformed = candidate_transformed_words(
        pairs, candidates, final_key_candidate=final_key_candidate
    )
    differences = candidate_difference_words(transformed)
    if not np.array_equal(differences[:, :, 0], np.broadcast_to(expected_left, differences[:, :, 0].shape)):
        raise AssertionError("post-zero-inverse left difference depends on candidate")
    right = differences[:, :, 1].astype(np.intp, copy=False)
    scores = log_rows[np.arange(len(pairs))[None, :], right].sum(axis=1, dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise ValueError("nonfinite candidate score")
    return scores


def rank_classical_domain(
    ciphertext_pairs: Sequence[tuple[Block, Block]],
    table: FullPredictedDdt,
    *,
    final_key_candidate: int | None = None,
    screen_pairs: int = 16,
    full_pairs: int = 32,
    rescore_top: int = 512,
    batch_size: int = 1024,
) -> DomainRanking:
    """Score the complete 16-bit domain, then fully rescore a fixed screen.

    Higher summed Bao log probability ranks first. Ties use ascending key word.
    Runtime includes stage-specific row acquisition, transforms, and sorts.
    """
    if table.rounds != (7 if final_key_candidate is None else 6):
        raise ValueError("DDT depth does not match candidate stage")
    if not 1 <= screen_pairs <= full_pairs <= len(ciphertext_pairs):
        raise ValueError("invalid pair counts")
    if not 1 <= rescore_top <= 1 << 16 or not 1 <= batch_size <= 1 << 16:
        raise ValueError("invalid ranking width or batch size")
    started = time.perf_counter()
    pairs = ciphertext_pairs[:full_pairs]
    t0 = time.perf_counter()
    anchor = candidate_difference_words(
        candidate_transformed_words(pairs, [0], final_key_candidate=final_key_candidate)
    )[0, :, 0]
    row_cache: dict[int, np.ndarray] = {}
    for left in anchor:
        word = int(left)
        if word not in row_cache:
            row_cache[word] = table.log_score_row(word)
    log_rows = np.stack([row_cache[int(left)] for left in anchor])
    if not np.all(np.isfinite(log_rows)):
        raise ValueError("nonfinite predicted-DDT score row")
    row_load_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    screen_scores = np.empty(1 << 16, dtype=np.float64)
    for start in range(0, 1 << 16, batch_size):
        candidates = range(start, min(start + batch_size, 1 << 16))
        screen_scores[start : start + len(candidates)] = _score_candidates(
            pairs[:screen_pairs], candidates, log_rows[:screen_pairs], anchor[:screen_pairs],
            final_key_candidate=final_key_candidate,
        )
    screen_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    candidates_all = np.arange(1 << 16, dtype=np.uint16)
    screened_order = np.lexsort((candidates_all, -screen_scores))
    finalists = candidates_all[screened_order[:rescore_top]]
    screen_tie_count = int(np.count_nonzero(screen_scores == screen_scores[screened_order[0]]))
    first_sort_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    full_scores = _score_candidates(
        pairs, finalists, log_rows, anchor, final_key_candidate=final_key_candidate
    )
    rescore_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    local_order = np.lexsort((finalists, -full_scores))
    order = finalists[local_order]
    full_tie_count = int(np.count_nonzero(full_scores == full_scores[local_order[0]]))
    sort_seconds = first_sort_seconds + time.perf_counter() - t0
    return DomainRanking(
        order=order,
        screen_seconds=screen_seconds,
        row_load_seconds=row_load_seconds,
        rescore_seconds=rescore_seconds,
        sort_seconds=sort_seconds,
        total_seconds=time.perf_counter() - started,
        unique_left_rows=len(row_cache),
        screen_tie_count=screen_tie_count,
        full_tie_count=full_tie_count,
    )
