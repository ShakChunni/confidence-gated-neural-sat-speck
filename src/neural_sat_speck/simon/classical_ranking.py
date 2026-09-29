"""Truth-blind SIMON candidate ranking from complete Markov-predicted DDTs.

The DDTs are offline public distributions. This module accepts ciphertext
pairs and explicit candidates only; it has no master-key or truth-label input.
"""

from __future__ import annotations

import json
import struct
import subprocess
import tempfile
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
    """Screened-domain result; ``order`` contains full-rescored words.

    Tie-count and collision semantics:
    - ``screen_tie_count``: Top-1 candidate tie count during initial screening
      (number of candidates tied for the highest screening score).
    - ``full_tie_count``: Top-1 candidate tie count during final rescoring
      (number of candidates tied for the highest rescore score).
    - ``screen_collision_count``: All-candidate score collision count during screening
      (total pairwise adjacent duplicate scores across all 65,536 sorted domain candidates).
    - ``full_collision_count``: All-candidate score collision count during rescoring
      (total pairwise adjacent duplicate scores among sorted rescored finalists).
    """

    order: np.ndarray
    screen_seconds: float
    row_load_seconds: float
    rescore_seconds: float
    sort_seconds: float
    total_seconds: float
    unique_left_rows: int
    screen_tie_count: int
    full_tie_count: int
    screen_collision_count: int = 0
    full_collision_count: int = 0
    screen_pairs_used: int = 16
    full_pairs_used: int = 0


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
    expected_rounds: int | None = None,
    screen_pairs: int = 16,
    full_pairs: int | None = None,
    rescore_top: int = 512,
    batch_size: int = 1024,
) -> DomainRanking:
    """Score the complete 16-bit domain, then fully rescore a fixed screen.

    Higher summed Bao log probability ranks first. Ties use ascending key word.
    Runtime includes stage-specific row acquisition, transforms, and sorts.
    """
    target_rounds = (
        expected_rounds
        if expected_rounds is not None
        else (7 if final_key_candidate is None else 6)
    )
    if table.rounds != target_rounds:
        raise ValueError(
            f"DDT depth {table.rounds} does not match candidate stage {target_rounds}"
        )
    actual_full_pairs = len(ciphertext_pairs) if full_pairs is None else full_pairs
    if not 1 <= screen_pairs <= actual_full_pairs <= len(ciphertext_pairs):
        raise ValueError("invalid pair counts")
    if not 1 <= rescore_top <= 1 << 16 or not 1 <= batch_size <= 1 << 16:
        raise ValueError("invalid ranking width or batch size")
    started = time.perf_counter()
    pairs = ciphertext_pairs[:actual_full_pairs]
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
    screen_collision_count = int(np.sum(np.diff(np.sort(screen_scores)) == 0))
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
    full_collision_count = int(np.sum(np.diff(np.sort(full_scores)) == 0))
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
        screen_collision_count=screen_collision_count,
        full_collision_count=full_collision_count,
        screen_pairs_used=screen_pairs,
        full_pairs_used=actual_full_pairs,
    )


def rank_stage1_ondemand(
    pairs: Sequence[tuple[Block, Block]],
    tool_path: str | Path = "/private/tmp/simon_queried_row_8r",
    ddt_7r_path: str | Path = "/private/tmp/simon_predicted_ddt_7r.bin",
    *,
    screen_pairs: int = 16,
    full_pairs: int | None = None,
    rescore_top: int = 512,
    batch_size: int = 1024,
) -> tuple[DomainRanking, dict[str, float]]:
    """Score 8R stage 1 candidates using on-demand double-precision queried rows from 7R table."""
    actual_full_pairs = len(pairs) if full_pairs is None else full_pairs
    if not 1 <= screen_pairs <= actual_full_pairs <= len(pairs):
        raise ValueError(
            f"Invalid pair counts: screen_pairs={screen_pairs}, "
            f"full_pairs={actual_full_pairs}, total_pairs={len(pairs)}"
        )
    t_start = time.perf_counter()
    tool_path = Path(tool_path)
    ddt_7r_path = Path(ddt_7r_path)

    # 1. Compute candidate-independent left difference words
    t0 = time.perf_counter()
    anchor = candidate_difference_words(
        candidate_transformed_words(pairs, [0], final_key_candidate=None)
    )[0, :, 0]
    unique_rows = sorted(set(int(x) for x in anchor))
    anchor_seconds = time.perf_counter() - t0

    # 2. Run C++ on-demand queried row engine
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="simon_queried_rows_") as temporary_dir:
        out_bin = Path(temporary_dir) / "rows.bin"
        cmd = [str(tool_path), str(ddt_7r_path), str(out_bin)] + [str(r) for r in unique_rows]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        tool_wall_seconds = time.perf_counter() - t0
        tool_metrics = json.loads(proc.stdout)

        # 3. Load queried rows; the temporary output is removed even if parsing fails.
        t0 = time.perf_counter()
        matrix = np.fromfile(out_bin, dtype=np.float64).reshape((len(unique_rows), 65536))
    row_map = {r: matrix[idx] for idx, r in enumerate(unique_rows)}

    log_rows = np.stack([np.log2(row_map[int(left)] + 1e-64) for left in anchor])
    row_load_seconds = time.perf_counter() - t0

    # 4. Screen all 65,536 candidates on screen_pairs
    t0 = time.perf_counter()
    screen_p = pairs[:screen_pairs]
    screen_log_rows = log_rows[:screen_pairs]
    screen_scores = np.empty(1 << 16, dtype=np.float64)

    for start in range(0, 1 << 16, batch_size):
        candidates = list(range(start, min(start + batch_size, 1 << 16)))
        trans = candidate_transformed_words(screen_p, candidates, final_key_candidate=None)
        diffs = candidate_difference_words(trans)
        right = diffs[:, :, 1].astype(np.intp, copy=False)
        scores = screen_log_rows[np.arange(screen_pairs)[None, :], right].sum(axis=1, dtype=np.float64)
        screen_scores[start : start + len(candidates)] = scores
    screen_seconds = time.perf_counter() - t0

    # 5. Rescore top rescore_top on full_pairs with deterministic tie breaking
    t0 = time.perf_counter()
    candidates_all = np.arange(1 << 16, dtype=np.uint16)
    screen_order = np.lexsort((candidates_all, -screen_scores))
    top_cands = candidates_all[screen_order[:rescore_top]]

    full_p = pairs[:actual_full_pairs]
    trans_rescore = candidate_transformed_words(full_p, top_cands, final_key_candidate=None)
    diffs_rescore = candidate_difference_words(trans_rescore)
    right_rescore = diffs_rescore[:, :, 1].astype(np.intp, copy=False)
    rescore_scores = log_rows[np.arange(actual_full_pairs)[None, :], right_rescore].sum(axis=1, dtype=np.float64)
    rescore_seconds = time.perf_counter() - t0

    # 6. Final sort
    t0 = time.perf_counter()
    rescore_order = np.lexsort((top_cands, -rescore_scores))
    final_order = top_cands[rescore_order]
    sort_seconds = time.perf_counter() - t0

    total_seconds = time.perf_counter() - t_start

    screen_tie_count = int(np.count_nonzero(screen_scores == screen_scores[screen_order[0]]))
    screen_collision_count = int(np.sum(np.diff(np.sort(screen_scores)) == 0))
    full_tie_count = int(np.count_nonzero(rescore_scores == rescore_scores[rescore_order[0]]))
    full_collision_count = int(np.sum(np.diff(np.sort(rescore_scores)) == 0))

    ranking = DomainRanking(
        order=final_order,
        screen_seconds=screen_seconds,
        row_load_seconds=row_load_seconds + tool_wall_seconds,
        rescore_seconds=rescore_seconds,
        sort_seconds=sort_seconds,
        total_seconds=total_seconds,
        unique_left_rows=len(unique_rows),
        screen_tie_count=screen_tie_count,
        full_tie_count=full_tie_count,
        screen_collision_count=screen_collision_count,
        full_collision_count=full_collision_count,
        screen_pairs_used=screen_pairs,
        full_pairs_used=actual_full_pairs,
    )

    diagnostics = {
        "anchor_seconds": anchor_seconds,
        "tool_wall_seconds": tool_wall_seconds,
        "nvme_read_seconds": tool_metrics.get("read_seconds", 0.0),
        "butterfly_arithmetic_seconds": tool_metrics.get("arithmetic_seconds", 0.0),
        "tool_peak_rss_gib": tool_metrics.get("peak_rss_gib", 0.0),
        "row_load_seconds": row_load_seconds,
        "screen_seconds": screen_seconds,
        "rescore_seconds": rescore_seconds,
        "sort_seconds": sort_seconds,
        "total_stage1_seconds": total_seconds,
        "screen_pairs_used": screen_pairs,
        "full_pairs_used": actual_full_pairs,
    }

    return ranking, diagnostics
