"""Compare real-table vectorized candidate scores with scalar source-order reads.

All ciphertext words and candidate words are fixed public synthetic values.
No key, truth label, verification pair, or attack instance is loaded.
"""

from __future__ import annotations

import argparse
import math
import random
import struct
from pathlib import Path

import numpy as np

from neural_sat_speck.simon.classical_ranking import (
    DENOMINATOR,
    FullPredictedDdt,
    _score_candidates,
)
from neural_sat_speck.simon.ranking_features import (
    candidate_difference_words,
    candidate_transformed_words,
)


def _f(value: int) -> int:
    def rol(shift: int) -> int:
        return ((value << shift) | (value >> (16 - shift))) & 0xFFFF

    return (rol(1) & rol(8)) ^ rol(2)


def _inverse(block: tuple[int, int], key: int) -> tuple[int, int]:
    left, right = block
    return right, left ^ _f(right) ^ key


def _pairs() -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    rng = random.Random(260926)
    return tuple(
        (
            (rng.randrange(1 << 16), rng.randrange(1 << 16)),
            (rng.randrange(1 << 16), rng.randrange(1 << 16)),
        )
        for _ in range(32)
    )


def _read_dense(handle: object, index: int) -> int:
    handle.seek(32 + index * 8)
    return struct.unpack("<Q", handle.read(8))[0]


def _read_sparse(handle: object, count: int, index: int) -> int:
    lower, upper = 0, count
    while lower < upper:
        middle = (lower + upper) // 2
        handle.seek(32 + middle * 12)
        found = struct.unpack("<I", handle.read(4))[0]
        if found < index:
            lower = middle + 1
        else:
            upper = middle
    if lower == count:
        return 0
    handle.seek(32 + lower * 12)
    found, mass = struct.unpack("<IQ", handle.read(12))
    return mass if found == index else 0


def _check_stage(
    path: Path, rounds: int, final_key_candidate: int | None
) -> int:
    table = FullPredictedDdt(path, rounds)
    pairs = _pairs()
    candidates = (0, 1, 0x1234, 0xFFFF)
    anchor = candidate_difference_words(
        candidate_transformed_words(pairs, [0], final_key_candidate=final_key_candidate)
    )[0, :, 0]
    rows = np.stack([table.log_score_row(int(left)) for left in anchor])
    actual = _score_candidates(
        pairs, candidates, rows, anchor, final_key_candidate=final_key_candidate
    )
    with path.open("rb") as handle:
        for candidate_index, candidate in enumerate(candidates):
            expected = 0.0
            for pair_index, pair in enumerate(pairs):
                states = []
                for block in pair:
                    if final_key_candidate is not None:
                        block = _inverse(block, final_key_candidate)
                    states.append(_inverse(_inverse(block, candidate), 0))
                left = states[0][0] ^ states[1][0]
                right = states[0][1] ^ states[1][1]
                assert left == int(anchor[pair_index])
                index = (left << 16) | right
                mass = (
                    _read_dense(handle, index)
                    if rounds == 7
                    else _read_sparse(handle, table.count, index)
                )
                expected += math.log2(mass / DENOMINATOR + 1e-64)
            if not math.isclose(float(actual[candidate_index]), expected, abs_tol=1e-10):
                raise AssertionError("scalar and vectorized classical candidate scores disagree")
    return len(candidates) * len(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sparse6", type=Path)
    parser.add_argument("dense7", type=Path)
    args = parser.parse_args()
    stage1 = _check_stage(args.dense7, 7, None)
    stage2 = _check_stage(args.sparse6, 6, 0x1234)
    print(f"truth-blind scalar-source lookup agreement: {stage1} stage-1 and {stage2} stage-2 candidate-pair scores")


if __name__ == "__main__":
    main()
