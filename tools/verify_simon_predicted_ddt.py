"""Independent checks of complete sparse-6R and dense-7R SIMON Markov tables.

This verifier reimplements Bao's scalar one-round diff_prob conditions. It
streams every stored mass and checks public, fixed-index seven-round entries
against a direct 6R-to-7R source-style sum. No attack data or truth labels enter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
from pathlib import Path

import numpy as np

DENOMINATOR = 1 << 60
SPARSE_RECORD = np.dtype([("index", "<u4"), ("mass", "<u8")])
SPARSE_CHUNK = 65536
DENSE_CHUNK_BYTES = 8 * 1024 * 1024


def _header(path: Path, expected_magic: bytes, expected_rounds: int) -> tuple[int, int, int]:
    with path.open("rb") as handle:
        raw = handle.read(32)
    if len(raw) != 32 or raw[:8] != expected_magic:
        raise ValueError(f"invalid {expected_rounds}R file magic/header: {path}")
    bits, rounds, initial, exponent, count = struct.unpack("<IIIIQ", raw[8:])
    if bits != 16 or rounds != expected_rounds or initial != 0x00000040 or exponent != 60:
        raise ValueError(f"unexpected {expected_rounds}R table metadata: {path}")
    expected_bytes = 32 + count * (12 if expected_rounds == 6 else 8)
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"{expected_rounds}R table length mismatch: {path}")
    if expected_rounds == 7 and count != 1 << 32:
        raise ValueError("7R table must cover all 2^32 indices")
    return bits, initial, count


def _source_exponent(a: int, b: int, c: int) -> int | None:
    """Return k for Bao diff_prob((a,b),(c,a)) == 2^-k, or None for zero."""

    def rol(value: int, shift: int) -> int:
        return ((value << shift) | (value >> (16 - shift))) & 0xFFFF

    gamma = rol(a, 2) ^ b ^ c
    if a == 0xFFFF:
        return 15 if gamma.bit_count() % 2 == 0 else None
    rola, rolb = rol(a, 8), rol(a, 1)
    variable_bits = rola | rolb
    if gamma & (variable_bits ^ 0xFFFF):
        return None
    double_bits = rol(a, 15) & (rola ^ 0xFFFF) & rolb
    if (gamma ^ rol(gamma, 7)) & double_bits:
        return None
    return (variable_bits ^ double_bits).bit_count()


def _sample_words() -> list[int]:
    rng = random.Random(0x53494D4F4E444454)
    return sorted({0, 1, 0x40, 0x80, 0x1234, 0x8000, 0xFFFF, *rng.sample(range(1 << 16), 32)})


def verify(sparse: Path, dense: Path) -> dict[str, int | str]:
    _header(sparse, b"SMDT0001", 6)
    _header(dense, b"SMDT0002", 7)
    sample_a, sample_c = _sample_words(), _sample_words()
    samples: dict[int, list[tuple[int, int]]] = {a: [] for a in sample_a}
    sparse_hash = hashlib.sha256()
    sparse_mass = sparse_support = previous_index = 0
    with sparse.open("rb") as handle:
        sparse_hash.update(handle.read(32))
        while raw := handle.read(SPARSE_RECORD.itemsize * SPARSE_CHUNK):
            if len(raw) % SPARSE_RECORD.itemsize:
                raise ValueError("partial sparse record")
            sparse_hash.update(raw)
            records = np.frombuffer(raw, dtype=SPARSE_RECORD)
            indices = records["index"]
            masses = records["mass"]
            if len(indices) == 0 or int(indices[0]) <= previous_index:
                raise ValueError("sparse indices are duplicate or unordered")
            if np.any(indices[1:] <= indices[:-1]) or np.any(masses == 0):
                raise ValueError("sparse index order or positive-mass invariant failed")
            maximum = int(masses.max())
            if maximum > DENOMINATOR or maximum * len(masses) > (1 << 64) - 1:
                raise ValueError("sparse mass exceeds safe uint64 chunk-sum bound")
            sparse_mass += int(masses.sum(dtype=np.uint64))
            sparse_support += len(records)
            previous_index = int(indices[-1])
            for a in sample_a:
                start = int(np.searchsorted(indices, a << 16, side="left"))
                end = int(np.searchsorted(indices, (a + 1) << 16, side="left"))
                samples[a].extend((int(index & 0xFFFF), int(mass)) for index, mass in records[start:end])
    if sparse_mass != DENOMINATOR:
        raise ValueError("sparse total mass is not one")

    dense_hash = hashlib.sha256()
    dense_mass = dense_support = 0
    with dense.open("rb") as handle:
        dense_hash.update(handle.read(32))
        while raw := handle.read(DENSE_CHUNK_BYTES):
            if len(raw) % 8:
                raise ValueError("partial dense record")
            dense_hash.update(raw)
            values = np.frombuffer(raw, dtype="<u8")
            maximum = int(values.max())
            if maximum > DENOMINATOR or maximum * len(values) > (1 << 64) - 1:
                raise ValueError("dense mass exceeds safe uint64 chunk-sum bound")
            dense_mass += int(values.sum(dtype=np.uint64))
            dense_support += int(np.count_nonzero(values))
    if dense_mass != DENOMINATOR:
        raise ValueError("dense total mass is not one")

    checked = positive = zero = 0
    with dense.open("rb") as handle:
        for a, row in samples.items():
            for c in sample_c:
                predicted = 0
                for b, mass in row:
                    exponent = _source_exponent(a, b, c)
                    if exponent is None:
                        continue
                    if mass & ((1 << exponent) - 1):
                        raise ValueError("source transition cannot use 2^60 denominator")
                    predicted += mass >> exponent
                handle.seek(32 + (((c << 16) | a) * 8))
                actual = struct.unpack("<Q", handle.read(8))[0]
                if actual != predicted:
                    raise ValueError(f"source transition mismatch at (c={c:#06x}, a={a:#06x})")
                checked += 1
                positive += actual > 0
                zero += actual == 0
    if not positive or not zero:
        raise ValueError("spot checks did not cover both positive and zero bins")
    return {
        "sparse6_sha256": sparse_hash.hexdigest(),
        "sparse6_support": sparse_support,
        "dense7_sha256": dense_hash.hexdigest(),
        "dense7_support": dense_support,
        "denominator_exponent": 60,
        "independent_source_spot_checks": checked,
        "positive_spot_checks": positive,
        "zero_spot_checks": zero,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sparse6", type=Path)
    parser.add_argument("dense7", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.sparse6, args.dense7), sort_keys=True))


if __name__ == "__main__":
    main()
