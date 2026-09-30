#!/usr/bin/env python3
"""Check opened training key replay against retained full dataset checksums.

Vectorized independent SIMON encryption avoids a million scalar generator calls.
No training, ranking, solver or new seed is used.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from neural_sat_speck.simon.b1_data import compute_content_checksum  # noqa: E402


def rotate(x, n):
    return (x << n) | (x >> (16 - n))


def reconstruct(count: int, seed: int, group_size: int | None):
    rng = np.random.default_rng(seed)
    keys = rng.integers(0, 65536, size=(count, 4), dtype=np.uint16)
    shape = (count, 2) if group_size is None else (count, group_size, 2)
    p0 = rng.integers(0, 65536, size=shape, dtype=np.uint16)
    positive = (count + 1) // 2
    labels = np.zeros(count, dtype=np.uint8)
    labels[:positive] = 1
    p1 = p0 ^ np.array([0, 64], dtype=np.uint16)
    p1[positive:] = rng.integers(0, 65536, size=p1[positive:].shape, dtype=np.uint16)
    permutation = rng.permutation(count)
    keys, labels, p0, p1 = keys[permutation], labels[permutation], p0[permutation], p1[permutation]
    round_keys = [keys[:, col] for col in (3, 2, 1, 0)]
    z0 = (1, 1, 1, 1, 1)  # only the five expansion steps needed at R9
    for i in range(5):
        last = round_keys[i + 3]
        tmp = ((last >> 3) | (last << 13)) ^ round_keys[i + 1]
        tmp ^= (tmp >> 1) | (tmp << 15)
        round_keys.append(round_keys[i] ^ tmp ^ np.uint16(0xFFFC ^ z0[i]))
    def encrypt(pts):
        left, right = pts[..., 0].copy(), pts[..., 1].copy()
        for key in round_keys:
            expanded = key if group_size is None else key[:, None]
            left, right = right ^ (rotate(left, 1) & rotate(left, 8)) ^ rotate(left, 2) ^ expanded, left
        return left, right
    l0, r0 = encrypt(p0)
    l1, r1 = encrypt(p1)
    words = np.stack([l0, r0, l1, r1], axis=-1)
    features = ((words[..., None] >> np.arange(15, -1, -1, dtype=np.uint16)) & 1).astype(np.uint8)
    return features, labels, keys


def main():
    metadata = json.loads((ROOT / "results/simon_training_log.json").read_text())["metadata"]
    evidence = {"numpy_version": np.__version__, "roles": {}, "all_content_checksums_match": True}
    for role, seed, count, group_size, field in (
        ("b0_train", 880001, 1_000_000, None, "b0_train_dataset_checksum"),
        ("b0_validation", 880002, 99_968, None, "b0_validation_dataset_checksum"),
        ("b1_train", 880001, 15_625, 64, "train_dataset_checksum"),
        ("b1_validation", 880002, 1_562, 64, "validation_dataset_checksum"),
    ):
        features, labels, keys = reconstruct(count, seed, group_size)
        checksum = compute_content_checksum(features, labels, keys)
        if checksum != metadata[field]:
            raise RuntimeError(f"Retained {role} checksum mismatch: {checksum} != {metadata[field]}")
        evidence["roles"][role] = {"seed": seed, "key_count": count, "content_sha256": checksum,
                                   "key_bytes_sha256": hashlib.sha256(keys.tobytes()).hexdigest()}
        print(f"{role}: retained full content checksum matches", flush=True)
    destination = ROOT / "results/simon_training_key_replay_audit.json"
    destination.write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"Evidence: {destination}")


if __name__ == "__main__":
    main()
