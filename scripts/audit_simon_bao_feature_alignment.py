"""Compare local candidate features with a pinned checkout of Bao's source.

Usage: python scripts/audit_simon_bao_feature_alignment.py --bao-repo PATH
This is a source-equivalence check on synthetic ciphertext words, not an attack.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np

from neural_sat_speck.simon.ranking_features import (
    bao_full_value_bits,
    bao_value_difference_bits,
    candidate_difference_words,
    candidate_transformed_words,
    candidate_value_words,
)

BAO_COMMIT = "09cc81320be9ba65eb4b5c932ac4a0b85d5e4bc8"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bao-repo", type=Path, required=True)
    args = parser.parse_args()
    repo = args.bao_repo.resolve()
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if commit != BAO_COMMIT:
        parser.error(f"Bao checkout must be pinned to {BAO_COMMIT}, got {commit}")
    path = repo / "simonNDvsDD" / "simon.py"
    spec = importlib.util.spec_from_file_location("bao_simon_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    bao = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bao)

    rng = np.random.default_rng(260926)
    checked = 0
    for _ in range(8):
        words = rng.integers(0, 1 << 16, size=(3, 2, 2), dtype=np.uint16)
        pairs = tuple(
            (
                (int(row[0, 0]), int(row[0, 1])),
                (int(row[1, 0]), int(row[1, 1])),
            )
            for row in words
        )
        for final_candidate in (None, 0x3A55):
            for candidate in (0, 1, 0x1234, 0xFFFF):
                ours_vv = candidate_value_words(
                    pairs, (candidate,), final_key_candidate=final_candidate
                )
                ours_vd = candidate_transformed_words(
                    pairs, (candidate,), final_key_candidate=final_candidate
                )
                ours_diff = candidate_difference_words(ours_vd)
                for pair_index, (cipher_a, cipher_b) in enumerate(pairs):
                    a, b = cipher_a, cipher_b
                    if final_candidate is not None:
                        a = bao.dec_one_round(a, final_candidate)
                        b = bao.dec_one_round(b, final_candidate)
                    a = bao.dec_one_round(a, candidate)
                    b = bao.dec_one_round(b, candidate)
                    source_vv = np.array((*a, *b), dtype=np.uint16)
                    np.testing.assert_array_equal(ours_vv[0, pair_index], source_vv)
                    source_vv_bits = bao.convert_to_binary(
                        [np.array([word], dtype=np.uint16) for word in source_vv]
                    )
                    np.testing.assert_array_equal(
                        bao_full_value_bits(ours_vv)[0, pair_index].reshape(1, 64),
                        source_vv_bits,
                    )

                    a = bao.dec_one_round(a, 0)
                    b = bao.dec_one_round(b, 0)
                    source_vd = np.array((*a, *b), dtype=np.uint16)
                    np.testing.assert_array_equal(ours_vd[0, pair_index], source_vd)
                    source_vd_bits = bao.convert_to_binary(
                        [
                            np.array([a[0]], dtype=np.uint16),
                            np.array([b[0]], dtype=np.uint16),
                            np.array([a[1] ^ b[1]], dtype=np.uint16),
                        ]
                    )
                    np.testing.assert_array_equal(
                        bao_value_difference_bits(ours_vd)[0, pair_index].reshape(1, 48),
                        source_vd_bits,
                    )
                    np.testing.assert_array_equal(
                        ours_diff[0, pair_index],
                        np.array((a[0] ^ b[0], a[1] ^ b[1]), dtype=np.uint16),
                    )
                    checked += 1
    print(json.dumps({"bao_commit": commit, "checked_candidate_pairs": checked}))


if __name__ == "__main__":
    main()
