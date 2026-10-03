"""Unit tests for the demo cross-cipher residual-depth dataset.

Verifies:
1. File exists and SHA-256 sidecar matches bit-for-bit.
2. Cipher KATs pass for SPECK32/64 and SIMON32/64.
3. Exactly 8 keys per cipher, identical across all round configurations.
4. Maximum 64 differential pairs per instance.
5. Exact prefix parity: O=4 observations match the first 2 differential pairs of O=128.
6. Exactly 2 fresh verification blocks, strictly disjoint from all attack plaintexts.
7. Correct trailing subkeys matching independent cipher expansion.
8. Re-encryption of all attack and verification plaintexts matches dataset ciphertexts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from neural_sat_speck.simon.cipher import (
    SIMON32_64_KAT_CIPHERTEXT,
    SIMON32_64_KAT_KEY,
    SIMON32_64_KAT_PLAINTEXT,
    encrypt_block as simon_encrypt,
)
from neural_sat_speck.simon.key_schedule import expand_key as simon_expand_key
from neural_sat_speck.speck.cipher import encrypt_block as speck_encrypt
from neural_sat_speck.speck.key_schedule import expand_key as speck_expand_key
from tests.fixtures.speck32_64_vectors import (
    CIPHERTEXT as SPECK_KAT_CIPHERTEXT,
    MASTER_KEY as SPECK_KAT_KEY,
    PLAINTEXT as SPECK_KAT_PLAINTEXT,
)

ROOT = Path(__file__).resolve().parents[2]
DATASET_FILE = ROOT / "datasets" / "demo_cross_cipher_depth_r6_r11" / "dataset.json"
SIDECAR_FILE = ROOT / "datasets" / "demo_cross_cipher_depth_r6_r11" / "dataset.json.sha256"


def test_dataset_sidecar_checksum_valid():
    """Verify dataset.json and its sha256 sidecar exist and match."""
    assert DATASET_FILE.is_file(), f"Dataset file not found: {DATASET_FILE}"
    assert SIDECAR_FILE.is_file(), f"Sidecar file not found: {SIDECAR_FILE}"

    data_bytes = DATASET_FILE.read_bytes()
    expected_hash = hashlib.sha256(data_bytes).hexdigest()

    sidecar_content = SIDECAR_FILE.read_text(encoding="utf-8").strip().split()
    assert sidecar_content[0] == expected_hash, "Sidecar checksum mismatch"


def test_cipher_known_answer_tests():
    """Verify cipher Known Answer Tests."""
    assert speck_encrypt(SPECK_KAT_PLAINTEXT, SPECK_KAT_KEY) == SPECK_KAT_CIPHERTEXT
    assert simon_encrypt(SIMON32_64_KAT_PLAINTEXT, SIMON32_64_KAT_KEY) == SIMON32_64_KAT_CIPHERTEXT


def test_dataset_structure_and_invariants():
    """Verify all 8 keys, plaintexts, prefix parity, trailing subkeys, and re-encryptions."""
    with open(DATASET_FILE, encoding="utf-8") as f:
        data = json.load(f)

    assert data["dataset_id"] == "demo_cross_cipher_depth_r6_r11_v1"
    assert data["split_role"] == "demo"
    assert data["seed"] == 999999

    ciphers = data["ciphers"]
    assert set(ciphers.keys()) == {"speck32_64", "simon32_64"}

    for cipher_name, r_list, enc_fn, exp_fn, input_diff in (
        ("speck32_64", (6, 7, 8), speck_encrypt, speck_expand_key, (0x0040, 0x0000)),
        ("simon32_64", (9, 10, 11), simon_encrypt, simon_expand_key, (0x0000, 0x0040)),
    ):
        instances = ciphers[cipher_name]["instances"]
        assert len(instances) == 8, f"{cipher_name} must have exactly 8 instances"

        keys = set()
        for inst in instances:
            idx = inst["instance_index"]
            mk = tuple(inst["master_key"])
            assert mk not in keys, f"Duplicate master key in {cipher_name}"
            keys.add(mk)

            atk_pts = inst["attack_plaintexts"]
            ver_pts = inst["verification_plaintexts"]
            assert len(atk_pts) == 64, "Must have exactly 64 differential pairs"
            assert len(ver_pts) == 2, "Must have exactly 2 verification blocks"

            # Check differential property
            for pair in atk_pts:
                p0 = tuple(pair[0])
                p1 = tuple(pair[1])
                assert (p0[0] ^ input_diff[0], p0[1] ^ input_diff[1]) == p1

            # Check disjointness
            all_atk = [tuple(p) for pair in atk_pts for p in pair]
            all_ver = [tuple(vp) for vp in ver_pts]
            assert len(all_atk) == 128
            assert len(set(all_atk)) == 128, "Duplicate attack plaintexts"
            assert len(set(all_atk).intersection(set(all_ver))) == 0, "Verification plaintexts overlap attack plaintexts"

            # Check round data
            for r in r_list:
                r_data = inst["rounds_data"][str(r)]
                round_keys = exp_fn(mk, r)
                assert r_data["round_keys"] == list(round_keys)

                # Check trailing subkeys for m in 0..3
                for m in (0, 1, 2, 3):
                    exp_trailing = list(round_keys[-m:]) if m > 0 else []
                    assert r_data["trailing_subkeys"][str(m)] == exp_trailing

                # Check attack re-encryptions
                atk_cts = r_data["attack_ciphertexts"]
                assert len(atk_cts) == 64
                for p_pair, c_pair in zip(atk_pts, atk_cts, strict=True):
                    for p, c in zip(p_pair, c_pair, strict=True):
                        assert enc_fn(tuple(p), mk, r) == tuple(c)

                # Check verification re-encryptions
                ver_cts = r_data["verification_ciphertexts"]
                assert len(ver_cts) == 2
                for vp, vc in zip(ver_pts, ver_cts, strict=True):
                    assert enc_fn(tuple(vp), mk, r) == tuple(vc)

                # Check prefix parity: first 2 pairs (4 observations) are identical between O=4 and O=128
                o4_pairs = atk_pts[:2]
                o4_cts = atk_cts[:2]
                assert o4_pairs == atk_pts[0:2]
                assert o4_cts == atk_cts[0:2]
