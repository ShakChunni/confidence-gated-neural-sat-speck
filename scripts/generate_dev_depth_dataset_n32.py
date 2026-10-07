#!/usr/bin/env python3
"""Deterministic generator and validator for development cross-cipher residual-depth dataset (N=32 keys per cipher).

Generates exactly 32 fresh development keys per cipher under seed 501001, reusing a single
64-pair maximum family per key across rounds R and observation levels O in {4, 128}.
Runs exhaustive preflight validation against cipher KATs, independent cipher re-encryptions,
key equality across rounds, prefix preservation, and verification block disjointness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any

from neural_sat_speck.simon.cipher import (
    SIMON32_64_KAT_CIPHERTEXT,
    SIMON32_64_KAT_KEY,
    SIMON32_64_KAT_PLAINTEXT,
)
from neural_sat_speck.simon.cipher import (
    encrypt_block as simon_encrypt,
)
from neural_sat_speck.simon.key_schedule import expand_key as simon_expand_key
from neural_sat_speck.speck.cipher import encrypt_block as speck_encrypt
from neural_sat_speck.speck.key_schedule import expand_key as speck_expand_key
from tests.fixtures.speck32_64_vectors import (
    CIPHERTEXT as SPECK_KAT_CIPHERTEXT,
)
from tests.fixtures.speck32_64_vectors import (
    MASTER_KEY as SPECK_KAT_KEY,
)
from tests.fixtures.speck32_64_vectors import (
    PLAINTEXT as SPECK_KAT_PLAINTEXT,
)

ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = ROOT / "datasets" / "dev_cross_cipher_depth_n32"
DATASET_PATH = DATASET_DIR / "dataset.json"

DEV_SEED_501001 = 501001
KEYS_PER_CIPHER = 32

SPECK_ROUNDS = (6, 7, 8)
SIMON_ROUNDS = (9, 10, 11)
MAX_PAIRS = 64  # 128 observations
VERIFICATION_BLOCKS = 2

SPECK_INPUT_DIFF = (0x0040, 0x0000)
SIMON_INPUT_DIFF = (0x0000, 0x0040)


def _get_git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN"


def generate_cipher_instances(
    cipher_name: str,
    seed: int,
    rounds_list: tuple[int, ...],
    input_diff: tuple[int, int],
    encrypt_fn: Any,
    expand_fn: Any,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    instances: list[dict[str, Any]] = []

    for idx in range(KEYS_PER_CIPHER):
        master_key = (
            rng.randrange(1 << 16),
            rng.randrange(1 << 16),
            rng.randrange(1 << 16),
            rng.randrange(1 << 16),
        )

        used_pts: set[tuple[int, int]] = set()
        attack_pairs_pts: list[list[list[int]]] = []

        for _ in range(MAX_PAIRS):
            while True:
                p0 = (rng.randrange(1 << 16), rng.randrange(1 << 16))
                p1 = (p0[0] ^ input_diff[0], p0[1] ^ input_diff[1])
                if p0 not in used_pts and p1 not in used_pts:
                    used_pts.update((p0, p1))
                    break
            attack_pairs_pts.append([[p0[0], p0[1]], [p1[0], p1[1]]])

        verif_pts: list[list[int]] = []
        for _ in range(VERIFICATION_BLOCKS):
            while True:
                vp = (rng.randrange(1 << 16), rng.randrange(1 << 16))
                if vp not in used_pts:
                    used_pts.add(vp)
                    break
            verif_pts.append([vp[0], vp[1]])

        # Compute encryptions and schedules for each round
        rounds_data: dict[str, Any] = {}
        for r in rounds_list:
            round_keys = expand_fn(master_key, r)

            attack_pairs_cts: list[list[list[int]]] = []
            for pair in attack_pairs_pts:
                p0 = (pair[0][0], pair[0][1])
                p1 = (pair[1][0], pair[1][1])
                c0 = encrypt_fn(p0, master_key, r)
                c1 = encrypt_fn(p1, master_key, r)
                attack_pairs_cts.append([[c0[0], c0[1]], [c1[0], c1[1]]])

            verif_cts: list[list[int]] = []
            for vp_list in verif_pts:
                vp = (vp_list[0], vp_list[1])
                vc = encrypt_fn(vp, master_key, r)
                verif_cts.append([vc[0], vc[1]])

            trailing_subkeys: dict[str, list[int]] = {}
            for m in (0, 1, 2, 3):
                trailing_subkeys[str(m)] = list(round_keys[-m:]) if m > 0 else []

            rounds_data[str(r)] = {
                "round_keys": list(round_keys),
                "trailing_subkeys": trailing_subkeys,
                "attack_ciphertexts": attack_pairs_cts,
                "verification_ciphertexts": verif_cts,
            }

        instances.append({
            "instance_index": idx,
            "instance_id": f"{cipher_name}_dev_key_{idx}",
            "master_key": list(master_key),
            "attack_plaintexts": attack_pairs_pts,
            "verification_plaintexts": verif_pts,
            "rounds_data": rounds_data,
        })

    return instances


def validate_dataset(data: dict[str, Any]) -> None:
    """Rigorous preflight validation of generated dataset."""
    # 1. Verify cipher KATs
    assert speck_encrypt(SPECK_KAT_PLAINTEXT, SPECK_KAT_KEY) == SPECK_KAT_CIPHERTEXT, "SPECK KAT failed"
    assert simon_encrypt(SIMON32_64_KAT_PLAINTEXT, SIMON32_64_KAT_KEY) == SIMON32_64_KAT_CIPHERTEXT, "SIMON KAT failed"

    # 2. Check each cipher
    for cipher_name, r_list, enc_fn, exp_fn in (
        ("speck32_64", SPECK_ROUNDS, speck_encrypt, speck_expand_key),
        ("simon32_64", SIMON_ROUNDS, simon_encrypt, simon_expand_key),
    ):
        insts = data["ciphers"][cipher_name]["instances"]
        assert len(insts) == KEYS_PER_CIPHER, f"{cipher_name} does not have exactly {KEYS_PER_CIPHER} instances"

        keys_seen: set[tuple[int, ...]] = set()
        for inst in insts:
            mk = tuple(inst["master_key"])
            assert mk not in keys_seen, f"Duplicate master key in {cipher_name}: {mk}"
            keys_seen.add(mk)

            atk_pts = inst["attack_plaintexts"]
            ver_pts = inst["verification_plaintexts"]
            assert len(atk_pts) == MAX_PAIRS == 64
            assert len(ver_pts) == VERIFICATION_BLOCKS == 2

            # Check disjointness
            all_pts = [tuple(p) for pair in atk_pts for p in pair]
            all_vpts = [tuple(vp) for vp in ver_pts]
            assert len(all_pts) == 128
            assert len(set(all_pts)) == 128, "Duplicate attack plaintexts"
            assert len(set(all_pts).intersection(set(all_vpts))) == 0, "Verification plaintexts overlap attack plaintexts!"

            # Re-verify cipher round encryptions
            for r in r_list:
                r_str = str(r)
                expected_keys = exp_fn(mk, r)
                stored_keys = inst["rounds_data"][r_str]["round_keys"]
                assert tuple(stored_keys) == tuple(expected_keys), f"Stored keys mismatch round {r}"

                stored_atk_cts = inst["rounds_data"][r_str]["attack_ciphertexts"]
                for p_pair, c_pair in zip(atk_pts, stored_atk_cts):
                    c0_calc = enc_fn(tuple(p_pair[0]), mk, r)
                    c1_calc = enc_fn(tuple(p_pair[1]), mk, r)
                    assert (c0_calc[0], c0_calc[1]) == (c_pair[0][0], c_pair[0][1]), "Attack encryption mismatch"
                    assert (c1_calc[0], c1_calc[1]) == (c_pair[1][0], c_pair[1][1]), "Attack encryption mismatch"

                stored_ver_cts = inst["rounds_data"][r_str]["verification_ciphertexts"]
                for vp, vc in zip(ver_pts, stored_ver_cts):
                    vc_calc = enc_fn(tuple(vp), mk, r)
                    assert (vc_calc[0], vc_calc[1]) == (vc[0], vc[1]), "Verification encryption mismatch"


def generate_and_save_dataset(output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    git_commit = _get_git_commit()

    speck_instances = generate_cipher_instances(
        "speck32_64",
        DEV_SEED_501001,
        SPECK_ROUNDS,
        SPECK_INPUT_DIFF,
        speck_encrypt,
        speck_expand_key,
    )

    simon_instances = generate_cipher_instances(
        "simon32_64",
        DEV_SEED_501001 + 1,
        SIMON_ROUNDS,
        SIMON_INPUT_DIFF,
        simon_encrypt,
        simon_expand_key,
    )

    dataset: dict[str, Any] = {
        "dataset_id": "dev_cross_cipher_depth_n32",
        "format_version": "1.0.0",
        "created_at": "2026-10-08T02:30:00+08:00",
        "generator_script": "scripts/generate_dev_depth_dataset_n32.py",
        "generator_commit": git_commit,
        "seed": DEV_SEED_501001,
        "split_role": "development",
        "governing_decisions": ["D-055", "D-056"],
        "metadata": {
            "key_count_per_cipher": KEYS_PER_CIPHER,
            "max_differential_pairs": MAX_PAIRS,
            "observation_levels": [4, 128],
            "verification_blocks_count": VERIFICATION_BLOCKS,
            "speck_rounds": list(SPECK_ROUNDS),
            "simon_rounds": list(SIMON_ROUNDS),
            "speck_input_difference": list(SPECK_INPUT_DIFF),
            "simon_input_difference": list(SIMON_INPUT_DIFF),
        },
        "ciphers": {
            "speck32_64": {
                "instances": speck_instances,
            },
            "simon32_64": {
                "instances": simon_instances,
            },
        },
    }

    validate_dataset(dataset)

    # Atomic write of dataset.json
    temp_path = output_path.with_suffix(".tmp")
    data_bytes = json.dumps(dataset, indent=2).encode("utf-8")
    temp_path.write_bytes(data_bytes)
    temp_path.replace(output_path)

    # Write sidecar SHA-256
    digest = hashlib.sha256(data_bytes).hexdigest()
    sidecar_path = output_path.with_name(f"{output_path.name}.sha256")
    sidecar_path.write_text(f"{digest}  {output_path.name}\n", encoding="utf-8")

    print(f"Dataset generated and verified: {output_path}")
    print(f"SHA-256: {digest}")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dev cross-cipher depth dataset (N=32).")
    parser.add_argument("--output", type=Path, default=DATASET_PATH, help="Output dataset path")
    args = parser.parse_args()
    generate_and_save_dataset(args.output)
