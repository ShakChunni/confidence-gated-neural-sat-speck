"""Native brute-force completion: KATs, schedule inversion, reference agreement.

Independent oracles: the Python SPECK/SIMON reference (neural_sat_speck.*),
the official KAT values in tests/fixtures, and Gohr's expand_key convention
(copied below from scripts/gohr_published_net.py lines 61-79, because that
module imports torch at import time).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from neural_sat_speck.simon import cipher as simon_cipher
from neural_sat_speck.simon import key_schedule as simon_schedule
from neural_sat_speck.speck import cipher as speck_cipher
from neural_sat_speck.speck import key_schedule as speck_schedule
from scripts import bruteforce_completion as bf
from tests.fixtures import simon32_64_vectors, speck32_64_vectors

KAT_KEY = (0x1918, 0x1110, 0x0908, 0x0100)
SPECK_KAT = ((0x6574, 0x694C), (0xA868, 0x42F2))
SIMON_KAT = ((0x6565, 0x6877), (0xC69B, 0xE9BB))
SEED = 20261011
SPACE16 = 1 << 16


def gohr_expand_key(k, rounds):
    """Gohr's SPECK32/64 expand_key (scripts/gohr_published_net.py), copied verbatim."""

    mask = 0xFFFF

    def ror(v, r):
        return ((v >> r) | (v << (16 - r))) & mask

    def rol(v, r):
        return ((v << r) | (v >> (16 - r))) & mask

    ks = [k[-1]]
    ell = list(reversed(k[:-1]))
    for i in range(rounds - 1):
        l_i, k_i = ell[i % 3], ks[i]
        new_l = ((ror(l_i, 7) + k_i) & mask) ^ i
        new_k = rol(k_i, 2) ^ new_l
        ell[i % 3] = new_l
        ks.append(new_k)
    return ks


def python_expand(cipher: str, key, rounds: int):
    module = speck_schedule if cipher == "speck" else simon_schedule
    return module.expand_key(key, rounds)


def python_encrypt(cipher: str, block, key, rounds: int):
    module = speck_cipher if cipher == "speck" else simon_cipher
    return module.encrypt_block(block, key, rounds)


def random_key(rng: np.random.Generator) -> tuple[int, int, int, int]:
    return tuple(int(x) for x in rng.integers(0, 1 << 16, size=4))


def known_blocks(
    cipher: str, rounds: int, key, rng: np.random.Generator, pairs: int = 64
):
    """64 random plaintext pairs (two independent uniform blocks each) = 128 blocks.

    The completer does not use any plaintext difference, so pairs are simply two
    uniformly random plaintexts; the grouping is for documentation only.
    """

    plain = rng.integers(0, 1 << 16, size=(pairs, 2, 2))
    pts = [(int(a), int(b)) for a, b in plain.reshape(-1, 2)]
    cts = [python_encrypt(cipher, block, key, rounds) for block in pts]
    return pts, cts


def true_guess(rks, m: int) -> int:
    """Pack the unknown window words k_{R-4}..k_{R-m-1} as in the C header."""

    rounds = len(rks)
    return sum(int(rks[rounds - 4 + t]) << (16 * t) for t in range(4 - m))


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    if shutil.which("clang") is None:
        pytest.skip("brute-force completer requires clang")
    build_dir = tmp_path_factory.mktemp("bruteforce_build")
    return bf.BruteforceLibrary(bf.build_library(build_dir))


def test_fixtures_match_task_known_answer_values():
    assert speck32_64_vectors.MASTER_KEY == KAT_KEY
    assert (speck32_64_vectors.PLAINTEXT, speck32_64_vectors.CIPHERTEXT) == SPECK_KAT
    assert simon32_64_vectors.MASTER_KEY == KAT_KEY
    assert (simon32_64_vectors.PLAINTEXT, simon32_64_vectors.CIPHERTEXT) == SIMON_KAT


def test_official_known_answers_through_c_encryption(lib):
    assert lib.encrypt_block("speck", 22, KAT_KEY, SPECK_KAT[0]) == SPECK_KAT[1]
    assert lib.encrypt_block("simon", 32, KAT_KEY, SIMON_KAT[0]) == SIMON_KAT[1]


def test_official_known_answers_recovered_by_completion(lib):
    """Full 2^16 m=3 completion on the official vectors (one block suffices here)."""

    for cipher, rounds, (pt, ct) in (
        ("speck", 22, SPECK_KAT),
        ("simon", 32, SIMON_KAT),
    ):
        rks = python_expand(cipher, KAT_KEY, rounds)
        result = lib.complete(
            cipher, rounds, rks[rounds - 3 : rounds], [pt], [ct], 0, SPACE16
        )
        assert result.found
        assert result.master_key == KAT_KEY
        assert result.found_guess == rks[rounds - 4]
        assert result.tested == rks[rounds - 4] + 1


@pytest.mark.parametrize(
    ("cipher", "rounds_range"),
    [("speck", range(4, 13)), ("simon", range(4, 15))],
)
def test_schedule_inversion_round_trip(lib, cipher, rounds_range):
    rng = np.random.default_rng(SEED)
    cases = 0
    for rounds in rounds_range:
        for _ in range(40):
            key = random_key(rng)
            window = python_expand(cipher, key, rounds)[rounds - 4 : rounds]
            assert lib.master_from_window(cipher, rounds, window) == key
            cases += 1
    assert cases >= 360


@pytest.mark.parametrize(("cipher", "max_rounds"), [("speck", 22), ("simon", 32)])
def test_arbitrary_windows_invert_consistently(lib, cipher, max_rounds):
    """The window-to-master-key map is a bijection, so no window can be inconsistent."""

    rng = np.random.default_rng(SEED + 5)
    for _ in range(1500):
        rounds = int(rng.integers(4, max_rounds + 1))
        window = tuple(int(x) for x in rng.integers(0, 1 << 16, size=4))
        key = lib.master_from_window(cipher, rounds, window)  # raises on inconsistency
        assert lib.expand(cipher, rounds, key)[rounds - 4 : rounds] == window


@pytest.mark.parametrize(("cipher", "rounds"), [("speck", 8), ("simon", 11)])
def test_full_m3_space_visits_distinct_master_keys(lib, cipher, rounds):
    rng = np.random.default_rng(SEED + 6)
    trailing = [int(x) for x in rng.integers(0, 1 << 16, size=3)]
    keys = {
        lib.master_from_window(cipher, rounds, [g, *trailing]) for g in range(SPACE16)
    }
    assert len(keys) == SPACE16


@pytest.mark.parametrize(("cipher", "max_rounds"), [("speck", 22), ("simon", 32)])
def test_expand_and_encrypt_agree_with_python_reference(lib, cipher, max_rounds):
    rng = np.random.default_rng(SEED + 1)
    for rounds in range(1, max_rounds + 1):
        for _ in range(20):
            key = random_key(rng)
            assert lib.expand(cipher, rounds, key) == python_expand(cipher, key, rounds)
            for block in (
                (0, 0),
                (0xFFFF, 0xFFFF),
                *[(int(a), int(b)) for a, b in rng.integers(0, 1 << 16, size=(2, 2))],
            ):
                assert lib.encrypt_block(cipher, rounds, key, block) == python_encrypt(
                    cipher, block, key, rounds
                )


def test_speck_expand_agrees_with_gohr_convention(lib):
    rng = np.random.default_rng(SEED + 2)
    for rounds in range(1, 23):
        for _ in range(20):
            key = random_key(rng)
            assert list(lib.expand("speck", rounds, key)) == gohr_expand_key(
                key, rounds
            )


@pytest.mark.parametrize(("cipher", "rounds"), [("speck", 8), ("simon", 11)])
def test_full_space_m3_completion_finds_true_key(lib, cipher, rounds):
    rng = np.random.default_rng(SEED + 3)
    key = random_key(rng)
    rks = python_expand(cipher, key, rounds)
    pts, cts = known_blocks(cipher, rounds, key, rng)
    result = lib.complete(
        cipher, rounds, rks[rounds - 3 : rounds], pts, cts, 0, SPACE16
    )
    assert result.found
    assert result.master_key == key
    assert result.found_guess == true_guess(rks, 3)
    assert result.tested == true_guess(rks, 3) + 1


@pytest.mark.parametrize(
    ("cipher", "rounds", "m"),
    [("speck", 8, 2), ("speck", 8, 1), ("simon", 11, 2), ("simon", 11, 1)],
)
def test_narrow_ranges_for_partial_trailing_keys(lib, cipher, rounds, m):
    rng = np.random.default_rng(SEED + 4 + m)
    key = random_key(rng)
    rks = python_expand(cipher, key, rounds)
    pts, cts = known_blocks(cipher, rounds, key, rng)
    space = 1 << (16 * (4 - m))
    g = true_guess(rks, m)
    trailing = rks[rounds - m : rounds]

    lo, hi = max(0, g - 1000), min(space, g + 1001)
    inside = lib.complete(cipher, rounds, trailing, pts, cts, lo, hi)
    assert inside.found
    assert inside.master_key == key
    assert inside.found_guess == g
    assert inside.tested == g - lo + 1

    if g + 1001 <= space:
        lo, hi = g + 1, g + 1001
    else:
        lo, hi = g - 1000, g
    outside = lib.complete(cipher, rounds, trailing, pts, cts, lo, hi)
    assert not outside.found
    assert outside.master_key is None
    assert outside.tested == hi - lo


@pytest.mark.parametrize(("cipher", "rounds"), [("speck", 8), ("simon", 11)])
def test_wrong_trailing_key_is_rejected_over_full_space(lib, cipher, rounds):
    rng = np.random.default_rng(SEED + 7)
    key = random_key(rng)
    rks = python_expand(cipher, key, rounds)
    pts, cts = known_blocks(cipher, rounds, key, rng)
    wrong = (rks[rounds - 3], rks[rounds - 2], rks[rounds - 1] ^ 1)
    result = lib.complete(cipher, rounds, wrong, pts, cts, 0, SPACE16)
    assert not result.found
    assert result.tested == SPACE16


def test_every_block_is_verified_not_only_the_first(lib):
    rng = np.random.default_rng(SEED + 8)
    key = random_key(rng)
    rks = python_expand("speck", key, 8)
    pts, cts = known_blocks("speck", 8, key, rng)
    trailing = rks[5:8]
    control = lib.complete("speck", 8, trailing, pts, cts, 0, SPACE16)
    assert control.found and control.master_key == key

    last_bad = list(cts)
    last_bad[-1] = ((cts[-1][0] ^ 1) & 0xFFFF, cts[-1][1])
    assert not lib.complete("speck", 8, trailing, pts, last_bad, 0, SPACE16).found

    first_bad = list(cts)
    first_bad[0] = (cts[0][0], cts[0][1] ^ 1)
    assert not lib.complete("speck", 8, trailing, pts, first_bad, 0, SPACE16).found


def test_invalid_arguments_are_rejected(lib):
    pts, cts = [(1, 2)], [(3, 4)]
    with pytest.raises(ValueError):
        lib.complete("speck", 3, [0], pts, cts, 0, SPACE16)  # R < 4 for completion
    with pytest.raises(ValueError):
        lib.complete("speck", 8, [0, 0, 0, 0], pts, cts, 0, SPACE16)  # m = 4
    with pytest.raises(ValueError):
        lib.complete("speck", 8, [], pts, cts, 0, SPACE16)  # m = 0
    with pytest.raises(ValueError):
        lib.complete(
            "speck", 8, [0, 0, 0], pts, cts, 0, SPACE16 + 1
        )  # beyond the space
    with pytest.raises(ValueError):
        lib.complete("speck", 8, [0, 0, 0], pts, cts, 5, 4)  # lo > hi
    with pytest.raises(ValueError):
        lib.complete("speck", 8, [0, 0, 0], [], [], 0, SPACE16)  # no blocks
    with pytest.raises(ValueError):
        lib.complete(2, 8, [0, 0, 0], pts, cts, 0, SPACE16)  # unknown cipher id
    with pytest.raises(ValueError):
        lib.complete(
            "speck", 8, [0x10000, 0, 0], pts, cts, 0, SPACE16
        )  # not a 16-bit word


def test_build_uses_frozen_flags_and_produces_library(tmp_path):
    if shutil.which("clang") is None:
        pytest.skip("brute-force completer requires clang")
    path = bf.build_library(tmp_path)
    assert Path(path).exists()
    assert "-Werror" in bf.COMPILER_FLAGS and "-O3" in bf.COMPILER_FLAGS


def test_quick_benchmark_cli_writes_complete_record(tmp_path, monkeypatch):
    if shutil.which("clang") is None:
        pytest.skip("brute-force completer requires clang")
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        monkeypatch.setenv(name, "1")
    output = tmp_path / "quick.json"
    assert bf.main(["--quick", "--output", str(output)]) == 0
    import json

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["mode"] == "quick_smoke_not_for_reporting"
    assert len(record["configs"]) == len(bf.BENCH_CONFIGS) == 8
    for row in record["configs"]:
        rejection = row["rejection"]
        assert rejection["repeats"] == len(rejection["samples_seconds"])
        assert rejection["full_space_label"] == (
            "measured" if row["unknown_bits"] == 16 else "extrapolated"
        )
        assert (row["correct_candidate"] is not None) == (row["m"] == 3)
