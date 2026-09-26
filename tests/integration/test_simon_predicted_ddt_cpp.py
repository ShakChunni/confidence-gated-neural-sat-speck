"""Cross-language checks for the bounded SIMON predicted-DDT generator."""

from __future__ import annotations

import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from neural_sat_speck.simon.predicted_ddt import predicted_distribution_toy


@pytest.fixture(scope="module")
def generator(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    binary = tmp_path_factory.mktemp("simon_ddt_cpp") / "generate"
    source = Path(__file__).resolve().parents[2] / "tools/simon_predicted_ddt.cpp"
    subprocess.run(
        [compiler, "-O2", "-std=c++17", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary


@pytest.fixture(scope="module")
def dense_generator(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    binary = tmp_path_factory.mktemp("simon_ddt_dense_cpp") / "generate_dense"
    source = Path(__file__).resolve().parents[2] / "tools/simon_predicted_ddt_dense7.cpp"
    subprocess.run(
        [compiler, "-O2", "-std=c++17", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary


@pytest.mark.parametrize("bits,left,right", [(3, 0, 1), (4, 3, 5), (6, 0, 1), (6, 63, 7)])
@pytest.mark.parametrize("rounds", [0, 1, 3, 6, 7])
def test_cpp_complete_toy_distribution_matches_independent_python_oracle(
    generator: Path, tmp_path: Path, bits: int, left: int, right: int, rounds: int
) -> None:
    path = tmp_path / "distribution.bin"
    result = subprocess.run(
        [str(generator), str(bits), str(rounds), str(left), str(right), "100000", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert f'"round":{rounds}' in result.stdout
    data = path.read_bytes()
    assert data[:8] == b"SMDT0001"
    stored_bits, stored_rounds, initial, denominator_exponent, count = struct.unpack_from(
        "<IIIIQ", data, 8
    )
    assert (stored_bits, stored_rounds, initial, denominator_exponent) == (
        bits, rounds, (left << bits) | right, 60
    )
    assert len(data) == 32 + count * 12
    denominator, expected = predicted_distribution_toy(bits, rounds, (left, right))
    observed = {
        (index >> bits, index & ((1 << bits) - 1)): mass
        for index, mass in struct.iter_unpack("<IQ", data[32:])
    }
    assert count == len(expected) == len(observed)
    assert observed == {state: mass * ((1 << 60) // denominator) for state, mass in expected.items()}


def test_cpp_full_width_rejects_unbounded_dense_round(generator: Path) -> None:
    result = subprocess.run(
        [str(generator), "16", "7", "0", "64", "80000000", "-"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "rounds <=6" in result.stderr


def test_cpp_record_cap_fails_without_silent_truncation(generator: Path, tmp_path: Path) -> None:
    path = tmp_path / "should_not_exist.bin"
    result = subprocess.run(
        [str(generator), "6", "4", "0", "1", "1", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "record cap exceeded" in result.stderr
    assert not path.exists()


def test_cpp_rejects_zero_difference_and_existing_output(generator: Path, tmp_path: Path) -> None:
    zero = subprocess.run(
        [str(generator), "16", "2", "0", "0", "100", "-"],
        capture_output=True,
        text=True,
    )
    assert zero.returncode != 0
    assert "zero input difference" in zero.stderr
    path = tmp_path / "existing.bin"
    path.write_bytes(b"keep this artifact")
    existing = subprocess.run(
        [str(generator), "6", "1", "0", "1", "100", str(path)],
        capture_output=True,
        text=True,
    )
    assert existing.returncode != 0
    assert path.read_bytes() == b"keep this artifact"


@pytest.mark.parametrize("malformed", ["16junk", "-1", "4294967312"])
def test_cpp_rejects_malformed_word_width(generator: Path, malformed: str) -> None:
    result = subprocess.run(
        [str(generator), malformed, "2", "0", "64", "100", "-"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "error:" in result.stderr


def _bao_source_probability(a: int, b: int, c: int) -> float:
    """Literal full-width translation of the pinned C++ diff_prob conditions."""
    mask = 0xFFFF

    def rol(value: int, shift: int) -> int:
        return ((value << shift) | (value >> (16 - shift))) & mask

    gamma = rol(a, 2) ^ b ^ c
    if a == mask:
        return 2.0 ** -15 if gamma.bit_count() % 2 == 0 else 0.0
    rola, rolb = rol(a, 8), rol(a, 1)
    variable_bits = rola | rolb
    if gamma & (variable_bits ^ mask):
        return 0.0
    double_bits = rol(a, 15) & (rola ^ mask) & rolb
    if (gamma ^ rol(gamma, 7)) & double_bits:
        return 0.0
    return 2.0 ** -(variable_bits ^ double_bits).bit_count()


def test_cpp_full_width_three_rounds_matches_source_style_transition(
    generator: Path, tmp_path: Path
) -> None:
    def load(rounds: int) -> dict[int, int]:
        path = tmp_path / f"r{rounds}.bin"
        subprocess.run(
            [str(generator), "16", str(rounds), "0", "64", "10000", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return dict(struct.iter_unpack("<IQ", path.read_bytes()[32:]))

    two_rounds = load(2)
    three_rounds = load(3)
    expected: dict[int, int] = {}
    for index, mass in two_rounds.items():
        a, b = index >> 16, index & 0xFFFF
        for c in range(1 << 16):
            probability = _bao_source_probability(a, b, c)
            if probability:
                expected_index = (c << 16) | a
                expected[expected_index] = expected.get(expected_index, 0) + int(
                    mass * probability
                )
    assert three_rounds == expected


@pytest.mark.parametrize("bits,left,right", [(4, 3, 5), (6, 0, 1), (6, 63, 7)])
def test_dense_seven_round_table_matches_independent_complete_oracle(
    generator: Path,
    dense_generator: Path,
    tmp_path: Path,
    bits: int,
    left: int,
    right: int,
) -> None:
    sparse = tmp_path / "sparse6.bin"
    dense = tmp_path / "dense7.bin"
    tiles = tmp_path / "tiles"
    subprocess.run(
        [str(generator), str(bits), "6", str(left), str(right), "100000", str(sparse)],
        check=True,
        capture_output=True,
        text=True,
    )
    probe = subprocess.run(
        [str(dense_generator), str(sparse), "probe", str(tiles), str(dense)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert not tiles.exists() and not dense.exists()
    generated = subprocess.run(
        [str(dense_generator), str(sparse), "generate", str(tiles), str(dense)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert probe.stdout.splitlines()[0] == generated.stdout.splitlines()[0]
    data = dense.read_bytes()
    assert data[:8] == b"SMDT0002"
    metadata = struct.unpack_from("<IIIIQ", data, 8)
    assert metadata == (bits, 7, (left << bits) | right, 60, 1 << (2 * bits))
    assert len(data) == 32 + 8 * (1 << (2 * bits))
    denominator, expected = predicted_distribution_toy(bits, 7, (left, right))
    observed = {
        (index >> bits, index & ((1 << bits) - 1)): mass[0]
        for index, mass in enumerate(struct.iter_unpack("<Q", data[32:]))
        if mass[0]
    }
    assert observed == {state: mass * ((1 << 60) // denominator) for state, mass in expected.items()}
    assert sum(mass[0] for mass in struct.iter_unpack("<Q", data[32:])) == 1 << 60


def test_dense_generator_rejects_corrupted_sparse_input(
    generator: Path, dense_generator: Path, tmp_path: Path
) -> None:
    sparse = tmp_path / "sparse6.bin"
    subprocess.run(
        [str(generator), "4", "6", "3", "5", "100000", str(sparse)],
        check=True,
        capture_output=True,
        text=True,
    )
    data = bytearray(sparse.read_bytes())
    data[-1] ^= 1
    sparse.write_bytes(data)
    result = subprocess.run(
        [str(dense_generator), str(sparse), "probe", str(tmp_path / "tiles"), str(tmp_path / "dense")],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "mass" in result.stderr
