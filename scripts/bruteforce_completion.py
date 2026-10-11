"""Build, wrap and benchmark the native brute-force key-completion baseline.

The C tool (tools/bruteforce_completion.{c,h}) completes a reduced-round
SPECK32/64 or SIMON32/64 master key from m fixed trailing round keys and known
blocks by enumerating the 16*(4-m) unknown bits of the four-round-key window
k_{R-4..R-1}. Conventions and return codes are documented in the header.

Build: clang with the frozen calibration COMPILER_FLAGS (single-threaded,
-O3, -Wall -Wextra -Werror) plus -fPIC -shared, into runtime/bruteforce_build/.

Benchmark (single worker, single thread):
    .venv/bin/python scripts/bruteforce_completion.py [--output PATH] [--quick]

Timing uses time.monotonic_ns around the ctypes call. Whole-space rejection is
measured for 16-bit spaces; 32- and 48-bit spaces are timed on a contiguous
2^24-guess slice and their full-space time is extrapolated. The correct-key
time is measured for m = 3 only, and the half-space figure is derived as
0.5 x full-space time.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_sat_speck.analysis.encryption_calibration import (  # noqa: E402
    COMPILER_FLAGS,
    quantile,
)
from neural_sat_speck.simon import cipher as simon_cipher  # noqa: E402
from neural_sat_speck.simon import key_schedule as simon_schedule  # noqa: E402
from neural_sat_speck.speck import cipher as speck_cipher  # noqa: E402
from neural_sat_speck.speck import key_schedule as speck_schedule  # noqa: E402

BUILD_DIR = ROOT / "runtime" / "bruteforce_build"
BENCH_DIR = ROOT / "runtime" / "bruteforce_benchmarks"
C_SOURCES = (
    ROOT / "tools" / "reference_encryption.c",
    ROOT / "tools" / "bruteforce_completion.c",
)
C_HEADERS = (
    ROOT / "tools" / "reference_encryption.h",
    ROOT / "tools" / "bruteforce_completion.h",
)
AWST = timezone(timedelta(hours=8))

CIPHER_IDS = {"speck": 0, "simon": 1}
WORD_LIMIT = 0xFFFF
U64_LIMIT = 2**64 - 1

# Benchmark design (fixed for the reported run; --quick changes it for smoke tests only).
BENCH_CONFIGS: tuple[tuple[str, int, int], ...] = (
    ("speck", 7, 1),
    ("speck", 7, 2),
    ("speck", 8, 2),
    ("speck", 8, 3),
    ("simon", 10, 1),
    ("simon", 10, 2),
    ("simon", 11, 2),
    ("simon", 11, 3),
)
BENCH_BLOCKS = 128
WHOLE_SPACE_MAX_BITS = 16
REPORT_SLICE_BITS = 24
REPORT_REJECTION_REPEATS = 30
REPORT_SLICE_REPEATS = 5
REPORT_CORRECT_KEYS = 20
QUICK_SLICE_BITS = 16
QUICK_REJECTION_REPEATS = 3
QUICK_SLICE_REPEATS = 2
QUICK_CORRECT_KEYS = 3


def library_filename() -> str:
    return (
        "libbruteforce_completion.dylib"
        if sys.platform == "darwin"
        else "libbruteforce_completion.so"
    )


def build_library(build_dir: Path = BUILD_DIR) -> Path:
    """Compile the completer and reference cipher into one shared library."""

    compiler = shutil.which("clang")
    if compiler is None:
        raise RuntimeError("clang is required to build the brute-force completer")
    build_dir.mkdir(parents=True, exist_ok=True)
    target = build_dir / library_filename()
    command = [
        compiler,
        *COMPILER_FLAGS,
        "-fPIC",
        "-shared",
        f"-I{ROOT / 'tools'}",
        *(str(source) for source in C_SOURCES),
        "-o",
        str(target),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"clang failed with exit code {completed.returncode}:\n{completed.stderr}"
        )
    return target


def _cipher_id(cipher: str | int) -> int:
    if isinstance(cipher, str):
        if cipher not in CIPHER_IDS:
            raise ValueError(f"unknown cipher {cipher!r}; expected 'speck' or 'simon'")
        return CIPHER_IDS[cipher]
    if isinstance(cipher, bool) or not isinstance(cipher, int):
        raise ValueError("cipher must be 'speck', 'simon' or an integer id")
    return cipher


def _words(values: Sequence[int], count: int, name: str) -> tuple[int, ...]:
    if len(values) != count:
        raise ValueError(
            f"{name} must contain exactly {count} words, got {len(values)}"
        )
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= WORD_LIMIT
        ):
            raise ValueError(f"{name} entries must be 16-bit unsigned integers")
    return tuple(int(value) for value in values)


def _u64(value: int, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= U64_LIMIT
    ):
        raise ValueError(f"{name} must be an unsigned 64-bit integer")
    return value


def _check_status(code: int) -> int:
    if code == -1:
        raise ValueError(
            "invalid argument to the brute-force completer (see tools/bruteforce_completion.h)"
        )
    if code == -2:
        raise RuntimeError(
            "internal schedule inconsistency in the brute-force completer"
        )
    return code


@dataclass(frozen=True)
class CompletionResult:
    """Outcome of one bf_complete call; tested counts every guess attempted."""

    found: bool
    master_key: tuple[int, int, int, int] | None
    tested: int
    found_guess: int | None


class BruteforceLibrary:
    """ctypes wrapper around the shared library built by build_library()."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lib = ctypes.CDLL(str(self.path))
        u16p = ctypes.POINTER(ctypes.c_uint16)
        u64p = ctypes.POINTER(ctypes.c_uint64)
        lib = self._lib
        lib.bf_expand.argtypes = [ctypes.c_uint, ctypes.c_uint, u16p, u16p]
        lib.bf_expand.restype = ctypes.c_int
        lib.bf_encrypt_block.argtypes = [ctypes.c_uint, ctypes.c_uint, u16p, u16p]
        lib.bf_encrypt_block.restype = ctypes.c_int
        lib.bf_master_from_window.argtypes = [ctypes.c_uint, ctypes.c_uint, u16p, u16p]
        lib.bf_master_from_window.restype = ctypes.c_int
        lib.bf_complete.argtypes = [
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            u16p,
            ctypes.c_size_t,
            u16p,
            u16p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            u16p,
            u64p,
            u64p,
        ]
        lib.bf_complete.restype = ctypes.c_int

    @staticmethod
    def _array(values: Sequence[int]) -> Any:
        return (ctypes.c_uint16 * len(values))(*values)

    def expand(
        self, cipher: str | int, rounds: int, key: Sequence[int]
    ) -> tuple[int, ...]:
        """Forward round keys k_0..k_{R-1} from the reference expansion."""

        cid = _cipher_id(cipher)
        key_arr = self._array(_words(key, 4, "key"))
        out = (ctypes.c_uint16 * 32)()
        _check_status(self._lib.bf_expand(cid, rounds, key_arr, out))
        return tuple(out[:rounds])

    def encrypt_block(
        self, cipher: str | int, rounds: int, key: Sequence[int], block: Sequence[int]
    ) -> tuple[int, int]:
        """Encrypt one (a, b) block under the master key."""

        cid = _cipher_id(cipher)
        key_arr = self._array(_words(key, 4, "key"))
        block_arr = self._array(_words(block, 2, "block"))
        _check_status(self._lib.bf_encrypt_block(cid, rounds, key_arr, block_arr))
        return (block_arr[0], block_arr[1])

    def master_from_window(
        self, cipher: str | int, rounds: int, window: Sequence[int]
    ) -> tuple[int, int, int, int]:
        """Invert the schedule from chronological k_{R-4..R-1} to the master key."""

        cid = _cipher_id(cipher)
        window_arr = self._array(_words(window, 4, "window"))
        out = (ctypes.c_uint16 * 4)()
        _check_status(self._lib.bf_master_from_window(cid, rounds, window_arr, out))
        return (out[0], out[1], out[2], out[3])

    def complete(
        self,
        cipher: str | int,
        rounds: int,
        trailing: Sequence[int],
        pt_blocks: Sequence[tuple[int, int]],
        ct_blocks: Sequence[tuple[int, int]],
        guess_lo: int,
        guess_hi: int,
    ) -> CompletionResult:
        """Enumerate guesses in [guess_lo, guess_hi) and verify every block on a match."""

        cid = _cipher_id(cipher)
        m = len(trailing)
        if not 1 <= m <= 3:
            raise ValueError("trailing must contain 1 to 3 round-key words")
        if len(pt_blocks) != len(ct_blocks) or not pt_blocks:
            raise ValueError(
                "pt_blocks and ct_blocks must be non-empty and the same length"
            )
        trailing_arr = self._array(_words(trailing, m, "trailing"))
        pt_arr = self._array(
            [w for block in pt_blocks for w in _words(block, 2, "pt block")]
        )
        ct_arr = self._array(
            [w for block in ct_blocks for w in _words(block, 2, "ct block")]
        )
        lo = _u64(guess_lo, "guess_lo")
        hi = _u64(guess_hi, "guess_hi")
        out_key = (ctypes.c_uint16 * 4)()
        tested = ctypes.c_uint64(0)
        found_guess = ctypes.c_uint64(0)
        code = self._lib.bf_complete(
            cid,
            rounds,
            m,
            trailing_arr,
            len(pt_blocks),
            pt_arr,
            ct_arr,
            lo,
            hi,
            out_key,
            ctypes.byref(tested),
            ctypes.byref(found_guess),
        )
        if _check_status(code) == 1:
            return CompletionResult(
                True,
                (out_key[0], out_key[1], out_key[2], out_key[3]),
                int(tested.value),
                int(found_guess.value),
            )
        return CompletionResult(False, None, int(tested.value), None)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def _reference(
    cipher: str,
) -> tuple[Callable[..., tuple[int, ...]], Callable[..., tuple[int, int]]]:
    if cipher == "speck":
        return speck_schedule.expand_key, speck_cipher.encrypt_block
    if cipher == "simon":
        return simon_schedule.expand_key, simon_cipher.encrypt_block
    raise ValueError(f"unknown cipher {cipher!r}")


def _known_blocks(
    cipher: str, rounds: int, key: tuple[int, int, int, int], seed: str, count: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Plaintext blocks and Python-reference ciphertexts under one master key."""

    _, encrypt = _reference(cipher)
    rng = random.Random(seed)
    pts = [
        (rng.randrange(WORD_LIMIT + 1), rng.randrange(WORD_LIMIT + 1))
        for _ in range(count)
    ]
    cts = [encrypt(block, key, rounds) for block in pts]
    return pts, cts


def _random_key(seed: str) -> tuple[int, int, int, int]:
    rng = random.Random(seed)
    return tuple(rng.randrange(WORD_LIMIT + 1) for _ in range(4))  # type: ignore[return-value]


def _timed(call: Callable[[], Any]) -> tuple[float, Any]:
    start = time.monotonic_ns()
    result = call()
    end = time.monotonic_ns()
    return (end - start) * 1e-9, result


def _summary(samples: list[float]) -> dict[str, float]:
    return {
        "median_seconds": statistics.median(samples),
        "q1_seconds": quantile(samples, 0.25),
        "q3_seconds": quantile(samples, 0.75),
        "iqr_seconds": quantile(samples, 0.75) - quantile(samples, 0.25),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
    }


def _benchmark_config(
    library: BruteforceLibrary, cipher: str, rounds: int, m: int, *, quick: bool
) -> dict[str, Any]:
    unknown_bits = 16 * (4 - m)
    space = 1 << unknown_bits
    seed_root = f"bf-bench-{cipher}-{rounds}-{m}"
    key = _random_key(f"{seed_root}-key")
    rks = _reference(cipher)[0](key, rounds)
    pts, cts = _known_blocks(cipher, rounds, key, f"{seed_root}-blocks", BENCH_BLOCKS)
    true_trailing = tuple(rks[rounds - m : rounds])
    wrong_trailing = true_trailing[:-1] + (true_trailing[-1] ^ 1,)

    whole = unknown_bits <= WHOLE_SPACE_MAX_BITS
    if whole:
        guesses = space
        repeats = QUICK_REJECTION_REPEATS if quick else REPORT_REJECTION_REPEATS
        scope = "whole_space"
    else:
        guesses = 1 << (QUICK_SLICE_BITS if quick else REPORT_SLICE_BITS)
        repeats = QUICK_SLICE_REPEATS if quick else REPORT_SLICE_REPEATS
        scope = "contiguous_slice"

    def reject() -> CompletionResult:
        return library.complete(cipher, rounds, wrong_trailing, pts, cts, 0, guesses)

    warm = reject()  # One discarded warm-up run per configuration.
    if warm.found or warm.tested != guesses:
        raise RuntimeError(
            f"wrong-key rejection sanity check failed for {cipher} R={rounds} m={m}"
        )
    samples: list[float] = []
    for _ in range(repeats):
        seconds, result = _timed(reject)
        if result.found or result.tested != guesses:
            raise RuntimeError(
                f"wrong-key rejection sanity check failed for {cipher} R={rounds} m={m}"
            )
        samples.append(seconds)

    median = statistics.median(samples)
    rate = guesses / median
    full_seconds = median if whole else space / rate
    rejection = {
        "scope": scope,
        "guesses_per_run": guesses,
        "repeats": repeats,
        "samples_seconds": samples,
        **_summary(samples),
        "median_seconds_per_guess": median / guesses,
        "guesses_per_second": rate,
        "full_space_seconds": full_seconds,
        "full_space_label": "measured" if whole else "extrapolated",
        "half_space_seconds": 0.5 * full_seconds,
        "half_space_label": "derived_0.5_x_full_space",
    }

    correct: dict[str, Any] | None = None
    if m == 3:
        n_keys = QUICK_CORRECT_KEYS if quick else REPORT_CORRECT_KEYS
        times: list[float] = []
        tested_counts: list[int] = []
        for index in range(n_keys):
            key_i = _random_key(f"{seed_root}-correct-{index}")
            rks_i = _reference(cipher)[0](key_i, rounds)
            pts_i, cts_i = _known_blocks(
                cipher,
                rounds,
                key_i,
                f"{seed_root}-correct-blocks-{index}",
                BENCH_BLOCKS,
            )
            trailing_i = tuple(rks_i[rounds - m : rounds])
            seconds, result = _timed(
                lambda t=trailing_i, p=pts_i, c=cts_i: library.complete(
                    cipher, rounds, t, p, c, 0, space
                )
            )
            if not result.found or result.master_key != key_i:
                raise RuntimeError(
                    f"correct-key completion failed for {cipher} R={rounds} key index {index}"
                )
            times.append(seconds)
            tested_counts.append(result.tested)
        correct = {
            "keys": n_keys,
            "samples_seconds": times,
            **_summary(times),
            "tested_guesses_median": statistics.median(tested_counts),
            "tested_guesses": tested_counts,
            "label": "measured_random_position_true_key",
        }

    return {
        "cipher": cipher,
        "rounds": rounds,
        "m": m,
        "unknown_bits": unknown_bits,
        "space_guesses": space,
        "rejection": rejection,
        "correct_candidate": correct,
    }


def _run_text(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def host_record() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "uname": list(platform.uname()),
        "python": sys.version.split()[0],
        "logical_cpus": os.cpu_count(),
        "cpu_brand": _run_text(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "clang_version": _run_text(["clang", "--version"]),
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "workers": 1,
    }


def repository_record() -> dict[str, Any]:
    status = _run_text(["git", "-C", str(ROOT), "status", "--porcelain"])
    return {
        "git_commit": _run_text(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "dirty_entries_at_run": None if status is None else len(status.splitlines()),
        "source_sha256": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (*C_SOURCES, *C_HEADERS)
        },
    }


def run_benchmark(library_path: Path, *, quick: bool) -> dict[str, Any]:
    library = BruteforceLibrary(library_path)
    host = host_record()
    # The host is shared with other applications; load is recorded, not controlled.
    host["loadavg_1_5_15_at_start"] = list(os.getloadavg())
    configs = [
        _benchmark_config(library, cipher, rounds, m, quick=quick)
        for cipher, rounds, m in BENCH_CONFIGS
    ]
    host["loadavg_1_5_15_at_end"] = list(os.getloadavg())
    return {
        "schema": "bruteforce_completion_benchmark_v1",
        "mode": "quick_smoke_not_for_reporting" if quick else "report",
        "timestamp_awst": datetime.now(AWST).isoformat(timespec="seconds"),
        "clock": "time.monotonic_ns around the ctypes call",
        "compiler_flags": ["clang", *COMPILER_FLAGS, "-fPIC", "-shared"],
        "library_path": str(library_path.relative_to(ROOT)),
        "library_sha256": _sha256(library_path),
        "host": host,
        "repository": repository_record(),
        "design": {
            "blocks_per_key": BENCH_BLOCKS,
            "whole_space_max_bits": WHOLE_SPACE_MAX_BITS,
            "slice_bits": QUICK_SLICE_BITS if quick else REPORT_SLICE_BITS,
            "rejection_repeats_whole_space": QUICK_REJECTION_REPEATS
            if quick
            else REPORT_REJECTION_REPEATS,
            "rejection_repeats_slice": QUICK_SLICE_REPEATS
            if quick
            else REPORT_SLICE_REPEATS,
            "correct_keys_m3": QUICK_CORRECT_KEYS if quick else REPORT_CORRECT_KEYS,
            "warm_up_runs_discarded": 1,
            "wrong_key": "true trailing words with the last word XOR 1",
        },
        "configs": configs,
    }


def _format_seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.4e}"


def print_summary(record: dict[str, Any]) -> None:
    header = (
        f"{'cipher':<6}{'R':>3}{'m':>3}{'bits':>5}  {'scope':<15}{'s/guess(med)':>14}"
        f"{'guess/s':>14}{'full-space s':>14}{'label':>13}{'0.5x full s':>14}{'correct med s':>15}"
    )
    print(header)
    print("-" * len(header))
    for row in record["configs"]:
        rej = row["rejection"]
        corr = row["correct_candidate"]
        print(
            f"{row['cipher']:<6}{row['rounds']:>3}{row['m']:>3}{row['unknown_bits']:>5}  "
            f"{rej['scope']:<15}{_format_seconds(rej['median_seconds_per_guess']):>14}"
            f"{rej['guesses_per_second']:>14.4e}{_format_seconds(rej['full_space_seconds']):>14}"
            f"{rej['full_space_label']:>13}{_format_seconds(rej['half_space_seconds']):>14}"
            f"{_format_seconds(corr['median_seconds'] if corr else None):>15}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON path (default: runtime/bruteforce_benchmarks/<AWST timestamp>.json)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="smoke mode with reduced repeats and 2^16 slices; output is not for reporting",
    )
    args = parser.parse_args(argv)

    os.environ.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
    )
    library_path = build_library(BUILD_DIR)
    record = run_benchmark(library_path, quick=args.quick)

    output = args.output
    if output is None:
        stamp = datetime.now(AWST).strftime("%Y%m%d_%H%M%S_AWST")
        output = BENCH_DIR / f"{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)

    print_summary(record)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
