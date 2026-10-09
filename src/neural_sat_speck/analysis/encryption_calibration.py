"""Correctness and descriptive accounting for native reference encryption rates."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

FIXTURE_STATE = 0xE5C0202610096002
MASK64 = (1 << 64) - 1
FIXED_KEY = (0x1918, 0x1110, 0x0908, 0x0100)
PROFILES = tuple(
    (cipher, rounds, scope)
    for scope in ("expanded_key", "schedule_charged")
    for cipher, rounds in (
        ("speck32_64", 7),
        ("speck32_64", 22),
        ("simon32_64", 10),
        ("simon32_64", 32),
    )
)
COMPILER_FLAGS = [
    "-std=c11",
    "-O3",
    "-DNDEBUG",
    "-Wall",
    "-Wextra",
    "-Werror",
    "-fno-vectorize",
    "-fno-slp-vectorize",
    "-fno-lto",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_evidence(path: Path, data: Any) -> None:
    """Atomic document replacement with detached content hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload)
    temporary.replace(path)
    path.with_name(path.name + ".sha256").write_text(f"{sha256(path)}  {path.name}\n")


def validate_manifest(manifest: dict[str, Any]) -> None:
    expected = {
        "schema": "reference-encryption-calibration-v1",
        "decision": "D-058",
        "evidence_class": "ENGINEERING",
        "profiles": [
            dict(zip(("cipher", "rounds", "scope"), p, strict=True)) for p in PROFILES
        ],
        "batch_blocks": 65536,
        "fixture_state_hex": "e5c0202610096002",
        "fixed_key_hex_words": ["1918", "1110", "0908", "0100"],
        "repetitions": 7,
        "order": "rotate_base_order_left_by_repetition_index",
        "warmup_seconds": 0.5,
        "minimum_measurement_seconds": 2.0,
        "invocation_timeout_seconds": 15,
        "timing_campaign_ceiling_seconds": 240,
        "reference_allowance_seconds": 300,
        "threads": 1,
        "darwin_qos": "USER_INITIATED_25",
        "process_nice": 0,
        "darwin_background_state": 0,
        "cpu_affinity": "none",
        "compiler": "clang",
        "arithmetic": "uint16 words; uint32 rotation intermediates; uint64 counters and digest",
        "clock": "CLOCK_MONOTONIC; double seconds",
        "native_peak_rss_ceiling_bytes": 67108864,
        "compiler_flags": COMPILER_FLAGS,
        "sources": [
            "tools/reference_encryption.c",
            "tools/reference_encryption.h",
            "tools/encryption_calibration.c",
        ],
        "summary": "median, linear-interpolated IQR, min, max; retain all samples",
        "attack_queries": 128,
        "fresh_verification_queries": 2,
        "offline_assets_in_online_allowance": False,
        "protected_data_access": False,
    }
    # Canonical JSON distinguishes boolean from integer (True == 1 in Python).
    if json.dumps(manifest, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("manifest differs from the frozen calibration profile")


def fixture(count: int) -> tuple[list[tuple[int, int]], list[tuple[int, ...]]]:
    if type(count) is not int or not 1 <= count <= 65536:
        raise ValueError("invalid fixture size")
    inputs, keys = [], []
    state = FIXTURE_STATE
    for i in range(count):
        p = (i * 0x9E3779B1 + 0x65656877) & 0xFFFFFFFF
        inputs.append((p >> 16, p & 0xFFFF))
        state = (state + 0x9E3779B97F4A7C15) & MASK64
        z = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        z ^= z >> 31
        keys.append(tuple((z >> r) & 0xFFFF for r in (48, 32, 16, 0)))
    if len(set(inputs)) != count or len(set(keys)) != count:
        raise ValueError("fixture unexpectedly repeats blocks or keys")
    return inputs, keys


def output_digest(outputs: Iterable[tuple[int, int]]) -> str:
    """FNV-style digest of packed output WORDS, not a cryptographic hash."""
    h = 0xCBF29CE484222325
    for a, b in outputs:
        h = ((h ^ ((a << 16) | b)) * 0x100000001B3) & MASK64
    return f"{h:016x}"


def verify_native(
    binary: Path,
    count: int = 65536,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    from neural_sat_speck.simon.cipher import encrypt_block as simon_encrypt
    from neural_sat_speck.simon.key_schedule import expand_key as simon_expand
    from neural_sat_speck.speck.cipher import encrypt_block as speck_encrypt
    from neural_sat_speck.speck.key_schedule import expand_key as speck_expand
    from tests.fixtures.simon_reference_oracle import (
        oracle_simon32_64_encrypt,
        oracle_simon32_64_key_schedule,
    )

    inputs, keys = fixture(count)
    cases: list[tuple[str, int, tuple[int, ...], tuple[int, int]]] = []
    word_edges = (0, 1, 0xFFFF, 0x8000, 0x7FFF, 0xAAAA, 0x5555, 0xFFFE)
    for cipher, maximum in (("speck32_64", 22), ("simon32_64", 32)):
        for rounds in range(1, maximum + 1):
            for i, w in enumerate(word_edges):
                key = (w, word_edges[(i + 1) % 8], w ^ 0xFFFF, w)
                cases.append((cipher, rounds, key, (w, word_edges[(i + 2) % 8])))
    for cipher, rounds, _ in PROFILES[:4]:
        indices = sorted({0, count - 1, *range(min(64, count))})
        for i in indices:
            cases.append((cipher, rounds, keys[i], inputs[i]))
    cases += [
        ("speck32_64", 22, FIXED_KEY, (0x6574, 0x694C)),
        ("simon32_64", 32, FIXED_KEY, (0x6565, 0x6877)),
        ("speck32_64", 7, FIXED_KEY, (0x6574, 0x694C)),
        ("simon32_64", 10, FIXED_KEY, (0x6565, 0x6877)),
    ]
    text = "".join(
        " ".join(map(str, (int(c == "simon32_64"), r, *k, *p))) + "\n"
        for c, r, k, p in cases
    )
    result = subprocess.run(
        [str(binary), "--verify"],
        input=text,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    if len(rows) != len(cases):
        raise ValueError("native verification omitted cases")
    for (c, r, k, p), row in zip(cases, rows, strict=True):
        encrypt, expand = (
            (simon_encrypt, simon_expand)
            if c == "simon32_64"
            else (speck_encrypt, speck_expand)
        )
        ct, rk = encrypt(p, k, rounds=r), expand(k, rounds=r)
        if row != {"ciphertext": list(ct), "round_keys": list(rk)}:
            raise ValueError(f"native/reference mismatch: {c}, R{r}, {k}, {p}")
        if c == "simon32_64" and (
            ct != oracle_simon32_64_encrypt(p, k, r)
            or rk != oracle_simon32_64_key_schedule(k, r)
        ):
            raise ValueError("SIMON clean-room oracle mismatch")
    official = ((0xA868, 0x42F2), (0xC69B, 0xE9BB), (0xE786, 0x574E))
    if any(tuple(rows[-4 + i]["ciphertext"]) != v for i, v in enumerate(official)):
        raise ValueError("known-answer vector mismatch")
    digests = {}
    for cipher, rounds, scope in PROFILES:
        progress(f"Checking all {count} fixture blocks: {cipher} R{rounds} {scope}")
        encrypt = simon_encrypt if cipher == "simon32_64" else speck_encrypt
        digest = output_digest(
            encrypt(p, FIXED_KEY if scope == "expanded_key" else k, rounds=rounds)
            for p, k in zip(inputs, keys, strict=True)
        )
        native = json.loads(
            subprocess.run(
                [str(binary), "--batch", cipher, str(rounds), scope, str(count)],
                text=True,
                capture_output=True,
                check=True,
                timeout=30,
            ).stdout
        )
        if (
            native["checksum"] != digest
            or native["batch_blocks"] != count
            or (native["batches"] != 1 or native["blocks"] != count)
        ):
            raise ValueError("full-batch output mismatch")
        digests[f"{cipher}:r{rounds}:{scope}"] = digest
    return {
        "status": "PASS",
        "all_round_case_count": len(cases),
        "fixture_blocks_per_profile": count,
        "fixture_profiles_checked": 8,
        "expected_digests": digests,
        "official_full_round_kats": 2,
        "speck_r7_regression_hex": ["e786", "574e"],
        "simon_r10_oracle_checked_hex": [f"{w:04x}" for w in rows[-1]["ciphertext"]],
    }


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low = math.floor(index)
    return ordered[low] + (ordered[math.ceil(index)] - ordered[low]) * (index - low)


def summarize_samples(
    samples: list[dict[str, Any]],
    manifest: dict[str, Any],
    digests: dict[str, str],
    *,
    darwin: bool = True,
    identity_sha256: str | None = None,
) -> list[dict[str, Any]]:
    validate_manifest(manifest)
    grid: dict[tuple[int, str, int, str], dict[str, Any]] = {}
    grouped: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for ordinal, row in enumerate(samples, 1):
        profile = (row["cipher"], row["rounds"], row["scope"])
        rep = row["repetition"]
        if type(rep) is not int or not 0 <= rep < manifest["repetitions"]:
            raise ValueError("invalid repetition")
        if type(row["rounds"]) is not int or profile not in PROFILES:
            raise ValueError("invalid sample profile")
        identity = (rep, *profile)
        if identity in grid:
            raise ValueError("duplicate sample")
        expected_rep, position = divmod(ordinal - 1, len(PROFILES))
        expected_profile = PROFILES[(position + expected_rep) % len(PROFILES)]
        if (
            rep != expected_rep
            or profile != expected_profile
            or type(row["ordinal"]) is not int
            or row["ordinal"] != ordinal
        ):
            raise ValueError("sample order differs from frozen counterbalancing")
        if identity_sha256 is not None and row["identity_sha256"] != identity_sha256:
            raise ValueError("sample execution identity mismatch")
        grid[identity] = row
        for field in (
            "batch_blocks",
            "batches",
            "blocks",
            "threads",
            "qos_class",
            "nice",
        ):
            if type(row[field]) is not int:
                raise ValueError(f"noninteger sample {field}")
        elapsed = row["elapsed_seconds"]
        if (
            type(elapsed) not in (int, float)
            or not math.isfinite(elapsed)
            or not (
                manifest["minimum_measurement_seconds"]
                <= elapsed
                < manifest["invocation_timeout_seconds"]
            )
        ):
            raise ValueError("invalid elapsed time")
        if (
            row["batch_blocks"] != 65536
            or row["batches"] < 1
            or (row["blocks"] != row["batches"] * row["batch_blocks"])
        ):
            raise ValueError("invalid block accounting")
        if (
            row["threads"] != 1
            or row["nice"] != 0
            or (darwin and row["qos_class"] != 25)
        ):
            raise ValueError("runtime profile drift")
        if type(row["peak_rss_bytes"]) is not int or not (
            0 < row["peak_rss_bytes"] <= manifest["native_peak_rss_ceiling_bytes"]
        ):
            raise ValueError("invalid native memory profile")
        if row["checksum"] != digests[f"{profile[0]}:r{profile[1]}:{profile[2]}"]:
            raise ValueError("timed output mismatch")
        grouped[profile].append(row["blocks"] / elapsed)
    if len(grid) != len(PROFILES) * manifest["repetitions"]:
        raise ValueError("incomplete sample grid")
    output = []
    for cipher, rounds, scope in PROFILES:
        values = grouped[cipher, rounds, scope]
        median = statistics.median(values)
        output.append(
            {
                "cipher": cipher,
                "rounds": rounds,
                "scope": scope,
                "n": len(values),
                "reference_unit": "expanded-key encryption"
                if scope == "expanded_key"
                else "key expansion plus encryption",
                "median_blocks_per_second": median,
                "q1_blocks_per_second": quantile(values, 0.25),
                "q3_blocks_per_second": quantile(values, 0.75),
                "iqr_blocks_per_second": quantile(values, 0.75)
                - quantile(values, 0.25),
                "min_blocks_per_second": min(values),
                "max_blocks_per_second": max(values),
                "reference_units_at_300s": 300 * median,
                "log2_reference_units_at_300s": math.log2(300 * median),
                "log2_reference_min": math.log2(300 * min(values)),
                "log2_reference_max": math.log2(300 * max(values)),
            }
        )
    return output
