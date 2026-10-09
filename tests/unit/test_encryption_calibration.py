"""Independent vectors, malformed native input and calibration accounting gates."""

from __future__ import annotations

import copy
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from neural_sat_speck.analysis.encryption_calibration import (
    COMPILER_FLAGS,
    PROFILES,
    fixture,
    output_digest,
    sha256,
    summarize_samples,
    validate_manifest,
    verify_native,
)
from scripts import calibrate_reference_encryptions as runner

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    compiler = shutil.which("clang")
    if not compiler:
        pytest.skip("native reference gate requires clang")
    build = tmp_path_factory.mktemp("reference-encryption")
    objects = []
    for source in ("reference_encryption", "encryption_calibration"):
        target = build / (source + ".o")
        subprocess.run(
            [
                compiler,
                *COMPILER_FLAGS,
                "-c",
                str(ROOT / "tools" / (source + ".c")),
                "-o",
                str(target),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        objects.append(target)
    executable = build / "reference"
    subprocess.run(
        [compiler, "-fno-lto", *map(str, objects), "-o", str(executable)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return executable


def policy():
    return json.loads(
        (ROOT / "docs/encryption_cost_calibration_manifest_20261009.json").read_text()
    )


def synthetic_samples():
    digests = {f"{c}:r{r}:{s}": "0123456789abcdef" for c, r, s in PROFILES}
    rows = []
    for rep in range(7):
        for c, r, s in PROFILES[rep:] + PROFILES[:rep]:
            rows.append(
                {
                    "cipher": c,
                    "rounds": r,
                    "scope": s,
                    "repetition": rep,
                    "ordinal": len(rows) + 1,
                    "batch_blocks": 65536,
                    "batches": 400,
                    "blocks": 400 * 65536,
                    "elapsed_seconds": 2.0 + rep * 0.1,
                    "checksum": "0123456789abcdef",
                    "qos_class": 25,
                    "nice": 0,
                    "threads": 1,
                    "peak_rss_bytes": 4 * 1024 * 1024,
                    "identity_sha256": "a" * 64,
                }
            )
    return rows, digests


def test_native_kats_all_prefixes_and_public_fixture(binary):
    receipt = verify_native(binary, count=257, progress=lambda _: None)
    assert receipt["status"] == "PASS"
    assert receipt["official_full_round_kats"] == 2
    assert receipt["fixture_profiles_checked"] == 8
    assert receipt["simon_r10_oracle_checked_hex"] == ["b397", "687c"]
    assert receipt["speck_r7_regression_hex"] == ["e786", "574e"]


def test_effective_cpu_policy_is_checked_without_timing(binary):
    row = json.loads(
        subprocess.run(
            [str(binary), "--policy-check"],
            check=True,
            text=True,
            capture_output=True,
            timeout=5,
        ).stdout
    )
    assert row == {
        "qos_class": 25 if sys.platform == "darwin" else 0,
        "nice": 0,
        "threads": 1,
    }


def test_observer_deadline_is_passed_to_subprocess(monkeypatch):
    monkeypatch.setattr(runner.time, "monotonic", lambda: 100.0)
    observed = []

    def fake_run(*args, **kwargs):
        observed.append(kwargs["timeout"])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner.observe(["example"], deadline=100.1)
    assert observed == pytest.approx([0.1])
    with pytest.raises(TimeoutError):
        runner.observe(["example"], deadline=99)
    assert len(observed) == 1


def test_pinned_source_is_the_actual_imported_module(tmp_path, monkeypatch):
    compiler = tmp_path / "compiler"
    compiler.write_bytes(b"fixture")
    identity = {
        "pinned_sha256": {name: sha256(ROOT / name) for name in runner.PINNED},
        "build_sha256": {},
        "compiler_path": str(compiler),
        "compiler_sha256": sha256(compiler),
    }
    runner.check_identity(identity)
    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda _: SimpleNamespace(__file__=str(compiler)),
    )
    with pytest.raises(ValueError, match="executed module"):
        runner.check_identity(identity)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "\n",
        "0 7 1 2\n",
        "0 7 1 2 3 4 5 6 7\n",
        "2 7 1 2 3 4 5 6\n",
        "0 23 1 2 3 4 5 6\n",
        "1 33 1 2 3 4 5 6\n",
        "0 7 -1 2 3 4 5 6\n",
        "0 7 184467440737095516161 2 3 4 5 6\n",
    ],
)
def test_native_rejects_malformed_verification(binary, text):
    p = subprocess.run(
        [str(binary), "--verify"], input=text, text=True, capture_output=True, timeout=5
    )
    assert p.returncode == 2


@pytest.mark.parametrize(
    "args",
    [
        ["--batch", "wrong", "7", "expanded_key", "64"],
        ["--batch", "speck32_64", "0", "expanded_key", "64"],
        ["--batch", "simon32_64", "32", "wrong", "64"],
        ["--batch", "simon32_64", "32", "expanded_key", "65537"],
        ["--measure", "speck32_64", "7", "expanded_key", "64", "nan", "2"],
        ["--measure", "speck32_64", "7", "expanded_key", "64", ".5", "inf"],
    ],
)
def test_native_rejects_invalid_arguments_before_work(binary, args):
    p = subprocess.run([str(binary), *args], text=True, capture_output=True, timeout=5)
    assert p.returncode == 2


def test_full_fixture_has_distinct_keys_and_blocks_and_known_digest():
    blocks, keys = fixture(65536)
    assert blocks[0] == (0x6565, 0x6877)
    assert len(set(blocks)) == len(set(keys)) == 65536
    assert output_digest([(0, 0)]) == "af63bd4c8601b7df"


def test_measured_conversion_uses_all_repetitions_and_one_profile_rate():
    rows, digests = synthetic_samples()
    result = summarize_samples(rows, policy(), digests, identity_sha256="a" * 64)
    assert len(result) == 8
    for item in result:
        expected = 400 * 65536 / 2.3
        assert item["median_blocks_per_second"] == pytest.approx(expected)
        assert item["reference_units_at_300s"] == pytest.approx(300 * expected)
        assert item["log2_reference_units_at_300s"] == pytest.approx(
            math.log2(300 * expected)
        )
        assert item["min_blocks_per_second"] == pytest.approx(400 * 65536 / 2.6)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("elapsed_seconds", float("nan")),
        ("elapsed_seconds", -1),
        ("elapsed_seconds", 1.99),
        ("batches", True),
        ("blocks", 1),
        ("checksum", "wrong"),
        ("qos_class", 9),
        ("nice", 5),
        ("threads", 8),
        ("peak_rss_bytes", 0),
        ("peak_rss_bytes", 100000000),
        ("identity_sha256", "b" * 64),
        ("ordinal", 3),
    ],
)
def test_sample_tampering_rejected(field, value):
    rows, digests = synthetic_samples()
    rows[0][field] = value
    with pytest.raises(ValueError):
        summarize_samples(rows, policy(), digests, identity_sha256="a" * 64)


def test_missing_duplicate_and_wrong_order_fail_closed():
    rows, digests = synthetic_samples()
    for bad in (rows[:-1], rows + [rows[0]], [rows[1], rows[0], *rows[2:]]):
        with pytest.raises(ValueError):
            summarize_samples(bad, policy(), digests)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("threads", True),
        ("repetitions", 5),
        ("minimum_measurement_seconds", 1),
        ("compiler_flags", ["-O0"]),
        ("protected_data_access", True),
    ],
)
def test_frozen_profile_drift_rejected(field, value):
    manifest = copy.deepcopy(policy())
    manifest[field] = value
    with pytest.raises(ValueError):
        validate_manifest(manifest)
