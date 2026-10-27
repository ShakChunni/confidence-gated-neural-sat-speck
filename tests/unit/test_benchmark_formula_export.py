"""Regression tests for the reproducible cryptographic benchmark formula artifact bundle."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from neural_sat_speck.sat.dimacs import decode_master_key_from_dimacs_solution

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "benchmark_formulas"


def test_benchmark_formula_manifest_integrity():
    """Verify manifest file, sidecar digest, and referenced file hashes."""
    manifest_path = ARTIFACTS_DIR / "manifest.json"
    assert manifest_path.is_file(), "manifest.json must exist in artifacts/benchmark_formulas"

    sidecar_path = ARTIFACTS_DIR / "manifest.json.sha256"
    assert sidecar_path.is_file(), "manifest.json.sha256 must exist"

    raw_manifest = manifest_path.read_bytes()
    expected_sha = hashlib.sha256(raw_manifest).hexdigest()
    assert expected_sha in sidecar_path.read_text(encoding="utf-8")

    manifest = json.loads(raw_manifest.decode("utf-8"))
    assert manifest["manifest_id"] == "benchmark_formula_artifacts_v1"
    assert len(manifest["instances"]) == 4

    for inst in manifest["instances"]:
        for file_type, file_info in inst["files"].items():
            fpath = ARTIFACTS_DIR / file_info["path"]
            assert fpath.is_file(), f"Referenced file {fpath} does not exist"
            actual_sha = hashlib.sha256(fpath.read_bytes()).hexdigest()
            assert actual_sha == file_info["sha256"], f"SHA mismatch for {file_info['path']}"


@pytest.mark.skipif(shutil.which("cadical") is None, reason="needs the cadical command")
def test_speck_subcritical_formula_solvable_by_cadical():
    """Verify that the exported SPECK R=7 m=2 pure CNF formula is solvable by CaDiCaL."""
    cnf_path = ARTIFACTS_DIR / "speck32_64_r7_m2_inst00_pure.cnf"
    varmap_path = ARTIFACTS_DIR / "speck32_64_r7_m2_inst00_varmap.json"
    assert cnf_path.is_file()
    assert varmap_path.is_file()

    varmap = json.loads(varmap_path.read_text(encoding="utf-8"))
    true_master_key = tuple(varmap["master_key"])
    master_key_bits = tuple(tuple(word) for word in varmap["master_key_bits"])

    proc = subprocess.run(
        ["cadical", "-q", str(cnf_path)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 10
    assert "s SATISFIABLE" in proc.stdout
    rec_key = decode_master_key_from_dimacs_solution(proc.stdout, master_key_bits)
    assert rec_key == true_master_key
