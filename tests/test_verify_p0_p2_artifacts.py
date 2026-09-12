"""Unit tests for P0-P2 publication claims manifest verification.

Tests:
1. Valid manifest verification.
2. Missing claim_id detection.
3. Duplicate claim_id detection.
4. Stale schema rejection (v1, v2).
5. Hash mismatch rejection.
6. Value mismatch rejection (exact and tolerance).
7. Prohibited causal wording rejection.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.verify_p0_p2_artifacts import (
    file_sha256,
    resolve_json_pointer,
    verify_claim,
    verify_manifest,
)


@pytest.fixture
def dummy_artifact(tmp_path: Path) -> tuple[Path, str, dict[str, Any]]:
    doc = {
        "schema": "neural_sat_speck.test_artifact.v3",
        "containment": {"top1": 26, "top4": 64},
        "timing": {"mean_latency": 0.571},
        "records": [{"id": 0, "status": "sat"}, {"id": 1, "status": "timeout"}],
    }
    art_path = tmp_path / "results" / "test_artifact.json"
    art_path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(doc, indent=2)
    art_path.write_text(content, encoding="utf-8")
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return art_path, digest, doc


def test_resolve_json_pointer():
    doc = {
        "a": {"b": [10, 20, {"c": "hello"}]},
        "slash/key": 42,
        "tilde~key": 99,
    }
    assert resolve_json_pointer(doc, "") == doc
    assert resolve_json_pointer(doc, "/a/b/0") == 10
    assert resolve_json_pointer(doc, "/a/b/2/c") == "hello"
    assert resolve_json_pointer(doc, "/slash~1key") == 42
    assert resolve_json_pointer(doc, "/tilde~0key") == 99

    with pytest.raises(ValueError):
        resolve_json_pointer(doc, "no_leading_slash")

    with pytest.raises(KeyError):
        resolve_json_pointer(doc, "/nonexistent")

    with pytest.raises(IndexError):
        resolve_json_pointer(doc, "/a/b/99")


def test_valid_manifest_and_claim(tmp_path: Path, dummy_artifact):
    art_path, digest, _ = dummy_artifact
    assert file_sha256(art_path) == digest
    rel_path = art_path.relative_to(tmp_path)

    claim = {
        "claim_id": "C-TEST-01",
        "artifact_path": str(rel_path),
        "expected_artifact_sha256": digest,
        "expected_schema": "neural_sat_speck.test_artifact.v3",
        "rejected_stale_schemas": ["neural_sat_speck.test_artifact.v1", "neural_sat_speck.test_artifact.v2"],
        "assertions": [
            {"pointer": "/containment/top4", "expected_value": 64, "comparison_type": "exact"},
            {"pointer": "/timing/mean_latency", "expected_value": 0.571, "comparison_type": "tolerance", "is_timing": True},
            {"pointer": "/records/0/status", "expected_value": "sat", "comparison_type": "exact"},
        ],
        "supported_wording": "Observed test containment is 64/64 with mean latency 0.571 s.",
        "prohibited_wording": ["proves", "definitively"],
    }

    errors = verify_claim(claim, repo_root=tmp_path)
    assert errors == []

    manifest = {
        "schema": "neural_sat_speck.publication_claims_manifest.v1",
        "claims": [claim],
    }
    summary = verify_manifest(manifest, repo_root=tmp_path)
    assert summary["success"] is True
    assert summary["passed_claims"] == ["C-TEST-01"]
    assert summary["failed_claims"] == {}


def test_missing_claim_id(tmp_path: Path, dummy_artifact):
    art_path, digest, _ = dummy_artifact
    rel_path = art_path.relative_to(tmp_path)

    claim = {
        "artifact_path": str(rel_path),
        "expected_artifact_sha256": digest,
    }
    errors = verify_claim(claim, repo_root=tmp_path)
    assert any("missing required 'claim_id'" in e for e in errors)

    manifest = {
        "schema": "neural_sat_speck.publication_claims_manifest.v1",
        "claims": [claim],
    }
    summary = verify_manifest(manifest, repo_root=tmp_path)
    assert summary["success"] is False
    assert any("missing 'claim_id'" in e for e in summary["manifest_errors"])


def test_duplicate_claim_id(tmp_path: Path, dummy_artifact):
    art_path, digest, _ = dummy_artifact
    rel_path = art_path.relative_to(tmp_path)

    claim1 = {
        "claim_id": "C-DUP",
        "artifact_path": str(rel_path),
        "expected_artifact_sha256": digest,
    }
    claim2 = {
        "claim_id": "C-DUP",
        "artifact_path": str(rel_path),
        "expected_artifact_sha256": digest,
    }
    manifest = {
        "schema": "neural_sat_speck.publication_claims_manifest.v1",
        "claims": [claim1, claim2],
    }
    summary = verify_manifest(manifest, repo_root=tmp_path)
    assert summary["success"] is False
    assert any("Duplicate claim_id 'C-DUP'" in e for e in summary["manifest_errors"])


def test_stale_schema_rejection(tmp_path: Path):
    stale_doc = {
        "schema": "neural_sat_speck.test_artifact.v2",
        "value": 42,
    }
    art_path = tmp_path / "stale.json"
    art_path.write_text(json.dumps(stale_doc), encoding="utf-8")
    digest = hashlib.sha256(art_path.read_bytes()).hexdigest()

    claim = {
        "claim_id": "C-STALE",
        "artifact_path": "stale.json",
        "expected_artifact_sha256": digest,
        "expected_schema": "neural_sat_speck.test_artifact.v3",
        "rejected_stale_schemas": ["neural_sat_speck.test_artifact.v1", "neural_sat_speck.test_artifact.v2"],
    }
    errors = verify_claim(claim, repo_root=tmp_path)
    assert any("Stale schema rejected" in e for e in errors)
    assert any("Schema mismatch" in e for e in errors)


def test_hash_mismatch_rejection(tmp_path: Path, dummy_artifact):
    art_path, _, _ = dummy_artifact
    rel_path = art_path.relative_to(tmp_path)

    claim = {
        "claim_id": "C-HASH-BAD",
        "artifact_path": str(rel_path),
        "expected_artifact_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
    }
    errors = verify_claim(claim, repo_root=tmp_path)
    assert any("SHA-256 hash mismatch" in e for e in errors)


def test_value_mismatch_rejection(tmp_path: Path, dummy_artifact):
    art_path, digest, _ = dummy_artifact
    rel_path = art_path.relative_to(tmp_path)

    claim = {
        "claim_id": "C-VAL-BAD",
        "artifact_path": str(rel_path),
        "expected_artifact_sha256": digest,
        "assertions": [
            {"pointer": "/containment/top4", "expected_value": 999, "comparison_type": "exact"},
            {"pointer": "/timing/mean_latency", "expected_value": 10.5, "comparison_type": "tolerance", "abs_tol": 0.01, "rel_tol": 0.01},
        ],
    }
    errors = verify_claim(claim, repo_root=tmp_path)
    assert any("Exact assertion failed at '/containment/top4'" in e for e in errors)
    assert any("Tolerance assertion failed at '/timing/mean_latency'" in e for e in errors)


def test_prohibited_wording_rejection(tmp_path: Path, dummy_artifact):
    art_path, digest, _ = dummy_artifact
    rel_path = art_path.relative_to(tmp_path)

    claim = {
        "claim_id": "C-WORDING-BAD",
        "artifact_path": str(rel_path),
        "expected_artifact_sha256": digest,
        "supported_wording": "This experiment definitively proves that screening is the sole mechanism.",
    }
    errors = verify_claim(claim, repo_root=tmp_path)
    assert any("Prohibited causal/overclaim term 'definitively'" in e for e in errors)
    assert any("Prohibited causal/overclaim term 'proves'" in e for e in errors)
    assert any("Prohibited causal/overclaim term 'sole mechanism'" in e for e in errors)
