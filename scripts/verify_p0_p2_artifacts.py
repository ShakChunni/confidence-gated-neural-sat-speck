#!/usr/bin/env python3
"""P0-P2 Publication Claims and Artifact Verification Tool.

Verifies that all scientific claims in docs/publication_claims_manifest.json
match raw immutable JSON artifacts in the repository:
1. Resolves and validates artifact paths and SHA-256 digests.
2. Explicitly rejects stale schemas (v1, v2).
3. Verifies schema versions and recorded dependency hashes.
4. Resolves RFC 6901 JSON pointers to verify exact deterministic values
   and timing tolerances.
5. Verifies that supported wording contains no prohibited causal terms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = DEFAULT_REPO_ROOT / "docs" / "publication_claims_manifest.json"

PROHIBITED_CAUSAL_TERMS = [
    "proves",
    "definitively",
    "decisively",
    "rules out",
    "sole mechanism",
    "strictly necessary",
    "bit-perfect",
    "guaranteed",
]


def file_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def resolve_json_pointer(doc: Any, pointer: str) -> Any:
    """Resolve an RFC 6901 JSON pointer in a parsed JSON document.

    Examples:
        "" -> doc
        "/containment/top4" -> doc["containment"]["top4"]
        "/records/0/status" -> doc["records"][0]["status"]
    """
    if pointer == "":
        return doc
    if not pointer.startswith("/"):
        raise ValueError(f"Invalid RFC 6901 pointer (must start with '/'): {pointer!r}")

    tokens = [t.replace("~1", "/").replace("~0", "~") for t in pointer[1:].split("/")]
    curr = doc
    for token in tokens:
        if isinstance(curr, dict):
            if token not in curr:
                raise KeyError(f"Key {token!r} not found at pointer {pointer!r} in dict keys {list(curr.keys())[:10]}")
            curr = curr[token]
        elif isinstance(curr, list):
            try:
                idx = int(token)
            except ValueError:
                raise IndexError(f"Index {token!r} is not an integer in list of len {len(curr)} at pointer {pointer!r}")
            if idx < 0 or idx >= len(curr):
                raise IndexError(f"Index {idx} out of range [0, {len(curr)}) at pointer {pointer!r}")
            curr = curr[idx]
        else:
            raise TypeError(f"Cannot traverse into scalar {type(curr).__name__} with token {token!r} at pointer {pointer!r}")
    return curr


def verify_claim(
    claim: dict[str, Any],
    repo_root: Path,
    skip_timing: bool = False,
) -> list[str]:
    """Verify a single claim against its underlying artifact.

    Returns a list of error strings. An empty list indicates success.
    """
    errors: list[str] = []
    claim_id = claim.get("claim_id")
    if not claim_id:
        return ["Claim is missing required 'claim_id' field"]

    # 1. Check artifact path and existence
    rel_path = claim.get("artifact_path")
    if not rel_path:
        return [f"[{claim_id}] Missing 'artifact_path'"]

    artifact_path = (repo_root / rel_path).resolve()
    if not artifact_path.exists():
        return [f"[{claim_id}] Artifact not found at {artifact_path} (rel: {rel_path})"]

    # 2. Verify artifact hash
    actual_hash = file_sha256(artifact_path)
    expected_hash = claim.get("expected_artifact_sha256")
    if expected_hash and actual_hash != expected_hash:
        errors.append(
            f"[{claim_id}] SHA-256 hash mismatch for {rel_path}:\n"
            f"  Expected: {expected_hash}\n"
            f"  Actual:   {actual_hash}"
        )

    # 3. Parse JSON
    try:
        with open(artifact_path, encoding="utf-8") as f:
            doc = json.load(f)
    except Exception as e:
        errors.append(f"[{claim_id}] Failed to parse JSON from {rel_path}: {e}")
        return errors

    # 4. Check schema validity and reject stale schemas
    actual_schema = doc.get("schema")
    rejected_stale = claim.get("rejected_stale_schemas", [])
    if actual_schema in rejected_stale:
        errors.append(
            f"[{claim_id}] Stale schema rejected for {rel_path}: '{actual_schema}' is in {rejected_stale}"
        )

    expected_schema = claim.get("expected_schema")
    if expected_schema and actual_schema != expected_schema:
        errors.append(
            f"[{claim_id}] Schema mismatch for {rel_path}: expected '{expected_schema}', got '{actual_schema}'"
        )

    # 5. Check dependency hashes if specified
    expected_deps = claim.get("expected_dependencies", {})
    if expected_deps:
        actual_deps = doc.get("provenance", {}).get("dependency_sha256", {})
        for dep_rel, exp_h in expected_deps.items():
            if dep_rel not in actual_deps:
                errors.append(f"[{claim_id}] Missing dependency {dep_rel} in artifact provenance")
            elif actual_deps[dep_rel] != exp_h:
                errors.append(
                    f"[{claim_id}] Dependency SHA-256 drift for {dep_rel}:\n"
                    f"  Expected in claim: {exp_h}\n"
                    f"  Found in artifact: {actual_deps[dep_rel]}"
                )

    # 6. Verify assertions via RFC 6901 JSON pointers
    assertions = claim.get("assertions", [])
    for idx, assertion in enumerate(assertions):
        pointer = assertion.get("pointer", "")
        expected_val = assertion.get("expected_value")
        comparison = assertion.get("comparison_type", "exact")
        is_timing = assertion.get("is_timing", False)

        if is_timing and skip_timing:
            continue

        try:
            actual_val = resolve_json_pointer(doc, pointer)
        except Exception as e:
            errors.append(f"[{claim_id}] Assertion #{idx} pointer error at {pointer!r}: {e}")
            continue

        if comparison == "exact":
            if actual_val != expected_val:
                errors.append(
                    f"[{claim_id}] Exact assertion failed at {pointer!r}:\n"
                    f"  Expected: {expected_val!r} (type {type(expected_val).__name__})\n"
                    f"  Actual:   {actual_val!r} (type {type(actual_val).__name__})"
                )
        elif comparison == "tolerance":
            rel_tol = assertion.get("rel_tol", 0.05)
            abs_tol = assertion.get("abs_tol", 0.001)
            try:
                act_f = float(actual_val)
                exp_f = float(expected_val)
                if not math.isclose(act_f, exp_f, rel_tol=rel_tol, abs_tol=abs_tol):
                    errors.append(
                        f"[{claim_id}] Tolerance assertion failed at {pointer!r}:\n"
                        f"  Expected: {exp_f} (rel_tol={rel_tol}, abs_tol={abs_tol})\n"
                        f"  Actual:   {act_f}"
                    )
            except (ValueError, TypeError) as e:
                errors.append(f"[{claim_id}] Float conversion failed at {pointer!r}: {e}")
        elif comparison == "contains":
            if expected_val not in actual_val:
                errors.append(
                    f"[{claim_id}] Contains assertion failed at {pointer!r}: {expected_val!r} not in {actual_val!r}"
                )
        else:
            errors.append(f"[{claim_id}] Unknown comparison_type '{comparison}' at assertion #{idx}")

    # 7. Check wording hygiene
    supported_text = claim.get("supported_wording", "")
    if isinstance(supported_text, list):
        supported_text = " ".join(supported_text)
    claim_prohibited = claim.get("prohibited_wording", [])
    combined_prohibited = set(PROHIBITED_CAUSAL_TERMS + claim_prohibited)
    lower_supported = supported_text.lower()
    for term in combined_prohibited:
        if term.lower() in lower_supported:
            errors.append(
                f"[{claim_id}] Prohibited causal/overclaim term '{term}' detected in supported_wording"
            )

    return errors


def verify_manifest(
    manifest: dict[str, Any],
    repo_root: Path,
    skip_timing: bool = False,
) -> dict[str, Any]:
    """Verify all claims in a publication claims manifest."""
    schema = manifest.get("schema")
    expected_manifest_schema = "neural_sat_speck.publication_claims_manifest.v1"
    if schema != expected_manifest_schema:
        return {
            "success": False,
            "manifest_errors": [
                f"Manifest schema mismatch: expected '{expected_manifest_schema}', got '{schema}'"
            ],
            "passed_claims": [],
            "failed_claims": {},
        }

    claims = manifest.get("claims", [])
    seen_ids: set[str] = set()
    manifest_errors: list[str] = []
    passed_claims: list[str] = []
    failed_claims: dict[str, list[str]] = {}

    for claim in claims:
        cid = claim.get("claim_id")
        if not cid:
            manifest_errors.append("Encountered claim with missing 'claim_id'")
            continue
        if cid in seen_ids:
            manifest_errors.append(f"Duplicate claim_id '{cid}' detected in manifest")
            continue
        seen_ids.add(cid)

        claim_errors = verify_claim(claim, repo_root, skip_timing=skip_timing)
        if claim_errors:
            failed_claims[cid] = claim_errors
        else:
            passed_claims.append(cid)

    success = len(manifest_errors) == 0 and len(failed_claims) == 0
    return {
        "success": success,
        "manifest_errors": manifest_errors,
        "passed_claims": passed_claims,
        "failed_claims": failed_claims,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to publication claims manifest JSON",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Repository root directory",
    )
    parser.add_argument(
        "--skip-timing",
        action="store_true",
        help="Skip timing tolerance assertions (deterministic metrics only)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose assertion details",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = args.manifest.resolve()
    repo_root = args.repo_root.resolve()

    if not manifest_path.exists():
        print(f"[✗] Manifest not found: {manifest_path}")
        return 1

    print("=== PUBLICATION CLAIMS & ARTIFACT RECONCILIATION AUDIT ===")
    print(f"Manifest:  {manifest_path}")
    print(f"Repo Root: {repo_root}")
    print(f"Mode:      {'Deterministic only' if args.skip_timing else 'Full (deterministic + timing tolerances)'}\n")

    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        print(f"[✗] Failed to parse manifest JSON: {e}")
        return 1

    summary = verify_manifest(manifest, repo_root, skip_timing=args.skip_timing)

    if summary["manifest_errors"]:
        print("[✗] MANIFEST STRUCTURE ERRORS:")
        for err in summary["manifest_errors"]:
            print(f"    - {err}")
        print()

    print(f"[i] Evaluated {len(summary['passed_claims']) + len(summary['failed_claims'])} claims:")
    for cid in summary["passed_claims"]:
        print(f"  [✓] {cid}: VERIFIED CLEAN")

    if summary["failed_claims"]:
        print("\n[✗] FAILED CLAIMS:")
        for cid, errors in summary["failed_claims"].items():
            print(f"  [✗] {cid}:")
            for err in errors:
                print(f"      {err}")
        print(f"\nAudit FAILED: {len(summary['failed_claims'])} claims failed reconciliation.")
        return 1

    print(f"\n[✓] ALL {len(summary['passed_claims'])} CLAIMS VERIFIED CLEAN AND IMMUTABLE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
