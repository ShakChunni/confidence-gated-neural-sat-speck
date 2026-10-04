"""Fail-closed identity and fixed-grid checks, with no solver imports."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

PROFILES = (
    "cms_wrapper",
    "cms_native_default",
    "cms_native_gauss",
    "cms_native_no_gauss",
    "cms_cnf_no_gauss",
    "cadical_cnf",
)
UNGUIDED_PROFILES = PROFILES[:3] + ("cadical_cnf",)
OUTCOMES = frozenset(
    {
        "recovered",
        "different_verified_key",
        "verification_failed",
        "unsat_alarm",
        "unknown",
        "global_deadline",
        "solver_error",
        "worker_exception",
        "environment_rejected",
        "infrastructure_interrupted",
        "audit_rejected",
    }
)
THREAD_ENVIRONMENT = {
    name: "1"
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
}


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_id(cipher: str, key_index: int, observations: int, trailing_keys: int) -> str:
    return f"{cipher}_k{key_index}_o{observations}_m{trailing_keys}"


def fixed_grid() -> list[dict[str, Any]]:
    """Counterbalance backend order by key/case; exactly 160 unique cells."""
    cells = []
    for key_index in range(4):
        for ci, (cipher, rounds) in enumerate((("speck32_64", 7), ("simon32_64", 10))):
            for oi, (observations, trailing_keys) in enumerate(
                ((4, 1), (4, 2), (4, 0), (128, 0))
            ):
                profiles = PROFILES if trailing_keys else UNGUIDED_PROFILES
                shift = (key_index + ci + oi) % len(profiles)
                ordered = profiles[shift:] + profiles[:shift]
                for profile in ordered:
                    ident = case_id(cipher, key_index, observations, trailing_keys)
                    cells.append(
                        {
                            "cell_id": f"{ident}__{profile}",
                            "case_id": ident,
                            "cipher": cipher,
                            "rounds": rounds,
                            "key_index": key_index,
                            "observations": observations,
                            "trailing_keys": trailing_keys,
                            "profile": profile,
                        }
                    )
    return cells


def checked_path(root: Path, record: dict[str, Any]) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError("Incomplete file identity")
    name, digest = record["path"], record["sha256"]
    if not isinstance(name, str) or not name:
        raise ValueError("Missing identity path")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Missing SHA-256 identity")
    path = Path(name)
    path = path if path.is_absolute() else root / path
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError(f"File identity mismatch: {name}")
    # Preserve a venv's executable symlink path: resolving it changes sys.prefix.
    return path.absolute()


def validate_manifest(manifest: dict[str, Any], root: Path) -> dict[str, Path]:
    """Validate authorization and all identities before a dataset or worker loads."""
    if manifest.get("schema") != "backend_demo_v1":
        raise ValueError("Unknown backend schema")
    if (
        manifest.get("status") != "APPROVED"
        or manifest.get("execution_authorized") is not True
    ):
        raise ValueError("Backend execution is not authorized")
    exact = {
        "seed": 999999,
        "role": "opened_demo_diagnostic",
        "key_count": 4,
        "cohort_key_count": 8,
        "pair_count": 64,
        "fresh_block_count": 2,
        "deadline_seconds": 300.0,
        "threads": 1,
        "repetitions": 1,
        "max_cell_starts": 160,
        "worst_case_cutoff_seconds": 48000.0,
        "assumption_policy": "unit_clauses_all_profiles",
        "interruption_policy": "retain_started_without_retry",
        "thread_environment": THREAD_ENVIRONMENT,
        "runtime_versions": {
            "cms": "5.14.7",
            "pycryptosat": "5.14.7",
            "cadical": "3.0.1",
        },
    }
    for key, expected in exact.items():
        value = manifest.get(key)
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"Backend profile drift: {key}")
    if canonical_bytes(manifest.get("cells")) != canonical_bytes(fixed_grid()):
        raise ValueError("Backend grid/order drift")
    if manifest.get("native_policy_review_passed") is not True:
        raise ValueError("Actual native policy and Gauss controls remain unreviewed")
    if manifest.get("launch_blockers") != []:
        raise ValueError("Recorded backend launch blockers remain open")
    validate_source_tree(manifest, root)
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Missing file identities")
    required = {
        "cohort",
        "catalog",
        "python",
        "pycryptosat",
        "cms",
        "cadical",
        "protocol",
    }
    if not required <= files.keys() or not any(k.startswith("source:") for k in files):
        raise ValueError("Missing source/runtime/cohort identities")
    sources = {
        f"source:{p.relative_to(root)}"
        for p in (root / "src" / "neural_sat_speck").rglob("*.py")
    }
    sources.update(
        f"source:scripts/{n}_backend_comparison.py"
        for n in ("prepare", "evaluate", "analyze")
    )
    if not sources <= files.keys():
        raise ValueError("Incomplete imported source identity closure")
    paths = {key: checked_path(root, rec) for key, rec in files.items()}
    for key in sources:
        if paths[key] != (root / key.removeprefix("source:")).absolute():
            raise ValueError("Source identity points outside the executed tree")
    catalog = json.loads(paths["catalog"].read_bytes())
    wanted = {cell["case_id"] for cell in fixed_grid()}
    if not isinstance(catalog, dict) or set(catalog) != wanted:
        raise ValueError("Incomplete case catalog")
    for ident, case in catalog.items():
        if case.get("case_id") != ident or not isinstance(case.get("files"), dict):
            raise ValueError("Malformed case identity")
        if set(case["files"]) != {"public", "truth", "native", "cnf", "varmap"}:
            raise ValueError("Incomplete case files")
        for kind, rec in case["files"].items():
            paths[f"{ident}:{kind}"] = checked_path(root, rec)
    return paths


def validate_source_tree(manifest: dict[str, Any], root: Path) -> None:
    """Require a clean committed worktree, rather than an asserted snapshot."""
    commit = manifest.get("source_commit_at_preparation")
    if (
        re.fullmatch(r"[0-9a-f]{40}", str(commit)) is None
        or manifest.get("source_isolation") != "git_worktree"
        or not (root / ".git").is_file()
    ):
        raise ValueError("Committed isolated source worktree is required")

    def git(*arguments):
        result = subprocess.run(
            ["git", *arguments], cwd=root, text=True, capture_output=True, timeout=5
        )
        if result.returncode:
            raise ValueError("Cannot verify committed source identity")
        return result.stdout.strip()

    if (
        git("rev-parse", "HEAD") != commit
        or Path(git("rev-parse", "--show-toplevel")).resolve() != root.resolve()
    ):
        raise ValueError("Committed source revision/root drift")
    controlled = [
        "src",
        "docs/CONTROLLED_BACKEND_COMPARISON_PROTOCOL_V2_20261003.md",
        "scripts/generate_demo_depth_dataset.py",
    ]
    controlled += [
        f"scripts/{name}_backend_comparison.py"
        for name in ("prepare", "evaluate", "analyze")
    ]
    if git("status", "--porcelain", "--untracked-files=all", "--", *controlled):
        raise ValueError("Executed source tree is modified or uncommitted")


def validate_authorization(
    path: Path, manifest_sha256: str, root: Path, *, output: Path | None = None
) -> None:
    """Require a separate reviewed receipt bound to the final manifest bytes."""
    receipt = read_checked_json(path)
    if (
        receipt.get("schema") != "backend_launch_authorization_v1"
        or receipt.get("status") != "AUTHOR_APPROVED"
    ):
        raise ValueError("Missing author-approved backend launch receipt")
    if receipt.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Authorization receipt names another manifest")
    if not re.fullmatch(r"D-[0-9]+", str(receipt.get("decision", ""))):
        raise ValueError("Missing recorded author decision")
    if (
        receipt.get("source_root") != str(root.resolve())
        or receipt.get("isolated_snapshot") is not True
    ):
        raise ValueError("Authorization does not name this isolated source root")
    if receipt.get("depth_campaign_closed") is not True:
        raise ValueError("Depth campaign must finish before backend execution")
    if output is not None and receipt.get("output_path") != str(output.resolve()):
        raise ValueError("Authorization names another output ledger")


def validate_record(record: dict[str, Any], cell: dict[str, Any]) -> None:
    if any(
        type(record.get(k)) is not type(v) or record.get(k) != v
        for k, v in cell.items()
    ):
        raise ValueError("Cell identity mismatch")
    if record.get("outcome") not in OUTCOMES:
        raise ValueError("Unknown final outcome")
    elapsed = record.get("online_seconds")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Invalid online duration")
    if (
        "duration_available" in record
        and type(record["duration_available"]) is not bool
    ):
        raise ValueError("Invalid duration-availability flag")
    for field in ("verified_known", "verified_fresh", "assigned_key_match"):
        if type(record.get(field)) is not bool:
            raise ValueError(f"Invalid verification flag: {field}")
    success = record["outcome"] == "recovered"
    if success != all(
        record[f] for f in ("verified_known", "verified_fresh", "assigned_key_match")
    ):
        raise ValueError("Contradictory recovery flags")
    if success and (elapsed <= 0 or elapsed >= 300.0):
        raise ValueError("Late or zero-cost recovery")
    if record["outcome"] in {
        "recovered",
        "different_verified_key",
        "verification_failed",
    } and (
        not isinstance(record.get("recovered_key"), list)
        or len(record["recovered_key"]) != 4
        or any(
            type(word) is not int or not 0 <= word < 65536
            for word in record["recovered_key"]
        )
    ):
        raise ValueError("Missing recovered key")
    if success and record.get("duration_available", True) is not True:
        raise ValueError("Recovery duration cannot be unavailable")
    outcome = record["outcome"]
    expected_status = {
        "recovered": "sat",
        "different_verified_key": "sat",
        "verification_failed": "sat",
        "unsat_alarm": "unsat",
        "unknown": "unknown",
    }
    if (
        outcome in expected_status
        and record.get("solver_status") != expected_status[outcome]
    ):
        raise ValueError("Outcome contradicts solver status")
    if outcome == "different_verified_key" and not (
        record["verified_known"]
        and record["verified_fresh"]
        and not record["assigned_key_match"]
    ):
        raise ValueError("Different-key outcome lacks independent verification")
    if (
        outcome == "verification_failed"
        and record["verified_known"]
        and record["verified_fresh"]
    ):
        raise ValueError("Verification-failure flags contradict outcome")
    if outcome in {"unknown", "unsat_alarm", "global_deadline"} and (
        any(
            record[f]
            for f in ("verified_known", "verified_fresh", "assigned_key_match")
        )
        or record.get("recovered_key") is not None
    ):
        raise ValueError("Failure unexpectedly claims verified key information")
    if outcome == "global_deadline" and elapsed < 300.0:
        raise ValueError("Global deadline outcome precedes frozen cutoff")
    if (
        outcome == "infrastructure_interrupted"
        and record.get("duration_available") is not False
    ):
        raise ValueError("Interrupted duration is unavailable")


def atomic_json(path: Path, value: Any) -> None:
    """Two-file persistence: mismatch after a crash intentionally fails closed."""
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_bytes(value)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    sidecar = path.with_name(path.name + ".sha256")
    temporary_sidecar = sidecar.with_name(sidecar.name + ".tmp")
    with temporary_sidecar.open("w") as stream:
        stream.write(f"{hashlib.sha256(raw).hexdigest()}  {path.name}\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary_sidecar.replace(sidecar)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_checked_json(path: Path) -> Any:
    sidecar = path.with_name(path.name + ".sha256")
    expected = f"{sha256_file(path)}  {path.name}\n"
    if not sidecar.is_file() or sidecar.read_text() != expected:
        raise ValueError(f"Checksum mismatch: {path}")

    def unique_pairs(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"Duplicate JSON key: {key}")
            value[key] = item
        return value

    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    return json.loads(
        path.read_bytes(),
        object_pairs_hook=unique_pairs,
        parse_constant=reject_constant,
    )
