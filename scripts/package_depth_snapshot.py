#!/usr/bin/env python3
"""Packages an immutable source snapshot for the cross-cipher depth map run.

Copies src/ into runtime/depth_map_snapshot/src/ and records SHA-256 digests
to isolate the running depth map from any subsequent repo edits by parallel processes.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT / "runtime" / "depth_map_snapshot"
SNAPSHOT_SRC = SNAPSHOT_DIR / "src"


def create_snapshot() -> dict[str, str]:
    if SNAPSHOT_DIR.exists():
        shutil.rmtree(SNAPSHOT_DIR)

    SNAPSHOT_SRC.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "src" / "neural_sat_speck", SNAPSHOT_SRC / "neural_sat_speck")

    # Record hashes
    manifest: dict[str, str] = {}
    for p in sorted(SNAPSHOT_SRC.rglob("*.py")):
        rel = str(p.relative_to(SNAPSHOT_SRC))
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        manifest[rel] = h

    manifest_file = SNAPSHOT_DIR / "snapshot_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    sidecar = SNAPSHOT_DIR / "snapshot_manifest.json.sha256"
    m_bytes = manifest_file.read_bytes()
    m_hash = hashlib.sha256(m_bytes).hexdigest()
    sidecar.write_text(f"{m_hash}  {manifest_file.name}\n", encoding="utf-8")

    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()

    info = {
        "git_commit": git_head,
        "snapshot_manifest_sha256": m_hash,
        "file_count": len(manifest),
    }
    (SNAPSHOT_DIR / "snapshot_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Snapshot created at {SNAPSHOT_DIR}")
    print(f"Commit: {git_head}")
    print(f"File count: {len(manifest)}")
    print(f"Manifest SHA-256: {m_hash}")
    return manifest


if __name__ == "__main__":
    create_snapshot()
