"""Pin an authorized opened-demo order diagnostic; generates no attack data."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_sat_speck.backend.guard import atomic_json  # noqa: E402
from neural_sat_speck.followups.queue_order import PROFILE, fixed_jobs  # noqa: E402
from neural_sat_speck.followups.runner import runtime_versions  # noqa: E402

EXPECTED = {
    "speck_stage1": "0a5afc9fc6f1d78bc218a157ef2680ba257aa8230c716d410e868be7bf0ec351",
    "speck_stage2": "a93d29a76a521b6f3fd0f1dff943df9f74de3857169b9d735c9aa478817954d5",
    "speck_classical": "7a1002c3213226cc8543608f262e9984e8c7a7a57d49e3302828ae5a69a699df",
    "simon_ddt": "52b3f3d4df3e20d64942f95ea2837f55b6d7591f1989e92dffd6212938e25c46",
    "simon_tool": "4fbceea2557a0f89908cd7dc8dcb37d4e6793eaf024486745f6cc230402efcae",
    "bao_vv9": "2363b843592529fe9409dc25dbc0f50c20565973a4a23e54e5478a03113a7070",
}


def file_pin(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.absolute()), "sha256": digest.hexdigest()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--asset-root", type=Path, default=ROOT)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pilot-report", type=Path)
    p.add_argument("--pilot-manifest", type=Path)
    args = p.parse_args()
    source, assets = args.source_root.absolute(), args.asset_root.absolute()
    if args.output.exists():
        raise FileExistsError("Never overwrite an approved diagnostic manifest")
    if subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=source,
        capture_output=True,
        text=True,
        check=True,
    ).stdout:
        raise ValueError("Freeze clean committed source before preparing the manifest")
    import pycryptosat

    files = {
        "cohort": assets / "datasets/demo_cross_cipher_depth_r6_r11/dataset.json",
        "python": Path(sys.executable),
        "pycryptosat": Path(pycryptosat.__file__),
        "protocol": source / "docs/QUEUE_ORDER_DIAGNOSTIC_PROTOCOL_20261005.md",
        "speck_stage1": assets
        / "models/speck32_64_phase11b_prime_stage1_r6_b1_mean.pt",
        "speck_stage2": assets / "models/speck32_64_phase6a_b1_mean_g64.pt",
        "speck_classical": assets / "models/classical_dist_r5_sparse_v2.json",
        "simon_ddt": assets / "runtime/resources/simon_predicted_ddt_7r.bin",
        "simon_tool": assets / "runtime/resources/simon_queried_row_8r",
        "tf_python": assets / "runtime/simon_tf_venv/bin/python3.12",
        "bao_vv9": assets
        / "runtime/bao-simon-audit/simonNDvsDD/SENet/ND_VV_Simon32_9R.h5",
    }
    for path in sorted((source / "src").rglob("*.py")):
        files["source:" + str(path.relative_to(source))] = path
    for name in (
        "run_neural_ranking_worker.py",
        "run_queue_order_neural_ranking.py",
        "prepare_queue_order_diagnostic.py",
        "evaluate_queue_order_diagnostic.py",
        "pilot_queue_order_diagnostic.py",
        "analyze_queue_order_diagnostic.py",
    ):
        files["source:scripts/" + name] = source / "scripts" / name
    pins = {name: file_pin(path) for name, path in files.items()}
    for name, expected in EXPECTED.items():
        if pins[name]["sha256"] != expected:
            raise ValueError(f"Frozen comparator asset differs: {name}")
    tf_probe = subprocess.run(
        [
            str(files["tf_python"]),
            "-c",
            "import importlib.metadata,json;print(json.dumps({n:importlib.metadata.version(n) for n in ('numpy','tensorflow','keras','h5py')}))",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    manifest = {
        "status": "APPROVED",
        "execution_authorized": True,
        "execution_kind": "full" if args.pilot_report is not None else "pilot",
        "authorization_decision": "D-050",
        "authorization": "Author instruction on2026-10-05: continue bounded research controls and start the next experiment",
        "source_root": str(source),
        "asset_root": str(assets),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "profile": PROFILE,
        "jobs": fixed_jobs(),
        "files": pins,
        "runtime_versions": runtime_versions(),
        "python_version": sys.version,
        "tensorflow_runtime_versions": json.loads(tf_probe.stdout),
        "pilot_gate": {
            "required": True,
            "authorization_decision": "D-050",
            "job_ids": [j["job_id"] for j in fixed_jobs()[:12]],
        },
        "scientific_limits": [
            "Opened N8 per cipher; descriptive only",
            "Shared ranking cost-accounted replay, not autonomous timing replication",
            "One fixed permutation per set; no across-shuffle population claim",
            "SPECK Stage2 table is sparse empirical; method remains hybrid",
            "No continuous all-thread QoS, GPU-memory or core-affinity certification",
            "No protected seed or frozen-cohort rerun",
            "Actual key0 pilot is separate; it may warm OS page caches before this opened diagnostic",
            "Author allowed battery before outcomes; timings depend on recorded power conditions",
        ],
    }
    if (args.pilot_report is None) != (args.pilot_manifest is None):
        raise ValueError("Pilot report and its manifest must be supplied together")
    if args.pilot_report is not None:
        manifest["files"]["pilot_report"] = file_pin(args.pilot_report)
        manifest["files"]["pilot_manifest"] = file_pin(args.pilot_manifest)
    atomic_json(args.output, manifest)
    print(str(args.output))
    print(file_pin(args.output)["sha256"])


if __name__ == "__main__":
    main()
