"""Freeze existing opened-demo inputs and source; generate no cryptographic data."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_sat_speck.backend.guard import atomic_json  # noqa: E402
from neural_sat_speck.followups.stage2_control import (  # noqa: E402
    PROFILE,
    asset_identity,
    fixed_jobs,
    streaming_sha256,
)

EXPECTED = {
    "vv9": "2363b843592529fe9409dc25dbc0f50c20565973a4a23e54e5478a03113a7070",
    "vv8": "f4e780e484f998ac7e3f05ff78adf47e9dcc1a7a405c9aec92bb06a2761a65fc",
    "ddt": "52b3f3d4df3e20d64942f95ea2837f55b6d7591f1989e92dffd6212938e25c46",
}
SCRIPTS = (
    "prepare_simon_matched_stage2_control.py",
    "run_simon_matched_stage2_worker.py",
    "evaluate_simon_matched_stage2_control.py",
    "analyze_simon_matched_stage2_control.py",
    "audit_simon_matched_stage2_pilot.py",
)


def pin(path: Path, *, stat: bool = False) -> dict:
    before = asset_identity(path)
    result = {"path": str(path.absolute()), "sha256": streaming_sha256(path)}
    if before != asset_identity(path):
        raise ValueError("File changed during streaming identity check")
    if stat:
        result["identity"] = before
    return result


def probe(python: Path, source: Path, scorer: str) -> dict:
    result = subprocess.run(
        [
            str(python),
            "-c",
            "import runpy,json,sys; m=runpy.run_path(sys.argv[1]); "
            "print(json.dumps(m['smoke'](sys.argv[2]),sort_keys=True))",
            str(source / "scripts/run_simon_matched_stage2_worker.py"),
            scorer,
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return json.loads(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot-report", type=Path)
    parser.add_argument("--pilot-manifest", type=Path)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.with_name(args.output.name + ".sha256").exists()
    ):
        raise FileExistsError("Never overwrite a frozen manifest")
    if (args.pilot_report is None) != (args.pilot_manifest is None):
        raise ValueError("Supply both pilot report and manifest or neither")
    source, assets = args.source_root.absolute(), args.asset_root.absolute()
    if subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=source,
        capture_output=True,
        text=True,
        check=True,
    ).stdout:
        raise ValueError("Freeze clean committed source first")
    files = {
        "cohort": assets / "datasets/demo_cross_cipher_depth_r6_r11/dataset.json",
        "python": Path(sys.executable).absolute(),
        "tf_python": assets / "runtime/simon_tf_venv/bin/python3.12",
        "protocol": source / "docs/SIMON_MATCHED_STAGE2_CONTROL_DESIGN_20261005.md",
        "execution_note": source
        / "docs/SIMON_MATCHED_STAGE2_EXIT_RACE_REPAIR_20261006.md",
    }
    for path in sorted((source / "src").rglob("*.py")):
        files["source:" + str(path.relative_to(source))] = path
    for name in SCRIPTS:
        files["source:scripts/" + name] = source / "scripts" / name
    paths = {
        "vv9": assets / "runtime/bao-simon-audit/simonNDvsDD/SENet/ND_VV_Simon32_9R.h5",
        "vv8": assets / "runtime/bao-simon-audit/simonNDvsDD/SENet/ND_VV_Simon32_8R.h5",
        "ddt": assets / "runtime/resources/simon_predicted_ddt_7r.bin",
    }
    pins = {name: pin(path, stat=True) for name, path in paths.items()}
    if any(pins[name]["sha256"] != value for name, value in EXPECTED.items()):
        raise ValueError("Published checkpoint or predicted DDT identity changed")
    runtime = {
        "python": probe(files["python"], source, "ddt7"),
        "tensorflow": probe(files["tf_python"], source, "vv9"),
    }
    if runtime["tensorflow"]["packages"].get("tensorflow") != "2.16.2":
        raise ValueError("TensorFlow runtime differs from audited environment")
    manifest = {
        "status": "APPROVED",
        "execution_authorized": True,
        "mode": "full" if args.pilot_report else "pilot",
        "authorization_decision": "D-053",
        "source_root": str(source),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "profile": PROFILE,
        "jobs": fixed_jobs() if args.pilot_report else fixed_jobs()[:3],
        "files": {name: pin(path) for name, path in files.items()},
        "assets": pins,
        "runtime": runtime,
        "limits": [
            "Opened N8; conditional on shared Bao VV9 parents; descriptive only",
            "No solver, recovery, verifier, new dataset or protected seed",
            "Costs are shared-work projections; no autonomous attack timing",
            "TF intra-op eight matches prior autonomous inference; not equal CPU threads",
            "AC or battery allowed before outcomes; raw samples retained",
            "RSS sampled every ten seconds; no instantaneous/all-thread certification",
            "Real key0 pilot excluded; it can warm filesystem caches",
        ],
    }
    if args.pilot_report:
        manifest["pilot"] = {
            "manifest": pin(args.pilot_manifest),
            "report": pin(args.pilot_report),
        }
    atomic_json(args.output, manifest)
    print(args.output)
    print(streaming_sha256(args.output))


if __name__ == "__main__":
    main()
