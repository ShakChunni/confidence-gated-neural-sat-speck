#!/usr/bin/env python3
"""Export the opened cohort and a pinned, unauthorized backend launch proposal."""

from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_sat_speck.backend.artifacts import export_cases  # noqa: E402
from neural_sat_speck.backend.guard import (  # noqa: E402
    THREAD_ENVIRONMENT,
    atomic_json,
    fixed_grid,
    read_checked_json,
    sha256_file,
)


def prepare(cohort: Path, output: Path, cms: Path, cadical: Path) -> dict:
    read_checked_json(cohort)
    export_cases(cohort, output)
    spec = importlib.util.find_spec("pycryptosat")
    if spec is None or spec.origin is None:
        raise RuntimeError("Missing pycryptosat runtime")
    import pycryptosat

    if pycryptosat.__version__ != "5.14.7":
        raise ValueError("Wrapper version differs from proposed diagnostic")
    files = {}
    named = {
        "cohort": cohort,
        "catalog": output / "catalog.json",
        "python": Path(sys.executable).absolute(),
        "pycryptosat": Path(spec.origin),
        "cms": cms,
        "cadical": cadical,
        "protocol": ROOT
        / "docs"
        / "CONTROLLED_BACKEND_COMPARISON_PROTOCOL_V2_20261003.md",
    }
    for name, path in named.items():
        files[name] = {"path": str(path.absolute()), "sha256": sha256_file(path)}
    venv_cfg = Path(sys.executable).parent.parent / "pyvenv.cfg"
    if venv_cfg.is_file():
        files["venv_config"] = {
            "path": str(venv_cfg.absolute()),
            "sha256": sha256_file(venv_cfg),
        }
    generator = ROOT / "scripts" / "generate_demo_depth_dataset.py"
    files["source:scripts/generate_demo_depth_dataset.py"] = {
        "path": str(generator.relative_to(ROOT)),
        "sha256": sha256_file(generator),
    }
    for path in sorted((ROOT / "src" / "neural_sat_speck").rglob("*.py")):
        name = str(path.relative_to(ROOT))
        files[f"source:{name}"] = {"path": name, "sha256": sha256_file(path)}
    for name in (
        "prepare_backend_comparison.py",
        "evaluate_backend_comparison.py",
        "analyze_backend_comparison.py",
    ):
        path = ROOT / "scripts" / name
        if not path.is_file():
            raise ValueError(f"Missing backend entry point: {name}")
        files[f"source:scripts/{name}"] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
        }
    tool_metadata = {}
    for name, tool in (("cms", cms), ("cadical", cadical)):
        tool_metadata[name] = {}
        for flag in ("--version", "--help"):
            proc = subprocess.run(
                [str(tool), flag], text=True, capture_output=True, timeout=5
            )
            tool_metadata[name][flag] = {
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        expected_version = "5.14.7" if name == "cms" else "3.0.1"
        version_output = tool_metadata[name]["--version"]
        if (
            version_output["returncode"] != 0
            or re.search(
                rf"(?<![0-9.]){re.escape(expected_version)}(?![0-9.])",
                version_output["stdout"] + version_output["stderr"],
            )
            is None
        ):
            raise ValueError(f"{name} version differs from fixed diagnostic")
        if sys.platform == "darwin":
            proc = subprocess.run(
                ["/usr/bin/otool", "-L", str(tool)],
                text=True,
                capture_output=True,
                timeout=5,
            )
            tool_metadata[name]["linked_libraries_raw"] = proc.stdout
            # Non-system linked libraries are part of the local runtime identity.
            for line in proc.stdout.splitlines()[1:]:
                name_path = line.strip().split(" (", 1)[0]
                lib = Path(name_path)
                if (
                    lib.is_absolute()
                    and lib.is_file()
                    and not name_path.startswith(("/usr/lib/", "/System/"))
                ):
                    files[f"library:{name_path}"] = {
                        "path": name_path,
                        "sha256": sha256_file(lib),
                    }
    manifest = {
        "schema": "backend_demo_v1",
        "status": "PROPOSED",
        "execution_authorized": False,
        "native_policy_review_passed": False,
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
        "cells": fixed_grid(),
        "files": files,
        "runtime_versions": {
            "cms": "5.14.7",
            "pycryptosat": "5.14.7",
            "cadical": "3.0.1",
        },
        "tools": tool_metadata,
        "python_version": sys.version,
        "source_commit_at_preparation": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_isolation": "git_worktree",
        "launch_blockers": [
            "Independent runner/formula/parser review and qualification required",
            "Actual native worker policy receipt and Gauss controls require review",
            "Freeze from committed isolated source and qualify native/CNF toy solves",
            "Depth campaign must finish before timed backend execution",
            "Author approval of concrete 160-cell profile required",
        ],
    }
    atomic_json(output / "manifest_proposed.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cms", type=Path, default=Path("/opt/homebrew/bin/cryptominisat5")
    )
    parser.add_argument(
        "--cadical", type=Path, default=Path("/opt/homebrew/bin/cadical")
    )
    args = parser.parse_args()
    prepare(
        args.cohort.resolve(),
        args.output_dir.resolve(),
        args.cms.resolve(),
        args.cadical.resolve(),
    )
    print("32 formula cases exported; proposed 160-cell manifest remains unauthorized.")


if __name__ == "__main__":
    main()
