"""Fixed opened-cohort identity for the conditioning-interface diagnostic."""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import subprocess
import sys
from pathlib import Path

from neural_sat_speck.backend.artifacts import case_inputs, validate_cohort
from neural_sat_speck.backend.formula import build_formula
from neural_sat_speck.backend.guard import (
    THREAD_ENVIRONMENT,
    atomic_json,
    canonical_bytes,
    checked_path,
    read_checked_json,
    sha256_file,
)

ARMS = ("permanent_units", "temporary_assumptions")
COHORT_SHA256 = "a8bf7ccdf468f49a0ba67cb7cd7454a5912fec7f96c86d9b5f2a0e5878c49dc5"
PROFILE = {
    "seed": 999999,
    "role": "opened_demo_diagnostic",
    "cohort_keys_per_cipher": 8,
    "observations": 4,
    "trailing_keys": 1,
    "fresh_blocks": 2,
    "deadline_seconds": 300.0,
    "solver_threads": 1,
    "solver_verbose": 0,
    "solver_time_limit_argument": None,
    "repetitions": 1,
    "process_group_rss_limit_bytes": 2_684_354_560,
    "sample_interval_seconds": 10.0,
    "thread_environment": THREAD_ENVIRONMENT,
    "power_policy": "battery_allowed_author_instruction_20261005",
    "interruption_policy": "retain_without_retry",
}
SCIENTIFIC_OUTCOMES = frozenset(
    {
        "recovered",
        "sat_model_not_fresh_verified",
        "verified_different_key",
        "solver_unknown",
        "global_deadline",
    }
)


def digest(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def seconds(value: object) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Invalid elapsed time")
    return float(value)


def fixed_cells(mode: str) -> list[dict]:
    if mode not in {"pilot", "full"}:
        raise ValueError("Unknown campaign mode")
    cells = []
    for key_index in range(1 if mode == "pilot" else 8):
        ciphers = (("speck32_64", 7), ("simon32_64", 10))
        if key_index % 2:
            ciphers = ciphers[::-1]
        for cipher, rounds in ciphers:
            ident = f"{cipher}_k{key_index}_o4_m1"
            arms = ARMS if (key_index // 2) % 2 == 0 else ARMS[::-1]
            for arm in arms:
                cells.append(
                    {
                        "cell_id": f"{ident}__{arm}",
                        "case_id": ident,
                        "cipher": cipher,
                        "rounds": rounds,
                        "key_index": key_index,
                        "observations": 4,
                        "trailing_keys": 1,
                        "arm": arm,
                    }
                )
    return cells


def formula_identity(formula: dict) -> dict:
    return {
        name: formula[name]
        for name in (
            "num_vars",
            "cnf_clauses",
            "xor_clauses",
            "assumption_literals",
            "varmap",
        )
    }


def file_identity(path: Path) -> dict:
    return {"path": str(path.absolute()), "sha256": sha256_file(path)}


def source_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
    ).strip()


def validate_source_root(root: Path, source_names: list[str]) -> None:
    for directory, dirs, files in os.walk(root / "src", followlinks=False):
        if any((Path(directory) / name).is_symlink() for name in dirs + files):
            raise ValueError("Symlinked source tree cannot be frozen")
    top = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=root, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True
    )
    tracked = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
        .decode()
        .split("\0")
    )
    if Path(top).resolve() != root.resolve() or dirty.strip():
        raise ValueError("Source must be a clean committed git top-level")
    for name in source_names:
        path = root / name
        if name not in tracked or path.resolve() != path.absolute():
            raise ValueError("Untracked or symlinked source cannot be frozen")


def runtime_support_paths() -> dict[str, Path]:
    """The linked local CMS/GMP libraries and the preserved venv configuration."""
    return {
        "pyvenv_config": Path(sys.executable).parent.parent / "pyvenv.cfg",
        "cms_library": Path("/opt/homebrew/lib/libcryptominisat5.5.14.dylib"),
        "gmp_library": Path("/opt/homebrew/opt/gmp/lib/libgmp.10.dylib"),
        "gmpxx_library": Path("/opt/homebrew/opt/gmp/lib/libgmpxx.4.dylib"),
    }


def load_manifest(path: Path) -> tuple[dict, dict[str, Path]]:
    """Check policy and file hashes before opening the cohort or dispatching."""
    manifest = read_checked_json(path)
    if (
        manifest.get("schema") != "conditioning_control_manifest_v1"
        or manifest.get("status") != "APPROVED"
        or manifest.get("author_decision") != "D-054"
        or manifest.get("execution_authorized") is not True
        or digest(manifest.get("profile")) != digest(PROFILE)
    ):
        raise ValueError("Unapproved or drifted conditioning profile")
    cells = fixed_cells(manifest.get("mode"))
    if digest(manifest.get("cells")) != digest(cells) or manifest.get(
        "campaign_ceiling_seconds"
    ) != (1500.0 if manifest["mode"] == "pilot" else 10800.0):
        raise ValueError("Fixed grid or campaign cap differs")
    root = Path(manifest.get("source_root", ""))
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("Missing immutable source root")
    if source_commit(root) != manifest.get("source_commit"):
        raise ValueError("Source revision drift")
    pins = manifest.get("files")
    if (
        not isinstance(pins, dict)
        or not {"cohort", "python", "pycryptosat"} <= pins.keys()
    ):
        raise ValueError("Missing runtime/cohort pins")
    source_names = sorted(
        str(p.relative_to(root)) for p in (root / "src").rglob("*.py")
    ) + [
        "scripts/prepare_conditioning_interface_control.py",
        "scripts/evaluate_conditioning_interface_control.py",
        "scripts/analyze_conditioning_interface_control.py",
        "scripts/run_conditioning_interface_campaign.py",
        "docs/CONDITIONING_INTERFACE_PROTOCOL_20261006.md",
    ]
    validate_source_root(root, source_names)
    expected = {"source:" + name for name in source_names}
    if {name for name in pins if name.startswith("source:")} != expected:
        raise ValueError("Incomplete source hash inventory")
    paths = {
        name: checked_path(root, pin) for name, pin in pins.items() if name != "cohort"
    }
    for name, expected_path in runtime_support_paths().items():
        if paths.get(name) != expected_path.absolute():
            raise ValueError("Missing or redirected linked runtime/configuration pin")
    for name in source_names:
        if paths["source:" + name] != (root / name).absolute():
            raise ValueError("Source pin points outside source root")
    if pins["cohort"]["sha256"] != COHORT_SHA256:
        raise ValueError("Different opened cohort")
    if Path(sys.executable).absolute() != paths[
        "python"
    ] or sys.version != manifest.get("python_version"):
        raise ValueError("Interpreter identity differs")
    import pycryptosat

    spec = importlib.util.find_spec("pycryptosat")
    if (
        spec is None
        or Path(spec.origin).absolute() != paths["pycryptosat"]
        or getattr(pycryptosat, "__version__", None) != "5.14.7"
        or manifest.get("solver_version") != "5.14.7"
    ):
        raise ValueError("Solver extension identity differs")
    paths["cohort"] = checked_path(root, pins["cohort"])
    cases = manifest.get("cases")
    if not isinstance(cases, dict) or set(cases) != {c["case_id"] for c in cells}:
        raise ValueError("Case inventory differs")
    for ident, case in cases.items():
        for role in ("public", "truth"):
            paths[f"{ident}:{role}"] = checked_path(root, case[role])
        if (
            not isinstance(case.get("formula_sha256"), str)
            or len(case["formula_sha256"]) != 64
        ):
            raise ValueError("Incomplete formula identity")
    return manifest, paths


def verify_cases(manifest: dict, paths: dict[str, Path]) -> dict:
    """Independent reference reconstruction; no solver calls or new draws."""
    data = read_checked_json(paths["cohort"])
    validate_cohort(data)
    visited = set()
    for cell in manifest["cells"]:
        ident = cell["case_id"]
        if ident in visited:
            continue
        visited.add(ident)
        public, truth = case_inputs(data, cell)
        if (
            read_checked_json(paths[f"{ident}:public"]) != public
            or read_checked_json(paths[f"{ident}:truth"]) != truth
        ):
            raise ValueError("Exported case differs from opened reference")
        formula = build_formula(cell["cipher"], public, 4, 1, include_cnf=False)
        if (
            len(formula["assumption_literals"]) != 16
            or digest(formula_identity(formula))
            != manifest["cases"][ident]["formula_sha256"]
        ):
            raise ValueError("Base formula identity differs")
    return data


def prepare(
    root: Path, cohort: Path, output: Path, mode: str, pilot: Path | None
) -> Path:
    """Pin reviewed source and existing opened data; create no fresh key family."""
    if output.exists():
        raise ValueError("Preparation output already exists")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True
    )
    if dirty.strip():
        raise ValueError("Preparation source is not a clean committed snapshot")
    if sha256_file(cohort) != COHORT_SHA256:
        raise ValueError("Opened cohort checksum differs")
    data = read_checked_json(cohort)
    validate_cohort(data)
    cells = fixed_cells(mode)
    if mode == "full" and pilot is None:
        raise ValueError("Full preparation requires a separately passed pilot")
    output.mkdir(parents=True)
    cases = {}
    for cell in cells:
        ident = cell["case_id"]
        if ident in cases:
            continue
        public, truth = case_inputs(data, cell)
        public_path, truth_path = (
            output / f"{ident}_public.json",
            output / f"{ident}_truth.json",
        )
        atomic_json(public_path, public)
        atomic_json(truth_path, truth)
        formula = build_formula(cell["cipher"], public, 4, 1, include_cnf=False)
        cases[ident] = {
            "public": file_identity(public_path),
            "truth": file_identity(truth_path),
            "formula_sha256": digest(formula_identity(formula)),
        }
    import pycryptosat

    sources = sorted((root / "src").rglob("*.py")) + [
        root / name
        for name in (
            "scripts/prepare_conditioning_interface_control.py",
            "scripts/evaluate_conditioning_interface_control.py",
            "scripts/analyze_conditioning_interface_control.py",
            "scripts/run_conditioning_interface_campaign.py",
            "docs/CONDITIONING_INTERFACE_PROTOCOL_20261006.md",
        )
    ]
    manifest = {
        "schema": "conditioning_control_manifest_v1",
        "status": "APPROVED",
        "author_decision": "D-054",
        "execution_authorized": True,
        "mode": mode,
        "source_root": str(root.absolute()),
        "source_commit": source_commit(root),
        "profile": PROFILE,
        "cells": cells,
        "cases": cases,
        "campaign_ceiling_seconds": 1500.0 if mode == "pilot" else 10800.0,
        "python_version": sys.version,
        "solver_version": pycryptosat.__version__,
        "files": {
            "cohort": file_identity(cohort),
            "python": file_identity(Path(sys.executable)),
            "pycryptosat": file_identity(Path(pycryptosat.__file__)),
            **{
                name: file_identity(path)
                for name, path in runtime_support_paths().items()
            },
            **{"source:" + str(p.relative_to(root)): file_identity(p) for p in sources},
        },
        "pilot_ledger": file_identity(pilot) if pilot is not None else None,
    }
    target = output / "manifest.json"
    atomic_json(target, manifest)
    load_manifest(target)
    return target
