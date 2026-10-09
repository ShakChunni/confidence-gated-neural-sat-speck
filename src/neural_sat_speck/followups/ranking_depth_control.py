"""Frozen opened-data backend × ranking-depth profile and input separation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

from neural_sat_speck.backend.artifacts import case_inputs, validate_cohort
from neural_sat_speck.backend.formula import build_formula
from neural_sat_speck.backend.guard import atomic_json, read_checked_json
from neural_sat_speck.followups.conditioning_control import (
    digest,
    formula_identity,
    runtime_support_paths,
    source_commit,
    validate_source_root,
)
from neural_sat_speck.followups.conditioning_runner import controlled_environment
from neural_sat_speck.followups.environment import same_python_environment

DESIGN = "configs/protocol/backend_ranking_depth_proposed_20261006.json"
D056_DESIGN = "configs/protocol/backend_ranking_depth_d056_n32.json"
SCRIPTS = (
    "scripts/run_backend_ranking_depth.py",
    "scripts/analyze_backend_ranking_depth.py",
    "docs/CONTRIBUTION_REVIEW_AND_NEXT_STUDY_20261006.md",
    "docs/PROPOSAL_D056_EXPANDED_N32_CONTROLS.md",
    DESIGN,
    D056_DESIGN,
)
SCIENTIFIC_OUTCOMES = {
    "recovered",
    "queue_exhausted",
    "global_deadline",
    "solver_unknown",
    "verification_failure",
    "distinct_verified_key",
}

# Prospective queue-cost campaign (D-060 model test, D-061 classical comparator).
# The protocol is docs/PROSPECTIVE_QUEUE_COST_TEST_PROTOCOL_20261009.md, sections 4-9.
ASSET_KEYS = (
    "cms",
    "tf_python",
    "bao_vv9",
    "simon_ddt",
    "speck_stage1",
    "speck_classical",
)
PROSPECTIVE_ASSET_KEYS = ASSET_KEYS + ("speck_classical_r6",)
PROSPECTIVE_CAMPAIGN = "prospective_queue_cost_d060"
PROSPECTIVE_AUTHOR_DECISION = "D-060"
PROSPECTIVE_FULL_DESIGN = "configs/protocol/prospective_queue_cost_d060_n32.json"
PROSPECTIVE_PILOT_DESIGN = "configs/protocol/prospective_queue_cost_d060_pilot_n32.json"
PROSPECTIVE_DESIGNS = (PROSPECTIVE_FULL_DESIGN, PROSPECTIVE_PILOT_DESIGN)
PROSPECTIVE_SEEDS = {"full": 601001, "pilot": 501001}
PROSPECTIVE_STAGE1_OPTIONS = ("neural", "classical_sparse_r6")
PROSPECTIVE_DISPATCH_MARGIN_SECONDS = 308
PROSPECTIVE_NON_IDENTITY_FIELDS = frozenset(
    {
        "seed",
        "selected_instances",
        "files",
        "campaign_role",
        "status",
        "execution_authorized",
        "campaign_ceiling_seconds",
    }
)
_PROSPECTIVE_SPECK_ARMS = (
    ("neural", "cms_native_gauss", None),
    ("neural", "cms_native_no_gauss", None),
    ("neural", "cms_native_gauss", 1.5),
    ("neural", "cms_native_no_gauss", 1.5),
    ("classical_sparse_r6", "cms_native_gauss", None),
    ("classical_sparse_r6", "cms_native_no_gauss", None),
)
_PROSPECTIVE_SIMON_ARMS = (
    ("neural", "cms_native_gauss", None),
    ("neural", "cms_native_no_gauss", None),
    ("neural", "cms_native_gauss", 10.0),
    ("neural", "cms_native_no_gauss", 10.0),
)
_CAP_TAGS = {None: "none", 1.5: "1p5", 10.0: "10"}


def streaming_hash(path: Path) -> str:
    """The frozen SIMON table is 32 GiB; hashing must have bounded memory."""
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def stat_identity(path: Path) -> dict:
    s = path.stat()
    return {
        "device": s.st_dev,
        "inode": s.st_ino,
        "size": s.st_size,
        "mtime_ns": s.st_mtime_ns,
    }


def file_identity(path: Path) -> dict:
    before = stat_identity(path)
    result = {
        "path": str(path.absolute()),
        "sha256": streaming_hash(path),
        "identity": before,
    }
    if stat_identity(path) != before:
        raise ValueError("File changed during hashing")
    return result


def checked_path(root: Path, pin: dict) -> Path:
    if (
        not isinstance(pin, dict)
        or not isinstance(pin.get("path"), str)
        or not isinstance(pin.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", pin["sha256"]) is None
    ):
        raise ValueError("Malformed file identity")
    path = Path(pin["path"])
    if not path.is_absolute():
        path = root / path
    before = stat_identity(path)
    if (
        not path.is_file()
        or streaming_hash(path) != pin["sha256"]
        or stat_identity(path) != before
        or ("identity" in pin and pin["identity"] != before)
    ):
        raise ValueError("File hash drift: " + str(path))
    return path.absolute()


def optional_terminal_json(
    path: Path, deadline_proven: bool
) -> tuple[object | None, bool]:
    """A killed atomic writer's uncommitted file/sidecar is not evidence.

    Preserve the bytes and report component metadata as unavailable on a proven
    deadline. Completed results, the parent ledger and guardian stay strict.
    """
    if not path.exists():
        return None, False
    try:
        return read_checked_json(path), False
    except (ValueError, FileNotFoundError):
        if not deadline_proven:
            raise
        return None, True


def distribution_metadata_paths() -> list[str]:
    """Wheel installations use METADATA; editable egg-info uses PKG-INFO."""
    paths = []
    for distribution in importlib.metadata.distributions():
        directory = Path(distribution._path)
        path = directory / "METADATA"
        if not path.is_file():
            path = directory / "PKG-INFO"
        if not path.is_file():
            raise ValueError("Distribution lacks a metadata file: " + str(directory))
        paths.append(str(path))
    return sorted(paths)


def runtime_identity(tf_python: Path, root: Path | None = None) -> dict:
    """Identify the loaded numeric stack without models or key-dependent inputs.

    TensorFlow imports can expose setuptools' vendored distribution metadata.
    Enumerate after the worker's imports, so the freeze and receipt describe
    the same environment. Every discovered metadata file is still byte-pinned.
    """
    base = {n: importlib.metadata.version(n) for n in ("numpy", "torch")}
    if root is None:
        root = Path(__file__).resolve().parents[3]
    tmpdir = root / "runtime/ranking_depth_tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    probe_environment = {
        **controlled_environment(root, tmpdir),
        "TF_NUM_INTRAOP_THREADS": "8",
        "TF_NUM_INTEROP_THREADS": "1",
    }
    probe = subprocess.run(
        [
            str(tf_python),
            "-c",
            "import importlib.metadata as m,importlib.util as u,json,sys; from pathlib import Path; "
            "import tensorflow as tf; "
            "tf.config.threading.set_intra_op_parallelism_threads(8); "
            "tf.config.threading.set_inter_op_parallelism_threads(1); "
            "import h5py,keras,numpy; "
            "names=('numpy','tensorflow','keras','h5py'); "
            "print(json.dumps({'executable':sys.executable,'version':sys.version,"
            "'packages':{n:m.version(n) for n in names},'origins':{n:u.find_spec(n).origin for n in names},"
            "'installed_packages':{d.metadata['Name']:d.version for d in m.distributions()},"
            "'metadata_paths':sorted(str(Path(d._path) / ('METADATA' if (Path(d._path) / 'METADATA').is_file() else 'PKG-INFO')) for d in m.distributions()) }))",
        ],
        env=probe_environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    return {
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "packages": base,
            "origins": {n: importlib.util.find_spec(n).origin for n in base},
            "installed_packages": {
                d.metadata["Name"]: d.version
                for d in importlib.metadata.distributions()
            },
            "metadata_paths": distribution_metadata_paths(),
        },
        "tensorflow": json.loads(probe.stdout),
    }


def runtime_files(runtime: dict, tf_python: Path) -> dict[str, Path]:
    files = {"tf_pyvenv_config": tf_python.parent.parent / "pyvenv.cfg"}
    for environment, record in runtime.items():
        for i, p in enumerate(sorted(record["metadata_paths"])):
            files[f"distribution_metadata:{environment}:{i}"] = Path(p)
        for name, origin in record["origins"].items():
            package_root = Path(origin).parent
            files[f"package:{environment}:{name}:init"] = Path(origin)
            # Pin the core compiled implementations actually used for inference.
            for pattern in (
                ("_C*.so", "lib/*.dylib")
                if name == "torch"
                else ("python/_pywrap_tensorflow_internal.so", "*.dylib")
                if name == "tensorflow"
                else ("_core/*multiarray*.so", "core/*multiarray*.so")
                if name == "numpy"
                else ("*.so",)
            ):
                for p in sorted(package_root.glob(pattern)):
                    files[
                        f"package:{environment}:{name}:{p.relative_to(package_root)}"
                    ] = p
    return files


def fixed_cells(mode: str, num_keys: int = 8) -> list[dict]:
    """Rotate all four arms within each key; alternate the leading cipher."""
    if mode not in {"pilot", "full"}:
        raise ValueError("Mode must be pilot or full")
    arms = [
        (1, "cms_native_gauss"),
        (2, "cms_native_gauss"),
        (1, "cms_native_no_gauss"),
        (2, "cms_native_no_gauss"),
    ]
    cells = []
    for index in range(1 if mode == "pilot" else num_keys):
        ciphers = ["speck32_64", "simon32_64"]
        if index % 2:
            ciphers.reverse()
        for cipher in ciphers:
            offset = (index + (cipher == "simon32_64")) % 4
            for trailing, backend in arms[offset:] + arms[:offset]:
                case = f"{cipher}_{index:02d}"
                cells.append(
                    {
                        "case_id": case,
                        "cipher": cipher,
                        "key_index": index,
                        "rounds": 7 if cipher == "speck32_64" else 10,
                        "trailing_keys": trailing,
                        "profile": backend,
                        "cell_id": f"{case}_m{trailing}_{backend}",
                    }
                )
    return cells


def _prospective_cell_id(
    case: str, stage1: str, profile: str, cap: float | None
) -> str:
    if stage1 == "classical_sparse_r6":
        return f"{case}_m2_classical_{profile}_capnone"
    return f"{case}_m2_{profile}_cap{_CAP_TAGS[cap]}"


def prospective_cells(mode: str, num_keys: int = 32) -> list[dict]:
    """Ten arms per key (SPECK six, SIMON four); 320 cells for 32 keys.

    Pilot mode runs one opened D-056 key per cipher (key index 0) across all
    applicable arms: 10 cells. Rotation mirrors ``fixed_cells``: the arm list
    rotates by key and cipher, and the leading cipher alternates by key.
    """
    if mode not in {"pilot", "full"}:
        raise ValueError("Mode must be pilot or full")
    cells = []
    for index in range(1 if mode == "pilot" else num_keys):
        ciphers = ["speck32_64", "simon32_64"]
        if index % 2:
            ciphers.reverse()
        for cipher in ciphers:
            arms = (
                _PROSPECTIVE_SPECK_ARMS
                if cipher == "speck32_64"
                else _PROSPECTIVE_SIMON_ARMS
            )
            offset = (index + (cipher == "simon32_64")) % len(arms)
            case = f"{cipher}_{index:02d}"
            for stage1, profile, cap in arms[offset:] + arms[:offset]:
                cells.append(
                    {
                        "case_id": case,
                        "cipher": cipher,
                        "key_index": index,
                        "rounds": 7 if cipher == "speck32_64" else 10,
                        "trailing_keys": 2,
                        "profile": profile,
                        "cell_id": _prospective_cell_id(case, stage1, profile, cap),
                        "candidate_cap_seconds": cap,
                        "stage1": stage1,
                    }
                )
    return cells


def validate_prospective_cell(cell: dict) -> None:
    """Exact field, arm and identifier checks for one prospective cell."""
    expected_fields = {
        "case_id",
        "cipher",
        "key_index",
        "rounds",
        "trailing_keys",
        "profile",
        "cell_id",
        "candidate_cap_seconds",
        "stage1",
    }
    if not isinstance(cell, dict) or set(cell) != expected_fields:
        raise ValueError("Prospective cell has unexpected fields")
    cipher = cell["cipher"]
    if cipher not in ("speck32_64", "simon32_64"):
        raise ValueError("Prospective cell cipher is outside the campaign")
    index = cell["key_index"]
    if type(index) is not int or not 0 <= index < 32:
        raise ValueError("Prospective cell key index is outside the cohort")
    if cell["case_id"] != f"{cipher}_{index:02d}":
        raise ValueError("Prospective cell case identifier differs")
    if cell["rounds"] != (7 if cipher == "speck32_64" else 10):
        raise ValueError("Prospective cell rounds differ from the campaign")
    if type(cell["trailing_keys"]) is not int or cell["trailing_keys"] != 2:
        raise ValueError("Prospective cells are two-key cells only")
    stage1, profile, cap = (
        cell["stage1"],
        cell["profile"],
        cell["candidate_cap_seconds"],
    )
    if stage1 not in PROSPECTIVE_STAGE1_OPTIONS:
        raise ValueError("Unknown Stage 1 option")
    if stage1 == "classical_sparse_r6" and cipher != "speck32_64":
        raise ValueError("Classical Stage 1 is a SPECK comparator only")
    if cap is not None and (type(cap) is not float or not 0 < cap < 300):
        raise ValueError("Candidate cap must be None or a float in (0, 300)")
    arms = (
        _PROSPECTIVE_SPECK_ARMS if cipher == "speck32_64" else _PROSPECTIVE_SIMON_ARMS
    )
    if (stage1, profile, cap) not in arms:
        raise ValueError("Prospective cell arm is not in the campaign")
    if cell["cell_id"] != _prospective_cell_id(cell["case_id"], stage1, profile, cap):
        raise ValueError("Prospective cell identifier differs from its arm")


def asset_keys_for(manifest: dict) -> tuple[str, ...]:
    """Prospective manifests add the pinned six-round Stage 1 table."""
    if manifest.get("campaign") == PROSPECTIVE_CAMPAIGN:
        return PROSPECTIVE_ASSET_KEYS
    return ASSET_KEYS


def prospective_source_names(root: Path) -> list[str]:
    """Source pins plus both prospective designs (D-056 source names unchanged)."""
    return source_names(root) + list(PROSPECTIVE_DESIGNS)


def names_for_manifest(manifest: dict, root: Path) -> list[str]:
    if manifest.get("campaign") == PROSPECTIVE_CAMPAIGN:
        return prospective_source_names(root)
    return source_names(root)


def prospective_identity(manifest: dict) -> dict:
    """Code, configuration and profile identity shared by pilot and full runs.

    The cohort pin, seed, selected keys, role and ceiling are excluded: the
    pilot runs on opened D-056 keys and the full run on the fresh cohort.
    """
    profile = manifest["profile"]
    return {
        "campaign": manifest.get("campaign"),
        "author_decision": manifest.get("author_decision"),
        "profile": digest(
            {
                k: v
                for k, v in profile.items()
                if k not in PROSPECTIVE_NON_IDENTITY_FIELDS
            }
        ),
        "profile_files": digest(
            {k: v for k, v in profile["files"].items() if k != "cohort"}
        ),
        "manifest_files": digest(
            {k: v for k, v in manifest["files"].items() if k != "cohort"}
        ),
        "runtime": digest(manifest["runtime"]),
        "source_commit": manifest["source_commit"],
        "python_version": manifest["python_version"],
    }


def source_names(root: Path) -> list[str]:
    return sorted(
        str(p.relative_to(root)) for p in (root / "src").rglob("*.py")
    ) + list(SCRIPTS)


def policy(root: Path, design_rel: str | Path | None = None) -> dict:
    """The reviewed proposal remains a nonexecutable reference, unchanged."""
    rel = Path(design_rel) if design_rel is not None else Path(DESIGN)
    if not rel.is_absolute():
        rel = root / rel
    return json.loads(rel.read_bytes())


def prepare(
    root: Path,
    output: Path,
    mode: str,
    tf_python: Path,
    asset_root: Path,
    pilot: Path | None = None,
    config_path: Path | None = None,
    author_approval: str | None = None,
) -> Path:
    if output.exists():
        raise ValueError("Refusing to overwrite preparation")
    root, asset_root = root.resolve(), asset_root.resolve()
    names = source_names(root)
    validate_source_root(root, names)
    profile = policy(root, config_path)
    if profile.get("campaign") == PROSPECTIVE_CAMPAIGN:
        return _prepare_prospective(
            root,
            output,
            mode,
            tf_python,
            asset_root,
            pilot,
            config_path,
            profile,
            author_approval,
        )
    if author_approval is not None:
        raise ValueError(
            "Author approval flag applies only to the prospective campaign"
        )
    if profile.get("confirmation_seed") is not None:
        raise ValueError("Phase 12 confirmation seed is strictly prohibited")
    seed = profile.get("seed")
    if seed not in (999999, 501001):
        raise ValueError("Only the opened demo family or authorized D-056 dev family is eligible")
    author_decision = "D-056" if seed == 501001 else "D-055"
    if mode == "full" and pilot is None:
        raise ValueError("Full preparation needs an independently audited pilot")
    num_keys = len(profile.get("selected_instances", list(range(8 if seed == 999999 else 32))))
    cells = fixed_cells(mode, num_keys=num_keys)
    files = {}
    for name, pin in profile["files"].items():
        p = Path(pin["path"])
        if not p.is_absolute():
            p = asset_root / p
        files[name] = {"path": str(p.absolute()), "sha256": pin["sha256"]}
        files[name]["identity"] = stat_identity(p)
    files.update(
        python=file_identity(Path(sys.executable)),
        tf_python=file_identity(tf_python.absolute()),
    )
    runtime = runtime_identity(tf_python, root)
    if runtime["tensorflow"]["packages"]["tensorflow"] != "2.16.2":
        raise ValueError("TensorFlow environment differs from the frozen comparator")
    files.update(
        {n: file_identity(p) for n, p in runtime_files(runtime, tf_python).items()}
    )
    files.update(
        {
            name: file_identity(p)
            for name, p in runtime_support_paths().items()
            if name != "pycryptosat"
        }
    )
    files.update({"source:" + n: file_identity(root / n) for n in names})
    # Check every pin before consuming even this already-opened cohort.
    paths = {name: checked_path(root, pin) for name, pin in files.items()}
    expected_cohort_sha = profile["files"]["cohort"]["sha256"]
    if files["cohort"]["sha256"] != expected_cohort_sha:
        raise ValueError("Wrong cohort SHA-256")
    data = read_checked_json(paths["cohort"])
    validate_cohort(data)
    output.mkdir(parents=True)
    cases = {}
    for cell in cells:
        ident = cell["case_id"]
        if ident in cases:
            continue
        # No true trailing key is included in attack inputs.
        public, truth = case_inputs(
            data, {**cell, "observations": 128, "trailing_keys": 0}
        )
        verification = {"verification_pairs": truth["verification_pairs"]}
        records = {}
        for role, value in (
            ("public", public),
            ("verification", verification),
            ("truth", truth),
        ):
            p = output / f"{ident}_{role}.json"
            atomic_json(p, value)
            records[role] = file_identity(p)
        formula = build_formula(cell["cipher"], public, 4, 0, include_cnf=False)
        records["base_formula_sha256"] = digest(formula_identity(formula))
        cases[ident] = records
    ceiling = 3600.0 if mode == "pilot" else float(profile.get("campaign_ceiling_seconds", 25200.0 if author_decision == "D-055" else 90000.0))
    manifest = {
        "schema": "backend_ranking_depth_manifest_v1",
        "status": "APPROVED",
        "author_decision": author_decision,
        "execution_authorized": True,
        "mode": mode,
        "profile": profile,
        "source_root": str(root),
        "source_commit": source_commit(root),
        "python_version": sys.version,
        "runtime": runtime,
        "runtime_binding_scope": "complete installed-version lock and distribution metadata; selected inference/core binary and entry-module hashes; transitive package code is not exhaustively byte-pinned",
        "files": files,
        "cases": cases,
        "cells": cells,
        "campaign_ceiling_seconds": ceiling,
        "pilot_audit": file_identity(pilot) if pilot else None,
    }
    path = output / "manifest.json"
    atomic_json(path, manifest)
    return path


def _prepare_prospective(
    root: Path,
    output: Path,
    mode: str,
    tf_python: Path,
    asset_root: Path,
    pilot: Path | None,
    config_path: Path | None,
    profile: dict,
    author_approval: str | None,
) -> Path:
    """Prepare the D-060/D-061 campaign. Approval is an explicit flag, never a default."""
    names = prospective_source_names(root)
    validate_source_root(root, names)
    if config_path is None:
        raise ValueError("Prospective preparation needs an explicit design file")
    design = Path(config_path)
    if not design.is_absolute():
        design = root / design
    design_rel = design.resolve().relative_to(root).as_posix()
    if design_rel not in PROSPECTIVE_DESIGNS:
        raise ValueError("Unregistered prospective design file")
    if mode not in PROSPECTIVE_SEEDS:
        raise ValueError("Mode must be pilot or full")
    if profile.get("campaign_role") != mode:
        raise ValueError("Design role differs from the requested mode")
    if (mode == "full") != (pilot is not None):
        raise ValueError(
            "Full preparation needs the pinned pilot audit; pilot preparation takes none"
        )
    if profile.get("confirmation_seed") is not None:
        raise ValueError("Phase 12 confirmation seed is strictly prohibited")
    if profile.get("seed") != PROSPECTIVE_SEEDS[mode]:
        raise ValueError("Seed family differs from the declared campaign role")
    if profile.get("author_decision") != PROSPECTIVE_AUTHOR_DECISION:
        raise ValueError("Prospective design carries the wrong decision label")
    if (
        profile.get("status") == "APPROVED"
        or profile.get("execution_authorized") is not False
    ):
        raise ValueError(
            "Designs ship unapproved; approval comes only from --author-approval"
        )
    if author_approval not in (None, PROSPECTIVE_AUTHOR_DECISION):
        raise ValueError("Author approval label must be exactly D-060")
    approved = author_approval == PROSPECTIVE_AUTHOR_DECISION
    selected = profile.get("selected_instances")
    if mode == "pilot" and selected != [0]:
        raise ValueError("Pilot covers exactly one opened key per cipher (index 0)")
    cells = prospective_cells(mode, num_keys=len(selected or []))
    for cell in cells:
        validate_prospective_cell(cell)
    ceiling = float(profile["campaign_ceiling_seconds"])
    if ceiling < len(cells) * PROSPECTIVE_DISPATCH_MARGIN_SECONDS:
        raise ValueError("Campaign ceiling cannot cover the cell count")
    if profile["files"]["cohort"]["sha256"] is None:
        raise ValueError(
            "Cohort SHA-256 is not pinned: generate and pin the fresh cohort after author authorization"
        )
    root, asset_root = root.resolve(), asset_root.resolve()
    files = {}
    for name, pin in profile["files"].items():
        p = Path(pin["path"])
        if not p.is_absolute():
            p = asset_root / p
        files[name] = {"path": str(p.absolute()), "sha256": pin["sha256"]}
        files[name]["identity"] = stat_identity(p)
    files.update(
        python=file_identity(Path(sys.executable)),
        tf_python=file_identity(tf_python.absolute()),
    )
    runtime = runtime_identity(tf_python, root)
    if runtime["tensorflow"]["packages"]["tensorflow"] != "2.16.2":
        raise ValueError("TensorFlow environment differs from the frozen comparator")
    files.update(
        {n: file_identity(p) for n, p in runtime_files(runtime, tf_python).items()}
    )
    files.update(
        {
            name: file_identity(p)
            for name, p in runtime_support_paths().items()
            if name != "pycryptosat"
        }
    )
    files.update({"source:" + n: file_identity(root / n) for n in names})
    paths = {name: checked_path(root, pin) for name, pin in files.items()}
    if files["cohort"]["sha256"] != profile["files"]["cohort"]["sha256"]:
        raise ValueError("Wrong cohort SHA-256")
    data = read_checked_json(paths["cohort"])
    validate_cohort(data)
    output.mkdir(parents=True)
    cases = {}
    for cell in cells:
        ident = cell["case_id"]
        if ident in cases:
            continue
        public, truth = case_inputs(
            data, {**cell, "observations": 128, "trailing_keys": 0}
        )
        verification = {"verification_pairs": truth["verification_pairs"]}
        records = {}
        for role, value in (
            ("public", public),
            ("verification", verification),
            ("truth", truth),
        ):
            p = output / f"{ident}_{role}.json"
            atomic_json(p, value)
            records[role] = file_identity(p)
        formula = build_formula(cell["cipher"], public, 4, 0, include_cnf=False)
        records["base_formula_sha256"] = digest(formula_identity(formula))
        cases[ident] = records
    manifest = {
        "schema": "backend_ranking_depth_manifest_v1",
        "status": "APPROVED" if approved else "NOT_APPROVED",
        "author_decision": PROSPECTIVE_AUTHOR_DECISION,
        "execution_authorized": approved,
        "campaign": PROSPECTIVE_CAMPAIGN,
        "campaign_role": mode,
        "design_path": design_rel,
        "mode": mode,
        "profile": profile,
        "source_root": str(root),
        "source_commit": source_commit(root),
        "python_version": sys.version,
        "runtime": runtime,
        "runtime_binding_scope": "complete installed-version lock and distribution metadata; selected inference/core binary and entry-module hashes; transitive package code is not exhaustively byte-pinned",
        "files": files,
        "cases": cases,
        "cells": cells,
        "campaign_ceiling_seconds": ceiling,
        "pilot_audit": file_identity(pilot) if pilot else None,
    }
    path = output / "manifest.json"
    atomic_json(path, manifest)
    return path


def load_manifest(path: Path, *, hashes: bool = True) -> tuple[dict, dict[str, Path]]:
    manifest = read_checked_json(path)
    root = Path(manifest.get("source_root", ""))
    prospective = manifest.get("campaign") == PROSPECTIVE_CAMPAIGN
    labels = (PROSPECTIVE_AUTHOR_DECISION,) if prospective else ("D-055", "D-056")
    if (
        not root.is_absolute()
        or manifest.get("schema") != "backend_ranking_depth_manifest_v1"
        or manifest.get("status") != "APPROVED"
        or manifest.get("author_decision") not in labels
        or manifest.get("execution_authorized") is not True
    ):
        raise ValueError("Unapproved ranking-depth execution")
    author_decision = manifest.get("author_decision")
    profile = manifest.get("profile", {})
    mode = manifest.get("mode")
    if prospective:
        if manifest.get("design_path") not in PROSPECTIVE_DESIGNS:
            raise ValueError("Unregistered prospective design")
        expected_profile = policy(root, manifest["design_path"])
        if (
            expected_profile.get("campaign_role") != mode
            or manifest.get("campaign_role") != mode
        ):
            raise ValueError("Campaign role differs from the manifest mode")
    else:
        expected_profile = policy(
            root, D056_DESIGN if author_decision == "D-056" else DESIGN
        )
    if digest(profile) != digest(expected_profile):
        raise ValueError("Ranking-depth scientific profile drift")
    num_keys = len(profile.get("selected_instances", list(range(8 if author_decision == "D-055" else 32))))
    if prospective:
        expected_cells = prospective_cells(mode, num_keys=num_keys)
        for cell in manifest.get("cells", []):
            validate_prospective_cell(cell)
        expected_ceiling = float(expected_profile["campaign_ceiling_seconds"])
    else:
        expected_cells = fixed_cells(mode, num_keys=num_keys)
        expected_ceiling = (
            3600.0
            if mode == "pilot"
            else float(
                expected_profile.get(
                    "campaign_ceiling_seconds",
                    25200.0 if author_decision == "D-055" else 90000.0,
                )
            )
        )
    if (
        manifest.get("cells") != expected_cells
        or manifest.get("campaign_ceiling_seconds") != expected_ceiling
    ):
        raise ValueError("Grid or campaign ceiling drift")
    names = prospective_source_names(root) if prospective else source_names(root)
    validate_source_root(root, names)
    if source_commit(root) != manifest.get("source_commit"):
        raise ValueError("Source revision drift")
    pins = manifest.get("files", {})
    expected_sources = {"source:" + n for n in names}
    minimum = (
        set(profile["files"])
        | {"python", "tf_python"}
        | set(runtime_support_paths())
    )
    if not isinstance(pins, dict) or not expected_sources | minimum <= pins.keys():
        raise ValueError("Incomplete runtime/source inventory")
    paths = {}
    for name, pin in pins.items():
        if hashes:
            paths[name] = checked_path(root, pin)
        else:
            p = Path(pin["path"])
            paths[name] = p if p.is_absolute() else root / p
    # Never invoke the TF executable before checking its pinned bytes.
    observed_runtime = runtime_identity(paths["tf_python"], root)
    if digest(observed_runtime) != digest(manifest.get("runtime")):
        raise ValueError("Imported numeric/inference runtime identity differs")
    compiled = runtime_files(observed_runtime, paths["tf_python"])
    if set(pins) != expected_sources | minimum | compiled.keys():
        raise ValueError("Incomplete runtime/source inventory")
    for name, expected_path in compiled.items():
        if paths[name] != expected_path.absolute():
            raise ValueError("Compiled runtime identity redirected")
    for name, pin in profile["files"].items():
        if pins[name]["sha256"] != pin["sha256"]:
            raise ValueError("Pinned scientific asset drift")
    for n in names:
        if paths["source:" + n] != (root / n).absolute():
            raise ValueError("Source identity redirected")
    for name, p in runtime_support_paths().items():
        if paths[name] != p.absolute():
            raise ValueError("Linked runtime/configuration drift")
    if not same_python_environment(
        sys.executable, paths["python"]
    ) or sys.version != manifest.get("python_version"):
        raise ValueError("Interpreter identity differs")
    if set(manifest.get("cases", {})) != {c["case_id"] for c in manifest["cells"]}:
        raise ValueError("Case inventory mismatch")
    for ident, case in manifest["cases"].items():
        for role in ("public", "verification", "truth"):
            paths[ident + ":" + role] = checked_path(root, case[role])
    return manifest, paths


def verify_cases(manifest: dict, paths: dict[str, Path]) -> None:
    """Replay already-opened keys; attack/fresh/key files must match the cohort."""
    data = read_checked_json(paths["cohort"])
    validate_cohort(data)
    for ident, case in manifest["cases"].items():
        cell = next(c for c in manifest["cells"] if c["case_id"] == ident)
        public, truth = case_inputs(
            data, {**cell, "observations": 128, "trailing_keys": 0}
        )
        if (
            read_checked_json(paths[ident + ":public"]) != public
            or read_checked_json(paths[ident + ":truth"]) != truth
            or read_checked_json(paths[ident + ":verification"])
            != {"verification_pairs": truth["verification_pairs"]}
            or digest(
                formula_identity(
                    build_formula(cell["cipher"], public, 4, 0, include_cnf=False)
                )
            )
            != case["base_formula_sha256"]
        ):
            raise ValueError(
                "Prepared public/fresh/truth/formula differs from opened cohort"
            )
