#!/usr/bin/env python3
"""Package checked existing evidence; verify/regenerate without running attacks.

See docs/WRITING_HANDOFF_PREPARATION_NOTE_20261004.md. This script copies only
explicitly pinned completed artifacts and committed project source. It never
loads a ranker, calls a solver, draws data or changes an original result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def load_json(path: Path) -> dict:
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"Duplicate JSON key: {key}")
            value[key] = item
        return value

    value = json.loads(path.read_bytes(), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def inside(root: Path, relative: str) -> Path:
    """Accept canonical portable relative paths, rejecting links and escapes."""
    p = PurePosixPath(relative)
    if (
        not relative
        or not p.parts
        or p.is_absolute()
        or ".." in p.parts
        or "\\" in relative
        or p.as_posix() != relative
    ):
        raise ValueError(f"Nonportable path: {relative}")
    path = root
    for part in p.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Symlink is not an artifact: {relative}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes bundle: {relative}")
    return path


def read_pinned(root: Path, entry: dict) -> bytes:
    data = inside(root, entry["path"]).read_bytes()
    if digest(data) != entry["sha256"] or (
        "bytes" in entry and len(data) != entry["bytes"]
    ):
        raise ValueError(f"Pinned source changed: {entry['path']}")
    return data


def committed_source(repo: Path) -> tuple[str, dict[str, bytes]]:
    """Snapshot selected tracked code, rejecting uncommitted source drift."""

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args])

    commit = git("rev-parse", "HEAD").decode().strip()
    if git("status", "--porcelain", "--untracked-files=all", "--", "src"):
        raise ValueError("Uncommitted or untracked project source drift")
    names = git("ls-tree", "-rz", "--name-only", commit).decode().split("\0")[:-1]
    if "LICENSE" not in names:
        raise ValueError("Committed root LICENSE is missing")
    selected = [
        name
        for name in names
        if (name.startswith("src/") and name.endswith(".py"))
        or name
        in {
            "LICENSE",
            "pyproject.toml",
            "requirements.txt",
            "README.md",
            "scripts/package_publication_evidence.py",
            "scripts/audit_backend_completed_evidence.py",
            "tests/unit/test_publication_evidence_package.py",
            "tests/unit/test_backend_completed_evidence.py",
        }
    ]
    if "scripts/package_publication_evidence.py" not in selected:
        raise ValueError("Commit packaging source before making a release")
    files = {}
    for name in selected:
        data = git("show", f"{commit}:{name}")
        try:
            working_data = inside(repo, name).read_bytes()
        except OSError as exc:
            if name == "LICENSE":
                raise ValueError(
                    "Committed root LICENSE is missing from working tree"
                ) from exc
            raise
        if data != working_data:
            raise ValueError(f"Uncommitted source drift: {name}")
        files[name] = data
    return commit, files


def build(repo: Path, catalog_path: Path, output: Path) -> dict:
    """Create a new package; never overwrite an existing destination."""
    repo = repo.resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite an evidence package")
    catalog = load_json(catalog_path)
    if catalog.get("schema") != 1 or not catalog.get("files"):
        raise ValueError("Invalid source catalog")
    entries = catalog["files"]
    if len({e["path"] for e in entries}) != len(entries):
        raise ValueError("Duplicate source paths")
    source_commit, source = committed_source(repo)
    pinned = {e["path"]: e for e in entries}
    guide = "docs/PORTABLE_EVIDENCE_REPRODUCTION_20261004.md"
    if any(
        name not in pinned
        for name in (catalog["formula_catalog"], catalog["cohort"], guide)
    ):
        raise ValueError("Unpinned formula catalog/cohort/reproduction guide")
    if "LICENSE" not in source:
        raise ValueError("Committed source omitted root LICENSE")
    original = json.loads(read_pinned(repo, pinned[catalog["formula_catalog"]]))
    if len(original) != 32:
        raise ValueError("Expected the completed 32-case diagnostic artifact")
    # Validate before creating output; originals remain read-only throughout.
    payload = {f"evidence/{e['path']}": read_pinned(repo, e) for e in entries}
    payload.update({f"source/{name}": data for name, data in source.items()})
    payload["LICENSE"] = source["LICENSE"]
    payload["source_catalog.json"] = catalog_path.read_bytes()
    portable = {}
    formula_root = inside(repo, catalog["formula_catalog"]).parent
    for case, entry in original.items():
        if case != entry["case_id"] or set(entry["files"]) != {
            "native",
            "cnf",
            "public",
            "truth",
            "varmap",
        }:
            raise ValueError("Invalid formula case/file matrix")
        copied = json.loads(json.dumps(entry))
        for kind, item in entry["files"].items():
            src = Path(item["path"])
            if src.is_symlink() or src.parent.resolve() != formula_root.resolve():
                raise ValueError("Formula source outside pinned artifact directory")
            data = src.read_bytes()
            if digest(data) != item["sha256"]:
                raise ValueError(f"Formula hash mismatch: {case}:{kind}")
            prefix = "verification_truth" if kind == "truth" else "formulas"
            rel = f"{prefix}/{src.name}"
            if rel in payload:
                raise ValueError("Colliding packaged file")
            payload[rel] = data
            sidecar = src.with_name(src.name + ".sha256")
            expected = f"{item['sha256']}  {src.name}\n".encode()
            if sidecar.is_symlink() or sidecar.read_bytes() != expected:
                raise ValueError(f"Formula sidecar mismatch: {case}:{kind}")
            payload[rel + ".sha256"] = expected
            copied["files"][kind]["path"] = rel
        portable[case] = copied
    payload["formula_catalog_portable.json"] = encode(portable)
    payload["REPRODUCE.md"] = payload[f"evidence/{guide}"]
    manifest = {
        "schema": 1,
        "source_commit": source_commit,
        "source_catalog_sha256": digest(payload["source_catalog.json"]),
        "cohort_path": f"evidence/{catalog['cohort']}",
        "original_formula_catalog_path": f"evidence/{catalog['formula_catalog']}",
        "license_provenance": {
            "source_path": "source/LICENSE",
            "package_path": "LICENSE",
            "sha256": digest(source["LICENSE"]),
            "bytes": len(source["LICENSE"]),
        },
        "limitations": catalog["limitations"],
        "files": {
            name: {"sha256": digest(data), "bytes": len(data)}
            for name, data in sorted(payload.items())
        },
    }
    try:
        current_license = inside(repo, "LICENSE").read_bytes()
    except OSError as exc:
        raise ValueError("Root LICENSE disappeared during package validation") from exc
    if current_license != source["LICENSE"]:
        raise ValueError("Root LICENSE changed during package validation")
    output.mkdir(parents=True)
    for name, data in payload.items():
        path = inside(output, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    encoded = encode(manifest)
    (output / "manifest.json").write_bytes(encoded)
    (output / "manifest.json.sha256").write_text(f"{digest(encoded)}  manifest.json\n")
    # Do not silently accept a source that changed while it was being copied.
    for entry in entries:
        read_pinned(repo, entry)
    return verify(output)


def verify(bundle: Path, expected_manifest_sha256: str | None = None) -> dict:
    """Check every listed byte/path and the original-to-portable formula map."""
    bundle = bundle.resolve()
    encoded = inside(bundle, "manifest.json").read_bytes()
    sha = digest(encoded)
    sidecar = inside(bundle, "manifest.json.sha256").read_text().split()
    if sidecar != [sha, "manifest.json"] or (
        expected_manifest_sha256 is not None and sha != expected_manifest_sha256
    ):
        raise ValueError("Package manifest checksum mismatch")
    manifest = load_json(bundle / "manifest.json")
    if manifest.get("schema") != 1 or not manifest.get("files"):
        raise ValueError("Invalid package manifest")
    listed = set(manifest["files"]) | {"manifest.json", "manifest.json.sha256"}
    actual = set()
    for path in bundle.rglob("*"):
        if path.is_symlink():
            raise ValueError("Symlink inside package")
        if path.is_file():
            actual.add(path.relative_to(bundle).as_posix())
    if actual != listed:
        raise ValueError("Missing or unexpected package files")
    for name, item in manifest["files"].items():
        if set(item) != {"sha256", "bytes"}:
            raise ValueError("Unexpected package file metadata")
        read_pinned(bundle, {**item, "path": name})
    source = load_json(bundle / "source_catalog.json")
    if (
        digest((bundle / "source_catalog.json").read_bytes())
        != manifest["source_catalog_sha256"]
    ):
        raise ValueError("Source catalog mismatch")
    for entry in source["files"]:
        read_pinned(bundle, {**entry, "path": f"evidence/{entry['path']}"})
    if "license_provenance" in manifest:
        provenance = manifest["license_provenance"]
        if (
            not isinstance(provenance, dict)
            or set(provenance) != {"source_path", "package_path", "sha256", "bytes"}
            or provenance.get("source_path") != "source/LICENSE"
            or provenance.get("package_path") != "LICENSE"
            or not isinstance(provenance.get("sha256"), str)
            or len(provenance["sha256"]) != 64
            or type(provenance.get("bytes")) is not int
            or provenance["bytes"] < 0
        ):
            raise ValueError("Invalid license provenance")
        expected = {
            "sha256": provenance["sha256"],
            "bytes": provenance["bytes"],
        }
        license_bytes = []
        for name in (provenance["source_path"], provenance["package_path"]):
            if manifest["files"].get(name) != expected:
                raise ValueError(
                    "License provenance differs from package file manifest"
                )
            license_bytes.append(read_pinned(bundle, {"path": name, **expected}))
        if license_bytes[0] != license_bytes[1]:
            raise ValueError("Packaged root/source LICENSE copies differ")
    if (
        manifest["cohort_path"] != f"evidence/{source['cohort']}"
        or manifest["original_formula_catalog_path"]
        != f"evidence/{source['formula_catalog']}"
    ):
        raise ValueError("Package source path binding mismatch")
    original = load_json(inside(bundle, manifest["original_formula_catalog_path"]))
    portable = load_json(bundle / "formula_catalog_portable.json")
    if set(portable) != set(original) or len(portable) != 32:
        raise ValueError("Portable formula case matrix mismatch")
    for case, entry in original.items():
        transformed = json.loads(json.dumps(entry))
        for kind, item in entry["files"].items():
            prefix = "verification_truth" if kind == "truth" else "formulas"
            rel = f"{prefix}/{Path(item['path']).name}"
            transformed["files"][kind]["path"] = rel
            read_pinned(bundle, {"path": rel, "sha256": item["sha256"]})
            if inside(bundle, rel + ".sha256").read_text() != (
                f"{item['sha256']}  {Path(item['path']).name}\n"
            ):
                raise ValueError("Portable formula sidecar mismatch")
        if portable[case] != transformed:
            raise ValueError("Portable metadata altered more than file paths")
    return {
        "files_checked": len(manifest["files"]),
        "formula_cases": len(portable),
        "manifest_sha256": sha,
        "source_commit": manifest["source_commit"],
    }


def mount_scalar_simon(bundle: Path) -> None:
    """Load scalar leaves without training convenience re-exports in __init__.

    The original SIMON initializer eagerly imports dataset/training interfaces.
    A namespace mounted at the sealed directory avoids executing those exports;
    cipher/schedule/encoder leaf source is still imported verbatim and hash-bound.
    """
    import importlib.machinery
    import importlib.util

    import neural_sat_speck

    name = "neural_sat_speck.simon"
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = [
        str(bundle / "source/src/neural_sat_speck/simon")
    ]
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    neural_sat_speck.simon = module


def regenerate(bundle: Path) -> dict:
    """Recreate formulas and reference witnesses using bundled source, no SAT."""
    result = verify(bundle)
    bundle = bundle.resolve()
    # Refuse accidentally imported project code from an installed/parent checkout.
    if any(
        n == "neural_sat_speck" or n.startswith("neural_sat_speck.")
        for n in sys.modules
    ):
        raise ValueError("Project modules already loaded; run in a fresh process")
    sys.path.insert(0, str(bundle / "source/src"))
    # Imports must not mutate the sealed package with __pycache__ files.
    sys.dont_write_bytecode = True
    mount_scalar_simon(bundle)
    from neural_sat_speck.backend.artifacts import (
        validate_case_inputs,
        validate_cohort,
    )
    from neural_sat_speck.backend.guard import fixed_grid

    manifest = load_json(bundle / "manifest.json")
    cohort = inside(bundle, manifest["cohort_path"])
    data = load_json(cohort)
    validate_cohort(data)
    portable = load_json(bundle / "formula_catalog_portable.json")
    paths = {"cohort": cohort, "catalog": bundle / "formula_catalog_portable.json"}
    for case, entry in portable.items():
        for kind, item in entry["files"].items():
            paths[f"{case}:{kind}"] = inside(bundle, item["path"])
    validate_case_inputs(data, paths)
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "bundled_reference_audit",
        bundle / "source/scripts/audit_backend_completed_evidence.py",
    )
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load bundled independent witness evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    witnesses = module.reference_witness_checks(paths, fixed_grid())
    if len(witnesses) != 32:
        raise ValueError("Incomplete reference witness matrix")
    project_modules = {
        name: value
        for name, value in sys.modules.items()
        if name == "neural_sat_speck" or name.startswith("neural_sat_speck.")
    }
    for value in project_modules.values():
        locations = (
            [value.__file__]
            if getattr(value, "__file__", None)
            else list(getattr(value, "__path__", []))
        )
        if not locations or any(
            not Path(path).resolve().is_relative_to(bundle / "source/src")
            for path in locations
        ):
            raise ValueError("Imported project code outside sealed bundle")
    # A second hash pass catches changes during regeneration.
    verify(bundle)
    return {
        **result,
        "formula_reproductions": 64,
        "reference_witnesses": 64,
        "recovery_solver_calls": 0,
        "dataset_draws": 0,
        "bundled_project_modules_checked": len(project_modules),
        "scalar_namespace_packages": ["neural_sat_speck.simon"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--repo", type=Path, default=Path.cwd())
    build_parser.add_argument("--catalog", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    for name in ("verify", "regenerate-formulas"):
        sub = commands.add_parser(name)
        sub.add_argument("--bundle", type=Path, required=True)
        if name == "verify":
            sub.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()
    if args.command == "build":
        result = build(args.repo, args.catalog, args.output)
    elif args.command == "verify":
        result = verify(args.bundle, args.expected_manifest_sha256)
    else:
        result = regenerate(args.bundle)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
