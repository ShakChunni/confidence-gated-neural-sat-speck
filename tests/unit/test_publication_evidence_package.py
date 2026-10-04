"""Adversarial checks for a portable evidence archive, with no experiment calls."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/package_publication_evidence.py"
SPEC = importlib.util.spec_from_file_location("publication_package", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def package(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    formulas = repo / "runtime/formulas"
    formulas.mkdir(parents=True)
    original = {}
    for index in range(32):
        case = f"case_{index:02}"
        files = {}
        for kind in ("native", "cnf", "public", "truth", "varmap"):
            path = formulas / f"{case}_{kind}.txt"
            value = f"{case}:{kind}\n".encode()
            path.write_bytes(value)
            path.with_name(path.name + ".sha256").write_text(
                f"{MODULE.digest(value)}  {path.name}\n"
            )
            files[kind] = {"path": str(path), "sha256": MODULE.digest(value)}
        original[case] = {"case_id": case, "files": files, "counts": {"n": 7}}
    cat = formulas / "catalog.json"
    cat.write_bytes(MODULE.encode(original))
    cohort = repo / "cohort.json"
    cohort.write_text('{"role":"synthetic_fixture"}\n')
    note = repo / "docs/PORTABLE_EVIDENCE_REPRODUCTION_20261004.md"
    note.parent.mkdir()
    note.write_text("Synthetic fixture reproduction note\n")
    entries = [
        {
            "path": str(p.relative_to(repo)),
            "sha256": MODULE.digest(p.read_bytes()),
            "bytes": p.stat().st_size,
        }
        for p in (cat, cohort, note)
    ]
    source_catalog = repo / "sources.json"
    source_catalog.write_bytes(
        MODULE.encode(
            {
                "schema": 1,
                "files": entries,
                "formula_catalog": "runtime/formulas/catalog.json",
                "cohort": "cohort.json",
                "limitations": ["Synthetic fixture only"],
            }
        )
    )
    monkeypatch.setattr(
        MODULE,
        "committed_source",
        lambda _: (
            "1" * 40,
            {"scripts/package_publication_evidence.py": b"# synthetic\n"},
        ),
    )
    output = tmp_path / "bundle"
    return repo, source_catalog, output


def test_relocation_retains_original_bytes_and_truth_separation(package, tmp_path):
    repo, catalog, output = package
    result = MODULE.build(repo, catalog, output)
    assert result["formula_cases"] == 32
    assert not (output / "formulas/case_00_truth.txt").exists()
    assert (output / "verification_truth/case_00_truth.txt").read_bytes() == (
        repo / "runtime/formulas/case_00_truth.txt"
    ).read_bytes()
    assert (output / "verification_truth/case_00_truth.txt.sha256").read_bytes() == (
        repo / "runtime/formulas/case_00_truth.txt.sha256"
    ).read_bytes()
    original = (repo / "runtime/formulas/catalog.json").read_bytes()
    assert (output / "evidence/runtime/formulas/catalog.json").read_bytes() == original
    relocated = tmp_path / "moved"
    output.rename(relocated)
    assert MODULE.verify(relocated, result["manifest_sha256"]) == result
    with pytest.raises(ValueError, match="checksum"):
        MODULE.verify(relocated, "0" * 64)


@pytest.mark.parametrize(
    "relative", [".", "../bad", "/absolute", "a/../b", "a\\b", "a//b"]
)
def test_nonportable_paths_fail_closed(tmp_path, relative):
    with pytest.raises(ValueError):
        MODULE.inside(tmp_path, relative)


def test_output_overwrite_and_source_tampering_are_rejected(package):
    repo, catalog, output = package
    (repo / "cohort.json").write_text("altered")
    with pytest.raises(ValueError, match="Pinned source"):
        MODULE.build(repo, catalog, output)
    assert not output.exists()
    output.mkdir()
    with pytest.raises(FileExistsError):
        MODULE.build(repo, catalog, output)


@pytest.mark.parametrize("change", ["missing", "extra", "altered", "symlink"])
def test_package_corruption_and_unexpected_files_are_rejected(package, change):
    repo, catalog, output = package
    MODULE.build(repo, catalog, output)
    target = output / "formulas/case_00_native.txt"
    if change == "missing":
        target.unlink()
    elif change == "extra":
        (output / "unlisted.txt").write_text("extra")
    elif change == "altered":
        target.write_text("changed")
    else:
        target.unlink()
        target.symlink_to(repo / "runtime/formulas/case_00_native.txt")
    with pytest.raises(ValueError):
        MODULE.verify(output)


def test_portable_metadata_cannot_be_changed_even_if_local_hashes_updated(package):
    _, catalog, output = package
    MODULE.build(package[0], catalog, output)
    path = output / "formula_catalog_portable.json"
    value = MODULE.load_json(path)
    value["case_00"]["counts"]["n"] = 100
    path.write_bytes(MODULE.encode(value))
    manifest_path = output / "manifest.json"
    manifest = MODULE.load_json(manifest_path)
    manifest["files"][path.name] = {
        "sha256": MODULE.digest(path.read_bytes()),
        "bytes": path.stat().st_size,
    }
    encoded = MODULE.encode(manifest)
    manifest_path.write_bytes(encoded)
    (output / "manifest.json.sha256").write_text(
        f"{MODULE.digest(encoded)}  manifest.json\n"
    )
    with pytest.raises(ValueError, match="metadata altered"):
        MODULE.verify(output)


def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"a":1,"a":2}')
    with pytest.raises(ValueError, match="Duplicate"):
        MODULE.load_json(path)


def test_original_detached_sidecar_corruption_rejects_packaging(package):
    repo, catalog, output = package
    (repo / "runtime/formulas/case_00_native.txt.sha256").write_text("wrong\n")
    with pytest.raises(ValueError, match="sidecar mismatch"):
        MODULE.build(repo, catalog, output)
    assert not output.exists()


def test_formula_symlink_outside_pinned_directory_rejected(package, tmp_path):
    repo, catalog, output = package
    path = repo / "runtime/formulas/case_00_native.txt"
    outside = tmp_path / "outside.txt"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        MODULE.build(repo, catalog, output)
    assert not output.exists()


def test_scalar_namespace_loads_leaf_without_eager_training_exports(tmp_path):
    base = tmp_path / "bundle/source/src/neural_sat_speck"
    scalar = base / "simon"
    scalar.mkdir(parents=True)
    (base / "__init__.py").write_text('"""Scalar fixture root."""\n')
    (scalar / "__init__.py").write_text("raise RuntimeError('eager training')\n")
    (scalar / "cipher.py").write_text("value = 7\n")
    code = """
import importlib.util, sys
from pathlib import Path
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('package', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
bundle = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(bundle / 'source/src'))
module.mount_scalar_simon(bundle)
from neural_sat_speck.simon import cipher
assert cipher.value == 7
assert Path(cipher.__file__).resolve().is_relative_to(bundle / 'source/src')
assert not list(bundle.rglob('*.pyc'))
"""
    ran = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code, str(SCRIPT), str(tmp_path / "bundle")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert ran.returncode == 0, ran.stderr
