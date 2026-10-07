"""Synthetic cryptographic and dispatch contracts; no opened-key experiment."""

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from neural_sat_speck.backend.artifacts import cipher_functions
from neural_sat_speck.backend.formula import build_formula
from neural_sat_speck.backend.guard import atomic_json, read_checked_json, sha256_file
from neural_sat_speck.backend.worker import _complete_assignment
from neural_sat_speck.followups import ranking_depth_control as control
from neural_sat_speck.followups import ranking_depth_runner as runner
from neural_sat_speck.followups import ranking_depth_worker as worker
from neural_sat_speck.followups.conditioning_control import digest, formula_identity


def toy(cipher, rounds=3):
    key = (0x1234, 0x5678, 0x9ABC, 0xDEF0)
    encrypt, expand = cipher_functions(cipher)
    pairs = []
    for p0, p1 in (((1, 2), (3, 4)), ((5, 6), (7, 8)), ((9, 10), (11, 12))):
        pairs.append(
            {
                "plaintext0": list(p0),
                "plaintext1": list(p1),
                "ciphertext0": list(encrypt(p0, key, rounds)),
                "ciphertext1": list(encrypt(p1, key, rounds)),
            }
        )
    public = {"instance_id": "synthetic", "rounds": rounds, "pairs": pairs}
    fresh = {
        "verification_pairs": [
            [list(p), list(encrypt(p, key, rounds))] for p in ((21, 22), (23, 24))
        ]
    }
    return public, fresh, key, list(expand(key, rounds))


def model_for_key(formula, key):
    from pycryptosat import Solver

    solver = Solver(threads=1, verbose=0)
    for clause in formula["cnf_clauses"]:
        solver.add_clause(clause)
    for record in formula["xor_clauses"]:
        solver.add_xor_clause(record["literals"], record["rhs"])
    for literal in formula["assumption_literals"]:
        solver.add_clause([literal])
    for word, value in zip(formula["varmap"]["master_key"], key, strict=True):
        for bit, v in enumerate(word):
            solver.add_clause([v if (value >> bit) & 1 else -v])
    sat, solution = solver.solve()
    assert sat is True
    return {i: solution[i] for i in range(1, formula["num_vars"] + 1)}


@pytest.mark.parametrize("cipher", ["speck32_64", "simon32_64"])
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("target_rounds", [False, True])
def test_conditioned_units_equal_reference_and_full_model(cipher, depth, target_rounds):
    rounds = (7 if cipher == "speck32_64" else 10) if target_rounds else 3
    public, _, key, keys = toy(cipher, rounds)
    base = build_formula(cipher, public, 4, 0, include_cnf=False)
    conditioned = worker.conditioned_formula(base, keys[-depth:])
    reference = build_formula(
        cipher,
        {**public, "trailing_round_keys": keys[-depth:]},
        4,
        depth,
        include_cnf=False,
    )
    assert conditioned["native_dimacs"] == reference["native_dimacs"]
    assert conditioned["assumption_literals"] == reference["assumption_literals"]
    assert conditioned["counts"]["assumption_units"] == 16 * depth
    assert base["assumption_literals"] == []
    assignment = model_for_key(conditioned, key)
    assert _complete_assignment(conditioned, assignment)
    assignment[abs(conditioned["assumption_literals"][0])] ^= True
    assert not _complete_assignment(conditioned, assignment)


@pytest.mark.parametrize("candidate", [[], [True], [-1], [65536], [1, 2, 3]])
def test_invalid_candidate_rejected(candidate):
    with pytest.raises(ValueError):
        worker.conditioned_formula({}, candidate)


@pytest.mark.parametrize("cipher", ["speck32_64", "simon32_64"])
def test_all_known_blocks_and_fresh_blocks_checked_without_key_truth(cipher):
    public, fresh, key, _ = toy(cipher)
    assert worker.verify_without_truth(cipher, 3, list(key), public, fresh) == {
        "verified_known": True,
        "verified_fresh": True,
    }
    bad = copy.deepcopy(public)
    bad["pairs"][-1]["ciphertext1"][0] ^= 1
    assert (
        worker.verify_without_truth(cipher, 3, list(key), bad, fresh)["verified_known"]
        is False
    )
    bad_fresh = copy.deepcopy(fresh)
    bad_fresh["verification_pairs"][0][1][0] ^= 1
    assert (
        worker.verify_without_truth(cipher, 3, list(key), public, bad_fresh)[
            "verified_fresh"
        ]
        is False
    )
    overlap = {
        "verification_pairs": [
            [public["pairs"][0]["plaintext0"], public["pairs"][0]["ciphertext0"]],
            fresh["verification_pairs"][1],
        ]
    }
    with pytest.raises(ValueError, match="overlaps"):
        worker.verify_without_truth(cipher, 3, list(key), public, overlap)


def test_grid_is_complete_fixed_and_counterbalanced():
    cells = control.fixed_cells("full")
    assert len(cells) == len({c["cell_id"] for c in cells}) == 64
    assert control.fixed_cells("pilot") == cells[:8]
    for cipher in ("speck32_64", "simon32_64"):
        by_key = [
            [c for c in cells if c["cipher"] == cipher and c["key_index"] == i]
            for i in range(8)
        ]
        assert all(len(rows) == 4 for rows in by_key)
        first = [(r[0]["trailing_keys"], r[0]["profile"]) for r in by_key]
        assert all(first.count(arm) == 2 for arm in set(first))
    with pytest.raises(ValueError):
        control.fixed_cells("unseen")


def test_streaming_hash_does_not_read_large_file_at_once(tmp_path, monkeypatch):
    p = tmp_path / "asset.bin"
    data = b"bounded-memory" * 1000
    p.write_bytes(data)

    def forbidden(*_):
        raise AssertionError("Whole-file allocation")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    assert control.streaming_hash(p) == hashlib.sha256(data).hexdigest()
    assert (
        control.checked_path(
            tmp_path, {"path": str(p), "sha256": hashlib.sha256(data).hexdigest()}
        )
        == p
    )


def test_distribution_metadata_pins_wheels_and_editable_installations(
    tmp_path, monkeypatch
):
    wheel, editable = tmp_path / "wheel.dist-info", tmp_path / "project.egg-info"
    wheel.mkdir()
    editable.mkdir()
    (wheel / "METADATA").write_text("Name: synthetic-wheel\n")
    (editable / "PKG-INFO").write_text("Name: synthetic-editable\n")
    monkeypatch.setattr(
        control.importlib.metadata,
        "distributions",
        lambda: [SimpleNamespace(_path=wheel), SimpleNamespace(_path=editable)],
    )
    assert control.distribution_metadata_paths() == sorted(
        [str(wheel / "METADATA"), str(editable / "PKG-INFO")]
    )
    (editable / "PKG-INFO").unlink()
    with pytest.raises(ValueError, match="lacks a metadata file"):
        control.distribution_metadata_paths()


def test_loaded_tf_inventory_matches_independent_import_only_worker(monkeypatch):
    """Catch import-exposed vendored metadata without models, data, or ranking."""
    root = Path(__file__).resolve().parents[2]
    python = root / "runtime/simon_tf_venv/bin/python"
    if not python.is_file():
        pytest.skip("Preserved TensorFlow environment is unavailable")
    monkeypatch.setenv("PYTHONPATH", "/synthetic/interpreter-injection")
    monkeypatch.setenv("DYLD_SYNTHETIC_INJECTION", "reject-this")
    calls = []
    actual_run = subprocess.run

    def capture_probe(*args, **kwargs):
        calls.append(kwargs["env"])
        return actual_run(*args, **kwargs)

    monkeypatch.setattr(control.subprocess, "run", capture_probe)
    frozen = control.runtime_identity(python)["tensorflow"]
    assert calls[0]["PYTHONPATH"] == str(root / "src")
    assert calls[0]["PYTHONNOUSERSITE"] == "1"
    assert "DYLD_SYNTHETIC_INJECTION" not in calls[0]
    env = {
        **runner.controlled_environment(root, root / "runtime/ranking_depth_tmp"),
        "TF_NUM_INTRAOP_THREADS": "8",
        "TF_NUM_INTEROP_THREADS": "1",
    }
    result = actual_run(
        [
            str(python),
            "-c",
            "import tensorflow,h5py,keras,numpy,json; "
            "from neural_sat_speck.followups.ranking_depth_ranking import _tf_runtime_identity; "
            "print(json.dumps(_tf_runtime_identity()))",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    worker_runtime = json.loads(result.stdout)
    assert worker_runtime["installed_packages"] == frozen["installed_packages"]
    assert worker_runtime["packages"] == {
        name: {"version": version, "origin": frozen["origins"][name]}
        for name, version in frozen["packages"].items()
    }
    # Bind all metadata, including same-name distributions in vendored paths.
    assert frozen["metadata_paths"]
    assert all(Path(path).is_file() for path in frozen["metadata_paths"])
    assert len(set(frozen["metadata_paths"])) == len(frozen["metadata_paths"])


@pytest.fixture
def worker_case(tmp_path, monkeypatch):
    public, fresh, key, keys = toy("speck32_64")
    p, f = tmp_path / "public.json", tmp_path / "fresh.json"
    atomic_json(p, public)
    atomic_json(f, fresh)
    directory = tmp_path / "worker"
    directory.mkdir()
    base = build_formula("speck32_64", public, 4, 0, include_cnf=False)
    payload = {
        "directory": str(directory),
        "cell": {"cipher": "speck32_64", "rounds": 3, "trailing_keys": 1},
        "repo_root": str(tmp_path),
        "source_commit": "a" * 40,
        "public_path": str(p),
        "public_sha256": sha256_file(p),
        "verification_path": str(f),
        "verification_sha256": sha256_file(f),
        "base_formula_sha256": digest(formula_identity(base)),
        "paths": {},
    }
    receipt = SimpleNamespace(
        git_commit="a" * 40, to_dict=lambda: {"git_commit": "a" * 40}
    )
    import neural_sat_speck.protocol.worker_telemetry as telemetry

    monkeypatch.setattr(telemetry, "certify_worker_or_raise", lambda _: receipt)
    parents = [keys[-1] ^ 1, keys[-1]]
    parents.extend(i for i in range(32) if i not in parents)
    parents = parents[:32]

    def rank(cipher, cts, depth, paths, directory, callback):
        assert cipher == "speck32_64" and depth == 1
        assert cts == [
            (tuple(pair["ciphertext0"]), tuple(pair["ciphertext1"]))
            for pair in public["pairs"]
        ]
        callback("stage1_start", {})
        callback("stage1_complete", {"stage1_parents": parents})
        return {
            "queue": [[w] for w in parents],
            "stage1_parents": parents,
            "ranking_receipts": {},
            "stage1_wall_seconds": 0.01,
            "stage2_wall_seconds": 0.0,
        }

    monkeypatch.setitem(
        sys.modules,
        "neural_sat_speck.followups.ranking_depth_ranking",
        SimpleNamespace(rank_queue=rank),
    )
    return payload, key, base, directory


def test_worker_unsat_advances_and_fresh_verified_sat_stops(worker_case, monkeypatch):
    payload, key, base, directory = worker_case
    calls = []

    def solve(formula, payload, attempt_dir, journal):
        attempt_dir.mkdir(parents=True)
        calls.append(formula["assumption_literals"])
        parsed = (
            {"status": "unsat", "assignment": {}}
            if len(calls) == 1
            else {"status": "sat", "assignment": model_for_key(formula, key)}
        )
        return {"parsed": parsed}

    monkeypatch.setattr(worker, "solve_candidate", solve)
    result = worker.execute(payload)
    assert result["outcome"] == "recovered" and result["attempt_count"] == 2
    assert (
        result["recovered_key"] == list(key)
        and result["verified_known"] is True
        and result["verified_fresh"] is True
    )
    assert len(calls) == 2
    assert read_checked_json(directory / "queue.json")["queue_sha256"]
    assert (
        read_checked_json(directory / "attempts/0000/attempt.json")["status"] == "unsat"
    )


def test_unknown_is_terminal_and_not_unsat(worker_case, monkeypatch):
    payload, _, _, _ = worker_case
    calls = []

    def solve(formula, payload, attempt_dir, journal):
        attempt_dir.mkdir(parents=True)
        calls.append(1)
        return {"parsed": {"status": "unknown", "assignment": {}}}

    monkeypatch.setattr(worker, "solve_candidate", solve)
    result = worker.execute(payload)
    assert result["outcome"] == "solver_unknown" and len(calls) == 1


def test_input_drift_prevents_ranking_or_solving(worker_case, monkeypatch):
    payload, _, _, _ = worker_case
    payload["public_sha256"] = "0" * 64
    monkeypatch.setattr(
        worker, "solve_candidate", lambda *_: pytest.fail("Unexpected solver call")
    )
    result = worker.execute(payload)
    assert (
        result["outcome"] == "worker_exception"
        and "Public input drift" in result["error"]
    )


@pytest.fixture
def manifest_fixture(tmp_path, monkeypatch):
    root = tmp_path / "snapshot"
    root.mkdir()
    design = copy.deepcopy(
        json.loads((Path(__file__).resolve().parents[2] / control.DESIGN).read_bytes())
    )
    files = {}
    for name in design["files"]:
        p = tmp_path / f"{name}.fixture"
        p.write_text("synthetic identity " + name)
        files[name] = control.file_identity(p)
        design["files"][name] = {"path": str(p), "sha256": files[name]["sha256"]}
    (root / control.DESIGN).parent.mkdir(parents=True)
    (root / control.DESIGN).write_text(json.dumps(design))
    (root / "src").mkdir()
    (root / "src/toy.py").write_text("# fixture source\n")
    names = ["src/toy.py", control.DESIGN]
    files.update({"source:" + n: control.file_identity(root / n) for n in names})
    files.update(
        python=control.file_identity(Path(sys.executable)),
        tf_python=control.file_identity(Path(sys.executable)),
    )
    runtime = {"fixture": "package identity without importing any model"}
    monkeypatch.setattr(control, "source_names", lambda _: names)
    monkeypatch.setattr(control, "validate_source_root", lambda *_: None)
    monkeypatch.setattr(control, "source_commit", lambda _: "a" * 40)
    monkeypatch.setattr(control, "runtime_support_paths", lambda: {})
    monkeypatch.setattr(control, "runtime_files", lambda *_: {})
    monkeypatch.setattr(control, "runtime_identity", lambda *_: runtime)
    case_files = {}
    for role in ("public", "verification", "truth"):
        p = tmp_path / f"{role}.json"
        atomic_json(p, {"fixture_only": True})
        case_files[role] = control.file_identity(p)
    manifest = {
        "schema": "backend_ranking_depth_manifest_v1",
        "status": "APPROVED",
        "author_decision": "D-055",
        "execution_authorized": True,
        "mode": "pilot",
        "profile": design,
        "source_root": str(root),
        "source_commit": "a" * 40,
        "python_version": sys.version,
        "runtime": runtime,
        "files": files,
        "cases": {
            c["case_id"]: {**case_files, "base_formula_sha256": "b" * 64}
            for c in control.fixed_cells("pilot")
        },
        "cells": control.fixed_cells("pilot"),
        "campaign_ceiling_seconds": 3600.0,
        "pilot_audit": None,
    }
    path = tmp_path / "manifest.json"
    atomic_json(path, manifest)
    return manifest, path


def test_manifest_checks_pins_without_loading_data(manifest_fixture):
    manifest, path = manifest_fixture
    checked, paths = control.load_manifest(path)
    assert checked == manifest and paths["python"] == Path(sys.executable)


@pytest.mark.parametrize(
    "drift",
    [
        "authorization",
        "deadline",
        "grid",
        "source_inventory",
        "source_hash",
        "runtime",
        "campaign_cap",
    ],
)
def test_manifest_drift_fails_before_cohort_or_worker(manifest_fixture, drift):
    manifest, path = manifest_fixture
    if drift == "authorization":
        manifest["execution_authorized"] = "true"
    elif drift == "deadline":
        manifest["profile"]["global_deadline_seconds"] = 301.0
    elif drift == "grid":
        manifest["cells"][-1] = manifest["cells"][0]
    elif drift == "source_inventory":
        del manifest["files"]["source:src/toy.py"]
    elif drift == "source_hash":
        Path(manifest["files"]["source:src/toy.py"]["path"]).write_text("changed")
    elif drift == "runtime":
        manifest["runtime"] = {"different": True}
    else:
        manifest["campaign_ceiling_seconds"] += 1
    atomic_json(path, manifest)
    with pytest.raises(ValueError):
        control.load_manifest(path)


def test_orphan_lookup_identifies_tf_child_by_unique_invocation_path(monkeypatch):
    token = "a" * 32
    output = (
        f"23456 /python -m neural_sat_speck.followups.ranking_depth_ranking --tf-stage1 --invocation-path /project/cells/{token}\n"
        f"34567 /bin/echo /project/cells/{token}\n"
        f"45678 /python -m neural_sat_speck.followups.ranking_depth_worker /project/cells/prefix{token}/payload.json\n"
    )
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *_, **__: SimpleNamespace(stdout=output)
    )
    assert runner.live_invocation_groups(token) == {23456}


def test_orphan_lookup_refuses_own_live_process_group(monkeypatch):
    token = "a" * 32
    output = f"{runner.os.getpgrp()} /python -m neural_sat_speck.followups.ranking_depth_worker /project/{token}/payload.json\n"
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *_, **__: SimpleNamespace(stdout=output)
    )
    with pytest.raises(RuntimeError, match="Unsafe"):
        runner.live_invocation_groups(token)
