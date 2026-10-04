"""Adversarial checks for backend artifact/cohort consistency.

The cohort, grid, and formula builder are deliberately tiny deterministic
fixtures. These tests exercise the validator's independent file comparisons;
they do not construct a cipher formula or contact a SAT solver.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from neural_sat_speck.backend import artifacts, formula
from neural_sat_speck.backend.guard import canonical_bytes, sha256_file

CELLS = [
    {
        "case_id": "speck32_64_k0_o4_m0",
        "cipher": "speck32_64",
        "rounds": 7,
        "key_index": 0,
        "observations": 4,
        "trailing_keys": 0,
    },
    {
        "case_id": "simon32_64_k0_o4_m1",
        "cipher": "simon32_64",
        "rounds": 10,
        "key_index": 0,
        "observations": 4,
        "trailing_keys": 1,
    },
]


def _case_inputs(_data: dict, cell: dict[str, Any]) -> tuple[dict, dict]:
    ident = cell["case_id"]
    public = {
        "instance_id": ident,
        "rounds": cell["rounds"],
        "pairs": [
            {
                "plaintext0": [1, 2],
                "ciphertext0": [3, 4],
                "plaintext1": [5, 6],
                "ciphertext1": [7, 8],
            },
            {
                "plaintext0": [9, 10],
                "ciphertext0": [11, 12],
                "plaintext1": [13, 14],
                "ciphertext1": [15, 16],
            },
        ],
    }
    if cell["trailing_keys"]:
        public["trailing_round_keys"] = [0x1234]
    truth = {
        "master_key": [1, 2, 3, 4],
        "verification_pairs": [[[20, 21], [22, 23]], [[24, 25], [26, 27]]],
    }
    return public, truth


def _formula(_cipher: str, public: dict, observations: int, trailing_keys: int) -> dict:
    ordered_observations = [
        {
            "plaintext": public["pairs"][i][f"plaintext{side}"],
            "ciphertext": public["pairs"][i][f"ciphertext{side}"],
        }
        for i in range(observations // 2)
        for side in (0, 1)
    ]
    assumptions = [-2] if trailing_keys else []
    native = "p cnf 20 1\n1 0\n"
    cnf = "p cnf 21 1\n1 0\n"
    counts = {
        "base_cnf": 1,
        "base_xor": 0,
        "assumption_units": len(assumptions),
        "native_records": 1,
        "pure_cnf_clauses": 1,
        "auxiliary_start": 21,
        "auxiliary_end": 21,
    }
    return {
        "num_vars": 20,
        "pure_cnf_num_vars": 21,
        "native_dimacs": native,
        "pure_cnf": cnf,
        "varmap": {"master_key": [[1, 2], [3, 4]]},
        "format_ordering": {"word_bits": "little-endian"},
        "known_observations": ordered_observations,
        "assumption_literals": assumptions,
        "counts": counts,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_bytes(value))


def _make_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict, dict[str, Path]]:
    """Write a minimal export-shaped bundle while mocking all expensive inputs."""
    monkeypatch.setattr(artifacts, "fixed_grid", lambda: copy.deepcopy(CELLS))
    monkeypatch.setattr(artifacts, "case_inputs", _case_inputs)
    monkeypatch.setattr(formula, "build_formula", _formula)

    data = {"synthetic_demo_fixture": True}
    cohort = tmp_path / "cohort.json"
    cohort.write_bytes(b"synthetic checked demo-like cohort\n")
    paths: dict[str, Path] = {"cohort": cohort}
    catalog: dict[str, dict] = {}

    for cell in CELLS:
        ident = cell["case_id"]
        public, truth = _case_inputs(data, cell)
        built = _formula(
            cell["cipher"], public, cell["observations"], cell["trailing_keys"]
        )
        varmap = {
            "mapping": built["varmap"],
            "format_ordering": built["format_ordering"],
            "native_num_vars": built["num_vars"],
            "pure_cnf_num_vars": built["pure_cnf_num_vars"],
        }
        values = {"public": public, "truth": truth, "varmap": varmap}
        files: dict[str, dict[str, str]] = {}
        for kind, value in values.items():
            path = tmp_path / f"{ident}_{kind}.json"
            _write_json(path, value)
            paths[f"{ident}:{kind}"] = path
            files[kind] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for kind, content, suffix in (
            ("native", built["native_dimacs"], ".dimacs"),
            ("cnf", built["pure_cnf"], ".cnf"),
        ):
            path = tmp_path / f"{ident}{suffix}"
            path.write_text(content)
            paths[f"{ident}:{kind}"] = path
            files[kind] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
        catalog[ident] = {
            "case_id": ident,
            "files": files,
            "counts": built["counts"],
            "known_observations": built["known_observations"],
            "assumption_literals": built["assumption_literals"],
            "cohort_sha256": sha256_file(cohort),
        }

    catalog_path = tmp_path / "catalog.json"
    _write_json(catalog_path, catalog)
    paths["catalog"] = catalog_path
    return data, paths


@pytest.mark.parametrize(
    ("case", "artifact", "mutate", "message"),
    [
        (
            "base",
            "public",
            lambda value: value["pairs"][0]["ciphertext1"].__setitem__(0, 99),
            "Backend case differs",
        ),
        (
            "base",
            "public",
            lambda value: value.__setitem__("master_key", [1, 2, 3, 4]),
            "Backend case differs",
        ),
        (
            "conditioned",
            "public",
            lambda value: value["trailing_round_keys"].__setitem__(0, 0x4321),
            "Backend case differs",
        ),
        (
            "base",
            "truth",
            lambda value: value["master_key"].__setitem__(0, 99),
            "Backend case differs",
        ),
        (
            "base",
            "truth",
            lambda value: value["verification_pairs"][1][1].__setitem__(0, 99),
            "Backend case differs",
        ),
        (
            "base",
            "varmap",
            lambda value: value["mapping"]["master_key"][0].__setitem__(0, 99),
            "Variable map differs",
        ),
        (
            "base",
            "varmap",
            lambda value: value["format_ordering"].__setitem__(
                "word_bits", "big-endian"
            ),
            "Variable map differs",
        ),
        ("base", "native", lambda _value: None, "Formula differs"),
        ("base", "cnf", lambda _value: None, "Formula differs"),
    ],
    ids=[
        "forged-observation",
        "truth-leaked-into-public",
        "forged-trailing-key-unit",
        "forged-master-truth",
        "forged-verification-pair",
        "forged-variable-map",
        "forged-bit-order",
        "forged-native-formula",
        "forged-pure-cnf-formula",
    ],
)
def test_semantic_validator_rejects_single_forged_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    artifact: str,
    mutate: Callable[[dict], None],
    message: str,
) -> None:
    data, paths = _make_bundle(tmp_path, monkeypatch)
    ident = CELLS[1]["case_id"] if case == "conditioned" else CELLS[0]["case_id"]
    path = paths[f"{ident}:{artifact}"]
    if artifact in {"native", "cnf"}:
        # Keep the manifest/catalog hash untouched when bytes are corrupted.
        path.write_bytes(path.read_bytes() + b"c forged\n")
    else:
        value = json.loads(path.read_bytes())
        mutate(value)
        _write_json(path, value)

    with pytest.raises(ValueError, match=message):
        artifacts.validate_case_inputs(data, paths)


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        (
            "counts",
            lambda value: value.__setitem__("base_xor", value["base_xor"] + 1),
        ),
        ("known_observations", lambda value: value[0]["ciphertext"].__setitem__(0, 99)),
        ("assumption_literals", lambda value: value.__setitem__(0, -3)),
    ],
    ids=["counts", "known-observations", "assumption-units"],
)
def test_semantic_validator_rejects_forged_catalog_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    mutate: Callable[[Any], Any],
) -> None:
    data, paths = _make_bundle(tmp_path, monkeypatch)
    catalog_path = paths["catalog"]
    catalog = json.loads(catalog_path.read_bytes())
    ident = (
        CELLS[1]["case_id"] if field == "assumption_literals" else CELLS[0]["case_id"]
    )
    value = catalog[ident][field]
    mutate(value)
    # Formula files and their recorded hashes remain unchanged.
    _write_json(catalog_path, catalog)

    with pytest.raises(ValueError, match="Catalog provenance/metadata mismatch"):
        artifacts.validate_case_inputs(data, paths)


def test_semantic_validator_accepts_canonical_minimal_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, paths = _make_bundle(tmp_path, monkeypatch)
    artifacts.validate_case_inputs(data, paths)


def test_catalog_observations_are_checked_against_formula_not_file_hash_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, paths = _make_bundle(tmp_path, monkeypatch)
    catalog = json.loads(paths["catalog"].read_bytes())
    ident = CELLS[0]["case_id"]
    # Alter the catalog semantic claim while leaving every formula and file hash intact.
    catalog[ident]["known_observations"][0]["plaintext"][0] += 1
    _write_json(paths["catalog"], catalog)
    with pytest.raises(ValueError, match="Catalog provenance/metadata mismatch"):
        artifacts.validate_case_inputs(data, paths)
