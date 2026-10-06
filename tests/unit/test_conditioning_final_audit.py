"""Adversarial checks of the separate final audit, with no solver calls."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def audit_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/audit_conditioning_interface_results.py"
    )
    spec = importlib.util.spec_from_file_location("conditioning_final_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_original_formula_check_rejects_parity_and_conditioning_errors(audit_module):
    formula = {
        "num_vars": 2,
        "cnf_clauses": [[1, -2]],
        "xor_clauses": [{"literals": [1, -2], "rhs": False}],
        "assumption_literals": [1, -2],
        "varmap": {"master_key": [[1, 2]]},
    }
    assert audit_module.check_assignment(formula, [None, True, False]) == [1]
    with pytest.raises(ValueError, match="XOR parity"):
        audit_module.check_assignment(formula, [None, True, True])
    with pytest.raises(ValueError, match="CNF clause"):
        audit_module.check_assignment(formula, [None, False, True])
    with pytest.raises(ValueError, match="trailing-key literal"):
        audit_module.check_assignment(
            {**formula, "cnf_clauses": [[1, -1]]}, [None, False, True]
        )
    with pytest.raises(ValueError, match="genuine Boolean"):
        audit_module.check_assignment(formula, [None, 1, False])


def test_duplicate_json_fields_are_rejected(audit_module):
    with pytest.raises(ValueError, match="Duplicate JSON field"):
        json.loads(
            '{"outcome":"global_deadline","outcome":"recovered"}',
            object_pairs_hook=audit_module.unique_object,
        )


def test_checksum_and_sidecar_filename_are_both_required(audit_module, tmp_path):
    path = tmp_path / "result.json"
    path.write_text('{"outcome":"recovered"}\n')
    sidecar = path.with_name(path.name + ".sha256")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar.write_text(f"{sha}  {path.name}\n")
    assert audit_module.checked_json(path)[0]["outcome"] == "recovered"
    path.write_text('{"outcome":"global_deadline"}\n')
    with pytest.raises(ValueError, match="Checksum/filename"):
        audit_module.checked_json(path)
    sidecar.write_text(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  other.json\n")
    with pytest.raises(ValueError, match="Checksum/filename"):
        audit_module.checked_json(path)


def test_summary_keeps_failures_and_recomputes_paired_denominators(audit_module):
    rows = []
    for index in range(8):
        for arm in ("permanent_units", "temporary_assumptions"):
            recovered = index in ({0, 1} if arm == "permanent_units" else {1})
            rows.append(
                {
                    "cell": {"cipher": "speck32_64", "arm": arm, "key_index": index},
                    "outcome": "recovered" if recovered else "global_deadline",
                    "parent_wall_seconds": 10.0 if recovered else 300.02,
                }
            )
    result = audit_module.recompute_summary(rows, "speck32_64")
    units = result["arms"]["permanent_units"]
    assert units["eligible"] == units["scheduled"] == 8
    assert units["par2_seconds_eligible"] == (2 * 10 + 6 * 600) / 8
    assert units["median_parent_seconds_eligible"] == 300.0
    assert result["discordances"] == {
        "permanent_units_only": 1,
        "temporary_assumptions_only": 0,
    }
    with pytest.raises(ValueError, match="denominator"):
        audit_module.recompute_summary(rows[:-1], "speck32_64")
