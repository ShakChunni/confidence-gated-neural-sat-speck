"""Reject misleading retrospective history evidence without invoking SAT."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts import analyze_simon_candidate_history_observation as analysis


def fixture_records():
    target = {
        "candidate_k8": 100,
        "candidate_k9": 200,
        "status": "sat",
        "solve_wall_seconds": 2.0,
        "rebuild_occurred": False,
    }
    wrong = {
        **target,
        "candidate_k8": 101,
        "status": "unsat",
        "solve_wall_seconds": 3.0,
    }
    records = [
        {
            "method": method,
            "instance_id": "key0",
            "outcome": "queue_exhausted",
            "verified_on_fresh_pairs": False,
            "complete_online_wall_seconds": 50.0,
            "candidate_attempts": [],
        }
        for method in analysis.METHODS
    ]
    records[0].update(
        outcome="verified_recovery",
        verified_on_fresh_pairs=True,
        ranking_wall_seconds=20.0,
        candidate_attempts=[wrong, target],
    )
    records[-1].update(
        outcome="candidate_unresolved",
        candidate_attempts=[
            {
                **target,
                "status": "solver_timeout",
                "solve_wall_seconds": 45.2,
                "rebuild_occurred": True,
            }
        ],
    )
    return records


def test_complete_grid_derives_only_matched_target_with_unsat_history():
    rows = analysis.derive_rows(fixture_records(), {"key0"})
    assert len(rows) == 1
    assert rows[0]["preceding_unsat_calls"] == 1
    assert rows[0]["preceding_solve_wall_seconds"] == 3.0
    assert rows[0]["target_solve_wall_seconds"] == 2.0
    assert "candidate_k8" not in rows[0]


@pytest.mark.parametrize(
    "defect,reason",
    [
        ("duplicate", "Duplicate method/key"),
        ("missing", "Incomplete method/key"),
        ("flag", "Non-Boolean"),
        ("false_flag", "Outcome/flag conflict"),
        ("wrong_target", "Target tuple differs"),
        ("rebuild", "Intervening solver rebuild"),
        ("unknown_prefix", "non-UNSAT"),
    ],
)
def test_corrupt_or_non_history_evidence_is_rejected(defect, reason):
    records = deepcopy(fixture_records())
    if defect == "duplicate":
        records.append(deepcopy(records[0]))
    elif defect == "missing":
        records.pop(1)
    elif defect == "flag":
        records[0]["verified_on_fresh_pairs"] = "true"
    elif defect == "false_flag":
        records[0]["verified_on_fresh_pairs"] = False
    elif defect == "wrong_target":
        records[0]["candidate_attempts"][-1]["candidate_k8"] += 1
    elif defect == "rebuild":
        records[0]["candidate_attempts"][0]["rebuild_occurred"] = True
    elif defect == "unknown_prefix":
        records[0]["candidate_attempts"][0]["status"] = "solver_timeout"
    with pytest.raises(ValueError, match=reason):
        analysis.derive_rows(records, {"key0"})


def test_false_master_key_and_failed_fresh_verification_are_rejected():
    key = (1, 2, 3, 4)
    known = ((0, 1), analysis.encrypt_block((0, 1), key, 10))
    fresh = ((2, 3), analysis.encrypt_block((2, 3), key, 10))
    instance = SimpleNamespace(
        master_key=key, rounds=10, known_pairs=(known,), verification_pairs=(fresh,)
    )
    record = {
        "instance_id": "key0",
        "outcome": "verified_recovery",
        "recovered_master_key": list(key),
        "complete_online_wall_seconds": 50,
    }
    assert analysis.verify_recoveries([record], {"key0": instance}) == 1
    record["recovered_master_key"] = [1, 2, 3, 5]
    with pytest.raises(ValueError, match="assigned master key"):
        analysis.verify_recoveries([record], {"key0": instance})
    record["recovered_master_key"] = list(key)
    instance.verification_pairs = ((fresh[0], (0, 0)),)
    with pytest.raises(ValueError, match="Known/fresh"):
        analysis.verify_recoveries([record], {"key0": instance})


def test_frozen_hash_sidecar_and_duplicate_fields(tmp_path):
    path = tmp_path / "ledger.json"
    path.write_text('{"records":[],"records":[1]}')
    sha = analysis.hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    sidecar.write_text(f"{sha}  {path.name}\n")
    with pytest.raises(ValueError, match="Duplicate JSON"):
        analysis.checked_json(path, sha)
    path.write_text('{"records":[]}')
    with pytest.raises(ValueError, match="Frozen input changed"):
        analysis.checked_json(path, sha)
    sha = analysis.hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Sidecar mismatch"):
        analysis.checked_json(path, sha)
