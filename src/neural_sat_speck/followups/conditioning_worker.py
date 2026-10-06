"""One fresh-solver cell for the opened conditioning-interface control.

The only arm-specific operation is whether the fixed trailing-key literals are
added as permanent unit clauses or passed once to ``solve`` as assumptions.
The parent guardian, not CryptoMiniSat, owns the hard deadline.
"""

from __future__ import annotations

import hashlib
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

from neural_sat_speck.backend.artifacts import verify_key
from neural_sat_speck.backend.formula import build_formula
from neural_sat_speck.backend.guard import (
    THREAD_ENVIRONMENT,
    atomic_json,
    canonical_bytes,
    read_checked_json,
    sha256_file,
)
from neural_sat_speck.backend.worker import Journal, _complete_assignment
from neural_sat_speck.followups.environment import power_sample
from neural_sat_speck.protocol.worker_telemetry import (
    certify_worker_or_raise,
    observe_worker_telemetry,
)

ARMS = frozenset({"permanent_units", "temporary_assumptions"})
ROUND_PROFILE = {"speck32_64": 7, "simon32_64": 10}
PAYLOAD_KEYS = frozenset(
    {
        "cell",
        "invocation_id",
        "directory",
        "repo_root",
        "source_commit",
        "python_path",
        "public_path",
        "public_sha256",
        "truth_path",
        "truth_sha256",
        "formula_sha256",
        "deadline_monotonic",
    }
)
CELL_KEYS = frozenset(
    {
        "cell_id",
        "case_id",
        "cipher",
        "rounds",
        "key_index",
        "observations",
        "trailing_keys",
        "arm",
    }
)
IDENTITY_FIELDS = (
    "num_vars",
    "cnf_clauses",
    "xor_clauses",
    "assumption_literals",
    "varmap",
)


class EnvironmentRejected(RuntimeError):
    """The worker could not certify its execution environment."""


def _sha256_formula(formula: dict[str, Any]) -> str:
    identity = {name: formula[name] for name in IDENTITY_FIELDS}
    return hashlib.sha256(canonical_bytes(identity)).hexdigest()


def _word(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= 0xFFFF:
        raise ValueError(f"{name} must be a genuine unsigned 16-bit integer")
    return value


def _block(value: object, name: str) -> list[int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two words")
    return [_word(value[0], f"{name}[0]"), _word(value[1], f"{name}[1]")]


def _validate_payload(payload: object) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(payload, dict) or set(payload) != PAYLOAD_KEYS:
        raise ValueError("Worker payload fields differ from the fixed interface")
    cell = payload["cell"]
    if not isinstance(cell, dict) or set(cell) != CELL_KEYS:
        raise ValueError("Cell fields differ from the fixed interface")
    cipher = cell["cipher"]
    if cipher not in ROUND_PROFILE:
        raise ValueError("Unsupported conditioning-control cipher")
    ints = ("rounds", "key_index", "observations", "trailing_keys")
    if any(type(cell.get(name)) is not int for name in ints):
        raise TypeError("Cell dimensions and indices must be genuine integers")
    if (
        cell["rounds"] != ROUND_PROFILE[cipher]
        or not 0 <= cell["key_index"] < 8
        or cell["observations"] != 4
        or cell["trailing_keys"] != 1
    ):
        raise ValueError("Cell differs from the fixed opened-demo profile")
    case_id = f"{cipher}_k{cell['key_index']}_o4_m1"
    if (
        cell["case_id"] != case_id
        or cell["cell_id"] != f"{case_id}__{cell['arm']}"
        or cell["arm"] not in ARMS
    ):
        raise ValueError("Cell ID, case ID, or arm does not match the fixed grid")

    invocation = payload["invocation_id"]
    if (
        not isinstance(invocation, str)
        or re.fullmatch(r"[0-9a-f]{32}", invocation) is None
    ):
        raise ValueError("Invocation ID must be a 32-character lowercase hex token")
    directory = payload["directory"]
    if (
        not isinstance(directory, str)
        or not Path(directory).is_absolute()
        or Path(directory).name != invocation
    ):
        raise ValueError("Worker directory must be absolute and bound to invocation")
    for name in ("repo_root", "python_path", "public_path", "truth_path"):
        value = payload[name]
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError(f"{name} must be an absolute path")
    if not Path(payload["repo_root"]).is_dir():
        raise ValueError("Repository root does not exist")
    if not Path(payload["python_path"]).is_file():
        raise ValueError("Pinned Python executable does not exist")
    if (
        not isinstance(payload["source_commit"], str)
        or re.fullmatch(r"[0-9a-f]{40}", payload["source_commit"]) is None
    ):
        raise ValueError("Source commit must be a 40-character lowercase hex ID")
    for name in ("formula_sha256", "public_sha256", "truth_sha256"):
        if (
            not isinstance(payload[name], str)
            or re.fullmatch(r"[0-9a-f]{64}", payload[name]) is None
        ):
            raise ValueError(f"{name} must be a SHA-256 digest")
    deadline = payload["deadline_monotonic"]
    if (
        type(deadline) not in (int, float)
        or not math.isfinite(deadline)
        or deadline <= 0
    ):
        raise ValueError("Parent monotonic deadline must be finite and positive")
    return payload, cell


def _validate_public(public: object, cell: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(public, dict) or set(public) != {
        "instance_id",
        "rounds",
        "pairs",
        "trailing_round_keys",
    }:
        raise ValueError("Public input contains unexpected or missing fields")
    if (
        public["instance_id"] != cell["case_id"]
        or type(public["rounds"]) is not int
        or public["rounds"] != cell["rounds"]
        or not isinstance(public["pairs"], list)
        or len(public["pairs"]) != 2
        or not isinstance(public["trailing_round_keys"], list)
        or len(public["trailing_round_keys"]) != 1
    ):
        raise ValueError("Public input differs from the fixed observation profile")
    for pair_index, pair in enumerate(public["pairs"]):
        if not isinstance(pair, dict) or set(pair) != {
            "plaintext0",
            "plaintext1",
            "ciphertext0",
            "ciphertext1",
        }:
            raise ValueError(f"Public pair {pair_index} has an invalid schema")
        for field in ("plaintext0", "plaintext1", "ciphertext0", "ciphertext1"):
            _block(pair[field], f"pairs[{pair_index}].{field}")
    _word(public["trailing_round_keys"][0], "trailing_round_keys[0]")
    return public


def _read_pinned_json(path: Path, expected_sha256: str) -> Any:
    """Read checked JSON and bind both its sidecar and canonical value to a pin."""
    value = read_checked_json(path)
    canonical_sha = hashlib.sha256(canonical_bytes(value)).hexdigest()
    if canonical_sha != expected_sha256 or sha256_file(path) != expected_sha256:
        raise ValueError(f"Pinned JSON identity differs: {path}")
    return value


def _validate_formula(formula: dict[str, Any], public: dict[str, Any]) -> None:
    if type(formula.get("num_vars")) is not int or formula["num_vars"] <= 0:
        raise ValueError("Formula has an invalid variable count")
    assumptions = formula.get("assumption_literals")
    if (
        not isinstance(assumptions, list)
        or len(assumptions) != 16
        or any(type(lit) is not int or lit == 0 for lit in assumptions)
    ):
        raise ValueError("Exactly sixteen valid final-round-key literals are required")
    varmap = formula.get("varmap")
    if not isinstance(varmap, dict):
        raise ValueError("Formula variable map is missing")
    master_bits = varmap.get("master_key")
    round_keys = varmap.get("round_keys")
    if (
        not isinstance(master_bits, list)
        or len(master_bits) != 4
        or any(not isinstance(word, list) or len(word) != 16 for word in master_bits)
        or not isinstance(round_keys, list)
        or len(round_keys) != formula["rounds"]
        or not isinstance(round_keys[-1], list)
        or len(round_keys[-1]) != 16
    ):
        raise ValueError("Formula variable map has an invalid key schedule shape")
    all_ids = [var for word in master_bits for var in word] + round_keys[-1]
    if any(
        type(var) is not int or not 1 <= var <= formula["num_vars"] for var in all_ids
    ):
        raise ValueError("Formula variable map references an invalid variable")
    expected = [
        var if (public["trailing_round_keys"][-1] >> bit) & 1 else -var
        for bit, var in enumerate(round_keys[-1])
    ]
    if assumptions != expected:
        raise ValueError("Formula assumptions do not encode the public trailing key")


def new_solver():
    """Construct the exact wrapper configuration used by both arms."""
    from pycryptosat import Solver

    return Solver(verbose=0, threads=1)


def _reference_known_check(
    cipher: str, rounds: int, key: list[int], public: dict[str, Any]
) -> bool:
    if cipher == "speck32_64":
        from neural_sat_speck.speck.cipher import encrypt_block
        from neural_sat_speck.speck.key_schedule import expand_key
    else:
        from neural_sat_speck.simon.cipher import encrypt_block
        from neural_sat_speck.simon.key_schedule import expand_key
    master = tuple(key)
    if expand_key(master, rounds)[-1] != public["trailing_round_keys"][-1]:
        return False
    return all(
        tuple(encrypt_block(tuple(pair[f"plaintext{side}"]), master, rounds))
        == tuple(pair[f"ciphertext{side}"])
        for pair in public["pairs"]
        for side in (0, 1)
    )


def _environment_gate(payload: dict[str, Any], directory: Path) -> None:
    """Persist raw observations first, then apply the existing certification gate."""
    observation_error = None
    power_error = None
    observed = None
    power = None
    try:
        observed = observe_worker_telemetry(Path(payload["repo_root"]))
        atomic_json(directory / "python_worker_observation.json", observed.to_dict())
    except Exception as error:  # preserve the independent power sample if available
        observation_error = f"{type(error).__name__}: {error}"
        atomic_json(
            directory / "python_worker_observation_error.json",
            {"error": observation_error, "observed_monotonic": time.perf_counter()},
        )
    try:
        power = power_sample()
        atomic_json(directory / "power_at_worker_start.json", power)
    except Exception as error:
        power_error = f"{type(error).__name__}: {error}"
        atomic_json(
            directory / "power_observation_error.json",
            {"error": power_error, "observed_monotonic": time.perf_counter()},
        )
    if observation_error or power_error:
        raise EnvironmentRejected(
            f"Initial environment observation failed: {observation_error or power_error}"
        )
    if observed.is_certified is not True:
        raise EnvironmentRejected(
            f"Initial worker telemetry rejected: {observed.rejection_reason}"
        )
    observed_record = observed.to_dict()
    if (
        observed_record.get("git_commit") != payload["source_commit"]
        or observed_record.get("python_executable") != payload["python_path"]
        or any(
            observed_record.get("thread_environment", {}).get(name) != value
            for name, value in THREAD_ENVIRONMENT.items()
        )
    ):
        raise EnvironmentRejected(
            "Initial worker source/runtime/thread identity differs"
        )
    try:
        receipt = certify_worker_or_raise(Path(payload["repo_root"]))
    except Exception as error:
        atomic_json(
            directory / "worker_certification_error.json",
            {
                "error": f"{type(error).__name__}: {error}",
                "observed_monotonic": time.perf_counter(),
            },
        )
        raise EnvironmentRejected(f"Worker certification failed: {error}") from error
    atomic_json(directory / "python_worker_receipt.json", receipt.to_dict())
    receipt_record = receipt.to_dict()
    if (
        receipt.is_certified is not True
        or receipt_record.get("git_commit") != payload["source_commit"]
        or receipt_record.get("python_executable") != payload["python_path"]
        or any(
            receipt_record.get("thread_environment", {}).get(name) != value
            for name, value in THREAD_ENVIRONMENT.items()
        )
    ):
        raise EnvironmentRejected(
            "Worker receipt source/runtime/thread identity differs"
        )
    if power.get("accepted") is not True:
        raise EnvironmentRejected("Worker power/sleep sample was rejected")


def _decode_model(
    formula: dict[str, Any], solution: object
) -> tuple[list[bool | None], dict[int, bool], list[int]]:
    if (
        not isinstance(solution, (tuple, list))
        or len(solution) != formula["num_vars"] + 1
    ):
        raise ValueError("SAT result lacks a complete indexed assignment")
    if solution[0] is not None or any(
        type(value) is not bool for value in solution[1:]
    ):
        raise ValueError(
            "SAT model must contain bool values at every one-based variable"
        )
    model: list[bool | None] = [None, *solution[1:]]
    assignment = {index: value for index, value in enumerate(model[1:], start=1)}
    if not _complete_assignment(formula, assignment):
        raise ValueError(
            "SAT model violates the original CNF/XOR formula or true units"
        )
    key = [
        sum(int(assignment[var]) << bit for bit, var in enumerate(word))
        for word in formula["varmap"]["master_key"]
    ]
    return model, assignment, key


def execute(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one fixed cell and return its auditable result record."""
    started = time.perf_counter()
    raw_cell = payload.get("cell") if isinstance(payload, dict) else None
    cell = raw_cell if isinstance(raw_cell, dict) else {}
    invocation = payload.get("invocation_id") if isinstance(payload, dict) else None
    formula_sha = payload.get("formula_sha256") if isinstance(payload, dict) else None
    directory_value = payload.get("directory") if isinstance(payload, dict) else None
    if not isinstance(directory_value, str) or not Path(directory_value).is_absolute():
        raise ValueError("Cannot journal worker without an absolute output directory")
    directory = Path(directory_value)
    directory.mkdir(parents=True, exist_ok=True)
    journal = Journal(directory / "journal.json")
    result: dict[str, Any] = {
        "schema": "conditioning_worker_result_v1",
        "cell_id": cell.get("cell_id"),
        "case_id": cell.get("case_id"),
        "invocation_id": invocation,
        "formula_sha256": formula_sha,
        "solver_status": "not_started",
        "formula_valid": False,
        "verified_known": False,
        "verified_fresh": False,
        "assigned_key_match": False,
        "recovered_key": None,
        "model": None,
        "outcome": "worker_exception",
        "error": None,
    }
    try:
        journal.transition("input_loading")
        payload, cell = _validate_payload(payload)
        public = _validate_public(
            _read_pinned_json(Path(payload["public_path"]), payload["public_sha256"]),
            cell,
        )

        journal.transition("environment_observation")
        _environment_gate(payload, directory)

        journal.transition("formula_construction")
        formula = build_formula(
            cell["cipher"],
            public,
            cell["observations"],
            cell["trailing_keys"],
            include_cnf=False,
        )
        _validate_formula(formula, public)
        observed_sha = _sha256_formula(formula)
        if observed_sha != payload["formula_sha256"]:
            raise ValueError("Rebuilt formula identity differs from the frozen payload")

        journal.transition("solver_loading_and_conditioning")
        solver = new_solver()
        for clause in formula["cnf_clauses"]:
            solver.add_clause(clause)
        for xor in formula["xor_clauses"]:
            solver.add_xor_clause(xor["literals"], xor["rhs"])
        assumptions = list(formula["assumption_literals"])
        if cell["arm"] == "permanent_units":
            for literal in assumptions:
                solver.add_clause([literal])

        journal.transition("solving")
        atomic_json(
            directory / "solver_dispatch.json",
            {
                "invocation_id": payload["invocation_id"],
                "cell_id": cell["cell_id"],
                "formula_sha256": observed_sha,
                "arm": cell["arm"],
                "assumptions": (
                    assumptions if cell["arm"] == "temporary_assumptions" else []
                ),
                "permanent_unit_count": (16 if cell["arm"] == "permanent_units" else 0),
                "solver_configuration": {"verbose": 0, "threads": 1},
                "requested_at_monotonic": time.perf_counter(),
            },
        )
        sat, solution = (
            solver.solve(assumptions)
            if cell["arm"] == "temporary_assumptions"
            else solver.solve()
        )
        if sat is None:
            result["solver_status"] = "unknown"
            result["outcome"] = "solver_unknown"
        elif sat is False:
            result["solver_status"] = "unsat"
            result["outcome"] = "correctness_alarm"
            result["error"] = "Checked true-key formula returned UNSAT"
        elif sat is not True:
            result["solver_status"] = "invalid_status"
            result["outcome"] = "correctness_alarm"
            result["error"] = "Solver returned a status outside True/False/None"
        else:
            result["solver_status"] = "sat"
            journal.transition("model_check_and_decode")
            try:
                model, assignment, key = _decode_model(formula, solution)
            except Exception as error:
                result["outcome"] = "correctness_alarm"
                result["error"] = f"{type(error).__name__}: {error}"
            else:
                result["model"] = model
                result["recovered_key"] = key
                result["formula_valid"] = True
                journal.transition("known_reference_verification")
                if not _reference_known_check(
                    cell["cipher"], cell["rounds"], key, public
                ):
                    result["outcome"] = "correctness_alarm"
                    result["error"] = (
                        "Decoded model failed independent known-block/key-schedule check"
                    )
                else:
                    result["verified_known"] = True
                    journal.transition("fresh_pair_verification")
                    truth = _read_pinned_json(
                        Path(payload["truth_path"]), payload["truth_sha256"]
                    )
                    verification = verify_key(
                        cell["cipher"], cell["rounds"], key, public, truth
                    )
                    if set(verification) != {
                        "verified_known",
                        "verified_fresh",
                        "assigned_key_match",
                    } or any(
                        type(verification[name]) is not bool for name in verification
                    ):
                        raise ValueError(
                            "Independent verifier returned malformed fields"
                        )
                    result.update(verification)
                    if not result["verified_known"]:
                        result["outcome"] = "correctness_alarm"
                        result["error"] = (
                            "Fresh verifier disagrees with prior known-block check"
                        )
                    elif result["verified_fresh"] and result["assigned_key_match"]:
                        result["outcome"] = "recovered"
                    elif result["verified_fresh"]:
                        result["outcome"] = "verified_different_key"
                    elif result["assigned_key_match"]:
                        result["outcome"] = "correctness_alarm"
                        result["error"] = (
                            "Truth key failed its stored fresh-pair checks"
                        )
                    else:
                        # SAT is valid, but this model is not freshly verified. Keep it
                        # as a valid nonunique-model outcome and continue the matrix.
                        result["outcome"] = "sat_model_not_fresh_verified"
    except EnvironmentRejected as error:
        result["outcome"] = "environment_rejected"
        result["error"] = f"{type(error).__name__}: {error}"
    except Exception as error:
        result["outcome"] = "worker_exception"
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        try:
            journal.transition(None)
        except Exception as error:
            result["outcome"] = "worker_exception"
            old_error = result.get("error")
            result["error"] = f"journal close failed: {error}; prior={old_error}"
        result["phase_seconds"] = dict(journal.completed)
        result["elapsed_seconds"] = time.perf_counter() - started
        result["completed_at_monotonic"] = time.perf_counter()
    return result


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: python -m neural_sat_speck.followups.conditioning_worker PAYLOAD.json"
        )
    payload_path = Path(sys.argv[1])
    payload = read_checked_json(payload_path)
    result = execute(payload)
    directory = Path(payload["directory"])
    atomic_json(directory / "worker_result.json", result)


if __name__ == "__main__":
    main()
