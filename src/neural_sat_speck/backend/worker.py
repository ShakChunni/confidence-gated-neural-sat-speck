"""One isolated backend cell; the coordinator enforces the hard deadline."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .environment import power_sample
from .guard import atomic_json


class Journal:
    def __init__(self, path: Path):
        self.path = path
        self.started = time.perf_counter()
        self.phase = None
        self.phase_started = None
        self.completed = {}

    def transition(self, phase: str | None) -> None:
        now = time.perf_counter()
        if self.phase is not None:
            self.completed[self.phase] = (
                self.completed.get(self.phase, 0.0) + now - self.phase_started
            )
        self.phase, self.phase_started = phase, now if phase else None
        atomic_json(
            self.path,
            {
                "phase": phase,
                "phase_started_monotonic": self.phase_started,
                "completed_seconds": self.completed,
                "worker_started_monotonic": self.started,
                "observed_monotonic": now,
            },
        )


def _complete_assignment(formula: dict, assignment: dict[int, bool]) -> bool:
    """Check the original constraints, including units, independently of status."""
    if any(
        type(assignment.get(v)) is not bool for v in range(1, formula["num_vars"] + 1)
    ):
        return False
    for clause in formula["cnf_clauses"] + [
        [lit] for lit in formula["assumption_literals"]
    ]:
        if not any(assignment[abs(lit)] == (lit > 0) for lit in clause):
            return False
    for xor in formula["xor_clauses"]:
        parity = False
        for lit in xor["literals"]:
            parity ^= assignment[abs(lit)] ^ (lit < 0)
        if parity != xor["rhs"]:
            return False
    return True


def execute(payload: dict) -> dict:
    from .artifacts import verify_key
    from .formula import build_formula
    from .solver_io import parse_solver_output, solver_arguments

    directory = Path(payload["directory"])
    journal = Journal(directory / "journal.json")
    cell = payload["cell"]
    result = {
        **cell,
        "outcome": "worker_exception",
        "verified_known": False,
        "verified_fresh": False,
        "assigned_key_match": False,
        "recovered_key": None,
    }
    try:
        journal.transition("input_loading")
        public = json.loads(Path(payload["public_path"]).read_bytes())
        journal.transition("environment_observation")
        from neural_sat_speck.protocol.worker_telemetry import certify_worker_or_raise

        receipt = certify_worker_or_raise(Path(payload["repo_root"]))
        atomic_json(directory / "python_worker_receipt.json", receipt.to_dict())
        if receipt.git_commit != payload["source_commit"]:
            raise ValueError("Python worker receipt source revision drift")
        power = power_sample()
        atomic_json(directory / "power_at_worker_start.json", power)
        if power.get("accepted") is False:
            raise RuntimeError(
                "Worker execution scheduling certification failed: power/sleep sample rejected"
            )
        journal.transition("formula_construction")
        formula = build_formula(
            cell["cipher"],
            public,
            cell["observations"],
            cell["trailing_keys"],
            include_cnf=False,
        )
        native = formula["native_dimacs"]
        if hashlib.sha256(native.encode()).hexdigest() != payload["native_sha256"]:
            raise ValueError("Rebuilt native formula differs from frozen artifact")
        pure = cell["profile"] in {"cms_cnf_no_gauss", "cadical_cnf"}
        formula_text, num_vars = native, formula["num_vars"]
        if pure:
            journal.transition("xor_to_cnf_conversion")
            from neural_sat_speck.sat.pure_cnf import (
                convert_native_xor_dimacs_to_pure_cnf,
            )

            formula_text, num_vars = convert_native_xor_dimacs_to_pure_cnf(
                native, formula["num_vars"]
            )
            if (
                hashlib.sha256(formula_text.encode()).hexdigest()
                != payload["cnf_sha256"]
            ):
                raise ValueError("Rebuilt CNF differs from frozen artifact")
            formula["counts"].update(
                pure_cnf_clauses=int(formula_text.splitlines()[0].split()[3]),
                auxiliary_start=formula["num_vars"] + 1
                if num_vars > formula["num_vars"]
                else None,
                auxiliary_end=num_vars if num_vars > formula["num_vars"] else None,
            )
        result["formula_counts"] = formula["counts"]
        result["native_num_vars"] = formula["num_vars"]
        result["submitted_num_vars"] = num_vars
        journal.transition("formula_io_and_solver_loading")
        profile = cell["profile"]
        if profile == "cms_wrapper":
            from pycryptosat import Solver

            solver = Solver(verbose=0, threads=1)
            for clause in formula["cnf_clauses"]:
                solver.add_clause(clause)
            for xor in formula["xor_clauses"]:
                solver.add_xor_clause(xor["literals"], xor["rhs"])
            for lit in formula["assumption_literals"]:
                solver.add_clause([lit])
            journal.transition("solving")
            sat, solution = solver.solve()
            journal.transition("model_parsing_and_decode")
            assignment = (
                {
                    v: solution[v]
                    for v in range(1, len(solution))
                    if type(solution[v]) is bool
                }
                if solution
                else {}
            )
            parsed = {
                "status": "sat"
                if sat is True
                else "unsat"
                if sat is False
                else "unknown",
                "assignment": assignment,
                "error": None,
            }
        else:
            path = directory / ("input.cnf" if pure else "input.dimacs")
            path.write_text(formula_text)
            arguments = solver_arguments(
                profile, payload["cms_path"], payload["cadical_path"], str(path)
            )
            launch_path = directory / "native_launch.json"
            atomic_json(
                launch_path,
                {
                    "repo_root": payload["repo_root"],
                    "arguments": arguments,
                    "receipt": str(directory / "native_launch_receipt.json"),
                },
            )
            result["solver_arguments"] = arguments
            with (
                (directory / "solver_stdout.txt").open("w") as stdout,
                (directory / "solver_stderr.txt").open("w") as stderr,
            ):
                journal.transition("native_launch_parse_and_solve")
                native_proc = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "neural_sat_speck.backend.native_launcher",
                        str(launch_path),
                    ],
                    stdout=stdout,
                    stderr=stderr,
                )
                atomic_json(
                    directory / "native_process.json",
                    {
                        "pid": native_proc.pid,
                        "process_group": os.getpgrp(),
                        "argv": arguments,
                    },
                )
                code = native_proc.wait()
            from .guard import read_checked_json

            launch_receipt = read_checked_json(directory / "native_launch_receipt.json")
            if (
                launch_receipt.get("pid") != native_proc.pid
                or launch_receipt.get("is_certified") is not True
                or launch_receipt.get("solver_arguments") != arguments
            ):
                raise ValueError(
                    "Native launch receipt does not match the actual child invocation"
                )
            if launch_receipt.get("git_commit") != payload["source_commit"]:
                raise ValueError("Native receipt source revision drift")
            result["native_launch_receipt"] = launch_receipt
            journal.transition("model_parsing_and_decode")
            stdout_text = (directory / "solver_stdout.txt").read_text()
            parsed = parse_solver_output(
                "cadical" if profile == "cadical_cnf" else "cms",
                code,
                stdout_text,
                required_variables=list(range(1, num_vars + 1)),
                num_vars=num_vars,
            )
            assignment = parsed["assignment"]
            result["solver_returncode"] = code
            result["gauss_evidence"] = parsed.get("gauss_evidence")
        result["solver_status"] = parsed["status"]
        if parsed["status"] == "unsat":
            result["outcome"] = "unsat_alarm"
        elif parsed["status"] == "unknown":
            result["outcome"] = "unknown"
        elif parsed["status"] != "sat":
            result["outcome"], result["error"] = "solver_error", parsed.get("error")
        elif not _complete_assignment(formula, assignment):
            result["outcome"], result["error"] = (
                "solver_error",
                "Returned assignment violates original formula or is incomplete",
            )
        else:
            key = [
                sum(int(assignment[var]) << bit for bit, var in enumerate(word))
                for word in formula["varmap"]["master_key"]
            ]
            result["recovered_key"] = key
            journal.transition("verification")
            # Truth is first loaded here, after the solver call and decode.
            truth = json.loads(Path(payload["truth_path"]).read_bytes())
            result.update(
                verify_key(cell["cipher"], cell["rounds"], key, public, truth)
            )
            if result["verified_known"] and result["verified_fresh"]:
                result["outcome"] = (
                    "recovered"
                    if result["assigned_key_match"]
                    else "different_verified_key"
                )
            else:
                result["outcome"] = "verification_failed"
        journal.transition(None)
    except Exception as error:
        result["outcome"] = (
            "environment_rejected"
            if "scheduling certification" in str(error)
            else "worker_exception"
        )
        result["error"] = f"{type(error).__name__}: {error}"
        journal.transition(None)
    result["completed_phase_seconds"] = journal.completed
    result["worker_seconds"] = time.perf_counter() - journal.started
    return result


def main() -> None:
    path = Path(sys.argv[1])
    payload = json.loads(path.read_bytes())
    atomic_json(Path(payload["directory"]) / "result.json", execute(payload))


if __name__ == "__main__":
    main()
