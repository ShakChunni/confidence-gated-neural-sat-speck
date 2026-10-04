"""Strict command construction and result parsing for backend diagnostics.

The parser recognizes SAT/UNSAT exit codes 10/20 and UNKNOWN exit code 0 for
CaDiCaL. CryptoMiniSat 5.14.7 reports UNKNOWN as `s INDETERMINATE` with exit
code 15; zero-exit normalization is deliberately not enabled. A SAT
result is accepted only when its model is terminated and assigns every
requested variable. It does not validate the model against a formula; callers
must do that separately.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_CMS_PROFILES = {
    "cms_native_default",
    "cms_native_gauss",
    "cms_native_no_gauss",
    "cms_cnf_no_gauss",
}
_KNOWN_PROFILES = _CMS_PROFILES | {"cadical_cnf"}
_ENGINE_ALIASES = {
    "cms": "cms",
    "cryptominisat": "cms",
    "cryptominisat5": "cms",
    "cadical": "cadical",
}


def _required_path(value: str, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def solver_arguments(
    profile: str,
    cms_path: str,
    cadical_path: str,
    formula_path: str,
) -> list[str]:
    """Return an argv vector for one fixed backend profile.

    Paths remain separate argv elements, so spaces and shell metacharacters do
    not receive shell interpretation. `cms_native_no_gauss` and
    `cms_cnf_no_gauss` set the documented matrix-treatment limit to zero;
    launch metadata must preserve that as a configured policy, not as proof
    that a particular runtime exercised or skipped Gaussian elimination.
    """
    if not isinstance(profile, str) or profile not in _KNOWN_PROFILES:
        raise ValueError(f"unknown solver profile: {profile!r}")
    formula = _required_path(formula_path, "formula_path")

    if profile == "cadical_cnf":
        return [_required_path(cadical_path, "cadical_path"), "-q", formula]

    cms = _required_path(cms_path, "cms_path")
    base = ["--verb", "1", "--threads", "1"]
    if profile == "cms_native_default":
        return [cms, *base, formula]

    # G/X differ only in the matrix cap. X/C differ only in input representation;
    # all three disable long-XOR discovery, retaining explicitly supplied XORs.
    matrix_options = [
        "--maxmatrixrows",
        "2000",
        "--maxmatrixcols",
        "2500",
        "--autodisablegauss",
        "0",
        "--minmatrixrows",
        "3",
        "--gaussusefulcutoff",
        "0.2",
    ]
    matrix_limit = (
        "0" if profile in {"cms_native_no_gauss", "cms_cnf_no_gauss"} else "5"
    )
    argv = [
        cms,
        *base,
        *matrix_options,
        "--maxnummatrices",
        matrix_limit,
        "--xor",
        "0",
    ]
    argv.append(formula)
    return argv


def _gauss_evidence(stdout: str) -> dict[str, object]:
    matching_lines = [
        line
        for line in stdout.splitlines()
        if "gauss" in line.casefold()
        or "gaussian" in line.casefold()
        or "matrix" in line.casefold()
        or re.search(r"\[g\s*\d+\]", line)
    ]
    return {
        "matching_log_lines": matching_lines,
        "matrix_count_samples": [
            int(match.group(1))
            for match in re.finditer(r"\[matrix\] Using ([0-9]+) matrices", stdout)
        ],
        "is_proof_of_gauss_use": False,
    }


def _result(
    status: str,
    assignment: dict[int, bool],
    error: str | None,
    gauss_evidence: dict[str, object],
) -> dict[str, object]:
    return {
        "status": status,
        "assignment": assignment,
        "error": error,
        "gauss_evidence": gauss_evidence,
    }


def parse_solver_output(
    engine: str,
    returncode: int,
    stdout: str,
    *,
    required_variables: Sequence[int],
    num_vars: int,
) -> dict[str, object]:
    """Parse SAT/UNSAT/UNKNOWN status and signed DIMACS `v` model lines.

    Model literals may span multiple `v` lines, with one terminal zero at the
    end of the final line. Comments and ordinary solver log lines are ignored.
    Repeated status lines, model text on non-SAT outcomes, malformed literals,
    model truncation, contradictory status/exit code, and incomplete requested
    variables produce `status="error"` with an empty assignment. CMS's native
    UNKNOWN exit code 15 and `INDETERMINATE` status are normalized to `unknown`.
    """
    evidence = _gauss_evidence(stdout if isinstance(stdout, str) else "")

    def error(message: str) -> dict[str, object]:
        return _result("error", {}, message, evidence)

    normalized_engine = (
        _ENGINE_ALIASES.get(engine.casefold()) if isinstance(engine, str) else None
    )
    if normalized_engine is None:
        return error(f"unsupported solver engine: {engine!r}")
    if not isinstance(stdout, str):
        return error("stdout must be a string")
    if type(returncode) is not int:
        return error(f"unexpected solver exit code: {returncode!r}")
    if type(num_vars) is not int or num_vars < 0:
        return error("num_vars must be a non-negative integer")

    try:
        required = list(required_variables)
    except TypeError:
        return error("required_variables must be a sequence of variable IDs")
    if any(type(variable) is not int for variable in required):
        return error("required_variables must contain only integer variable IDs")
    if len(set(required)) != len(required):
        return error("required_variables contains duplicate IDs")
    if any(variable < 1 or variable > num_vars for variable in required):
        return error("required variable ID is outside 1..num_vars")

    statuses: list[str] = []
    model_lines: list[list[str]] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        marker = fields[0]
        if marker.startswith("#") or marker == "c":
            continue
        if marker == "s":
            if len(fields) != 2:
                return error(f"malformed status line: {raw_line!r}")
            status_token = fields[1].upper()
            if status_token not in {
                "SATISFIABLE",
                "UNSATISFIABLE",
                "UNKNOWN",
                "INDETERMINATE",
            }:
                return error(f"unrecognized status line: {raw_line!r}")
            if normalized_engine == "cms" and status_token == "INDETERMINATE":
                status_token = "UNKNOWN"
            elif status_token == "INDETERMINATE":
                return error(
                    f"INDETERMINATE status is not valid for {normalized_engine}"
                )
            statuses.append(status_token)
        elif marker == "v":
            model_lines.append(fields[1:])

    if len(statuses) > 1:
        return error("repeated solver status lines")

    exit_statuses = {
        "cms": {
            10: "SATISFIABLE",
            15: "UNKNOWN",
            20: "UNSATISFIABLE",
        },
        "cadical": {0: "UNKNOWN", 10: "SATISFIABLE", 20: "UNSATISFIABLE"},
    }
    if returncode not in exit_statuses[normalized_engine]:
        return error(f"unexpected {normalized_engine} exit code: {returncode!r}")
    expected = exit_statuses[normalized_engine][returncode]
    if statuses and statuses[0] != expected:
        return error(
            f"status {statuses[0]} contradicts exit code {returncode} ({expected})"
        )
    if not statuses:
        return error(f"exit code {returncode} has no explicit solver status")

    if expected == "UNKNOWN":
        if model_lines:
            return error("UNKNOWN result unexpectedly contains a model")
        return _result("unknown", {}, None, evidence)
    if expected == "UNSATISFIABLE":
        if model_lines:
            return error("UNSAT result unexpectedly contains model lines")
        return _result("unsat", {}, None, evidence)

    if not model_lines:
        return error("SAT result has no model lines")

    assignment: dict[int, bool] = {}
    terminated = False
    for fields in model_lines:
        if terminated:
            return error("repeated model line after terminal zero")
        for field_index, token in enumerate(fields):
            if re.fullmatch(r"[+-]?[0-9]+", token) is None:
                return error(f"non-integer model literal: {token!r}")
            literal = int(token, 10)
            if literal == 0:
                if field_index != len(fields) - 1:
                    return error("model contains literals after terminal zero")
                terminated = True
                break
            variable = abs(literal)
            if variable < 1 or variable > num_vars:
                return error(f"model variable {variable} is outside 1..num_vars")
            value = literal > 0
            previous = assignment.get(variable)
            if previous is not None and previous != value:
                return error(f"conflicting model literals for variable {variable}")
            assignment[variable] = value

    if not terminated:
        return error("SAT model is truncated: missing terminal zero")
    missing = [variable for variable in required if variable not in assignment]
    if missing:
        return error(f"SAT model omits required variables: {missing}")
    return _result("sat", assignment, None, evidence)
