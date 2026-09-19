"""Native CryptoMiniSat CLI runner for secondary sensitivity baselines.

Implements the predeclared sensitivity baselines:
- sat_only_native_gauss_4obs: Native cryptominisat5 CLI with 4 SAT observations (2 pairs).
- sat_only_native_gauss_all_observations: Native cryptominisat5 CLI with 128 observations (64 pairs).

Accounting rules:
- Full wall-clock accounting includes DIMACS serialization, subprocess launch,
  pipe communication, log parsing, and independent fresh-pair verification.
- Outcome mapped to the 4 orthogonal failure taxonomy fields:
  counts_as_verified_recovery, counts_as_unconditional_failure,
  is_resolved_sat_state, failure_domain.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from typing import Any

from neural_sat_speck.attacks.recovery import (
    independent_verification_pairs,
    known_pairs_for_attack_instance,
    verify_master_key,
)
from neural_sat_speck.data.attack import AttackInstance
from neural_sat_speck.protocol.manifest_validator import classify_attack_outcome
from neural_sat_speck.sat.dimacs import (
    decode_master_key_from_dimacs_solution,
    export_speck_to_dimacs,
)


@dataclass(frozen=True)
class NativeGaussSensitivityResult:
    """Detailed result of a native CryptoMiniSat sensitivity evaluation."""

    method_id: str
    instance_id: str
    observations: int
    success: bool
    status: str
    category: str
    counts_as_verified_recovery: bool
    counts_as_unconditional_failure: bool
    is_resolved_sat_state: bool
    failure_domain: str
    recovered_master_key: tuple[int, int, int, int] | None
    timing_seconds: dict[str, float]
    matrix_diagnostics: dict[str, Any]
    returncode: int
    stdout_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "instance_id": self.instance_id,
            "observations": self.observations,
            "success": self.success,
            "status": self.status,
            "category": self.category,
            "counts_as_verified_recovery": self.counts_as_verified_recovery,
            "counts_as_unconditional_failure": self.counts_as_unconditional_failure,
            "is_resolved_sat_state": self.is_resolved_sat_state,
            "failure_domain": self.failure_domain,
            "recovered_master_key": list(self.recovered_master_key) if self.recovered_master_key else None,
            "timing_seconds": dict(self.timing_seconds),
            "matrix_diagnostics": dict(self.matrix_diagnostics),
            "returncode": self.returncode,
            "stdout_summary": self.stdout_summary,
        }


def parse_native_cms_log(stdout: str) -> dict[str, Any]:
    """Extract matrix usage and solver CPU time diagnostics from CryptoMiniSat stdout."""
    matrix_detected = False
    matrix_dimensions = None
    matrix_count_used = 0
    unused_reason = None
    autodisable_observed = False
    solver_version = None
    solver_cpu_seconds = 0.0

    for line in stdout.splitlines():
        line_s = line.strip()
        if "CryptoMiniSat version" in line_s:
            solver_version = line_s
        if "Good matrix" in line_s or "Good   matrix" in line_s:
            matrix_detected = True
        if "UNused matrix" in line_s:
            matrix_detected = True
            parts = line_s.split()
            if "x" in parts:
                idx = parts.index("x")
                matrix_dimensions = f"{parts[idx - 1]} x {parts[idx + 1]}"
        if "Too many columns in matrix" in line_s or "Too many rows in matrix" in line_s:
            unused_reason = line_s
        if "density:" in line_s and matrix_dimensions is None:
            parts = line_s.split()
            if "x" in parts:
                idx = parts.index("x")
                matrix_dimensions = f"{parts[idx - 1]} x {parts[idx + 1]}"
        if "Using 1 matrices" in line_s:
            matrix_count_used = 1
        elif "Using 0 matrices" in line_s:
            matrix_count_used = 0
        if "autodisable" in line_s.lower() or "auto-disable" in line_s.lower():
            autodisable_observed = True
        if "Total time" in line_s:
            match = re.search(r"Total time\s*:\s*([\d\.]+)", line_s)
            if match:
                try:
                    solver_cpu_seconds = float(match.group(1))
                except ValueError:
                    pass

    return {
        "solver_version": solver_version,
        "matrix_detected": matrix_detected,
        "matrix_dimensions": matrix_dimensions,
        "matrix_used": matrix_count_used > 0,
        "matrix_count_used": matrix_count_used,
        "unused_reason": unused_reason,
        "autodisable_observed": autodisable_observed,
        "solver_cpu_seconds": solver_cpu_seconds,
    }


def run_native_gauss_sensitivity(
    instance: AttackInstance,
    method_id: str = "sat_only_native_gauss_4obs",
    *,
    rounds: int = 7,
    timeout_seconds: float = 300.0,
    maxmatrixcols: int = 2500,
    autodisablegauss: int = 0,
    maxmatrixrows: int | None = None,
    threads: int = 1,
    binary_path: str = "cryptominisat5",
    verification_pairs_count: int = 2,
    verification_seed: int = 999999,
) -> NativeGaussSensitivityResult:
    """Execute exact SAT solving using native cryptominisat5 CLI with full wall-clock accounting.

    Parameters
    ----------
    instance : AttackInstance
        The attack instance to solve.
    method_id : str
        Sensitivity method identifier ('sat_only_native_gauss_4obs' or 'sat_only_native_gauss_all_observations').
    rounds : int
        Cipher round count (default: 7).
    timeout_seconds : float
        Per-instance solver timeout in seconds (default: 300.0).
    maxmatrixcols : int
        Maximum matrix column count for Gauss elimination (default: 2500).
    autodisablegauss : int
        Whether to auto-disable Gauss elimination (0=no, 1=yes; default: 0).
    maxmatrixrows : int | None
        Optional maximum matrix rows (e.g. 50000 for 128 observations).
    threads : int
        Thread count (strictly 1 under protocol).
    binary_path : str
        Name or path of cryptominisat5 executable.
    verification_pairs_count : int
        Number of fresh pairs for verification (default: 2).
    verification_seed : int
        RNG seed for fresh verification pairs (default: 999999).

    Returns
    -------
    NativeGaussSensitivityResult
        Complete accounting and diagnostic report.
    """
    all_known = known_pairs_for_attack_instance(instance)
    if method_id == "sat_only_native_gauss_4obs":
        known_pairs = all_known[:4]
        observations = 4
    elif method_id == "sat_only_native_gauss_all_observations":
        known_pairs = all_known
        observations = len(all_known)
    else:
        raise ValueError(f"Unsupported sensitivity method_id: {method_id!r}")

    verification_pairs = independent_verification_pairs(
        instance.master_key,
        rounds=rounds,
        pair_count=verification_pairs_count,
        seed=verification_seed,
    )

    t_wall_start = time.perf_counter()

    # 1. DIMACS serialization
    t_dimacs_start = time.perf_counter()
    dimacs_model = export_speck_to_dimacs(known_pairs=known_pairs, rounds=rounds)
    dimacs_seconds = time.perf_counter() - t_dimacs_start

    elapsed_before_sub = time.perf_counter() - t_wall_start
    remaining_seconds = max(0.0, timeout_seconds - elapsed_before_sub)

    timed_out = False
    stdout = ""
    stderr = ""
    returncode = -1
    sub_seconds = 0.0
    cmd: list[str] = []

    if remaining_seconds <= 0.0:
        timed_out = True
    else:
        # 2. Subprocess execution with clamped maxtime
        cmd = [
            binary_path,
            "--verb", "1",
            "--random", "0",
            "--threads", str(threads),
            "--maxmatrixcols", str(maxmatrixcols),
            "--autodisablegauss", str(autodisablegauss),
            "--maxtime", str(max(1, int(remaining_seconds))),
        ]
        if maxmatrixrows is not None:
            cmd.extend(["--maxmatrixrows", str(maxmatrixrows)])

        t_sub_start = time.perf_counter()
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = proc.communicate(
                input=dimacs_model.dimacs_text,
                timeout=remaining_seconds + 2.0,
            )
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            returncode = proc.returncode
            timed_out = True
        except FileNotFoundError as err:
            returncode = 127
            stderr = str(err)
        finally:
            sub_seconds = time.perf_counter() - t_sub_start

    # 3. Log parsing and solution extraction
    t_parse_start = time.perf_counter()
    diag = parse_native_cms_log(stdout)
    diag["command"] = " ".join(cmd) if cmd else ""
    diag["returncode"] = returncode
    diag["stderr"] = stderr

    is_sat = "s SATISFIABLE" in stdout
    is_unsat = "s UNSATISFIABLE" in stdout
    is_unknown = "s UNKNOWN" in stdout

    recovered_key = None
    parse_seconds = time.perf_counter() - t_parse_start

    # 4. Fresh-pair verification
    t_verif_start = time.perf_counter()
    verified = False
    status = "indeterminate"

    elapsed_now = time.perf_counter() - t_wall_start
    if timed_out or returncode == 15 or elapsed_now >= timeout_seconds:
        if elapsed_now >= timeout_seconds:
            status = "global_timeout"
        else:
            status = "solver_timeout"
    elif returncode == 127:
        status = "solver_exception"
    elif is_sat:
        t_decode_start = time.perf_counter()
        recovered_key = decode_master_key_from_dimacs_solution(
            stdout, dimacs_model.master_key_bits
        )
        parse_seconds += (time.perf_counter() - t_decode_start)

        if recovered_key is not None:
            verif_res = verify_master_key(
                recovered_key,
                known_pairs,
                verification_pairs,
                rounds=rounds,
            )
            verified = verif_res.verified
            status = "sat_and_verified" if verified else "verification_failure"
        else:
            status = "decoding_failure"
    elif is_unsat or returncode == 20:
        status = "unsat"
    elif is_unknown:
        status = "solver_unknown"
    else:
        status = "solver_exception"

    verif_seconds = time.perf_counter() - t_verif_start
    total_wall_seconds = time.perf_counter() - t_wall_start

    classified = classify_attack_outcome(status, success=verified, unresolved_timeouts=1 if status == "solver_timeout" else 0)

    # First and last 5 lines of stdout for audit
    lines = stdout.splitlines()
    summary_lines = lines[:5] + (["..."] if len(lines) > 10 else []) + lines[-5:] if len(lines) > 5 else lines
    stdout_summary = "\n".join(summary_lines)

    return NativeGaussSensitivityResult(
        method_id=method_id,
        instance_id=instance.instance_id,
        observations=observations,
        success=verified,
        status=status,
        category=classified["category"],
        counts_as_verified_recovery=classified["counts_as_verified_recovery"],
        counts_as_unconditional_failure=classified["counts_as_unconditional_failure"],
        is_resolved_sat_state=classified["is_resolved_sat_state"],
        failure_domain=classified["failure_domain"],
        recovered_master_key=recovered_key,
        timing_seconds={
            "dimacs_generation": round(dimacs_seconds, 6),
            "subprocess_wall": round(sub_seconds, 6),
            "solver_cpu": round(diag["solver_cpu_seconds"], 6),
            "parse": round(parse_seconds, 6),
            "verification": round(verif_seconds, 6),
            "total_wall": round(total_wall_seconds, 6),
        },
        matrix_diagnostics=diag,
        returncode=returncode,
        stdout_summary=stdout_summary,
    )
