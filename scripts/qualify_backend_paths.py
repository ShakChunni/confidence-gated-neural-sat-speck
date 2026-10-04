#!/usr/bin/env python3
"""Bounded public-fixture qualification, never a project attack campaign.

Uses an explicitly supplied committed source tree. Keys/plaintexts below are
public constants; no seeded cohort or protected data is loaded. Native toy
calls have five-second subprocess deadlines; full guarded cells have ten.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import uuid
from pathlib import Path


def qualify(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("Preserve earlier qualification evidence")
    output.mkdir(parents=True)
    sys.path.insert(0, str(source / "src"))
    from neural_sat_speck.backend.artifacts import cipher_functions, verify_key
    from neural_sat_speck.backend.environment import sleep_assertion
    from neural_sat_speck.backend.evidence import audit_cell_evidence
    from neural_sat_speck.backend.formula import build_formula
    from neural_sat_speck.backend.guard import (
        PROFILES,
        THREAD_ENVIRONMENT,
        atomic_json,
        read_checked_json,
        sha256_file,
    )
    from neural_sat_speck.backend.runner import now, run_cell
    from neural_sat_speck.backend.solver_io import (
        parse_solver_output,
        solver_arguments,
    )
    from neural_sat_speck.sat.pure_cnf import (
        convert_native_xor_dimacs_to_pure_cnf,
    )

    python = Path(sys.executable).absolute()  # retain the venv symlink
    cms = Path("/opt/homebrew/bin/cryptominisat5").resolve()
    cadical = Path("/opt/homebrew/bin/cadical").resolve()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    report = {
        "classification": "ENGINEERING_PUBLIC_CONSTANT_FIXTURES",
        "source_root": str(source),
        "source_commit": commit,
        "harness_sha256": sha256_file(Path(__file__)),
        "tool_hashes": {str(p): sha256_file(p) for p in (python, cms, cadical)},
        "passed": False,
        "toy_cases": [],
        "worker_cases": [],
        "limits": [
            "Not attack recovery or timing evidence",
            "Pre-exec calling-thread receipts do not certify all native threads",
            "Configured Gauss options do not establish elimination was exercised",
        ],
    }
    atomic_json(output / "qualification.json", report)
    # Three variables with a unique SAT witness, and its contradictory variant.
    for unsat in (False, True):
        native = (
            f"p cnf 3 {4 if unsat else 3}\n1 0\n-2 0\n"
            + ("3 0\n" if unsat else "")
            + "x 1 2 3 0\n"
        )
        pure, count = convert_native_xor_dimacs_to_pure_cnf(native, 3)
        for profile in PROFILES[1:]:
            ident = f"toy_{profile}_{'unsat' if unsat else 'sat'}"
            path = output / (
                ident + (".cnf" if profile in PROFILES[-2:] else ".dimacs")
            )
            path.write_text(pure if profile in PROFILES[-2:] else native)
            argv = solver_arguments(profile, str(cms), str(cadical), str(path))
            proc = subprocess.run(argv, text=True, capture_output=True, timeout=5)
            (output / f"{ident}_stdout.txt").write_text(proc.stdout)
            (output / f"{ident}_stderr.txt").write_text(proc.stderr)
            parsed = parse_solver_output(
                "cadical" if profile == "cadical_cnf" else "cms",
                proc.returncode,
                proc.stdout,
                num_vars=count if profile in PROFILES[-2:] else 3,
                required_variables=[1, 2, 3],
            )
            expected = "unsat" if unsat else "sat"
            if parsed["status"] != expected or (
                not unsat and parsed["assignment"] != {1: True, 2: False, 3: False}
            ):
                raise ValueError(f"Public native toy qualification failed: {ident}")
            report["toy_cases"].append(
                {
                    "id": ident,
                    "argv": argv,
                    "returncode": proc.returncode,
                    "status": expected,
                }
            )
    from pycryptosat import Solver

    for unsat in (False, True):
        solver = Solver(verbose=0, threads=1)
        solver.add_clause([1])
        solver.add_clause([-2])
        if unsat:
            solver.add_clause([3])
        solver.add_xor_clause([1, 2, 3], True)
        sat, assignment = solver.solve(time_limit=1.0)
        if sat is not (not unsat) or (
            not unsat and list(assignment[1:4]) != [True, False, False]
        ):
            raise ValueError("Public wrapper toy qualification failed")
        report["toy_cases"].append(
            {"id": f"toy_wrapper_{unsat}", "status": "unsat" if unsat else "sat"}
        )

    # Four true round keys at R4 determine the public master key in both ciphers.
    # These twelve cases exercise the actual child/guardian/native/verification paths.
    key = (0x1918, 0x1110, 0x0908, 0x0100)
    pts = ((0x1021, 0x2345), (0xFEDC, 0xAB89), (0x1032, 0x2362), (0xFEBD, 0xAB7C))
    fresh_pts = ((0x3245, 0x6789), (0x1234, 0x5678))
    ledger_path = output / "qualification_ledger.json"
    ledger = {"starts": {}, "records": {}}
    with sleep_assertion():
        for cipher in ("speck32_64", "simon32_64"):
            encrypt, expand = cipher_functions(cipher)
            public = {
                "rounds": 4,
                "pairs": [],
                "trailing_round_keys": list(expand(key, 4)),
            }
            for first, second in ((pts[0], pts[1]), (pts[2], pts[3])):
                public["pairs"].append(
                    {
                        "plaintext0": list(first),
                        "plaintext1": list(second),
                        "ciphertext0": list(encrypt(first, key, 4)),
                        "ciphertext1": list(encrypt(second, key, 4)),
                    }
                )
            truth = {
                "master_key": list(key),
                "verification_pairs": [
                    [list(pt), list(encrypt(pt, key, 4))] for pt in fresh_pts
                ],
            }
            formula = build_formula(cipher, public, 4, 4)
            case = f"public_{cipher}_r4_m4"
            paths = {"python": python, "cms": cms, "cadical": cadical}
            for kind, value in (
                ("public", public),
                ("truth", truth),
                (
                    "varmap",
                    {
                        "native_num_vars": formula["num_vars"],
                        "pure_cnf_num_vars": formula["pure_cnf_num_vars"],
                    },
                ),
            ):
                path = output / f"{case}_{kind}.json"
                atomic_json(path, value)
                paths[f"{case}:{kind}"] = path
            for kind, text in (
                ("native", formula["native_dimacs"]),
                ("cnf", formula["pure_cnf"]),
            ):
                path = output / f"{case}_{kind}.dimacs"
                path.write_text(text)
                paths[f"{case}:{kind}"] = path
            for profile in PROFILES:
                cell = {
                    "cell_id": f"{case}__{profile}",
                    "case_id": case,
                    "cipher": cipher,
                    "rounds": 4,
                    "observations": 4,
                    "trailing_keys": 4,
                    "key_index": 0,
                    "profile": profile,
                }
                invocation, started = uuid.uuid4().hex, now()
                directory = output / "workers" / f"{cell['cell_id']}_{invocation}"
                tmpdir = source / "runtime" / "qualification_tmp" / invocation
                tmpdir.mkdir(parents=True)
                payload = {
                    "cell": cell,
                    "directory": str(directory),
                    "repo_root": str(source),
                    "invocation": invocation,
                    "started_utc": started,
                    "record_process": str(ledger_path),
                    "public_path": str(paths[f"{case}:public"]),
                    "truth_path": str(paths[f"{case}:truth"]),
                    "native_sha256": sha256_file(paths[f"{case}:native"]),
                    "cnf_sha256": sha256_file(paths[f"{case}:cnf"]),
                    "cms_path": str(cms),
                    "cadical_path": str(cadical),
                    "python_path": str(python),
                    "source_commit": commit,
                }
                ledger["starts"][cell["cell_id"]] = {
                    "cell_id": cell["cell_id"],
                    "invocation": invocation,
                    "started_utc": started,
                    "process_group": None,
                }
                atomic_json(ledger_path, ledger)
                env = {
                    "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                    "LANG": "C",
                    "LC_ALL": "C",
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPYCACHEPREFIX": str(tmpdir / "pycache"),
                    **THREAD_ENVIRONMENT,
                    "TMPDIR": str(tmpdir),
                    "PYTHONPATH": str(source / "src"),
                }
                result = run_cell(payload, env, timeout=10.0)
                atomic_json(directory / "coordinator_received.json", result)
                result["raw_artifacts"] = {
                    p.name: sha256_file(p)
                    for p in directory.iterdir()
                    if p.is_file() and not p.name.endswith(".tmp")
                }
                result.update(pre_dispatch_seconds=0.0, post_dispatch_audit_seconds=0.0)
                ledger = read_checked_json(ledger_path)
                ledger["records"][cell["cell_id"]] = result
                atomic_json(ledger_path, ledger)  # retain failures before asserting
                if (
                    result["outcome"] != "recovered"
                    or result.get("guardian_healthy") is not True
                ):
                    raise ValueError(
                        f"Public full worker qualification failed: {cell['cell_id']}: {result}"
                    )
                if not all(
                    verify_key(
                        cipher, 4, result["recovered_key"], public, truth
                    ).values()
                ):
                    raise ValueError("Independent public fresh verification failed")
                audit = audit_cell_evidence(
                    result,
                    ledger["starts"][cell["cell_id"]],
                    {
                        "cells": [cell],
                        "source_commit_at_preparation": commit,
                        "deadline_seconds": 10.0,
                    },
                    paths,
                    ledger_path,
                )
                report["worker_cases"].append(audit)
                atomic_json(output / "qualification.json", report)
                print(f"Qualified {cell['cell_id']}", flush=True)
    report["passed"] = True
    atomic_json(output / "qualification.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    qualify(arguments.source_root, arguments.output_dir)
