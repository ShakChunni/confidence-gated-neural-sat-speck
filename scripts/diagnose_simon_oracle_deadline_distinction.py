#!/usr/bin/env python3
"""Diagnostic: 45-second advisory oracle request vs true 300-second parent-bounded solve.

Predeclared diagnostic per Target B of docs/DEPTH_BACKEND_DESIGN_TASK_20261003.md:
- Uses the fixed 8 opened demo keys (seed 999999, R=10, 4 observations / 2 pairs).
- Evaluates conditioned exact CryptoMiniSat solving on true two trailing subkeys (k8, k9):
  Treatment 1: 45.0-second advisory solver request (matching the confirmation oracle diagnostic).
  Treatment 2: 300.0-second parent monotonic watchdog deadline.
- Distinguishes whether the 10/64 unresolved cells in the confirmation campaign
  represent true search intractability or truncation by the 45-second advisory cutoff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from neural_sat_speck.data.seed_policy import CLI_DEMO_SEED  # noqa: E402
from neural_sat_speck.protocol.worker_telemetry import (  # noqa: E402
    certify_worker_or_raise,
    observe_worker_telemetry,
)
from neural_sat_speck.simon.attack_instance import (  # noqa: E402
    SimonAttackInstanceConfig,
    generate_simon_attack_instances,
)
from neural_sat_speck.simon.key_schedule import expand_key  # noqa: E402
from neural_sat_speck.simon.sat_model import SimonSatModel  # noqa: E402
from neural_sat_speck.simon.verification import verify_simon_master_key  # noqa: E402


@dataclass(frozen=True)
class OracleDiagnosticCell:
    instance_index: int
    master_key: list[int]
    true_trailing_keys: list[int]
    treatment: str  # "advisory_45s" or "parent_bounded_300s"
    solve_status: str
    solve_wall_seconds: float
    verified: bool
    timed_out: bool


def run_oracle_diagnostic(
    output_path: Path | None = None,
    verify_telemetry: bool = True,
) -> dict[str, Any]:
    """Execute the paired 45s advisory vs 300s parent-bounded oracle diagnostic."""
    if verify_telemetry:
        certify_worker_or_raise(repo_root=REPO_ROOT)
    telemetry = observe_worker_telemetry(repo_root=REPO_ROOT)

    cfg = SimonAttackInstanceConfig(
        instance_count=8,
        pair_count=64,  # Generate 64-pair cohort, use first 2 pairs (4 obs)
        rounds=10,
        seed=CLI_DEMO_SEED,
        split_role="demo",
    )
    instances = generate_simon_attack_instances(cfg)
    cells: list[OracleDiagnosticCell] = []

    for idx, inst in enumerate(instances):
        # 4 observations = first 2 pairs
        pairs = inst.pairs[:2]
        known_obs = [(p.plaintext0, p.ciphertext0) for p in pairs] + [
            (p.plaintext1, p.ciphertext1) for p in pairs
        ]
        r_keys = expand_key(inst.master_key, 10)
        trailing_2 = [r_keys[8], r_keys[9]]

        # Treatment 1: 45.0s advisory solve
        model_45 = SimonSatModel.from_known_pairs(known_obs, rounds=10)
        assump_45 = model_45.assumptions_for_trailing_round_subkeys(trailing_2)
        t0 = time.perf_counter()
        res_45 = model_45.solve(assump_45, time_limit=45.0)
        elapsed_45 = time.perf_counter() - t0

        verified_45 = False
        if res_45.satisfiable and res_45.solution is not None:
            dec_key = model_45.master_key_from_solution(res_45.solution)
            v = verify_simon_master_key(dec_key, known_obs, inst.verification_pairs, rounds=10)
            verified_45 = v.verified

        cells.append(
            OracleDiagnosticCell(
                instance_index=idx,
                master_key=list(inst.master_key),
                true_trailing_keys=trailing_2,
                treatment="advisory_45s",
                solve_status=res_45.status,
                solve_wall_seconds=elapsed_45,
                verified=verified_45,
                timed_out=bool(res_45.timed_out),
            )
        )

        # Treatment 2: 300.0s parent-bounded solve
        model_300 = SimonSatModel.from_known_pairs(known_obs, rounds=10)
        assump_300 = model_300.assumptions_for_trailing_round_subkeys(trailing_2)
        t0 = time.perf_counter()
        res_300 = model_300.solve(assump_300, time_limit=300.0)
        elapsed_300 = time.perf_counter() - t0

        verified_300 = False
        if res_300.satisfiable and res_300.solution is not None:
            dec_key = model_300.master_key_from_solution(res_300.solution)
            v = verify_simon_master_key(dec_key, known_obs, inst.verification_pairs, rounds=10)
            verified_300 = v.verified

        cells.append(
            OracleDiagnosticCell(
                instance_index=idx,
                master_key=list(inst.master_key),
                true_trailing_keys=trailing_2,
                treatment="parent_bounded_300s",
                solve_status=res_300.status,
                solve_wall_seconds=elapsed_300,
                verified=verified_300,
                timed_out=bool(res_300.timed_out),
            )
        )

    summary = {
        "telemetry": telemetry.to_dict(),
        "instance_count": len(instances),
        "cells": [asdict(c) for c in cells],
    }

    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(summary, indent=2)
        target.write_text(raw, encoding="utf-8")
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        sidecar = target.with_name(f"{target.name}.sha256")
        sidecar.write_text(f"{digest}  {target.name}\n", encoding="utf-8")

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "simon_feasibility" / "oracle_deadline_distinction_diagnostic.json",
        help="Target output JSON path",
    )
    parser.add_argument(
        "--no-telemetry-check",
        action="store_true",
        help="Skip strict worker telemetry certification",
    )
    args = parser.parse_args()

    print("Running SIMON oracle deadline distinction diagnostic on 8 opened demo keys...")
    res = run_oracle_diagnostic(output_path=args.output, verify_telemetry=not args.no_telemetry_check)
    print(f"Completed diagnostic: {len(res['cells'])} cells evaluated. Saved to {args.output}")


if __name__ == "__main__":
    main()
