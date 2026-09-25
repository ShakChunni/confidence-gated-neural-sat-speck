#!/usr/bin/env python3
"""SIMON32/64 exact-recovery feasibility grid and Go/No-Go pilot.

Evaluates an empirical grid over round counts R in {6, 7, 8}, assumed trailing
keys A in {0, 1, 2} (residual depths R_res = R - A), observation counts
N in {2, 4, 16, 64} (1, 2, 8, 32 differential pairs), and backend configurations
(in-process pycryptosat wrapper vs native cryptominisat5 with XOR/Gauss).

Records formula size (variables, CNF clauses, XOR clauses), solver outcome,
fresh-pair verified key recovery, wall time, and failure causes. Produces a
reproducible JSON artifact and a Markdown Go/No-Go summary table.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from neural_sat_speck.data.seed_policy import CLI_DEMO_SEED
from neural_sat_speck.simon.attack_instance import (
    DEFAULT_SIMON_INPUT_DIFFERENCE,
    SimonAttackInstance,
    SimonAttackInstanceConfig,
    generate_simon_attack_instances,
)
from neural_sat_speck.simon.sat_model import SimonSatModel
from neural_sat_speck.simon.verification import verify_simon_master_key


@dataclass(frozen=True)
class SimonGridPointResult:
    """Outcome of one grid cell evaluation on a SIMON attack instance."""

    instance_id: str
    rounds: int
    assumed_trailing_keys: int
    residual_depth: int
    observation_count: int
    pair_count: int
    backend: str
    variables: int
    cnf_clauses: int
    xor_clauses: int
    total_clauses: int
    satisfiable: bool
    timed_out: bool | None
    solver_status: str
    wall_seconds: float
    verified_recovery: bool
    known_pairs_verified: bool
    independent_pairs_verified: bool
    failure_cause: str | None


def evaluate_grid_point(
    instance: SimonAttackInstance,
    rounds: int,
    assumed_keys: int,
    obs_count: int,
    backend: str,
    timeout: float,
) -> SimonGridPointResult:
    """Evaluate one SIMON attack instance at a specific grid configuration."""
    residual_depth = rounds - assumed_keys
    pair_count = obs_count // 2
    selected_pairs = instance.pairs[:pair_count]

    # Materialise known pairs: 2 observations per differential pair
    known_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for p in selected_pairs:
        known_pairs.append((p.plaintext0, p.ciphertext0))
        known_pairs.append((p.plaintext1, p.ciphertext1))

    # Build exact SAT model
    model = SimonSatModel.from_known_pairs(known_pairs, rounds=rounds)

    # Build assumptions for trailing keys if A > 0
    assumptions: tuple[int, ...] = ()
    if assumed_keys > 0:
        true_trailing = instance.expected_trailing_subkeys(assumed_keys)
        assumptions = model.assumptions_for_trailing_round_subkeys(true_trailing)

    # Solve under selected backend
    t0 = time.perf_counter()
    if backend == "pycryptosat":
        solve_res = model.solve(assumptions, time_limit=timeout)
    elif backend == "native_gauss":
        solve_res = model.solve_native_cli(
            assumptions,
            time_limit=timeout,
            enable_gauss=True,
            max_matrix_cols=2500,
        )
    elif backend == "native_default":
        solve_res = model.solve_native_cli(
            assumptions,
            time_limit=timeout,
            enable_gauss=False,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")
    elapsed = time.perf_counter() - t0

    # Verification
    verified = False
    known_ok = False
    fresh_ok = False
    failure_cause: str | None = None

    if solve_res.satisfiable and solve_res.solution is not None:
        rec_key = model.master_key_from_solution(solve_res.solution)
        verif_res = verify_simon_master_key(
            rec_key,
            known_pairs,
            instance.verification_pairs,
            rounds=rounds,
        )
        known_ok = verif_res.known_pairs_verified
        fresh_ok = verif_res.independent_pairs_verified
        verified = verif_res.verified
        if not verified:
            failure_cause = "spurious_key_unverified"
    elif solve_res.timed_out:
        failure_cause = "solver_timeout"
    else:
        failure_cause = "unsat"

    return SimonGridPointResult(
        instance_id=instance.instance_id,
        rounds=rounds,
        assumed_trailing_keys=assumed_keys,
        residual_depth=residual_depth,
        observation_count=obs_count,
        pair_count=pair_count,
        backend=backend,
        variables=model.variable_count,
        cnf_clauses=model.cnf_clause_count,
        xor_clauses=model.xor_clause_count,
        total_clauses=model.total_clause_count,
        satisfiable=solve_res.satisfiable,
        timed_out=solve_res.timed_out,
        solver_status=solve_res.status,
        wall_seconds=elapsed,
        verified_recovery=verified,
        known_pairs_verified=known_ok,
        independent_pairs_verified=fresh_ok,
        failure_cause=failure_cause,
    )


def run_pilot(
    rounds_list: list[int],
    assumed_list: list[int],
    obs_list: list[int],
    backends: list[str],
    timeout: float,
    instance_count: int,
    output_path: Path,
) -> None:
    """Run full feasibility grid on demo instances."""
    print("=" * 80)
    print("SIMON32/64 EXACT-RECOVERY FEASIBILITY PILOT (ELIGIBILITY GATE)")
    print(f"Rounds: {rounds_list} | Assumed trailing keys: {assumed_list}")
    print(f"Observations: {obs_list} | Backends: {backends} | Timeout: {timeout}s")
    print(f"Instances per cell: {instance_count} | Seed: {CLI_DEMO_SEED} (DEMO)")
    print("=" * 80)

    # Pre-generate max pairs needed across all instances
    max_pairs = max(obs_list) // 2

    # Generate instances for each round count
    instances_by_rounds: dict[int, list[SimonAttackInstance]] = {}
    for r in rounds_list:
        cfg = SimonAttackInstanceConfig(
            instance_count=instance_count,
            pair_count=max_pairs,
            rounds=r,
            seed=CLI_DEMO_SEED,
            split_role="demo",
            verification_pair_count=2,
            input_difference=DEFAULT_SIMON_INPUT_DIFFERENCE,
        )
        instances_by_rounds[r] = list(generate_simon_attack_instances(cfg))

    all_results: list[SimonGridPointResult] = []

    print(
        f"{'R':>2} {'A':>2} {'Rres':>4} {'N':>3} {'Backend':<13} "
        f"{'Vars':>5} {'CNF':>5} {'XOR':>5} {'Status':<14} {'Verif':<6} {'Wall(s)':>7} {'Failure'}"
    )
    print("-" * 80)

    for r in rounds_list:
        for a in assumed_list:
            if a >= r:
                continue
            r_res = r - a
            for n_obs in obs_list:
                for b in backends:
                    cell_results: list[SimonGridPointResult] = []
                    for inst in instances_by_rounds[r]:
                        res = evaluate_grid_point(
                            instance=inst,
                            rounds=r,
                            assumed_keys=a,
                            obs_count=n_obs,
                            backend=b,
                            timeout=timeout,
                        )
                        cell_results.append(res)
                        all_results.append(res)

                    # Print aggregated line for cell
                    mean_wall = sum(x.wall_seconds for x in cell_results) / len(cell_results)
                    verif_count = sum(1 for x in cell_results if x.verified_recovery)
                    verif_str = f"{verif_count}/{len(cell_results)}"
                    sample = cell_results[0]
                    fail_str = sample.failure_cause or "None"

                    print(
                        f"{r:>2} {a:>2} {r_res:>4} {n_obs:>3} {b:<13} "
                        f"{sample.variables:>5} {sample.cnf_clauses:>5} {sample.xor_clauses:>5} "
                        f"{sample.solver_status:<14} {verif_str:<6} {mean_wall:>7.3f}s {fail_str}"
                    )

    # Save artifact
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "title": "SIMON32/64 Exact-Recovery Feasibility Pilot Grid",
            "date": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "seed": CLI_DEMO_SEED,
            "split_role": "demo",
            "rounds_evaluated": rounds_list,
            "assumed_keys_evaluated": assumed_list,
            "observations_evaluated": obs_list,
            "backends_evaluated": backends,
            "timeout_seconds": timeout,
            "instances_per_cell": instance_count,
        },
        "total_evaluations": len(all_results),
        "results": [asdict(r) for r in all_results],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("=" * 80)
    print(f"Results saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SIMON32/64 feasibility pilot grid")
    parser.add_argument("--rounds", nargs="+", type=int, default=[6, 7, 8])
    parser.add_argument("--assumed-keys", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--observations", nargs="+", type=int, default=[2, 4, 16, 64])
    parser.add_argument(
        "--backends",
        nargs="+",
        type=str,
        default=["pycryptosat", "native_gauss"],
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--instances", type=int, default=2)
    parser.add_argument(
        "--output",
        type=str,
        default="results/simon_feasibility/simon_feasibility_pilot.json",
    )
    args = parser.parse_args()

    run_pilot(
        rounds_list=args.rounds,
        assumed_list=args.assumed_keys,
        obs_list=args.observations,
        backends=args.backends,
        timeout=args.timeout,
        instance_count=args.instances,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
