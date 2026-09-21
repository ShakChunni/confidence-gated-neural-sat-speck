#!/usr/bin/env python3
"""Dedicated Adjudication Runner for Phase 11 7-Round SPECK32/64 Beam Configurations.

Evidence Class: DEVELOPMENT (Decision D-030).
Supervisor Authorization: 2026-09-19.

Compares:
  Configuration A (Reference):
    - M1 = 32 final-round k6 candidates
    - M2 = 32 conditional k5 candidates per retained k6
    - Candidate budget = 1,024 tuples
  Configuration B (Wider Stage 1):
    - M1 = 64 final-round k6 candidates
    - M2 = 16 conditional k5 candidates per retained k6
    - Candidate budget = 1,024 tuples

Controls:
  - Both configurations evaluated on identical N=64 instances.
  - Counterbalanced execution order: A then B on even indices, B then A on odd indices.
  - Independent IsolatedSolverFactory per treatment per instance.
  - Zero cross-treatment cache, solver state, or deadline sharing.
  - Monotonic 300.0s global cutoff enforced independently per configuration.
  - Per-candidate solver timeout: 10.0s.
  - Screened Stage 2: 16-pair screen, top-512 retention, 64-pair rescore.
  - Atomic checkpointing with fail-closed resume validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import torch

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from neural_sat_speck.attacks.group_ranking import rank_pair_group_subkeys  # noqa: E402
from neural_sat_speck.attacks.multi_round_instance import (  # noqa: E402
    partial_decrypt_instance,
)
from neural_sat_speck.attacks.multi_round_recovery import (  # noqa: E402
    STATUS_GLOBAL_TIMEOUT,
    run_multi_round_key_recovery,
)
from neural_sat_speck.attacks.recovery import (  # noqa: E402
    independent_verification_pairs,
    known_pairs_for_attack_instance,
)
from neural_sat_speck.attacks.screened_group_ranking import (  # noqa: E402
    rank_pair_group_subkeys_screened,
)
from neural_sat_speck.data.attack import AttackInstance, load_attack_shard  # noqa: E402
from neural_sat_speck.neural.group_model import PairGroupRanker  # noqa: E402
from neural_sat_speck.protocol.manifest_validator import (  # noqa: E402
    classify_attack_outcome,
    verify_attack_dataset_integrity,
)
from neural_sat_speck.sat.isolation import IsolatedSolverFactory  # noqa: E402
from neural_sat_speck.speck.key_schedule import expand_key  # noqa: E402

DEFAULT_DATASET: Final[Path] = (
    REPO_ROOT / "datasets" / "phase11_beam_adjudication_r7" / "attack_instances.json"
)
DEFAULT_STAGE1_MODEL: Final[Path] = (
    REPO_ROOT / "models" / "speck32_64_phase11b_prime_stage1_r6_b1_mean.pt"
)
DEFAULT_STAGE2_MODEL: Final[Path] = (
    REPO_ROOT / "models" / "speck32_64_phase6a_b1_mean_g64.pt"
)
DEFAULT_OUTPUT: Final[Path] = (
    REPO_ROOT
    / "results"
    / "phase11_scaling_diagnostics"
    / "beam_adjudication_results.json"
)

ROUNDS: Final[int] = 7
SAT_OBSERVATIONS_4OBS: Final[int] = 4
VERIFICATION_PAIRS: Final[int] = 2
VERIFICATION_SEED: Final[int] = 999999
GLOBAL_CUTOFF_SECONDS: Final[float] = 300.0
CANDIDATE_SOLVER_TIMEOUT: Final[float] = 10.0
CANDIDATE_BUDGET: Final[int] = 1024
SCREEN_PAIRS: Final[int] = 16
SCREEN_TOP_K: Final[int] = 512
BATCH_SIZE: Final[int] = 1024

CONFIG_SPECS: Final[dict[str, dict[str, Any]]] = {
    "A": {
        "config_id": "A",
        "display_name": "Configuration A (Reference: M1=32, M2=32)",
        "m1": 32,
        "m2": 32,
        "candidate_budget": CANDIDATE_BUDGET,
    },
    "B": {
        "config_id": "B",
        "display_name": "Configuration B (Wider Stage 1: M1=64, M2=16)",
        "m1": 64,
        "m2": 16,
        "candidate_budget": CANDIDATE_BUDGET,
    },
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_pytorch_ranker(path: Path, device: torch.device) -> PairGroupRanker:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    variant = payload.get("variant", "b1_mean")
    model = PairGroupRanker(variant=variant)
    state = payload["state_dict"] if "state_dict" in payload else payload
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _save_atomic(data: dict[str, Any], target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, target_path)


def run_single_configuration_attack(
    instance: AttackInstance,
    *,
    config_id: str,
    stage1_model: PairGroupRanker,
    stage2_model: PairGroupRanker,
    device: torch.device,
    global_timeout: float = GLOBAL_CUTOFF_SECONDS,
    solver_timeout: float = CANDIDATE_SOLVER_TIMEOUT,
    candidate_budget: int = CANDIDATE_BUDGET,
) -> dict[str, Any]:
    """Run one hierarchical attack configuration with strict deadline clamping and solver isolation."""
    cfg = CONFIG_SPECS[config_id]
    m1 = cfg["m1"]
    m2 = cfg["m2"]

    method_started = time.perf_counter()
    method_deadline = method_started + global_timeout

    # Preprocessing
    t_prep_start = time.perf_counter()
    subkeys = expand_key(instance.master_key, ROUNDS)
    true_k6 = subkeys[6]
    true_k5 = subkeys[5]
    true_tuple = (true_k6, true_k5)

    known_all = known_pairs_for_attack_instance(instance)  # 128 obs (64 pairs)
    known_4 = known_all[:SAT_OBSERVATIONS_4OBS]  # 4 obs (2 pairs)
    verification = independent_verification_pairs(
        instance.master_key,
        rounds=ROUNDS,
        pair_count=VERIFICATION_PAIRS,
        seed=VERIFICATION_SEED,
    )
    preprocessing_seconds = time.perf_counter() - t_prep_start

    # Check deadline
    if time.perf_counter() >= method_deadline:
        wall_sec = time.perf_counter() - method_started
        return _build_timeout_record(
            instance, config_id, m1, m2, wall_sec, preprocessing_seconds
        )

    # Stage 1: 6-round ranking
    t_rank_start = time.perf_counter()
    ranking_s1 = rank_pair_group_subkeys(
        instance,
        stage1_model,
        candidate_batch_size=BATCH_SIZE,
        device=device,
    )
    s1_order = ranking_s1.candidate_order
    true_k6_rank = s1_order.index(true_k6) + 1
    k6_in_beam = bool(true_k6_rank <= m1)
    s1_survivors = s1_order[:m1]

    # Stage 2: screened 5-round ranking per surviving k6
    true_k5_rank: int | None = None
    k5_in_beam: bool = False
    all_candidates: list[tuple[tuple[int, int], int]] = []
    timed_out_in_ranking = False

    for r1, k6_cand in enumerate(s1_survivors, start=1):
        if time.perf_counter() >= method_deadline:
            timed_out_in_ranking = True
            break

        desc_instance = partial_decrypt_instance(
            instance,
            k6_cand,
            expected_next_subkey=0,
            label=f"{instance.instance_id}-r5",
        )
        ranking_s2 = rank_pair_group_subkeys_screened(
            desc_instance,
            stage2_model,
            screen_pairs=SCREEN_PAIRS,
            screen_top_k=SCREEN_TOP_K,
            candidate_batch_size=BATCH_SIZE,
            device=device,
        )
        s2_order = ranking_s2.candidate_order

        if k6_cand == true_k6:
            true_k5_rank = s2_order.index(true_k5) + 1
            k5_in_beam = bool(true_k5_rank <= m2)

        for r2, k5_cand in enumerate(s2_order[:m2], start=1):
            all_candidates.append(((k6_cand, k5_cand), r1 + r2))

    ranking_screening_sorting_seconds = time.perf_counter() - t_rank_start

    if timed_out_in_ranking and not all_candidates:
        wall_sec = time.perf_counter() - method_started
        return _build_timeout_record(
            instance, config_id, m1, m2, wall_sec, preprocessing_seconds
        )

    # Sort candidates by (rank_sum, (k6, k5))
    ordered_tuples = sorted(all_candidates, key=lambda entry: (entry[1], entry[0]))

    # Diagnostic reachability
    if k6_in_beam and k5_in_beam:
        true_tuple_position = next(
            (idx for idx, (cand, _) in enumerate(ordered_tuples, start=1) if cand == true_tuple),
            None,
        )
        true_tuple_reachable = bool(
            true_tuple_position is not None and true_tuple_position <= candidate_budget
        )
    else:
        true_tuple_position = None
        true_tuple_reachable = False

    # Check deadline before SAT
    remaining_global = max(0.0, method_deadline - time.perf_counter())
    if remaining_global <= 0.0:
        wall_sec = time.perf_counter() - method_started
        return _build_record(
            instance=instance,
            config_id=config_id,
            m1=m1,
            m2=m2,
            success=False,
            status=STATUS_GLOBAL_TIMEOUT,
            candidates_attempted=0,
            unresolved_timeouts=0,
            recovered_master_key=None,
            fresh_pair_verification_passed=False,
            wall_sec=wall_sec,
            preprocessing_sec=preprocessing_seconds,
            ranking_sec=ranking_screening_sorting_seconds,
            solver_sec=0.0,
            rebuilding_sec=0.0,
            verification_sec=0.0,
            true_k6_rank=true_k6_rank,
            k6_in_beam=k6_in_beam,
            true_k5_rank=true_k5_rank,
            k5_in_beam=k5_in_beam,
            true_tuple_reachable=true_tuple_reachable,
            true_tuple_position=true_tuple_position,
        )

    # Isolated SAT solving
    candidate_round_keys = [
        {6: p[0], 5: p[1]} for p, _ in ordered_tuples[:candidate_budget]
    ]

    factory = IsolatedSolverFactory.from_pairs(
        known_4,
        rounds=ROUNDS,
        instance_id=f"{instance.instance_id}_config_{config_id}",
    )
    solver_model = factory.build_isolated_model(f"exact_sat_config_{config_id}")

    eff_solver_timeout = min(solver_timeout, remaining_global)
    recovery_res = run_multi_round_key_recovery(
        solver_model,
        candidate_round_keys,
        known_pairs=known_4,
        verification_pairs=verification,
        rounds=ROUNDS,
        solver_timeout=eff_solver_timeout,
        global_timeout=remaining_global,
        strategy="hierarchical_multi_round_beam",
        solver_factory=factory,
    )

    wall_sec = time.perf_counter() - method_started
    unresolved = sum(1 for a in recovery_res.attempts if a.timed_out)

    # Rebuilding time is time spent reconstructing isolated models upon candidate timeouts
    rebuilding_sec = sum(
        a.solver_seconds for a in recovery_res.attempts if a.timed_out
    )

    return _build_record(
        instance=instance,
        config_id=config_id,
        m1=m1,
        m2=m2,
        success=recovery_res.success,
        status=recovery_res.status,
        candidates_attempted=recovery_res.candidates_attempted,
        unresolved_timeouts=unresolved,
        recovered_master_key=(
            list(recovery_res.recovered_master_key)
            if recovery_res.recovered_master_key is not None
            else None
        ),
        fresh_pair_verification_passed=bool(recovery_res.success),
        wall_sec=wall_sec,
        preprocessing_sec=preprocessing_seconds,
        ranking_sec=ranking_screening_sorting_seconds,
        solver_sec=float(recovery_res.solver_seconds),
        rebuilding_sec=float(rebuilding_sec),
        verification_sec=float(recovery_res.verification_seconds),
        true_k6_rank=true_k6_rank,
        k6_in_beam=k6_in_beam,
        true_k5_rank=true_k5_rank,
        k5_in_beam=k5_in_beam,
        true_tuple_reachable=true_tuple_reachable,
        true_tuple_position=true_tuple_position,
    )


def _build_timeout_record(
    instance: AttackInstance,
    config_id: str,
    m1: int,
    m2: int,
    wall_sec: float,
    prep_sec: float,
) -> dict[str, Any]:
    return _build_record(
        instance=instance,
        config_id=config_id,
        m1=m1,
        m2=m2,
        success=False,
        status=STATUS_GLOBAL_TIMEOUT,
        candidates_attempted=0,
        unresolved_timeouts=0,
        recovered_master_key=None,
        fresh_pair_verification_passed=False,
        wall_sec=wall_sec,
        prep_sec=prep_sec,
        ranking_sec=0.0,
        solver_sec=0.0,
        rebuilding_sec=0.0,
        verification_sec=0.0,
        true_k6_rank=None,
        k6_in_beam=False,
        true_k5_rank=None,
        k5_in_beam=False,
        true_tuple_reachable=False,
        true_tuple_position=None,
    )


def _build_record(
    *,
    instance: AttackInstance,
    config_id: str,
    m1: int,
    m2: int,
    success: bool,
    status: str,
    candidates_attempted: int,
    unresolved_timeouts: int,
    recovered_master_key: list[int] | None,
    fresh_pair_verification_passed: bool,
    wall_sec: float,
    preprocessing_sec: float,
    ranking_sec: float,
    solver_sec: float,
    rebuilding_sec: float,
    verification_sec: float,
    true_k6_rank: int | None,
    k6_in_beam: bool,
    true_k5_rank: int | None,
    k5_in_beam: bool,
    true_tuple_reachable: bool,
    true_tuple_position: int | None,
) -> dict[str, Any]:
    classified = classify_attack_outcome(
        status,
        success=bool(success),
        unresolved_timeouts=unresolved_timeouts,
    )
    par2 = wall_sec if success else 2.0 * GLOBAL_CUTOFF_SECONDS

    # Failure decomposition
    failure_domain = classified["failure_domain"]
    if not success:
        if not k6_in_beam:
            failure_category = "stage1_ranking_failure"
        elif not k5_in_beam:
            failure_category = "stage2_ranking_failure"
        elif not true_tuple_reachable:
            failure_category = "tuple_reachability_failure"
        elif status == STATUS_GLOBAL_TIMEOUT:
            failure_category = "global_timeout_before_true_tuple"
        elif unresolved_timeouts > 0:
            failure_category = "candidate_solver_timeout"
        else:
            failure_category = "other_failure"
    else:
        failure_category = "none"

    return {
        "instance_id": instance.instance_id,
        "configuration": config_id,
        "config_display_name": CONFIG_SPECS[config_id]["display_name"],
        "m1": m1,
        "m2": m2,
        "candidate_budget": CANDIDATE_BUDGET,
        "verified_recovery": bool(success),
        "status": status,
        "failure_cause": None if success else status,
        "failure_domain": failure_domain,
        "failure_category": failure_category,
        "candidates_attempted": candidates_attempted,
        "unresolved_timeouts": unresolved_timeouts,
        "recovered_master_key": recovered_master_key,
        "fresh_pair_verification_passed": fresh_pair_verification_passed,
        "total_method_wall_seconds": round(wall_sec, 6),
        "par2_contribution": round(par2, 6),
        "preprocessing_seconds": round(preprocessing_sec, 6),
        "ranking_screening_sorting_seconds": round(ranking_sec, 6),
        "solver_seconds": round(solver_sec, 6),
        "rebuilding_seconds": round(rebuilding_sec, 6),
        "verification_seconds": round(verification_sec, 6),
        "true_k6_rank": true_k6_rank,
        "k6_in_beam": k6_in_beam,
        "true_k5_rank": true_k5_rank,
        "k5_in_beam": k5_in_beam,
        "true_tuple_reachable": true_tuple_reachable,
        "true_tuple_position": true_tuple_position,
        "counts_as_verified_recovery": classified["counts_as_verified_recovery"],
        "counts_as_unconditional_failure": classified["counts_as_unconditional_failure"],
        "is_resolved_sat_state": classified["is_resolved_sat_state"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to development dataset JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to output JSON checkpoint/results",
    )
    parser.add_argument(
        "--stage1-model",
        type=Path,
        default=DEFAULT_STAGE1_MODEL,
        help="Stage 1 6-round ResNet checkpoint",
    )
    parser.add_argument(
        "--stage2-model",
        type=Path,
        default=DEFAULT_STAGE2_MODEL,
        help="Stage 2 5-round ResNet checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="PyTorch device (mps, cpu, cuda)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume execution from existing checkpoint",
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=64,
        help="Number of instances to evaluate (default: 64)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start instance index (default: 0)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("=== Phase 11 SPECK32/64 Beam-Configuration Adjudication ===")
    print("Evidence class: DEVELOPMENT (Decision D-030)")

    # 1. Dataset verification
    dataset_info = verify_attack_dataset_integrity(
        args.dataset,
        manifest_data=None,
        is_prospective=False,
    )
    print(
        f"[✓] Dataset validated: {dataset_info['dataset_id']} "
        f"(N={dataset_info['instance_count']}, seed={dataset_info['seed']}, SHA256={dataset_info['sha256'][:12]}...)"
    )

    # 2. Device determination
    device_name = (
        args.device
        if (
            (args.device == "mps" and torch.backends.mps.is_available())
            or (args.device == "cuda" and torch.cuda.is_available())
        )
        else "cpu"
    )
    device_obj = torch.device(device_name)
    print(f"[✓] Execution device: {device_obj}")

    # 3. Model loading
    print(f"Loading Stage 1 model from {args.stage1_model}...")
    stage1_pyt = load_pytorch_ranker(args.stage1_model, device=device_obj)
    stage1_hash = _sha256_file(args.stage1_model)

    print(f"Loading Stage 2 model from {args.stage2_model}...")
    stage2_pyt = load_pytorch_ranker(args.stage2_model, device=device_obj)
    stage2_hash = _sha256_file(args.stage2_model)

    # 4. Resume check or fresh state
    output_path = args.output
    records_by_config: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}

    if args.resume and output_path.exists():
        print(f"Resuming from existing checkpoint at {output_path}...")
        existing_data = json.loads(output_path.read_text(encoding="utf-8"))
        ex_meta = existing_data.get("metadata", {})
        if ex_meta.get("dataset_sha256") != dataset_info["sha256"]:
            raise SystemExit("Resume error: dataset SHA-256 mismatch!")
        if ex_meta.get("stage1_model_sha256") != stage1_hash:
            raise SystemExit("Resume error: Stage 1 model SHA-256 mismatch!")
        if ex_meta.get("stage2_model_sha256") != stage2_hash:
            raise SystemExit("Resume error: Stage 2 model SHA-256 mismatch!")
        records_by_config = existing_data.get(
            "records_by_configuration", {"A": [], "B": []}
        )
        print(
            f"[✓] Loaded {len(records_by_config['A'])} records for Config A, "
            f"{len(records_by_config['B'])} records for Config B."
        )

    # 5. Load dataset instances
    shard = load_attack_shard(args.dataset, expected_dataset_id=dataset_info["dataset_id"])
    selected_instances = shard.instances[args.start : args.start + args.instances]
    if not selected_instances:
        raise SystemExit("No instances selected for evaluation.")

    print(
        f"\nStarting adjudication: {len(selected_instances)} instances "
        f"(indices {args.start}..{args.start + len(selected_instances) - 1})"
    )
    print("Configurations: A (M1=32, M2=32) vs B (M1=64, M2=16). Budget: 1,024 tuples.")
    print("Counterbalancing: Even index -> A then B; Odd index -> B then A.\n")

    metadata = {
        "evidence_class": "DEVELOPMENT",
        "supervisor_authorization": "Decision D-030 (2026-09-19)",
        "protocol_status": "OPENED_PERMANENTLY_RETIRED",
        "dataset_path": str(args.dataset),
        "dataset_sha256": dataset_info["sha256"],
        "dataset_id": dataset_info["dataset_id"],
        "dataset_seed": dataset_info["seed"],
        "stage1_model_path": str(args.stage1_model),
        "stage1_model_sha256": stage1_hash,
        "stage2_model_path": str(args.stage2_model),
        "stage2_model_sha256": stage2_hash,
        "rounds": ROUNDS,
        "sat_observations": SAT_OBSERVATIONS_4OBS,
        "verification_pairs": VERIFICATION_PAIRS,
        "verification_seed": VERIFICATION_SEED,
        "candidate_solver_timeout": CANDIDATE_SOLVER_TIMEOUT,
        "global_cutoff_seconds": GLOBAL_CUTOFF_SECONDS,
        "candidate_budget": CANDIDATE_BUDGET,
        "screen_pairs": SCREEN_PAIRS,
        "screen_top_k": SCREEN_TOP_K,
        "batch_size": BATCH_SIZE,
        "device": str(device_obj),
    }

    # Main evaluation loop
    for idx, instance in enumerate(selected_instances):
        global_idx = args.start + idx
        instance_id = instance.instance_id

        # Counterbalancing schedule:
        # Even index: A then B
        # Odd index: B then A
        if global_idx % 2 == 0:
            exec_order = ["A", "B"]
        else:
            exec_order = ["B", "A"]

        print(
            f"--- Instance [{global_idx:02d}] {instance_id} (Order: {' -> '.join(exec_order)}) ---"
        )

        for cfg_id in exec_order:
            # Check if already completed
            already_done = any(
                r.get("instance_id") == instance_id
                for r in records_by_config[cfg_id]
            )
            if already_done:
                print(f"  Config {cfg_id:<28} [ALREADY RECORDED]")
                continue

            rec = run_single_configuration_attack(
                instance,
                config_id=cfg_id,
                stage1_model=stage1_pyt,
                stage2_model=stage2_pyt,
                device=device_obj,
                global_timeout=GLOBAL_CUTOFF_SECONDS,
                solver_timeout=CANDIDATE_SOLVER_TIMEOUT,
                candidate_budget=CANDIDATE_BUDGET,
            )
            rec["execution_order"] = exec_order
            rec["instance_index"] = global_idx
            records_by_config[cfg_id].append(rec)

            status_str = f"recov={rec['verified_recovery']} status={rec['status']}"
            tuples_str = f"tuples={rec['candidates_attempted']}"
            wall_str = f"wall={rec['total_method_wall_seconds']:6.2f}s"
            k6_str = f"k6_rk={rec['true_k6_rank']} (in={rec['k6_in_beam']})"
            k5_str = f"k5_rk={rec['true_k5_rank']} (in={rec['k5_in_beam']})"
            print(
                f"  Config {cfg_id}  {status_str:<32} {tuples_str:<12} {wall_str:<14} {k6_str:<16} {k5_str:<16}"
            )

        # Atomic checkpoint after every instance
        checkpoint_data = {
            "metadata": metadata,
            "evidence_class": "DEVELOPMENT",
            "protocol_status": "OPENED_PERMANENTLY_RETIRED",
            "generated_at": datetime.now(UTC).isoformat(),
            "records_by_configuration": records_by_config,
        }
        _save_atomic(checkpoint_data, output_path)

    print("\n" + "=" * 80)
    print(f"{'BEAM ADJUDICATION RUN COMPLETE':^80}")
    print("=" * 80)
    print(f"Output saved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
