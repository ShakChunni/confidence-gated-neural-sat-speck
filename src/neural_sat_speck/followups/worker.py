"""Isolated ranking or queue traversal; an independent guardian owns deadlines."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from neural_sat_speck.backend.guard import atomic_json, read_checked_json
from neural_sat_speck.backend.worker import Journal
from neural_sat_speck.followups.environment import power_sample
from neural_sat_speck.followups.queue_order import (
    block,
    ordered_hash,
    permutation,
    queue,
)


def verify_observations(
    cipher: str, rounds: int, key: list[int], observations: list
) -> bool:
    if cipher == "speck32_64":
        from neural_sat_speck.speck.cipher import encrypt_block
    elif cipher == "simon32_64":
        from neural_sat_speck.simon.cipher import encrypt_block
    else:
        raise ValueError("Unsupported cipher")
    return bool(observations) and all(
        encrypt_block(block(pt), tuple(key), rounds) == block(ct)
        for pt, ct in observations
    )


def new_model(cipher: str, known: list, rounds: int):
    if cipher == "speck32_64":
        from neural_sat_speck.sat.model import SpeckSatModel

        factory = SpeckSatModel
    elif cipher == "simon32_64":
        from neural_sat_speck.simon.sat_model import SimonSatModel

        factory = SimonSatModel
    else:
        raise ValueError("Unsupported cipher")
    return factory.from_known_pairs(
        [(block(pt), block(ct)) for pt, ct in known[:4]],
        rounds=rounds,
    )


def assumptions(model, cipher: str, candidate: tuple[int, int], rounds: int) -> tuple:
    if cipher == "simon32_64":
        return model.assumptions_for_trailing_round_subkeys(candidate)
    from neural_sat_speck.sat.multi_round import assumptions_for_round_keys

    return assumptions_for_round_keys(
        model,
        {rounds - 2: candidate[0], rounds - 1: candidate[1]},
    )


def traverse(
    public: dict,
    fresh: list,
    candidates: tuple,
    job: dict,
    deadline: float,
    journal: Journal,
    directory: Path,
) -> dict:
    """One fresh incremental solver; unresolved calls rebuild within this arm."""
    cipher, rounds = job["cipher"], job["rounds"]
    known = public["known_observations"]
    if len(known) != 128 or len(fresh) != 2:
        raise ValueError("Replay needs128 known observations and2 fresh blocks")
    journal.transition("initial_formula_construction")
    model = new_model(cipher, known, rounds)
    attempts = []
    unresolved = verification_failures = 0
    result = dict(
        outcome="queue_exhausted",
        success=False,
        verified_known=False,
        verified_fresh=False,
        recovered_key=None,
    )
    for index, candidate in enumerate(candidates, start=1):
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            result["outcome"] = "global_deadline"
            break
        atomic_json(
            directory / "candidate_progress.json",
            {
                "candidate_index": index,
                "candidate": list(candidate),
                "completed_attempts": attempts,
            },
        )
        journal.transition("solving")
        solve_started = time.perf_counter()
        solved = model.solve(
            assumptions(model, cipher, candidate, rounds),
            time_limit=min(job["candidate_request_seconds"], remaining),
        )
        attempt = {
            "candidate_index": index,
            "candidate": list(candidate),
            "solver_status": solved.status,
            "solve_wall_seconds": time.perf_counter() - solve_started,
            "verified_known": False,
            "verified_fresh": False,
        }
        # Close solving before verification; journal phases do not overlap.
        journal.transition("decoding")
        if solved.status == "sat":
            if solved.solution is None:
                raise ValueError("SAT result lacks an assignment")
            key = list(model.master_key_from_solution(solved.solution))
            journal.transition("verification")
            known_ok = verify_observations(cipher, rounds, key, known)
            fresh_ok = verify_observations(cipher, rounds, key, fresh)
            attempt.update(verified_known=known_ok, verified_fresh=fresh_ok)
            attempts.append(attempt)
            if known_ok and fresh_ok:
                result.update(
                    outcome="verified_recovery",
                    success=True,
                    verified_known=True,
                    verified_fresh=True,
                    recovered_key=key,
                    recovered_candidate_index=index,
                )
                break
            verification_failures += 1
            journal.transition("rebuilding_formula")
            model = new_model(cipher, known, rounds)
        elif solved.status == "unsat":
            attempts.append(attempt)
        elif solved.status in {"solver_timeout", "unsat_or_timeout"}:
            attempts.append(attempt)
            unresolved += 1
            if time.perf_counter() >= deadline:
                result["outcome"] = "global_deadline"
                break
            journal.transition("rebuilding_formula")
            model = new_model(cipher, known, rounds)
        else:
            raise ValueError(f"Unexpected solver status: {solved.status}")
        atomic_json(
            directory / "candidate_progress.json",
            {
                "candidate_index": index,
                "candidate": list(candidate),
                "completed_attempts": attempts,
            },
        )
    else:
        if verification_failures:
            result["outcome"] = "verification_failure"
        elif unresolved:
            result["outcome"] = "unresolved_candidates"
    journal.transition(None)
    result.update(
        attempts=attempts,
        candidates_tested=len(attempts),
        unresolved_candidates=unresolved,
        verification_failures=verification_failures,
    )
    return result


def execute(payload: dict) -> dict:
    directory = Path(payload["directory"])
    job = payload["job"]
    journal = Journal(directory / "journal.json")
    result = {"job_id": job["job_id"], "outcome": "worker_exception", "success": False}
    try:
        journal.transition("environment_observation")
        from neural_sat_speck.protocol.worker_telemetry import certify_worker_or_raise

        receipt = certify_worker_or_raise(Path(payload["repo_root"]))
        atomic_json(directory / "worker_receipt.json", receipt.to_dict())
        if receipt.git_commit != payload["source_commit"]:
            raise ValueError("Worker source commit differs from registered source")
        if power_sample().get("accepted") is not True:
            raise RuntimeError("Observed worker power/sleep policy violation")
        journal.transition("input_loading")
        public = payload["public"]
        if set(public) != {
            "cipher",
            "rounds",
            "instance_id",
            "ciphertext_pairs",
            "known_observations",
        }:
            raise ValueError("Public input has unexpected fields or truth material")
        if job["operation"] == "rank":
            if "fresh" in payload or "preparation" in payload:
                raise ValueError(
                    "Ranking cannot receive fresh blocks or an existing queue"
                )
            journal.transition("ranking")
            from neural_sat_speck.followups.ranking import rank_queue

            os.environ["QUEUE_ORDER_TF_RECEIPT"] = str(
                directory / "tensorflow_worker_receipt.json"
            )
            os.environ["QUEUE_ORDER_SOURCE_COMMIT"] = payload["source_commit"]

            ranked_result = rank_queue(
                public,
                job,
                {name: Path(path) for name, path in payload["paths"].items()},
            )
            ranked = queue(ranked_result["ranked"])
            journal.transition("queue_persistence")
            shuffled = permutation(ranked, job)
            result.update(
                outcome="queue_ready",
                ranked=[list(v) for v in ranked],
                shuffled=[list(v) for v in shuffled],
                set_sha256=ordered_hash(sorted(ranked)),
                ranked_sha256=ordered_hash(ranked),
                shuffled_sha256=ordered_hash(shuffled),
                ranking_details={
                    k: v for k, v in ranked_result.items() if k != "ranked"
                },
            )
        else:
            prep = payload["preparation"]
            candidates = queue(prep[job["order"]])
            if ordered_hash(candidates) != prep[job["order"] + "_sha256"]:
                raise ValueError("Replay queue identity differs")
            result.update(
                traverse(
                    public,
                    payload["fresh"],
                    candidates,
                    job,
                    payload["deadline_monotonic"],
                    journal,
                    directory,
                )
            )
    except TimeoutError as exc:
        result.update(
            outcome="ranking_timeout"
            if job["operation"] == "rank"
            else "worker_exception",
            error=str(exc),
        )
    except Exception as exc:
        result.update(outcome="worker_exception", error=f"{type(exc).__name__}: {exc}")
    finally:
        journal.transition(None)
    result["phase_seconds"] = journal.completed
    return result


def main() -> None:
    path = Path(sys.argv[1])
    payload = read_checked_json(path)
    result = execute(payload)
    atomic_json(Path(payload["directory"]) / "result.json", result)


if __name__ == "__main__":
    main()
