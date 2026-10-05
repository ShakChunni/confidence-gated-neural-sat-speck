"""Identity, public-input and permutation guards for queue-order diagnostics.

No neural or SAT imports: rejected manifests cannot enter an attack boundary.
"""

from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path

from neural_sat_speck.backend.guard import (
    canonical_bytes,
    read_checked_json,
)

POLICIES = (
    ("speck32_64", "screened_neural", 7, 10.0),
    ("speck32_64", "neural_classical_hybrid", 7, 10.0),
    ("simon32_64", "classical_8parent", 10, 45.0),
    ("simon32_64", "bao_ddt_hybrid_8parent", 10, 45.0),
)
PROFILE = {
    "schema": "queue_order_demo_v1",
    "role": "opened_demo_diagnostic",
    "seed": 999999,
    "key_count_per_cipher": 8,
    "direct_pair_count": 64,
    "candidate_observations": 4,
    "verification_blocks": 2,
    "global_cutoff_seconds": 300.0,
    "shuffle_control_seed": 20261005,
    "accounting": "shared_preparation_charged_equally_plus_physical_replay",
    "maximum_starts": 96,
    "repetitions": 1,
    "retry_started_jobs": False,
    "speck_device": "mps",
    "candidate_batch_size": 1024,
    "tensorflow_intra_op_threads": 8,
    "tensorflow_inter_op_threads": 1,
    "numeric_threads": 1,
    "solver_threads": 1,
    "process_group_rss_limit_bytes": 2684354560,
    "power_policy": "battery_allowed_author_instruction_20261005",
    "policies": [list(p) for p in POLICIES],
}
FINAL_OUTCOMES = frozenset(
    {
        "verified_recovery",
        "different_verified_key",
        "queue_exhausted",
        "unresolved_candidates",
        "global_deadline",
        "ranking_timeout",
        "infrastructure_interrupted",
        "environment_rejected",
        "worker_exception",
        "verification_failure",
    }
)


def checked_path(root: Path, record: dict) -> Path:
    """Stream large DDT hashes; a whole-file read would exceed laptop memory."""
    import re

    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError("Incomplete file identity")
    name, expected = record["path"], record["sha256"]
    if (
        not isinstance(name, str)
        or not name
        or not isinstance(expected, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected) is None
    ):
        raise ValueError("Malformed file identity")
    path = Path(name)
    if not path.is_absolute():
        path = root / path
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"File identity mismatch: {name}")
    return path.absolute()


def fixed_jobs() -> list[dict]:
    result = []
    for key in range(8):
        shift = key % len(POLICIES)
        for cipher, policy, rounds, request in POLICIES[shift:] + POLICIES[:shift]:
            base = f"{cipher}_k{key}_{policy}"
            common = dict(
                case_id=base,
                cipher=cipher,
                policy=policy,
                rounds=rounds,
                key_index=key,
                candidate_request_seconds=request,
            )
            result.append({**common, "job_id": base + "__prepare", "operation": "rank"})
            orders = ("ranked", "shuffled") if key % 2 == 0 else ("shuffled", "ranked")
            for order in orders:
                result.append(
                    {
                        **common,
                        "job_id": base + "__" + order,
                        "operation": "solve",
                        "order": order,
                    }
                )
    return result


def word(value: object) -> int:
    if type(value) is not int or not 0 <= value <= 0xFFFF:
        raise ValueError("Expected a genuine 16-bit integer word")
    return value


def block(value: object) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Expected a two-word block")
    return word(value[0]), word(value[1])


def queue(value: object) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 1024:
        raise ValueError("A complete policy queue must contain exactly1024 tuples")
    result = tuple(block(v) for v in value)
    if len(set(result)) != len(result):
        raise ValueError("Duplicate candidate tuples")
    return result


def ordered_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def permutation(value: object, job: dict) -> tuple[tuple[int, int], ...]:
    ranked = queue(value)
    domain = (
        f"queue_order_diagnostic:20261005:{job['cipher']}:"
        f"{job['key_index']}:{job['policy']}"
    )
    seed = int.from_bytes(hashlib.sha256(domain.encode()).digest(), "big")
    shuffled = list(ranked)
    random.Random(seed).shuffle(shuffled)
    return tuple(shuffled)


def validate_preparation(result: dict, job: dict) -> None:
    if result.get("job_id") != job["case_id"] + "__prepare":
        raise ValueError("Preparation identity differs")
    ranked, shuffled = queue(result.get("ranked")), queue(result.get("shuffled"))
    if set(ranked) != set(shuffled) or shuffled != permutation(ranked, job):
        raise ValueError("Shuffled membership/stream differs")
    if result.get("set_sha256") != ordered_hash(sorted(ranked)):
        raise ValueError("Candidate set hash differs")
    for order, values in (("ranked", ranked), ("shuffled", shuffled)):
        if result.get(order + "_sha256") != ordered_hash(values):
            raise ValueError("Candidate order hash differs")


def public_case(dataset: dict, job: dict) -> dict:
    """Strip all truth and fresh observations before ranking dispatch."""
    if (
        type(dataset.get("seed")) is not int
        or dataset.get("seed") != 999999
        or dataset.get("split_role") != "demo"
    ):
        raise ValueError("Only the registered opened demo cohort is eligible")
    meta = dataset.get("metadata", {})
    if (
        meta.get("key_count_per_cipher"),
        meta.get("max_differential_pairs"),
        meta.get("verification_blocks_count"),
    ) != (8, 64, 2):
        raise ValueError("Unexpected demo cohort dimensions")
    instances = dataset["ciphers"][job["cipher"]]["instances"]
    if type(job.get("key_index")) is not int or not 0 <= job["key_index"] < 8:
        raise ValueError("Invalid demo key_index; expected an integer in0..7")
    if (
        len(instances) != 8
        or any(type(r["instance_index"]) is not int for r in instances)
        or [r["instance_index"] for r in instances] != list(range(8))
    ):
        raise ValueError("Duplicate/missing demo key index")
    row = instances[job["key_index"]]
    pt, ct = (
        row["attack_plaintexts"],
        row["rounds_data"][str(job["rounds"])]["attack_ciphertexts"],
    )
    if len(pt) != 64 or len(ct) != 64:
        raise ValueError("The policy must receive all64 direct pairs")
    pairs = [[list(block(v)) for v in pair] for pair in ct]
    if any(len(pair) != 2 for pair in ct) or any(len(pair) != 2 for pair in pt):
        raise ValueError("A direct pair must have two blocks")
    known = [
        [list(block(p)), list(block(c))]
        for pp, cc in zip(pt, ct, strict=True)
        for p, c in zip(pp, cc, strict=True)
    ]
    return {
        "cipher": job["cipher"],
        "rounds": job["rounds"],
        "instance_id": row["instance_id"],
        "ciphertext_pairs": pairs,
        "known_observations": known,
    }


def fresh_observations(dataset: dict, job: dict) -> list:
    public_case(dataset, job)
    row = dataset["ciphers"][job["cipher"]]["instances"][job["key_index"]]
    pt = row["verification_plaintexts"]
    ct = row["rounds_data"][str(job["rounds"])]["verification_ciphertexts"]
    if (
        len(pt) != 2
        or len(ct) != 2
        or len(set(map(block, pt))) != 2
        or set(map(block, pt))
        & {block(p) for pair in row["attack_plaintexts"] for p in pair}
    ):
        raise ValueError("Verification must use two separate fresh blocks")
    return [[list(block(p)), list(block(c))] for p, c in zip(pt, ct, strict=True)]


def validate_manifest(manifest: dict, root: Path) -> dict[str, Path]:
    if (
        manifest.get("status") != "APPROVED"
        or manifest.get("execution_authorized") is not True
        or canonical_bytes(manifest.get("profile")) != canonical_bytes(PROFILE)
        or canonical_bytes(manifest.get("jobs")) != canonical_bytes(fixed_jobs())
    ):
        raise ValueError("Unapproved or drifting diagnostic profile/grid")
    if canonical_bytes(manifest.get("pilot_gate")) != canonical_bytes(
        {
            "required": True,
            "authorization_decision": "D-050",
            "job_ids": [j["job_id"] for j in fixed_jobs()[:12]],
        }
    ) or manifest.get("execution_kind") not in {"pilot", "full"}:
        raise ValueError("Missing fixed pilot gate or execution kind")
    files = manifest.get("files", {})
    required = {
        "cohort",
        "python",
        "pycryptosat",
        "protocol",
        "speck_stage1",
        "speck_stage2",
        "speck_classical",
        "simon_ddt",
        "simon_tool",
        "tf_python",
        "bao_vv9",
    }
    source_paths = {str(p.relative_to(root)) for p in (root / "src").rglob("*.py")}
    source_paths |= {
        "scripts/run_neural_ranking_worker.py",
        "scripts/run_queue_order_neural_ranking.py",
        "scripts/evaluate_queue_order_diagnostic.py",
    }
    required |= {"source:" + path for path in source_paths}
    if manifest["execution_kind"] == "full":
        required |= {"pilot_report", "pilot_manifest"}
    if not required <= set(files):
        raise ValueError("Missing imported source/runtime/data identities")
    paths = {key: checked_path(root, record) for key, record in files.items()}
    if manifest["execution_kind"] == "full":
        pilot = read_checked_json(paths["pilot_report"])
        if pilot.get("readiness_passed") is not True or pilot.get(
            "source_commit"
        ) != manifest.get("source_commit"):
            raise ValueError("Passed actual pilot required")
    for name in source_paths:
        if paths["source:" + name] != (root / name).absolute():
            raise ValueError("Pinned source points outside the executed snapshot")
    return paths


def seconds(value: object) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Expected a finite nonnegative duration")
    return float(value)


def validate_replay(record: dict, job: dict, preparation: dict | None) -> None:
    if (
        record.get("job_id") != job["job_id"]
        or record.get("outcome") not in FINAL_OUTCOMES
    ):
        raise ValueError("Unexpected replay identity/outcome")
    success = record.get("success")
    if type(success) is not bool or success != (
        record["outcome"] == "verified_recovery"
    ):
        raise ValueError("Recovery flag must agree with final outcome")
    for field in ("verified_known", "verified_fresh"):
        if type(record.get(field)) is not bool:
            raise ValueError("Verification flags must be genuine booleans")
    if preparation is not None:
        if seconds(record.get("common_preparation_seconds")) != seconds(
            preparation["common_preparation_seconds"]
        ):
            raise ValueError("Common preparation cost differs across arms")
        if record.get("queue_sha256") != preparation[job["order"] + "_sha256"]:
            raise ValueError("Replay used a different candidate order")
        if "attempts" in record:
            attempts = record["attempts"]
            if (
                not isinstance(attempts, list)
                or type(record.get("candidates_tested")) is not int
                or len(attempts) != record["candidates_tested"]
                or len(attempts) > 1024
            ):
                raise ValueError("Invalid completed candidate trace")
            for index, attempt in enumerate(attempts, start=1):
                if (
                    type(attempt.get("candidate_index")) is not int
                    or attempt["candidate_index"] != index
                    or block(attempt.get("candidate"))
                    != block(preparation[job["order"]][index - 1])
                ):
                    raise ValueError("Candidate trace differs from queue prefix")
                seconds(attempt.get("solve_wall_seconds"))
                if attempt.get("solver_status") not in {
                    "sat",
                    "unsat",
                    "solver_timeout",
                    "unsat_or_timeout",
                }:
                    raise ValueError("Unexpected candidate solver status")
                if any(
                    type(attempt.get(v)) is not bool
                    for v in ("verified_known", "verified_fresh")
                ):
                    raise ValueError("Malformed candidate verification flags")
    if record.get("timing_unavailable") is True:
        if record["outcome"] != "infrastructure_interrupted" or success:
            raise ValueError("Only unavailable interrupted trials may lack timing")
        if (
            record.get("charged_online_seconds") is not None
            or record.get("replay_wall_seconds") is not None
        ):
            raise ValueError("Unavailable timing cannot be a zero placeholder")
        return
    common = seconds(record.get("common_preparation_seconds"))
    wall = seconds(record.get("replay_wall_seconds"))
    complete = seconds(record.get("charged_online_seconds"))
    if not math.isclose(complete, common + wall, abs_tol=1e-8):
        raise ValueError("Charged cost does not include common preparation")
    if (
        not success
        and record["verified_known"]
        and record["verified_fresh"]
        and record.get("late_verified_result") is not True
    ):
        raise ValueError(
            "Both verification flags on a failure need an explicit late-result record"
        )
    if success:
        if (
            wall <= 0
            or complete >= 300
            or record.get("verified_known") is not True
            or record.get("verified_fresh") is not True
        ):
            raise ValueError("Unverified or late recovery")
        key = record.get("recovered_key")
        if not isinstance(key, list) or len(key) != 4:
            raise ValueError("Missing recovered master key")
        for value in key:
            word(value)


def load_manifest(path: Path, root: Path) -> tuple[dict, dict[str, Path]]:
    manifest = read_checked_json(path)
    return manifest, validate_manifest(manifest, root)
