#!/usr/bin/env python3
"""Eleven-round SIMON32/64 key-recovery campaign (D-067 draft, not yet frozen).

One cell = one key x one arm under a 300 s clock that starts before any model
or table is opened. Three ranked stages fix the last three round keys
(k10, k9, k8); the remaining 16 schedule bits are then enumerated by the
native brute-force completer (tools/bruteforce_completion.c, as in D-065).

  S11-C  9R rows (full 8R table)  -> 8R rows (7R table) -> 7R table
  S11-H  Bao VV10                 -> 8R rows (7R table) -> 7R table
  S11-N  Bao VV10                 -> Bao VV9            -> Bao VV8

Each stage screens all 65,536 words on 16 pairs and rescores the top 512 on
all 64 pairs (as at 10 rounds). The queue is ordered by summed stage ranks,
then stage-1, stage-2 and stage-3 rank, then words. A completion is accepted
only if the master key reproduces all 128 known blocks (inside the C call)
and 2 independent fresh blocks (Python reference). Cells run serially and arm
order rotates by key index. The 8R rows for all stage-2 parents come from one
pass over the 7R table. TensorFlow intra-op threads (1 or 8) must be selected
explicitly on the CLI; inter-op and numerical-library threads are one.

The deadline is cooperative, as in the SPECK runner: checks surround ranking
calls and each completion chunk. A synchronous library call can overrun the
clock, but a late result is never counted as recovered. Actual elapsed time
is retained. This draft is not a frozen SIMON execution protocol.

Run with the TensorFlow interpreter (runtime/simon_tf_venv/bin/python3.12).
Refuses to overwrite an existing ledger.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

# Set the fixed numerical profile before importing NumPy or optional Torch.
for _variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = "1"

import numpy as np  # noqa: E402
from bruteforce_completion import BruteforceLibrary, build_library  # noqa: E402

from neural_sat_speck.data.seed_policy import PROTECTED_RESEARCH_SEEDS  # noqa: E402
from neural_sat_speck.simon import published_vv  # noqa: E402
from neural_sat_speck.simon.attack_instance import (  # noqa: E402
    FROZEN_EVALUATION_SEEDS,
    PERMITTED_SIMON_ROLES,
    SimonAttackInstanceConfig,
    generate_simon_attack_instances,
)
from neural_sat_speck.simon.cipher import decrypt_round  # noqa: E402
from neural_sat_speck.simon.classical_ranking import (  # noqa: E402
    FullPredictedDdt,
    rank_classical_domain,
    rank_stage1_ondemand,
)
from neural_sat_speck.simon.key_schedule import expand_key  # noqa: E402
from neural_sat_speck.simon.ranking_features import (  # noqa: E402
    candidate_difference_words,
    candidate_transformed_words,
)
from neural_sat_speck.simon.verification import verify_simon_master_key  # noqa: E402

ROUNDS = 11
PAIRS = 64
SCREEN_PAIRS = 16
RESCORE_TOP = 512
BUDGET = 300.0
BF_CHUNK = 1 << 16  # the whole 16-bit space in one call
DEMO_SEED = 999999
TEMP_ROOT = REPO / "runtime/simon11_tmp"
PROHIBITED_SEEDS = (
    PROTECTED_RESEARCH_SEEDS
    | FROZEN_EVALUATION_SEEDS
    | {
        98001,
        97001,
        99001,
        101001,
        201001,
        300001,
        406737,
        669981,
        6503664708853956672,
        701004,
    }
)

# VV10 is not in the original loader's pin list. Its HDF5 has the same
# 616-layer topology and 449,457 parameters as VV8/VV9, and its Lambda
# bytecode is byte-identical to VV8's K.clip(x, -10, 10) except one marshal
# FLAG_REF bit on the empty line table. Pinned at Bao commit 09cc813.
VV10_PINS = (
    "628b9d4fb2399332313f0cf367b655676b9d2d671b103520225edc9ac1b0bf2f",
    "bdfb15f1710fa1f287691b1c21e5754d65c01e6aebc38a5b00d0e28036de7274",
)

ASSETS = {
    "table7": REPO
    / "results/simon_confirmation/runtime_backup/resources/simon_predicted_ddt_7r.bin",
    "table8": REPO / "runtime/higher_round_tables/simon_ddt_r8_full.bin",
    "row_tool": REPO / "runtime/higher_round_tables/simon_queried_row_next",
    "vv8": REPO / "runtime/bao-simon-audit/simonNDvsDD/SENet/ND_VV_Simon32_8R.h5",
    "vv9": REPO / "runtime/bao-simon-audit/simonNDvsDD/SENet/ND_VV_Simon32_9R.h5",
    "vv10": REPO / "runtime/bao_vv_extra/ND_VV_Simon32_10R.h5",
}
SOURCES = [
    "scripts/run_simon_higher_round.py",
    "tools/simon_queried_row_next.cpp",
    "tools/simon_full_table_next.cpp",
    "src/neural_sat_speck/simon/classical_ranking.py",
    "src/neural_sat_speck/simon/published_vv.py",
    "src/neural_sat_speck/simon/published_vv_ranking.py",
    "src/neural_sat_speck/simon/ranking_features.py",
    "scripts/bruteforce_completion.py",
    "tools/bruteforce_completion.c",
]
# Dependencies missing from the original draft's provenance list.
SUPPORT_SOURCES = [
    "tools/bruteforce_completion.h",
    "tools/reference_encryption.c",
    "tools/reference_encryption.h",
    "src/neural_sat_speck/analysis/encryption_calibration.py",
    "src/neural_sat_speck/simon/__init__.py",
    "src/neural_sat_speck/data/seed_policy.py",
    "src/neural_sat_speck/simon/attack_instance.py",
    "src/neural_sat_speck/simon/cipher.py",
    "src/neural_sat_speck/simon/key_schedule.py",
    "src/neural_sat_speck/simon/primitives.py",
    "src/neural_sat_speck/simon/verification.py",
    "src/neural_sat_speck/speck/cipher.py",
    "src/neural_sat_speck/speck/key_schedule.py",
    "src/neural_sat_speck/speck/primitives.py",
]

ARMS = {
    "S11-C": {"stages": ("c9", "c8", "c7"), "widths": (8, 8, 16)},
    "S11-H": {"stages": ("vv10", "c8", "c7"), "widths": (8, 8, 16)},
    "S11-N": {"stages": ("vv10", "vv9", "vv8"), "widths": (1, 1, 128)},
}

BF: BruteforceLibrary | None = None
TF = None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def peel(pairs, key: int):
    """Undo one round with a fixed round key on both blocks of every pair."""
    return [(decrypt_round(a, key), decrypt_round(b, key)) for a, b in pairs]


def queue_order(paths):
    """paths: (r1, r2, r3, (k10, k9, k8)) with zero-based ranks.

    Returns chronological word tuples (k8, k9, k10) sorted by rank sum, then
    stage-1, stage-2, stage-3 rank, then words. Duplicates keep the first.
    """
    seen, out = set(), []
    for r1, r2, r3, (k10, k9, k8) in sorted(
        paths, key=lambda p: (p[0] + p[1] + p[2], p[0], p[1], p[2], p[3])
    ):
        words = (k8, k9, k10)
        if words not in seen:
            seen.add(words)
            out.append(words)
    return out


def _score_rows(
    pairs,
    log_rows,
    screen_pairs=SCREEN_PAIRS,
    rescore_top=RESCORE_TOP,
    batch=1024,
    *,
    check_budget=None,
):
    """Screen and rescore one domain from per-pair log rows (as rank_stage1_ondemand)."""
    if not 1 <= screen_pairs <= len(pairs) or not 1 <= rescore_top <= 1 << 16:
        raise ValueError("invalid screen/rescore counts")
    if not 1 <= batch <= 1024:
        raise ValueError("candidate batch must be between 1 and 1024")
    log_rows = np.asarray(log_rows, dtype=np.float64)
    if log_rows.shape != (len(pairs), 1 << 16):
        raise ValueError("queried log-row shape mismatch")
    if not np.all(np.isfinite(log_rows)) or np.any(log_rows > 0):
        raise ValueError("invalid queried log probabilities")
    anchor = candidate_difference_words(candidate_transformed_words(pairs, [0]))[
        0, :, 0
    ]
    screen = np.empty(1 << 16, dtype=np.float64)
    sp, sl = pairs[:screen_pairs], log_rows[:screen_pairs]
    for start in range(0, 1 << 16, batch):
        if check_budget is not None:
            check_budget()
        cands = list(range(start, min(start + batch, 1 << 16)))
        differences = candidate_difference_words(candidate_transformed_words(sp, cands))
        if not np.all(differences[:, :, 0] == anchor[None, :screen_pairs]):
            raise ValueError("queried left difference depends on candidate")
        right = differences[:, :, 1].astype(np.intp)
        screen[start : start + len(cands)] = sl[
            np.arange(screen_pairs)[None, :], right
        ].sum(axis=1, dtype=np.float64)
    if check_budget is not None:
        check_budget()
    words = np.arange(1 << 16, dtype=np.uint16)
    top = words[np.lexsort((words, -screen))[:rescore_top]]
    right = candidate_difference_words(candidate_transformed_words(pairs, top))[
        :, :, 1
    ].astype(np.intp)
    full = log_rows[np.arange(len(pairs))[None, :], right].sum(axis=1, dtype=np.float64)
    return top[np.lexsort((top, -full))]


def _validate_table(path: Path, expected_rounds: int) -> None:
    """Check depth/storage before passing a dense table to the generic tool."""
    with Path(path).open("rb") as handle:
        header = handle.read(32)
    if len(header) != 32:
        raise ValueError("truncated dense-table header")
    bits, rounds, initial, exponent, count = struct.unpack("<IIIIQ", header[8:])
    magic = b"SMDT0002" if expected_rounds == 7 else b"SMDT0003"
    expected_exponent = 60 if expected_rounds == 7 else 0
    if expected_rounds not in (7, 8) or (
        header[:8],
        bits,
        rounds,
        initial,
        exponent,
        count,
    ) != (magic, 16, expected_rounds, 0x40, expected_exponent, 1 << 32):
        raise ValueError("dense-table depth/storage mismatch")
    if Path(path).stat().st_size != 32 + (1 << 32) * 8:
        raise ValueError("dense-table length mismatch")


def rank_ondemand_batch(pair_sets, tool: Path, table: Path, *, check_budget=None):
    """8R (or 9R) rows for several domains from one pass over the dense table."""
    if not pair_sets or any(len(p) != PAIRS for p in pair_sets):
        raise ValueError("each queried domain must contain 64 pairs")
    _validate_table(table, 7)
    anchors = [
        candidate_difference_words(candidate_transformed_words(p, [0]))[0, :, 0]
        for p in pair_sets
    ]
    rows = sorted({int(v) for a in anchors for v in a})
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="simon11_rows_", dir=TEMP_ROOT) as tmp:
        out = Path(tmp) / "rows.bin"
        proc = subprocess.run(
            [str(tool), str(table), str(out)] + [str(r) for r in rows],
            capture_output=True,
            text=True,
            check=True,
            timeout=check_budget() if check_budget is not None else None,
        )
        metrics = json.loads(proc.stdout)
        if not isinstance(metrics, dict) or metrics.get("input_rounds") != 7:
            raise ValueError("queried-row tool input depth mismatch")
        if metrics.get("num_query_rows") != len(rows):
            raise ValueError("queried-row tool row count mismatch")
        if out.stat().st_size != len(rows) * (1 << 16) * 8:
            raise ValueError("queried-row binary length mismatch")
        matrix = np.fromfile(out, dtype="<f8").reshape(len(rows), 1 << 16)
        if not np.all(np.isfinite(matrix)) or np.any((matrix < 0) | (matrix > 1)):
            raise ValueError("invalid queried-row probabilities")
    index = {r: i for i, r in enumerate(rows)}
    orders = []
    for pairs, anchor in zip(pair_sets, anchors):
        if check_budget is not None:
            check_budget()
        log_rows = np.stack([np.log2(matrix[index[int(v)]] + 1e-64) for v in anchor])
        orders.append(_score_rows(pairs, log_rows, check_budget=check_budget))
    return orders, metrics


def load_vv(name: str):
    rounds = int(name[2:])
    previous = published_vv._CHECKPOINTS.get(10)
    try:
        if rounds == 10:
            published_vv._CHECKPOINTS[10] = VV10_PINS
        model, _ = published_vv.load_published_vv_checkpoint(ASSETS[name], rounds)
    finally:
        if rounds == 10:
            if previous is None:
                published_vv._CHECKPOINTS.pop(10, None)
            else:
                published_vv._CHECKPOINTS[10] = previous
    return model


def rank_vv(model, pairs):
    from neural_sat_speck.simon.published_vv_ranking import rank_neural_domain

    return np.asarray(
        rank_neural_domain(
            pairs, model, screen_pairs=SCREEN_PAIRS, rescore_top=RESCORE_TOP
        ).order
    )


def position(order, word) -> int | None:
    hit = np.flatnonzero(np.asarray(order) == word)
    return int(hit[0]) + 1 if hit.size else None


class _DeadlineReached(TimeoutError):
    """The cell has no remaining online time."""


def _retained(order, width: int) -> list[int]:
    values = np.asarray(order)
    if values.ndim != 1 or not width <= len(values) <= RESCORE_TOP:
        raise ValueError("ranker returned an incomplete or oversized shortlist")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("ranker returned noninteger words")
    if np.any((values < 0) | (values > 0xFFFF)):
        raise ValueError("ranker returned an invalid word")
    if len(np.unique(values)) != len(values):
        raise ValueError("ranker returned duplicate words")
    return [int(word) for word in values[:width]]


def run_cell(arm_name: str, x) -> dict:
    """Run one isolated ranking/completion cell; truth is joined only at exit."""
    t0 = time.perf_counter()
    rec = {
        "arm": arm_name,
        "key_id": x.instance_id,
        "rounds": ROUNDS,
        "outcome": "error",
        "attempts": 0,
        "verification_rejections": 0,
        "guesses_tested": 0,
        "queue_len": None,
        "queue_complete": False,
        "true_queue_position": None,
        "stage1_true_position": None,
        "stage1_true_rank": None,
        "verification_s": 0.0,
        "recovered_key": None,
        "exact_key_match": None,
        "completer": "bf",
        "words": 3,
    }
    stage1_order = None
    parents, queue, wrong = [], [], []
    models = {}
    valid_instance = False

    def left() -> float:
        remaining = BUDGET - (time.perf_counter() - t0)
        if remaining <= 0:
            raise _DeadlineReached("300-second cell deadline")
        return remaining

    def model(name):
        left()
        if name not in models:
            models[name] = load_vv(name)
            left()
        return models[name]

    try:
        left()
        arm = ARMS[arm_name]
        s1, s2, s3 = arm["stages"]
        m1, m2, m3 = arm["widths"]
        rec["widths"] = list(arm["widths"])
        rec["phase"] = "input_validation"
        if x.rounds != ROUNDS or len(x.pairs) != PAIRS:
            raise ValueError("cell requires 11 rounds and 64 direct pairs")
        known, fresh = tuple(x.known_pairs), tuple(x.verification_pairs)
        if len(known) != 2 * PAIRS or len(fresh) != 2:
            raise ValueError("cell requires 128 known and two fresh observations")
        if any(
            (p.plaintext0[0] ^ p.plaintext1[0], p.plaintext0[1] ^ p.plaintext1[1])
            != (0, 0x40)
            for p in x.pairs
        ):
            raise ValueError("attack pairs must use input difference (0, 0x0040)")
        plaintexts = [pt for pt, _ in known + fresh]
        if len(set(plaintexts)) != len(plaintexts):
            raise ValueError("known and fresh plaintexts must all be distinct")
        valid_instance = True
        pairs = [(p.ciphertext0, p.ciphertext1) for p in x.pairs]

        rec["phase"] = "stage1"
        started = time.perf_counter()
        left()
        if s1 == "c9":
            _validate_table(ASSETS["table8"], 8)
            left()
            ranking, diag = rank_stage1_ondemand(
                pairs,
                tool_path=ASSETS["row_tool"],
                ddt_7r_path=ASSETS["table8"],
                screen_pairs=SCREEN_PAIRS,
                full_pairs=PAIRS,
                rescore_top=RESCORE_TOP,
            )
            stage1_order = ranking.order
            rec["stage1_rows"] = diag
        else:
            stage1_order = rank_vv(model(s1), pairs)
        left()
        parents = _retained(stage1_order, m1)
        level = [(r1, k10, peel(pairs, k10)) for r1, k10 in enumerate(parents)]
        rec["stage1_s"] = time.perf_counter() - started

        rec["phase"] = "stage2"
        started = time.perf_counter()
        left()
        if s2 == "c8":
            orders, diag = rank_ondemand_batch(
                [p for _, _, p in level],
                ASSETS["row_tool"],
                ASSETS["table7"],
                check_budget=left,
            )
            if len(orders) != len(level):
                raise ValueError("Stage 2 returned the wrong number of parent domains")
            rec["stage2_rows"] = diag
        else:
            orders = None
        children = []
        for index, (r1, k10, peeled) in enumerate(level):
            left()
            order = orders[index] if orders is not None else rank_vv(model(s2), peeled)
            left()
            for r2, k9 in enumerate(_retained(order, m2)):
                children.append((r1, r2, k10, k9, peel(peeled, k9)))
        rec["stage2_s"] = time.perf_counter() - started

        rec["phase"] = "stage3"
        started = time.perf_counter()
        left()
        table = (
            FullPredictedDdt(ASSETS["table7"], expected_rounds=7)
            if s3 == "c7"
            else None
        )
        left()
        paths = []
        for r1, r2, k10, k9, peeled in children:
            left()
            if table is not None:
                order = rank_classical_domain(
                    peeled,
                    table,
                    expected_rounds=7,
                    screen_pairs=SCREEN_PAIRS,
                    full_pairs=PAIRS,
                    rescore_top=RESCORE_TOP,
                ).order
            else:
                order = rank_vv(model(s3), peeled)
            left()
            for r3, k8 in enumerate(_retained(order, m3)):
                paths.append((r1, r2, r3, (k10, k9, k8)))
        queue = queue_order(paths)
        left()
        rec["stage3_s"] = time.perf_counter() - started
        rec.update(
            queue_len=len(queue),
            queue_complete=True,
            prep_s=time.perf_counter() - t0,
            outcome="queue_exhausted",
        )
        if BF is None:
            raise RuntimeError("native completer has not been initialized")
        known_pt, known_ct = [pt for pt, _ in known], [ct for _, ct in known]
        rec["phase"] = "completion"
        for words in queue:
            left()
            rec["attempts"] += 1
            lo, candidate_seconds, hit = 0, 0.0, None
            while lo < 1 << 16:
                left()
                hi = min(1 << 16, lo + BF_CHUNK)
                started = time.perf_counter()
                result = BF.complete(
                    "simon", ROUNDS, list(words), known_pt, known_ct, lo, hi
                )
                candidate_seconds += time.perf_counter() - started
                rec["guesses_tested"] += result.tested
                left()
                if not result.found:
                    lo = hi
                    continue
                if (
                    result.master_key is None
                    or result.found_guess is None
                    or not lo <= result.found_guess < hi
                ):
                    raise ValueError("native completer returned an invalid hit")
                rec["phase"] = "verification"
                started = time.perf_counter()
                verification = verify_simon_master_key(
                    result.master_key, known, fresh, ROUNDS
                )
                rec["verification_s"] += time.perf_counter() - started
                rec["verification"] = verification.to_dict()
                left()
                if verification.verified:
                    hit = list(result.master_key)
                    break
                rec["verification_rejections"] += 1
                lo = result.found_guess + 1
                rec["phase"] = "completion"
            if hit is not None:
                rec.update(
                    outcome="recovered",
                    recovered_key=hit,
                    correct_solve_s=candidate_seconds,
                )
                break
            wrong.append(candidate_seconds)
    except (_DeadlineReached, subprocess.TimeoutExpired):
        rec["outcome"] = "deadline"
    except Exception as error:
        rec.update(outcome="error", error_type=type(error).__name__, error=str(error))
    finally:
        # Stop the online clock before joining evaluation truth.
        rec["total_s"] = time.perf_counter() - t0
        rec.setdefault("prep_s", rec["total_s"])
        rec["wrong_n"] = len(wrong)
        rec["wrong_mean_s"] = float(np.mean(wrong)) if wrong else None
        rec["wrong_median_s"] = float(np.median(wrong)) if wrong else None
        rec["deadline_overrun_s"] = max(0.0, rec["total_s"] - BUDGET)
        if rec["outcome"] == "recovered" and rec["total_s"] > BUDGET:
            rec["outcome"] = "deadline"
            rec["late_verified_key"] = rec["recovered_key"]
            rec["recovered_key"] = None
        # Evaluation-only truth join. No ranking, queue or stopping decision
        # receives these values; in particular, absence never stops search.
        try:
            if valid_instance:
                ks = expand_key(x.master_key, ROUNDS)
                if stage1_order is not None:
                    rec["stage1_true_position"] = position(stage1_order, ks[-1])
                    rec["stage1_true_rank"] = position(parents, ks[-1])
                if rec["queue_complete"]:
                    truth = tuple(ks[-3:])
                    rec["true_queue_position"] = (
                        queue.index(truth) if truth in queue else None
                    )
                if rec["recovered_key"] is not None:
                    rec["exact_key_match"] = tuple(rec["recovered_key"]) == tuple(
                        x.master_key
                    )
        except Exception as error:
            rec.update(outcome="error", diagnostic_error=str(error))
    return rec


def validate_seed_role(seed: int, role: str | None = None) -> str:
    """Validate labels without drawing keys or opening an evaluation family."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 1 << 64:
        raise ValueError("seed must be a nonnegative 64-bit integer")
    if seed in PROHIBITED_SEEDS:
        raise ValueError(f"Seed {seed} is protected and cannot be used by this runner")
    if role is None:
        if seed != DEMO_SEED:
            raise ValueError("a non-demo seed requires an explicit --role")
        role = "demo"
    if role not in PERMITTED_SIMON_ROLES:
        raise ValueError("unsupported SIMON data role")
    if (role == "demo") != (seed == DEMO_SEED):
        raise ValueError("seed 999999 and role demo must be used together")
    # Reuse role-specific exclusions too, without drawing an instance.
    SimonAttackInstanceConfig(
        instance_count=1,
        pair_count=PAIRS,
        rounds=ROUNDS,
        seed=seed,
        split_role=role,
        verification_pair_count=2,
    )
    return role


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--role", choices=sorted(PERMITTED_SIMON_ROLES))
    parser.add_argument("--keys", type=int, required=True)
    parser.add_argument("--first-key", type=int, default=0)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument(
        "--tf-intra-op-threads", type=int, choices=(1, 8), required=True
    )
    args = parser.parse_args(argv)
    args.command_line = list(sys.argv[1:] if argv is None else argv)
    try:
        args.role = validate_seed_role(args.seed, args.role)
        if args.keys <= 0 or args.first_key < 0:
            raise ValueError("--keys must be positive and --first-key nonnegative")
        arms = args.arms.split(",")
        if not arms or len(arms) != len(set(arms)) or any(a not in ARMS for a in arms):
            raise ValueError("--arms must be distinct known arms separated by commas")
    except ValueError as error:
        parser.error(str(error))
    return args


def create_ledger(path: Path, header: dict) -> None:
    """Exclusive creation; never truncate a prior or interrupted campaign."""
    line = json.dumps(header, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def append_ledger(path: Path, record: dict) -> None:
    """Append to an existing ledger; deletion must not create a new headerless file."""
    line = json.dumps(record, allow_nan=False) + "\n"
    flags = os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def configure_tensorflow(intra_op: int) -> None:
    global TF
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(intra_op)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    TF = tf


def make_header(args, arms: list[str], library: Path) -> dict:
    sources = list(dict.fromkeys(SOURCES + SUPPORT_SOURCES))
    commit = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "--", *sources],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    needed = set()
    for arm in arms:
        for stage in ARMS[arm]["stages"]:
            needed.update(
                {"table8", "row_tool"}
                if stage == "c9"
                else {"table7", "row_tool"}
                if stage == "c8"
                else {"table7"}
                if stage == "c7"
                else {stage}
            )
    return {
        "type": "header",
        "schema": "simon11_campaign.v1",
        "protocol_status": "D-067 draft; not frozen",
        "seed": args.seed,
        "role": args.role,
        "keys": args.keys,
        "first_key": args.first_key,
        "arms": arms,
        "command": [sys.executable, str(Path(__file__).resolve()), *args.command_line],
        "rounds": ROUNDS,
        "pairs": PAIRS,
        "screen_pairs": SCREEN_PAIRS,
        "rescore_top": RESCORE_TOP,
        "budget_s": BUDGET,
        "method_configurations": {a: ARMS[a] for a in arms},
        "input_difference": [0, 0x40],
        "fresh_blocks": 2,
        "clock_policy": "cooperative; late recoveries rejected; preflight hashing/build excluded",
        "queue_order": "rank sum, r1, r2, r3, newest-first words; deduplicate; output chronological",
        "truth_position_scope": "Stage 1: 1-based within rescored 512; queue: 0-based",
        "stage1_true_rank_scope": "1-based within retained Stage-1 parents only",
        "git_commit": commit,
        "runner_dirty": bool(dirty),
        "source_status": dirty,
        "script_sha256": sha256(Path(__file__)),
        "source_shas": {p: sha256(REPO / p) for p in sources},
        "assets": {
            k: {"path": str(ASSETS[k]), "sha256": sha256(ASSETS[k])}
            for k in sorted(needed)
        },
        "vv10_pins": list(VV10_PINS),
        "bf_library": {
            "path": str(library),
            "sha256": sha256(library),
            "chunk": BF_CHUNK,
        },
        "tf_intra_op_threads": args.tf_intra_op_threads,
        "tf_inter_op_threads": 1,
        "tensorflow": TF.__version__ if TF is not None else None,
        "tensorflow_device": "CPU",
        "numerical_threads": 1,
        "temporary_directory": str(TEMP_ROOT),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "byteorder": sys.byteorder,
        "started_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.role != "demo":
        raise SystemExit(
            "D-067 is unfrozen: non-demo execution needs its own protocol and approval"
        )
    if args.ledger.exists() or args.ledger.is_symlink():
        raise SystemExit("Refusing to overwrite existing ledger")
    if sys.byteorder != "little":
        raise SystemExit("the supplied row tool requires a little-endian host")
    arms = args.arms.split(",")
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(TEMP_ROOT)
    if "torch" in sys.modules:
        sys.modules["torch"].set_num_threads(1)
    if any(stage.startswith("vv") for a in arms for stage in ARMS[a]["stages"]):
        configure_tensorflow(args.tf_intra_op_threads)
    global BF
    library = build_library()
    BF = BruteforceLibrary(library)
    create_ledger(args.ledger, make_header(args, arms, library))
    instances = generate_simon_attack_instances(
        SimonAttackInstanceConfig(
            instance_count=args.first_key + args.keys,
            pair_count=PAIRS,
            rounds=ROUNDS,
            seed=args.seed,
            split_role=args.role,
            verification_pair_count=2,
        )
    )
    for index in range(args.first_key, args.first_key + args.keys):
        order = arms[index % len(arms) :] + arms[: index % len(arms)]
        for arm_position, arm in enumerate(order):
            record = run_cell(arm, instances[index])
            record.update(key_index=index, arm_position=arm_position)
            append_ledger(args.ledger, record)
            print(
                json.dumps(
                    {
                        k: record[k]
                        for k in ("key_index", "arm", "outcome", "attempts", "total_s")
                    }
                ),
                flush=True,
            )
            if TF is not None:
                TF.keras.backend.clear_session()
            gc.collect()
            if record["outcome"] == "error":
                return 1  # preserve the error; do not repeat invalid setup across the cohort
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
