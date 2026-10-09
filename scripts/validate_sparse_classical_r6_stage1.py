#!/usr/bin/env python3
"""Engineering validation of the sparse 6-round table as a 7-round SPECK Stage-1 ranker.

Uses synthetic instances only (demo seed 999999; no cohort keys): 64 direct
chosen-plaintext pairs with input difference (0x0040, 0) per key, 7 rounds.
Reports, for each synthetic key, the rank of the true final-round subkey under
``rank_classical_subkeys`` and the ranking wall time. Under D-061 these numbers
are reported but do not decide inclusion; inclusion needs only a working table
and ranking under 120 s per key.

Run as:
    python3 -I scripts/validate_sparse_classical_r6_stage1.py --table T --keys 32 --output OUT.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from neural_sat_speck.attacks.classical_ranking import (
    ClassicalDistributionTable,
    rank_classical_subkeys,
)
from neural_sat_speck.backend.artifacts import cipher_functions
from neural_sat_speck.data.attack import AttackInstance, AttackPair
from neural_sat_speck.data.seed_policy import SeedRole, validate_seed_for_role

ROUNDS = 7
PAIRS = 64


def synthetic_instance(rng: random.Random, encrypt, expand) -> tuple[AttackInstance, int]:
    key = tuple(rng.randrange(65536) for _ in range(4))
    pairs = []
    for _ in range(PAIRS):
        p0 = (rng.randrange(65536), rng.randrange(65536))
        p1 = (p0[0] ^ 0x0040, p0[1])
        pairs.append(
            AttackPair(
                plaintext0=p0,
                plaintext1=p1,
                ciphertext0=tuple(encrypt(p0, key, ROUNDS)),
                ciphertext1=tuple(encrypt(p1, key, ROUNDS)),
            )
        )
    final = expand(key, ROUNDS)[-1]
    return (
        AttackInstance(
            instance_id="synthetic-demo",
            master_key=key,
            final_round_subkey=final,
            pairs=tuple(pairs),
        ),
        final,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table", type=Path, required=True)
    p.add_argument("--keys", type=int, default=32)
    p.add_argument("--seed", type=int, default=999999)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    validate_seed_for_role(args.seed, SeedRole.DEMO.value)
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    table = ClassicalDistributionTable.load(args.table)
    if table.rounds != ROUNDS - 1:
        raise SystemExit(f"Table has {table.rounds} rounds; Stage 1 for R7 needs 6")
    encrypt, expand = cipher_functions("speck32_64")
    rng = random.Random(args.seed)
    rows = []
    for i in range(args.keys):
        inst, final = synthetic_instance(rng, encrypt, expand)
        t = time.perf_counter()
        ranking = rank_classical_subkeys(inst, table)
        secs = time.perf_counter() - t
        order = list(ranking.candidate_order)
        rows.append({"index": i, "true_rank_1based": order.index(final) + 1, "seconds": secs})
    ranks = sorted(r["true_rank_1based"] for r in rows)
    summary = {
        "table_sha256_note": "see table .generation.json",
        "keys": args.keys,
        "seed": args.seed,
        "rounds": ROUNDS,
        "pairs": PAIRS,
        "top1": sum(r <= 1 for r in ranks),
        "top8": sum(r <= 8 for r in ranks),
        "top32": sum(r <= 32 for r in ranks),
        "median_rank": ranks[len(ranks) // 2],
        "max_seconds": max(r["seconds"] for r in rows),
        "rows": rows,
    }
    args.output.write_text(json.dumps(summary, indent=2))
    print({k: v for k, v in summary.items() if k != "rows"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
