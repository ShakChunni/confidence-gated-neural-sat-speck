"""Build, but never solve, a fixed public SIMON four-/64-observation formula.

Each invocation creates one fresh pycryptosat solver. The synthetic public key
is used only to produce self-consistent known observations; no attack result,
true trailing-key rank, or protected seed is accessed.

Run from the repository root with ``PYTHONPATH=.`` so the independent test
oracle is importable without copying its implementation into this script.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import resource
import time

from neural_sat_speck.simon.cipher import encrypt_block
from neural_sat_speck.simon.sat_model import SimonSatModel
from tests.fixtures.simon_reference_oracle import oracle_simon32_64_encrypt

PUBLIC_SYNTHETIC_KEY = (0x1234, 0x5678, 0x9ABC, 0xDEF0)


def _known_observations() -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    rng = random.Random(260926)
    observations = []
    plaintexts = set()
    for _ in range(32):
        pt0 = (rng.randrange(1 << 16), rng.randrange(1 << 16))
        pt1 = (pt0[0], pt0[1] ^ 0x0040)
        if pt0 in plaintexts or pt1 in plaintexts:
            raise AssertionError("synthetic plaintext collision")
        plaintexts.update((pt0, pt1))
        for pt in (pt0, pt1):
            ciphertext = encrypt_block(pt, PUBLIC_SYNTHETIC_KEY, 9)
            if ciphertext != oracle_simon32_64_encrypt(pt, PUBLIC_SYNTHETIC_KEY, 9):
                raise AssertionError("scalar and independent SIMON oracle disagree")
            observations.append((pt, ciphertext))
    return tuple(observations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=int, choices=(4, 64), required=True)
    args = parser.parse_args()
    known = _known_observations()
    started = time.perf_counter()
    model = SimonSatModel.from_known_pairs(known[: args.observations], rounds=9)
    build_seconds = time.perf_counter() - started
    expected = {
        "variables": 384 + 320 * args.observations,
        "cnf_clauses": 496 * args.observations,
        "xor_clauses": 320 + 144 * args.observations,
    }
    observed = {
        "variables": model.variable_count,
        "cnf_clauses": model.cnf_clause_count,
        "xor_clauses": model.xor_clause_count,
    }
    if observed != expected:
        raise AssertionError(f"SIMON formula-size recurrence mismatch: {observed} != {expected}")
    if len(model.known_pairs) != args.observations:
        raise AssertionError("formula silently dropped observations")
    print(
        json.dumps(
            {
                "classification": "ENGINEERING_PUBLIC_SYNTHETIC_FORMULA_SIZE_ONLY",
                "cipher": "SIMON32/64",
                "rounds": 9,
                "direct_plaintext_pairs_generated": 32,
                "known_observations_in_formula": args.observations,
                "public_synthetic_key_only": True,
                "scalar_independent_oracle_equivalent": True,
                "solve_calls": 0,
                "fresh_solver_per_invocation": True,
                "backend": "pycryptosat in-memory native XOR, one thread",
                "formula": observed,
                "expected_formula": expected,
                "build_seconds": build_seconds,
                "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
