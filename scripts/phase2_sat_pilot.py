"""Run small unknown-key CryptoMiniSat feasibility pilots."""

from __future__ import annotations

import argparse
import json
from typing import Final

from neural_sat_speck.sat.model import SpeckSatModel
from neural_sat_speck.speck.cipher import encrypt_block
from tests.fixtures.speck32_64_vectors import MASTER_KEY

PLAINTEXTS: Final[tuple[tuple[int, int], ...]] = (
    (0x6574, 0x694C),
    (0x0000, 0x0000),
    (0x1234, 0xABCD),
    (0xDEAD, 0xBEEF),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an exact reduced-round SPECK SAT pilot."
    )
    parser.add_argument(
        "--rounds",
        type=int,
        nargs="+",
        default=[2, 4, 7],
    )
    parser.add_argument("--pair-count", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not 1 <= args.pair_count <= len(PLAINTEXTS):
        raise SystemExit(
            f"--pair-count must be between 1 and {len(PLAINTEXTS)}"
        )

    results = []
    for rounds in args.rounds:
        plaintexts = PLAINTEXTS[: args.pair_count]
        pairs = tuple(
            (plaintext, encrypt_block(plaintext, MASTER_KEY, rounds))
            for plaintext in plaintexts
        )
        model = SpeckSatModel.from_known_pairs(pairs, rounds)
        result = model.solve(time_limit=args.timeout)

        recovered_key = None
        verified = False
        if result.satisfiable and result.solution is not None:
            try:
                recovered_key = model.master_key_from_solution(result.solution)
            except ValueError:
                recovered_key = None
            if recovered_key is not None:
                verified = all(
                    encrypt_block(plaintext, recovered_key, rounds)
                    == ciphertext
                    for plaintext, ciphertext in pairs
                )

        results.append(
            {
                "rounds": rounds,
                "pair_count": args.pair_count,
                "timeout_seconds": args.timeout,
                "satisfiable": result.satisfiable,
                "timed_out": result.timed_out,
                "verified": verified,
                "recovered_master_key": (
                    [f"{word:04x}" for word in recovered_key]
                    if recovered_key is not None
                    else None
                ),
                "variable_count": model.variable_count,
                "runtime_seconds": result.elapsed_seconds,
            }
        )

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
