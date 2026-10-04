#!/usr/bin/env python3
"""Run only an approved fixed backend manifest; not a default experiment launcher."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True
sys.pycache_prefix = str(ROOT / "runtime" / "backend_bootstrap" / os.urandom(16).hex())

from neural_sat_speck.backend.runner import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorization-receipt", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    ledger = run(
        args.manifest.resolve(),
        args.output.resolve(),
        authorization_path=args.authorization_receipt.resolve(),
        resume=args.resume,
    )
    print(f"{len(ledger['records'])}/160: {ledger['status']}")


if __name__ == "__main__":
    main()
