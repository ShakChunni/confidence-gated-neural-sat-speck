"""Run the fixed opened-demo identical-set queue-order diagnostic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_sat_speck.followups.runner import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    ledger = run(args.manifest, args.output.absolute(), ROOT, resume=args.resume)
    print(ledger["status"], flush=True)
    if ledger["status"] != "COMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
