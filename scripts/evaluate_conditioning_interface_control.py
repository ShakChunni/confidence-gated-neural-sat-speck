"""Execute the frozen pilot/full conditioning-interface schedule serially."""

import argparse
from pathlib import Path

from neural_sat_speck.followups.conditioning_runner import execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ledger = execute(args.manifest.absolute(), args.output.absolute())
    if ledger["status"] != "COMPLETE":
        raise SystemExit(f"Campaign stopped: {ledger['stop_reason']}")


if __name__ == "__main__":
    main()
