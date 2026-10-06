"""Pin existing demo inputs for the reviewed interface control; no new keys."""

import argparse
from pathlib import Path

from neural_sat_speck.followups.conditioning_control import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--pilot-ledger", type=Path)
    args = parser.parse_args()
    print(
        prepare(
            args.source_root.absolute(),
            args.cohort.absolute(),
            args.output.absolute(),
            args.mode,
            args.pilot_ledger.absolute() if args.pilot_ledger else None,
        )
    )


if __name__ == "__main__":
    main()
