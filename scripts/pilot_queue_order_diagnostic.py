"""Run all four actual policy paths before the fresh queue-order diagnostic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_sat_speck.followups.pilot import run  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = run(args.manifest, args.output.absolute(), ROOT)
    print(report["status"], flush=True)
    if report.get("readiness_passed") is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
