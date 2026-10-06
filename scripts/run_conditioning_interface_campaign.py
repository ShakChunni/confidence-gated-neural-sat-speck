"""Run a separate mechanical pilot, then the authorized fixed opened campaign."""

import argparse
import subprocess
import sys
from pathlib import Path

from neural_sat_speck.followups.conditioning_control import prepare
from neural_sat_speck.followups.conditioning_runner import execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root, cohort, output = (
        p.absolute() for p in (args.source_root, args.cohort, args.output_root)
    )
    if output.exists():
        raise SystemExit("Campaign output root exists; no retry or overwrite")
    output.mkdir(parents=True)
    pilot_manifest = prepare(root, cohort, output / "pilot_inputs", "pilot", None)
    pilot = execute(pilot_manifest, output / "pilot")
    if pilot["status"] != "COMPLETE":
        raise SystemExit(f"Pilot failed its mechanical gate: {pilot['stop_reason']}")
    # prepare + execute re-audit the actual checked pilot before a full dispatch.
    full_manifest = prepare(
        root, cohort, output / "full_inputs", "full", output / "pilot" / "ledger.json"
    )
    full = execute(full_manifest, output / "full")
    if full["status"] != "COMPLETE":
        raise SystemExit(f"Full campaign stopped: {full['stop_reason']}")
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/analyze_conditioning_interface_control.py"),
            "--ledger",
            str(output / "full" / "ledger.json"),
            "--manifest",
            str(full_manifest),
            "--output",
            str(output / "analysis.json"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
