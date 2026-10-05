"""Execute the fixed pilot or full opened-demo Stage 2 ranking control."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_sat_speck.followups.stage2_runner import (  # noqa: E402
    CampaignCeilingReached,
    run,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()

    def deadline_reached(signum, frame):
        raise CampaignCeilingReached("Fixed eight-hour campaign ceiling reached")

    previous = signal.signal(signal.SIGALRM, deadline_reached)
    signal.setitimer(signal.ITIMER_REAL, 28800.0)
    try:
        ledger = run(
            args.manifest.absolute(),
            args.output.absolute(),
            ROOT,
            campaign_started=started,
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    print(ledger["status"], flush=True)
    if ledger["status"] != "COMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
