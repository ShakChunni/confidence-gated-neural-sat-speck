"""Use the existing deadline guardian and stop if its coordinator disappears."""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from pathlib import Path

from neural_sat_speck.backend import watchdog
from neural_sat_speck.backend.guard import atomic_json, read_checked_json


def main() -> None:
    payload = read_checked_json(Path(sys.argv[1]))
    parent = os.getppid()
    if parent != payload["coordinator_pid"]:
        raise RuntimeError("Guardian was not started by its recorded coordinator")
    done = threading.Event()

    def monitor_parent():
        while not done.wait(0.05):
            # Live parent relationship avoids signalling a saved/recycled PID.
            if os.getppid() != parent:
                atomic_json(
                    Path(payload["directory"]) / "coordinator_loss.json",
                    {
                        "coordinator_pid": parent,
                        "observed_monotonic": time.perf_counter(),
                    },
                )
                os.kill(os.getpid(), signal.SIGTERM)
                return

    observer = threading.Thread(target=monitor_parent, daemon=True)
    observer.start()
    try:
        watchdog.main()
    except RuntimeError:
        path = Path(payload["receipt"])
        if path.exists():
            receipt = read_checked_json(path)
            if (
                receipt.get("error")
                == "RuntimeError: Supervisor stopped before dispatch"
                and "worker_pid" not in receipt
                and (Path(payload["directory"]) / "coordinator_loss.json").exists()
            ):
                observed = time.perf_counter()
                receipt.update(
                    status="coordinator_stop",
                    worker_not_started=True,
                    process_group_absent=True,
                    observed_monotonic=observed,
                    cleanup_completed_monotonic=observed,
                )
                atomic_json(path, receipt)
                return
        raise
    finally:
        done.set()
        observer.join(timeout=1)


if __name__ == "__main__":
    main()
