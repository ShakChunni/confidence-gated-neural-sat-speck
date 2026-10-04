"""Record the native launch thread, then exec the solver with the same PID.

This is a pre-exec snapshot, not continuous certification of native threads.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from neural_sat_speck.backend.guard import atomic_json


def main() -> None:
    payload = json.loads(Path(sys.argv[1]).read_bytes())
    from neural_sat_speck.protocol.worker_telemetry import certify_worker_or_raise

    receipt = certify_worker_or_raise(Path(payload["repo_root"]))
    raw = receipt.to_dict()
    raw["observation_scope"] = "launch_thread_before_exec_same_pid"
    raw["solver_arguments"] = payload["arguments"]
    atomic_json(Path(payload["receipt"]), raw)
    os.execv(payload["arguments"][0], payload["arguments"])


if __name__ == "__main__":
    main()
