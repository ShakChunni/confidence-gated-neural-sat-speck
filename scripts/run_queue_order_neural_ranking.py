"""Certify the actual TensorFlow child, then execute the preserved VV worker."""

from __future__ import annotations

import importlib.util
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_sat_speck.backend.guard import atomic_json  # noqa: E402


def telemetry_module():
    # The TensorFlow environment intentionally has no SAT extension. Loading
    # this pinned stdlib-only file avoids protocol/__init__'s SAT imports.
    spec = importlib.util.spec_from_file_location(
        "queue_order_tf_telemetry",
        ROOT / "src/neural_sat_speck/protocol/worker_telemetry.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    receipt = telemetry_module().certify_worker_or_raise(ROOT)
    target = Path(os.environ["QUEUE_ORDER_TF_RECEIPT"])
    atomic_json(target, receipt.to_dict())
    if receipt.git_commit != os.environ["QUEUE_ORDER_SOURCE_COMMIT"]:
        raise ValueError("TensorFlow child source commit differs")
    # Preserve stdin and --stdin argv, model loading and all scoring semantics.
    runpy.run_path(
        str(ROOT / "scripts/run_neural_ranking_worker.py"), run_name="__main__"
    )


if __name__ == "__main__":
    main()
