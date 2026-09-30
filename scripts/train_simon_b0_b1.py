#!/usr/bin/env python3
"""CLI runner for manifest-driven SIMON32/64 B0->B1 training.

Governing Rules and Invariants (AGENTS.md, D-029, D-034, D-037):
- Rejects unverified development or unseen attack seeds.
- Preflight checks execute audit_training_key_roles and verify pair features
  against both the scalar cipher reference and the clean-room independent reference oracle.
- Checkpoint selection is strictly by minimum validation BCE loss (exact ties pick earlier epoch).
- Weak but finite validation performance (e.g. 50.8%) is retained and reported as descriptive.
- Aborts on NaN/Inf loss, zero gradients across the architecture, or non-finite weights.
- Full multi-hour training on 1,000,000 pairs / 15,625 groups is guarded under D-037 and
  requires explicit --execute-full-training confirmation. Use --smoke-test for pipeline validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neural_sat_speck.simon.b1_training_runner import (  # noqa: E402
    ManifestValidationError,
    run_b0_b1_training_pipeline,
)

DEFAULT_MANIFEST = REPO_ROOT / "docs" / "SIMON_B1_TRAINING_MANIFEST_PROPOSED.json"


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manifest-driven SIMON32/64 B0->B1 faithful training runner."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Path to JSON training manifest (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run fast smoke-test training on tiny synthetic dataset (16 groups, 2 epochs).",
    )
    parser.add_argument(
        "--execute-full-training",
        action="store_true",
        help="Explicitly authorize full 1,000,000-pair / 15,625-group training under D-037.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Execute dataset generation, key role audit, and cryptographic oracle checks without training.",
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=None,
        help="Destination path for final B1 checkpoint (default: models/simon/simon_b1_*.pt).",
    )
    parser.add_argument(
        "--output-b0-checkpoint",
        type=Path,
        default=None,
        help="Optional destination path for pretrained B0 checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device (default: cpu). Primary comparison requires CPU.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional path to write full JSON execution logs and metrics.",
    )
    parser.add_argument(
        "--smoke-groups",
        type=int,
        default=16,
        help="Number of B1 training groups for smoke-test mode (default: 16).",
    )
    parser.add_argument(
        "--smoke-val-groups",
        type=int,
        default=8,
        help="Number of B1 validation groups for smoke-test mode (default: 8).",
    )
    parser.add_argument(
        "--smoke-epochs",
        type=int,
        default=2,
        help="Number of epochs for smoke-test mode (default: 2).",
    )
    parser.add_argument(
        "--smoke-batch-size",
        type=int,
        default=4,
        help="Batch size (in groups) for smoke-test mode (default: 4).",
    )

    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)

    if not parsed.smoke_test and not parsed.execute_full_training:
        print(
            "ERROR: Full B0/B1 dataset generation/training is guarded under D-037 / AGENTS.md.\n"
            "Full training requires ~21,600s compute budget across 1,000,000 pairs.\n"
            "Specify one of:\n"
            "  --smoke-test             (Fast verification on tiny synthetic dataset)\n"
            "  --smoke-test --preflight-only (Fast preflight check on tiny synthetic dataset)\n"
            "  --execute-full-training  (Explicit confirmation to execute full training/generation)\n",
            file=sys.stderr,
        )
        return 1

    try:
        results = run_b0_b1_training_pipeline(
            manifest_path=parsed.manifest,
            output_checkpoint=parsed.output_checkpoint,
            output_b0_checkpoint=parsed.output_b0_checkpoint,
            smoke_test=parsed.smoke_test,
            smoke_b1_train_groups=parsed.smoke_groups,
            smoke_b1_val_groups=parsed.smoke_val_groups,
            smoke_b0_train_pairs=parsed.smoke_groups * 4,
            smoke_b0_val_pairs=parsed.smoke_val_groups * 4,
            smoke_epochs=parsed.smoke_epochs,
            smoke_batch_size_groups=parsed.smoke_batch_size,
            smoke_batch_size_pairs=parsed.smoke_batch_size * 4,
            device=parsed.device,
            preflight_only=parsed.preflight_only,
            execute_full_training=parsed.execute_full_training,
            verbose=True,
        )
    except ManifestValidationError as e:
        print(f"MANIFEST ERROR: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"DEFECT ABORT: {e}", file=sys.stderr)
        return 3
    except ValueError as e:
        print(f"VALIDATION ERROR: {e}", file=sys.stderr)
        return 4
    except Exception as e:
        print(f"UNHANDLED ERROR: {e}", file=sys.stderr)
        return 5

    if parsed.log_file:
        parsed.log_file.parent.mkdir(parents=True, exist_ok=True)
        # Serialize results to json (ensuring non-serializable objects are converted)
        def _default(obj: Any) -> Any:
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        parsed.log_file.write_text(json.dumps(results, indent=2, default=_default), encoding="utf-8")
        print(f"Wrote structured execution logs to {parsed.log_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
