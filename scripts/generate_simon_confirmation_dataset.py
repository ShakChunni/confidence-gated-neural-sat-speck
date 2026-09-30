#!/usr/bin/env python3
"""Dedicated authorized dataset generator for SIMON32/64 R=10 unseen confirmation.

Fails closed unless --authorized is passed AND confirmation_dataset_generation_authorized
is True in the manifest.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from neural_sat_speck.simon.confirmation_guard import (  # noqa: E402
    ConfirmationPreflightError,
    generate_confirmation_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate SIMON32/64 confirmation dataset.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to confirmation manifest JSON.")
    parser.add_argument("--output-dataset", type=Path, required=True, help="Path to write dataset JSON.")
    parser.add_argument("--output-certificate", type=Path, required=True, help="Path to write certificate JSON.")
    parser.add_argument("--authorized", action="store_true", help="Explicit authorization token.")
    args = parser.parse_args()

    try:
        ds_path, cert_path = generate_confirmation_dataset(
            manifest_raw=args.manifest,
            output_dataset_path=args.output_dataset,
            output_cert_path=args.output_certificate,
            authorization_token=args.authorized,
        )
        print(f"[+] Confirmation dataset successfully generated at {ds_path}")
        print(f"[+] Disjoint-family certificate generated at {cert_path}")
        return 0
    except ConfirmationPreflightError as exc:
        print(f"[!] AUTHORIZATION/PREFLIGHT REJECTION: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[!] GENERATION ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
