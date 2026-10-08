"""Prepare or execute the explicitly authorized D-057 D-056 continuation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "execute"))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-original-ledger-sha256", required=True)
    parser.add_argument("--original-root", type=Path)
    parser.add_argument("--expected-continuation-manifest-sha256")
    args = parser.parse_args()

    source_src = str(args.source_root.resolve() / "src")
    if source_src not in sys.path:
        sys.path.insert(0, source_src)
    from neural_sat_speck.backend.guard import sha256_file
    from neural_sat_speck.followups.ranking_depth_recovery import (
        execute_continuation,
        prepare_continuation,
    )

    if args.phase == "prepare":
        if args.original_root is None or args.expected_continuation_manifest_sha256:
            parser.error("prepare requires --original-root and no continuation hash")
        manifest = prepare_continuation(
            source_root=args.source_root,
            original_root=args.original_root,
            output_root=args.output_root,
            expected_ledger_sha256=args.expected_original_ledger_sha256,
        )
        print(f"prepared_manifest={manifest}")
        print(f"prepared_manifest_sha256={sha256_file(manifest)}")
        return 0

    if args.original_root is not None or not args.expected_continuation_manifest_sha256:
        parser.error("execute requires --expected-continuation-manifest-sha256 only")
    result = execute_continuation(
        output_root=args.output_root,
        expected_manifest_sha256=args.expected_continuation_manifest_sha256,
        expected_ledger_sha256=args.expected_original_ledger_sha256,
    )
    print(f"continuation_status={result['status']}")
    print(f"continuation_records={len(result['records'])}")
    if result["status"] != "COMPLETE":
        print(f"continuation_stop_reason={result.get('stop_reason')}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
