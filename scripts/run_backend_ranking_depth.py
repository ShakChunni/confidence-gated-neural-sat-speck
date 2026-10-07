"""Prepare and run one authorized D-055 mode; full requires a passed pilot audit."""

import argparse
from pathlib import Path

from neural_sat_speck.followups.ranking_depth_control import prepare
from neural_sat_speck.followups.ranking_depth_runner import execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--tf-python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--pilot-audit", type=Path)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    output = args.output_root.absolute()
    if output.exists():
        raise SystemExit("Output exists: no retry, overwrite or resume")
    manifest = prepare(
        args.source_root,
        output / "inputs",
        args.mode,
        args.tf_python,
        args.asset_root,
        args.pilot_audit,
        config_path=args.config,
    )
    result = execute(manifest, output / "execution")
    if result["status"] != "COMPLETE":
        raise SystemExit(f"Campaign stopped: {result['stop_reason']}")


if __name__ == "__main__":
    main()
