"""Independently audit and describe the fixed opened interface comparison."""

import argparse
from collections import Counter
from pathlib import Path
from statistics import median

from neural_sat_speck.backend.guard import atomic_json
from neural_sat_speck.followups.conditioning_control import ARMS, SCIENTIFIC_OUTCOMES
from neural_sat_speck.followups.conditioning_runner import audit_ledger


def analyze(ledger_path: Path, manifest_path: Path) -> dict:
    ledger, manifest = audit_ledger(ledger_path, manifest_path)
    rows = list(ledger["records"].values())
    summary = {
        "role": "finite_opened_cohort_descriptive",
        "mode": manifest["mode"],
        "status": ledger["status"],
        "manifest_sha256": ledger["manifest_sha256"],
        "source_commit": manifest["source_commit"],
        "ciphers": {},
    }
    for cipher in ("speck32_64", "simon32_64"):
        result = {"arms": {}, "pairs": []}
        for arm in ARMS:
            arm_rows = [
                r
                for r in rows
                if r["cell"]["cipher"] == cipher and r["cell"]["arm"] == arm
            ]
            eligible = [r for r in arm_rows if r["outcome"] in SCIENTIFIC_OUTCOMES]
            times = [min(r["parent_wall_seconds"], 300.0) for r in eligible]
            success = [r for r in eligible if r["outcome"] == "recovered"]
            result["arms"][arm] = {
                "scheduled": len(arm_rows),
                "eligible": len(eligible),
                "recovered": len(success),
                "outcomes": dict(Counter(r["outcome"] for r in arm_rows)),
                "median_parent_seconds_eligible": median(times) if times else None,
                "par2_seconds_eligible": sum(
                    min(r["parent_wall_seconds"], 300.0)
                    if r["outcome"] == "recovered"
                    else 600.0
                    for r in eligible
                )
                / len(eligible)
                if eligible
                else None,
                "mean_parent_seconds_success": sum(
                    r["parent_wall_seconds"] for r in success
                )
                / len(success)
                if success
                else None,
            }
        for index in range(1 if manifest["mode"] == "pilot" else 8):
            pair = {
                r["cell"]["arm"]: r
                for r in rows
                if r["cell"]["cipher"] == cipher and r["cell"]["key_index"] == index
            }
            if set(pair) != set(ARMS):
                raise ValueError("Missing paired row")
            eligible = all(r["outcome"] in SCIENTIFIC_OUTCOMES for r in pair.values())
            result["pairs"].append(
                {
                    "key_index": index,
                    "eligible": eligible,
                    "outcomes": {a: r["outcome"] for a, r in pair.items()},
                    "restricted_parent_seconds": {
                        a: min(r["parent_wall_seconds"], 300.0) for a, r in pair.items()
                    }
                    if eligible
                    else None,
                }
            )
        eligible_pairs = [p for p in result["pairs"] if p["eligible"]]
        result["discordances"] = {
            arm + "_only": sum(
                p["outcomes"][arm] == "recovered"
                and p["outcomes"][other] != "recovered"
                for p in eligible_pairs
            )
            for arm, other in (ARMS, ARMS[::-1])
        }
        result["twofold_cost_flag"] = {
            arm: sum(
                p["restricted_parent_seconds"][arm] * 2
                <= p["restricted_parent_seconds"][other]
                for p in eligible_pairs
            )
            >= 4
            for arm, other in (ARMS, ARMS[::-1])
        }
        summary["ciphers"][cipher] = result
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Analysis output exists; use a versioned path")
    summary = analyze(args.ledger.absolute(), args.manifest.absolute())
    atomic_json(args.output.absolute(), summary)
    print(summary)


if __name__ == "__main__":
    main()
