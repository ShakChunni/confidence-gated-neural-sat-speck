"""Audit D-058 samples and export existing-policy deadline summaries."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from collections import defaultdict
from pathlib import Path

from neural_sat_speck.analysis.encryption_calibration import (
    sha256,
    summarize_samples,
    write_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


def read_checked(path: Path, *, require_sidecar: bool = True) -> dict:
    side = path.with_name(path.name + ".sha256")
    if (require_sidecar and not side.exists()) or (
        side.exists() and side.read_text().split()[0] != sha256(path)
    ):
        raise ValueError(f"sidecar missing/mismatched: {path}")
    return json.loads(path.read_text())


def deadline_row(
    study: str,
    method: str,
    rows: list[dict],
    n: int,
    success: str,
    wall: str,
    identity: str,
) -> dict:
    if len(rows) != n or len({r[identity] for r in rows}) != n:
        raise ValueError(f"incomplete/duplicate cohort: {study}/{method}")
    for row in rows:
        if type(row[success]) is not bool or type(row[wall]) not in (int, float):
            raise ValueError("invalid success/time types")
        if not math.isfinite(row[wall]) or row[wall] <= 0:
            raise ValueError("invalid recorded online time")
        if row[success] and row[wall] > 300:
            raise ValueError("late success cannot enter cumulative recovery")
    return {
        "study": study,
        "method": method,
        "n": n,
        **{
            f"recovered_by_{t}s": sum(r[success] and r[wall] <= t for r in rows)
            for t in (100, 200, 300)
        },
    }


def cumulative_recovery() -> dict:
    source_names = [
        "results/phase11_prospective/seven_round_prospective_results.json",
        "results/simon_confirmation_replacement/simon_confirmation_comparators.json",
        "docs/backend_ranking_depth_evidence_20261009/analysis_v2.json",
    ]
    speck, simon, depth = [
        read_checked(ROOT / name, require_sidecar=i != 0)
        for i, name in enumerate(source_names)
    ]
    expected_hashes = (
        "b8d06d276d18a5d50df96cbe5d599c830fa6c46c2a35083dc35265a6772ff606",
        "8cc9da7fbdf5207d94e0b108452819244a9dd4aee3dd28de14419af05e397b61",
        "7eb12c76ded7ab078109ef0d387440152d1281175054a55dafc38b5b3644ade8",
    )
    for name, expected_hash in zip(source_names, expected_hashes, strict=True):
        if sha256(ROOT / name) != expected_hash:
            raise ValueError(f"reviewed source identity changed: {name}")
    if speck["metadata"]["protocol_status"] != "FROZEN_AUTHORIZED":
        raise ValueError("wrong frozen SPECK evidence")
    expected = {
        "hierarchical_neural_screened": 110,
        "hierarchical_neural_classical": 111,
        "hierarchical_neural_full": 0,
        "random_sat": 0,
        "sat_only_4obs": 0,
        "sat_only_all_observations": 0,
    }
    if set(speck["records_by_method"]) != set(expected):
        raise ValueError("incomplete SPECK method matrix")
    result = []
    for method, rows in speck["records_by_method"].items():
        row = deadline_row(
            "D031_SPECK_R7",
            method,
            rows,
            128,
            "counts_as_verified_recovery",
            "online_wall_seconds",
            "instance_id",
        )
        if row["recovered_by_300s"] != expected[method]:
            raise ValueError("frozen SPECK count drift")
        result.append(row)
    groups = defaultdict(list)
    for row in simon["records"]:
        row = dict(row)
        if type(row["verified_on_fresh_pairs"]) is not bool:
            raise ValueError("invalid fresh-verification flag")
        row["success"] = row["outcome"] == "verified_recovery"
        if row["success"] != row["verified_on_fresh_pairs"]:
            raise ValueError("SIMON recovery/verification mismatch")
        groups[row["method"]].append(row)
    if len(groups) != 9 or len(simon["records"]) != 576:
        raise ValueError("incomplete SIMON method matrix")
    for method, rows in sorted(groups.items()):
        result.append(
            deadline_row(
                "SIMON_REPLACEMENT_R10",
                method,
                rows,
                64,
                "success",
                "complete_online_wall_seconds",
                "instance_id",
            )
        )
    groups = defaultdict(list)
    for row in depth["cells"]:
        method = f"{row['cipher']}:m{row['trailing_keys']}:{row['profile']}"
        groups[method].append(row)
    if len(groups) != 8 or len(depth["cells"]) != 256:
        raise ValueError("incomplete N32 conditioning matrix")
    for method, rows in sorted(groups.items()):
        result.append(
            deadline_row(
                "D056_DEVELOPMENT",
                method,
                rows,
                32,
                "assigned_verified_recovery",
                "parent_wall_seconds",
                "case_id",
            )
        )
    return {
        "interpretation": "recorded policies at earlier times; not shorter-cutoff reruns",
        "input_sha256": {name: sha256(ROOT / name) for name in source_names},
        "rows": result,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(stream.getvalue())
    path.with_name(path.name + ".sha256").write_text(f"{sha256(path)}  {path.name}\n")


def offline_assets() -> dict:
    speck_path = ROOT / "docs/seven_round_protocol_manifest.json"
    simon_path = (
        ROOT / "docs/SIMON_UNSEEN_CONFIRMATION_REPLACEMENT_MANIFEST_FROZEN.json"
    )
    speck = read_checked(speck_path, require_sidecar=False)
    simon = read_checked(simon_path, require_sidecar=False)
    resources, hashes = (
        simon["development_resource_locations"],
        simon["provenance_hashes"],
    )
    items = []
    for name in ("stage1_model", "stage2_neural_model", "stage2_classical_table"):
        asset = speck["artifacts"][name]
        items.append(("SPECK " + name, ROOT / asset["path"], asset["sha256"]))
    for name, hash_name in (
        ("ddt_7r", "simon_predicted_ddt_7r_bin_sha256"),
        ("b1_checkpoint", "b1_checkpoint_sha256"),
    ):
        path = Path(resources[name])
        items.append(
            (
                "SIMON " + name,
                path if path.is_absolute() else ROOT / path,
                hashes[hash_name],
            )
        )
    for rounds in (8, 9):
        path = Path(resources["bao_vv_directory"]) / f"ND_VV_Simon32_{rounds}R.h5"
        items.append(
            (f"SIMON Bao VV{rounds}", path, hashes[f"vv{rounds}_checkpoint_sha256"])
        )
    result = []
    for label, path, pinned_hash in items:
        stat = path.stat()
        result.append(
            {
                "label": label,
                "path": str(path.relative_to(ROOT)),
                "logical_bytes": stat.st_size,
                "allocated_bytes": stat.st_blocks * 512,
                "pinned_sha256": pinned_hash,
            }
        )
    return {
        "scope": "storage only; hashes from frozen manifests, not rehashed large tables in calibration",
        "manifest_sha256": {
            str(p.relative_to(ROOT)): sha256(p) for p in (speck_path, simon_path)
        },
        "assets": result,
        "offline_wall_time": "not measured by this calibration; consult original generation/training logs; missing costs are not zero",
        "data_per_attack": {
            "chosen_plaintext_block_queries": 128,
            "fresh_verification_block_queries": 2,
            "total_queries_if_verified": 130,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    folder = args.evidence.resolve()
    identity = read_checked(folder / "execution_identity.json")
    raw = read_checked(folder / "samples.json")
    correctness = read_checked(folder / "correctness.json")
    saved = read_checked(folder / "summary.json")
    if raw["status"] != "COMPLETE" or raw["identity_sha256"] != sha256(
        folder / "execution_identity.json"
    ):
        raise ValueError("incomplete or mismatched run")
    if identity["correctness_sha256"] != sha256(folder / "correctness.json"):
        raise ValueError("wrong correctness receipt")
    if saved["raw_sha256"] != sha256(folder / "samples.json"):
        raise ValueError("summary binds different samples")
    recomputed = summarize_samples(
        raw["samples"],
        identity["manifest"],
        correctness["expected_digests"],
        identity_sha256=raw["identity_sha256"],
    )
    if saved["profiles"] != recomputed:
        raise ValueError("summary does not reproduce exactly")
    cumulative = cumulative_recovery()
    write_evidence(folder / "cumulative_recovery.json", cumulative)
    write_csv(folder / "cumulative_recovery.csv", cumulative["rows"])
    write_evidence(folder / "offline_assets.json", offline_assets())
    receipt = {
        "status": "PASS",
        "sample_count": len(raw["samples"]),
        "summary_reproduced_exactly": True,
        "source_sha256": sha256(Path(__file__)),
        "scope": "derived sample accounting and existing-policy timing; prior recovery audits retained",
    }
    write_evidence(folder / "audit.json", receipt)
    print(json.dumps({"audit": receipt, "profiles": recomputed}, indent=2))


if __name__ == "__main__":
    main()
