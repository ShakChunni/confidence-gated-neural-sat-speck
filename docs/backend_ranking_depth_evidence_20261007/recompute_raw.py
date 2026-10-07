"""Read-only independent arithmetic, cipher and detached-hash checks for D-055.

Uses the frozen scalar ciphers, not the experiment analyzer or its aggregate
tables. Never imports a ranker or invokes a solver. Supply a new output path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    assert (
        file_hash(path) == path.with_name(path.name + ".sha256").read_text().split()[0]
    )
    return json.loads(path.read_bytes())


def main():
    if not __debug__:
        raise RuntimeError(
            "Run without -O/PYTHONOPTIMIZE: audit assertions are required"
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    ledger_path = args.root / "execution/ledger.json"
    manifest_path = args.root / "inputs/manifest.json"
    ledger, manifest = read_json(ledger_path), read_json(manifest_path)
    analysis = read_json(args.analysis)
    assert ledger["status"] == "COMPLETE" and len(manifest["cells"]) == 64
    ids = [c["cell_id"] for c in manifest["cells"]]
    assert len(set(ids)) == 64
    assert set(ids) == set(ledger["starts"]) == set(ledger["records"])
    assert ids == ledger["execution_order"]
    hash_count, fragments = 0, []
    for sidecar in sorted(args.root.rglob("*.sha256")):
        target = sidecar.with_name(sidecar.name.removesuffix(".sha256"))
        if not target.is_file() or file_hash(target) != sidecar.read_text().split()[0]:
            # A killed atomic writer may leave uncommitted terminal bytes.
            # They cannot establish a recovered key or a completed attempt.
            owning = [
                r
                for r in ledger["records"].values()
                if Path(r["directory"]) in sidecar.parents
            ]
            assert len(owning) == 1 and owning[0]["outcome"] == "global_deadline"
            fragments.append(str(sidecar.relative_to(args.root)))
        else:
            hash_count += 1
    groups, cells, counts = {}, [], Counter()
    known_checks = fresh_checks = 0
    sample_count, max_rss = 0, 0
    for cell in manifest["cells"]:
        row = ledger["records"][cell["cell_id"]]
        assert row["cell"] == cell
        directory = Path(row["directory"])
        assert ledger["starts"][cell["cell_id"]]["invocation_id"] == directory.name
        public = read_json(Path(manifest["cases"][cell["case_id"]]["public"]["path"]))
        truth = read_json(Path(manifest["cases"][cell["case_id"]]["truth"]["path"]))
        if cell["cipher"] == "speck32_64":
            from neural_sat_speck.speck.cipher import encrypt_block
            from neural_sat_speck.speck.key_schedule import expand_key
        else:
            assert cell["cipher"] == "simon32_64"
            from neural_sat_speck.simon.cipher import encrypt_block
            from neural_sat_speck.simon.key_schedule import expand_key
        true_tuple = list(
            expand_key(tuple(truth["master_key"]), cell["rounds"])[
                -cell["trailing_keys"] :
            ]
        )
        recovered = row["outcome"] == "recovered"
        t = row["parent_wall_seconds"]
        assert isinstance(t, (int, float)) and t > 0
        if recovered:
            assert t <= 300
            key = tuple(row["worker_result"]["recovered_key"])
            assert key == tuple(truth["master_key"])
            used = set()
            for pair in public["pairs"]:
                for s in (0, 1):
                    pt = tuple(pair[f"plaintext{s}"])
                    used.add(pt)
                    assert encrypt_block(pt, key, cell["rounds"]) == tuple(
                        pair[f"ciphertext{s}"]
                    )
                    known_checks += 1
            fresh = truth["verification_pairs"]
            assert len(fresh) == 2 and len({tuple(x[0]) for x in fresh}) == 2
            for pt, ct in fresh:
                assert tuple(pt) not in used
                assert encrypt_block(tuple(pt), key, cell["rounds"]) == tuple(ct)
                fresh_checks += 1
        for p in directory.glob("attempts/[0-9][0-9][0-9][0-9]/attempt.json"):
            try:
                attempt = read_json(p)
            except (AssertionError, FileNotFoundError):
                assert row["outcome"] == "global_deadline"
                continue
            counts[attempt["status"]] += 1
            assert not (
                attempt["candidate"] == true_tuple and attempt["status"] == "unsat"
            )
        for sample in row["environment_samples"]:
            sample_count += 1
            max_rss = max(max_rss, sample.get("memory", {}).get("group_rss_bytes", 0))
        group = (cell["cipher"], cell["trailing_keys"], cell["profile"])
        measured = {
            "key_index": cell["key_index"],
            "recovered": int(recovered),
            "par2": t if recovered else 600.0,
            "outcome": row["outcome"],
        }
        groups.setdefault(group, []).append(measured)
        cells.append(measured)
    assert max_rss <= 2.5 * 2**30
    summaries, interactions = [], []
    for (cipher, depth, profile), rows in sorted(groups.items()):
        assert len(rows) == 8 and {r["key_index"] for r in rows} == set(range(8))
        derived = {
            "cipher": cipher,
            "trailing_keys": depth,
            "profile": profile,
            "recovered": sum(r["recovered"] for r in rows),
            "mean_par2_seconds": statistics.mean(r["par2"] for r in rows),
        }
        reference = next(
            x
            for x in analysis["summaries"]
            if (x["cipher"], x["trailing_keys"], x["profile"])
            == (cipher, depth, profile)
        )
        assert derived["recovered"] == reference["recovered"]
        assert abs(derived["mean_par2_seconds"] - reference["mean_par2_seconds"]) < 1e-9
        summaries.append(derived)
    for cipher in ("speck32_64", "simon32_64"):
        effects = {}
        for profile in ("cms_native_gauss", "cms_native_no_gauss"):
            one = {x["key_index"]: x for x in groups[cipher, 1, profile]}
            two = {x["key_index"]: x for x in groups[cipher, 2, profile]}
            effects[profile] = {
                "recovery_difference": sum(
                    two[i]["recovered"] - one[i]["recovered"] for i in range(8)
                )
                / 8,
                "par2_difference_seconds": sum(
                    two[i]["par2"] - one[i]["par2"] for i in range(8)
                )
                / 8,
            }
        g, x = effects["cms_native_gauss"], effects["cms_native_no_gauss"]
        derived = {
            "cipher": cipher,
            "recovery_interaction_G_minus_X": g["recovery_difference"]
            - x["recovery_difference"],
            "par2_interaction_G_minus_X_seconds": g["par2_difference_seconds"]
            - x["par2_difference_seconds"],
            "effects": effects,
        }
        reference = next(z for z in analysis["interactions"] if z["cipher"] == cipher)
        assert (
            abs(
                derived["recovery_interaction_G_minus_X"]
                - reference["recovery_interaction_G_minus_X"]
            )
            < 1e-9
        )
        assert (
            abs(
                derived["par2_interaction_G_minus_X_seconds"]
                - reference["par2_interaction_G_minus_X_seconds"]
            )
            < 1e-9
        )
        interactions.append(derived)
    output = {
        "passed": True,
        "schema": "d055_raw_recompute_v1",
        "ledger_sha256": file_hash(ledger_path),
        "manifest_sha256": file_hash(manifest_path),
        "valid_detached_hashes": hash_count,
        "uncommitted_deadline_fragments": fragments,
        "recoveries": sum(c["recovered"] for c in cells),
        "known_block_checks": known_checks,
        "fresh_block_checks": fresh_checks,
        "completed_attempt_statuses": dict(counts),
        "resource_samples": sample_count,
        "sampled_max_group_rss_bytes": max_rss,
        "summaries": summaries,
        "interactions": interactions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, sort_keys=True, indent=2) + "\n")
    args.output.with_name(args.output.name + ".sha256").write_text(
        file_hash(args.output) + "  " + args.output.name + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in output.items() if k not in ("summaries", "interactions")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
