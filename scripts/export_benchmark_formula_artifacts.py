#!/usr/bin/env python3
"""Export reproducible benchmark SAT/CNF formulas, variable maps, and manifests.

Governed by Target B of docs/DEPTH_BACKEND_DESIGN_TASK_20261003.md:
- Exports native-XOR DIMACS and pure-CNF DIMACS formulas for SPECK and SIMON.
- Records exact variable mappings (master key, round keys, state bits).
- Exports assumption vectors and solver regeneration CLI commands.
- Calculates SHA-256 digests for all exported artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from neural_sat_speck.data.attack import (  # noqa: E402
    AttackInstanceConfig,
    generate_attack_instances,
)
from neural_sat_speck.data.seed_policy import CLI_DEMO_SEED  # noqa: E402
from neural_sat_speck.sat.dimacs import export_speck_to_dimacs  # noqa: E402
from neural_sat_speck.sat.pure_cnf import (  # noqa: E402
    convert_native_xor_dimacs_to_pure_cnf,
)
from neural_sat_speck.simon.attack_instance import (  # noqa: E402
    SimonAttackInstanceConfig,
    generate_simon_attack_instances,
)
from neural_sat_speck.simon.key_schedule import (  # noqa: E402
    expand_key as simon_expand_key,
)
from neural_sat_speck.simon.sat_model import SimonSatModel  # noqa: E402
from neural_sat_speck.speck.key_schedule import (  # noqa: E402
    expand_key as speck_expand_key,
)


def export_speck_instance_artifacts(
    out_dir: Path,
    instance_idx: int = 0,
    rounds: int = 7,
    trailing_keys: int = 1,
) -> dict[str, Any]:
    """Export benchmark formula artifacts for one SPECK instance."""
    cfg = AttackInstanceConfig(
        instance_count=8,
        pair_count=64,
        rounds=rounds,
        seed=CLI_DEMO_SEED,
        split_role="validation",
    )
    inst = generate_attack_instances(cfg).instances[instance_idx]
    # 4 observations = first 2 pairs
    pairs = inst.pairs[:2]
    known = [(p.plaintext0, p.ciphertext0) for p in pairs] + [
        (p.plaintext1, p.ciphertext1) for p in pairs
    ]

    r_keys = speck_expand_key(inst.master_key, rounds)
    pinned: dict[int, int] = {}
    for i in range(trailing_keys):
        r_idx = rounds - 1 - i
        pinned[r_idx] = r_keys[r_idx]

    t0 = time.perf_counter()
    model = export_speck_to_dimacs(known, rounds=rounds, pinned_round_keys=pinned)
    build_seconds = time.perf_counter() - t0

    prefix = f"speck32_64_r{rounds}_m{trailing_keys}_inst{instance_idx:02d}"

    # 1. Native XOR DIMACS
    dimacs_file = out_dir / f"{prefix}_native_xor.dimacs"
    dimacs_file.write_text(model.dimacs_text, encoding="utf-8")
    dimacs_sha = hashlib.sha256(model.dimacs_text.encode("utf-8")).hexdigest()

    # 2. Pure CNF DIMACS
    cnf_text, num_cnf_vars = convert_native_xor_dimacs_to_pure_cnf(
        model.dimacs_text, model.num_variables
    )
    cnf_file = out_dir / f"{prefix}_pure.cnf"
    cnf_file.write_text(cnf_text, encoding="utf-8")
    cnf_sha = hashlib.sha256(cnf_text.encode("utf-8")).hexdigest()

    # 3. Variable Map
    varmap = {
        "cipher": "speck32_64",
        "rounds": rounds,
        "master_key": list(inst.master_key),
        "pinned_round_keys": pinned,
        "master_key_bits": [list(word) for word in model.master_key_bits],
        "round_key_bits": [list(word) for word in model.round_keys],
        "native_xor_variables": model.num_variables,
        "pure_cnf_variables": num_cnf_vars,
        "num_cnf_clauses": model.num_cnf_clauses,
        "num_xor_clauses": model.num_xor_clauses,
    }
    varmap_file = out_dir / f"{prefix}_varmap.json"
    varmap_raw = json.dumps(varmap, indent=2)
    varmap_file.write_text(varmap_raw, encoding="utf-8")
    varmap_sha = hashlib.sha256(varmap_raw.encode("utf-8")).hexdigest()

    return {
        "cipher": "speck32_64",
        "rounds": rounds,
        "assumed_keys": trailing_keys,
        "nominal_residual_depth": rounds - trailing_keys,
        "build_wall_seconds": build_seconds,
        "files": {
            "native_xor_dimacs": {"path": dimacs_file.name, "sha256": dimacs_sha},
            "pure_cnf": {"path": cnf_file.name, "sha256": cnf_sha},
            "varmap": {"path": varmap_file.name, "sha256": varmap_sha},
        },
        "regeneration_commands": {
            "cms_native": f"cryptominisat5 --verb 0 --threads 1 --maxmatrixcols 2500 --autodisablegauss 0 {dimacs_file.name}",
            "cms_pure_cnf": f"cryptominisat5 --verb 0 --threads 1 --xor 0 --zero-exit-status {cnf_file.name}",
            "cadical": f"cadical -q {cnf_file.name}",
        },
    }


def export_simon_instance_artifacts(
    out_dir: Path,
    instance_idx: int = 0,
    rounds: int = 10,
    trailing_keys: int = 2,
) -> dict[str, Any]:
    """Export benchmark formula artifacts for one SIMON instance."""
    cfg = SimonAttackInstanceConfig(
        instance_count=8,
        pair_count=64,
        rounds=rounds,
        seed=CLI_DEMO_SEED,
        split_role="demo",
    )
    inst = generate_simon_attack_instances(cfg)[instance_idx]
    pairs = inst.pairs[:2]
    known = [(p.plaintext0, p.ciphertext0) for p in pairs] + [
        (p.plaintext1, p.ciphertext1) for p in pairs
    ]

    r_keys = simon_expand_key(inst.master_key, rounds)
    trailing = [r_keys[rounds - trailing_keys + i] for i in range(trailing_keys)]

    t0 = time.perf_counter()
    model = SimonSatModel.from_known_pairs(known, rounds=rounds)
    assumptions = model.assumptions_for_trailing_round_subkeys(trailing)
    dimacs_text = model.to_dimacs(assumptions=assumptions)
    cnf_text, total_cnf_vars = model.to_dimacs_cnf(assumptions=assumptions)
    build_seconds = time.perf_counter() - t0

    prefix = f"simon32_64_r{rounds}_m{trailing_keys}_inst{instance_idx:02d}"

    # 1. Native XOR DIMACS
    dimacs_file = out_dir / f"{prefix}_native_xor.dimacs"
    dimacs_file.write_text(dimacs_text, encoding="utf-8")
    dimacs_sha = hashlib.sha256(dimacs_text.encode("utf-8")).hexdigest()

    # 2. Pure CNF DIMACS
    cnf_file = out_dir / f"{prefix}_pure.cnf"
    cnf_file.write_text(cnf_text, encoding="utf-8")
    cnf_sha = hashlib.sha256(cnf_text.encode("utf-8")).hexdigest()

    # 3. Variable Map
    varmap = {
        "cipher": "simon32_64",
        "rounds": rounds,
        "master_key": list(inst.master_key),
        "trailing_round_keys": trailing,
        "assumption_literals": list(assumptions),
        "master_key_bits": [list(word) for word in model.master_key_bits],
        "round_key_bits": [list(word) for word in model.round_keys],
        "native_xor_variables": model.variable_count,
        "pure_cnf_variables": total_cnf_vars,
        "num_cnf_clauses": model.cnf_clause_count,
        "num_xor_clauses": model.xor_clause_count,
    }
    varmap_file = out_dir / f"{prefix}_varmap.json"
    varmap_raw = json.dumps(varmap, indent=2)
    varmap_file.write_text(varmap_raw, encoding="utf-8")
    varmap_sha = hashlib.sha256(varmap_raw.encode("utf-8")).hexdigest()

    return {
        "cipher": "simon32_64",
        "rounds": rounds,
        "assumed_keys": trailing_keys,
        "nominal_residual_depth": rounds - trailing_keys,
        "build_wall_seconds": build_seconds,
        "files": {
            "native_xor_dimacs": {"path": dimacs_file.name, "sha256": dimacs_sha},
            "pure_cnf": {"path": cnf_file.name, "sha256": cnf_sha},
            "varmap": {"path": varmap_file.name, "sha256": varmap_sha},
        },
        "regeneration_commands": {
            "cms_native": f"cryptominisat5 --verb 0 --threads 1 --maxmatrixcols 2500 --autodisablegauss 0 {dimacs_file.name}",
            "cms_pure_cnf": f"cryptominisat5 --verb 0 --threads 1 --xor 0 --zero-exit-status {cnf_file.name}",
            "cadical": f"cadical -q {cnf_file.name}",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "benchmark_formulas",
        help="Directory to store exported benchmark artifacts",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting reproducible benchmark formulas to {out_dir}...")
    manifest = {
        "manifest_id": "benchmark_formula_artifacts_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "description": "Cryptographic SAT benchmark instances for SPECK32/64 and SIMON32/64 in native-XOR and pure CNF formats",
        "instances": [],
    }

    # SPECK boundary: R=7, m=1 (d=6) and R=7, m=2 (d=5)
    for m in (1, 2):
        res = export_speck_instance_artifacts(out_dir, instance_idx=0, rounds=7, trailing_keys=m)
        manifest["instances"].append(res)
        print(f"Exported SPECK R=7 m={m} (d={7-m})")

    # SIMON boundary: R=10, m=1 (d=9) and R=10, m=2 (d=8)
    for m in (1, 2):
        res = export_simon_instance_artifacts(out_dir, instance_idx=0, rounds=10, trailing_keys=m)
        manifest["instances"].append(res)
        print(f"Exported SIMON R=10 m={m} (d={10-m})")

    manifest_path = out_dir / "manifest.json"
    raw_manifest = json.dumps(manifest, indent=2)
    manifest_path.write_text(raw_manifest, encoding="utf-8")
    manifest_sha = hashlib.sha256(raw_manifest.encode("utf-8")).hexdigest()
    (out_dir / "manifest.json.sha256").write_text(f"{manifest_sha}  manifest.json\n", encoding="utf-8")

    print(f"Benchmark formula artifact export complete: {len(manifest['instances'])} instances packaged.")


if __name__ == "__main__":
    main()
