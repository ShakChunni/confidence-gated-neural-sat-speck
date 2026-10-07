"""Verify and export the recorded depth cohort, without drawing any new keys."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .guard import atomic_json, fixed_grid, sha256_file


def _words(value: Any, count: int) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError("Invalid word count")
    if any(type(w) is not int or not 0 <= w < 65536 for w in value):
        raise ValueError("Words must be unsigned 16-bit integers")
    return tuple(value)


def cipher_functions(cipher: str):
    if cipher == "speck32_64":
        from neural_sat_speck.speck.cipher import encrypt_block
        from neural_sat_speck.speck.key_schedule import expand_key
    elif cipher == "simon32_64":
        from neural_sat_speck.simon.cipher import encrypt_block
        from neural_sat_speck.simon.key_schedule import expand_key
    else:
        raise ValueError("Unknown cipher")
    return encrypt_block, expand_key


def validate_cohort(data: dict[str, Any]) -> None:
    """Replay seed draw and check all target-round known/fresh observations."""
    seed = data.get("seed")
    role = data.get("split_role")
    if (
        type(seed) is not int
        or (seed, role) not in ((999999, "demo"), (501001, "development"))
    ):
        raise ValueError("Only the recorded opened demo family or authorized D-056 development family is eligible")
    if set(data.get("ciphers", {})) != {"speck32_64", "simon32_64"}:
        raise ValueError("Expected exactly two ciphers")
    if seed == 999999:
        for cipher, rounds, diff in (
            ("speck32_64", 7, (64, 0)),
            ("simon32_64", 10, (0, 64)),
        ):
            encrypt, expand = cipher_functions(cipher)
            instances = data["ciphers"][cipher]["instances"]
            if len(instances) != 8:
                raise ValueError("Expected eight cohort keys per cipher")
            rng = random.Random(999999)
            seen_keys = set()
            for index, instance in enumerate(instances):
                expected_key = tuple(rng.randrange(65536) for _ in range(4))
                key = _words(instance["master_key"], 4)
                if (
                    key != expected_key
                    or key in seen_keys
                    or instance.get("instance_index") != index
                ):
                    raise ValueError("Key-draw or cohort order mismatch")
                seen_keys.add(key)
                expected_pts, used = [], set()
                for _ in range(64):
                    while True:
                        p0 = (rng.randrange(65536), rng.randrange(65536))
                        p1 = (p0[0] ^ diff[0], p0[1] ^ diff[1])
                        if p0 not in used and p1 not in used:
                            used.update((p0, p1))
                            expected_pts.append([list(p0), list(p1)])
                            break
                expected_fresh = []
                for _ in range(2):
                    while True:
                        pt = (rng.randrange(65536), rng.randrange(65536))
                        if pt not in used:
                            used.add(pt)
                            expected_fresh.append(list(pt))
                            break
                if (
                    instance["attack_plaintexts"] != expected_pts
                    or instance["verification_plaintexts"] != expected_fresh
                ):
                    raise ValueError("Maximum-pair/fresh draw mismatch")
                rd = instance["rounds_data"][str(rounds)]
                if rd["round_keys"] != list(expand(key, rounds)):
                    raise ValueError("Key schedule mismatch")
                expected_ct = [
                    [list(encrypt(tuple(pt), key, rounds)) for pt in pair]
                    for pair in expected_pts
                ]
                fresh_ct = [list(encrypt(tuple(pt), key, rounds)) for pt in expected_fresh]
                if (
                    rd["attack_ciphertexts"] != expected_ct
                    or rd["verification_ciphertexts"] != fresh_ct
                ):
                    raise ValueError("Known/fresh encryption mismatch")
    else:  # seed == 501001 (D-056 development family)
        for cipher, rounds, diff, c_seed in (
            ("speck32_64", 7, (64, 0), 501001),
            ("simon32_64", 10, (0, 64), 501002),
        ):
            encrypt, expand = cipher_functions(cipher)
            instances = data["ciphers"][cipher]["instances"]
            if len(instances) != 32:
                raise ValueError("Expected 32 cohort keys per cipher for D-056")
            rng = random.Random(c_seed)
            seen_keys = set()
            for index, instance in enumerate(instances):
                expected_key = tuple(rng.randrange(65536) for _ in range(4))
                key = _words(instance["master_key"], 4)
                if (
                    key != expected_key
                    or key in seen_keys
                    or instance.get("instance_index") != index
                ):
                    raise ValueError("Key-draw or cohort order mismatch")
                seen_keys.add(key)
                expected_pts, used = [], set()
                for _ in range(64):
                    while True:
                        p0 = (rng.randrange(65536), rng.randrange(65536))
                        p1 = (p0[0] ^ diff[0], p0[1] ^ diff[1])
                        if p0 not in used and p1 not in used:
                            used.update((p0, p1))
                            expected_pts.append([list(p0), list(p1)])
                            break
                expected_fresh = []
                for _ in range(2):
                    while True:
                        pt = (rng.randrange(65536), rng.randrange(65536))
                        if pt not in used:
                            used.add(pt)
                            expected_fresh.append(list(pt))
                            break
                if (
                    instance["attack_plaintexts"] != expected_pts
                    or instance["verification_plaintexts"] != expected_fresh
                ):
                    raise ValueError("Maximum-pair/fresh draw mismatch")
                rd = instance["rounds_data"][str(rounds)]
                if rd["round_keys"] != list(expand(key, rounds)):
                    raise ValueError("Key schedule mismatch")
                expected_ct = [
                    [list(encrypt(tuple(pt), key, rounds)) for pt in pair]
                    for pair in expected_pts
                ]
                fresh_ct = [list(encrypt(tuple(pt), key, rounds)) for pt in expected_fresh]
                if (
                    rd["attack_ciphertexts"] != expected_ct
                    or rd["verification_ciphertexts"] != fresh_ct
                ):
                    raise ValueError("Known/fresh encryption mismatch")


def case_inputs(data: dict[str, Any], cell: dict[str, Any]) -> tuple[dict, dict]:
    """Separate solver-visible inputs from assigned-key/fresh verification truth."""
    instance = data["ciphers"][cell["cipher"]]["instances"][cell["key_index"]]
    rd = instance["rounds_data"][str(cell["rounds"])]
    pairs = [
        {
            "plaintext0": pts[0],
            "plaintext1": pts[1],
            "ciphertext0": cts[0],
            "ciphertext1": cts[1],
        }
        for pts, cts in zip(
            instance["attack_plaintexts"][: cell["observations"] // 2],
            rd["attack_ciphertexts"][: cell["observations"] // 2],
            strict=True,
        )
    ]
    public = {"instance_id": cell["case_id"], "rounds": cell["rounds"], "pairs": pairs}
    if cell["trailing_keys"]:
        public["trailing_round_keys"] = rd["round_keys"][-cell["trailing_keys"] :]
    truth = {
        "master_key": instance["master_key"],
        "verification_pairs": [
            [pt, ct]
            for pt, ct in zip(
                instance["verification_plaintexts"],
                rd["verification_ciphertexts"],
                strict=True,
            )
        ],
    }
    return public, truth


def verify_key(cipher: str, rounds: int, key: Any, public: dict, truth: dict) -> dict:
    key = _words(key, 4)
    encrypt, _ = cipher_functions(cipher)
    known = all(
        encrypt(_words(pair[f"plaintext{side}"], 2), key, rounds)
        == _words(pair[f"ciphertext{side}"], 2)
        for pair in public["pairs"]
        for side in (0, 1)
    )
    fresh = truth.get("verification_pairs")
    if not isinstance(fresh, list) or len(fresh) != 2:
        raise ValueError("Exactly two fresh verification blocks required")
    used = {tuple(p[f"plaintext{s}"]) for p in public["pairs"] for s in (0, 1)}
    fresh_pts = [_words(pt, 2) for pt, _ in fresh]
    if len(set(fresh_pts)) != 2 or used.intersection(fresh_pts):
        raise ValueError("Verification plaintext overlap")
    return {
        "verified_known": known,
        "verified_fresh": all(
            encrypt(_words(pt, 2), key, rounds) == _words(ct, 2) for pt, ct in fresh
        ),
        "assigned_key_match": key == _words(truth["master_key"], 4),
    }


def export_cases(cohort: Path, output: Path) -> dict:
    """Export all 32 canonical cases; existing files cannot be overwritten."""
    from .formula import build_formula

    cohort_sha = sha256_file(cohort)
    data = json.loads(cohort.read_bytes())
    validate_cohort(data)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite backend bundle: {output}")
    output.mkdir(parents=True)
    catalog = {}
    for cell in fixed_grid():
        ident = cell["case_id"]
        if ident in catalog:
            continue
        public, truth = case_inputs(data, cell)
        formula = build_formula(
            cell["cipher"], public, cell["observations"], cell["trailing_keys"]
        )
        files = {}
        values = {
            "public": public,
            "truth": truth,
            "varmap": {
                "mapping": formula["varmap"],
                "format_ordering": formula["format_ordering"],
                "native_num_vars": formula["num_vars"],
                "pure_cnf_num_vars": formula["pure_cnf_num_vars"],
            },
        }
        for kind, value in values.items():
            path = output / f"{ident}_{kind}.json"
            atomic_json(path, value)
            files[kind] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for kind, value, suffix in (
            ("native", formula["native_dimacs"], ".dimacs"),
            ("cnf", formula["pure_cnf"], ".cnf"),
        ):
            path = output / f"{ident}{suffix}"
            path.write_text(value)
            digest = sha256_file(path)
            path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n")
            files[kind] = {"path": str(path.resolve()), "sha256": digest}
        catalog[ident] = {
            "case_id": ident,
            "files": files,
            "counts": formula["counts"],
            "known_observations": formula["known_observations"],
            "assumption_literals": formula["assumption_literals"],
            "cohort_sha256": cohort_sha,
        }
    if sha256_file(cohort) != cohort_sha:
        raise ValueError("Cohort changed during export; partial bundle is not accepted")
    atomic_json(output / "catalog.json", catalog)
    return catalog


def validate_case_inputs(data: dict, paths: dict[str, Path]) -> None:
    """Reject a rehashed catalog whose public/truth inputs changed the cohort."""
    from .formula import build_formula
    from .guard import canonical_bytes

    catalog = json.loads(paths["catalog"].read_bytes())
    cohort_sha = sha256_file(paths["cohort"])
    checked = set()
    for cell in fixed_grid():
        ident = cell["case_id"]
        if ident in checked:
            continue
        public, truth = case_inputs(data, cell)
        actual_public = json.loads(paths[f"{ident}:public"].read_bytes())
        actual_truth = json.loads(paths[f"{ident}:truth"].read_bytes())
        if canonical_bytes(public) != canonical_bytes(actual_public) or canonical_bytes(
            truth
        ) != canonical_bytes(actual_truth):
            raise ValueError(f"Backend case differs from recorded cohort: {ident}")
        formula = build_formula(
            cell["cipher"], public, cell["observations"], cell["trailing_keys"]
        )
        varmap = {
            "mapping": formula["varmap"],
            "format_ordering": formula["format_ordering"],
            "native_num_vars": formula["num_vars"],
            "pure_cnf_num_vars": formula["pure_cnf_num_vars"],
        }
        if canonical_bytes(varmap) != canonical_bytes(
            json.loads(paths[f"{ident}:varmap"].read_bytes())
        ):
            raise ValueError(f"Variable map differs from canonical encoder: {ident}")
        import hashlib

        for kind, name in (("native", "native_dimacs"), ("cnf", "pure_cnf")):
            if hashlib.sha256(formula[name].encode()).hexdigest() != sha256_file(
                paths[f"{ident}:{kind}"]
            ):
                raise ValueError(
                    f"Formula differs from canonical encoder: {ident}:{kind}"
                )
        expected_meta = {
            "cohort_sha256": cohort_sha,
            "counts": formula["counts"],
            "known_observations": formula["known_observations"],
            "assumption_literals": formula["assumption_literals"],
        }
        if any(
            canonical_bytes(catalog[ident].get(k)) != canonical_bytes(v)
            for k, v in expected_meta.items()
        ):
            raise ValueError(f"Catalog provenance/metadata mismatch: {ident}")
        checked.add(ident)
