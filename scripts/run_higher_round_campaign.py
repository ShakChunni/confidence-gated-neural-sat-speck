#!/usr/bin/env python3
"""Seven- and eight-round SPECK32/64 key-recovery campaign (D-063).

One cell = one key x one arm under a 300 s clock that starts before any model
or table is loaded. Arms:

  R7-B1 project B1 Stage 1 (production)    -> 5-round table Stage 2             -> SAT, 2 words
  R7-G  Gohr's published net6 Stage 1      -> 5-round table Stage 2             -> SAT, 2 words
  R7-C  sparse 6-round table Stage 1 (D-061) -> 5-round table Stage 2           -> SAT, 2 words
  R8-G  Gohr's published net7 Stage 1      -> 6-round table -> 5-round table    -> SAT, 3 words
  R8-C  sparse 7-round table Stage 1       -> 6-round table -> 5-round table    -> SAT, 3 words
  R8-F  full 2^32-entry sampled 7-round table Stage 1 (D-064) -> 6-round -> 5-round -> SAT, 3 words
  *-BF  the same arm with native brute-force completion instead of SAT (D-065)

Stage widths: R7 32 parents x 32 children (as D-060); R8 32 x 8 x 2. Queues are
ordered by summed stage ranks (ties by tuple). Each candidate is solved by a
fresh native CryptoMiniSat 5.14.7 process, matrix capacity 5, 4 observations,
assumptions as permanent unit clauses, killed at the parent deadline. A SAT
model is accepted only if the decoded master key reproduces all 64 known pairs
and 2 independent fresh blocks. Cells run serially; arm order rotates by key.
Neural inference runs on MPS in batches of 512 candidates x 64 pairs = 32,768
rows (MPS returns wrong outputs for rows >= 65,536 in one batch, torch 2.13).
R7-B1 calls the production ranker (_rank_speck_stage1: 1,024 candidates x 64
pairs per MPS call). Torch runs single-threaded for every arm, as in production.
BF arms (D-065) replace each SAT call with tools/bruteforce_completion.c: the
16(4-m) unknown schedule bits are enumerated in 2^22-guess chunks with the cell
deadline checked between chunks; a match must reproduce all 128 known blocks
(inside the C call) and the 2 fresh blocks, exactly as a SAT model must.
R8-F loads the 4 GiB int8 table (log2(p/2^-32) in 1/16-bit units, built on a
GPU by colab_build_speck_full_ddt.py from 2^42 samples) inside the cell clock.

Refuses to overwrite an existing ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from bruteforce_completion import BruteforceLibrary, build_library  # noqa: E402
from gohr_published_net import dec_one, enc, expand_key, load_gohr, to_bits  # noqa: E402
from neural_sat_speck.attacks.classical_ranking import ClassicalDistributionTable, rank_classical_subkeys  # noqa: E402
from neural_sat_speck.attacks.recovery import independent_verification_pairs  # noqa: E402
from neural_sat_speck.backend.formula import build_formula, render_native_dimacs  # noqa: E402
from neural_sat_speck.backend.solver_io import parse_solver_output, solver_arguments  # noqa: E402
from neural_sat_speck.data.attack import AttackInstance, AttackInstanceConfig, AttackPair, generate_attack_instances  # noqa: E402
from neural_sat_speck.followups.ranking_depth_ranking import _configure_torch, _rank_speck_stage1  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CMS = "/opt/homebrew/Cellar/cryptominisat/5.14.7/bin/cryptominisat5"
BUDGET, NP, OBS, CH = 300.0, 64, 4, 512
ASSETS = {
    "b1": REPO / "models/speck32_64_phase11b_prime_stage1_r6_b1_mean.pt",
    "net6": REPO / "runtime/gohr_deep_speck/net6_small.npz",
    "net7": REPO / "runtime/gohr_deep_speck/net7_small.npz",
    "t7": REPO / "docs/paper/evidence/higher_round_probe_20261010/classical_dist_r7_sparse_twopass_mc20_probe.json",
    "t6": REPO / "models/classical_dist_r6_sparse_twopass_mc20_v1.json",
    "t5": REPO / "models/classical_dist_r5_sparse_v2.json",
    "tfull": REPO / "runtime/higher_round_tables/speck_ddt_r7_full.npy",
    "cms": Path(CMS),
    "bf_source": REPO / "tools/bruteforce_completion.c",
    "bf_reference": REPO / "tools/reference_encryption.c",
    "bf_header": REPO / "tools/bruteforce_completion.h",
    "bf_wrapper": REPO / "scripts/bruteforce_completion.py",
}
ARMS = {
    "R7-B1": {"rounds": 7, "stage1": "b1", "later": ["t5"], "widths": [32, 32]},
    "R7-G": {"rounds": 7, "stage1": "net6", "later": ["t5"], "widths": [32, 32]},
    "R7-C": {"rounds": 7, "stage1": "t6", "later": ["t5"], "widths": [32, 32]},
    "R8-G": {"rounds": 8, "stage1": "net7", "later": ["t6", "t5"], "widths": [32, 8, 2]},
    "R8-C": {"rounds": 8, "stage1": "t7", "later": ["t6", "t5"], "widths": [32, 8, 2]},
    "R8-F": {"rounds": 8, "stage1": "tfull", "later": ["t6", "t5"], "widths": [32, 8, 2]},
}
for _a in ("R7-B1", "R7-G", "R7-C", "R8-G", "R8-F"):
    ARMS[_a + "-BF"] = dict(ARMS[_a], completer="bf")
BF_CHUNK = 1 << 22
BF = None  # BruteforceLibrary, built once in main()
CANDS = np.arange(65536, dtype=np.uint32)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def table_top(c0, c1, table, top):
    pairs = tuple(AttackPair((0, 0), (0, 0), (int(a[0]), int(a[1])), (int(b[0]), int(b[1]))) for a, b in zip(c0, c1))
    return [int(v) for v in rank_classical_subkeys(AttackInstance("p", (0, 0, 0, 0), 0, pairs), table).candidate_order[:top]]


def net_order(net, c0, c1):
    scores = np.empty(65536)
    for s in range(0, 65536, CH):
        k = np.repeat(CANDS[s:s + CH], len(c0))
        a = dec_one(np.tile(c0[:, 0], CH), np.tile(c0[:, 1], CH), k)
        b = dec_one(np.tile(c1[:, 0], CH), np.tile(c1[:, 1], CH), k)
        with torch.no_grad():
            z = net.logits(torch.from_numpy(to_bits(*a, *b)).to("mps")).cpu().numpy()
        scores[s:s + CH] = z.reshape(CH, len(c0)).sum(1)
    return np.lexsort((CANDS, -scores))  # descending score, ascending word on ties


def full_order(llr, c0, c1):
    scores = np.empty(65536, dtype=np.int64)
    for s in range(0, 65536, 4096):
        k = np.repeat(CANDS[s:s + 4096], len(c0))
        a = dec_one(np.tile(c0[:, 0], 4096), np.tile(c0[:, 1], 4096), k)
        b = dec_one(np.tile(c1[:, 0], 4096), np.tile(c1[:, 1], 4096), k)
        scores[s:s + 4096] = llr[((a[0] ^ b[0]) << 16) | (a[1] ^ b[1])].astype(np.int32).reshape(4096, len(c0)).sum(1)
    return np.lexsort((CANDS, -scores))  # descending score, ascending word on ties


def dec_pairs(c, k):
    return np.stack(dec_one(c[:, 0], c[:, 1], k), 1)


def run_cell(arm_name: str, x: AttackInstance, fresh, rounds: int) -> dict:
    arm = ARMS[arm_name]
    ks = expand_key(list(x.master_key), rounds)
    m = len(arm["widths"])
    truth = tuple(ks[-m:])  # chronological trailing round keys
    c0 = np.array([p.ciphertext0 for p in x.pairs], dtype=np.uint32)
    c1 = np.array([p.ciphertext1 for p in x.pairs], dtype=np.uint32)
    rec = {"arm": arm_name, "key_id": x.instance_id, "rounds": rounds, "words": m}
    t0 = time.perf_counter()
    if arm["stage1"] == "b1":
        order, _, _ = _rank_speck_stage1(tuple((p.ciphertext0, p.ciphertext1) for p in x.pairs), {"speck_stage1": ASSETS["b1"]})
        parents = [int(v) for v in order[:arm["widths"][0]]]
    elif arm["stage1"].startswith("net"):
        net = load_gohr(str(ASSETS[arm["stage1"]])).to("mps")
        order = net_order(net, c0, c1)
        parents = [int(v) for v in order[:arm["widths"][0]]]
        rec["stage1_true_position"] = int(np.flatnonzero(order == ks[-1])[0]) + 1
    elif arm["stage1"] == "tfull":
        llr = np.load(ASSETS["tfull"])
        rec["stage1_load_s"] = time.perf_counter() - t0
        order = full_order(llr, c0, c1)
        del llr
        parents = [int(v) for v in order[:arm["widths"][0]]]
        rec["stage1_true_position"] = int(np.flatnonzero(order == ks[-1])[0]) + 1
    else:
        parents = table_top(c0, c1, ClassicalDistributionTable.load(str(ASSETS[arm["stage1"]])), arm["widths"][0])
    rec["stage1_s"] = time.perf_counter() - t0
    rec["stage1_true_rank"] = parents.index(ks[-1]) + 1 if ks[-1] in parents else None
    tables = [ClassicalDistributionTable.load(str(ASSETS[t])) for t in arm["later"]]
    # Each entry: (rank_sum, word tuple newest-first, decrypted c0, c1)
    level = [(r, (k,), dec_pairs(c0, k), dec_pairs(c1, k)) for r, k in enumerate(parents)]
    for depth, table in enumerate(tables, start=1):
        nxt = []
        for r, words, d0, d1 in level:
            for rc, k in enumerate(table_top(d0, d1, table, arm["widths"][depth])):
                nxt.append((r + rc, words + (k,), dec_pairs(d0, k), dec_pairs(d1, k)))
            if time.perf_counter() - t0 > BUDGET:
                break
        level = nxt
    queue = sorted((r, tuple(reversed(w))) for r, w, _, _ in level)  # chronological words
    rec["prep_s"] = time.perf_counter() - t0
    rec["queue_len"] = len(queue)
    pos = [i for i, (_, w) in enumerate(queue) if w == truth]
    rec["true_queue_position"] = pos[0] if pos else None
    known = [(p.plaintext0, p.ciphertext0) for p in x.pairs] + [(p.plaintext1, p.ciphertext1) for p in x.pairs]
    rec.update(outcome="queue_exhausted", attempts=0, verification_rejections=0, completer=arm.get("completer", "sat"))
    wrong = []
    if arm.get("completer") == "bf":
        known_pt, known_ct = [pt for pt, _ in known], [ct for _, ct in known]
        space = 1 << (16 * (4 - m))
        for i, (_, words) in enumerate(queue):
            if BUDGET - (time.perf_counter() - t0) <= 0:
                rec["outcome"] = "deadline"
                break
            rec["attempts"] = i + 1
            s, lo, hit = time.perf_counter(), 0, None
            while lo < space:
                if time.perf_counter() - t0 >= BUDGET:
                    rec["outcome"] = "deadline"
                    break
                r = BF.complete("speck", rounds, list(words), known_pt, known_ct, lo, min(space, lo + BF_CHUNK))
                if r.found:
                    kk = expand_key(list(r.master_key), rounds)
                    if all(tuple(int(v) for v in enc(np.uint32(pt[0]), np.uint32(pt[1]), kk)) == tuple(ct) for pt, ct in fresh):
                        hit = list(r.master_key)
                        break
                    rec["verification_rejections"] += 1
                    lo = r.found_guess + 1
                else:
                    lo = min(space, lo + BF_CHUNK)
            dt = time.perf_counter() - s
            if hit is not None:
                rec.update(outcome="recovered", correct_solve_s=dt, recovered_key=hit)
                break
            if rec["outcome"] == "deadline":
                break
            wrong.append(dt)
        queue = []  # completion done; skip the SAT loop
    else:
        public = {"rounds": rounds, "pairs": [{"plaintext0": list(p.plaintext0), "ciphertext0": list(p.ciphertext0),
                                                "plaintext1": list(p.plaintext1), "ciphertext1": list(p.ciphertext1)} for p in x.pairs]}
        base = build_formula("speck32_64", public, OBS, 0, include_cnf=False)  # SAT arms only; BF arms are not charged for it
        word_vars = base["varmap"]["round_keys"][-m:]
    with tempfile.TemporaryDirectory() as tmp:
        for i, (_, words) in enumerate(queue):
            left = BUDGET - (time.perf_counter() - t0)
            if left <= 0:
                rec["outcome"] = "deadline"
                break
            lits = [v if (w >> b) & 1 else -v for w, var in zip(words, word_vars) for b, v in enumerate(var)]
            path = Path(tmp) / "c.cnf"
            path.write_text(render_native_dimacs(base["num_vars"], base["cnf_clauses"], base["xor_clauses"], lits))
            s = time.perf_counter()
            rec["attempts"] = i + 1
            try:
                pr = subprocess.run(solver_arguments("cms_native_gauss", CMS, "", str(path)), capture_output=True, text=True, timeout=left)
            except subprocess.TimeoutExpired:
                rec["outcome"] = "deadline"
                break
            dt = time.perf_counter() - s
            parsed = parse_solver_output("cms", pr.returncode, pr.stdout, required_variables=list(range(1, base["num_vars"] + 1)), num_vars=base["num_vars"])
            if parsed["status"] == "sat":
                key = [sum(int(parsed["assignment"][v]) << b for b, v in enumerate(w)) for w in base["varmap"]["master_key"]]
                kk = expand_key(key, rounds)

                def ok(pt, ct):
                    return tuple(int(v) for v in enc(np.uint32(pt[0]), np.uint32(pt[1]), kk)) == tuple(ct)
                if all(ok(pt, ct) for pt, ct in known) and all(ok(pt, ct) for pt, ct in fresh):
                    rec.update(outcome="recovered", correct_solve_s=dt, recovered_key=key)
                    break
                rec["verification_rejections"] += 1
            elif parsed["status"] == "unsat":
                wrong.append(dt)
            else:
                rec["outcome"] = "solver_" + parsed["status"]
                break
    rec["total_s"] = time.perf_counter() - t0
    rec["wrong_n"] = len(wrong)
    rec["wrong_mean_s"] = float(np.mean(wrong)) if wrong else None
    rec["wrong_median_s"] = float(np.median(wrong)) if wrong else None
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--role", choices=["demo", "fresh"], required=True)
    ap.add_argument("--keys", type=int, required=True)
    ap.add_argument("--first-key", type=int, default=0)
    ap.add_argument("--arms", default="R7-B1,R7-G,R7-C,R8-G,R8-C")
    ap.add_argument("--ledger", type=Path, required=True)
    args = ap.parse_args()
    _configure_torch(torch)
    global BF
    bf_lib = build_library()
    BF = BruteforceLibrary(bf_lib)
    if args.role == "demo" and args.seed != 999999:
        raise SystemExit("demo role requires seed 999999")
    if args.ledger.exists():
        raise SystemExit("Refusing to overwrite existing ledger")
    arms = args.arms.split(",")
    assert all(a in ARMS for a in arms)
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "scripts/run_higher_round_campaign.py", "scripts/gohr_published_net.py",
                            "scripts/bruteforce_completion.py", "tools/bruteforce_completion.c", "tools/bruteforce_completion.h"],
                           capture_output=True, text=True).stdout.strip()
    header = {"type": "header", "seed": args.seed, "role": args.role, "keys": args.keys, "first_key": args.first_key, "arms": arms,
              "git_commit": commit, "runner_dirty": bool(dirty), "script_sha256": sha256(Path(__file__)),
              "assets": {k: {"path": str(v), "sha256": sha256(v)} for k, v in ASSETS.items()},
              "bf_library": {"path": str(bf_lib), "sha256": sha256(bf_lib), "chunk": BF_CHUNK},
              "python": sys.version, "torch": torch.__version__, "platform": platform.platform(),
              "started_local": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    with args.ledger.open("x") as f:
        f.write(json.dumps(header) + "\n")
    data = {}
    for rounds in sorted({ARMS[a]["rounds"] for a in arms}):
        data[rounds] = generate_attack_instances(AttackInstanceConfig(
            instance_count=args.first_key + args.keys, pair_count=NP, rounds=rounds, seed=args.seed, split_role="test")).instances
    for j in range(args.first_key, args.first_key + args.keys):
        order = arms[j % len(arms):] + arms[:j % len(arms)]
        for a in order:
            rounds = ARMS[a]["rounds"]
            x = data[rounds][j]
            fresh = independent_verification_pairs(x.master_key, rounds, 2, args.seed * 1000 + j)
            rec = run_cell(a, x, fresh, rounds)
            rec["key_index"] = j
            rec["arm_position"] = order.index(a)
            print(json.dumps({k: rec[k] for k in ("key_index", "arm", "outcome", "stage1_true_rank", "true_queue_position", "total_s")}), flush=True)
            with args.ledger.open("a") as f:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
