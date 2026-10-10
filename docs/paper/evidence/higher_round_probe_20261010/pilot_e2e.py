"""8-round SPECK end-to-end pilot (demo seed 999999, engineering only, not frozen).

One cell = one key under a 300 s clock that starts before ranking:
  Stage 1 (neural: Gohr net7, all 64 pairs | classical: sparse 7-round table) -> top P parents
  Stage 2: 6-round sparse table after 1-round decryption -> top C children per parent
  Stage 3: 5-round sparse table after 2-round decryption -> top G grandchildren
  Queue sorted by summed stage ranks; each candidate (k6, k7, k8) solved by a fresh
  native CryptoMiniSat 5.14.7 process (matrix capacity 5, 4 observations, permanent
  unit clauses), parent deadline enforced; a SAT model is accepted only if the
  decoded master key reproduces all 64 known pairs and 2 fresh blocks.
"""

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from gohr_torch import dec_one, enc, expand_key, load_gohr, to_bits
from neural_sat_speck.attacks.classical_ranking import ClassicalDistributionTable, rank_classical_subkeys
from neural_sat_speck.backend.formula import build_formula, render_native_dimacs
from neural_sat_speck.backend.solver_io import parse_solver_output, solver_arguments
from neural_sat_speck.data.attack import AttackInstance, AttackInstanceConfig, AttackPair, generate_attack_instances

REPO = Path("/Users/ashfaq/Dev/comp-sci-project")
CMS = "/opt/homebrew/Cellar/cryptominisat/5.14.7/bin/cryptominisat5"
R, NP, P, C, G, BUDGET = 8, 64, 16, 8, 2, 300.0
arm, key_ids, out = sys.argv[1], [int(k) for k in sys.argv[2].split(",")], Path(sys.argv[3])
profile = sys.argv[4] if len(sys.argv) > 4 else "cms_native_gauss"

insts = generate_attack_instances(AttackInstanceConfig(
    instance_count=max(key_ids), pair_count=NP, rounds=R, seed=999999, split_role="test")).instances
cands = np.arange(65536, dtype=np.uint32)


def ranked_by_table(c0, c1, table, top):
    pairs = tuple(AttackPair((0, 0), (0, 0), (int(a[0]), int(a[1])), (int(b[0]), int(b[1]))) for a, b in zip(c0, c1))
    r = rank_classical_subkeys(AttackInstance("p", (0, 0, 0, 0), 0, pairs), table)
    return list(r.candidate_order[:top])


def ranked_by_net(net, dev, c0, c1, top):
    scores = np.empty(65536)
    for s in range(0, 65536, 4096):
        k = np.repeat(cands[s:s + 4096], len(c0))
        a = dec_one(np.tile(c0[:, 0], 4096), np.tile(c0[:, 1], 4096), k)
        b = dec_one(np.tile(c1[:, 0], 4096), np.tile(c1[:, 1], 4096), k)
        with torch.no_grad():
            z = net.logits(torch.from_numpy(to_bits(*a, *b)).to(dev)).cpu().numpy()
        scores[s:s + 4096] = z.reshape(4096, len(c0)).sum(1)
    order = np.lexsort((cands, -scores))  # descending score, ascending word on ties
    return [int(v) for v in order[:top]]


def dec_pairs(c, k):
    return np.stack(dec_one(c[:, 0], c[:, 1], k), 1)


for kid in key_ids:
    x = insts[kid - 1]
    ks = expand_key(list(x.master_key), R)
    c0 = np.array([p.ciphertext0 for p in x.pairs], dtype=np.uint32)
    c1 = np.array([p.ciphertext1 for p in x.pairs], dtype=np.uint32)
    rng = np.random.default_rng(10_000 + kid)
    fresh_p = rng.integers(0, 65536, (2, 2), dtype=np.uint32)
    fresh_c = np.stack(enc(fresh_p[:, 0], fresh_p[:, 1], ks), 1)
    rec = {"arm": arm, "key": kid, "profile": profile, "P": P, "C": C, "G": G}
    t0 = time.perf_counter()
    if arm == "neural":
        dev = torch.device("mps")
        net = load_gohr(str(REPO / "runtime/gohr_deep_speck/net7_small.npz")).to(dev)
        parents = ranked_by_net(net, dev, c0, c1, P)
    else:
        t7 = ClassicalDistributionTable.load(str(REPO / "docs/paper/evidence/higher_round_probe_20261010/classical_dist_r7_sparse_twopass_mc20_probe.json"))
        parents = ranked_by_table(c0, c1, t7, P)
    rec["stage1_s"] = time.perf_counter() - t0
    rec["parent_rank_of_true"] = parents.index(ks[-1]) + 1 if ks[-1] in parents else None
    t6 = ClassicalDistributionTable.load(str(REPO / "models/classical_dist_r6_sparse_twopass_mc20_v1.json"))
    t5 = ClassicalDistributionTable.load(str(REPO / "models/classical_dist_r5_sparse_v2.json"))
    triples = []
    for rp, k8 in enumerate(parents):
        d0, d1 = dec_pairs(c0, k8), dec_pairs(c1, k8)
        for rc, k7 in enumerate(ranked_by_table(d0, d1, t6, C)):
            e0, e1 = dec_pairs(d0, k7), dec_pairs(d1, k7)
            for rg, k6 in enumerate(ranked_by_table(e0, e1, t5, G)):
                triples.append((rp + rc + rg, rp, rc, rg, k6, k7, k8))
        if time.perf_counter() - t0 > BUDGET:
            break
    triples.sort()
    rec["prep_s"] = time.perf_counter() - t0
    truth = (ks[-3], ks[-2], ks[-1])
    pos = [i for i, t in enumerate(triples) if t[4:] == truth]
    rec["true_queue_position"] = pos[0] if pos else None
    rec["queue_len"] = len(triples)
    public = {"rounds": R, "pairs": [{"plaintext0": list(map(int, p.plaintext0)), "ciphertext0": list(map(int, p.ciphertext0)),
                                       "plaintext1": list(map(int, p.plaintext1)), "ciphertext1": list(map(int, p.ciphertext1))} for p in x.pairs]}
    base = build_formula("speck32_64", public, 4, 0, include_cnf=False)
    words = base["varmap"]["round_keys"][-3:]
    rec.update(outcome="queue_exhausted", attempts=0, wrong_s=[], correct_s=None)
    with tempfile.TemporaryDirectory() as tmp:
        for i, t in enumerate(triples):
            left = BUDGET - (time.perf_counter() - t0)
            if left <= 0:
                rec["outcome"] = "deadline"
                break
            lits = [v if (w >> b) & 1 else -v for w, var in zip(t[4:], words) for b, v in enumerate(var)]
            path = Path(tmp) / f"{i}.cnf"
            path.write_text(render_native_dimacs(base["num_vars"], base["cnf_clauses"], base["xor_clauses"], lits))
            s = time.perf_counter()
            try:
                pr = subprocess.run(solver_arguments(profile, CMS, "", str(path)), capture_output=True, text=True, timeout=left)
            except subprocess.TimeoutExpired:
                rec["outcome"] = "deadline"
                rec["attempts"] = i + 1
                break
            dt = time.perf_counter() - s
            rec["attempts"] = i + 1
            parsed = parse_solver_output("cms", pr.returncode, pr.stdout, required_variables=list(range(1, base["num_vars"] + 1)), num_vars=base["num_vars"])
            if parsed["status"] == "sat":
                key = [sum(int(parsed["assignment"][v]) << b for b, v in enumerate(w)) for w in base["varmap"]["master_key"]]
                kk = expand_key(key, R)
                ok_known = np.array_equal(np.stack(enc(np.array([p.plaintext0[0] for p in x.pairs], dtype=np.uint32), np.array([p.plaintext0[1] for p in x.pairs], dtype=np.uint32), kk), 1), c0) and \
                    np.array_equal(np.stack(enc(np.array([p.plaintext1[0] for p in x.pairs], dtype=np.uint32), np.array([p.plaintext1[1] for p in x.pairs], dtype=np.uint32), kk), 1), c1)
                ok_fresh = np.array_equal(np.stack(enc(fresh_p[:, 0], fresh_p[:, 1], kk), 1), fresh_c)
                if ok_known and ok_fresh:
                    rec.update(outcome="recovered", correct_s=dt, matches_truth=list(key) == list(x.master_key))
                    break
                rec.setdefault("verification_rejections", 0)
                rec["verification_rejections"] += 1
            elif parsed["status"] == "unsat":
                rec["wrong_s"].append(dt)
            else:
                rec["outcome"] = "solver_" + parsed["status"]
                break
    rec["total_s"] = time.perf_counter() - t0
    w = rec.pop("wrong_s")
    rec["wrong_n"], rec["wrong_median_s"] = len(w), (float(np.median(w)) if w else None)
    rec["wrong_mean_s"] = float(np.mean(w)) if w else None
    print(json.dumps(rec), flush=True)
    with out.open("a") as f:
        f.write(json.dumps(rec) + "\n")
