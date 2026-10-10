"""Does the production 7-round SPECK Stage-1 ranker (B1, batch 1024) give identical scores on CPU and MPS?"""
import numpy as np, torch
from neural_sat_speck.attacks.group_ranking import rank_pair_group_subkeys
from neural_sat_speck.followups.ranking import _load_pair_group_ranker
from neural_sat_speck.neural.group_model import PairGroupRanker
from neural_sat_speck.data.attack import AttackInstanceConfig, generate_attack_instances
M = "/Users/ashfaq/Dev/comp-sci-project/models/speck32_64_phase11b_prime_stage1_r6_b1_mean.pt"
inst = generate_attack_instances(AttackInstanceConfig(instance_count=3, pair_count=64, rounds=7, seed=999999, split_role="test")).instances
for dev in ("cpu", "mps"):
    pass
res = {}
for dev in ("cpu", "mps"):
    model = _load_pair_group_ranker(torch, PairGroupRanker, M, torch.device(dev)) if _load_pair_group_ranker.__code__.co_argcount == 4 else _load_pair_group_ranker(M, torch.device(dev))
    res[dev] = [rank_pair_group_subkeys(x, model, candidate_batch_size=1024, device=torch.device(dev)) for x in inst]
for i, x in enumerate(inst):
    a, b = np.array(res["cpu"][i].candidate_scores), np.array(res["mps"][i].candidate_scores)
    ra = list(res["cpu"][i].candidate_order).index(x.final_round_subkey) + 1
    rb = list(res["mps"][i].candidate_order).index(x.final_round_subkey) + 1
    print(f"key {i+1}: maxabsdiff={np.abs(a-b).max():.3g}  true rank cpu={ra} mps={rb}  top32 overlap={len(set(res['cpu'][i].candidate_order[:32]) & set(res['mps'][i].candidate_order[:32]))}/32")
