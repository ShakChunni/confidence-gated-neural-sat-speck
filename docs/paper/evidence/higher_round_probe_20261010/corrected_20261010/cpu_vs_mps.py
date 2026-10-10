import numpy as np, torch
from gohr_torch import dec_one, load_gohr, to_bits, make_train_data
from neural_sat_speck.data.attack import AttackInstanceConfig, generate_attack_instances
G = "/Users/ashfaq/Dev/comp-sci-project/runtime/gohr_deep_speck/net7_small.npz"
cpu, mps = load_gohr(G), load_gohr(G).to("mps")
x = generate_attack_instances(AttackInstanceConfig(instance_count=2, pair_count=64, rounds=8, seed=999999, split_role="test")).instances[1]
c0 = np.array([p.ciphertext0 for p in x.pairs], dtype=np.uint32); c1 = np.array([p.ciphertext1 for p in x.pairs], dtype=np.uint32)
for k in [x.final_round_subkey, x.final_round_subkey ^ 0x400, 0x0000, 0x0400, 0x0800, 0x0C00, 0x1234, 0x1634]:
    kk = np.full(64, k, dtype=np.uint32)
    X = torch.from_numpy(to_bits(*dec_one(c0[:,0], c0[:,1], kk), *dec_one(c1[:,0], c1[:,1], kk)))
    with torch.no_grad():
        a = cpu.logits(X).numpy(); b = mps.logits(X.to("mps")).cpu().numpy()
    print(f"k={k:04x} cpu_sum={a.sum():9.2f} mps_sum={b.sum():9.2f} maxabsdiff={np.abs(a-b).max():.3g}")
# batch-size effect on MPS
X, Y = make_train_data(262144, 7, np.random.default_rng(1))
with torch.no_grad():
    a = cpu.logits(torch.from_numpy(X)).numpy(); b = mps.logits(torch.from_numpy(X).to("mps")).cpu().numpy()
print("random 7r batch 262144: cpu acc", ((a>0)==Y).mean(), "mps acc", ((b>0)==Y).mean(), "maxabsdiff", np.abs(a-b).max())
