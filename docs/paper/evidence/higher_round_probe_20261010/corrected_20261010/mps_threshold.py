import numpy as np, torch
from gohr_torch import load_gohr, make_train_data
G = "/Users/ashfaq/Dev/comp-sci-project/runtime/gohr_deep_speck/net7_small.npz"
cpu, mps = load_gohr(G), load_gohr(G).to("mps")
X, Y = make_train_data(70000, 7, np.random.default_rng(2)); X = torch.from_numpy(X)
with torch.no_grad():
    ref = cpu.logits(X).numpy()
    for n in (32768, 65535, 65536, 65537, 70000):
        out = mps.logits(X[:n].to("mps")).cpu().numpy()
        d = np.abs(out - ref[:n]); bad = np.nonzero(d > 1e-3)[0]
        print(f"batch {n}: maxdiff {d.max():.3g}  first bad row {bad[0] if len(bad) else None}  n_bad {len(bad)}")
print(torch.__version__)
