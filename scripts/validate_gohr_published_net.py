"""Check the PyTorch rebuild reproduces Gohr's published accuracies (6r ~0.788, 7r ~0.616)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))



import numpy as np
import torch

from gohr_published_net import load_gohr, make_train_data

root = sys.argv[1]
rng = np.random.default_rng(999999)
for r in (6, 7):
    net = load_gohr(f"{root}/net{r}_small.npz")
    X, Y = make_train_data(10**6, r, rng)
    with torch.no_grad():
        z = torch.sigmoid(torch.cat([net.logits(torch.from_numpy(X[i:i + 100000])) for i in range(0, len(X), 100000)])).numpy()
    print(f"net{r} on {r} rounds: accuracy {np.mean((z > 0.5) == Y):.4f}  TPR {np.mean(z[Y == 1] > 0.5):.4f}  TNR {np.mean(z[Y == 0] <= 0.5):.4f}")
