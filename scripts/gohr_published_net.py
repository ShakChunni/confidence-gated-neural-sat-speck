"""Gohr's published single-block ResNet distinguisher (deep_speck, Keras 2.1.3) rebuilt in PyTorch.

Weights are copied verbatim from the published HDF5 files; architecture follows
single_block_resnet.json: Reshape(4,16) -> Permute -> Conv1D(32,1) BN ReLU ->
[Conv1D(32,3) BN ReLU Conv1D(32,3) BN ReLU] + shortcut -> Flatten ->
Dense(64) BN ReLU -> Dense(64) BN ReLU -> Dense(1) sigmoid. BN eps 1e-3.
"""

import numpy as np
import torch
from torch import nn

MASK = 0xFFFF


class GohrNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv1d(4, 32, 1)
        self.b0 = nn.BatchNorm1d(32, eps=1e-3)
        self.c1 = nn.Conv1d(32, 32, 3, padding=1)
        self.b1 = nn.BatchNorm1d(32, eps=1e-3)
        self.c2 = nn.Conv1d(32, 32, 3, padding=1)
        self.b2 = nn.BatchNorm1d(32, eps=1e-3)
        self.d0 = nn.Linear(512, 64)
        self.e0 = nn.BatchNorm1d(64, eps=1e-3)
        self.d1 = nn.Linear(64, 64)
        self.e1 = nn.BatchNorm1d(64, eps=1e-3)
        self.d2 = nn.Linear(64, 1)

    def logits(self, bits):  # bits: (N, 64) float, word-major MSB-first
        x = bits.view(-1, 4, 16)  # (N, C=4 words, L=16 bit positions)
        a = torch.relu(self.b0(self.c0(x)))
        h = torch.relu(self.b1(self.c1(a)))
        h = torch.relu(self.b2(self.c2(h)))
        h = (a + h).permute(0, 2, 1).reshape(-1, 512)  # Keras flattens (L, C)
        h = torch.relu(self.e0(self.d0(h)))
        h = torch.relu(self.e1(self.d1(h)))
        return self.d2(h).squeeze(1)


def load_gohr(npz_path):
    w = np.load(npz_path)
    net = GohrNet()
    t = lambda k: torch.tensor(w[k])
    for j, mod in enumerate([net.c0, net.c1, net.c2]):
        mod.weight.data = t(f"conv1d{j}/kernel:0").permute(2, 1, 0).contiguous()
        mod.bias.data = t(f"conv1d{j}/bias:0")
    for j, mod in enumerate([net.b0, net.b1, net.b2, net.e0, net.e1]):
        mod.weight.data = t(f"batch_normalization{j}/gamma:0")
        mod.bias.data = t(f"batch_normalization{j}/beta:0")
        mod.running_mean.data = t(f"batch_normalization{j}/moving_mean:0")
        mod.running_var.data = t(f"batch_normalization{j}/moving_variance:0")
    for j, mod in enumerate([net.d0, net.d1, net.d2]):
        mod.weight.data = t(f"dense{j}/kernel:0").T.contiguous()
        mod.bias.data = t(f"dense{j}/bias:0")
    return net.eval()


# ---- SPECK32/64 (Gohr's speck.py conventions: (x, y) = (left, right)) ----
def ror(v, r):
    return ((v >> r) | (v << (16 - r))) & MASK


def rol(v, r):
    return ((v << r) | (v >> (16 - r))) & MASK


def expand_key(k, rounds):  # k = (l2, l1, l0, k0) as in Gohr's speck.py
    ks = [k[-1]]
    l = list(reversed(k[:-1]))
    for i in range(rounds - 1):
        l_i, k_i = l[i % 3], ks[i]
        new_l = ((ror(l_i, 7) + k_i) & MASK) ^ i
        new_k = rol(k_i, 2) ^ new_l
        l[i % 3] = new_l
        ks.append(new_k)
    return ks


def enc(x, y, ks):
    for k in ks:
        x = ((ror(x, 7) + y) & MASK) ^ k
        y = rol(y, 2) ^ x
    return x, y


def dec_one(x, y, k):
    y = ror(y ^ x, 2)
    x = rol(((x ^ k) - y) & MASK, 7)
    return x, y


def to_bits(c0l, c0r, c1l, c1r):
    words = np.stack([c0l, c0r, c1l, c1r], axis=1).astype(np.uint32)
    shifts = np.arange(15, -1, -1, dtype=np.uint32)
    return ((words[:, :, None] >> shifts) & 1).reshape(-1, 64).astype(np.float32)


def make_train_data(n, rounds, rng, diff=(0x0040, 0)):
    y = rng.integers(0, 2, n)
    keys = rng.integers(0, 65536, (4, n), dtype=np.uint32)
    p0l = rng.integers(0, 65536, n, dtype=np.uint32)
    p0r = rng.integers(0, 65536, n, dtype=np.uint32)
    p1l, p1r = p0l ^ diff[0], p0r ^ diff[1]
    rand = y == 0
    p1l[rand] = rng.integers(0, 65536, rand.sum(), dtype=np.uint32)
    p1r[rand] = rng.integers(0, 65536, rand.sum(), dtype=np.uint32)
    ks = expand_key(list(keys), rounds)
    c0l, c0r = enc(p0l, p0r, ks)
    c1l, c1r = enc(p1l, p1r, ks)
    return to_bits(c0l, c0r, c1l, c1r), y
