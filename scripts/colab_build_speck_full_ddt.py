"""Build a full 32-bit sampled difference table for (R-1)-round SPECK32/64 on a CUDA GPU.

Run on Google Colab with an A100 (40 or 80 GB) or L4 (24 GB) runtime:

    !python colab_build_speck_full_ddt.py --rounds 7 --log2-samples 42 --out /content/drive/MyDrive/speck_ddt_r7.npy

What it does
  * Samples plaintext pairs with input difference (0x0040, 0x0000) and
    independent uniformly random round keys (the same model as the project's
    sparse tables), encrypts both for --rounds rounds, and counts the 32-bit
    output difference (x in the high 16 bits, y in the low 16 bits) in 2^32
    uint32 counters on the GPU (16 GB).
  * Converts counts to log2(p / 2^-32) with +0.5 smoothing and stores them as
    int8 in units of 1/16 bit (4 GB), clipped to [-8, +7.94] bits.
  * Writes a JSON sidecar with the settings, totals and a SHA-256 of the table.

Self-test: before sampling, the GPU SPECK is checked against the published
SPECK32/64 test vector (key 1918 1110 0908 0100, plaintext 6574 694c ->
ciphertext a868 42f2) and against a NumPy reference on random inputs.
"""

import argparse
import hashlib
import json
import time

import cupy as cp
import numpy as np

KERNEL = r"""
extern "C" {
__device__ __forceinline__ unsigned long long splitmix(unsigned long long &s) {
    unsigned long long z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
__device__ __forceinline__ unsigned int ror16(unsigned int v, int r) { return ((v >> r) | (v << (16 - r))) & 0xFFFF; }
__device__ __forceinline__ unsigned int rol16(unsigned int v, int r) { return ((v << r) | (v >> (16 - r))) & 0xFFFF; }

__global__ void sample(unsigned int *counts, unsigned long long seed, unsigned long long base,
                       unsigned long long n, int rounds) {
    unsigned long long i = base + blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= base + n) return;
    unsigned long long s = seed ^ (i * 0xD1B54A32D192ED03ULL);
    unsigned long long r0 = splitmix(s), r1 = splitmix(s);
    unsigned int x0 = r0 & 0xFFFF, y0 = (r0 >> 16) & 0xFFFF;
    unsigned int x1 = x0 ^ 0x0040, y1 = y0;
    unsigned long long pool = r1; int left = 4;
    for (int k = 0; k < rounds; k++) {
        if (left == 0) { pool = splitmix(s); left = 4; }
        unsigned int rk = pool & 0xFFFF; pool >>= 16; left--;
        x0 = ((ror16(x0, 7) + y0) & 0xFFFF) ^ rk; y0 = rol16(y0, 2) ^ x0;
        x1 = ((ror16(x1, 7) + y1) & 0xFFFF) ^ rk; y1 = rol16(y1, 2) ^ x1;
    }
    unsigned int d = ((x0 ^ x1) << 16) | (y0 ^ y1);
    atomicAdd(&counts[d], 1u);
}

__global__ void enc_test(const unsigned int *xin, const unsigned int *yin, const unsigned int *ks,
                         unsigned int *xout, unsigned int *yout, int n, int rounds) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int x = xin[i], y = yin[i];
    for (int k = 0; k < rounds; k++) {
        x = ((ror16(x, 7) + y) & 0xFFFF) ^ ks[k * n + i]; y = rol16(y, 2) ^ x;
    }
    xout[i] = x; yout[i] = y;
}
}
"""


def np_enc(x, y, ks):
    for k in ks:
        x = ((((x >> 7) | (x << 9)) & 0xFFFF) + y & 0xFFFF) ^ k
        y = (((y << 2) | (y >> 14)) & 0xFFFF) ^ x
    return x, y


def expand_key(k, rounds):  # k = (l2, l1, l0, k0), Gohr's convention
    ks = [k[3]]
    l = list(reversed(k[:3]))
    for i in range(rounds - 1):
        l_new = ((((l[i] >> 7) | (l[i] << 9)) & 0xFFFF) + ks[i] & 0xFFFF) ^ i
        ks.append((((ks[i] << 2) | (ks[i] >> 14)) & 0xFFFF) ^ l_new)
        l.append(l_new)
    return ks


def self_test(mod):
    enc_test = mod.get_function("enc_test")
    ks = expand_key([0x1918, 0x1110, 0x0908, 0x0100], 22)
    x, y = np_enc(np.uint32(0x6574), np.uint32(0x694C), [np.uint32(k) for k in ks])
    assert (int(x), int(y)) == (0xA868, 0x42F2), "NumPy reference fails the SPECK32/64 test vector"
    rng = np.random.default_rng(1)
    n, r = 4096, 7
    xin = rng.integers(0, 1 << 16, n, dtype=np.uint32)
    yin = rng.integers(0, 1 << 16, n, dtype=np.uint32)
    kk = rng.integers(0, 1 << 16, (r, n), dtype=np.uint32)
    xo, yo = cp.zeros(n, cp.uint32), cp.zeros(n, cp.uint32)
    enc_test(((n + 255) // 256,), (256,), (cp.asarray(xin), cp.asarray(yin), cp.asarray(kk.ravel()), xo, yo, np.int32(n), np.int32(r)))
    ex, ey = np_enc(xin, yin, list(kk))
    assert np.array_equal(cp.asnumpy(xo), ex) and np.array_equal(cp.asnumpy(yo), ey), "GPU SPECK disagrees with NumPy"
    print("self-test passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--log2-samples", type=int, default=42)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    mod = cp.RawModule(code=KERNEL)
    self_test(mod)
    sample = mod.get_function("sample")
    counts = cp.zeros(1 << 32, dtype=cp.uint32)
    total, batch, threads = 1 << a.log2_samples, 1 << 30, 256
    t0 = time.time()
    for base in range(0, total, batch):
        n = min(batch, total - base)
        sample(((n + threads - 1) // threads,), (threads,),
               (counts, np.uint64(a.seed), np.uint64(base), np.uint64(n), np.int32(a.rounds)))
        if (base // batch) % 64 == 0:
            cp.cuda.Device().synchronize()
            print(f"{(base + n) / total:6.1%}  {time.time() - t0:7.0f} s", flush=True)
    cp.cuda.Device().synchronize()
    sample_s = time.time() - t0
    assert int(counts.sum(dtype=cp.uint64)) == total, "count total mismatch"
    max_count = int(counts.max())
    # int8 log-likelihood ratio in 1/16-bit units, computed in slices to bound memory.
    q = np.empty(1 << 32, dtype=np.int8)
    denom = total + 0.5 * (1 << 32)
    step = 1 << 28
    for s in range(0, 1 << 32, step):
        llr = cp.log2((counts[s:s + step].astype(cp.float64) + 0.5) / denom * float(1 << 32))
        q[s:s + step] = cp.asnumpy(cp.clip(cp.rint(llr * 16), -128, 127).astype(cp.int8))
    np.save(a.out, q)
    h = hashlib.sha256()
    with open(a.out, "rb") as f:
        while chunk := f.read(1 << 26):
            h.update(chunk)
    sha = h.hexdigest()
    meta = {"rounds": a.rounds, "log2_samples": a.log2_samples, "seed": a.seed, "input_difference": [64, 0],
            "independent_round_keys": True, "difference_layout": "x<<16 | y", "llr_unit_bits": 1 / 16,
            "smoothing": 0.5, "max_count": max_count, "sample_seconds": round(sample_s, 1),
            "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(), "table_sha256": sha}
    json.dump(meta, open(a.out + ".json", "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
