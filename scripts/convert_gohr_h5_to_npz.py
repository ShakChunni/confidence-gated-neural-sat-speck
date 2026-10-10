"""Dump Gohr's HDF5 weights to npz in layer order (conv1d*, batch_normalization*, dense*)."""
import sys
import h5py
import numpy as np
src, dst = sys.argv[1], sys.argv[2]
f = h5py.File(src, "r")
names = [n.decode() if isinstance(n, bytes) else n for n in f.attrs["layer_names"]]
out = {}
for prefix in ("conv1d", "batch_normalization", "dense"):
    for j, n in enumerate([n for n in names if n.startswith(prefix)]):
        def grab(k, o, j=j, prefix=prefix):
            if isinstance(o, h5py.Dataset):
                out[f"{prefix}{j}/{k.split('/')[-1]}"] = np.array(o)
        f[n].visititems(grab)
np.savez(dst, **out)
print(dst, len(out), "arrays")
