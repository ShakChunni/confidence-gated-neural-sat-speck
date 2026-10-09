"""Frozen Step 5 count predictions from the queue-cost model (D-060 protocol §6).

For each prospective arm, per-key recovery probabilities come from the full N=32
arm distributions (optionally with every solve time scaled by 0.9/1.0/1.1 and a
per-candidate cap); 90% intervals for the count of 32 come from resampling
keys and Bernoulli outcomes.

Run as:
    python3 -I scripts/queue_cost_step5_predictions.py --input-dir DIR --output FILE
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit_queue_cost_model import KM, arm_of, dists, kq, load_dataset, p_recover, pooled  # noqa: E402

ARMS = [
    ("speck32_64", 2, "cms_native_gauss", None),
    ("speck32_64", 2, "cms_native_no_gauss", None),
    ("speck32_64", 2, "cms_native_gauss", 1.5),
    ("speck32_64", 2, "cms_native_no_gauss", 1.5),
    ("simon32_64", 2, "cms_native_gauss", None),
    ("simon32_64", 2, "cms_native_no_gauss", None),
    ("simon32_64", 2, "cms_native_gauss", 10.0),
    ("simon32_64", 2, "cms_native_no_gauss", 10.0),
]


def predictions(ds) -> dict:
    out: dict = {}
    for scale in (0.9, 1.0, 1.1):
        for cipher, depth, profile, tau in ARMS:
            arm = (cipher, depth, profile)
            rkm = KM([(t * scale, c) for t, c in pooled(ds, arm)])
            skm = KM([(t * scale, c) for t, c in pooled(ds, arm, field="s")])
            _, _, o = dists(ds, arm)
            rng = np.random.default_rng(59101)
            pv = np.array(
                [
                    p_recover(float(c["prep_seconds"]), kq(c), rkm, skm, o, rng, tau=tau)
                    for c in ds.cells
                    if arm_of(c) == arm
                ]
            )
            rr = np.random.default_rng(59102 if cipher == "speck32_64" else 59103)
            idx = rr.integers(0, 32, (20000, 32))
            counts = (rr.random((20000, 32)) < pv[idx]).sum(1)
            lo, med, hi = np.quantile(counts, [0.05, 0.5, 0.95])
            out.setdefault(f"{cipher}|{profile}|cap={tau}", {})[scale] = {
                "expected": round(float(pv.sum()), 2),
                "pi90": [int(lo), int(hi)],
                "median": int(med),
            }
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    out = predictions(load_dataset(args.input_dir))
    args.output.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
