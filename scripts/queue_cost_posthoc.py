"""POST HOC (labelled) checks on the queue-cost model; not part of the protocol.

Reports (1) the SIMON/SPECK d1 counterfactual that swaps BOTH the wrong-candidate
and the true-candidate observations between the G and X profiles; (2) the
distribution of inter-attempt overheads; and (3) the sensitivity of the cap
predictions to +-10% solver-speed scaling.

Run as:
    python3 -I scripts/queue_cost_posthoc.py --input-dir DIR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit_queue_cost_model import (  # noqa: E402
    KM,
    Dataset,
    arm_of,
    dists,
    kq,
    load_dataset,
    p_recover,
    pooled,
)


def count(ds: Dataset, target, r_arm=None, s_arm=None, scale=1.0, tau=None):
    rkm = KM([(t * scale, c) for t, c in pooled(ds, r_arm or target)])
    skm = KM([(t * scale, c) for t, c in pooled(ds, s_arm or target, field="s")])
    _, _, o = dists(ds, target)
    rng = np.random.default_rng(59005)
    total = sum(
        p_recover(float(c["prep_seconds"]), kq(c), rkm, skm, o, rng, tau=tau)
        for c in ds.cells
        if arm_of(c) == target
    )
    return round(total, 2)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Post hoc checks on the queue-cost model."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="directory containing cells.csv and attempts.csv",
    )
    args = parser.parse_args(argv)
    ds = load_dataset(args.input_dir)

    for ci in ("simon32_64", "speck32_64"):
        G, X = (ci, 1, "cms_native_gauss"), (ci, 1, "cms_native_no_gauss")
        print(
            ci,
            "d1 X with G r+s:",
            count(ds, X, G, G),
            " G with X r+s:",
            count(ds, G, X, X),
        )

    o = []
    for c in ds.cells:
        o += ds.obs[c["cell_id"]]["o"]
    o = np.array(o)
    print(
        "overhead n",
        len(o),
        "quantiles",
        np.quantile(o, [0, 0.5, 0.9, 0.99, 1]).round(3),
    )

    for ci, taus in (("speck32_64", (1, 1.5, 2, 3)), ("simon32_64", (5, 10, 20))):
        for p in ("cms_native_gauss", "cms_native_no_gauss"):
            a = (ci, 2, p)
            for tau in taus:
                print(
                    a,
                    "tau",
                    tau,
                    [count(ds, a, scale=s, tau=tau) for s in (0.9, 1.0, 1.1)],
                )


if __name__ == "__main__":
    main()
