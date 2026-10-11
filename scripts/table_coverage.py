#!/usr/bin/env python3
"""Probability mass covered by a sparse classical table and expected hits per key.

Each retained weight is log2(p / 2^-32), so an entry's empirical probability is
2^(w - 32). Covered mass is the sum over retained entries; with 64 correctly
decrypted pairs, the expected number of pairs whose difference has a retained
entry is 64 x mass, and the chance of no hit at all is (1 - mass)^64.

Usage: table_coverage.py TABLE [TABLE ...]
"""

import hashlib
import json
import sys
from pathlib import Path

for arg in sys.argv[1:]:
    path = Path(arg)
    d = json.loads(path.read_text())
    mass = sum(2 ** (w - 32) for w in d["weights"].values())
    print(json.dumps({
        "table": arg, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rounds": d["rounds"], "samples_log2": d["sample_count"].bit_length() - 1, "min_count": d["min_count"],
        "entries": d["weights_count"], "covered_mass": mass,
        "expected_hits_64_pairs": 64 * mass, "p_no_hit_64_pairs": (1 - mass) ** 64,
    }))
