"""Validate the full 8R SIMON table built by tools/simon_full_table_next.cpp.

Checks, all against independent sources:
  1. rows of the full table are bit-identical to rows from the audited
     tools/simon_queried_row_8r (used for the D-041 10-round campaign);
  2. the table is finite, non-negative and has total mass close to 1;
  3. random entries match a direct Python evaluation of the SIMON one-round
     transition from the 7R table (written here from the round equations, not
     from the C++ code);
  4. 9R rows from simon_queried_row_next on the 8R table match the same Python
     evaluation from the 8R table.

Usage: python scripts/validate_simon_r8_table.py OUT_JSON
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path("/Users/ashfaq/Dev/comp-sci-project")
T7 = REPO / "results/simon_confirmation/runtime_backup/resources/simon_predicted_ddt_7r.bin"
T8 = REPO / "runtime/higher_round_tables/simon_ddt_r8_full.bin"
OLD_TOOL = REPO / "results/simon_confirmation/runtime_backup/resources/simon_queried_row_8r"
NEXT_TOOL = REPO / "runtime/higher_round_tables/simon_queried_row_next"
W = 1 << 16
HEADER = 32


def rol(x: int, s: int) -> int:
    return ((x << s) | (x >> (16 - s))) & 0xFFFF


def f(x: int) -> int:
    return (rol(x, 1) & rol(x, 8)) ^ rol(x, 2)


def span(a: int) -> np.ndarray:
    """All output differences of f(x ^ a) ^ f(x) minus the offset f(a): a linear space."""
    gens = [f(a ^ (1 << i)) ^ f(1 << i) ^ f(a) for i in range(16)]
    space = {0}
    for g in gens:
        if g not in space:
            space |= {v ^ g for v in space}
    return np.array(sorted(space), dtype=np.int64)


def header(path: Path) -> dict:
    with path.open("rb") as handle:
        raw = handle.read(HEADER)
    return {
        "magic": raw[:8].decode(),
        "bits": int.from_bytes(raw[8:12], "little"),
        "rounds": int.from_bytes(raw[12:16], "little"),
        "initial": int.from_bytes(raw[16:20], "little"),
        "exponent": int.from_bytes(raw[20:24], "little"),
        "count": int.from_bytes(raw[24:32], "little"),
    }


def t7_row(m7, a: int) -> np.ndarray:
    return m7[a * W:(a + 1) * W].astype(np.float64) / float(1 << 60)


def t8_row(m8, a: int) -> np.ndarray:
    return np.array(m8[a * W:(a + 1) * W], dtype=np.float64)


def transition(row_a: np.ndarray, a: int, c: int) -> float:
    """P(next left = c, next right = a) = mean over v in span(a) of in[a][c ^ f(a) ^ v]."""
    return float(row_a[(c ^ f(a)) ^ span(a)].mean())


def tool_rows(tool: Path, table: Path, rows: list[int]) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "rows.bin"
        subprocess.run([str(tool), str(table), str(out)] + [str(r) for r in rows],
                       check=True, capture_output=True, text=True)
        return np.fromfile(out, dtype=np.float64).reshape(len(rows), W)


def main() -> None:
    rng = np.random.default_rng(20261027)
    res: dict = {"t8_header": header(T8), "t8_bytes": T8.stat().st_size}
    assert res["t8_header"] == {"magic": "SMDT0003", "bits": 16, "rounds": 8, "initial": 0x40,
                                "exponent": 0, "count": 1 << 32}, res["t8_header"]
    assert res["t8_bytes"] == HEADER + 8 * (1 << 32)
    m7 = np.memmap(T7, dtype="<u8", mode="r", offset=HEADER)
    m8 = np.memmap(T8, dtype="<f8", mode="r", offset=HEADER)

    # 1. bit-identical rows against the audited tool
    rows = [0x0000, 0x0001, 0x0040, 0x4000, 0x8000, 0xFFFF] + [int(x) for x in rng.integers(0, W, 10)]
    t0 = time.time()
    old = tool_rows(OLD_TOOL, T7, rows)
    mism = [r for i, r in enumerate(rows) if not np.array_equal(old[i], t8_row(m8, r))]
    maxdiff = max(float(np.max(np.abs(old[i] - t8_row(m8, r)))) for i, r in enumerate(rows))
    res["check1_rows"] = {"rows": rows, "bit_identical": len(mism) == 0, "mismatched_rows": mism,
                          "max_abs_diff": maxdiff, "seconds": round(time.time() - t0, 1)}

    # 2. finite, non-negative, total mass
    t0 = time.time()
    total, neg, nonfinite = 0.0, 0, 0
    chunk = 1 << 26
    for s in range(0, 1 << 32, chunk):
        x = np.asarray(m8[s:s + chunk])
        nonfinite += int((~np.isfinite(x)).sum())
        neg += int((x < 0).sum())
        total += float(x.sum(dtype=np.float64))
    res["check2_mass"] = {"total_mass": total, "negative": neg, "nonfinite": nonfinite,
                          "seconds": round(time.time() - t0, 1)}

    # 3. random entries against the Python transition from the 7R table
    entries = [(int(c), int(a)) for c, a in rng.integers(0, W, (64, 2))]
    errs = []
    for c, a in entries:
        want = transition(t7_row(m7, a), a, c)
        got = float(m8[(c << 16) | a])
        errs.append(abs(want - got) / max(abs(want), 1e-300) if want else abs(got))
    res["check3_python_entries"] = {"entries": len(entries), "max_rel_err": max(errs),
                                    "nonzero_entries": sum(float(m8[(c << 16) | a]) > 0 for c, a in entries)}

    # 4. 9R rows from the next-row tool on the 8R table
    rows9 = [0x0000, 0x0040] + [int(x) for x in rng.integers(0, W, 2)]
    t0 = time.time()
    r9 = tool_rows(NEXT_TOOL, T8, rows9)
    errs9 = []
    for i, c in enumerate(rows9):
        for a in [int(x) for x in rng.integers(0, W, 16)]:
            want = transition(t8_row(m8, a), a, c)
            got = float(r9[i, a])
            errs9.append(abs(want - got) / max(abs(want), 1e-300) if want else abs(got))
    res["check4_r9_rows"] = {"rows": rows9, "entries": len(errs9), "max_rel_err": max(errs9),
                             "row_sums": [float(r9[i].sum()) for i in range(len(rows9))],
                             "finite_nonneg": bool(np.isfinite(r9).all() and (r9 >= 0).all()),
                             "seconds": round(time.time() - t0, 1)}

    res["pass"] = (res["check1_rows"]["bit_identical"] and res["check2_mass"]["negative"] == 0
                   and res["check2_mass"]["nonfinite"] == 0 and abs(total - 1.0) < 1e-6
                   and res["check3_python_entries"]["max_rel_err"] < 1e-9
                   and res["check4_r9_rows"]["max_rel_err"] < 1e-9 and res["check4_r9_rows"]["finite_nonneg"])
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
