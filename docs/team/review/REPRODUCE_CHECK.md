# Reproduction check on Windows

> Sessions: first pass on 12 October 2026 (Levels 1 and 2, before the eight-round replication); this pass on 27 October 2026, after the d066 evidence landed.

Purpose: run Levels 1 and 2 of `docs/REPRODUCE.md` on a second operating system
and record whether the guide works. The guide's own check on line 92 and line
126 was made on macOS with Python 3.12; this one is on Windows.

Run on 27 October 2026, 15:12 to 15:36 AWST, on Windows 11 (build 26200, host
DESKTOP-1NHQSUD), PowerShell 5.1, Intel Core i5-1135G7, 8 GiB, from a fresh
clone of `master` at `292a491`. Full setup detail, including the two
environment problems I had to work around, is in
[CLEAN_CLONE_TEST.md](CLEAN_CLONE_TEST.md).

## Summary

| Level | What the guide claims | What happened here | Verdict |
|---|---|---|---|
| 1 | tests pass, `kat` verified | `kat` verified; **77 failed, 22 errors, 1,231 passed, 136 skipped** | fails |
| 2 | three analyses match, both figure profiles pass | three analyses match exactly; both profiles pass every check; figures are pixel-identical | passes |

## Level 1: tests and the known-answer test

```
python -m pytest -q
neural-sat-speck kat
```

`neural-sat-speck kat` passed in 4 s:

```
SPECK32/64 Known-Answer Test (KAT):
  Key:        ['0x1918', '0x1110', '0x0908', '0x0100']
  Plaintext:  (0x6574, 0x694c)
  Ciphertext: (0xa868, 0x42f2)
  Expected:   (0xa868, 0x42f2)
  Verified:   True
```

`pytest` did not pass. The summary was

```
77 failed, 1231 passed, 136 skipped, 22 errors in 304.95s
```

against the guide's macOS figure of 2,108 passed, 64 skipped, none failed. The
collection difference is expected: `tests/conftest.py` deliberately deselects
the POSIX runner modules on Windows, so 1,466 tests are collected here instead
of 2,172.

The failures are systematic, not flaky, and they have one dominant cause:

| Cause | Count | Location |
|---|---:|---|
| `PermissionError` when opening a directory to `fsync` it | 62 | `src/neural_sat_speck/backend/guard.py:353` |
| `FileNotFoundError` from a subprocess call to a solver binary that is not installed | 8 | tests that shell out to `cryptominisat5`, `cadical` or `clang` |
| `ValueError` in evidence packaging and Stage 2 control | 20 | `scripts/package_publication_evidence.py:138`, `src/neural_sat_speck/followups/stage2_control.py:1386`, `src/neural_sat_speck/backend/artifacts.py:315` |
| `AssertionError` | 14 | several |
| `AttributeError`, `TimeoutExpired` | 3 | several |

The directory-`fsync` failure is the one to fix first. It is a single call:

```python
descriptor = os.open(path.parent, os.O_RDONLY)
```

This is correct on POSIX and raises `PermissionError: [Errno 13]` on Windows,
so every writer that goes through the guard's JSON path fails its test. Two
exclusions would also be honest: the tests that need an external solver binary
should skip when the binary is absent, the way the guide already says they do
for `clang`.

I did not change any code. The failure list is in
[`evidence/pytest_windows_summary.txt`](evidence/pytest_windows_summary.txt).

### Two caveats that belong with these numbers

1. The suite ran with `torch==2.7.1`, not the newest version the requirements
   allow, because this machine lacks the Visual C++ 2019 to 2022 runtime and
   torch 2.14.1 cannot load without it. See `CLEAN_CLONE_TEST.md`, section 3.
2. `pip` could not reach PyPI on this network, so the pinned wheels were
   fetched separately and installed with `pip install --no-index`. The pins in
   `requirements.txt` (`pytest==8.4.2`, `ruff==0.16.2`, `pycryptosat==5.14.7`)
   were installed exactly as pinned.

Neither caveat explains the 62 guard failures; those tests do not touch torch.

## Level 2: analyses and figures from the saved results

### The three analyses

```
python scripts/analyze_d063.py docs/paper/evidence/d063_campaign_20261011/ledger_k00-31.jsonl
python scripts/analyze_d064.py docs/paper/evidence/d064_campaign_20261011/ledger_k00-31.jsonl
python scripts/analyze_d065.py docs/paper/evidence/d065_predictions/predictions.json d065_summary.json docs/paper/evidence/d065_campaign_20261012/ledger_k00-31.jsonl
```

All three ran in under a second each and printed the same JSON as the saved
`summary.json` in the same folder. I compared the parsed objects, not the bytes:
**0 differences** for d063, d064 and d065, across every key including the
per-key records. The guide's claim that "all three analyses matched their saved
summaries exactly" therefore holds on Windows as well.

### The two figure profiles

```
python scripts/generate_report_figures.py --profile report
python scripts/generate_report_figures.py --profile lncs
```

Both profiles passed every internal check in 6.2 s and 5.5 s. The guide's
warning is accurate: the image bytes differ from the committed ones, and the
input hashes and checks do not. Details, including the one thing that does not
match, are in [`FIGURE_CHECK.md`](FIGURE_CHECK.md).

## One thing the guide gets slightly wrong on Windows

`docs/REPRODUCE.md` line 124 says the manifest records input hashes under
`figures/<profile>/manifest.json`. On Windows the same script writes the input
paths with backslashes (`docs\paper\evidence\...`), while the committed
manifests use forward slashes. The hash values are identical; only the keys
differ. So a reader who diffs `manifest.json` between the two operating systems
sees a diff even though every input matches. The safest fix is to normalise the
separator with `Path.as_posix()` when the manifest is written.

## Verdict

Level 2 is reproducible on Windows, exactly and quickly. Level 1 is not, and
the reason is one line in `src/`. The guide's Windows paragraph should not say
that the suite still runs until that line is fixed; it currently over-promises
in a way a marker can reproduce in ten minutes.
