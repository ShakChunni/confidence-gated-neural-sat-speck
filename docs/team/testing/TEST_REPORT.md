# Hamza - testing and reproduction report

Run date: 31 October 2026 (Australia/Perth). Source: `78c1bd27257a364bf822d105a298085c7b963c90`, fresh clone of `ShakChunni/confidence-gated-neural-sat-speck`, branch `hamza/testing`. Existing tracked files were not edited. An earlier ZIP investigation is separate and is not the result reported below.

## Environment and scope

Windows 11 build 26200, AMD64; 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz; 15.69 GiB usable physical RAM. Python 3.12.14, pytest 8.4.2, pycryptosat 5.14.7, NumPy 2.5.4, PyTorch 2.15.0, matplotlib 3.11.2. Reused an existing environment whose installed packages satisfy requirements.txt; this was not a clean dependency-install test. `PYTHONPATH` points to this checkout's `src` and root. TEMP and TMP point to a writable workspace directory. CLI checks use the equivalent `python -m neural_sat_speck.cli` entry point.

Unit tests exercise cipher/key schedule primitives, SAT encodings, ranking, neural inference, statistics and execution guards. Integration tests combine components and verification paths. System checks exercise CLI KAT/key verification, deterministic data generation and the bounded SAT pilot. POSIX-dependent modules are not collected on Windows; skips do not establish those paths work here.

## Counts and full suite

Independent collection: **1,466 total = 1,377 unit + 59 integration + 30 root-level tests**. Counts below partition collected unit files; exact file groups and collection output are in TEST_COUNTS.md.

| Area | Collected tests |
|---|---:|
| cipher | 141 |
| key schedule | 19 |
| SAT encoding | 119 |
| solver isolation | 79 |
| ranking and neural | 390 |
| statistics | 68 |
| brute-force completer | 24 |
| run guards | 223 |
| other | 314 |
| **Total unit** | **1,377** |

Full-suite result: **72 failed, 1236 passed, 136 skipped, 22 errors in 474.15s (0:07:54)**. Process exit 1; measured wall time 476.99 seconds. Full output, commands and tracebacks: `pytest_full_20261031.txt`. These failures are retained, not repaired or classified wholesale as harmless. Consult the traceback for each failure; Windows capability and path handling affect this run. Collection checks ran alongside part of the suite, so this elapsed time is not a performance benchmark.

## System checks and reproduction

- KAT, 22-round key verification and seven-round key verification: Verified True. Generated two six-round instances with 64 pairs each, seed 999999, outside the repository.
- Bounded SAT pilot: two and four rounds verified known and independent pairs; seven rounds reached `solver_timeout` under its 10-second limit. At two rounds the returned master key differs in unused words; block agreement does not demonstrate unique recovery of all 64 key bits. Exact measured times and output: `system_checks.txt`.
- Level 2: D063, D064 and D065 rebuilt JSON exactly equal their matching saved summaries after JSON parsing. Report and LNCS figure runs exited 0 and passed built-in input-hash/count checks. Outputs stayed outside the repository; see LEVEL2_ANALYSES.md.
- Seven-round ranked demo: **not run**, because this machine is Windows/Intel and the assignment requires an Apple Silicon Mac. Both supplied tables parsed and their SHA-256 values match the saved D063 header. They were not rebuilt or committed. No new demo ledger or backup video exists.
- Colab: **not run**. Opening the specified notebook produced “Notebook not found” / GitHub API 404 and requested separate Colab GitHub authorization. CLI authorization does not authorize Colab. No cells ran and no final-cell output is claimed.

## Three historical problems caught

These are documented historical results, not experiments rerun here. `docs/timeline/10_published_networks_eight_rounds.md` records wrong scores from large MPS GPU batches; CPU comparison showed the earlier production B1 configuration was unaffected. The same file records a sparse seven-round table scoring every candidate zero because decrypted differences missed retained entries; the apparent neural win required comparison against a complete table. `docs/timeline/06_simon.md` records the compromised SIMON confirmation run: laptop sleep, background-priority throttling and temporary-runtime cleanup undermined timing. The records were retained but not accepted as clean evidence; a separately governed replacement was used.

## Rerun and remaining work

Use Python 3.11-3.13, follow docs/REPRODUCE.md, and work on `hamza/testing`. Set a writable TEMP/TMP and `PYTHONPATH=.;src` on Windows. Run `python -m pytest -q tests`; collect all, unit and integration tests with `--collect-only -q`. Replay the exact system and Level 2 commands in the companion evidence, choosing new external output paths and always supplying figure `--output-root`. Do not overwrite earlier evidence or change source to make tests pass. Only seed 999999 is used for generated demo data; no fresh campaign was launched.

An Apple Silicon demo/video and an authorized Colab run remain outstanding. Ashfaq should review the recorded failures and merge the pull request; master must not be modified by this testing work.
