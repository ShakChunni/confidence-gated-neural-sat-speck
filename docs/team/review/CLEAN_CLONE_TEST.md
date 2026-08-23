# Clean-clone setup test

> Sessions: first setup run on 23 August 2026; this clean-clone run, with the corrected instructions, on 27 October 2026.

Purpose: follow the repository's own instructions from a new folder, on a
machine that had never built this project, and record what happened. This is
the outside-reader test the brief asks for.

Run on 27 October 2026, 15:12 to 15:26 AWST.
Machine: Windows 11 (build 26200, host DESKTOP-1NHQSUD), PowerShell 5.1, Intel Core
i5-1135G7, 8 GiB.
Repository state: `master` at `292a491`.
Clone location: a new folder, `review-work\clean_clone\`, outside the working
copy.

The run is reproducible from the transcript at the end of this file: every
command, its exit code and its wall time is recorded there.

## Result in one line

The clone and the environment build work. **The test suite does not pass on
Windows as the guide implies:** 1,231 passed, 77 failed, 22 errors, 136
skipped. Of the 99 failing or erroring tests, 62 are one bug in
`src/neural_sat_speck/backend/guard.py` line 353, where the durability helper
tries to `fsync` a directory, which Windows refuses.

## Steps, as written in the instructions

| # | Command (from `README.md` / `docs/REPRODUCE.md`) | Outcome | Time |
|---:|---|---|---:|
| 1 | `git clone https://github.com/ShakChunni/confidence-gated-neural-sat-speck.git` | works | 6.4 s |
| 2 | `py -3.12 -m venv .venv` | **fails**: no 3.12 registered with the Python launcher | 0.1 s |
| 3 | `.\.venv\Scripts\Activate.ps1` | works, once a `.venv` exists | 1 s |
| 4 | `python -m pip install --upgrade pip` | **hangs** on this network (see below) | > 480 s, stopped |
| 5 | `python -m pip install -r requirements.txt` | works only after two fixes (see below) | 153.7 s |
| 6 | `python -m pip install --editable .` | works | 11.2 s |
| 7 | `$env:PYTHONPATH = ".;src"` | works, and is needed | instant |
| 8 | `python -m pytest -q` | **99 tests fail or error** | 305 s |
| 9 | `neural-sat-speck kat` | works: `Verified: True` | 4 s |

## What went wrong, in the order a reader would hit it

### 1. `py -3.12` assumes a Python the machine may not have

The only Python on this machine is 3.14, which the project correctly excludes
(there is no `pycryptosat` wheel for it). The launcher answers:

```
[ERROR] No runtime installed that matches 3.12. Try running "py install 3.12".
```

Step 2 is therefore a dead end on a clean Windows laptop. The guide says "Use
64-bit CPython 3.11, 3.12 or 3.13", so the fix is small: tell the reader to
check `py -0p` and use whichever of 3.11 to 3.13 is installed, instead of
hard-coding 3.12 in the one command they must run first.

For this test I created the environment with a CPython 3.12.14 already on the
machine. That is the only deviation from the written steps that affects what
runs afterwards.

### 2. `pip` could not reach PyPI from this network

`python -m pip install --upgrade pip` ran for more than eight minutes with no
output and no open connection, and a single-package download behaved the same
way. Plain `urllib` from the same interpreter reached PyPI in 0.4 s, so this is
pip's HTTP stack on this network, not the repository.

To get past it without changing what gets installed, I downloaded the exact
pinned wheels with `urllib` and installed them with
`pip install --no-index --find-links <dir>`. `requirements.txt` pins
`pytest==8.4.2`, `ruff==0.16.2` and `pycryptosat==5.14.7`; those three pins were
installed at the pinned versions.

This is an environment problem, not a repository problem, so it is not a README
fix. It is recorded here because any reader behind a filtering proxy will hit
it.

### 3. `torch` fails to import on Windows without the Visual C++ runtime

`import torch` failed with

```
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed.
Error loading "...\torch\lib\c10.dll" or one of its dependencies.
```

The machine has the Visual C++ 2015 runtime (`msvcp140.dll` 14.00.23026) and no
2019 to 2022 runtime. The application log confirms the fault: `python.exe`
crashes inside `msvcp140.dll`, version 14.0.23026.0. Torch 2.14.1 needs a newer
runtime than that.

What I did: installed `torch==2.7.1`, which loads against the runtime present.
This is a deviation and it matters, so it is stated plainly: the suite below ran
with torch 2.7.1, not with the newest torch the requirements allow.

What a reader should do instead: install the Microsoft Visual C++ 2015-2022
Redistributable (x64) before `pip install`. That is a one-line prerequisite the
guide does not mention, and adding it is the second README fix.

### 4. The test suite fails on Windows

```
77 failed, 1231 passed, 136 skipped, 22 errors in 304.95s
```

The guide's Windows paragraph says that about 700 runner tests are not
collected, and that "the cipher, SAT model, ranking, verification, statistics
and figure tests all still run". The cipher, ranking, verification and figure
tests do run and pass. But the suite is not green, and the causes are
systematic, not random:

| Cause | Tests | Where |
|---|---:|---|
| `PermissionError` opening a directory to `fsync` it | 62 | `src/neural_sat_speck/backend/guard.py:353` |
| `FileNotFoundError` from a subprocess call (a solver binary that is not installed) | 8 | `tests/unit/test_*` that shell out |
| `ValueError` in evidence packaging and Stage 2 control | 20 | `scripts/package_publication_evidence.py:138`, `src/neural_sat_speck/followups/stage2_control.py:1386`, `src/neural_sat_speck/backend/artifacts.py:315` |
| `AssertionError` | 14 | several |
| `AttributeError`, `TimeoutExpired` | 3 | several |

The dominant one is worth naming precisely, because it is one line:

```python
descriptor = os.open(path.parent, os.O_RDONLY)
```

On POSIX this opens a directory so the code can `fsync` it after replacing a
file, which is the correct durability idiom. On Windows, opening a directory
with `os.open` raises `PermissionError: [Errno 13]`, so every caller of the
guard's JSON writer fails. That is 62 of the 99.

I did not change the code. The unit's instructions for this task say not to,
and a fix belongs to whoever owns `src/`.

## What did work, end to end

- The clone is small and fast: 50 MB working tree in 6.4 s.
- `pip install --editable .` builds and installs the CLI in 11 s.
- `neural-sat-speck kat` prints the published test vector and `Verified: True`.
- The three Level 2 analyses and both figure profiles reproduce exactly; that is
  the subject of `REPRODUCE_CHECK.md` and `FIGURE_CHECK.md`.

## Recommended document fixes

Two changes, both to steps that did not work as written. They are in the next
commit, and they touch nothing but the setup instructions:

1. `docs/REPRODUCE.md`, Windows block: stop hard-coding `py -3.12`; tell the
   reader to check `py -0p` and use any installed 3.11 to 3.13.
2. `docs/REPRODUCE.md` and `README.md`: add the Microsoft Visual C++
   2015-2022 Redistributable to the Windows prerequisites, with the
   `WinError 1114` symptom, so the next reader does not lose an hour to it.

The failing suite is not a documentation problem. It is recorded here and in
`REPRODUCE_CHECK.md`, and it needs a code fix by the owner of `src/`.

## Transcript

The full pytest output is 4,000 lines of progress dots and tracebacks, so it is
not committed. The part that matters is: the summary line, and the name of
every test that failed or errored. Both are in
[`evidence/pytest_windows_summary.txt`](evidence/pytest_windows_summary.txt),
copied from the run without editing. The times in the table above come from the
same transcript.
