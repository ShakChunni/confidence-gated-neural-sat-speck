# Reproducing the results

This guide says what a reader can rerun, on which systems, and how long it
takes. It covers Windows 10/11, macOS, Linux and Google Colab.

The project is an experiment, so "reproduce" has four levels. Levels 1 to 3
run on any laptop in under half an hour. Level 4 is the full campaigns, which
need large tables and many hours.

| Level | What it shows | Time | Needs |
|---|---|---|---|
| 1. Tests and known-answer test | The ciphers, SAT model, ranking, verification and statistics code behave as specified | about 5 to 15 min | Python only |
| 2. Analyses and figures from the saved results | The published numbers and figures follow from the saved ledgers | about 1 min | The Python install below |
| 3. Short live demo | The attack pipeline runs end to end on the demo seed | about 1 min | The Python install below |
| 4. Full campaigns | New attack runs under the 300 s clock | 7 to 10 h per campaign | 4 to 32 GiB tables, published networks, a C compiler |

## Quickest route: Docker, on any system

This is the simplest way to check the numbers on Windows, macOS or Linux. It
needs only Docker and Git. The image runs on Linux, so the Windows limits below
do not apply. It takes about 1 minute to build and 6 minutes to run on a recent
laptop.

```bash
git clone https://github.com/ShakChunni/confidence-gated-neural-sat-speck.git
cd confidence-gated-neural-sat-speck
docker build -t project-check .
docker run --rm project-check
```

It runs the known-answer test, checks every recorded `.sha256` file, rebuilds
four saved campaign analyses and compares each with its saved summary, checks
the figure counts and input hashes, and runs the test suite.

The script `scripts/docker_check.sh` leaves out a few tests and says why. They
need macOS tools, the exact Python used for the frozen runs, or precise process
timing. None of them reads a saved result. Tests that need local data or solver
commands skip themselves. The image was checked on 1 Nov 2026 on 64-bit Arm;
the Intel build was built and its known-answer test passed.

### What each study can be rebuilt from

| Study | What the repository holds | Exact rebuild |
|---|---|---|
| Published networks, full table, completer choice, replication | Per-key ledgers and analysis scripts in `docs/paper/evidence/` | Yes, in the Docker run |
| Seven-round SPECK comparison | Per-key results and `scripts/analyze_seven_round_results.py` in `results/phase11_prospective/` | Yes, by running the script |
| Queue-cost rule on development cells | Cell tables and scripts in `docs/queue_cost_model_evidence_20261009/` | Yes, by running the scripts |
| SIMON comparison, second-stage study, fresh-key cost test, depth map, backend study | Per-key tables or saved summaries; the full raw files are too large or stay local | No, only the saved summaries can be read |
| Six-round reliability and the seven-round gate | Saved records in `results/`, without their inputs | No, only the saved records can be read |

Rerunning a campaign gives new timings, so it is a replication. The interval
in the report for the seven-round comparison uses Newcombe Method 10 and differs
slightly from the older interval in the saved analysis file.

## Setup

Use 64-bit CPython 3.11, 3.12 or 3.13. All three have `pycryptosat` wheels
for Windows, macOS and Linux, so no separate SAT solver install is needed.
Python 3.14 has no `pycryptosat` wheel yet.

- **macOS: use Python 3.13.** The watchdog tests need `os.waitid`, which
  Python on macOS has only from 3.13. On 3.11 or 3.12 those 16 tests (36
  with parameters) skip themselves.
- **Linux:** the `pycryptosat` wheels need glibc 2.34 or newer (Ubuntu 22.04
  or newer, which includes Colab).

**Windows (PowerShell)**

```powershell
git clone https://github.com/ShakChunni/confidence-gated-neural-sat-speck.git
Set-Location confidence-gated-neural-sat-speck
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --editable .
$env:PYTHONPATH = ".;src"
```

If PowerShell blocks activation, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.

`py -3.12` needs a 3.12 runtime registered with the Python launcher, and a
machine with only 3.14 installed answers `No runtime installed that matches
3.12`. Run `py -0p` to see which versions are there and use any of 3.11 to 3.13.

Windows also needs the Microsoft Visual C++ 2015-2022 Redistributable (x64)
before the `pip` steps. Without it the first `import torch` fails with
`OSError: [WinError 1114] A dynamic link library (DLL) initialization routine
failed`, which names `torch\lib\c10.dll` and hides the real cause.

**macOS or Linux**

```bash
git clone https://github.com/ShakChunni/confidence-gated-neural-sat-speck.git
cd confidence-gated-neural-sat-speck
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --editable .
export PYTHONPATH=.:src
```

**Google Colab.** Open `scripts/colab_reproduce.ipynb` in Colab
(File, Open notebook, GitHub tab, or upload the file) and run the cells in
order. It runs Levels 1 to 3 on a free CPU runtime. If the repository is
private, the notebook explains how to add a read-only GitHub token as a Colab
secret.

The commands below use `/` in paths. On Windows, `\` also works.

## Level 1: tests and the known-answer test

```bash
python -m pytest -q
neural-sat-speck kat
```

Some tests need things a fresh clone or a given system does not have. They
skip themselves and print the reason. A skip is not a failure. Every other
test should pass on macOS and in Docker. Windows has known failures, listed
under Known platform limits.

- **Solver commands.** A few tests call the `cryptominisat5` or `cadical`
  command-line solvers. The Python solver from `pycryptosat` does not need them.
- **clang.** The native brute-force completer and the encryption calibration
  are built with clang only, as frozen in the calibration. On Colab, the
  notebook has an optional cell that installs it.
- **Local data.** A few tests read an opened dataset that is git-ignored.
- **macOS only.** The worker checks of CPU priority and QoS run only on macOS.
- **`os.waitid`.** The real watchdog tests need it. Python on macOS has it
  only from 3.13, so on macOS with 3.11 or 3.12 those tests skip.
- **Windows.** The experiment runners use POSIX process groups, file locks
  and resource limits. On Windows their test modules are not collected (about
  700 of the 2,171 tests), and the pytest header says so. The cipher, SAT
  model, ranking, verification, statistics and figure tests still run, but the
  suite does not pass on Windows. See Known platform limits.
  The list is in `tests/conftest.py`.

Checked on 1 Nov 2026 on an Apple M5 with Python 3.14 (a local solver build):
2,143 passed, 28 skipped, none failed.

The known-answer test checks SPECK32/64 against the published test vector
(key `1918 1110 0908 0100`, plaintext `6574 694c`, ciphertext `a868 42f2`).

## Level 2: analyses and figures from the saved results

The ledgers for the eight-round and brute-force campaigns are saved under
`docs/paper/evidence/`. Each analysis script is the frozen, pre-registered
one. Its output should be identical to the saved `summary.json`.

```bash
python scripts/analyze_d063.py docs/paper/evidence/d063_campaign_20261011/ledger_k00-31.jsonl
python scripts/analyze_d064.py docs/paper/evidence/d064_campaign_20261011/ledger_k00-31.jsonl
python scripts/analyze_d065.py docs/paper/evidence/d065_predictions/predictions.json d065_summary.json docs/paper/evidence/d065_campaign_20261012/ledger_k00-31.jsonl
```

The first two print JSON. Compare it with `summary.json` in the same folder.
The third writes `d065_summary.json`.

The paper figures are redrawn from the saved results:

```bash
python scripts/generate_report_figures.py --profile report --force
python scripts/generate_report_figures.py --profile lncs --force
```

Before saving, the script checks every input file against its recorded
SHA-256 and every drawn count against the published count. It stops without
writing anything if either differs. The image bytes can differ slightly with
another matplotlib version, so do not expect identical file hashes. The report
profile draws seven figures. The input hashes and checks in
`figures/<profile>/manifest.json` should not change.

This was checked on 27 Oct 2026 on a fresh clone with Python 3.12: all three
analyses matched their saved summaries exactly, and both figure profiles
passed their checks.

## Level 3: short live demo

All demo runs use the demo seed `999999`.

```bash
python scripts/phase2_sat_pilot.py --rounds 2 4 7 --pair-count 4 --timeout 10
neural-sat-speck generate-data --instances 2 --pairs 64 --rounds 6 --seed 999999 --output demo_instances.json
neural-sat-speck verify-key --key 0x1918 0x1110 0x0908 0x0100 --rounds 7
```

The first command recovers the full 64-bit key by exact SAT solving from 4
known pairs and checks it on fresh pairs. At 2 and 4 rounds it takes well
under a second. At 7 rounds, 4 pairs and 10 s are usually not enough, and the
run ends as `solver_timeout`. That is the expected result, and it is why the
paper ranks round keys before solving.

## Level 4: full campaigns

The full campaigns are not needed to check the paper. Each one needs:

- the multi-gigabyte difference tables (4 GiB for SPECK at seven rounds,
  32 GiB for SIMON at seven rounds), built by `scripts/colab_build_speck_full_ddt.ipynb`
  and the tools in `tools/`;
- Gohr's published SPECK networks (github.com/agohr/deep_speck, converted with
  `scripts/convert_gohr_h5_to_npz.py`) and Bao's published SIMON networks. They
  are not stored here and there is no download script. Their expected hashes
  are in the saved ledger headers and in the Stage 2 control code;
- a C compiler for the brute-force completer (`tools/bruteforce_completion.c`);
- one machine running serially for 7 to 10 hours per campaign.

The eight-round SPECK campaigns use `scripts/run_higher_round_campaign.py`,
with one analysis script per campaign (`scripts/analyze_d063.py` to
`scripts/analyze_d066.py`). The ranking-depth and queue-cost studies use
`scripts/run_backend_ranking_depth.py` and
`scripts/launch_prospective_queue_cost.py`. `docs/timeline/` says what each
campaign tested and where its saved evidence is. Runs use only the demo seed
unless a protocol is frozen and a fresh seed is approved. Timings depend on the machine, so a rerun on
other hardware is a replication, not a byte-for-byte reproduction.

## Known platform limits

- On Windows, two separate runs of Level 1 finished with 77 and 72
  failed tests and 22 errors each (see `docs/team/review` and
  `docs/team/testing`). In each run, 62 of them come from one helper that
  syncs a directory to disk, which Windows refuses. That helper is part of
  the frozen backend, so it stays unchanged. Level 2 passes on Windows.
  Paths in a `manifest.json` written on Windows use backslashes, so its keys
  differ from the saved copy while the hash values match.
- Some experiment runners use POSIX process groups to stop a solver at the
  deadline. They run on macOS and Linux. On Windows, use WSL or Colab for
  Level 4.
- The SIMON neural arms need TensorFlow 2.16 and `h5py` in a separate
  environment (Python 3.12, NumPy 1.26). Levels 1 to 3 do not need it.
- `.gitattributes` turns off line-ending conversion, so a Windows checkout
  keeps the exact bytes that the hash checks expect.
- **Windows, Level 1.** The durability guard opens a directory to `fsync` it
  (`src/neural_sat_speck/backend/guard.py`). That is correct on POSIX and raises
  `PermissionError: [Errno 13]` on Windows, so the tests that write through the
  guard fail, and the tests that call an external solver binary error when the
  binary is not installed. A fresh Windows clone on 27 October 2026 gave 1,231
  passed, 77 failed, 22 errors and 136 skipped. Level 2 is unaffected.
