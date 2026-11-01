# Containment Is Not Recovery: When Key Ranking Repays Its Cost

This repository holds the code, data generators, tests and saved results for a
COMP6002 research project at Curtin University. We study reduced-round key
recovery on SPECK32/64 and SIMON32/64. A ranker orders candidates for the last
round keys, and an exact SAT solver (CryptoMiniSat) finishes the 64-bit master
key from the best candidates.

Rankers are usually judged by accuracy, or by whether the right candidate
reaches the shortlist. Neither counts the time it takes. We put ranking, search
and verification on one 300 s clock per key, with 64 chosen-plaintext pairs. A
key counts only if it matches the assigned key and reproduces two fresh blocks
that the attack never saw.

This is a reduced-round study. It does not attack full-round SPECK or SIMON.

## Main findings

- **Containment is not recovery.** On seven-round SPECK, a screened neural
  pipeline and a neural-classical hybrid recover 110 and 111 of 128 fresh keys.
  By 100 s the hybrid has recovered 104 and the neural pipeline 29.
- **A simple cost rule explains most outcomes.** Preparation time, the correct
  candidate's queue position and the cost of rejecting a wrong candidate
  explain 94.5% of 256 development cells. Containment alone explains 61.7%.
  The rule's count predictions then passed a fresh-key test on both ciphers.
- **The rule can choose the method.** Before new keys were drawn it chose SAT
  at seven rounds and brute-force completion at eight. Both choices held up.
- **Neural ranking pays only sometimes.** Our own pair-aware network lost to a
  classical table at seven rounds. At eight rounds, Gohr's published network
  beat a complete 4 GiB classical table, 24 to 18 on 32 fresh keys. On a
  second fresh set of 64 keys it won 49 to 41 (exact McNemar p = 0.021).
- **On SIMON at ten rounds**, a classical table recovered 53 of 64 fresh keys
  and a published-network hybrid 49. We could not tell them apart (p = 0.34).
- **The original confidence-gate idea failed.** Better prediction of ranking
  failure did not give a threshold that held on new keys, and it never beat a
  plain fixed search.

The story of how we got here, stage by stage, is in
[`docs/timeline/`](docs/timeline/README.md).

## Repository layout

| Path | What it holds |
|---|---|
| `src/neural_sat_speck/` | Ciphers, data generation, neural rankers, SAT models, attacks, statistics and the CLI |
| `tests/` | Unit and integration tests, including known-answer tests for both ciphers |
| `scripts/` | Experiment runners, analyses and figure generators |
| `tools/` | Native C/C++ code: reference encryption, brute-force completer, SIMON table builders |
| `configs/` | Dataset, training and experiment configurations |
| `models/` | Small trained models and classical tables used by the attacks |
| `datasets/` | Generated attack instances with manifests and hashes |
| `results/`, `docs/paper/evidence/` | Saved ledgers, analyses and checksums from the campaigns |
| `artifacts/benchmark_formulas/` | SPECK and SIMON SAT formulas in native-XOR and CNF form, with variable maps |
| `figures/` | Paper and report figures (PDF and PNG) |
| `docs/` | Setup, reproduction, dataset notes and the project timeline |

Large tables (4 GiB and 32 GiB) and the published networks are not stored here.
`docs/REPRODUCE.md` explains how to build or obtain them and check their
hashes. It also gives one Docker command that checks the saved results on any
system.

## Quick start

Use 64-bit Python 3.11, 3.12 or 3.13. On macOS use 3.13.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

python -m pytest -q
```

On Windows (PowerShell), check which versions the launcher can see with
`py -0p`, then create the environment with any installed 3.11 to 3.13, for
example `py -3.12 -m venv .venv`, and activate it with
`.\.venv\Scripts\Activate.ps1`. Windows also needs the Microsoft Visual C++
2015-2022 Redistributable (x64) before the `pip` steps; without it the first
`import torch` fails with `OSError: [WinError 1114]`.

A short check of the cipher and a small demo dataset:

```bash
neural-sat-speck kat
neural-sat-speck generate-data \
  --instances 10 --pairs 64 --rounds 6 --seed 999999 \
  --output data/demo_instances.json
```

Seed `999999` is the demo seed. It has no evaluation role.

## Reproducing the results

[`docs/REPRODUCE.md`](docs/REPRODUCE.md) gives step-by-step commands for
Windows, macOS, Linux and Google Colab. It has four levels: the tests, the
analyses and figures rebuilt from the saved ledgers, a short live demo, and the
full campaigns. The first three run on a laptop in under half an hour. In
Colab, open [`scripts/colab_reproduce.ipynb`](scripts/colab_reproduce.ipynb)
and run the cells in order.

## How the experiments were run

- Every compared method gets its own solver instance. Solver reuse is allowed
  only within one method's sequence of candidates.
- Timeouts, proven UNSAT results and verified recoveries are recorded
  separately. A timeout is never counted as a wrong candidate.
- Fresh-key campaigns had their protocol, predictions and analysis fixed and
  committed before the keys were generated. Each ran once, and every failure is
  kept.
- All campaign timings come from one laptop (Apple M5, 16 GB). Times on other
  machines will differ.

## Licence

All rights reserved. See [`LICENSE`](LICENSE). The literature review submitted for the course
is included as `Literature Review (Cryptanalysis).pdf`.
