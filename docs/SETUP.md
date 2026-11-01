# Development setup

These are the early development notes. To reproduce the results, use
[`REPRODUCE.md`](REPRODUCE.md) instead.

These instructions cover the source-verified scalar SPECK32/64 implementation,
the deterministic Dataset v1 foundation, the bounded CryptoMiniSat feasibility
pilot, the Phase 3 neural distinguisher baseline, the Phase 4 final-round
ranking baseline, and the bounded Phase 5 neural-SAT integration pilot.
Confidence gating and fallback selection remain later phases.

The supported interpreter is **64-bit CPython 3.11**. The Python version limit
is declared in `pyproject.toml`; the 64-bit requirement comes from the
available `pycryptosat` wheels.

## Installed packages

`requirements.txt` installs the test, lint, SAT, and Phase 3 neural dependencies:

- `pytest` for the test suite;
- `ruff` for linting;
- `pycryptosat` for the CryptoMiniSat Python interface;
- `numpy` for numerical support used by the training pipeline;
- `torch` for the compact Phase 3 distinguisher.

The project is then installed in editable mode.

## Windows 10/11

Install [Git for Windows](https://git-scm.com/download/win) and 64-bit
[CPython 3.11](https://www.python.org/downloads/release/python-3119/).
Select **Add python.exe to PATH** in the Python installer, or use the `py`
launcher below.

In PowerShell, clone and enter the repository:

```powershell
git clone https://github.com/ShakChunni/confidence-gated-neural-sat-speck.git
Set-Location confidence-gated-neural-sat-speck
```

Create and activate an isolated environment:

```powershell
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --editable .
```

If PowerShell blocks activation, allow it for the current process only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Verify the installation:

```powershell
python -c "import sys, pycryptosat; print(sys.version); print(pycryptosat.__file__)"
python -m pytest -q
python -m ruff check .
```

Run the pilots from the repository root. The root is included in
`PYTHONPATH` because the pilot reads its trusted test-vector fixture from
`tests/`:

```powershell
$env:PYTHONPATH = ".;src"
python scripts\phase1_data_pilot.py --sample-count 256 --rounds 7 --seed 1001 --split-role train
python scripts\phase2_sat_pilot.py --rounds 2 4 7 --pair-count 4 --timeout 10
python scripts\export_dataset_v1.py --classifier-config configs\data\phase3_classifier_r5.json --output-dir datasets\phase3_classifier_r5 --skip-attack --include-commit
python scripts\train_distinguisher.py --config configs\training\phase3_baseline.json
python scripts\export_dataset_v1.py --attack-config configs\data\phase4_attack_r6.json --output-dir datasets\phase4_attack_r6 --skip-classifier --include-commit
python scripts\rank_subkeys.py --config configs\experiments\phase4_ranking_r6.json
python scripts\plot_phase4_figures.py
python scripts\run_phase5_integration.py --config configs\experiments\phase5_integration_r6.json
python scripts\summarize_phase5.py
python scripts\export_dataset_v1.py --attack-config configs\data\phase6_confidence_attack_r6.json --output-dir datasets\phase6_confidence_r6 --skip-classifier --include-commit
python scripts\rank_subkeys.py --config configs\experiments\phase6_confidence_ranking_r6.json --split-role validation --instance-index 0 --output results\phase6_ranking_r6_validation_0000.json
python scripts\evaluate_phase6_confidence.py --config configs\experiments\phase6_confidence_r6.json
python scripts\plot_phase6_confidence.py
python scripts\export_dataset_v1.py --attack-config configs\data\phase6_pair64_attack_r6.json --output-dir datasets\phase6_pair64_r6 --skip-classifier --include-commit
python scripts\rank_subkeys.py --config configs\experiments\phase6_pair64_ranking_r6.json --split-role validation --instance-index 0 --output results\phase6_pair64_ranking_r6_validation_0000.json
python scripts\rank_subkeys.py --config configs\experiments\phase6_pair64_ranking_r6.json --pair-count 32 --split-role validation --instance-index 0 --output results\phase6_pair64_prefix32_ranking_r6_validation_0000.json
python scripts\compare_pair_counts.py
```

The `PYTHONPATH` setting lasts only for the current PowerShell session.

## macOS

Install [Git](https://git-scm.com/download/mac) and CPython 3.11 from the
[official macOS downloads](https://www.python.org/downloads/release/python-3119/).
Homebrew is also supported:

```bash
brew install python@3.11
```

Use the native interpreter for the machine architecture. Check it first:

```bash
python3.11 --version
```

In Terminal, clone and install:

```bash
git clone https://github.com/ShakChunni/confidence-gated-neural-sat-speck.git
cd confidence-gated-neural-sat-speck
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --editable .
```

Verify the installation and run the checks:

```bash
python -c 'import sys, pycryptosat; print(sys.version); print(pycryptosat.__file__)'
python -m pytest -q
python -m ruff check .
```

Run the pilots:

```bash
export PYTHONPATH=.:src
python scripts/phase1_data_pilot.py --sample-count 256 --rounds 7 --seed 1001 --split-role train
python scripts/phase2_sat_pilot.py --rounds 2 4 7 --pair-count 4 --timeout 10
python scripts/export_dataset_v1.py --classifier-config configs/data/phase3_classifier_r5.json --output-dir datasets/phase3_classifier_r5 --skip-attack --include-commit
python scripts/train_distinguisher.py --config configs/training/phase3_baseline.json
python scripts/export_dataset_v1.py --attack-config configs/data/phase4_attack_r6.json --output-dir datasets/phase4_attack_r6 --skip-classifier --include-commit
python scripts/rank_subkeys.py --config configs/experiments/phase4_ranking_r6.json
python scripts/plot_phase4_figures.py
python scripts/run_phase5_integration.py --config configs/experiments/phase5_integration_r6.json
python scripts/summarize_phase5.py
python scripts/export_dataset_v1.py --attack-config configs/data/phase6_confidence_attack_r6.json --output-dir datasets/phase6_confidence_r6 --skip-classifier --include-commit
python scripts/rank_subkeys.py --config configs/experiments/phase6_confidence_ranking_r6.json --split-role validation --instance-index 0 --output results/phase6_ranking_r6_validation_0000.json
python scripts/evaluate_phase6_confidence.py --config configs/experiments/phase6_confidence_r6.json
python scripts/plot_phase6_confidence.py
python scripts/export_dataset_v1.py --attack-config configs/data/phase6_pair64_attack_r6.json --output-dir datasets/phase6_pair64_r6 --skip-classifier --include-commit
python scripts/rank_subkeys.py --config configs/experiments/phase6_pair64_ranking_r6.json --split-role validation --instance-index 0 --output results/phase6_pair64_ranking_r6_validation_0000.json
python scripts/rank_subkeys.py --config configs/experiments/phase6_pair64_ranking_r6.json --pair-count 32 --split-role validation --instance-index 0 --output results/phase6_pair64_prefix32_ranking_r6_validation_0000.json
python scripts/compare_pair_counts.py
```

The `PYTHONPATH` setting lasts only for the current Terminal session.

## CryptoMiniSat notes

The normal route is the pinned `pycryptosat==5.14.7` wheel in
`requirements.txt`; the Python model does not require a separate
`cryptominisat5` executable. The current PyPI release publishes CPython 3.11
wheels for Windows x64, macOS arm64, and macOS x86_64. Check the
[pycryptosat release files](https://pypi.org/project/pycryptosat/5.14.7/)
if platform availability changes.

If pip reports that no compatible wheel exists:

1. Confirm that the active interpreter is 64-bit CPython 3.11.
2. Upgrade pip inside the activated environment.
3. Retry `python -m pip install -r requirements.txt`.
4. If a source build is genuinely required, follow the
   [official CryptoMiniSat build instructions](https://github.com/msoos/cryptominisat).
   The official instructions require CMake and GMP on macOS. On Windows they
   use MSYS2 and the MINGW64 toolchain; this is an advanced fallback, not the
   normal project setup.

Do not replace CryptoMiniSat with another SAT solver without recording and
reviewing that project decision.

## Common problems

### `python` is the wrong version

Activate `.venv` and run `python --version`. It must report Python 3.11.x.
Recreate the environment with `py -3.11` on Windows or `python3.11` on macOS
if necessary.

### `ModuleNotFoundError: neural_sat_speck`

The editable install was skipped or the virtual environment is not active:

```text
python -m pip install --editable .
```

### `ModuleNotFoundError: tests` when running the pilot

Run the command from the repository root and set the platform-specific
`PYTHONPATH` shown above (`.;src` on Windows, `.:src` on macOS).

### The test count differs

Run the tests from the repository root in the activated environment. The test
count is intentionally not fixed in this document; use the command output as
the source of truth.

## Reproducibility guidance

Record the operating system, CPU architecture, Python version, dependency
versions, repository commit, and pilot configuration with every experiment.
Do not commit `.venv`, large generated datasets, checkpoints, or results; the
repository `.gitignore` is configured for these local artefacts. The tiny
Dataset v1 fixtures under `tests/fixtures/dataset_v1/` are inspectable
exceptions; regenerate them with `scripts/export_dataset_v1.py` and see
[DATASETS.md](DATASETS.md).

The Phase 3 baseline export is reproducible but deliberately not committed.
Its protocol configuration is
[`configs/data/phase3_classifier_r5.json`](../configs/data/phase3_classifier_r5.json),
and its training configuration is
[`configs/training/phase3_baseline.json`](../configs/training/phase3_baseline.json).
The exporter writes a manifest and three checksummed shards. The training
script verifies those files before training and evaluates the test split only
after validation-based model selection.

The Phase 5 configuration is
[`configs/experiments/phase5_integration_r6.json`](../configs/experiments/phase5_integration_r6.json).
It consumes a complete Phase 4 ranking result, uses the same 32-pair validation
instance for all methods, and keeps its JSON result under the ignored
`results/` directory. The integration runner does not tune a confidence
threshold, modify the SAT model between methods, or treat a timeout as UNSAT.

The Phase 6 commands use a separately seeded 16-instance validation benchmark.
Repeat the ranking command for each validation index, changing both
`--instance-index` and the output filename, before running the confidence
evaluator. The evaluator refuses non-validation ranking results and does not
select a threshold or fallback.

## Phase 6A/6B pair-aware ranker study

The pair-aware study adds a separate group-data format and two validation-only
attack roles. It does not change Dataset v1, B0, or the held-out test split.
The source basis is Gohr's single-pair rank-and-sum baseline, Chen et al.'s
multi-pair derived-feature distinguisher, and the permutation-invariant Deep
Sets construction.

From the repository root, export and verify the group data and both validation
roles:

```text
PYTHONPATH=.:src python scripts/export_phase6a_group_dataset.py --config configs/data/phase6a_pair_group_r5_g64.json --include-commit
PYTHONPATH=.:src python scripts/export_phase6a_attack_validation.py --config configs/data/phase6a_selection_attack_r6.json --include-commit
PYTHONPATH=.:src python scripts/export_phase6a_attack_validation.py --config configs/data/phase6a_confirmation_attack_r6.json --include-commit
PYTHONPATH=.:src python scripts/phase6a_feasibility.py
```

Train B1, B2, and B3 from the frozen Phase 3 B0 encoder:

```text
PYTHONPATH=.:src python scripts/train_phase6a_rankers.py --config configs/training/phase6a_pair_group_rankers.json
```

Run complete-domain rankings for every method in the selection role. For the
confirmation role, run only `b0` and the method selected from the selection
role. Thus the frozen study has 128 selection jobs and 64 confirmation jobs,
192 complete-domain rankings in total. The following is one example:

```text
PYTHONPATH=.:src python scripts/rank_phase6a_candidates.py --config configs/experiments/phase6a_ranking.json --method b0 --role selection --instance-index 0
```

The reference PyTorch path is the default. An opt-in CPU NumPy path is
available for timing experiments:

```text
PYTHONPATH=.:src python scripts/rank_phase6a_candidates.py --config configs/experiments/phase6a_ranking.json --method b1_mean --role selection --instance-index 0 --inference-backend numpy
```

It uses the same checkpoint weights but is only an execution optimisation.
Compare its logits and candidate order with the reference path before using
it for a reported result; the PyTorch path remains authoritative if ordering
differs.

Omit `--candidate-count` for the full 65,536-candidate domain. The prefix
option is only for feasibility checks. Once the 192 ranking result files are
present, summarize them with:

```text
PYTHONPATH=.:src python scripts/summarize_phase6a.py --config configs/experiments/phase6a_ranking.json
```

The summary uses fixed top-16 and median-rank rules to select a candidate from
the selection role and checks it independently on confirmation. It does not
open a held-out test split or select a confidence fallback.

The recorded Phase 6A outcome selected B1 on the selection role, but B1 did
not pass the confirmation median-rank condition. This is retained as a
validation result; it does not promote B1 to the preferred ranker or open the
SAT pilot. Large datasets, checkpoints, and ranking JSON files remain local
and ignored by Git.
