# Development setup

These instructions cover the source-verified scalar SPECK32/64 implementation,
the deterministic Dataset v1 foundation, the bounded CryptoMiniSat feasibility
pilot, the Phase 3 neural distinguisher baseline, and the Phase 4 final-round
ranking baseline. Confidence gating and neural-SAT integration are later
phases and are not included here.

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
