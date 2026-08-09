# Confidence-Gated Neural-SAT SPECK

This repository supports the COMP6002 research project:

**Learning When to Trust Neural Guidance: Confidence-Gated Incremental XOR-SAT Key Recovery for Reduced-Round SPECK32/64**

The current implementation phase is the Phase 2 SAT-only feasibility pilot built
on the source-verified scalar and deterministic data foundation. Neural training,
confidence gating, full subkey ranking, and large dataset generation remain
deferred.

## Current implementation

- Source-backed SPECK32/64 research notes and representation decisions.
- A Python 3.11 src layout.
- Explicit 16-bit masking, rotation, and modular addition/subtraction primitives.
- Scalar key expansion and word-level encryption/decryption.
- Scalar final-round partial decryption.
- A deterministic Gohr-style baseline generator with machine-readable metadata.
- A bounded exact CryptoMiniSat model for reduced-round feasibility tests.
- Unit tests for primitives, key expansion, cipher, partial decryption, and data.

The current API is word-level: a block is represented as (x, y) and a master key
uses the word order recorded in docs/RESEARCH_NOTES.md. Byte serialization is not
fixed until the team reviews the documented source ambiguity.

## Development setup

Use Python 3.11. Complete Windows and macOS instructions are in
[docs/SETUP.md](docs/SETUP.md).

~~~text
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --editable .
~~~

Run the available checks:

~~~text
pytest
ruff check .
~~~

Run a small in-memory Phase 1 generation pilot:

~~~text
PYTHONPATH=src python scripts/phase1_data_pilot.py --sample-count 256 --rounds 7 --seed 1001 --split-role train
~~~

Run the Phase 2 SAT-only pilot:

~~~text
PYTHONPATH=.:src python scripts/phase2_sat_pilot.py --rounds 2 4 7 --pair-count 4 --timeout 10
~~~

The repository does not commit virtual environments, generated datasets, model
checkpoints, or experiment output. See the project instructions and
docs/RESEARCH_NOTES.md before selecting cipher or dataset parameters.
