# Confidence-Gated Neural-SAT SPECK

This repository supports the COMP6002 research project:

**Learning When to Trust Neural Guidance: Confidence-Gated Incremental XOR-SAT Key Recovery for Reduced-Round SPECK32/64**

The current implementation phase is the source-verified scalar foundation. Neural
training, SAT encoding, confidence gating, full subkey ranking, and large dataset
generation are intentionally deferred.

## Current implementation

- Source-backed SPECK32/64 research notes and representation decisions.
- A Python 3.11 src layout.
- Explicit 16-bit masking, rotation, and modular addition/subtraction primitives.
- Unit tests for the primitive layer.

The current API is word-level: a block is represented as (x, y) and a master key
will use the word order recorded in docs/RESEARCH_NOTES.md. Byte serialization is
not fixed until the team reviews the documented source ambiguity.

## Development setup

Use Python 3.11:

~~~text
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
~~~

Run the available checks:

~~~text
pytest
ruff check .
~~~

The repository does not commit virtual environments, generated datasets, model
checkpoints, or experiment output. See the project instructions and
docs/RESEARCH_NOTES.md before selecting cipher or dataset parameters.
