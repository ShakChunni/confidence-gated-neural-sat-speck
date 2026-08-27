# Confidence-Gated Neural-SAT SPECK

This repository supports the COMP6002 research project:

**Learning When to Trust Neural Guidance: Confidence-Gated Incremental XOR-SAT Key Recovery for Reduced-Round SPECK32/64**

The local `PLAN.md` is an untracked working-instructions file and is not part
of the public repository. The current implementation includes the bounded
Phase 4 full-domain final-round subkey-ranking baseline, built on the
source-verified scalar, deterministic data, bounded SAT, and Phase 3 neural
foundations. Confidence gating and neural-SAT integration remain later phases.

## Current implementation

- Source-backed SPECK32/64 research notes and representation decisions.
- A Python 3.11 src layout.
- Explicit 16-bit masking, rotation, and modular addition/subtraction primitives.
- Scalar key expansion and word-level encryption/decryption.
- Scalar final-round partial decryption.
- A deterministic Gohr-style baseline generator with machine-readable metadata.
- Dataset v1 export: separate classifier and attack fixtures, manifests, and checksums.
- A bounded exact CryptoMiniSat model for reduced-round feasibility tests.
- A verified PyTorch Dataset v1-compatible loader and compact residual distinguisher.
- A committed Phase 3 configuration with checksummed train/validation/test inputs.
- A Phase 4 ranker for all 65,536 final-round subkey candidates with deterministic
  tie handling, score aggregation, timing, memory, and provenance metadata.
- Unit and integration tests for primitives, key expansion, cipher, partial decryption,
  data, SAT feasibility, neural training, checkpoint reload, and candidate ranking.

The current API is word-level: a block is represented as (x, y), and a master key
uses the explicit word order (l[2], l[1], l[0], k[0]). Byte serialization is not
fixed because the verified SPECK32/64 vector is word-oriented and the reviewed
implementation guide does not define a 32-bit-block byte API.

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

Regenerate the inspectable Dataset v1 fixtures:

~~~text
PYTHONPATH=src python scripts/export_dataset_v1.py
~~~

The classifier and attack formats are specified in
[docs/DATASETS.md](docs/DATASETS.md). Do not merge those two tasks.

Run a small in-memory Phase 1 generation pilot:

~~~text
PYTHONPATH=src python scripts/phase1_data_pilot.py --sample-count 256 --rounds 7 --seed 1001 --split-role train
~~~

Run the Phase 2 SAT-only pilot:

~~~text
PYTHONPATH=.:src python scripts/phase2_sat_pilot.py --rounds 2 4 7 --pair-count 4 --timeout 10
~~~

Generate the local Phase 3 baseline shards from the committed protocol and
then train from those verified shards:

~~~text
PYTHONPATH=.:src python scripts/export_dataset_v1.py --classifier-config configs/data/phase3_classifier_r5.json --output-dir datasets/phase3_classifier_r5 --skip-attack --include-commit
PYTHONPATH=.:src python scripts/train_distinguisher.py --config configs/training/phase3_baseline.json
~~~

The exporter and trainer keep large data, checkpoints, and summaries out of
Git. The training script does not call the generator; it verifies the manifest,
file hashes, content hashes, schema, shape, labels, and split metadata before
training, selects weights using validation AUC, and evaluates the held-out test
split once after that selection.

Generate the local Phase 4 attack benchmark and rank every candidate for the
declared test instance:

~~~text
PYTHONPATH=.:src python scripts/export_dataset_v1.py --attack-config configs/data/phase4_attack_r6.json --output-dir datasets/phase4_attack_r6 --skip-classifier --include-commit
PYTHONPATH=.:src python scripts/rank_subkeys.py --config configs/experiments/phase4_ranking_r6.json
~~~

The Phase 4 configuration deliberately uses six-round attack instances with
the five-round Phase 3 classifier. Correct inverse removal of the final round
therefore leaves the five-round related state expected by the model. The
ranker only orders candidates; it does not call CryptoMiniSat, apply a
confidence gate, or recover a master key. A smaller candidate prefix can be
used for a quick pilot with `--candidate-count`; the declared run uses all
65,536 values.

The baseline follows Gohr's published SPECK32/64 input representation and
residual-network design ([paper, Section 4](https://eprint.iacr.org/2019/037.pdf);
[supplementary implementation](https://github.com/agohr/deep_speck/blob/master/train_nets.py)).
This repository uses a smaller four-block PyTorch network, BCE-with-logits
training, and explicit seeded data splits; these are recorded implementation
choices, not claims that the original training results have been reproduced.

The repository does not commit virtual environments, large generated datasets,
model checkpoints, or experiment output. The MIT licence covers the code and
the synthetic Dataset v1 artefacts. See the project instructions before
selecting cipher or dataset parameters.
