# Confidence-Gated Neural-SAT SPECK

This repository supports the COMP6002 research project:

**Learning When to Trust Neural Guidance: Confidence-Gated Incremental XOR-SAT Key Recovery for Reduced-Round SPECK32/64**

The local `PLAN.md` is an untracked working-instructions file and is not part
of the public repository. The current implementation includes the full
cryptographic, dataset, SAT, neural, ranking, and prospective adjudication
foundations through Phase 6B, alongside the proposal figure generation suite
and the Phase 6C-A calibration/fallback checkpoint.

The pair-aware ranker (**B1**) is the promoted headline neural method following
prospective Phase 6B adjudication on 64 fresh validation keys; the Gohr-style
single-pair model (**B0**) serves as the frozen comparator. The held-out attack
test split remains strictly closed and unopened.

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
- A Phase 5 runner comparing SAT-only, seeded random-order, and neural-order
  candidate testing with temporary final-subkey assumptions and fresh-key
  verification.
- A Phase 6 post-hoc confidence layer with reproducible margins,
  score-concentration, entropy, and validation-only risk--coverage metrics.
- A Phase 6A group dataset and three separately identified pair-aware rankers
  (B1 mean pooling, B2 mean-plus-variance pooling, and B3 learned
  reliability-weighted pooling), with complete-domain ranking and a frozen
  selection/confirmation protocol.
- A Phase 6B prospective adjudication on 64 fresh validation keys, promoting B1
  over B0 (+14.06 pp top-16 recall, 22.10% lower mean attempts) and recording H1
  as an ablation.
- A publication and proposal figure generator producing 8 publication-grade
  diagrams in SVG, 300 DPI PNG, and vector PDF formats.
- A Phase 6C-A confidence calibration and diversity fallback module, recording a
  fail-closed threshold gate on initial validation calibration.
- Unit and integration tests covering cryptographic primitives, key schedule,
  cipher, partial decryption, datasets, SAT models, neural training, NumPy/PyTorch
  equivalence, full candidate ranking, calibration, and fallback logic.

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

Generate the local Phase 4 attack benchmark and rank every candidate for the
declared test instance:

~~~text
PYTHONPATH=.:src python scripts/export_dataset_v1.py --attack-config configs/data/phase4_attack_r6.json --output-dir datasets/phase4_attack_r6 --skip-classifier --include-commit
PYTHONPATH=.:src python scripts/rank_subkeys.py --config configs/experiments/phase4_ranking_r6.json
PYTHONPATH=.:src python scripts/plot_phase4_figures.py
~~~

Run the bounded Phase 5 integration pilot after the Phase 4 ranking result has
been generated:

~~~text
PYTHONPATH=.:src python scripts/run_phase5_integration.py --config configs/experiments/phase5_integration_r6.json
PYTHONPATH=.:src python scripts/summarize_phase5.py
~~~

Run the Phase 6B prospective adjudication comparison across 64 fresh validation keys:

~~~text
PYTHONPATH=.:src python scripts/run_phase6b_h1_adjudication.py --config configs/experiments/phase6b_h1_adjudication.json
~~~

Generate all 8 publication and proposal figures (SVG, 300 DPI PNG, and vector PDF):

~~~text
PYTHONPATH=.:src python scripts/plot_phase6b_proposal_figures.py
~~~

The 8 generated proposal figures in `results/figures/proposal/` are:
1. `methodology_pipeline`: Full confidence-gated neural–XOR-SAT framework.
2. `b1_pair_aware_architecture`: Pair-aware permutation-invariant scoring network (Deep Sets).
3. `b0_b1_prospective_results`: Prospective validation recall curves and metrics on 64 fresh keys.
4. `validation_protocol`: Leakage-resistant multi-stage development protocol.
5. `research_positioning`: Four-strand research taxonomy mapping.
6. `exact_recovery_verification`: Incremental CryptoMiniSat solving and fresh-pair verification.
7. `confidence_fallback_mechanism`: Confidence calibration and diversity fallback routing.
8. `evaluation_matrix`: Comparative evaluation matrix across attack configurations.

The baseline follows Gohr's published SPECK32/64 input representation and
residual-network design ([paper, Section 4](https://eprint.iacr.org/2019/037.pdf);
[supplementary implementation](https://github.com/agohr/deep_speck/blob/master/train_nets.py)).
This repository uses a compact four-block PyTorch network, BCE-with-logits
training, and explicit seeded data splits.

The repository does not commit virtual environments, large generated datasets,
model checkpoints, or experiment output. The MIT licence covers the code and
the synthetic Dataset v1 artefacts. See the project instructions before
selecting cipher or dataset parameters.
