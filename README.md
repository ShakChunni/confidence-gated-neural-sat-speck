# Confidence-Gated Neural-SAT SPECK

This repository supports the COMP6002 research project:

**Learning When to Trust Neural Guidance: Confidence-Gated Incremental XOR-SAT Key Recovery for Reduced-Round SPECK32/64**

The local `PLAN.md` is an untracked working-instructions file and is not part
of the public repository. The current implementation includes the bounded
Phase 5 incremental neural-SAT integration pilot and the Phase 6 validation-only
confidence-feature pilot, built on the
source-verified scalar, deterministic data, exact SAT, Phase 3 neural, and
Phase 4 ranking foundations. A frozen confidence threshold and fallback
selection remain later decisions.

The current baseline checkpoint is frozen at commit `b5597a4`. Future ranker
improvements must use new configurations and remain comparable with this
baseline; this freeze does not claim that the Phase 6 confidence-gate exit
criteria have been met.

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
PYTHONPATH=.:src python scripts/plot_phase4_figures.py
~~~

The Phase 4 configuration deliberately uses six-round attack instances with
the five-round Phase 3 classifier. Correct inverse removal of the final round
therefore leaves the five-round related state expected by the model. The
ranker only orders candidates; it does not call CryptoMiniSat, apply a
confidence gate, or recover a master key. A smaller candidate prefix can be
used for a quick pilot with `--candidate-count`; the declared run uses all
65,536 values.

The plotting command creates dependency-free SVG figures from the local
ranking JSON files. The corresponding local PNG previews are rendered with
the system SVG renderer and are not committed.

Run the bounded Phase 5 integration pilot after the Phase 4 ranking result has
been generated:

~~~text
PYTHONPATH=.:src python scripts/run_phase5_integration.py --config configs/experiments/phase5_integration_r6.json
PYTHONPATH=.:src python scripts/summarize_phase5.py
~~~

The pilot uses all 32 related pairs from one validation instance, a 32-candidate
budget, and a ten-second per-solve timeout. It compares the same exact model
under three candidate-order strategies and verifies any returned key against
fresh plaintext/ciphertext pairs. A time-limit-ambiguous solver result is
reported as inconclusive. The ranking time is included in the neural method's
online total, but the ranking is deliberately recorded as precomputed input to
this bounded integration result.

The summary command aggregates the local validation result files descriptively;
it does not choose a timeout, candidate budget, confidence threshold, or
fallback policy. The Phase 5 validation extension currently records 3/4
neural-order recoveries, 0/4 SAT-only recoveries, and 0/4 random-order
recoveries under the bounded pilot settings. These are validation observations
only, not final performance claims.

Run the first Phase 6 confidence-feature analysis on the separately generated
validation benchmark:

~~~text
PYTHONPATH=.:src python scripts/export_dataset_v1.py --attack-config configs/data/phase6_confidence_attack_r6.json --output-dir datasets/phase6_confidence_r6 --skip-classifier --include-commit
PYTHONPATH=.:src python scripts/evaluate_phase6_confidence.py --config configs/experiments/phase6_confidence_r6.json
PYTHONPATH=.:src python scripts/plot_phase6_confidence.py
~~~

The Phase 6 evaluator consumes complete Phase 4 ranking JSON files for the
validation split only. It uses a predeclared top-16 target and reports the
accepted-prefix risk--coverage behaviour of four post-hoc features; it does
not inspect the held-out test split, fit a threshold, choose a fallback, or
use SAT outcomes as a confidence signal. The ranking files for this pilot are
generated with the same five-round checkpoint and six-round attack protocol,
but use an independently seeded 16-instance validation benchmark.

In the current local pilot, 8/16 true final-round subkeys were in top-16.
The top-1/top-16 margin produced a two-instance zero-error accepted prefix;
top-1/top-2 margin produced one, and concentration/entropy produced none.
This is evidence that score sharpness is not automatically reliable
cryptanalytic confidence. No threshold or fallback has been selected from
this small pilot, and the held-out test split remains closed.

The repository also contains a separate source-aligned 64-pair pilot. Gohr's
published key-ranking experiment combines scores over 64 ciphertext pairs;
our paired check uses the same four keys with the first 32 pairs versus all 64.
The full 64-pair ranking improved the true rank in 3/4 cases (median 11.0 to
5.5), while top-16 recall remained 3/4. This is a small protocol observation,
not a final performance claim.

The paired analysis can be regenerated with
`scripts/compare_pair_counts.py` after producing both ranking prefixes. It is
kept separate from the frozen 32-pair Phase 4 baseline.

The next bounded study compares three new, separately identified pair-aware
rankers against frozen B0: B1 mean pooling, B2 mean-plus-variance pooling, and
B3 learned reliability-weighted pooling. The source basis is Gohr's ranking
baseline, Chen et al.'s multi-pair distinguisher, and Deep Sets. The group
dataset and attack roles are new validation-only artefacts; Dataset v1 and the
held-out test split remain untouched.

Generate the local group data and validation roles:

~~~text
PYTHONPATH=.:src python scripts/export_phase6a_group_dataset.py --config configs/data/phase6a_pair_group_r5_g64.json --include-commit
PYTHONPATH=.:src python scripts/export_phase6a_attack_validation.py --config configs/data/phase6a_selection_attack_r6.json --include-commit
PYTHONPATH=.:src python scripts/export_phase6a_attack_validation.py --config configs/data/phase6a_confirmation_attack_r6.json --include-commit
PYTHONPATH=.:src python scripts/phase6a_feasibility.py
~~~

Train the three candidates from the frozen B0 encoder:

~~~text
PYTHONPATH=.:src python scripts/train_phase6a_rankers.py --config configs/training/phase6a_pair_group_rankers.json
~~~

Run one complete-domain ranking job. Repeat for each method (`b0`, `b1_mean`,
`b2_moments`, `b3_weighted`), validation role, and index from 0 through 31:

~~~text
PYTHONPATH=.:src python scripts/rank_phase6a_candidates.py --config configs/experiments/phase6a_ranking.json --method b1_mean --role selection --instance-index 0
~~~

The optional `--candidate-count` is only a feasibility prefix. Omit it for the
declared 65,536-candidate domain. After all 256 ranking jobs are complete,
run `PYTHONPATH=.:src python scripts/summarize_phase6a.py`. The summary applies
the frozen selection and confirmation gates; it does not inspect the held-out
test split or implement confidence fallback.

The plotting command creates a local SVG risk--coverage figure (and an
optional local PNG preview can be rendered with `rsvg-convert`).

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
