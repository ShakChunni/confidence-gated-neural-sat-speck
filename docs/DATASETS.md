# Dataset v1

**Status: frozen.** The two-task split, MIT licence, word-level representation,
Gohr-style classifier protocol, and plaintext-difference attack construction
are the official Dataset v1 contract. Later official neural work must load
these shards. Do not regenerate ad hoc pairs for a reported baseline.

This repository packages its own synthetic SPECK32/64 data so different neural
architectures can be compared on the same examples. It is not a third-party
dataset import and it is not a claim that this project invented neural
cryptanalysis.

## Interoperability profile

The benchmark payload is deliberately architecture-neutral. A classifier shard
contains ordinary JSON lists of binary features and labels; an attack shard
contains ordinary JSON word tuples, related pairs, and metadata. A consumer may
convert these values to NumPy, PyTorch, TensorFlow, or another tensor library
without changing the bit order or cryptographic data. This follows the current
direction of public multi-architecture frameworks such as
[NeuroDiffTorch](https://github.com/12xyzz/NeuroDiffTorch), while keeping this
repository's generator, checksums, and split policy independent of any one
framework.

The 2026 literature check did not find a single accepted frozen SPECK32/64
pair benchmark that can be imported as a common reference. Dataset v1 is
therefore the project's reproducible benchmark contract, not a claim that an
external model has already been reproduced here. The contract is frozen; any
new representation or negative-sample convention requires a new dataset
version and an explicit plan change.

Dataset v1 has two separate tasks. Do not mix them.

1. **Classifier benchmark** — related versus random ciphertext pairs.
2. **Attack benchmark** — several related pairs under one secret key.

Large generated files are not committed. The repository keeps the generator,
frozen configuration, checksums, and a tiny inspectable fixture.

## Why a fixed dataset

Gohr's 2019 SPECK32/64 work generated training data on the fly from
`/dev/urandom` (ePrint 2019/037, Section 4.3). Later neural-cryptanalysis
projects usually publish code or pretrained models, not a frozen pair set.
SAT4CryptoBench is an adjacent SAT-instance benchmark, not a substitute for
this neural pair dataset.

A seeded, checksummed export lets another model consume exactly the same
examples. That is the intended contribution of Dataset v1.

## Shared conventions

- Cipher: reduced-round SPECK32/64.
- Default rounds: 7.
- Input difference: `(0x0040, 0x0000)`.
- Word order: block `(x, y)`; master key `(l[2], l[1], l[0], k[0])`.
- Feature bit order, classifier only: first ciphertext `x` MSB to LSB, then
  `y`, then the second ciphertext `x` and `y`.
- Feature dtype: `uint8` bits in `{0, 1}`.
- Licence: MIT, covering the generator and the synthetic artefacts.
- Format version: `1.0.0`.

Byte serialization of SPECK32/64 remains intentionally unset. Dataset v1 uses
the documented word-level representation only.

## Classifier benchmark

Task identifier: `related_vs_random`.

Schema:

- shard: `neural_sat_speck.classifier_shard.v1`
- manifest: `neural_sat_speck.classifier_manifest.v1`

Each sample is one ciphertext pair and a binary label:

- label `1`: `P1 = P0 XOR (0x0040, 0x0000)`
- label `0`: a fresh random `P1`

Each sample has an independently generated 64-bit master key. Train,
validation, and test splits use distinct seeds and therefore disjoint keys.
The public shard contains features and labels only; master keys are not
exported.

The tiny committed fixture uses four samples per split:

| Split | Seed | Samples |
|---|---:|---:|
| train | 1001 | 4 |
| validation | 2001 | 4 |
| test | 3001 | 4 |

Configuration: [`configs/data/dataset_v1_classifier.json`](../configs/data/dataset_v1_classifier.json).
Fixture: [`tests/fixtures/dataset_v1/classifier/`](../tests/fixtures/dataset_v1/classifier/).

## Phase 3 baseline export

The canonical Dataset v1 fixture remains a seven-round reference. Phase 2
identified five rounds as the first measurable, bounded operating point, so
Phase 3 uses a separately named export with the same classifier protocol and
feature schema rather than changing the frozen fixture:

- dataset id: `speck32_64_classifier_phase3_r5`;
- rounds: `5`;
- samples: 20,000 train, 4,000 validation, and 4,000 test;
- seeds: 51,001, 52,001, and 53,001 respectively;
- configuration: [`configs/data/phase3_classifier_r5.json`](../configs/data/phase3_classifier_r5.json).

This is an explicit protocol-compatible Phase 3 baseline, not a third-party
dataset and not a key-ranking dataset. Run `scripts/export_dataset_v1.py`
with that configuration to create the local
`datasets/phase3_classifier_r5/classifier/` directory. The exporter records
the repository commit when `--include-commit` is used. The training script
requires the existing manifest and verifies all three shard hashes before it
starts; it never generates replacement samples.

Load a shard with the standard library only:

```python
import json
from pathlib import Path

shard = json.loads(Path("tests/fixtures/dataset_v1/classifier/test.json").read_text())
features = shard["features"]  # list[list[int]], shape (n, 64), values in {0, 1}
labels = shard["labels"]      # list[int], values in {0, 1}
```

The same JSON can be converted to NumPy, PyTorch, or TensorFlow tensors by the
caller. The repository does not hide a model-specific preprocess step.

## Phase 4 ranking benchmark

Phase 4 uses a separately named attack export so the frozen seven-round
Dataset v1 fixture is unchanged. Configuration:
[`configs/data/phase4_attack_r6.json`](../configs/data/phase4_attack_r6.json).
It has six-round plaintext-difference instances, 32 related pairs per
instance, and disjoint train/validation/test seeds. The test split contains
four independent attack keys. The six-round choice is paired with the
verified five-round Phase 3 classifier: a correct inverse final-round
candidate exposes a five-round related state.

The ranking experiment configuration is
[`configs/experiments/phase4_ranking_r6.json`](../configs/experiments/phase4_ranking_r6.json).
It enumerates every value from `0x0000` through `0xffff`, applies one inverse
round to both ciphertexts of every pair, encodes the resulting words in the
existing MSB-first layout, and sums raw classifier logits over pairs. Raw
logits preserve the ordering of summed log-odds. Ties are resolved by
ascending candidate value. The result JSON is local and ignored by Git.

Generate the local benchmark and run the ranking command from the repository
root:

```text
PYTHONPATH=.:src python scripts/export_dataset_v1.py --attack-config configs/data/phase4_attack_r6.json --output-dir datasets/phase4_attack_r6 --skip-classifier --include-commit
PYTHONPATH=.:src python scripts/rank_subkeys.py --config configs/experiments/phase4_ranking_r6.json
```

Phase 4 measures candidate-ranking behaviour. The bounded Phase 5 integration
consumes its complete-domain result; it does not alter the dataset or the
ranking scores.

## Phase 5 recovery pilot

The Phase 5 configuration is
[`configs/experiments/phase5_integration_r6.json`](../configs/experiments/phase5_integration_r6.json).
It uses all 32 related pairs from one validation instance as 64 exact
plaintext/ciphertext observations. SAT-only, seeded random-order, and
neural-order methods receive the same observations, reduced-round model,
candidate budget, timeout, and fresh verification pairs. The only intended
method difference is candidate order; the neural ranking time is carried from
the recorded Phase 4 result and labelled as precomputed input.

The runner records every candidate solve and treats a time-limit-ambiguous
false result as inconclusive. A SAT model is a recovery only after the decoded
master key matches the key schedule, all supplied observations, and fresh
independent plaintext/ciphertext pairs. The pilot result is local and ignored
by Git; it is an integration/correctness result, not a final speed claim.

The descriptive validation extension uses the same frozen settings on all four
validation instances. The neural order verifies 3/4 keys (the successful
candidate positions are 1, 19, and 1); one instance is timeout-inconclusive.
SAT-only verifies 0/4 because all four bounded solves time out, and random
order verifies 0/4 (three timeouts and one exhausted 32-candidate budget).
Run `scripts/summarize_phase5.py` to regenerate the local aggregate. These
figures are pilot evidence and must not be used to set the later confidence
threshold or final evaluation policy.

## Phase 6 confidence benchmark extension

The Phase 6 confidence pilot uses a separately named export so the Phase 4
ranking baseline remains immutable. Its configuration is
[`configs/data/phase6_confidence_attack_r6.json`](../configs/data/phase6_confidence_attack_r6.json).
It keeps the same six-round cipher, `(0x0040, 0x0000)` plaintext difference,
32 related pairs per key, word-level representation, and disjoint split policy,
but expands the validation split to 16 independently generated attack keys.
The train and test shards are generated for a complete manifest and remain
closed to this validation-only analysis.

The ranking configuration is
[`configs/experiments/phase6_confidence_ranking_r6.json`](../configs/experiments/phase6_confidence_ranking_r6.json)
and the evaluator configuration is
[`configs/experiments/phase6_confidence_r6.json`](../configs/experiments/phase6_confidence_r6.json).
It fixes `top_k = 16` before reading the validation outcomes and derives four
post-hoc features from each complete 65,536-candidate ranking:
top-1/top-2 margin, top-1/top-16 margin, stable top-score softmax
concentration, and normalized softmax entropy. It reports descriptive
risk--coverage curves, where risk is the fraction of accepted instances whose
true final-round subkey is outside top-16. No threshold or fallback is fitted
by this pilot. The held-out test split is not read by the evaluator.

Generate the local extension and then rank each validation instance with the
existing Phase 4 ranking runner before running the evaluator. The generated
shards, ranking outputs, and confidence result are local ignored artefacts.
The extension is a benchmark-generation configuration and reproducibility
boundary, not a claim that 16 validation keys are sufficient for final
statistical calibration.

On the current local run, 8/16 true final-round subkeys were in top-16. The
top-1/top-16 margin had a two-instance zero-error accepted prefix; top-1/top-2
margin had one, and concentration and entropy had none. This negative result
is retained as validation evidence that a sharp neural score distribution is
not by itself a reliable gate. No threshold, fallback, or held-out test result
is inferred from it.

The dependency-free figure helper
`scripts/plot_phase6_confidence.py` renders these local curves as an SVG for
inspection and later paper drafting. The figure is generated from the saved
validation result and is not a substitute for the raw ranking evidence.

### Pair-count protocol check

Gohr's published key-ranking experiment combines the neural scores over 64
ciphertext pairs ([paper, Section 5](https://eprint.iacr.org/2019/037.pdf)).
The separate configuration
[`configs/data/phase6_pair64_attack_r6.json`](../configs/data/phase6_pair64_attack_r6.json)
therefore creates four-key validation instances with 64 related pairs. The
same instances are ranked twice: once with the first 32 pairs using the
`--pair-count 32` pilot override, and once with all 64 pairs. The tracked
`scripts/compare_pair_counts.py` checks that the keys and pair prefixes match
before calculating the paired rank changes.

In the current local four-key check, all-64-pair ranking improved the true rank
in three cases and left one unchanged; median rank changed from 11.0 to 5.5.
Top-16 recall was 3/4 for both settings. This supports treating pair count as
an experimental variable and justifies a larger validation run, but it is not
enough to select the final pair budget or establish an attack advantage.

## Phase 6A/6B pair-group benchmark

The ranker-improvement study uses a new dataset id,
`speck32_64_pair_group_phase6a_r5_g64`; it does not modify Dataset v1. Its
shard schema is `neural_sat_speck.pair_group_shard.v1` and its manifest schema
is `neural_sat_speck.pair_group_manifest.v1`. Shards are compressed NumPy
arrays with an embedded metadata document. The feature array has shape
`(groups, 64, 4, 16)` and dtype `uint8`; labels have shape `(groups,)` and
dtype `uint8`. The master-key array is included only for local audit and key
separation checks.

Each positive group contains 64 pairs generated under one master key. Its
second plaintext is the first plaintext XOR `(0x0040, 0x0000)`, and both
blocks are encrypted for five rounds. Each negative group also uses one key
for all 64 pairs, but every second plaintext is independently random. Labels
are balanced, train and validation keys are disjoint, and the generator uses
an explicit NumPy `Generator` seeded per split. The five-round choice matches
the six-round ranking attack after inverse removal of one final round.

The frozen group configuration contains 4,096 training groups (seed 61001) and
1,024 validation groups (seed 62001). It is exported with
`scripts/export_phase6a_group_dataset.py` using
`configs/data/phase6a_pair_group_r5_g64.json`. Large `.npz` shards stay local
and ignored by Git; their content and file checksums are recorded in the
manifest.

The attack comparison uses two validation-only manifests, each with 32
independent six-round keys and 64 related pairs: selection seed 64001 and
confirmation seed 65001. The exporter creates no test shard or held-out test
material. Use `scripts/export_phase6a_attack_validation.py` with the two
corresponding configurations. B0, B1, B2, and B3 must consume the same
ciphertext pairs in each role.

The model identifiers are `speck32_64_b1_mean_g64`,
`speck32_64_b2_moments_g64`, and `speck32_64_b3_weighted_g64`. B1 is mean
pooling of shared B0 pair embeddings; B2 adds their population variance; B3
uses a normalized learned reliability-weighted mean. The hypotheses are
motivated by Chen et al.'s multi-pair derived-feature work and the
permutation-invariant construction in Deep Sets. They are controlled study
variants, not claims of architectural priority.

## Attack benchmark

Task identifier: `single_key_related_pairs`.

Schema:

- shard: `neural_sat_speck.attack_shard.v1`
- manifest: `neural_sat_speck.attack_manifest.v1`

Each instance has one secret master key and several related pairs. Dataset v1
uses the frozen plaintext-difference construction: every pair satisfies
`P1 = P0 XOR (0x0040, 0x0000)` and both blocks are encrypted under that
instance key. This is intentionally not Gohr's later first-round-difference
ranking arrangement. That arrangement remains documented as a separate source
protocol and is not silently mixed into either Dataset v1 or the Phase 4
baseline.

The fixture includes the synthetic master key and the true final-round subkey
so later ranking or SAT recovery can be verified. These keys are research
labels, not a secrecy challenge.

| Split | Seed | Instances | Pairs each |
|---|---:|---:|---:|
| train | 4001 | 2 | 2 |
| validation | 6001 | 2 | 2 |
| test | 5001 | 2 | 2 |

Configuration: [`configs/data/dataset_v1_attack.json`](../configs/data/dataset_v1_attack.json).
Fixture: [`tests/fixtures/dataset_v1/attack/`](../tests/fixtures/dataset_v1/attack/).

Words are four lowercase hexadecimal digits. Convert them with `int(word, 16)`.

## Regeneration

From the repository root, after the development setup in
[SETUP.md](SETUP.md):

```text
PYTHONPATH=src python scripts/export_dataset_v1.py
```

The command rewrites the inspectable fixture. It is not a large-scale
generator. Do not commit large local exports from `datasets/`.

Identical configuration, seed, and `generated_at` timestamp must reproduce the
same shard bytes and the same SHA-256 checksums. If `repository_commit` is
included, the same commit must also be used. The manifest records both a
content checksum over the cryptographic payload and a file checksum over the
canonical JSON.

## What Dataset v1 is not

- Not a full-cipher break.
- Not a replacement for SAT4CryptoBench.
- Not a published 10-million-sample training dump.
- Not a PyTorch or Keras dataset class.
- Not a freeze of Gohr's first-round-difference ranking protocol. That
  construction is deferred to Phase 4.

## Frozen contract

The following values are frozen for Dataset v1. Changing them requires a new
dataset version and a plan change-log entry.

| Item | Frozen value |
|---|---|
| Classifier dataset id | `speck32_64_classifier_v1` |
| Attack dataset id | `speck32_64_attack_v1` |
| Format version | `1.0.0` |
| Cipher | SPECK32/64 |
| Rounds | 7 |
| Input difference | `(0x0040, 0x0000)` |
| Classifier task | `related_vs_random` |
| Attack task | `single_key_related_pairs` |
| Attack pair construction | `plaintext_difference` |
| Classifier key policy | one independent key per sample |
| Attack key policy | one key per instance |
| Feature width / dtype | 64 binary `uint8` bits |
| Licence | MIT |
| Train / validation / test | disjoint seeds and keys |

Authoritative files: this document, the local `PLAN.md` section 1J,
`configs/data/dataset_v1_classifier.json`, and
`configs/data/dataset_v1_attack.json`. The Phase 3 variant is defined by
`configs/data/phase3_classifier_r5.json` and does not modify the canonical
Dataset v1 fixture.
