# Dataset v1

**Status: frozen.** The two-task split, MIT licence, word-level representation,
Gohr-style classifier protocol, and plaintext-difference attack construction
are the official Dataset v1 contract. Later official neural work must load
these shards. Do not regenerate ad hoc pairs for a reported baseline.

This repository packages its own synthetic SPECK32/64 data so different neural
architectures can be compared on the same examples. It is not a third-party
dataset import and it is not a claim that this project invented neural
cryptanalysis.

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

## Attack benchmark

Task identifier: `single_key_related_pairs`.

Schema:

- shard: `neural_sat_speck.attack_shard.v1`
- manifest: `neural_sat_speck.attack_manifest.v1`

Each instance has one secret master key and several related pairs. Dataset v1
uses the frozen plaintext-difference construction: every pair satisfies
`P1 = P0 XOR (0x0040, 0x0000)` and both blocks are encrypted under that
instance key. This is intentionally not Gohr's later first-round-difference
ranking arrangement. That ranking protocol is out of Dataset v1 and remains a
Phase 4 decision.

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
