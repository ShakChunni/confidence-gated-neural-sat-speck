# 02. Distinguisher, complete-domain ranking and the B1 ranker

**Dates:** 26 to 29 August 2026

## What we set out to do

Train a neural distinguisher, use it to rank all 65,536 candidates for the
last round key, and connect that ranking to the SAT solver. Then test whether
looking at many ciphertext pairs together ranks better than scoring each pair
on its own.

## What we built

- **Five-round distinguisher.** Trained on Dataset v1 to tell real pairs from
  random ones. Training used only verified frozen shards, chose the model on
  validation data, and reproduced exactly on a same-seed rerun.
- **B0 ranker.** For a six-round attack, try every 16-bit value of the last
  round key, undo one round on each of the 64 pairs, score the result with the
  five-round network and add the scores. This follows Gohr's `key_rank.py`.
  Ties break by candidate value, so the order is always the same.
- **SAT link.** The best candidates are passed to the solver one at a time as
  temporary assumptions on the last round key. The solver keeps what it learns
  within one method, and each candidate is removed after its call.
- **B1, B2 and B3.** Pair-aware rankers that encode each pair with the B0
  encoder and pool all 64 pair embeddings (mean, mean plus second moment, and
  learned weights). We fixed these three before looking at any attack result.

## What we found

The study was pre-registered with a selection set and a separate confirmation
set of 32 keys each.

| Set | B0 top-16 | B1 top-16 | B0 median rank | B1 median rank |
|---|---:|---:|---:|---:|
| Selection | 24/32 | 30/32 | 9.0 | 8.0 |
| Confirmation | 26/32 | 31/32 | 7.5 | 8.5 |

B1 won on top-16 both times. On confirmation, however, its median rank was
worse, and our rule needed both. So on 28 August we did **not** promote B1. B2
and B3 were weaker and were kept as negative results.

We then ran a second, larger pre-registered test on 64 new validation keys.
These count as development keys, because they decided the promotion.
B1 put the true key in its top 16 for 62 of 64 keys, against 53 for B0. It
also cut the mean number of capped attempts from 9.41 to 7.33. A small reranker
on top of B1 (H1) made attempts worse and failed its own gate. B1 was promoted
on 29 August as the preferred ranker, with B0 kept as the comparator.

We also checked that the fast NumPy ranking path agreed with the reference
PyTorch path. Both gave the same true ranks on all 64 keys.

## What we decided and why

- Keep B0 unchanged as the comparison point, so any B1 gain is measured against
  a fixed baseline.
- Describe B1 as a compact pair-aware adaptation. Multi-pair distinguishers
  and permutation-invariant pooling both exist in the literature. B1 is not the
  novelty of this project.
- Do not add more variants after seeing results. The family was B1, B2 and B3,
  and it stayed that way.

## Questions and answers

**Why rank all 65,536 candidates instead of sampling?**
The last round key is only 16 bits, so the complete domain is cheap enough. A
complete ranking also means the true key always has a well-defined position.

**Why did you not promote B1 after the first study?**
Our pre-registered rule needed both a top-16 gain and a better median rank on
the confirmation set. B1 failed the second part. We kept that result and ran a
new, fixed test rather than relax the rule.

**Is B1 a new architecture?**
No. It is a mean-pooled multi-pair version of the B0 encoder. Chen et al. and
Deep Sets already cover the ideas. We use it as one ranker among several.

**Does a good distinguisher accuracy mean a good attack?**
No. Classifier accuracy, key rank and verified recovery are different
measurements. This stage is where we started reporting them separately.
