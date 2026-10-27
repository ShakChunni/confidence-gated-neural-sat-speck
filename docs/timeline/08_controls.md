# 08. Controls: queue order, Stage 2, conditioning, ranking depth, encryption cost

**Dates:** 5 to 9 October 2026

## Why controls

By early October the main comparisons were done. Each one, however, changed
several things at once. These controls change one thing at a time, on opened
keys, so we can say where a difference comes from. They are diagnostics, not
new attack claims. Each one had a separate pilot that had to pass mechanical
checks before the full run. The pilot never used the success count as a gate.

## Queue order (5 October)

Same 1,024 candidates, ranked order against one fixed shuffle, eight keys per
cipher.

| Pipeline | Ranked | Shuffled |
|---|---:|---:|
| SPECK screened neural | 7/8 | 7/8 |
| SPECK neural-classical hybrid | 7/8 | 7/8 |
| SIMON classical | 7/8 | 6/8 |
| SIMON Bao hybrid | 6/8 | 7/8 |

Ranked order cut PAR-2 on SPECK and on SIMON classical, but the SIMON Bao hybrid
did worse ranked. In one SIMON case the true candidate was first in the queue
and still not recovered in time, while the shuffle found it. Position alone
did not decide recovery. The first attempt at this run failed on a TensorFlow
import problem before any useful result. We kept it, fixed the import, piloted
again and restarted from the beginning.

## Matched SIMON Stage 2 (5 to 6 October)

Same Stage 1 parents, with Bao's eight-round network against the classical
seven-round table for Stage 2. Both held the true pair for 2/8 keys with one
parent and 7/8 with eight parents. The projected median Stage 2 time was 115 s
and 500 s for the network, against 59 s for the table in both cases. Same
coverage, much lower cost for the table.

## Conditioning interface (6 October)

Fixing the true round key as permanent unit clauses, or as temporary
assumptions in each call. SPECK: 4/8 with clauses, 6/8 with assumptions. SIMON:
6/8 with clauses, 3/8 with assumptions. The two ciphers went opposite ways on
eight keys each, so we make no general claim.

## Backend by ranking depth (6 to 9 October)

Two depths (Stage 1 only, or Stage 1 plus Stage 2) crossed with two native
solver profiles (XOR matrix capacity 5, or 0). A first run used eight demo
keys. We then expanded to 32 development keys per cipher (256 cells).

| Cipher, matrix capacity | Stage 1 only | With Stage 2 |
|---|---:|---:|
| SPECK, 5 | 4/32 | 18/32 |
| SPECK, 0 | 2/32 | 19/32 |
| SIMON, 5 | 23/32 | 27/32 |
| SIMON, 0 | 3/32 | 22/32 |

Every arm held the true candidate on the same 27 of 32 keys per cipher. Yet
recovery ranged from 2 to 27. Stage 2 helps most when the solver is weak at
the remaining depth. This table is the clearest case of "containment is not
recovery" in the project.

One cell failed on a monitoring error in the 32-key run. With explicit
approval we retried that cell once and continued the 114 cells that had not
started, with the same settings. All records are kept, and dropping that key
does not change the finding.

## Encryption cost calibration (9 October)

To express 300 s in cipher terms we measured single-thread reference
encryption on this laptop. Full-round SPECK ran at 68.2 million blocks per
second and full-round SIMON at 29.7 million. So 300 s is about 2^34.3 SPECK or
2^33.1 SIMON encryptions here. This is a scale for the clock, not a count of
the work the solver does.

## What we decided and why

- Report containment and recovery side by side everywhere. They differ.
- The 32-key ranking-depth data became the development set for a cost model,
  which is the next stage.

## Questions and answers

**Why run controls on opened keys?**
They explain where differences come from. They do not make new claims about
attack strength, so they do not need fresh keys.

**Why was one cell retried when your rules say no retries?**
The failure came from our monitor, not the solver, and the retry was approved
in writing before it ran. We report it openly, and the result holds without that key.

**Why does the same containment give such different recovery?**
Recovery also depends on how long preparation takes and how long each wrong
candidate takes to reject. If rejection is slow, the solver may not reach the
true candidate before time runs out.
