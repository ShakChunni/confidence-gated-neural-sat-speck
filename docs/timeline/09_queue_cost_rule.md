# 09. The queue-cost rule and its fresh-key test

**Dates:** 9 to 10 October 2026

## What we set out to do

The 32-key ranking-depth study showed the same containment giving very
different recovery. We wanted a simple explanation that could also predict
new runs.

## The rule

A key is recovered when the solver reaches the correct candidate before the
clock runs out. That depends on three measured things:

1. **preparation time**, before the first solver call;
2. **queue position** of the correct candidate;
3. **rejection time** for each wrong candidate, plus the time to solve the
   correct one.

If preparation plus the rejections ahead of the correct candidate plus its
solve time fits inside 300 s, the rule predicts recovery. We fixed and hashed
the rule before looking at any per-attempt solver times.

## Retrospective result (32 development keys per cipher)

Solver times for each key came from the other 31 keys. The rule predicted the
outcome correctly in **242 of 256 cells (94.5%)**. "Recovered if contained"
got 61.7%, and "each arm's usual outcome" got 76.6%. The predicted count was
within 3 of the actual count in all eight arms.

The rule also explains the table in stage 08. With one key fixed on SPECK, a
wrong candidate took a median 125 s to reject, so the solver could reject only
one or two before time ran out. With two keys fixed, rejection took 1 to 2 s,
and the solver could work through a long queue.

## Fresh-key test (32 new keys per cipher)

Before drawing the keys, we committed the predictions: a count and a 90%
interval for each of eight arms. Four of the arms added a per-candidate cap
(1.5 s on SPECK, 10 s on SIMON). The rule predicted that the same cap would
hurt opposite matrix settings on the two ciphers.

- **All 8 arm counts fell inside their intervals.** The rule needed at least 7.
  One arm (SPECK, matrix on, with the cap) was at the upper edge: 5 against an
  expected 1.
- **Both direction predictions held.** On SPECK the capped arm with the matrix
  off beat the matrix on by 12 keys. On SIMON the opposite setting won, also by
  12. Both survive Holm correction.
- Cell accuracy on the conditional test was 0.90, against 0.78 for "recovered
  if contained".

The run stopped at cell 300 of 320 on a file-check race in our coordinator. We
kept the 299 finished cells, fixed the race, piloted again, and ran the 21
missing cells with the same keys and settings. The decision to continue did not
use any outcome.

## Classical SPECK Stage 1

The same run included two new arms with a classical SPECK Stage 1, a sparse
six-round table built by sampling. An exact table needed about 64 GiB to
build, more than this 16 GiB laptop has. Under the same classical Stage 2:

| Profile | Classical Stage 1 | Neural Stage 1 (B1) | Holm p |
|---|---:|---:|---:|
| Matrix on | 30/32 | 21/32 | 0.022 |
| Matrix off | 31/32 | 21/32 | 0.013 |

The classical table held the true key in its top 32 for 32/32 keys, against
24/32 for the neural Stage 1, and was about three times cheaper to prepare
(20 s against 61 s).

## What we decided and why

- The rule became the paper's central tool. It explains recovery. It is not an
  attack tool, because it uses the correct candidate's position, which an
  attacker does not know.
- Our own B1 lost to a classical table at seven rounds. That led us to ask
  whether a stronger, published network would do better, which is the next
  stage.

## Questions and answers

**Is 94.5% a fair number if you built the rule on the same data?**
It is retrospective, and we say so. That is why we ran the fresh-key test with
committed predictions. The rule passed there too.

**Why is a cap useful in a test of the rule?**
A cap changes rejection time in a known way. If the rule is right, it should
predict which setting gets hurt. It did, in opposite directions on the two
ciphers.

**Does the classical win mean neural ranking is worse?**
It means our B1 was worse here, on SPECK at seven rounds, on this laptop.
It says nothing about other networks, rounds or hardware.

**Is the sparse table exact or optimal?**
Neither. It is sampled, and entries seen fewer than 20 times were dropped to
fit in memory.
