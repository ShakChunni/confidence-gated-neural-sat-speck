# 03. Confidence gate, first comparison and the solver-state bug

**Dates:** 29 August to 4 September 2026

## What we set out to do

This was the original idea of the project. If the network could tell when its
own ranking was reliable, the attack could trust B1 when confident and fall
back to a wider or different search when not. We then wanted one held-out
comparison of all methods on six rounds.

## What we ran

**Scalar confidence gate.** We fixed one feature before fitting: the gap
between the 1st and 8th best scores, divided by the spread of all 65,536
scores. A two-parameter logistic fit turned it into a probability. The rule
was to accept the largest set of keys with at most 10% error and at least 16
accepted keys.

**Six-round comparison.** 64 held-out keys from the opened historical set, five methods: SAT-only, random
candidate order, B0, B1 and a B1/B0 fallback with 32 candidates.

## What went wrong

**The gate failed.** No threshold met the rule. The first point with 16
accepted keys already had 37.5% error. The feature barely separated good
rankings from bad ones (ROC-AUC about 0.52 for top-8 failure on the opened keys). We stopped there rather than try
more features on the same records.

**The first comparison had a bug.** All five methods shared one solver object.
When a neural method found the key, the solver kept learned clauses and
variable phases, and the next method started with that help. SAT-only looked
like 12/64. With a fresh solver per method it was **0/64**. Our first write-up
also said solving the correct candidate took under 100 ms. The measured median
was about 0.48 s. We corrected both.

**A blind seed leaked.** The seed planned for final confirmation had been
written into a CLI default and the README. We retired it, and another seed that
had been generated too early was limited to debugging.

## What we found (after the fix)

| Method (6 rounds, 64 keys, 10 s per call) | Verified recoveries |
|---|---:|
| SAT-only | 0 |
| Random order, 16 candidates | 0 |
| B0, 16 candidates | 50 |
| B1, 16 candidates | 58 |
| B1/B0 fallback, 32 candidates | 63 |

B1 against B0 is 8 keys apart, but the paired McNemar test gives p = 0.077. On
these keys, which we had already looked at, that is promising, not proven. The
fallback result used two full rankings and twice the candidates, so it is not a
fair win either.

## What we decided and why

- **Every compared method gets its own solver.** We built a solver factory that
  refuses to hand the same solver object to two methods, with tests. This rule
  holds for every later experiment.
- **Status codes.** The solver result now separates SAT, proven UNSAT and
  timeout, because the Python binding returns `None` on timeout.
- **Seed rules.** A small module rejects retired and quarantined seeds for any
  evaluation role. The demo seed became 999999.
- **A second try at reliability.** Instead of one scalar, G1 uses seven
  ranking-stability features (for example, how much the top 16 changes when
  the pairs are split). R1 reranks a short B1 list when G1 predicts failure.
  The full protocol was frozen on 3 September, before any new data existed.

## Questions and answers

**What exactly was the solver bug?**
One CryptoMiniSat instance was reused across methods. Learned clauses and
saved phases from a solved method made later methods faster. The fix gives
each method its own instance. Reuse is allowed only within one method's
sequence of candidates.

**Why not keep adding confidence features until one works?**
Repeated searching on the same 64 records would find something by chance. We
kept the failure and wrote a new protocol to test on new keys.

**Why is p = 0.077 not enough?**
It is above 0.05, and these keys were not fresh. We report the 8-key gap as
an observation and not as evidence that B1 is better.

**Is the 63/64 fallback the best method?**
No. It used more candidates and an extra ranking. Under an equal budget the
comparison is different, which is why later work fixed the budget first.
