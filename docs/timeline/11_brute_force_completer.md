# 11. SAT against a brute-force completer

**Dates:** 11 to 12 October 2026

## What we set out to do

Feedback on the draft asked a fair question. Once the attack has fixed some
round keys, is a SAT solver even needed? The remaining key schedule bits could
be enumerated directly. We also wanted to test the cost rule in a stronger
way: let it choose the method before the keys exist, then check the choice.

## What we built

- **Brute-force completer.** A native single-thread C program. Given m fixed
  trailing round keys, it tries every value of the remaining 16(4 - m) schedule
  bits, inverts the key schedule to a master key and tests the known blocks.
  It is built on the same verified reference schedules, with known-answer and
  inversion round-trip tests.
- **Per-candidate costs.** We measured rejection and completion cost for SPECK
  and SIMON at one, two and three fixed keys, and also expressed them in
  reference-encryption units (stage 08).

## What the costs said

- With 32 or 48 unknown bits left, SAT is much faster per candidate, by a
  factor of about 2^1.8 to 2^17 depending on the case.
- With 16 unknown bits left (eight-round SPECK), brute force is about 2^7.7
  times faster than SAT. Rejecting a wrong candidate took 1.3 ms.

So the cost rule picks **SAT at seven rounds** and **brute force at eight**.

## Prescriptive test (32 fresh keys, 8 arms)

We froze the design, the rule's choices, the comparison arms, the predictions
and the analysis script before the fresh seed was generated. A pilot on demo
keys came first.

- **Seven rounds (rule chose SAT).** SAT finished ahead of brute force on 28
  keys and behind on none (one-sided p = 3.7e-9; other files use two-sided values, and this would be 7.5e-9). This held for all three Stage 1
  rankers after Holm correction.
- **Eight rounds (rule chose brute force).** Both recovered the same 18 keys.
  Brute force was cheaper on 17 and dearer on 1, by 0.13 s (one-sided p = 7.2e-5; two-sided 1.4e-4).
- **Ranker order.** The size of the SAT advantage at seven rounds followed the
  predicted order B1, then the table, then Gohr's network (23, 16 and 6 keys of 32
per ranker; the 28 above is the pooled comparison).
  This fits the rule. B1 puts the true candidate deepest in the queue, so slow
  brute-force rejections (61 s each at seven rounds) cost it most.

One part failed. Only 5 of 8 arm counts were inside their predicted ranges.
B1 at seven rounds recovered 26, above its range of 14 to 24. The two
eight-round network arms recovered 18, below their range of 22 to 29, because
only 18 of 32 true chains reached the queue. Using the observed preparation
times and queue positions, the rule matches 255 of 256 cells. That check came
after the run, and we label it as such.

## What we decided and why

- The completer choice is a cost question, and the rule answers it correctly
  in advance.
- Count predictions are only as good as the containment estimate behind them.
  We report the failed calibration as a failure.

## Questions and answers

**Does brute force make SAT pointless?**
Only when few bits remain. With 16 unknown bits, enumeration is faster. With
32 or more, SAT is faster by a wide margin.

**Why did the count predictions miss?**
They assumed more true chains would reach the queue than did. The rule itself
was right about which completer wins. The containment input was off.

**What does "prescriptive" mean here?**
The rule chose the method from development measurements before any fresh key
existed. The fresh run then checked whether that choice was right.

**Is 255 of 256 a result?**
It is a post hoc consistency check using observed values. We report it, but
not as a prediction.
