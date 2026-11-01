# 05. Two-stage seven-round attack and the 128-key comparison

**Dates:** 10 to 25 September 2026

## What we set out to do

Find a way past the seven-round wall, then run one fair, pre-registered
comparison of neural, hybrid, random-order and SAT-only guidance at seven
rounds.

## What we learned about the solver

On opened keys we varied how many trailing round keys were fixed.

- Seven rounds with two true round keys fixed: 8/8 recovered, median 1.2 s.
  Eight rounds with two fixed failed, with three fixed it solved. Nine rounds
  with four fixed solved in about 1 ms. Under this setup, the solver finished
  when the unfixed depth was five rounds or less.
- The Python binding never used CryptoMiniSat's XOR (Gaussian elimination)
  engine. The XOR matrix was larger than the default column limit of 1,000.
  The binding's time limit also counts CPU seconds, not wall time.
- Exporting the same formula and running the native solver with a larger
  matrix limit solved 3 of 3 seven-round keys that had timed out, in about 48
  to 59 s. These were diagnostic runs, and their files are not in this
  repository.

So the RED gate was a limit of one configuration and budget, not proof that
seven rounds cannot be solved.

## The two-stage attack

Stage 1 ranks the last round key with a new six-round pair-group network.
Stage 2 ranks the round key before it, for each of the top Stage 1 candidates,
with the frozen five-round B1. Pairs of candidates go to the solver in order
of summed rank. With two round keys fixed, the solver only faces five rounds.

The first version took about 570 s per key, because Stage 2 ran nine complete
rankings. A **screened** Stage 2 fixed this: score all 65,536 candidates on 16
pairs, keep the best 512, and rescore those on all 64 pairs.

On 16 opened keys, with 32 Stage 1 and 32 Stage 2 candidates, the attack
recovered 13/16 with no solver timeouts. All three failures were Stage 1
misses.

## The audit that cut our claims back

On 12 September we audited our own records and found overclaims:

- the 13/16 run used the Python wrapper, not the native XOR setting we had
  credited;
- the reported 51x screening speed-up compared timings from two different
  runs;
- a G2-gated cascade, a plain wide search and a verification cascade all
  recovered 13/16, so routing gave no advantage.

The same day a **sparse classical ranker** built from sampled difference
statistics put the true six-round key in its top 4 on all 64 opened keys, in
0.57 s each. Our neural rankers did not do that. From then on we could not
claim neural ranking was better at six rounds, and the paper's question became
a fair cost comparison.

## Choosing the beam, then the main comparison

A development run on 64 new keys compared 32 x 32 against 64 x 16 candidates.
Both recovered 56/64. The 32 x 32 setting had lower PAR-2 (169 s against
213 s), so the pre-registered tie rule chose it.

The main comparison then ran once on 128 fresh keys, six methods, a 1,024
candidate budget and one 300 s clock per key that includes ranking.

| Method | Recovered of 128 | PAR-2 | Median time |
|---|---:|---:|---:|
| Neural Stage 1, classical Stage 2 (hybrid) | 111 | 143 s | 72 s |
| Neural Stage 1, screened neural Stage 2 | 110 | 178 s | 109 s |
| Neural Stage 1, full neural Stage 2 | 0 | 600 s | 308 s |
| Random order | 0 | 600 s | 96 s |
| SAT-only, 4 observations | 0 | 600 s | 302 s |
| SAT-only, all 128 observations | 0 | 600 s | 302 s |

Medians above 300 s come from the cooperative stop: a failed cell can run a
little past the budget before it stops. No counted recovery was late.

The primary test, screened neural against SAT-only, was 110 to 0 (exact
McNemar p = 1.5e-33). Screened neural and the hybrid differed by one key. The
hybrid was faster: by 100 s it had recovered 104 keys and screened neural 29.
The full neural Stage 2 never reached the solver, because ranking used the
whole 300 s. Stage 1 held the true key in its top 32 for 111 of 128 keys.

## What we decided and why

- Report the hybrid as **neural-classical**, not classical, because its Stage
  1 is neural.
- The seven-round result is frozen. No retuning on these 128 keys.
- On 17 September the scope grew: a second cipher (SIMON32/64), a depth map, a
  backend comparison and a reusable formula package. A negative result in any
  of them would still be reported.

## Questions and answers

**Why does a 110 to 0 win over SAT-only not prove neural ranking is good?**
It shows that fixing two round keys makes seven rounds solvable in time. The
hybrid with a classical Stage 2 did the same with fewer candidates and less
time. The useful comparison is between rankers, not ranking against no ranking.

**Why did the full neural method get zero?**
Ranking 32 Stage 2 domains on all 64 pairs is about 2.1 million network calls.
That took the whole 300 s clock, so no candidate reached the solver.

**What is the "residual depth" idea, and is it a law?**
It is the number of rounds left unfixed. In our solver setup, depth five or
less solved and more did not. That is an observation for one solver and
budget, not a general law, and we withdrew the word "law".

**Why 128 keys?**
A power calculation for a paired test with a 12.5 point difference gave about
82% power at 128 keys.
