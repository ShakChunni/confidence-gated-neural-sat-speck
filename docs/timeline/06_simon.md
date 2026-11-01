# 06. SIMON32/64: reference, feasibility, development and confirmation

**Dates:** 19 September to 3 October 2026

## What we set out to do

Check whether the SPECK findings carry over to a cipher with a different
round function. SIMON32/64 keeps 16-bit words and 16-bit round keys, so the
same attack shape works, but it uses AND and rotations instead of modular
addition. We chose SIMON before seeing any SIMON result, and agreed that we
would not swap to a third cipher if SIMON went badly.

## Getting a fair operating point

**Reference first.** A SIMON implementation with its key schedule, inverse
and known-answer tests, and a SAT model checked against it, came before any
experiment.

**Nine rounds was too easy.** A fixed grid on eight demo keys picked nine
rounds, four observations and two fixed round keys. At a 10 s solve limit,
plain SAT recovered 0/8, while the two-key oracle recovered 8/8. But when we
gave plain SAT the full 300 s clock, it recovered all **8/8**, with a median
of 50 s. Ranking cannot help where plain SAT already succeeds, so we did not
train a SIMON ranker for that setting.

**Ten rounds.** We moved to ten rounds with 64 pairs. Under 300 s, plain SAT
recovered 0/8 with both 4 and 64 observations, and the two-key oracle 8/8.
That leaves real room for ranking.

## Building a credible classical comparator

Bao et al. publish a Markov model for SIMON difference distributions, but
their code needs about 64 GiB of memory. We wrote an exact memory-bounded
version. It produced a complete six-round table (71 million non-zero entries)
and a complete seven-round table (2.17 billion entries) without sampling or
pruning. A checker streamed all the mass and compared 1,521 entries against
Bao's own transition. These are exact evaluations of Bao's model, not exact
cipher statistics.

For the neural side we used Bao's published SIMON networks. They loaded in
TensorFlow 2.16 after one safe change to an old layer, and every one of
449,457 weights matched the original files. We also trained a SIMON version
of our B1 ranker.

## Development comparison (32 opened keys)

| Method | Recovered of 32 |
|---|---:|
| Two-key oracle (diagnostic) | 30 |
| Classical table, both stages | 26 |
| Bao network Stage 1, classical Stage 2 | 22 |
| Bao networks, both stages | 9 |
| SIMON B1 hybrid | 0 |
| Random order, SAT-only | 0 |

## Confirmation on fresh keys

**Attempt 1 (30 September to 2 October).** 64 fresh keys, nine methods. It
finished with 75 recoveries in 576 cells, but we did not accept it. Three problems hit
the machine during the run:

- the restart helper ran the workers at macOS background priority, which
  throttles CPU;
- the laptop slept during the run;
- macOS cleaned `/private/tmp`, where the runtime lived, which broke one
  TensorFlow cell.

Median classical ranking time jumped from 19 s to 97 s after the restart. We
kept all 576 records but treated the run as compromised.

**Replacement (2 to 3 October).** A new seed, the same methods, persistent
directories and normal priority.

| Method | Recovered of 64 |
|---|---:|
| Two-key oracle (diagnostic) | 54 |
| Classical table | 53 |
| Bao hybrid | 49 |
| Bao fully neural | 26 |
| SIMON B1 hybrid, random, SAT-only | 0 |

The last row merges five arms, each of which recovered no key.

All 182 recoveries were checked independently. Both main methods beat SAT-only
after Holm correction. Classical against the Bao hybrid was 4 keys apart,
p = 0.34, so we cannot tell them apart. Our SIMON B1 never finished ranking
inside the clock on any key.

## What we decided and why

- The SIMON comparison closes here. We did not rerun to get a different
  answer.
- On SIMON at ten rounds, the classical table was at least as good as the
  published network, and far better than our own B1.
- System problems now count as threats to the result. Later runs record power,
  sleep, process priority and storage location.

## Questions and answers

**Why did you change from nine to ten rounds? Is that not cherry-picking?**
At nine rounds plain SAT recovered every key inside the clock, so the
comparison had no room for any ranker to help. We moved before any ranking
outcome existed. The ten-round choice came from SAT and oracle controls, not
from ranking results.

**Why throw away a confirmation run with 75 recoveries?**
We did not throw it away. It is kept and reported. We did not accept it as
clean evidence because the machine was throttled and interrupted, which changes
timing under a fixed clock.

**Is the classical table exact?**
It is an exact evaluation of Bao's Markov prediction. It is not the exact
distribution for a fixed key.

**Why did SIMON B1 get zero?**
It was too slow. Ranking did not finish within 300 s on any key, so the solver
never ran. That is a cost result, not an accuracy result.
