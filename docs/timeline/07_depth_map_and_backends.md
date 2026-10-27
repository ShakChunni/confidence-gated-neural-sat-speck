# 07. Residual-depth map, solver backends and the formula package

**Dates:** 3 to 4 October 2026

## What we set out to do

Two questions came out of the seven-round work. How does recovery change with
round count and the number of fixed round keys, on both ciphers? And how much
of the result belongs to the solver and its settings rather than the ranker?

## Depth map

Eight demo keys per cipher, 300 s per cell, one Python-wrapper CryptoMiniSat
setting, and the true trailing round keys fixed.

- SPECK: (rounds, fixed keys) = (6,0), (6,1), (7,1), (7,2), (8,2), (8,3).
- SIMON: (9,0), (9,1), (10,1), (10,2), (11,2), (11,3).
- 4 or 128 observations, from one 64-pair family.

Of 192 cells, 151 recovered and 41 hit the deadline. The same unfixed depth did
not give the same outcome across the two ciphers, so a single "depth" number
does not summarise recovery.

During review we found that the first analysis had written zero for timing and
formula size where values were missing. That made some cells look cheap. The
raw files stayed unchanged, and a corrected analysis was saved as a new
version.

## Backend comparison

Four demo keys per cipher, SPECK at seven rounds and SIMON at ten, with one or
two true round keys fixed, and six solver profiles. Each entry is recoveries
out of 4.

| Profile | SPECK, 1 key | SIMON, 1 key | Both, 2 keys |
|---|---:|---:|---:|
| Python wrapper | 3 | 3 | 4 |
| Native defaults | 3 | 3 | 4 |
| Native, explicit XOR matrix settings | 4 | 4 | 4 |
| Same settings, matrix off | 2 | 0 | 4 |
| Same settings, pure CNF | 3 | 2 | 4 |
| CaDiCaL on the same CNF | 1 | 4 | 4 |

With two round keys fixed, every profile recovered every key. With one key
fixed, the solver profile mattered a lot, and no profile was best on both
ciphers. CaDiCaL was the weakest on SPECK and among the best on SIMON.

The run stopped once when the laptop came off AC power. That stop was a
problem with our setup, not a solver result. We resumed only the unstarted
cells, did not rerun the interrupted one, and report that group as three
completed trials plus one unavailable trial.

## Formula package

We exported the SPECK and SIMON formulas in native-XOR and pure CNF form, with
variable maps, assumptions, hashes and the expected verification. The package
was rebuilt from scratch in a clean Python environment, and all 64 formula
representations and their reference witnesses passed.

## What we decided and why

- No universal solver claim. Solver choice interacts with the cipher and with
  how much is fixed.
- Keep one backend inside each main comparison. Mixing solvers would confuse
  ranker effects with solver effects.
- Power and sleep checks became part of every later launch.

## Questions and answers

**Does the depth map show a law?**
No. It shows outcomes for these keys, this solver and this clock. The two
ciphers behaved differently at the same nominal depth.

**Why does fixing more round keys help so much?**
Each fixed 16-bit round key removes one round of unknowns. With two fixed, all
six profiles solved every case. With one, the remaining search is hard and
sensitive to solver settings.

**Why is only one interrupted cell excluded and not rerun?**
The protocol said no retries. Rerunning one cell after seeing others could
bias the result. We count it as unavailable and use three keys for any paired
contrast that involves it.
