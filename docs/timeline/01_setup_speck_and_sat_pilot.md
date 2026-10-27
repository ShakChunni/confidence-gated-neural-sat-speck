# 01. SPECK reference, data generator and first SAT pilot

**Dates:** 6 to 13 August 2026

## What we set out to do

Before any attack, we needed a cipher we could trust. Cryptographic code can
run without errors and still be wrong. A wrong rotation or a swapped word order
gives plausible output and silently ruins every later result. So the first
stage built a slow, clear reference implementation of SPECK32/64 and a
deterministic data generator. It also checked whether CryptoMiniSat can solve
reduced-round SPECK at all.

## What we built

- **Scalar SPECK32/64.** 16-bit words, rotations by 7 and 2, addition modulo
  2^16 and the 22-round key schedule. The master key is four words in the order
  (l2, l1, l0, k0), and a block is (x, y), as in the original specification
  and Gohr's code. We did not choose a byte layout, because the SPECK32/64
  source does not define one.
- **Tests first.** Rotation, masking, wraparound, one-round inverse, full round
  trip, reduced-round prefixes and the published known-answer vector all had
  to pass before any data was generated.
- **Data generator.** Gohr's real-versus-random task with input difference
  (0x0040, 0x0000), a seeded generator, balanced labels and a checksum over
  every exported shard. Dataset v1 was frozen on 13 August as canonical JSON
  plus a manifest.
- **SAT model.** An exact Boolean model of SPECK in CryptoMiniSat: native XOR
  constraints for XOR and sum bits, exact carry clauses for addition, rotation
  as wiring, the full key schedule, and known plaintext and ciphertext pairs.

## What we found

With four known pairs and a 1 s limit, the solver recovered the key up to four
rounds. Five rounds needed about 1.2 s. Six and seven rounds timed out.

| Rounds | Variables | Result at 1 s |
|---:|---:|---|
| 4 | 1,408 | key recovered in 0.02 s |
| 5 | 1,728 | timeout (solved in 1.17 s with 5 s) |
| 6 | 2,048 | timeout, also at 5 s |
| 7 | 2,368 | timeout |

Two smaller findings mattered later. First, with one or two pairs the solver
returned a key that encrypted those pairs correctly but was not the real key.
Consistency is not uniqueness, so every later recovery is checked on fresh
pairs. Second, a timeout is not a proof that a candidate is wrong. We kept
"timeout" and "UNSAT" as separate outcomes from this point on.

## What we decided and why

- Six rounds became the first attack target, because plain SAT could not do it
  and there was room for ranking to help.
- We would rank the last round key and give the solver only the best
  candidates, instead of asking it to solve the whole problem.
- We wrote our own data format rather than import a third-party dataset. Gohr's
  code had no licence file we could find, so we used his paper as the design
  source and wrote the code ourselves.

## Questions and answers

**Why write your own SPECK instead of using a library?**
We needed to control word order, round indexing and the key schedule exactly,
and to test each piece. The known-answer test checks the result against the
published vector.

**How do you know the SAT model is the same cipher as the Python code?**
Every solver model is decoded and re-encrypted with the Python reference. Tests
also fix a known key and check every intermediate state, and a deliberately
wrong ciphertext must give UNSAT.

**Why does a solver timeout not mean the key is wrong?**
The solver stopped before it finished the search. It proved nothing either way.
Treating timeouts as "wrong" would hide failures and inflate the success of any
method that tries more candidates.

**Why fresh verification pairs?**
With few pairs, many keys fit. A key that also reproduces pairs the attack never
used is very unlikely to be a coincidence. We still do not call verification
infallible.
