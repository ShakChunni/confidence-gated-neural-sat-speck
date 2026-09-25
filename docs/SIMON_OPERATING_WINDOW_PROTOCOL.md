# SIMON32/64 demo operating-window adjudication

**Status:** Predeclared engineering protocol; demo data only.

**Evidence class:** `ENGINEERING/DEVELOPMENT`, not a neural attack or prospective validation.
**Authority:** D-029 permits bounded SIMON reference and feasibility work. This document does not authorize protected data or neural training.

## Question and inputs

The two-key feasibility pilot at 6–8 rounds established a working reference/SAT path, but SAT without key assumptions already verified both demo keys at eight rounds with four known plaintext/ciphertext observations. We need a harder operating point before building a ranker. This study asks whether a bounded SIMON range contains a configuration where unconditioned SAT is difficult at a fixed cutoff while correct trailing-key assumptions remain cheap.

Use the tested SIMON32/64 reference, one fixed demo seed `999999`, eight independently generated master keys per round count, and the fixed demo plaintext difference `(0x0000, 0x0040)`. An observation is **one 32-bit plaintext/ciphertext mapping**; a differential pair contains **two** such observations. This input difference is a demo setting, not a frozen or source-validated attack difference.

## Grid and timing

- Tier 1: rounds `R = {8, 9, 10}`. If no Tier-1 cell passes the screen below, run the already-declared Tier 2: `R = {11, 12}`.
- In each tier: known observations `N = {4, 16}` (two or eight differential pairs); correct consecutive trailing-key assumptions `A = {0, 1, 2}`; residual depth `d = R - A`.
- One backend: in-process, single-threaded `pycryptosat`; one fresh SAT model per instance/cell. Solver time limit: 10 seconds per call. Record both solver status and monotonic wall time; the binding's time limit is not asserted to be a strict wall-clock interrupt.
- Record model-build wall time, solve wall time, fresh-pair verification wall time, total wall time, cutoff overshoot, variable/CNF/XOR counts, key/instance ID, and explicit SAT/UNSAT/timeout/unknown/exception/verification-failure status. A satisfying model is not a verified recovery unless the decoded key passes two fresh pairs.
- The `A > 0` branches are **oracle-conditioned engineering controls**. True trailing keys may be read only for these controls and post-hoc diagnostics, never as a candidate-selection procedure.

The eight keys are paired across all cells within each round count. All failures and timeouts remain in the denominator. The pilot reports observed SAT-model outcomes, not key uniqueness; a first model that fails fresh verification is recorded as such.

## Result-independent screen and candidate probe

A `(R, N, A)` cell with `A` in `{1, 2}` passes the first screen only if, on the same eight keys:

1. unconditioned `A=0` fresh-pair-verified recovery is at most `4/8` within its 10-second solver cutoff;
2. true-key-conditioned recovery is at least `7/8` within its 10-second solver cutoff; and
3. median conditioned solve wall time is at most 1 second, with timed-out cases retained as 10-second censored values.

Among passing cells, select the lexicographically first by `(R, A, N)`; prefer one assumed key before two at the same round count. On that cell only, test 32 distinct deterministic pseudo-random **wrong** trailing-key tuples per key. Exclude the true tuple, use an independent fresh model for each wrong candidate, cap each solve at 10 seconds, and stop an instance's probe at 180 seconds of total build-plus-solve wall time. Fresh models make this a conservative throughput diagnostic, not an incremental-SAT performance claim. A wrong-candidate probe passes if all 32 attempts finish within 180 seconds, at least 30 end in a resolved SAT/UNSAT state, and none yield a fresh-pair-verified key, on at least `7/8` keys. Record incomplete probes, unresolved solver timeouts, and all construction cost.

If the first selected cell fails the throughput probe, test the next passing cell in the same fixed order. If no Tier-1 cell passes both checks, execute Tier 2. If no cell passes by the end of Tier 2, report **no operating window found in this bounded grid** and do not train a SIMON ranker from this study. Passing the gate permits designing a separate classical/neural ranking study; it does not itself establish that either ranker has attack signal.

## Invariants, falsification, and deliverables

- The prior pilot JSON and all SPECK protected artifacts remain byte-identical.
- No protected SIMON seed, dataset, model training, or unseen outcome is used.
- The new runner refuses to overwrite an existing result; it writes atomically and records the source/configuration hash and environment.
- An independent reference-cipher check validates every recovered key. Incorrect SAT/UNSAT classification, false verification, or a mismatch to the scalar reference invalidates the gate.
- Unit tests cover the grid, observation semantics, selection order, timeout/censoring, wrong-tuple uniqueness, failure classification, and overwrite protection.
- Deliver the complete machine-readable grid, a small result summary and hash, exact regeneration command, and a go/no-go decision with limitations. Any backend comparison or neural claim requires a separate protocol.
