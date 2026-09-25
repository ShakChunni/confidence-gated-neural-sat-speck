# SIMON nine-round SAT-only 300-second control

**Status:** Fixed demo-only sensitivity protocol, before execution. **Evidence class:** ENGINEERING. This control does not authorize SIMON model training or protected evaluation.

## Question

The earlier SIMON operating-window study selected nine rounds, four known plaintext/ciphertext observations, and two correct trailing-key assumptions. Unguided SAT timed out on all eight demo keys at a ten-second *solve-call* limit. Our intended attack comparison has a 300-second *complete online* limit. We need to know what unguided SAT actually does at that limit before investing in key ranking.

## Fixed run

Use the same eight deterministic demo instances from seed `999999`, nine rounds, input difference `(0x0000, 0x0040)`, and the first four individual known plaintext/ciphertext observations from each instance. Four observations are two differential pairs. For each key, build a fresh single-threaded in-memory `pycryptosat` model; pass **no** round-key or master-key assumptions; make exactly one solve call; decode any SAT model and check it on the two held-out verification observations. Run all eight keys in index order regardless of interim results. Do not change the solver, cipher, formula, pair subset, or key family during the run.

Start a monotonic 300.0-second clock before launching each isolated worker. The clock includes worker startup, model construction, solving, decoding, and verification. Pass the remaining budget to the solver and impose a parent-process wall-clock watchdog at 300.0 seconds because the solver's own time limit is not a strict wall-clock interrupt. Count a recovery only if fresh verification finishes before the complete 300-second deadline. Record SAT, unresolved solver limit, proven UNSAT, verification failure, exception, and parent watchdog separately; a finite-time-limit `False` result near the limit is unresolved, not proven UNSAT. If an unconditioned formula is proven UNSAT, investigate it as an implementation inconsistency: the previously tested correct-key-conditioned formula is satisfiable on these same instances.

The primary descriptive result is verified recoveries out of eight inside 300 seconds, with individual online wall times and censored failures. No significance test or population claim is justified by these eight opened demo keys. If at least five keys recover, the earlier `≤4/8` SAT-only screening condition would not hold at 300 seconds; this does not alter the frozen ten-second study. If all eight recover, a verified-recovery advantage over SAT-only is unavailable on this demo set, although a time advantage may still exist. In either case, a later study must compare all online observations available to the ranker, not just these four SAT constraints.

## Integrity and stop rule

The worker receives only four known observations and two verification observations; it does not receive the master key or true trailing keys. Truth is used only after the run for a post-hoc key check. The runner refuses to overwrite results, saves each completed key atomically, validates source/configuration hashes on resume, and writes a detached result checksum at completion. The existing D-032 artifact and frozen SPECK results remain untouched. Any protected SIMON dataset, neural training, or end-to-end attack requires a separate fixed development design and authorization. A solver crash or contradictory SAT/verification result is retained and investigated, not silently retried on the same evidence.
