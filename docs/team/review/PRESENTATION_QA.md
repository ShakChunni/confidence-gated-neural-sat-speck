# Presentation question sheet

Fifteen questions we are likely to get, with a short answer each, in my own
words. Numbers are the ones in `docs/timeline/` and in the saved evidence; the
file each answer comes from is named so we can point at it on the night.

Compiled 27 October 2026.

**1. What does the project show, in one sentence?**
A shortlist that contains the right key is not a recovery. What decides
recovery is how long preparation takes, where the correct candidate sits in the
queue, and how long each wrong candidate takes to reject. Put those three on
one 300 s clock and a simple sum of them predicts most of what happens
(files 05, 08, 09).

**2. Why is accuracy or key rank not enough to judge a ranker?**
They measure the ranker alone. The attack also pays to build the ranking and to
reject every wrong candidate that comes before the right one. In the ranking
depth study, every arm held the true candidate on the same 27 of 32 keys, yet
recovery ranged from 2 to 27 out of 32. Same containment, very different
outcome (file 08).

**3. How do you know a recovered key is the right key?**
It has to match the assigned key and reproduce the ciphertext for two fresh
plaintext blocks that the attack never used. With only a few pairs, many keys
fit the pairs we have; the extra blocks make a coincidence very unlikely, and
we still do not call verification infallible (file 01).

**4. Why is a timeout not the same as a wrong candidate?**
The solver stopped before it finished, so it proved nothing either way. If a
timeout counted as wrong, a method that tries more candidates would look better
than it is (file 01).

**5. What was the solver bug, and why did it matter?**
All five methods shared one CryptoMiniSat object, so clauses and phases learnt
by one method helped the next. With a fresh solver per method, SAT-only went
from 12 of 64 to 0 of 64. Every comparison after that gives each method its own
solver, and a factory with tests refuses to hand one solver to two methods
(file 03).

**6. The confidence gate reached AUC 0.85. Why call it a failure?**
AUC only orders keys by risk; using the model needs a fixed threshold. The
threshold chosen on development keys let through 5.2% error on new keys when 2%
was planned (11 of 211 trusted keys failed), and routing never beat a plain
fixed search of 32 candidates: 63 of 64 against 64 of 64 (file 04).

**7. Why stop working on the gate instead of trying more features?**
Because another model on the same 64 keys would be a search for a better
number, not a test. We froze a new protocol and ran it once; it failed in the
same way, so we kept the negative result and moved to the cost question
(files 03, 04).

**8. Why did your own network lose to a classical table?**
At seven rounds a sparse six-round table built from sampled difference
statistics put the true key in its top 4 on all 64 opened keys, at 0.57 s each,
and it was about three times cheaper to prepare than B1 (20 s against 61 s).
Our B1 was the weaker ranker in that setting (files 05, 09).

**9. Then why does Gohr's network beat a classical table at eight rounds?**
At eight rounds the attack fixes three round keys, so a wrong candidate costs
about 0.25 s to reject and the whole 512-entry queue fits inside the clock.
Recovery follows containment, and Gohr's network put the true chain in the queue
more often: 49 keys against 41 for the complete table (files 10, 12).

**10. Is 49 against 41 a real difference?**
Yes, under the primary test frozen before the keys existed. The two arms
disagreed on 10 keys, the network won 9 and the table 1, exact McNemar
p = 0.021. The earlier 24 against 18 on 32 keys was not significant after
correction, which is why we replicated on twice as many keys (file 12).

**11. Why isn't the 26 to 0 result the headline?**
Because the table it beat scored every candidate zero: on every key, none of
the 64 decrypted pairs hit a stored entry. The neural win was against a ranker
with no signal, so we built the complete four-gigabyte table and reported that
comparison instead (file 10).

**12. Why did you move SIMON from nine to ten rounds? Isn't that cherry-picking?**
At nine rounds plain SAT already recovered 8 of 8 keys inside the 300 s clock,
so ranking had nothing to contribute. We moved to ten rounds before any ranking
outcome existed, and the choice came from SAT and oracle controls, not from
ranking results (file 06).

**13. What is the cost rule, exactly?**
A key is recovered when preparation time, plus the rejection time of every
wrong candidate ahead of the correct one, plus the correct candidate's solve
time, still fits inside 300 s. It predicted 242 of 256 development cells
(94.5%) and then passed a fresh-key test in which all eight arm counts landed
inside intervals committed before the keys were drawn (file 09).

**14. If the rule needs the correct key's position, is it an attack?**
No. An attacker does not know that position, so the rule explains and predicts
rather than recovers. Its practical use is choosing between methods, and that
use was tested: it chose SAT at seven rounds and brute-force completion at
eight before the keys existed, and both choices held up (file 11).

**15. What are the limits?**
One laptop and one solver family for the main runs, two ciphers at a few round
counts each, classical tables that are sampled or model-based rather than exact
fixed-key distributions, and several controls on eight opened keys that support
description, not general claims. Nothing here says anything about full-round
SPECK or SIMON, and timings elsewhere will differ (file 12).

## Follow-ups to have ready

- PAR-2 is the mean time per key with a failure charged twice the limit, so a
  method that fails half its keys cannot look fast.
- 300 s is about 2^34.3 full-round SPECK encryptions on the reference laptop
  (file 08). That is a scale for the clock, not a count of solver work.
- The six methods in the 128-key comparison each got one solver, one budget of
  1,024 candidates and the same clock (file 05).
