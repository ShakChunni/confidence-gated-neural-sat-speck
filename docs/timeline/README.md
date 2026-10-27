# Project timeline

These notes tell the story of the project in the order it happened. Each file
covers one stage. It says what we set out to test, what we ran, what went
wrong, what we found and why we made the next decision. Each file ends with
questions a reader is likely to ask, with short answers.

All times are Perth time (AWST). Every recovered key in these notes was
accepted only after it matched the assigned key and reproduced fresh
plaintext and ciphertext blocks that the attack never saw.

## The short version

We started with a simple plan. A neural network ranks candidates for the last
round key of reduced-round SPECK32/64, and a SAT solver finishes the master
key from the best few candidates. A confidence gate would decide when to trust
the network.

The plan did not survive contact with the evidence, and the changes are the
main result.

- The confidence gate failed. Better failure prediction did not give a
  threshold that held on new keys, and it never beat a plain fixed search.
- Our first comparison had a solver bug that made SAT-only look better than it
  was. Fixing it gave us the rule that every method gets its own solver.
- A good classical ranker matched or beat our own network at six and seven
  rounds. The paper therefore became a fair cost comparison, not a claim that
  neural ranking wins.
- Putting ranking, search and verification on one 300 s clock showed that
  "the right candidate is in the shortlist" is not the same as "the key is
  recovered". A simple cost rule explains most outcomes.
- At eight rounds, Gohr's published network recovers more keys than a complete
  classical table. So neural ranking pays, but only sometimes.

## Stages

| File | Dates | Stage |
|---|---|---|
| [01](01_setup_speck_and_sat_pilot.md) | 6 to 13 Aug | SPECK reference, data generator and first SAT pilot |
| [02](02_neural_rankers.md) | 26 to 29 Aug | Distinguisher, complete-domain ranking and the pair-aware B1 ranker |
| [03](03_confidence_gate_and_first_comparison.md) | 29 Aug to 4 Sep | Confidence gate, first six-round comparison and the solver-state bug |
| [04](04_six_round_reliability_and_seven_round_gate.md) | 4 to 10 Sep | Prospective reliability tests and the seven-round solver gate |
| [05](05_seven_round_hierarchy_and_comparison.md) | 10 to 25 Sep | Two-stage seven-round attack and the 128-key comparison |
| [06](06_simon.md) | 19 Sep to 3 Oct | SIMON32/64: reference, feasibility, development and confirmation |
| [07](07_depth_map_and_backends.md) | 3 to 4 Oct | Residual-depth map, solver backends and the formula package |
| [08](08_controls.md) | 5 to 9 Oct | Controls: queue order, Stage 2, conditioning, ranking depth, encryption cost |
| [09](09_queue_cost_rule.md) | 9 to 10 Oct | The queue-cost rule and its fresh-key test |
| [10](10_published_networks_eight_rounds.md) | 10 to 11 Oct | Published networks at seven and eight rounds |
| [11](11_brute_force_completer.md) | 11 to 12 Oct | SAT against a brute-force completer |
| [12](12_replication_and_what_is_left.md) | 26 to 27 Oct | Eight-round replication, SIMON eleven-round preparation and open work |

## Terms used in every file

- **Stage 1, Stage 2.** The attack ranks the last round key first (Stage 1),
  then the round key before it for each kept candidate (Stage 2).
- **Queue.** The ordered list of candidate key pairs sent to the solver.
- **Containment.** The true candidate is somewhere in the queue.
- **Recovery.** The solver returns a master key that matches the assigned key
  and reproduces the fresh blocks within the time limit.
- **Preparation time.** Time spent before the first solver call: loading
  models or tables, partial decryption and ranking.
- **PAR-2.** Mean time per key, with a failure counted as twice the time limit.
- **Opened, development, fresh.** Opened keys have been looked at and are used
  for building and debugging. Development keys choose settings. Fresh keys are
  drawn once, after the protocol, predictions and analysis are fixed.
- **Demo seed 999999.** Used for pilots and examples. It has no evaluation role.

## Where the evidence is

Saved ledgers, analyses and hashes for the later campaigns are in
`docs/paper/evidence/`, `docs/backend_ranking_depth_evidence_*` and
`docs/queue_cost_model_evidence_20261009/`. `docs/REPRODUCE.md` explains how to
rebuild the numbers and figures from them.
