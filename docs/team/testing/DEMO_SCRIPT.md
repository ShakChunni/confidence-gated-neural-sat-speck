# Seven-round presentation demo

Status: not run on this Windows machine. The assignment requires an Apple Silicon Mac. No backup video has been recorded; prepare it before presenting. Both supplied classical-table JSON files are available, but neither has been used to claim a recovery.

On the configured Mac, from the repository root, run the following, using a new ledger filename for the actual presentation date:

```bash
python scripts/run_higher_round_campaign.py --seed 999999 --role demo --keys 1 --arms R7-G,R7-C --ledger docs/team/testing/demo_run_YYYYMMDD.jsonl
```

Beforehand, follow the assignment's setup for CryptoMiniSat, Gohr's converted network and the supplied classical tables. Do not rebuild the tables or overwrite an existing ledger.

Say: "These two arms compare Gohr's neural network with the classical difference table. Ranking proposes candidates; recovery still requires exact verification. Recovered means the key reproduces the known blocks and two independent fresh blocks."

Point to `outcome`, `true_queue_position` and `total_s`. These fields are assigned in `scripts/run_higher_round_campaign.py`; they also occur in the saved campaign evidence. No new demo ledger was produced here. `total_s` is per-arm elapsed time, not the full command's wall-clock duration.

Timing: the assignment reports 22.3 seconds for Gohr and 25.4 seconds for the classical arm on Ashfaq's laptop. These are historical figures, not measurements on this machine or a guaranteed presentation duration.

If the live run fails, switch to the previously recorded backup video and clearly identify it as a recording. Until that video exists, use the saved evidence and explicitly state that live recovery is unavailable; do not claim a successful live run.
