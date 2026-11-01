# 12. Eight-round replication, SIMON eleven-round preparation and open work

**Dates:** 26 to 27 October 2026

## What we set out to do

At eight rounds, 32 keys gave a gap between Gohr's network and the complete
table that was not significant after correction (stage 10). We replicated
with twice the keys and made recovery the primary test. That choice was
informed by the earlier result, and the protocol says so.

## Eight-round replication (64 fresh keys)

The protocol was hashed at 02:24 AWST on 27 October, before the seed made
any key. The analysis script was hashed at 03:11, before any outcome was read.
The run took 02:30 to 09:41 with no reruns, no missing cells and no cell over
310 s.

| Arm | Recovered of 64 | True chain in queue |
|---|---:|---:|
| Gohr seven-round network | 49 | 49 |
| Complete seven-round table | 41 | 41 |

- **Primary (recovery).** The arms disagreed on 10 keys. The network won 9 and
  the table 1 (exact McNemar p = 0.021). Under the frozen rule this reads
  "network recovers more".
- **Stage 1 ranking.** The network ranked the true key better on 44 keys and
  worse on 12 (Holm p = 4.2e-5).
- **Cost.** The network had lower PAR-2 on 31 keys and higher on 19 (Holm
  p = 0.12). This one is not significant.
- Both arms fell inside their predicted ranges.

Across all three eight-round campaigns, every key with its true chain in the
queue was recovered and no other key was. Containment decides at eight
rounds. Nothing can finish before preparation ends. Preparation took 150 to 170 s,
and the first recovery came at about 152 s.

Pooling these 64 keys with the earlier 32 gives 73 against 59 recoveries
(p = 0.0005). We report that as a secondary description only.

## Four matched comparisons, one picture

Gohr's network against a classical Stage 1 under the same later stages:

| Setting | Network | Classical |
|---|---:|---:|
| Seven rounds, 32 keys | 32 | 30 |
| Eight rounds, sparse table, 32 keys | 26 | 0 |
| Eight rounds, complete table, 32 keys | 24 | 18 |
| Eight rounds, complete table, 64 keys | 49 | 41 |

The network recovered at least as many keys in all four, and significantly
more in two. Separately, a classical Stage 1 matched or beat our own B1 at
seven rounds in all three comparisons, and significantly in two.

## SIMON at eleven rounds (prepared, not run)

To test whether the eight-round SPECK picture carries over, we prepared an
eleven-round SIMON design with the same shape: three ranked stages and the
brute-force completer.

- A complete eight-round SIMON table was built on 27 October (09:42 to 10:02)
  and passed all checks: 16 rows identical to an audited tool, total mass 1,
  and 128 entries matching an independent Python calculation exactly. The table
  is too large for this repository and is kept locally.
- The protocol draft is **not frozen**. It still needs a two-key demo pilot,
  a thread setting, predictions, a seed and the analysis script hash.

## What the project shows

1. A shortlist that contains the correct candidate is not a recovery. Time
   to prepare and time to reject wrong candidates decide whether the solver
   gets there.
2. A simple cost rule built from preparation time, queue position and
   rejection time explains most outcomes and predicted a fresh cohort.
3. Neural ranking pays only sometimes. Our B1 lost to classical tables. Gohr's
   network beat a complete table at eight rounds.
4. Confidence gating did not help. Better failure prediction did not give a
   threshold that held on new keys.

## Limits

- One laptop (Apple M5, 16 GB), one solver family for the main runs, and a
  300 s clock. Timing on other machines will differ.
- Two ciphers at a few round counts each. Nothing here concerns full-round
  SPECK or SIMON.
- The classical tables are sampled or model-based, not exact fixed-key
  distributions.
- Several controls used eight opened keys and support description, not
  general claims.

## Questions and answers

**Why replicate instead of reporting the 32-key result?**
The 32-key gap was not significant after correction. A larger fresh cohort
with a frozen primary test is the honest way to settle it.

**Is changing the primary test to recovery a form of tuning?**
It was informed by the earlier run, and we disclose that. It was fixed before
the new keys existed, and the analysis was hashed before any outcome was read.

**Why is there no final blind confirmation?**
We kept one in reserve for a fully frozen method. It is listed as future work.
The fresh-key campaigns above are pre-registered tests, each run once.

**What would change your conclusion?**
For the cost rule: keys that fail even though the true candidate sits early
in the queue and wrong candidates are cheap to reject. For the eight-round
neural result: a stronger classical table that matches Gohr's network on
containment.
