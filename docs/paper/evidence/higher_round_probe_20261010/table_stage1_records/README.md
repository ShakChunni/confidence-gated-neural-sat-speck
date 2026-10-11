# Per-key Stage-1 records for the 7-round sparse SPECK tables (8-round attack)

Generated 2026-10-11 by `scripts/record_classical_stage1_ranks.py TABLE 8 32 OUT`
(demo seed 999999, 32 development keys, 64 pairs). One line per key, then a
summary line. `pipeline_position` is the true last word's position in the
order the attack uses (descending score, ties by ascending word); `midrank`
is tie-aware.

| Table | Samples, min count | SHA-256 | Top 32 | Best position | Keys where every word ties |
|---|---|---|---:|---:|---:|
| probe table (used in the fresh campaign) | 2^30, 20 | e9cafb4f…e52d | 0/32 | 1,531 | 32/32 |
| 2^32 table | 2^32, 20 | 1669553b…a9667 | 0/32 | 1,531 | 27/32 |
| 2^32 table | 2^32, 5 | b0ffb1bd…5bde | 0/32 | 350 | 0/32 |

The 2^32 tables themselves are in `runtime/higher_round_tables/` (gitignored,
0.9 MB and 12 MB); generation metadata is in `../corrected_20261010/`.

Fresh-key check (seed 701001, the 32 campaign keys, read-only): with the probe
table, every candidate scored zero on all 32 keys, so the Stage-1 top 32 was
the words 0–31 on every key.
