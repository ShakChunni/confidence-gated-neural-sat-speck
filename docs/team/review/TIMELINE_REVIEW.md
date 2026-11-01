# Timeline review

This is a reader's review of `docs/timeline/`, written by a team member who did
not do the technical work, against the saved evidence in the repository. Part one covers files 01 to
06; part two, in the next commit, covers 07 to 12.

Reviewed on 27 October 2026 on Windows 11 (build 26200), from `master` at
commit `292a491`.

Method. I read each file once for meaning, then checked every number I could
against a saved file: `docs/paper/evidence/` summaries, `results/phase11_*`,
`figures/report/manifest.json`, and the analysis scripts. I also recomputed the
statistics that can be recomputed from the counts printed in the text. I did
not rerun the campaigns.

Severity labels: **wrong** means a number or statement that contradicts another
part of the repository. **unclear** means a reader cannot follow it without
guessing. **missing** means something a marker would reasonably ask for is not
there.

## Summary

| File | Wrong | Unclear | Missing |
|---|---:|---:|---:|
| 01 setup and SAT pilot | 0 | 2 | 0 |
| 02 neural rankers | 0 | 1 | 1 |
| 03 confidence gate and first comparison | 0 | 2 | 0 |
| 04 six-round reliability and seven-round gate | 0 | 1 | 1 |
| 05 seven-round hierarchy and comparison | 0 | 1 | 1 |
| 06 SIMON | 0 | 2 | 1 |

No wrong number was found in files 01 to 06. The problems are places where a
careful reader has to guess, and two places where a result sounds stronger than
the evidence supports.

## 01. SPECK reference, data generator and first SAT pilot

**Unclear, line 37 to 42.** The table's third column is headed "Result at 1 s",
but the five-round row reports "timeout (solved in 1.17 s with 5 s)". A reader
cannot tell from the table whether five rounds pass or fail the stated limit.
Either put the second limit in the cell ("5 s: 1.17 s") or add a footnote.

**Unclear, line 34 and 37.** The text says four known pairs were used, and the
table lists variable counts of 1,408 to 2,368. It does not say whether the count
is for one key plus four pairs or per pair. This matters because the jump from
four to five rounds is only 320 variables, which is what a reader would use to
sanity-check the encoding.

Everything else in this file is internally consistent. The claim on line 44
that one or two pairs can give a wrong key that still matches those pairs is
the reason fresh verification pairs appear everywhere later; that link is well
made.

## 02. Distinguisher, complete-domain ranking and the B1 ranker

**Unclear, line 44.** "It also cut the mean number of capped attempts from 9.41
to 7.33." This is the only place the project uses "capped attempts", and the cap
is never defined (candidate count? attempts per key?). Define it in one clause,
or link the term to the place where the cap is set.

**Missing, line 42.** The second test is described as "64 fresh validation
keys". `docs/timeline/README.md` lines 63 to 65 define three categories:
opened, development and fresh, and it defines fresh as keys drawn once after
the protocol, predictions and analysis are fixed. If these 64 keys were used to
promote B1, they are development keys under that definition. Calling them
"fresh validation keys" mixes two categories, and a marker checking the fresh
key rules will stop here.

The tables on lines 33 to 36 are consistent with the text: B1 wins top-16 on
both sets, loses median rank on confirmation, so it is not promoted on 28
August. That sequence is clear and honest.

## 03. Confidence gate, first comparison and the solver-state bug

**Unclear, line 20.** "64 held-out keys" does not say held out from what. If
these are the stage 02 confirmation keys, they had already been looked at, which
is exactly why the McNemar result on line 51 is reported as an observation. The
file half-says this ("these keys, which we had already looked at") but never
states which set it is.

**Unclear, and possibly wrong, line 27 (with stage 04, line 24).** Stage 03 says
the scalar gate's ROC-AUC was "about 0.52". Stage 04 says the same scalar's
ROC-AUC was "0.480". Both sentences describe the gate predicting a top-16
failure. One of two things is true: the two numbers come from different key
sets, or one of them is wrong. The files do not say which key set each AUC was
computed on. This is the one cross-file inconsistency in the first half, and it
is cheap to fix: name the key set next to each AUC, or align the two numbers.

The solver-bug section (lines 30 to 35) is the strongest part of the file: it
gives the wrong number (12/64), the right number (0/64), and the mechanism. It
would be stronger still with a one-line pointer to the test that now prevents
solver reuse.

## 04. Six-round reliability and the seven-round solver gate

**Unclear, line 33 to 34.** "Seven-round gate: RED. 0/64 recovered and all 64
timed out" is true of the setting that was tested, but the file does not say
that this was the Python wrapper with a 10 s limit, and stage 05 lines 22 to 24
show the native solver recovering 3 of 3 comparable keys in 48 to 59 s. A
reader who stops at file 04 takes away a stronger claim than the evidence
supports. Add one sentence at line 34 pointing forward to file 05.

**Missing, line 31.** "5.2% (11 of 211 trusted keys failed)" is correct
arithmetic (11/211 = 5.21%), but the count of trusted keys is never explained:
211 of 256 keys were trusted, so 45 were untrusted by the model. One clause
saying how many keys were routed to the wide search would close the gap.

The G2 threshold story (line 28 to 31) is otherwise precise: the AUC rises, the
threshold does not transport, and the decision to stop is justified by the
prospective test rather than by preference.

## 05. Two-stage seven-round attack and the 128-key comparison

**Unclear, line 74.** The full-neural Stage 2 row reports a median time of
308 s under a 300 s clock. That is possible with launch and teardown overhead,
but the table does not say so, and a reader will assume an error. One clause
under the table fixes it.

**Missing, line 22 to 24.** The native-solver claim ("solved 3 of 3 seven-round
keys that had timed out, in about 48 to 59 s") has no path to the files behind
it, while every later campaign has an evidence folder and a hash. Either point
to where those three runs are stored or say that they were diagnostic and not
kept.

**Checked, line 79 and 80.** The p-value is the exact two-sided McNemar value
for 110 against 0: 2 x 2^-110 = 1.54e-33, reported as 1.5e-33. That is right.
It is also the only place in the first six files where the convention is
stated by the size of the number rather than in words; stage 11 uses one-sided
values without saying so, which I note in part two.

The audit section (lines 44 to 59) is the part of the timeline that most needs
to stay exactly as it is. It records three overclaims, the date they were
found, and what replaced them.

## 06. SIMON32/64

**Missing, line 58 to 59.** Attempt 1 "finished with 75 recoveries". Out of how
many? The next sentence says nine methods and 64 keys, so the denominator is
576 cells, and 75 is a low yield that the paragraph then explains. Give the
number as "75 of 576 cells".

**Unclear, line 36.** The complete seven-round table has "2.17 billion
entries". Whether that fits on a 16 GiB laptop depends on the entry size, which
is not given: 2.17e9 entries at one byte is 2.2 GB, at eight bytes it is 17 GB.
Since line 32 says Bao's own code needed about 64 GiB, the reader wants to know
what the memory-bounded version stores.

**Unclear, line 74 to 80.** The replacement table collapses three methods into
one row ("SIMON B1 hybrid, random, SAT-only") at zero, next to a paragraph that
talks about nine methods. Give the table the same number of rows as the run had
arms, or say in the caption that the three zero arms are merged because each
was zero on every key.

**Checked.** The recovery counts are consistent with the text: 54 + 53 + 49 +
26 = 182 independent checks (line 83), and 53 against 49 is the 4-key gap
quoted with p = 0.34 on line 84. Nine methods x 64 keys = 576 records (line 69).

## Numbers checked against saved evidence

| Claim | Where in the timeline | Checked against | Result |
|---|---|---|---|
| 32, 30, 19, 26, 0 recoveries in the first eight-round campaign | 10, lines 28 to 34 | `docs/paper/evidence/d063_campaign_20261011/summary.json` | match |
| 24 and 18 on the complete table | 10, lines 52 to 56 | `docs/paper/evidence/d064_campaign_20261011/summary.json` | match |
| 8 arm counts, B1 at 26 above its 14 to 24 range | 11, lines 48 to 52 | `docs/paper/evidence/d065_campaign_20261012/summary.json` | match |
| 49 and 41 on 64 keys | 12, lines 19 to 22 | `docs/paper/evidence/d066_campaign_20261027/summary.json` | match |
| 111 and 110 of 128 | 05, lines 70 to 77 | `results/phase11_prospective/seven_round_prospective_analysis.json` (screened recovery rate 0.8594 = 110/128) | match |
| Figure counts 32, 30, 19, 26, 0 | 10 | `figures/report/manifest.json`, `checks.fig_recovery_d063` | match |

## What I would ask the author

1. Name the key set next to every AUC and every p-value, and state whether a
   p-value is one- or two-sided. The numbers are mostly right; the conventions
   are not written down.
2. Give every diagnostic run a place on disk, or state that it was not kept.
   Files 05 and 06 both make claims about runs that a reader cannot open.
3. Keep the "what went wrong" sections short, but keep them. They are the part
   of the timeline that makes the results believable.

## Part two: files 07 to 12

Part two covers the controls, the cost rule, the published-network campaigns, the
brute-force completer and the replication. Same method as part one: read for
meaning, then check every number that can be checked against a saved file, and
recompute the statistics that can be recomputed from the printed counts.

| File | Wrong | Unclear | Missing |
|---|---:|---:|---:|
| 07 depth map and backends | 0 | 1 | 1 |
| 08 controls | 0 | 2 | 0 |
| 09 queue-cost rule | 0 | 2 | 0 |
| 10 published networks at eight rounds | 0 | 0 | 1 |
| 11 brute-force completer | 1 | 1 | 0 |
| 12 replication and what is left | 0 | 1 | 1 |

### 07. Residual-depth map, solver backends and the formula package

**Unclear, line 24 to 27.** "During review we found that the first analysis had
written zero for timing and formula size where values were missing ... a
corrected analysis was saved as a new version." This is exactly the kind of
correction a marker wants to see, and it is the only place in the timeline where
a correction has no file path. Name the corrected file.

**Missing, line 56 to 58.** "The package was rebuilt from scratch in a clean
Python environment, and all 64 formula representations and their reference
witnesses passed." 64 is an odd number to leave unexplained: it is not 2
ciphers x 2 forms x 16 rounds. Say how the 64 are counted, or where the check
log is.

**Checked, line 20.** 151 + 41 = 192 cells, which matches 8 demo keys x 12
settings x 2 observation counts. The arithmetic is right.

### 08. Controls

**Unclear, line 36 to 38.** "The projected median Stage 2 time was 115 s and
500 s for the network, against 59 s for the table in both cases." Which network
run is 115 s and which is 500 s? Two numbers are given for one method and no
condition is attached to either.

**Unclear, line 65 to 68.** "One cell failed on a monitoring error ... With
explicit approval we retried that cell once and continued the 114 cells that had
not started." Of 256 cells, one failed and 114 had not started, so 141 had
already finished. State that, because the sentence currently leaves the reader
unable to tell how much of the grid was affected.

**Checked, line 72 to 76.** 68.2 million blocks per second x 300 s = 2.05e10 =
2^34.25, reported as 2^34.3; 29.7 million x 300 s = 8.91e9 = 2^33.05, reported
as 2^33.1. Both round correctly to one decimal place.

### 09. The queue-cost rule and its fresh-key test

**Unclear, line 44 to 46.** "All 8 arm counts fell inside their intervals ... One
arm (SPECK, matrix on, with the cap) was at the upper edge: 5 against an
expected 1." If the expected count was 1, an interval that contains 5 must be
very wide. Print the interval next to the count so the reader can see that the
test was still passed rather than stretched.

**Unclear, line 53 to 56.** "The run stopped at cell 300 of 320 ... We kept the
299 finished cells." The difference between 300 and 299 is one cell that was in
flight when the coordinator failed. One clause fixes it.

**Checked, line 27 to 30.** 242/256 = 94.5%, 158/256 = 61.7% and 196/256 =
76.6% all match the text exactly.

### 10. Published networks at seven and eight rounds

**Missing, line 59.** "Every true candidate that reached the queue was
recovered, in all eight-round arms (68 of 68)." The count is right: 26 + 0 + 24
+ 18 = 68 across the four eight-round arms. But the arm counts come from two
different campaigns and key sets, and the sentence does not say so. Name the
two campaigns, or a reader will think 68 keys were run together.

The p-values in this file are exact two-sided McNemar values: six discordant
keys against zero gives 2 x 0.5^6 = 0.031, as printed, and the Holm adjustment
to 0.063 is right for two tests.

### 11. SAT against a brute-force completer

**Wrong convention, line 39 and 42.** The p-values in this file are one-sided,
while every other stage reports two-sided values without saying so. Twenty-eight
discordant keys against none gives a two-sided exact p of 2 x 0.5^28 = 7.5e-9;
the file prints 3.7e-9, which is 0.5^28, the one-sided value. The same holds for
17 against 1: two-sided 1.4e-4, printed 7.2e-5. The numbers are therefore not
wrong in themselves, but they are on a different scale from stage 05 (110
against 0, printed 1.5e-33, which is two-sided) and stage 12 (9 against 1,
printed 0.021, two-sided). Make the convention explicit or align it.

**Unclear, line 44 to 46.** "The size of the SAT advantage ... followed the
predicted order B1, then the table, then Gohr's network (23, 16 and 6 keys)."
These three numbers are counts out of 32 keys each, but line 39 says SAT
finished ahead on 28 keys. A reader will try to reconcile 23 + 16 + 6 with 28
and fail. Say that the 28 is the pooled comparison and the 23, 16 and 6 are
per-ranker counts.

**Checked.** The 5 of 8 arms inside their intervals is confirmed by the saved
summary: B1 at 26 is above its 14 to 24 range, and both eight-round arms at 18
are below 22 to 29.

### 12. Eight-round replication, SIMON eleven-round preparation and open work

**Unclear, line 35.** "No cell finished before 150 s, because preparation took
about 153 s." If preparation took 153 s, no cell could finish before 153 s, so
the 150 s statement cannot fail. The interesting claim is the tighter one: no
cell finished sooner than about 153 s, because nothing can happen before
preparation ends. Rewrite it that way.

**Missing, line 55 to 65.** The SIMON eleven-round section reports a table built
and checked, and a protocol that is not frozen, but gives no path to either. The
eight-round table is a substantial artefact; say where it is stored.

**Checked.** 49 + 24 = 73 and 41 + 18 = 59, so the pooled sentence is
arithmetically right, and 9 against 1 gives the printed p = 0.021. The four
matched comparisons on lines 46 to 49 agree with stages 10 and 11.

## What part two adds up to

The second half is more careful than the first: every campaign has a folder, a
hash and a protocol, and the corrections are written down rather than hidden.
Two things need attention before submission:

1. **The p-value convention in stage 11.** It is the only place where a printed
   p-value cannot be reproduced from its own counts with the convention used
   everywhere else. A marker who recomputes it will find 7.5e-9, not 3.7e-9.
2. **Corrections without paths.** Stage 07 says a corrected analysis was saved
   as a new version, and stage 12 says a table was built and checked, but
   neither names a file. Everything else in the timeline names its evidence;
   these two should too.

## Numbers checked against saved evidence, part two

| Claim | Where in the timeline | Checked against | Result |
|---|---|---|---|
| 26 and 0 at eight rounds, 32 keys | 10, lines 28 to 34 | `d063_campaign_20261011/summary.json` | match |
| 24 and 18, complete table | 10, lines 52 to 56 | `d064_campaign_20261011/summary.json` | match |
| 8 arm counts and intervals | 11, lines 48 to 52 | `d065_campaign_20261012/summary.json` | match |
| 49 and 41 on 64 keys | 12, lines 19 to 22 | `d066_campaign_20261027/summary.json` | match |
| 4/32, 2/32, 18/32, 19/32, 23/32, 27/32, 3/32, 22/32 | 08, lines 53 to 58 | `figures/report/manifest.json`, `checks.fig_second_stage` | match |
| 8 of 8 prediction intervals, one at the edge | 09, line 44 | `figures/report/manifest.json`, `checks.fig_prospective_predictions` | match on the count of 8 |
