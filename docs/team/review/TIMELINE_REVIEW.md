# Timeline review

This is a reader's review of `docs/timeline/`, written by someone outside the
team, against the saved evidence in the repository. Part one covers files 01 to
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
