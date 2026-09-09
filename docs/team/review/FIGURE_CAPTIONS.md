# Figure caption check

> Sessions: first pass on 9 September 2026 against the mid-term report; this pass on 27 October 2026 against the publication figures.

Checked 27 October 2026 on Windows 11, from `master` at `292a491`.

Status: **partly done.** The report PDF is not in the repository — `.gitignore`
excludes `main.tex` and `main.pdf` — so the captions themselves were not
available to check against. What I could do is the harder half: I extracted the
text of each figure in `figures/report/`, read what every panel and legend
actually asserts, and checked each count against `docs/timeline/` and the saved
evidence. When the report PDF arrives, only the "caption says" column is left
to fill.

How each figure was read: the text layer of `figures/report/<name>.pdf` was
extracted, so the labels quoted below are the ones printed inside the figure,
not my paraphrase.

## The seven report figures

The report profile writes **seven** PNGs, not six: `fig_pipeline`,
`fig_recovery_main`, `fig_second_stage`, `fig_recovery_d063`,
`fig_recovery_d063_d066`, `fig_affordable_depth`, `fig_prospective_predictions`.
A caption list with six entries will be missing one of these.

| Figure | What the figure shows (from its own labels) | Counts agree with | Caption risk |
|---|---|---|---|
| `fig_pipeline` | The attack flow: data, Stage 1 ranking, later stages, queue, exact completion, verification, under one 300 s clock; the cost equation `cTi = Pi + sum(Rj + Hj) + Si` | Definitions match `docs/timeline/README.md` and file 09 | The equation uses symbols (`§`, `∙`, `¤`) that render as different glyphs when the figure text is extracted; if the caption quotes the formula, use the same symbols as the figure |
| `fig_recovery_main` | (a) SPECK 7 rounds, N = 128: hybrid 111/128, screened 110/128, four arms at 0/128. (b) SIMON 10 rounds, N = 64: classical 53/64, Bao hybrid 49/64, Bao neural 26/64, five arms at 0/64 | file 05 lines 70 to 77; file 06 lines 74 to 80; `figures/report/manifest.json` | The zero arms are counted, not named. A caption that says "the remaining methods recovered nothing" should say how many, or name them, because 4 and 5 are different sets |
| `fig_second_stage` | (a) SPECK 7 rounds: 4/32, 2/32 (Stage 1 only) and 18/32, 19/32 (two-stage), with containment 27/32 marked. (b) SIMON 10 rounds: 23/32, 3/32 and 27/32, 22/32 | file 08 lines 53 to 60, exactly | The panel title says "Two-stage hybrid"; the caption must not call it a new attack, since file 08 calls these controls |
| `fig_recovery_d063` | (a) SPECK 7 rounds, N = 32: Gohr 32/32, classical 30/32, B1 19/32. (b) SPECK 8 rounds, N = 32: Gohr 26/32, **sparse** classical tables 0/32 | file 10 lines 28 to 34; d063 summary | **Main risk.** Panel (b) is the *sparse, sampled* table that scored zero. If the caption calls it "the classical table", it contradicts the next figure, where a complete table scores 41/64 |
| `fig_recovery_d063_d066` | (a) SPECK 7 rounds, N = 32: 32/32, 30/32, 19/32. (b) SPECK 8 rounds, N = 64: Gohr 49/64, **full 7R table** 41/64 | file 12 lines 19 to 22; d066 summary | Same risk in reverse: the caption must say "complete 4 GiB table" here, and say why the N went from 32 to 64 |
| `fig_affordable_depth` | Affordable depth k* against median correct-candidate position + 1, for 12 of 13 pipelines, labelled 53/64, 49/64, 26/64, 111/128, 110/128, 4/32, 2/32, 18/32, 19/32, 23/32, 27/32, 22/32 | Caption file in the repo; file 08 and file 12 counts | The caption is honest about SPECK positions using recovered keys only. Keep that sentence; without it the SPECK and SIMON points look like the same measurement |
| `fig_prospective_predictions` | Eight arms of the fresh-key cost-rule test, each with its 90% interval fixed in advance, marked inside or outside | file 09 lines 44 to 49 | The caption must say "eight arms" and that one arm sat at the upper edge (5 against an expected 1), or a reader will assume all eight were comfortable |

## The two figures that do have captions in the repository

These were checked properly, caption against figure.

### `figures/fig8_affordable_depth.caption.txt`

| Caption says | Figure shows | Counts match | Fix needed |
|---|---|---|---|
| 13 pipelines from the break-even table, 12 plotted | 12 labelled points | yes | none |
| Blue circles incremental solver, orange squares fresh solver process, hollow markers fewer than 10 completed rejections | same three encodings used | yes | none |
| Not plotted: N=32 SIMON, 1 word, matrices off | 12 points, so one of 13 is absent | yes | none, the caption already says it |
| SPECK rejection time is approximate and the solve term is zero | consistent with the method used in file 09 | yes | none |

This caption is the model the other six should follow: it says what is
excluded, why, and where a number is approximate.

### `figures/fig6b_backend_profiles_appendix_v2.caption.txt`

| Caption says | Figure shows | Counts match | Fix needed |
|---|---|---|---|
| four opened keys per cipher, descriptive only | the backend table in file 07 lines 35 to 42 | yes | none |
| all configurations recover 4/4 with two key words supplied | matches file 07 line 44 | yes | none |
| one SIMON trial infrastructure-unavailable, 0/3 of 4 | matches file 07 lines 48 to 51 | yes | none |
| failures charged 600 s in PAR-2 | matches the PAR-2 definition | yes | none |

The caption contains two typographic artefacts: `â€` (a mis-decoded non-breaking
space) before "One SIMON trial", and a footnote mark that has lost its number.
Both are visible in the PDF if the caption is reproduced as-is.

## What to do when the report PDF arrives

1. For each figure in the PDF, write the caption text in a new "caption says"
   column next to the rows above.
2. Check three things per row: the count in the caption equals the count in the
   figure; the caption names the same method and the same N; the caption does
   not describe the sparse table and the complete table with the same words.
3. Check that every figure in the PDF appears in this table. Seven report
   figures exist; if the PDF shows six, one is unused and should be removed
   from the profile rather than left in `figures/report/`.
4. Re-check the two caption files above against their PDF pages, since the
   figure text can change while the caption file does not.
