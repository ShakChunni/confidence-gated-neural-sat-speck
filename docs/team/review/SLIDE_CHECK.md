# Slide number check

> Sessions: first pass on 4 September 2026 against the mid-term material; this pass on 27 October 2026 against the final report.

Checked 27 October 2026 on Windows 11, from `master` at `292a491`.

What I could check: the repository contains no `.pptx`. The deck is generated
by `scripts/generate_presentation.py` (420 lines), so every number and claim
the slides will carry is in that script as a string. I checked each number in
it against the report's numbers (`README.md`, `docs/timeline/`, the evidence
summaries) and each headline claim against the report's wording.

Headline: the numbers are almost all right; **the slide story is not the report
story.** The deck is still built around the confidence gate as a working
contribution, and the report records it as a failure.

## Numbers on the slides

| Slide | Number shown | Where it comes from | Matches |
|---|---|---|---|
| 3 (line 221) | 81.25% timeout rate, pure SAT at 6 rounds, 10.0 s | 52 of 64 keys; not printed in the timeline or the report | cannot verify from the repo |
| 3 (line 223) | 0.0% recovery, random order | file 03, line 46 | yes |
| 3 (line 234) | ~481 ms median to deduce the remaining bits | file 03, line 35 ("about 0.48 s") | yes |
| 9 (line 380) | AUC ≈ 0.52 for the scalar gate | file 03, line 27 | yes for stage 03, but file 04 line 24 gives 0.480 for the same gate |
| 9 (line 385) | 98.44% (63/64), 5 of 6 misses rescued | file 03, line 49 | yes, but see the claim problem below |
| 10 (line 404) | 0/64 at 10 s; historical 18.75% | 18.75% = 12/64; file 03, lines 32 to 33 | yes |
| 10 (line 408) | 78.12% (50/64) for B0 | file 03, line 48 | yes |
| 10 (line 410) | 90.62% (58/64) for B1, +12.50 pp | file 03, line 49; 8/64 = 12.5 pp | yes |
| 10 (line 412) | 98.44% (63/64), 9.11 mean tries | file 03, line 49 | yes |
| 10 (line 414) | 100% verified on 2 fresh plaintexts | file 01, verification rule | yes |
| 13 (line 450) | 0.00% to 90.62% | file 03 | yes |
| 13 (line 451) | ~481 ms | file 03 | yes |
| 13 (line 463) | 98.44% | file 03 | yes |

Every percentage that appears on the deck is arithmetically consistent with the
six-round, 10 s comparison in stage 03. None of them is wrong as a number.

**But they are the wrong numbers for the report.** The report's headline results
are the 300 s clock and the fresh-key campaigns: 110 and 111 of 128 at seven
rounds, 49 against 41 at eight rounds, 53 against 49 on SIMON. A deck whose
biggest number is 98.44% from an opened-key 10 s comparison will read, next to
the report, as a different project.

## Claims that contradict the report

| Slide, line | What the slide says | What the report says | Action |
|---|---|---|---|
| Title, line 154 | "Learning When to Trust Neural Guidance: Confidence-Gated ... Key Recovery" | Report title is "Containment Is Not Recovery: When Key Ranking Repays Its Cost" | Retitle the deck to the report's claim |
| 3, line 235 | "3. Confidence Gating: Dynamically detects when neural rankings are unreliable and routes to a safe fallback" | `README.md` line 34: "The original confidence-gate idea failed"; file 04: the threshold did not transport and routing never beat a fixed search | Replace with the negative finding, or remove the gate from the contribution list |
| 9, line 381 | The gate "failed" only because of "pseudo-differential ghost peaks ... frequently confidently wrong" | Files 03 and 04 give three separate reasons: no threshold met the rule, the threshold did not transfer, and a fixed search was already as good | Keep the ghost-peak explanation as a hypothesis, not as the recorded cause |
| 9, line 385 | "Rescued 5 of 6 B1 misses ... achieving 98.44%" | File 03, lines 52 to 54: the fallback used two full rankings and twice the candidates, "so it is not a fair win either" | Say it used double the budget, or drop the comparison |
| 13, line 449 | "Proved that Deep Sets neural guidance (B1) breaks the 6-round exponential SAT cliff" | File 09: B1 lost to a classical table at seven rounds; `README.md` line 28: neural ranking pays only sometimes | Change to a statement about ranking plus SAT against SAT alone, not about B1 |
| 13, line 463 | "Proved that architectural diversity fallback provides deterministic recovery robustness" | Nothing in the repo supports "deterministic"; the fallback is described as not a fair win | Delete "proved" and "deterministic" |
| 2, line 178 | "22 total rounds; round 6 is the exact SAT phase-transition boundary" | File 01: six rounds was chosen because plain SAT could not solve it; no phase-transition measurement exists in the repo | Drop "exact"; say six rounds is where plain SAT stopped finishing inside the limit |
| 2, line 179 | "Noise floor at round 8 (max differential probability < 2^-32)" | Not in the timeline or the evidence | Cite it or remove it |

## What also needs to be on the slides, because it is in the report

1. The 300 s clock and the containment-not-recovery distinction. The pipeline
   figure (`fig_pipeline`) and the cost equation already carry this; the deck
   never uses them.
2. The eight-round result: 24 to 18 on 32 keys, then 49 to 41 on 64 fresh keys,
   exact McNemar p = 0.021.
3. The SIMON ten-round result: 53 against 49, p = 0.34, so the two methods
   could not be told apart.
4. The cost rule: 242 of 256 cells (94.5%), and eight of eight fresh arm counts
   inside committed intervals.

## How to redo this check when the real deck arrives

One row per slide: slide number, number shown, where it comes from (report
section, timeline file, or evidence file), matches or not. The table above is
the same check done on the generator, so it can be reused as the checklist.

If the final deck is produced by hand rather than by this script, run the same
three tests on it: every percentage must name its key set and clock; every
"proved" must be a result that survived a fresh-key test; and no slide may
present the confidence gate as a contribution.
