# Link and path check

> Sessions: first pass on 11 September 2026 for the mid-term report; this pass on 27 October 2026 for the final report.

Scope: every link and file path in `README.md` and `docs/REPRODUCE.md`.

Checked on 27 October 2026 on Windows 11 (build 26200) from a fresh clone of
`master` at commit `292a491`. Method: each target was resolved relative to the
repository root and tested for existence; the clone URL was tested by cloning
it; the two external systems (Google Colab, a GPU runtime) were not opened.

Result: **40 references checked, 40 work, 0 broken.** No link or path needed a
fix, so this commit changes no existing file.

## README.md

| # | Line | What it points to | Type | Works or broken |
|---:|---:|---|---|---|
| 1 | 39 | `docs/timeline/README.md` | in-repo file | works |
| 2 | 39 | `docs/timeline/` | in-repo directory | works |
| 3 | 45 | `src/neural_sat_speck/` | in-repo directory | works |
| 4 | 46 | `tests/` | in-repo directory | works |
| 5 | 47 | `scripts/` | in-repo directory | works |
| 6 | 48 | `tools/` | in-repo directory | works |
| 7 | 49 | `configs/` | in-repo directory | works |
| 8 | 50 | `models/` | in-repo directory | works |
| 9 | 51 | `datasets/` | in-repo directory | works |
| 10 | 52 | `results/` | in-repo directory | works |
| 11 | 52 | `docs/paper/evidence/` | in-repo directory | works |
| 12 | 53 | `artifacts/benchmark_formulas/` | in-repo directory | works |
| 13 | 54 | `figures/` | in-repo directory | works |
| 14 | 55 | `docs/` | in-repo directory | works |
| 15 | 58 | `docs/REPRODUCE.md` | in-repo file | works |
| 16 | 76 | `.\.venv\Scripts\Activate.ps1` | created by setup | works after step 3 of the quick start; not expected to exist in a fresh clone |
| 17 | 91 | `docs/REPRODUCE.md` | in-repo file | works |
| 18 | 95 | `scripts/colab_reproduce.ipynb` | in-repo file | works |
| 19 | 119 | `LICENSE` | in-repo file | works |
| 20 | 120 | `Literature Review (Cryptanalysis).pdf` | in-repo file | works |

Line 120 points at a file name that contains spaces and brackets. It resolves
correctly, and the parentheses are escaped by Markdown as written, so the link
also works when the repository is opened on GitHub.

## docs/REPRODUCE.md

| # | Line | What it points to | Type | Works or broken |
|---:|---:|---|---|---|
| 21 | 32 | `https://github.com/ShakChunni/confidence-gated-neural-sat-speck.git` | external clone URL | works, cloned for this check |
| 22 | 58 | `scripts/colab_reproduce.ipynb` | in-repo file | works |
| 23 | 90 | `tests/conftest.py` | in-repo file | works |
| 24 | 105 | `docs/paper/evidence/d063_campaign_20261011/ledger_k00-31.jsonl` | in-repo file | works |
| 25 | 106 | `docs/paper/evidence/d064_campaign_20261011/ledger_k00-31.jsonl` | in-repo file | works |
| 26 | 107 | `docs/paper/evidence/d065_predictions/predictions.json` | in-repo file | works |
| 27 | 107 | `docs/paper/evidence/d065_campaign_20261012/ledger_k00-31.jsonl` | in-repo file | works |
| 28 | 110 | `summary.json` (in each evidence folder) | in-repo file | works: `d063_campaign_20261011/summary.json` and `d064_campaign_20261011/summary.json` both exist |
| 29 | 116 | `scripts/generate_report_figures.py --profile report` | in-repo file | works |
| 30 | 117 | `scripts/generate_report_figures.py --profile lncs` | in-repo file | works |
| 31 | 124 | `figures/<profile>/manifest.json` | in-repo file | works: `figures/report/manifest.json` and `figures/lncs/manifest.json` both exist |
| 32 | 151 | `scripts/colab_build_speck_full_ddt.ipynb` | in-repo file | works |
| 33 | 152 | `tools/` | in-repo directory | works |
| 34 | 156 | `tools/bruteforce_completion.c` | in-repo file | works |
| 35 | 159 | `scripts/run_higher_round_campaign.py` | in-repo file | works |
| 36 | 160 | `scripts/analyze_d063.py` | in-repo file | works |
| 37 | 161 | `scripts/analyze_d066.py` | in-repo file | works |
| 38 | 162 | `scripts/run_backend_ranking_depth.py` | in-repo file | works |
| 39 | 163 | `scripts/launch_prospective_queue_cost.py` | in-repo file | works |
| 40 | 175 | `.gitattributes` | in-repo file | works |

## Things that look like links but are not

Three strings in `docs/REPRODUCE.md` are code symbols, not paths, so a
path checker reports them as missing: `os.waitid` (lines 23 and 84), and
`summary.json` where the name is introduced before the folder is given
(line 102). None of them needs a change.

## What this check does not cover

- The two Colab notebooks are checked for existence only. Running them in
  Colab is Hamza's task, not this one.
- Only `README.md` and `docs/REPRODUCE.md` were in scope. The link tables
  inside `docs/timeline/` were not rechecked here.
