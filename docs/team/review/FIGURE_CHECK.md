# Figure reproducibility check

> Sessions: first pass on 9 September 2026 against the mid-term report figures; this pass on 27 October 2026 against the publication figures.

Purpose: run the figure script the task list names, compare its two
`manifest.json` files and the PNGs with the committed ones, and record the
machine, because image bytes depend on matplotlib.

Run on 27 October 2026, 15:34 AWST, on Windows 11 (build 26200), Python 3.12.14,
**matplotlib 3.11.2**, numpy 2.5.4, from a fresh clone of `master` at `292a491`.

## What I ran

The task list gives the command with a POSIX output root (`--output-root
/tmp/figcheck`). On Windows I used a scratch folder instead, which is the same
test:

```
python scripts/generate_report_figures.py --profile report --output-root <scratch>/figcheck --force
python scripts/generate_report_figures.py --profile lncs   --output-root <scratch>/figcheck --force
```

Both exited 0: the report profile in 6.2 s, the lncs profile in 5.5 s. Nothing
from the scratch folder was committed.

Note: the script creates a matplotlib cache directory on first import. On this
machine `%LOCALAPPDATA%\matplotlib` was not writable, so matplotlib fell back to
a temporary directory and printed a warning. That is a machine quirk, not a
script problem, but it is worth knowing before a demo.

## The two manifests

| Item | report | lncs |
|---|---|---|
| `checks` (every drawn count vs the published count) | identical to the committed manifest | identical |
| `printed_width_in` | identical (6.0 in) | identical (4.75 in) |
| `inputs_sha256` values | identical, all six | identical, all six |
| `inputs_sha256` keys | **differ**: backslashes on Windows | **differ** |
| `outputs` hashes | differ | differ |
| `outputs` sizes in inches | identical | identical |

So every number the script checks agrees with the committed manifest. The
counts in `checks` are the published ones:

| Figure | Counts checked |
|---|---|
| `fig_recovery_main` | 111, 110 (SPECK 7 rounds), 53, 49, 26 (SIMON 10 rounds) |
| `fig_recovery_d063` | 32, 30, 19 (7 rounds), 26, 0 (8 rounds) |
| `fig_recovery_d063_d066` | 32, 30, 19 and 49, 41 |
| `fig_second_stage` | 4, 18, 2, 19 (SPECK) and 23, 27, 3, 22 (SIMON) |
| `fig_affordable_depth` | 12 plotted points |
| `fig_prospective_predictions` | 8 arms inside their intervals |

All of them match the saved evidence and the timeline, and they match the
numbers quoted in the report.

## The PNGs

There are **seven** PNGs per profile, not six: `fig_pipeline`,
`fig_recovery_main`, `fig_second_stage`, `fig_recovery_d063`,
`fig_recovery_d063_d066`, `fig_affordable_depth`, `fig_prospective_predictions`.
The task list says six, so one figure was probably added after the list was
written.

Byte-for-byte, none of the fourteen regenerated PNGs matches the committed one.
Pixel-for-pixel, **all fourteen match exactly**: same dimensions, and an image
difference with no non-zero pixel in either profile.

That is the expected result and it confirms the guide's warning on
`docs/REPRODUCE.md` line 121 to 124: the image bytes carry metadata that changes
between matplotlib versions, while the content does not. A reader who compares
hashes will think the figures changed; a reader who looks at them will see they
did not. Worth saying in one sentence in the guide.

## The one real mismatch

`manifest.json` records input paths as dictionary keys. The committed manifests
use forward slashes:

```
"docs/paper/evidence/d063_campaign_20261011/ledger_k00-31.jsonl"
```

On Windows the same script writes:

```
"docs\\paper\\evidence\\d063_campaign_20261011\\ledger_k00-31.jsonl"
```

The hash values are identical in both cases; only the key strings differ. So
`figures/report/manifest.json` is not comparable across operating systems, and a
reader who diffs it will see six added and six removed entries where nothing
actually changed. Normalising with `Path.as_posix()` when the manifest is
written would make the cross-OS check exact.

## Verdict

The figures are reproducible: every checked count matches, and both profiles
draw pixel-identical images on a different operating system and a different
matplotlib version. Two documentation-sized fixes would make the check cleaner:
say seven figures, and say that identical bytes are not expected.

This is also the cleanest evidence in the project for the "different OS"
requirement: the same counts, the same images, from a different matplotlib, on
Windows.
