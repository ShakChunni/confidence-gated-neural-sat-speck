# From Neural Ranking to Verified Key Recovery

Research artefact for:

**Learning When to Trust Neural Guidance: Confidence-Gated Incremental XOR-SAT Key Recovery for Reduced-Round SPECK32/64**

The repository compares neural, classical, hybrid, random-order, and SAT-only
guidance for exact reduced-round SPECK32/64 key recovery. The central question is
whether an apparent neural ranking or reliability improvement survives complete
online-cost accounting and fresh-pair-verified master-key recovery. It is a
reduced-round scientific study, not a claim to break deployed full-round SPECK.

## Current scientific state

| Evidence | Result | Interpretation |
|---|---|---|
| Corrected Phase 7, seed `98001`, `N=64` | B1 `58/64`; B0 `50/64`; paired exact McNemar `p=0.0768`; isolated SAT-only `0/64` at 10 s | `RETROSPECTIVE`: B1 is promising, not confirmed superior. |
| Phase 10 prospective validation, seed `101001`, `N=64` | G1 failure-risk AUC `0.7759`; Sequential-32 `64/64`; adaptive policy `63/64` | `PROSPECTIVE_VALIDATION`: useful discrimination, no operational advantage over the static policy. |
| Phase 10-G2 prospective validation, seed `201001`, `N=256` | G2 AUC `0.8513` versus G1 `0.7194`; paired delta `0.1319`, `p=0.02688`; frozen trusted-region target missed | `PROSPECTIVE_VALIDATION`: improved discrimination, negative threshold transport. Six-round model search is closed. |
| Phase 11 prospective evaluation, seed `669981`, `N=128` (D-031) | Screened `110/128` (85.94%, Wilson [0.7886, 0.9092], PAR-2 178.31 s); Classical hybrid `111/128` (86.72%, Wilson [0.7976, 0.9154], PAR-2 143.24 s); Full neural `0/128` (ranking exhausted 300 s budget, 0 SAT attempts); Random SAT `0/128`; SAT-only 4-obs and all-obs `0/128`. Primary contrast paired risk difference +85.94% [78.29%, 90.92%], exact McNemar $p = 1.54 \times 10^{-33}$. Candidate reduction 29.9% overall (304.4 vs 213.3) and 52.3% on recoveries (186.6 vs 89.1; median 144 vs 38). | `PROSPECTIVE_VALIDATION`: the predeclared RQ1 contrast favoured neural guidance over SAT-only at 7 rounds under the common resource envelope. The classical Stage-2 hybrid had similar recovery with lower runtime and fewer candidate attempts. |
| Phase 11A frozen seven-round oracle gate, seed `300001`, `N=64` | `0/64`, all 64 timed out at 10 s | Frozen `NEGATIVE`/RED result for the specified backend and envelope. |
| Six-round sparse statistical comparator on opened seed `98001` | Top-4 containment `64/64`, mean sorted rank `1.75`, mean ranking time approximately `0.545 s` | `RETROSPECTIVE`: the observed result is stronger than B0/B1 on this family and rules out a six-round neural-superiority narrative. |
| Matched screening and controlled solver audits | Screening retained the recorded top-32 ordering in `4/4` true-prefix cases with mean paired speedup `30.60x`; revised native matrix settings solved `2/3` opened diagnostics within 60 s versus `0/3` for each default treatment | `ENGINEERING/DIAGNOSTIC`: useful controls, not prospective recovery claims or universal causal conclusions. |
| SIMON R10 replacement confirmation, 64 keys (D-041) | Classical `53/64`; Bao hybrid `49/64`; Bao fully neural `26/64`; B1-style hybrid, random controls and SAT-only `0/64`; true-tuple diagnostic `54/64` | Fixed-method comparison closed with [documented provenance limits](docs/SIMON_CONFIRMATION_REPLACEMENT_CLOSURE_20261003.md). All 182 recoveries independently checked. Classical versus hybrid recovery difference is inconclusive (`p=0.34375`); B1 stopped during ranking in all 64 cells. |
| Cross-cipher depth map, eight opened demo keys per cipher (D-043) | `151/192` verified recoveries, `41/192` parent timeouts | [Reviewed closure](docs/CROSS_CIPHER_DEPTH_CLOSURE_20261004.md) preserves raw results and corrects unavailable timeout metrics. Equal nominal depth does not summarize observed recovery; source/runtime/chronology limits prevent universal or causal claims. |
| Controlled backend diagnostic, four opened keys per cipher (D-044/D-045) | `80` recoveries, `79` deadlines and `1` infrastructure interruption across `160` assigned cells. With one true trailing key, explicit matrix profile `4/4` on both ciphers versus matched zero-matrix `2/4` SPECK and `0/4` SIMON; all profiles `4/4` with two true keys | [Reviewed closure](docs/BACKEND_COMPARISON_CLOSURE_20261004.md). The interrupted unguided control is `0/3` completed trials with one unavailable; conditioned comparisons are unaffected. All recoveries, 66 saved native models and 64 formula witnesses independently checked. No universal backend/structure claim. |

Every reported recovered master key must pass independent fresh-pair verification. Neural models order final-round subkey candidates; they neither prove a candidate nor directly recover the full 64-bit key.

## Governance and stop boundary

- Seeds `98001`, `101001`, `201001`, `300001`, `406737`, and `669981` are opened/retired for their recorded roles; `97001` and `99001` remain quarantined or retired.
- Do not generate or open a Phase 12 blind seed without an entirely frozen protocol and explicit author approval.
- Phase 11 seven-round SPECK comparative evaluation is completed and locked. No further modifications to seven-round SPECK protocols, models, or datasets are permitted.
- The hierarchical evaluation used the ordinary Python `pycryptosat` wrapper path under strict per-method solver isolation. Engineering diagnostics on native Gaussian-elimination settings remain separate.

Start with:

- [`PLAN.md`](PLAN.md) for the governing phase boundary;
- [`docs/PUBLICATION_ROADMAP_2026.md`](docs/PUBLICATION_ROADMAP_2026.md) for the publication-readiness audit and next steps;
- [`docs/PUBLICATION_EXPERIMENT_DESIGN.md`](docs/PUBLICATION_EXPERIMENT_DESIGN.md) for the proposed—not authorized—experiment design;
- [`docs/EXPANSION_LITERATURE_2026.md`](docs/EXPANSION_LITERATURE_2026.md) for the primary-source claim boundary;
- [`docs/DECISIONS.md`](docs/DECISIONS.md) and [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) for the audit trail.

## Writing and artifact handoff (2026-10-04)

The author owns manuscript writing in a separate project. Use the
[writing prompt and evidence packet](docs/writing_handoff_20261004/00_WRITING_PROMPT.md),
the [current release review](docs/PUBLICATION_PREPARATION_RELEASE_20261004.md),
[final PDF/PNG figures, CSVs and plotting code](docs/FINAL_FIGURES_AND_CSV_TASK_20261004.md).
The [portable evidence guide](docs/PORTABLE_EVIDENCE_REPRODUCTION_20261004.md)
documents checked local formula regeneration and its limits. Historical
publication-planning paragraphs below do not supersede this current status.

## Implemented system

- Tested scalar and vectorized SPECK32/64 implementations with known-answer checks.
- Deterministic synthetic training and attack-instance generation with seed governance.
- B0 single-pair and B1 permutation-invariant multi-pair complete-domain rankers.
- Exact incremental SAT recovery with per-method solver isolation.
- Fresh-pair master-key verification and explicit outcome records.
- G1/G2 reliability analysis, shortlist/cascade tooling, and preserved negative results.
- Hierarchical seven-round and solver-configuration diagnostics on opened development data.

## Installation and validation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

python -m pytest -q
ruff check .
```

The demo seed is `999999`. Do not substitute a protected, quarantined, or unregistered seed.

```bash
neural-sat-speck kat
neural-sat-speck generate-data \
  --instances 10 --pairs 64 --rounds 6 --seed 999999 \
  --output data/demo_instances.json
```

Historical result scripts are retained for auditability. Before executing an experiment script, check its protocol, seed role, backend, solver lifetime, timeout, candidate budget, hardware record, and evidence classification. A successful script run is not by itself a scientific confirmation.

## Publication direction

The recommended paper is a prospective empirical comparison rather than a new-attack
claim. Its contribution is to separate classifier quality, full-domain key rank,
top-k failure discrimination, threshold transport, and verified recovery under one
resource envelope. The completed evidence already shows that improved reliability
discrimination did not yield a better fixed policy, a credible classical ranker can
reverse the apparent neural advantage, and solver configuration can move the
observed seven-round boundary.

The seven-round SPECK comparative evaluation is complete and frozen (D-031, $N=128$, Seed `669981`), prospectively confirming that neural subkey guidance provides an operational advantage over SAT-only and random baselines ($85.94\%$ vs $0.00\%$, McNemar $p = 1.54 \times 10^{-33}$), while a classical difference distribution table at Stage 2 matches recovery ($86.72\%$) with 32% lower runtime and 52% candidate reduction on recoveries. Full unpruned neural ranking failed due to runtime exhaustion during scoring (0 SAT attempts). The SPECK publication account is closed.

The SIMON32/64 reference, two-key SAT pilot, and [predeclared demo operating-window study](docs/SIMON_OPERATING_WINDOW_PROTOCOL.md) are complete. At nine rounds with four known observations, unguided SAT timed out on 8/8 demo keys at a ten-second solve limit, while two correct trailing-key assumptions recovered 8/8. A separate [300-second online SAT-only control](docs/SIMON_300S_SAT_CONTROL_PROTOCOL.md) then recovered all eight of those same demo keys without any subkey assumption (median 49.57 s). The [paper evidence note](docs/SIMON_FEASIBILITY_EVIDENCE.md) gives the grid, complete timing, hashes, and limits. This is a useful negative feasibility check for that exact setting, not an end to the approved cross-cipher study. The next task is to fix an information- and resource-matched SIMON comparison, then complete the empirical depth map, controlled backend study, artifact, and cross-cipher paper synthesis. Neural training and protected evaluation keep their separate gates.

The working paper title is **From Neural Ranking to Verified Key Recovery: A
Controlled Study of Classical, Neural, and XOR-SAT Guidance on Reduced-Round
SPECK32/64**. The current primary target is ACNS 2027 Cycle 2, subject to the
evidence gate and supervisor approval. The project is not positioned as a new
neural architecture, a general SAT-solver contribution, a round-record attack, or a
full-round cryptanalytic break.

The approved course-time expansion keeps that SPECK study first and then tests
external validity on SIMON32/64. It adds a predeclared empirical map of residual
depth and solver configuration, a controlled native-XOR/wrapper/XOR-to-CNF backend
comparison, and a reproducible SPECK/SIMON formula artifact. This planning approval
does not open protected data: SIMON must first pass specification, known-answer,
reference-implementation, classical/neural-baseline, and exact-solver feasibility
gates, and every outcome-bearing study needs its own frozen protocol and approval.
See `PLAN.md`, Decision D-029, and `docs/PUBLICATION_ROADMAP_2026.md`.
