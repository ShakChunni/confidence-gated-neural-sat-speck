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
| Phase 11A frozen seven-round oracle gate, seed `300001`, `N=64` | `0/64`, all 64 timed out at 10 s | Frozen `NEGATIVE`/RED result for the specified backend and envelope. Phase 11B/11C remain blocked. |
| Phase 11B′ hierarchical evaluation on opened seed `300001` | `13/16` verified, Wilson 95% CI approximately `[0.570, 0.934]` | `DEVELOPMENT` only. It does not overturn the frozen RED gate or authorize protected work. |
| Six-round sparse statistical comparator on opened seed `98001` | Top-4 containment `64/64`, mean sorted rank `1.75`, mean ranking time approximately `0.545 s` | `RETROSPECTIVE`: the observed result is stronger than B0/B1 on this family and rules out a six-round neural-superiority narrative. |
| Matched screening and controlled solver audits | Screening retained the recorded top-32 ordering in `4/4` true-prefix cases with mean paired speedup `30.60x`; revised native matrix settings solved `2/3` opened diagnostics within 60 s versus `0/3` for each default treatment | `ENGINEERING/DIAGNOSTIC`: useful controls, not prospective recovery claims or universal causal conclusions. |

Every reported recovered master key must pass independent fresh-pair verification. Neural models order final-round subkey candidates; they neither prove a candidate nor directly recover the full 64-bit key.

## Governance and stop boundary

- Seeds `98001`, `101001`, `201001`, and `300001` are opened/retired for their recorded roles; `97001` and `99001` remain quarantined or retired.
- Do not generate or open a Phase 12 blind seed without an entirely frozen protocol and explicit author approval.
- Do not run a protected seven-round campaign from the current Phase 11B′ document. A September 2026 audit found a conflict between D-026 and higher-priority `AGENTS.md`/`PLAN.md`; D-027 places the proposed freeze and authorization on hold.
- The hierarchical evaluation used the ordinary Python `pycryptosat` wrapper path. A separate engineering diagnostic used the native CryptoMiniSat binary with explicit Gaussian-elimination settings. Those results and causal claims must remain separate.

Start with:

- [`PLAN.md`](PLAN.md) for the governing phase boundary;
- [`docs/PUBLICATION_ROADMAP_2026.md`](docs/PUBLICATION_ROADMAP_2026.md) for the publication-readiness audit and next steps;
- [`docs/PUBLICATION_EXPERIMENT_DESIGN.md`](docs/PUBLICATION_EXPERIMENT_DESIGN.md) for the proposed—not authorized—experiment design;
- [`docs/EXPANSION_LITERATURE_2026.md`](docs/EXPANSION_LITERATURE_2026.md) for the primary-source claim boundary;
- [`docs/DECISIONS.md`](docs/DECISIONS.md) and [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) for the audit trail.

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

The remaining publication-critical experiment is one frozen, unseen seven-round
comparison. The planned methods are neural-neural hierarchical ranking,
neural-classical hybrid ranking, the strongest feasible fully classical hierarchy,
random tuple order, and SAT-only recovery. The existing method named
`hierarchical_classical` is scientifically a hybrid because Stage 1 is neural; paper
text and future records must label it accordingly. The experiment must use one
global cutoff, isolated solvers, complete timing, and no post-open tuning.

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
