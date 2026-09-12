# Neural-Guided Exact Key Recovery for Reduced-Round SPECK32/64

Research artefact for:

**Learning When to Trust Neural Guidance: Confidence-Gated Incremental XOR-SAT Key Recovery for Reduced-Round SPECK32/64**

The repository studies whether neural ranking can reduce the cost of exact incremental SAT key recovery, when its guidance can be trusted, and when a bounded fallback helps. It is a reduced-round scientific study—not a claim to break deployed, full-round SPECK.

## Current scientific state

| Evidence | Result | Interpretation |
|---|---|---|
| Corrected Phase 7, seed `98001`, `N=64` | B1 `58/64`; B0 `50/64`; paired exact McNemar `p=0.0768`; isolated SAT-only `0/64` at 10 s | `RETROSPECTIVE`: B1 is promising, not confirmed superior. |
| Phase 10 prospective validation, seed `101001`, `N=64` | G1 failure-risk AUC `0.7759`; Sequential-32 `64/64`; adaptive policy `63/64` | `PROSPECTIVE_VALIDATION`: useful discrimination, no operational advantage over the static policy. |
| Phase 10-G2 prospective validation, seed `201001`, `N=256` | G2 AUC `0.8513` versus G1 `0.7194`; paired delta `0.1319`, `p=0.02688`; frozen trusted-region target missed | `PROSPECTIVE_VALIDATION`: improved discrimination, negative threshold transport. Six-round model search is closed. |
| Phase 11A frozen seven-round oracle gate, seed `300001`, `N=64` | `0/64`, all 64 timed out at 10 s | Frozen `NEGATIVE`/RED result for the specified backend and envelope. Phase 11B/11C remain blocked. |
| Phase 11B′ hierarchical evaluation on opened seed `300001` | `13/16` verified, Wilson 95% CI approximately `[0.570, 0.934]` | `DEVELOPMENT` only. It does not overturn the frozen RED gate or authorize protected work. |

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

The recommended paper is an empirical methods/reliability paper: neural guidance helps candidate ordering, but discrimination, calibration, and operational utility are different properties; exact-solver configuration and accounting can change the apparent boundary. The minimum credible extension is a strong full-distribution classical comparator, controlled same-device solver/screening ablations, a frozen simple seven-round method, and one genuinely unseen prospective evaluation—only after explicit authorization.

The current preferred realistic target is ACNS 2027, subject to the official call and supervisor approval. The project is not presently positioned as a general SAT-solver contribution or a full-round cryptanalytic break.
