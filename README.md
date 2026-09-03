# Confidence-Gated Neural-SAT SPECK

This repository contains the complete research artefact for the scientific study:

**Learning When to Trust Neural Guidance: Confidence-Gated Incremental XOR-SAT Key Recovery for Reduced-Round SPECK32/64**

The repository implements, tests, and evaluates a closed-loop neuro-symbolic cryptanalytic system combining **permutation-invariant Deep Sets neural distinguishers**, **selective classification / diversity fallback gating**, and **exact incremental XOR-aware SAT solving via CryptoMiniSat**.

---

## 1. Final Experimental Results (Phase 7 Held-Out Evaluation)

The final empirical benchmark evaluated all 5 comparative methods on an independently generated **held-out attack test benchmark** of **$N=64$ independent master keys** on 6-round SPECK32/64 ($2^{16} = 65,536$ complete candidate subkey domain per instance, $\Delta P = (0\text{x}0040, 0\text{x}0000)$):

| Method | Solved Keys / 64 | Exact Success Rate | 95% Wilson CI | Median Online Time (s) | PAR-2 ($T=100\text{s}$) | Key Empirical Finding |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **1. SAT-Only Baseline** | $0 / 64$ | **$0.00\%$** | $[0.0\%, 5.7\%]$ | $10.17\text{s}$ | $200.00\text{s}$ | Times out on 100% of keys on isolated solver. |
| **2. Random-Order + SAT ($k=16$)** | $0 / 64$ | **$0.00\%$** | $[0.0\%, 5.7\%]$ | $3.80\text{s}$ | $200.00\text{s}$ | Exhausts $k=16$ budget on 16 UNSATs on every key. |
| **3. Baseline B0 (Gohr 2019) + SAT** | $50 / 64$ | **$78.12\%$** | $[66.6\%, 86.5\%]$ | $78.72\text{s}$ | $108.70\text{s}$ | Solves majority, but misses 14 keys outside top-16. |
| **4. Proposed B1 (Deep Sets) + SAT** | **$58 / 64$** | **$90.62\%$** | $[81.0\%, 95.6\%]$ | **$79.04\text{s}$** | **$92.14\text{s}$** | Observed $+12.50\text{ pp}$ advantage over B0 (exact McNemar $p = 0.0768$). |
| **5. Historical Eager B1/B0 Fallback** | **$63 / 64$** | **$\mathbf{98.44\%}$** | $[91.7\%, 99.7\%]$ | $156.41\text{s}$ | $200.00\text{s}$ | Rescues 5 of 6 B1 misses using doubled budget ($k=32$) and two neural rankings. |

---

## 2. Research Questions & Scientific Findings

### RQ1 — Effectiveness (**RETROSPECTIVE / PROMISING, NOT PROSPECTIVELY CONFIRMED**)
* **Finding:** On the corrected historical Phase 7 benchmark (seed `98001`, $N=64$), unguided SAT-only and random search solved $0/64$ ($0.00\%$), while pair-aware neural ranking ($B1$) achieved $58/64$ ($90.62\%$) and baseline $B0$ achieved $50/64$ ($78.12\%$). While $B1$ showed an observed $+12.50$ percentage-point advantage over $B0$, the paired difference was not statistically significant at the conventional 0.05 level (exact McNemar $p = 0.0768$). When the true subkey is assumed, clean CryptoMiniSat deduction requires a median of $476.7\text{ ms}$ (mean $605.4\text{ ms}$). Prospective confirmation is required.

### RQ2 — Reliability (**CALIBRATION BREAKDOWN IDENTIFIED**)
* **Finding:** In symmetric cryptanalysis, a single scalar score-margin feature $(s_{(1)} - s_{(8)}) / \sigma_{\text{all}}$ is **insufficient** to reliably gate top-$k$ containment ($\text{ROC-AUC} \approx 0.52$). Wrong candidates can receive high neural scores and wide margins. Modular carry clustering in ARX ciphers is hypothesized as an exploratory mechanism ("possible structured false peaks"). All reported master-key recoveries must pass independent fresh-pair verification.

### RQ3 — Fallback Resilience (**RETROSPECTIVE / DIAGNOSTIC**)
* **Finding:** The historical eager diversity fallback (interleaving Deep Sets $B1$ with Gohr baseline $B0$ candidates up to $k=32$) recovered **$63 / 64$ keys ($98.44\%$)**, rescuing $5$ of the $6$ keys that $B1$ missed in isolation. However, computing two full rankings doubles the offline ranking latency, motivating Phase 10's investigation of ranking stability ($G_1$) and shortlist rescue ($R_1$).

---

## 3. Publication Figures & LaTeX Tables Suite

### Publication Figures Suite
All figures are generated in vector SVG, 300 DPI PNG, and vector PDF formats:

* **Evaluation Results (`results/figures/evaluation/`):**
  1. `final_success_rates_6round.pdf`: Comparative success rate bar chart across all 5 methods.
  2. `ecdf_solved_vs_attempts.pdf`: Empirical cumulative distribution (ECDF) of solved keys vs candidate budget ($k=1 \dots 32$).
  3. `runtime_and_par2_comparison.pdf`: Median online runtime and PAR-2 penalized runtime benchmark.
* **Methodology & Architecture (`results/figures/proposal/`):**
  1. `methodology_pipeline.pdf`: End-to-end neuro-symbolic attack workflow.
  2. `b1_pair_aware_architecture.pdf`: Permutation-invariant Deep Sets ResNet encoder.
  3. `b0_b1_prospective_results.pdf`: Prospective validation distribution comparison.
  4. `validation_protocol.pdf`: Zero-contamination seed isolation protocol across 24 seeds.
  5. `exact_recovery_verification.pdf`: Algebraic master-key completion and 2-pair verification.
  6. `confidence_fallback_mechanism.pdf`: Selective gating and diversity fallback routing.
  7. `research_positioning.pdf`: Positioning against Gohr (2019), Soos (2009), and Chen (2023).
  8. `evaluation_matrix.pdf`: Five-method comparative evaluation matrix.

### LaTeX Tables (`results/tables/`)
* `results/tables/phase7_results_table.tex`: Ready-to-embed LaTeX `table` environment.
* `results/tables/phase7_results_table.md`: Markdown summary table.

---

## 4. Command-Line Interface (`neural-sat-speck`)

The repository installs a standalone CLI tool for experimentation and demonstration:

```bash
# 1. Verify official Known-Answer Test (KAT)
neural-sat-speck kat

# 2. Generate synthetic chosen-plaintext attack instances
neural-sat-speck generate-data --instances 10 --pairs 64 --rounds 6 --seed 999999 --output data/demo_instances.json

# 3. Explain confidence calibration and routing decision
neural-sat-speck explain-confidence --ranking results/phase7_final_evaluation/records/instance_0000.json

# 4. Verify candidate master key against fresh independent plaintexts
neural-sat-speck verify-key --key 0x1918 0x1110 0x0908 0x0100 --rounds 6 --pairs 4 --seed 12345
```

---

## 5. Development & Reproduction

### Installation
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Running Tests & Quality Checks
```bash
pytest
ruff check .
```

### Reproducing Phase 7 Evaluation & Figures
```bash
# Run Phase 7 comparative evaluation
python scripts/run_phase7_final_evaluation.py

# Generate Phase 7 evaluation figures and LaTeX tables
python scripts/plot_phase7_evaluation_figures.py

# Generate Proposal & Architecture figures
python scripts/plot_phase6b_proposal_figures.py
```

---

## 6. Implementation Summary

* **Cipher:** Reference word-level SPECK32/64 ($16$-bit word, $32$-bit block, $64$-bit key), scalar round functions, key expansion, full encryption/decryption, and 1-round partial decryption inversion.
* **Neural Models:** Single-pair Gohr ResNet baseline ($B0$) and multi-pair permutation-invariant Deep Sets group model ($B1$).
* **Exact Solver:** Incremental CryptoMiniSat 5.14.7 Boolean encoding with native XOR Gaussian elimination, temporary subkey assumptions (`solve(assumptions)`), and algebraic master-key deduction (median ~481ms under clean solver conditions).
* **Verification:** Cryptographic verification against 4 known attack pairs in solver constraints plus 2 fresh independent plaintexts (bounding the theoretical false positive rate to at most $2^{-32}$ under ideal cipher assumptions; 4 pairs bound to at most $2^{-64}$).
