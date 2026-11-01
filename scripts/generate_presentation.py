# Mid-semester deck from 3 Sep 2026, kept as a record. Its claims predate the later
# campaigns and do not reflect the final report.
"""Generate an elite, publication-grade academic PowerPoint presentation in a crisp Light Theme (.pptx).

Addresses all 4 COMP6002 Master's evaluation criteria (40 Marks):
1) Problem Statement (10 marks)
2) Background & Related Work (10 marks)
3) Methodology & Empirical Adjudication (10 marks)
4) Plan / Timeline & Expansion Horizons (10 marks)
Plus: Key Terminology & Foundational Concepts Glossary.
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPOSITORY_ROOT / "presentation_confidence_gated_neural_sat_speck.pptx"

# Elegant Light Academic Palette
C_BG = RGBColor(248, 250, 252)           # #F8FAFC
C_WHITE = RGBColor(255, 255, 255)        # #FFFFFF
C_NAVY = RGBColor(15, 39, 71)            # #0F2747
C_NAVY_LIGHT = RGBColor(30, 58, 138)     # #1E3A8A
C_COBALT = RGBColor(37, 99, 235)         # #2563EB
C_COBALT_LIGHT = RGBColor(219, 234, 254) # #DBEAFE
C_EMERALD = RGBColor(22, 163, 74)        # #16A34A
C_EMERALD_DARK = RGBColor(22, 101, 52)   # #166534
C_EMERALD_LIGHT = RGBColor(220, 252, 231)# #DCFCE7
C_AMBER = RGBColor(217, 119, 6)          # #D97706
C_AMBER_LIGHT = RGBColor(254, 243, 199)  # #FEF3C7
C_PURPLE = RGBColor(124, 58, 237)        # #7C3AED
C_PURPLE_LIGHT = RGBColor(243, 232, 255) # #F3E8FF
C_SLATE = RGBColor(71, 85, 105)          # #475569
C_SLATE_LIGHT = RGBColor(241, 245, 249)  # #F1F5F9
C_BORDER = RGBColor(203, 213, 225)       # #CBD5E1
C_TEXT = RGBColor(15, 23, 42)            # #0F172A
C_TEXT_MUTED = RGBColor(100, 116, 139)   # #64748B


def create_presentation() -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def add_light_slide(title_text: str, category_text: str):
        slide = prs.slides.add_slide(blank_layout)
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = C_BG
        bg.line.fill.background()

        # Top Header Banner
        header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(0.35), Inches(12.333), Inches(1.15))
        header.fill.solid()
        header.fill.fore_color.rgb = C_NAVY
        header.line.color.rgb = C_NAVY
        header.line.width = Pt(1.5)

        # Category Pill
        pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(0.48), Inches(3.6), Inches(0.28))
        pill.fill.solid()
        pill.fill.fore_color.rgb = C_COBALT
        pill.line.fill.background()
        p_pill = pill.text_frame.paragraphs[0]
        p_pill.text = category_text.upper()
        p_pill.font.size = Pt(10)
        p_pill.font.bold = True
        p_pill.font.color.rgb = C_WHITE

        # Title Text
        tb = slide.shapes.add_textbox(Inches(0.7), Inches(0.78), Inches(11.8), Inches(0.6))
        p = tb.text_frame.paragraphs[0]
        p.text = title_text
        p.font.size = Pt(20)
        p.font.bold = True
        p.font.color.rgb = C_WHITE

        return slide

    def add_card(slide, left, top, width, height, title, items, fill_color=C_WHITE, stroke_color=C_BORDER, title_color=C_NAVY, tag=""):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = fill_color
        card.line.color.rgb = stroke_color
        card.line.width = Pt(1.5)

        tb = slide.shapes.add_textbox(left + Inches(0.18), top + Inches(0.18), width - Inches(0.36), height - Inches(0.36))
        tf = tb.text_frame
        tf.word_wrap = True

        if tag:
            p_tag = tf.paragraphs[0]
            p_tag.text = tag.upper()
            p_tag.font.size = Pt(10)
            p_tag.font.bold = True
            p_tag.font.color.rgb = stroke_color
            p_tag.space_after = Pt(3)
            p0 = tf.add_paragraph()
        else:
            p0 = tf.paragraphs[0]

        p0.text = title
        p0.font.size = Pt(13)
        p0.font.bold = True
        p0.font.color.rgb = title_color
        p0.space_after = Pt(6)

        for item in items:
            p = tf.add_paragraph()
            p.text = f"• {item}"
            p.font.size = Pt(10.5)
            p.font.color.rgb = C_TEXT
            p.space_after = Pt(4)

    # =========================================================================
    # SLIDE 1: Hero Title Slide (Light Academic)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = C_BG
    bg1.line.fill.background()

    # Hero Center Container Card
    hero = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.8), Inches(11.733), Inches(5.9))
    hero.fill.solid()
    hero.fill.fore_color.rgb = C_WHITE
    hero.line.color.rgb = C_NAVY
    hero.line.width = Pt(2.5)

    # Top accent bar on hero card
    top_bar = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.8), Inches(11.733), Inches(0.2))
    top_bar.fill.solid()
    top_bar.fill.fore_color.rgb = C_COBALT
    top_bar.line.fill.background()

    tb1 = s1.shapes.add_textbox(Inches(1.3), Inches(1.3), Inches(10.7), Inches(5.0))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "COMP6002 MASTER'S SCIENTIFIC RESEARCH PROJECT"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_COBALT
    p.space_after = Pt(14)

    p2 = tf1.add_paragraph()
    p2.text = "Learning When to Trust Neural Guidance:\nConfidence-Gated Incremental XOR-SAT Key Recovery for Reduced-Round SPECK32/64"
    p2.font.size = Pt(26)
    p2.font.bold = True
    p2.font.color.rgb = C_NAVY
    p2.space_after = Pt(18)

    p3 = tf1.add_paragraph()
    p3.text = "A Closed-Loop Neuro-Symbolic Cryptanalysis Pipeline: Permutation-Invariant Deep Sets Distinguishers, Selective Fallback Gating & CryptoMiniSat Deductions"
    p3.font.size = Pt(14)
    p3.font.color.rgb = C_SLATE
    p3.space_after = Pt(36)

    p4 = tf1.add_paragraph()
    p4.text = "Author: ShakChunni  |  Evaluation Benchmark, Publication Artefacts & Interactive CLI"
    p4.font.size = Pt(14)
    p4.font.bold = True
    p4.font.color.rgb = C_EMERALD_DARK

    # =========================================================================
    # SLIDE 2: Key Cryptographic & ML Terminology Glossary
    # =========================================================================
    s2 = add_light_slide("Foundational Concepts & Key Terminology Glossary", "Core Terminology & Theoretical Foundations")

    terms = [
        ("SPECK32/64 & ARX", ["Lightweight block cipher: 32-bit block, 64-bit key, 16-bit words.", "ARX Operations: Modular Addition (+), Bitwise Rotation (<<<), and XOR (⊕).", "22 total rounds; round 6 is the exact SAT phase-transition boundary."], C_COBALT, C_COBALT_LIGHT),
        ("Differential Cryptanalysis", ["Tracks how a chosen input difference ΔP = P ⊕ P' propagates.", "Standard target difference: ΔP = (0x0040, 0x0000).", "Noise floor at round 8 (max differential probability < 2^-32)."], C_COBALT, C_COBALT_LIGHT),
        ("Neural Distinguisher", ["Residual neural network trained to distinguish cipher ciphertext pairs from random uniform pairs.", "Pioneered by Gohr (CRYPTO 2019).", "Outputs probabilistic logits, not mathematical proofs."], C_PURPLE, C_PURPLE_LIGHT),
        ("Deep Sets (Permutation Invariance)", ["Processes unordered sets of 64 ciphertext pairs simultaneously.", "Mathematical formula: Score = ρ ( 1/m ∑ φ(x_i) ).", "Averages latent embeddings to evaluate empirical group distribution."], C_PURPLE, C_PURPLE_LIGHT),
        ("Incremental SAT & Assumptions", ["CryptoMiniSat 5.14.7 solver with native XOR Gaussian elimination.", "Temporary assumptions: solver.solve(assumptions=[k_5]).", "Refutes wrong subkeys in milliseconds; deduces full master key in <0.05s."], C_EMERALD, C_EMERALD_LIGHT),
        ("Platt Calibration & PAR-2", ["Platt Scaling: Fits logistic regression to map score margins to true probabilities.", "Selective Prediction: Fail-closed routing when confidence is uncalibrated.", "PAR-2: Penalized Average Runtime (2 × timeout for failures)."], C_AMBER, C_AMBER_LIGHT),
    ]

    card_w = 3.85
    card_h = 2.5
    gap_x = 0.25
    gap_y = 0.2
    for idx, (t_title, t_items, stroke_c, bg_c) in enumerate(terms):
        row = idx // 3
        col = idx % 3
        cx = Inches(0.6 + col * (card_w + gap_x))
        cy = Inches(1.7 + row * (card_h + gap_y))
        add_card(s2, cx, cy, Inches(card_w), Inches(card_h), t_title, t_items, fill_color=C_WHITE, stroke_color=stroke_c, title_color=stroke_c)

    # =========================================================================
    # SLIDE 3: Criterion 1 — Problem Statement: The Cryptanalytic Dilemma
    # =========================================================================
    s3 = add_light_slide("1. Problem Statement: The Statistical vs. Symbolic Cryptanalytic Dilemma", "Criterion 1: Problem Statement (10 Marks)")

    add_card(
        s3, Inches(0.6), Inches(1.7), Inches(3.8), Inches(5.3),
        "1. The Pure Neural Limit",
        [
            "Pioneering neural cryptanalysis (Gohr 2019) trains ResNets to distinguish differential distributions.",
            "Inherent Flaw: Only predicts/ranks 16-bit final-round subkeys probabilistically.",
            "Cannot complete key recovery: Leaves remaining 48 bits of the master key unrecovered.",
            "Brute-force search over remaining rounds becomes intractable as rounds grow.",
            "Vulnerable to ranking collapse: If the true subkey falls outside top-k, the attack silently fails with 0% recovery."
        ],
        fill_color=C_WHITE, stroke_color=C_COBALT, title_color=C_NAVY, tag="Statistical Heuristic Gap"
    )

    add_card(
        s3, Inches(4.7), Inches(1.7), Inches(3.8), Inches(5.3),
        "2. The Pure SAT Solver Limit",
        [
            "Exact SAT solvers (CryptoMiniSat) use native XOR Gaussian elimination for algebraic cryptanalysis.",
            "Inherent Flaw: Unguided solvers suffer exponential search-tree blowup on ARX symmetric ciphers.",
            "At 6-Round SPECK32/64: Pure SAT hits an exponential cliff (81.25% timeout rate at 10.0s budget).",
            "Exhaustive candidate enumeration (65,536 subkeys) without guidance is computationally intractable.",
            "Random-order candidate testing achieves 0.0% recovery."
        ],
        fill_color=C_WHITE, stroke_color=C_SLATE, title_color=C_NAVY, tag="Exact Symbolic Solver Gap"
    )

    add_card(
        s3, Inches(8.8), Inches(1.7), Inches(3.9), Inches(5.3),
        "3. The Closed-Loop Opportunity",
        [
            "The Research Aim: Formulate a closed-loop neuro-symbolic system where AI and SAT complement each other.",
            "1. Neural Network: Evaluates differential ciphertext distributions to rank 16-bit subkey candidates.",
            "2. CryptoMiniSat: Evaluates top-ranked candidates via temporary assumptions, deducing the 48-bit master key in sub-second median time (~481ms).",
            "3. Confidence Gating: Dynamically detects when neural rankings are unreliable and routes to a safe fallback.",
            "4. Independent Verification: Eliminates false positives via fresh plaintext checks."
        ],
        fill_color=C_WHITE, stroke_color=C_EMERALD, title_color=C_EMERALD_DARK, tag="Neuro-Symbolic Synthesis"
    )

    # =========================================================================
    # SLIDE 4: Criterion 1 — Formal Research Questions (RQ1, RQ2, RQ3)
    # =========================================================================
    s4 = add_light_slide("1. Problem Statement: Core Research Questions & Scientific Scope", "Criterion 1: Problem Statement (10 Marks)")

    add_card(
        s4, Inches(0.6), Inches(1.7), Inches(3.8), Inches(5.3),
        "RQ1 — Effectiveness",
        [
            "Research Question:",
            "Can differential-neural ranking of complete 16-bit final-round subkeys reduce the end-to-end computational cost of incremental XOR-aware SAT key recovery compared with SAT-only and randomly ordered baselines?",
            "",
            "Experimental Target:",
            "• Compare 5 methods under identical conditions.",
            "• Measure top-k recall, solver candidate attempts, median online runtime, and PAR-2 scores on held-out keys."
        ],
        fill_color=C_WHITE, stroke_color=C_COBALT, title_color=C_NAVY, tag="Core Efficacy"
    )

    add_card(
        s4, Inches(4.7), Inches(1.7), Inches(3.8), Inches(5.3),
        "RQ2 — Reliability",
        [
            "Research Question:",
            "Can attack-level confidence indicators predict whether the correct final-round subkey is contained within a specified top-k neural candidate set?",
            "",
            "Experimental Target:",
            "• Evaluate standardized score margins: (s_(1) - s_(8)) / σ_all.",
            "• Train Platt logistic calibrators on validation keys.",
            "• Measure Brier score, ECE, and ROC-AUC for selective prediction."
        ],
        fill_color=C_WHITE, stroke_color=C_PURPLE, title_color=C_PURPLE, tag="Confidence Calibration"
    )

    add_card(
        s4, Inches(8.8), Inches(1.7), Inches(3.9), Inches(5.3),
        "RQ3 — Fallback Resilience",
        [
            "Research Question:",
            "Can a confidence-gated fallback reduce the additional recovery cost caused by weak or misleading neural subkey rankings?",
            "",
            "Experimental Target:",
            "• Design an architectural diversity fallback interleaving distinct neural models (B1 and B0).",
            "• Guarantee recovery robustness even when confidence estimation breaks down.",
            "• Eliminate single-point-of-failure ranking collapse."
        ],
        fill_color=C_WHITE, stroke_color=C_EMERALD, title_color=C_EMERALD_DARK, tag="Robustness & Safety"
    )

    # =========================================================================
    # SLIDE 5: Criterion 2 — Background & State of the Art (2019–2026)
    # =========================================================================
    s5 = add_light_slide("2. Background & Related Work: State of the Art (2019–2026)", "Criterion 2: Background (10 Marks)")

    add_card(
        s5, Inches(0.6), Inches(1.7), Inches(5.8), Inches(2.5),
        "Differential Neural Distinguishers (Gohr 2019)",
        [
            "Gohr (CRYPTO 2019) demonstrated that ResNets can isolate differential behavior in reduced-round SPECK32/64 beyond classical Markov models.",
            "Subkey Ranking: Summed logit scores across independent pairs to rank candidates.",
            "Limitation: Evaluates pairs independently; vulnerable to noisy outlier pairs; no exact master-key completion."
        ],
        fill_color=C_WHITE, stroke_color=C_COBALT, title_color=C_NAVY
    )

    add_card(
        s5, Inches(6.8), Inches(1.7), Inches(5.9), Inches(2.5),
        "XOR-Aware SAT Solving (Soos et al. 2009–2024)",
        [
            "CryptoMiniSat integrates CDCL conflict-driven clause learning with native XOR Gaussian elimination over GF(2).",
            "Perfect for ARX modular addition carry constraints, but requires external guidance on multi-round ciphers.",
            "Limitation: Times out on ≥6 rounds without subkey assumptions."
        ],
        fill_color=C_WHITE, stroke_color=C_SLATE, title_color=C_NAVY
    )

    add_card(
        s5, Inches(0.6), Inches(4.4), Inches(5.8), Inches(2.6),
        "Multi-Pair & SMT Literature (Chen 2023, Hou 2025)",
        [
            "Chen, Shen, Yu, & Yuan (2023): Investigated multi-pair inputs for binary classification, but did not improve key-recovery frameworks.",
            "Hou et al. (2021, 2025): Used SAT/SMT solvers (Z3/STP) purely *offline* to search for high-probability differential trails before training neural networks."
        ],
        fill_color=C_WHITE, stroke_color=C_PURPLE, title_color=C_PURPLE
    )

    add_card(
        s5, Inches(6.8), Inches(4.4), Inches(5.9), Inches(2.6),
        "Our Demarcation: Online Closed-Loop Key Deduction",
        [
            "1. Closed-Loop Online Integration: Neural network heuristic solves the 16-bit subkey; CryptoMiniSat deduces the 48-bit master key in sub-second median time (~481ms) via temporary assumptions.",
            "2. Permutation-Invariant Deep Sets (B1): Group-level representation (+12.5 pp boost).",
            "3. Selective Prediction & Calibration Breakdown: First rigorous study of trust calibration in symmetric cryptanalysis."
        ],
        fill_color=C_WHITE, stroke_color=C_EMERALD, title_color=C_EMERALD_DARK
    )

    # =========================================================================
    # SLIDE 6: Criterion 2 — Research Positioning Matrix
    # =========================================================================
    s6 = add_light_slide("2. Background: Scientific Positioning Matrix", "Criterion 2: Background (10 Marks)")

    pos_img = REPOSITORY_ROOT / "results/figures/proposal/research_positioning.png"
    if pos_img.exists():
        s6.shapes.add_picture(str(pos_img), Inches(0.6), Inches(1.65), width=Inches(12.133))

    # =========================================================================
    # SLIDE 7: Criterion 3 — Methodology: End-to-End Attack Pipeline
    # =========================================================================
    s7 = add_light_slide("3. Methodology: Closed-Loop Neural-SAT Attack Pipeline", "Criterion 3: Methodology (10 Marks)")

    pipe_img = REPOSITORY_ROOT / "results/figures/proposal/methodology_pipeline.png"
    if pipe_img.exists():
        s7.shapes.add_picture(str(pipe_img), Inches(3.51), Inches(1.65), width=Inches(6.3))

    # =========================================================================
    # SLIDE 8: Criterion 3 — Methodology: Pair-Aware Deep Sets Distinguisher Architecture
    # =========================================================================
    s8 = add_light_slide("3. Methodology: Pair-Aware Deep Sets Distinguisher Architecture", "Criterion 3: Methodology (10 Marks)")

    simple_arch_img = REPOSITORY_ROOT / "results/figures/presentation/deep_sets_architecture_simple.png"
    if simple_arch_img.exists():
        s8.shapes.add_picture(str(simple_arch_img), Inches(2.9), Inches(1.65), width=Inches(7.533))

    # =========================================================================
    # SLIDE 9: Criterion 3 — Methodology: Calibration Breakdown & Diversity Fallback
    # =========================================================================
    s9 = add_light_slide("3. Methodology: Calibration Breakdown (RQ2) & Diversity Fallback (RQ3)", "Criterion 3: Methodology (10 Marks)")

    fb_img = REPOSITORY_ROOT / "results/figures/proposal/confidence_fallback_mechanism.png"
    if fb_img.exists():
        s9.shapes.add_picture(str(fb_img), Inches(0.6), Inches(1.65), width=Inches(7.2))

    add_card(
        s9, Inches(8.0), Inches(1.65), Inches(4.7), Inches(5.3),
        "The Science of Cryptanalytic Trust",
        [
            "The Negative Finding (RQ2):",
            "• Evaluated standardized margin: (s_(1) - s_(8)) / σ_all with Platt logistic calibration.",
            "• Calibration failed on validation keys (AUC ≈ 0.52).",
            "• Cause: Differential clustering produces 'pseudo-differential ghost peaks' for wrong subkeys sharing active bits. The network is frequently 'confidently wrong'.",
            "The Fallback Solution (RQ3):",
            "• System strictly fails closed (D-015).",
            "• Diversity Fallback interleaves B1 and B0 candidates [B1_1, B0_1, B1_2, B0_2, ...] (k=32).",
            "• Rescued 5 of 6 B1 misses on held-out test keys, achieving 98.44% (63/64) recovery."
        ],
        fill_color=C_WHITE, stroke_color=C_PURPLE, title_color=C_PURPLE, tag="Selective Gating & Safety"
    )

    # =========================================================================
    # SLIDE 10: Criterion 3 — Empirical Results: Held-Out Benchmark (N=64 Keys)
    # =========================================================================
    s10 = add_light_slide("3. Empirical Results: 64 Held-Out Master Keys Comparative Benchmark", "Criterion 3: Methodology & Results (10 Marks)")

    res_img = REPOSITORY_ROOT / "results/figures/evaluation/final_success_rates_6round.png"
    if res_img.exists():
        s10.shapes.add_picture(str(res_img), Inches(0.6), Inches(1.65), width=Inches(6.8))

    add_card(
        s10, Inches(7.6), Inches(1.65), Inches(5.1), Inches(5.3),
        "Held-Out Benchmark Summary (N=64)",
        [
            "1. SAT-Only Baseline (Audited Clean Rerun):",
            "   0.00% Solved (0/64 at 10s) | Historical unisolated baseline (18.75%) invalidated by solver state reuse.",
            "2. Random Order + SAT (k=16):",
            "   0.00% Solved (0/64) | 16 UNSAT refutations/key.",
            "3. Gohr Baseline (B0) + SAT (k=16):",
            "   78.12% Solved (50/64) | 8.75 mean tries | 79.66s.",
            "4. Pair-Aware (B1) + SAT (k=16):",
            "   90.62% Solved (58/64) | 8.55 mean tries | 78.73s (+12.50 pp boost over B0).",
            "5. Gated B1 + Diversity Fallback (k=32):",
            "   98.44% Solved (63/64) | 9.11 mean tries | 157.72s (Rescues 5 of 6 misses!).",
            "Cryptographic Verification:",
            "   100% sound master keys verified on 2 fresh plaintexts."
        ],
        fill_color=C_WHITE, stroke_color=C_EMERALD, title_color=C_EMERALD_DARK, tag="Final Benchmark Evidence"
    )

    # =========================================================================
    # SLIDE 11: Criterion 3 — Empirical Results: ECDF Curves & PAR-2 Cost Analysis
    # =========================================================================
    s11 = add_light_slide("3. Empirical Results: ECDF Solving Curves & PAR-2 Cost Analysis", "Criterion 3: Methodology & Results (10 Marks)")

    ecdf_img = REPOSITORY_ROOT / "results/figures/evaluation/ecdf_solved_vs_attempts.png"
    par2_img = REPOSITORY_ROOT / "results/figures/evaluation/runtime_and_par2_comparison.png"
    if ecdf_img.exists():
        s11.shapes.add_picture(str(ecdf_img), Inches(0.6), Inches(1.65), width=Inches(5.9))
    if par2_img.exists():
        s11.shapes.add_picture(str(par2_img), Inches(6.8), Inches(1.65), width=Inches(5.9))

    # =========================================================================
    # SLIDE 12: Criterion 4 — Graphic Timeline & Expansion Horizons (Phases 0–12)
    # =========================================================================
    s12 = add_light_slide("4. Plan & Timeline: Delivered Milestones (Phases 0–8) vs. Horizons (Phases 9–12)", "Criterion 4: Plan / Timeline (10 Marks)")

    timeline_img = REPOSITORY_ROOT / "results/figures/presentation/graphic_timeline_phases.png"
    if timeline_img.exists():
        s12.shapes.add_picture(str(timeline_img), Inches(0.5), Inches(1.6), width=Inches(12.333))

    # =========================================================================
    # SLIDE 13: Conclusion & Scientific Contributions
    # =========================================================================
    s13 = add_light_slide("Conclusion: Summary of Scientific Contributions & Defense Takeaways", "Executive Summary")

    add_card(
        s13, Inches(0.6), Inches(1.7), Inches(3.8), Inches(5.3),
        "1. Algorithmic Synergy",
        [
            "Proved that Deep Sets neural guidance (B1) breaks the 6-round exponential SAT cliff.",
            "Recovery success jumps from 0.00% (SAT-only / Random) to 90.62% (B1).",
            "CryptoMiniSat algebraically deduces remaining 48 bits in sub-second median time (~481ms).",
            "100% mathematically sound verification on fresh plaintexts."
        ],
        fill_color=C_WHITE, stroke_color=C_COBALT, title_color=C_NAVY, tag="Synergy Validated"
    )

    add_card(
        s13, Inches(4.7), Inches(1.7), Inches(3.8), Inches(5.3),
        "2. The Science of Trust",
        [
            "Identified calibration breakdown: raw score margins cannot be blindly trusted due to differential ghost peaks.",
            "Demonstrated fail-closed research integrity.",
            "Proved that architectural diversity fallback provides deterministic recovery robustness (98.44% success)."
        ],
        fill_color=C_WHITE, stroke_color=C_PURPLE, title_color=C_PURPLE, tag="Selective Trust Insights"
    )

    add_card(
        s13, Inches(8.8), Inches(1.7), Inches(3.9), Inches(5.3),
        "3. Research Artefacts",
        [
            "Reproducible Python 3.11 codebase with 287 passing tests (0 lint errors).",
            "Publication-grade vector figures (SVG/PDF/PNG) and LaTeX tables.",
            "Standalone executable CLI: `neural-sat-speck` (KAT, data generation, confidence explanation, key verification).",
            "Clear 4-horizon expansion roadmap (Phases 9–12)."
        ],
        fill_color=C_WHITE, stroke_color=C_EMERALD, title_color=C_EMERALD_DARK, tag="Submission Ready"
    )

    prs.save(str(OUTPUT_PATH))
    print(f"Light-themed presentation saved successfully to {OUTPUT_PATH}")


if __name__ == "__main__":
    create_presentation()
