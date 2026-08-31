"""Generate custom high-impact vector graphics for the presentation in a crisp Light Theme.

Creates:
1. graphic_timeline_phases.svg (Light-themed Phases 0 to 12 timeline with 'Where We Are At')
"""

from __future__ import annotations

import subprocess
from html import escape
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPOSITORY_ROOT / "results/figures/presentation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Elegant Light Academic Palette
C = {
    "bg": "#F8FAFC",
    "card": "#FFFFFF",
    "card_border": "#CBD5E1",
    "navy": "#0F2747",
    "navy_light": "#1E3A8A",
    "cobalt": "#2563EB",
    "cobalt_light": "#DBEAFE",
    "emerald": "#16A34A",
    "emerald_dark": "#166534",
    "emerald_light": "#DCFCE7",
    "amber": "#D97706",
    "amber_light": "#FEF3C7",
    "purple": "#7C3AED",
    "purple_light": "#F3E8FF",
    "rose": "#E11D48",
    "rose_light": "#FFE4E6",
    "slate": "#475569",
    "slate_light": "#F1F5F9",
    "white": "#FFFFFF",
    "text": "#0F172A",
    "text_muted": "#64748B",
    "grid": "#E2E8F0",
}


def _svg_wrap(w: int, h: int, body: str, bg: str = C["bg"]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">\n'
        f'<rect width="{w}" height="{h}" fill="{bg}"/>\n'
        f'{body}\n'
        f'</svg>\n'
    )


def _rect(x: float, y: float, w: float, h: float, fill: str, stroke: str, stroke_w: float = 1.0, rx: float = 8.0, dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_w:.1f}"{dash_attr}/>'


def _text(x: float, y: float, text: str, size: int = 14, color: str = C["text"], weight: str = "normal", anchor: str = "start", style: str = "normal") -> str:
    style_attr = f' font-style="{style}"' if style != "normal" else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" font-size="{size}px" fill="{color}" font-weight="{weight}" text-anchor="{anchor}"{style_attr}>{escape(str(text))}</text>'


def _line(x1: float, y1: float, x2: float, y2: float, color: str, width: float = 2.0, dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width:.1f}"{dash_attr}/>'


def generate_timeline() -> str:
    w, h = 1800, 950
    body = []

    # Title Banner (Deep Navy Accent on Light Canvas)
    body.append(_rect(40, 20, w - 80, 80, fill=C["navy"], stroke=C["navy"], stroke_w=1.5, rx=10))
    body.append(_text(60, 50, "RESEARCH MILESTONE TIMELINE & STRATEGIC EXPANSION ROADMAP", size=12, color=C["cobalt_light"], weight="bold"))
    body.append(_text(60, 82, "Delivered Foundation (Phases 0–8) vs. Future Research Expansion (Phases 9–12)", size=22, color=C["white"], weight="bold"))
    body.append(_text(w - 60, 68, "COMP6002 Master's Research Artefact", size=14, color=C["cobalt_light"], anchor="end"))

    # Connecting Spine (Horizontal Rails)
    # Rail 1 (Completed): Y = 310
    body.append(_line(80, 310, 1720, 310, color=C["emerald"], width=6))
    # Rail 2 (Future): Y = 680
    body.append(_line(80, 680, 1720, 680, color=C["purple"], width=6, dash="8,8"))

    # Section Headers
    body.append(_rect(60, 120, 480, 45, fill=C["emerald_light"], stroke=C["emerald"], stroke_w=2, rx=8))
    body.append(_text(300, 148, "✓ DELIVERED CORE MILESTONES (PHASES 0 TO 8)", size=14, color=C["emerald_dark"], weight="bold", anchor="middle"))

    # WHERE WE ARE AT BADGE
    body.append(_rect(1200, 120, 540, 45, fill=C["amber_light"], stroke=C["amber"], stroke_w=2, rx=8))
    body.append(_text(1470, 148, "📍 CURRENT STATUS: PHASE 8 COMPLETE (SUBMISSION-READY)", size=13, color=C["navy"], weight="bold", anchor="middle"))

    # Row 1: Completed Phases (0-1, 2-3, 4-5, 6A-B, 6C, 7, 8)
    completed_nodes = [
        ("Phase 0 & 1", "Cipher & Datasets", ["Scalar SPECK32/64 KAT", "16-bit word primitives", "Dataset v1 deterministic export"], "✓ Complete", C["white"], C["emerald"]),
        ("Phase 2 & 3", "SAT & Baseline Neural", ["CryptoMiniSat 5.14.7 model", "Gohr ResNet baseline (B0)", "Zero-leakage seed isolation"], "✓ Complete", C["white"], C["emerald"]),
        ("Phase 4 & 5", "Ranking & Incremental SAT", ["65,536-domain ranker", "solve(assumptions=[k5])", "2 fresh-pair verification"], "✓ Complete", C["white"], C["emerald"]),
        ("Phase 6A & 6B", "Deep Sets B1 Promotion", ["Permutation-invariant group model", "+14.06 pp recall gain over B0", "22.1% lower solver attempts"], "✓ Complete", C["white"], C["emerald"]),
        ("Phase 6C", "Calibration & Fallback", ["Standardized margin metric", "Calibration breakdown unmasked", "Fail-closed diversity routing"], "✓ Complete", C["white"], C["emerald"]),
        ("Phase 7", "Held-Out Evaluation", ["64 unseen test keys (seed 98001)", "98.44% (63/64) key recovery", "Rescued 5 of 6 B1 misses"], "✓ Complete", C["white"], C["emerald"]),
        ("Phase 8", "Artefacts & CLI", ["Vector SVG/PDF publication plots", "LaTeX & Markdown tables", "neural-sat-speck executable CLI"], "✓ Complete", C["white"], C["emerald"]),
    ]

    card_w = 220
    gap = 20
    for idx, (p_title, p_sub, bullets, p_status, bg_c, stroke_c) in enumerate(completed_nodes):
        cx = 60 + idx * (card_w + gap)
        cy = 185
        body.append(_rect(cx, cy, card_w, 240, fill=bg_c, stroke=stroke_c, stroke_w=2, rx=10))

        # Node on rail
        body.append(_rect(cx + card_w/2 - 16, 294, 32, 32, fill=C["emerald"], stroke=C["white"], stroke_w=2, rx=16))
        body.append(_text(cx + card_w/2, 315, "✓", size=16, color=C["white"], weight="bold", anchor="middle"))

        # Card Content
        body.append(_text(cx + 15, cy + 28, p_title, size=14, color=C["emerald_dark"], weight="bold"))
        body.append(_text(cx + 15, cy + 48, p_sub, size=11, color=C["cobalt"], weight="bold"))
        body.append(_line(cx + 12, cy + 60, cx + card_w - 12, cy + 60, color=C["grid"], width=1))

        for b_idx, bullet in enumerate(bullets):
            body.append(_text(cx + 12, cy + 82 + b_idx * 20, f"• {bullet}", size=10, color=C["slate"]))

        # Status badge at bottom
        body.append(_rect(cx + 15, cy + 195, card_w - 30, 26, fill=C["emerald_light"], stroke=C["emerald"], stroke_w=1, rx=6))
        body.append(_text(cx + card_w/2, cy + 213, p_status, size=11, color=C["emerald_dark"], weight="bold", anchor="middle"))

    # Row 2: Future Expansion Phases (9, 10, 11, 12)
    body.append(_rect(60, 480, 560, 45, fill=C["purple_light"], stroke=C["purple"], stroke_w=2, rx=8))
    body.append(_text(340, 508, "🚀 FUTURE RESEARCH EXPANSION ROADMAP (PHASES 9 TO 12)", size=14, color=C["purple"], weight="bold", anchor="middle"))

    future_nodes = [
        ("Phase 9: Scalability Study", "Rounds 4, 5, 6 Phase Transitions", ["Map the full SAT cliff boundary", "R4: <0.02s, R5: ~0.5s, R6: timeouts", "Demonstrates exact neural necessity", "Target: IEEE S&P / TIFS Benchmark"], "Planned Horizon 1", C["white"], C["purple"]),
        ("Phase 10: Multi-Feature Gating", "3D Calibrated Selective Trust", ["Combine 1) Score Margin ratio,", "2) Jaccard Pair-Subset Stability,", "3) Softmax Distribution Entropy", "Target: AUC > 0.85 calibration"], "Planned Horizon 2", C["white"], C["purple"]),
        ("Phase 11: Bit-Sensitivity", "7-Round Subkey Pruning", ["Chen et al. sensitivity ranking", "Guess only top 10 subkey bits", "CryptoMiniSat deduces 54 bits", "Bypasses 2^32 candidate barrier"], "Planned Horizon 3", C["white"], C["purple"]),
        ("Phase 12: Neutral Bits", "8–11 Round Deep Sets Attacks", ["14 neutral bits for ΔP=(0x0040, 0x0000)", "Structured pools of 2^11 ciphertexts", "Full algebraic key deduction", "Target: ASIACRYPT / EUROCRYPT"], "Planned Horizon 4", C["white"], C["amber"]),
    ]

    card_w2 = 380
    gap2 = 30
    for idx, (p_title, p_sub, bullets, p_status, bg_c, stroke_c) in enumerate(future_nodes):
        cx = 60 + idx * (card_w2 + gap2)
        cy = 550
        body.append(_rect(cx, cy, card_w2, 260, fill=bg_c, stroke=stroke_c, stroke_w=2, rx=10, dash="6,4"))

        # Node on rail
        body.append(_rect(cx + card_w2/2 - 16, 664, 32, 32, fill=stroke_c, stroke=C["white"], stroke_w=2, rx=16))
        body.append(_text(cx + card_w2/2, 685, str(idx + 9), size=14, color=C["white"], weight="bold", anchor="middle"))

        # Card Content
        body.append(_text(cx + 20, cy + 32, p_title, size=16, color=C["purple"] if stroke_c == C["purple"] else C["amber"], weight="bold"))
        body.append(_text(cx + 20, cy + 56, p_sub, size=13, color=C["cobalt"], weight="bold"))
        body.append(_line(cx + 15, cy + 70, cx + card_w2 - 15, cy + 70, color=C["grid"], width=1))

        for b_idx, bullet in enumerate(bullets):
            body.append(_text(cx + 20, cy + 96 + b_idx * 24, f"• {bullet}", size=12, color=C["slate"]))

        # Status badge at bottom
        body.append(_rect(cx + 20, cy + 215, card_w2 - 40, 30, fill=C["slate_light"], stroke=stroke_c, stroke_w=1.5, rx=6))
        body.append(_text(cx + card_w2/2, cy + 235, p_status, size=12, color=C["purple"] if stroke_c == C["purple"] else C["amber"], weight="bold", anchor="middle"))

    # Bottom Invariant Footer
    body.append(_rect(40, 840, w - 80, 80, fill=C["white"], stroke=C["grid"], stroke_w=1.5, rx=8))
    body.append(_text(w/2, 870, "Scientific Invariant: All completed milestones (Phases 0–8) are 100% reproducible with 287 passing tests & zero data contamination.", size=13, color=C["emerald_dark"], weight="bold", anchor="middle"))
    body.append(_text(w/2, 895, "Expansion phases (9–12) follow pre-registered protocols to prevent post-hoc evaluation bias.", size=12, color=C["text_muted"], anchor="middle"))

    return _svg_wrap(w, h, "\n".join(body))


def main() -> None:
    timeline_svg = OUTPUT_DIR / "graphic_timeline_phases.svg"
    timeline_svg.write_text(generate_timeline(), encoding="utf-8")
    
    png_path = OUTPUT_DIR / "graphic_timeline_phases.png"
    pdf_path = OUTPUT_DIR / "graphic_timeline_phases.pdf"
    
    subprocess.run(["rsvg-convert", "-d", "300", "-p", "300", str(timeline_svg), "-o", str(png_path)], check=True)
    subprocess.run(["rsvg-convert", "-f", "pdf", str(timeline_svg), "-o", str(pdf_path)], check=True)
    print(f"Generated light-themed graphic timeline: {png_path} and {pdf_path}")


if __name__ == "__main__":
    main()
