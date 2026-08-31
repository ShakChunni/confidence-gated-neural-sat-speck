"""Generate a clean, spacious, minimalist architecture diagram for presentation slides.

Flow:
64 ciphertext pairs
      ↓
Shared neural encoder (same network for every pair)
      ↓
64 learned representations
      ↓
Mean pooling (combine the whole group)
      ↓
Small prediction head
      ↓
One candidate-subkey score
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
    "bg": "#FFFFFF",
    "card_bg": "#F8FAFC",
    "navy": "#0F2747",
    "cobalt": "#2563EB",
    "cobalt_light": "#EFF6FF",
    "cobalt_border": "#3B82F6",
    "emerald": "#16A34A",
    "emerald_light": "#F0FDF4",
    "emerald_border": "#22C55E",
    "purple": "#7C3AED",
    "purple_light": "#FAF5FF",
    "purple_border": "#A855F7",
    "amber": "#D97706",
    "amber_light": "#FFFBEB",
    "amber_border": "#F59E0B",
    "rose": "#E11D48",
    "rose_light": "#FFF1F2",
    "rose_border": "#F43F5E",
    "text": "#0F172A",
    "text_muted": "#475569",
    "arrow": "#64748B",
}


def _svg_wrap(w: int, h: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">\n'
        f'<rect width="{w}" height="{h}" fill="{C["bg"]}"/>\n'
        f'{body}\n'
        f'</svg>\n'
    )


def _rect(x: float, y: float, w: float, h: float, fill: str, stroke: str, stroke_w: float = 2.0, rx: float = 12.0) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_w:.1f}"/>'


def _text(x: float, y: float, text: str, size: int = 16, color: str = C["text"], weight: str = "normal", anchor: str = "middle") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" font-size="{size}px" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{escape(str(text))}</text>'


def _arrow_down(x: float, y1: float, y2: float) -> str:
    return (
        f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y2:.1f}" stroke="{C["arrow"]}" stroke-width="3" stroke-linecap="round"/>\n'
        f'<polygon points="{x-6:.1f},{y2-8:.1f} {x+6:.1f},{y2-8:.1f} {x:.1f},{y2:.1f}" fill="{C["arrow"]}"/>'
    )


def generate_simple_architecture() -> str:
    w, h = 1000, 720
    body = []

    # Clean Card Steps (Vertical Stack)
    card_w = 640
    card_h = 75
    cx = (w - card_w) / 2
    
    stages = [
        ("64 Ciphertext Pairs", "Evaluated together under trial subkey k", C["cobalt_light"], C["cobalt_border"], C["cobalt"]),
        ("Shared Neural Encoder", "Same 4-block ResNet processes every pair identically", C["emerald_light"], C["emerald_border"], C["emerald"]),
        ("64 Learned Representations", "Extracts 64 latent feature vectors (e₁, e₂, ..., e₆₄)", C["amber_light"], C["amber_border"], C["amber"]),
        ("Mean Pooling", "Combines the whole group into one vector: e_group = 1/64 ∑ eᵢ", C["purple_light"], C["purple_border"], C["purple"]),
        ("Small Prediction Head", "Compact 3-layer MLP evaluates joint group characteristics", C["rose_light"], C["rose_border"], C["rose"]),
        ("One Candidate-Subkey Score", "Outputs Score(k) — repeated for all 65,536 keys to rank candidates", C["card_bg"], C["navy"], C["navy"]),
    ]

    start_y = 30
    gap = 40

    for idx, (title, subtitle, fill_c, stroke_c, title_c) in enumerate(stages):
        cy = start_y + idx * (card_h + gap)
        # Card
        body.append(_rect(cx, cy, card_w, card_h, fill=fill_c, stroke=stroke_c, stroke_w=2.0, rx=12))
        # Title
        body.append(_text(w / 2, cy + 32, title, size=19, color=title_c, weight="bold"))
        # Subtitle
        body.append(_text(w / 2, cy + 56, subtitle, size=13, color=C["text_muted"]))

        # Arrow down to next step
        if idx < len(stages) - 1:
            arrow_y1 = cy + card_h + 4
            arrow_y2 = arrow_y1 + gap - 8
            body.append(_arrow_down(w / 2, arrow_y1, arrow_y2))

    return _svg_wrap(w, h, "\n".join(body))


def main() -> None:
    svg_path = OUTPUT_DIR / "deep_sets_architecture_simple.svg"
    svg_path.write_text(generate_simple_architecture(), encoding="utf-8")
    
    png_path = OUTPUT_DIR / "deep_sets_architecture_simple.png"
    pdf_path = OUTPUT_DIR / "deep_sets_architecture_simple.pdf"
    
    subprocess.run(["rsvg-convert", "-d", "300", "-p", "300", str(svg_path), "-o", str(png_path)], check=True)
    subprocess.run(["rsvg-convert", "-f", "pdf", str(svg_path), "-o", str(pdf_path)], check=True)
    print(f"Generated clean architecture graphic: {png_path}")


if __name__ == "__main__":
    main()
