"""Cria referências sintéticas (desenhadas por código) para testar o pipeline sem fotos de ninguém.

    python scripts/make_demo_refs.py examples/demo/refs
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

out = Path(sys.argv[1] if len(sys.argv) > 1 else "examples/demo/refs")
out.mkdir(parents=True, exist_ok=True)

# 1) um "personagem": boneco simples de massinha (esfera laranja com olhos)
im = Image.new("RGB", (768, 768), "#dfe7ef")
d = ImageDraw.Draw(im)
d.ellipse((184, 160, 584, 560), fill="#f08a24", outline="#7a3e00", width=6)
d.ellipse((290, 290, 350, 360), fill="white"); d.ellipse((418, 290, 478, 360), fill="white")
d.ellipse((310, 315, 335, 345), fill="black"); d.ellipse((438, 315, 463, 345), fill="black")
d.arc((300, 360, 470, 470), 20, 160, fill="#7a3e00", width=8)
d.rectangle((0, 600, 768, 768), fill="#9b7653")
im.save(out / "personagem.png")

# 2) uma "roupa": chapéu de chef desenhado
im = Image.new("RGB", (768, 768), "white")
d = ImageDraw.Draw(im)
for x in (230, 330, 430):
    d.ellipse((x, 180, x + 160, 340), fill="#f4f4f4", outline="#999", width=4)
d.rectangle((260, 300, 520, 470), fill="#f4f4f4", outline="#999", width=4)
im.save(out / "chapeu.png")
print("ok:", sorted(p.name for p in out.iterdir()))
