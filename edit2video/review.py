"""Etapa 2: folha de contato (review.html + review.png) e escolha da variação de cada imagem."""
from __future__ import annotations

import html

from .common import image_path, selection, write_json


def choose(project, pairs=(), auto=False):
    sel = selection(project)
    for item in pairs:
        name, v = item.split("=")
        if name not in {s["name"] for s in project.images}:
            raise SystemExit(f"imagem desconhecida: {name}")
        sel[name] = int(v)
    if auto:
        for s in project.images:
            sel.setdefault(s["name"], 1)
    write_json(project.out / "selection.json", sel)
    print("selection.json:", sel)


def sheet(project):
    from PIL import Image, ImageDraw

    sel = selection(project)
    rows, thumbs = [], []
    for s in project.images:
        vs = sorted(p for p in project.dir("images").glob(f"{s['name']}_v*.png"))
        cells = []
        for p in vs:
            v = int(p.stem.rsplit("_v", 1)[1])
            mark = " escolhida" if sel.get(s["name"]) == v else ""
            cells.append(f'<figure class="{mark}"><img src="images/{p.name}" loading="lazy"><figcaption>v{v}</figcaption></figure>')
            thumbs.append((s["name"], v, p))
        rows.append(f"<section><h2>{html.escape(s['name'])}</h2><p>{html.escape(s['prompt'][:300])}…</p>"
                    f"<div class='row'>{''.join(cells) or '<i>sem imagens ainda</i>'}</div></section>")
    (project.out / "review.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Revisão</title><style>"
        "body{font-family:sans-serif;background:#111;color:#eee;margin:16px}.row{display:flex;gap:8px;flex-wrap:wrap}"
        "figure{margin:0;flex:1 1 280px;max-width:480px}img{width:100%;border:3px solid transparent}"
        ".escolhida img{border-color:#4c4}figcaption{text-align:center}p{color:#aaa}</style>"
        "<h1>Escolha uma variação por imagem</h1>" + "".join(rows))

    if thumbs:
        tw, th = 320, 240
        per_row = max(len([t for t in thumbs if t[0] == s["name"]]) for s in project.images) or 1
        names = [s["name"] for s in project.images if any(t[0] == s["name"] for t in thumbs)]
        g = Image.new("RGB", (per_row * tw, len(names) * th), "#111")
        d = ImageDraw.Draw(g)
        for r, name in enumerate(names):
            for c, (_, v, p) in enumerate([t for t in thumbs if t[0] == name]):
                im = Image.open(p).convert("RGB")
                im.thumbnail((tw - 4, th - 4))
                g.paste(im, (c * tw + 2, r * th + 2))
                d.text((c * tw + 8, r * th + 6), f"{name} v{v}", fill="#ff0")
        g.save(project.out / "review.png")
    print("ok:", project.out / "review.html")


def missing(project) -> list:
    sel = selection(project)
    needed = {v["first"] for v in project.videos} | {v["last"] for v in project.videos if v.get("last")}
    needed |= {r for s in project.images for r in s.get("refs") or []}
    return [s["name"] for s in project.images if s["name"] in needed and s["name"] not in sel]
