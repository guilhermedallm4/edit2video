"""Utilidades: registro de tempos e seeds, escolha de variações, divisão entre GPUs."""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else (default if default is not None else {})


def write_json(path: Path, data):
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False))


@contextmanager
def timed(project, stage: str, item: str, **info):
    t = time.time()
    yield info
    rec = {"stage": stage, "item": item, "seconds": round(time.time() - t, 1), "when": time.strftime("%F %T"), **info}
    with open(project.out / "tempos.jsonl", "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[{stage}] {item}: {rec['seconds']} s", flush=True)


def split(items: list, part: str) -> list:
    """'i/n' -> i-ésima de n partes (para rodar um processo por GPU)."""
    i, n = map(int, part.split("/"))
    return items[i::n]


def image_path(project, name: str, variation: int) -> Path:
    return project.dir("images") / f"{name}_v{variation}.png"


def selection(project) -> dict:
    return read_json(project.out / "selection.json")


def resolve_image(project, name: str) -> Path:
    """Nome de imagem gerada -> variação escolhida; nome de referência -> o arquivo original."""
    if name in project.references:
        return project.references[name]
    sel = selection(project)
    if name not in sel:
        raise SystemExit(f"imagem '{name}' sem variação escolhida: rode `edit2video review --choose {name}=N` "
                         f"(ou --auto para usar a v1)")
    return image_path(project, name, int(sel[name]))
