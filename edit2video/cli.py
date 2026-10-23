"""Linha de comando: edit2video {edit,review,animate,finish,run} projeto.yaml [opções]."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from . import config


def _parallel(project_file: str, stage: str, gpus: str, extra: list[str]):
    """Um processo independente por GPU, cada um com uma parte dos itens (sem sincronizar as GPUs)."""
    gl = gpus.split(",")
    procs = [subprocess.Popen([sys.executable, "-m", "edit2video.cli", stage, project_file, *extra,
                               "--part", f"{i}/{len(gl)}"], env={**os.environ, "CUDA_VISIBLE_DEVICES": g})
             for i, g in enumerate(gl)]
    if any(p.wait() for p in procs):
        raise SystemExit(f"{stage} falhou em alguma GPU")


def _levels(proj) -> list[list[str]]:
    """Agrupa as imagens: nível 0 usa só referências; nível k usa imagens de níveis anteriores."""
    names = {s["name"] for s in proj.images}
    level = {}
    for s in proj.images:  # a validação garante que refs geradas vêm antes no YAML
        deps = [r for r in s.get("refs") or [] if r in names]
        level[s["name"]] = 1 + max((level[d] for d in deps), default=-1)
    return [[n for n, l in level.items() if l == k] for k in range(max(level.values(), default=-1) + 1)]


def _opts(a, *names) -> list[str]:
    out = []
    for n in names:
        v = getattr(a, n, None)
        if v in (None, False, []):
            continue
        flag = "--" + n
        out += [flag] if v is True else [flag, *map(str, v if isinstance(v, list) else [v])]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="edit2video", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, gpu=True):
        p.add_argument("project", help="arquivo YAML do projeto")
        p.add_argument("--only", nargs="*", help="só estes nomes")
        p.add_argument("--force", action="store_true", help="refaz mesmo se a saída existir")
        if gpu:
            p.add_argument("--part", default="0/1", help=argparse.SUPPRESS)
            p.add_argument("--gpus", help="ex.: 0,1 -> um processo por GPU")

    p = sub.add_parser("edit", help="gera as imagens editadas (Qwen-Image-Edit)")
    common(p)
    p.add_argument("--variations", type=int, help="sobrescreve o nº de variações por imagem")

    p = sub.add_parser("review", help="folha de contato e escolha das variações")
    common(p, gpu=False)
    p.add_argument("--choose", nargs="*", default=[], metavar="NOME=V")
    p.add_argument("--auto", action="store_true", help="v1 para quem ainda não tem escolha")

    p = sub.add_parser("animate", help="gera os vídeos (Wan2.2-I2V)")
    common(p)
    p.add_argument("--frames", type=int, help="nº de quadros (vira 4k+1)")
    p.add_argument("--mode", choices=list(config.MODES))
    p.add_argument("--seeds", type=int, nargs="+")

    p = sub.add_parser("finish", help="interpola, faz upscale e (opcional) emenda")
    common(p, gpu=False)

    p = sub.add_parser("run", help="edit -> review -> animate -> finish (para na revisão se faltar escolha)")
    common(p)
    p.add_argument("--auto", action="store_true", help="escolhe a v1 sem parar na revisão")

    a = ap.parse_args(argv)
    proj = config.load(a.project)
    from . import review

    if a.cmd == "edit":
        if a.gpus:
            return _parallel(a.project, "edit", a.gpus, _opts(a, "only", "force", "variations"))
        from . import edit
        edit.run(proj, a.only, a.part, a.force, a.variations)
    elif a.cmd == "review":
        if a.choose or a.auto:
            review.choose(proj, a.choose, a.auto)
        review.sheet(proj)
    elif a.cmd == "animate":
        if a.gpus:
            return _parallel(a.project, "animate", a.gpus, _opts(a, "only", "force", "frames", "mode", "seeds"))
        from . import animate
        animate.run(proj, a.only, a.part, a.force, a.frames, a.mode, a.seeds)
    elif a.cmd == "finish":
        from . import finish
        finish.run(proj, a.only, a.force)
    elif a.cmd == "run":
        from . import animate, edit, finish
        for k, names in enumerate(_levels(proj)):
            if k > 0:  # este nível usa imagens geradas: precisa da escolha das anteriores
                if a.auto:
                    review.choose(proj, auto=True)
                deps = {r for s in proj.images if s["name"] in names for r in s.get("refs") or []}
                falta = [n for n in deps if n in {s["name"] for s in proj.images} and n not in review.selection(proj)]
                if falta:
                    review.sheet(proj)
                    raise SystemExit(f"Escolha as variações de {falta} (review --choose) e rode `run` de novo.")
            if a.gpus:
                _parallel(a.project, "edit", a.gpus, ["--only", *names] + (["--force"] if a.force else []))
            else:
                edit.run(proj, names, force=a.force)
        if a.auto:
            review.choose(proj, auto=True)
        review.sheet(proj)
        falta = review.missing(proj)
        if falta:
            raise SystemExit(f"Revise {proj.out / 'review.html'} e escolha, por exemplo:\n  edit2video review "
                             f"{a.project} --choose " + " ".join(f"{n}=1" for n in falta))
        if a.gpus:
            _parallel(a.project, "animate", a.gpus, ["--force"] if a.force else [])
        else:
            animate.run(proj, force=a.force)
        finish.run(proj, force=a.force)


if __name__ == "__main__":
    main()
