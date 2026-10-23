"""Etapa 4: acabamento com ffmpeg (interpolação de quadros, upscale e emenda opcional com crossfade)."""
from __future__ import annotations

import subprocess


def _ff(cfg, *args):
    cmd = [cfg["ffmpeg"], "-hide_banner", "-loglevel", "error", "-y", *map(str, args)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"ffmpeg falhou:\n{' '.join(cmd)}\n{r.stderr}")


def _duration(cfg, path) -> float:
    probe = cfg["ffmpeg"].replace("ffmpeg", "ffprobe")
    r = subprocess.run([probe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, check=True)
    return float(r.stdout)


def run(project, clips=None, force=False):
    cfg = project.finish
    src = sorted(project.dir("clips").glob("*.mp4"))
    if clips:
        src = [p for p in src if any(p.stem.startswith(c) for c in clips)]
    if not src:
        raise SystemExit("finish: nenhum clipe em outputs/clips (rode `animate` antes)")

    vf = []
    if cfg["interpolation"] == "minterpolate":
        vf.append(f"minterpolate=fps={cfg['fps']}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1")
    else:
        vf.append(f"fps={cfg['fps']}")
    # cabe em width x height mantendo a proporção (barras pretas se preciso)
    vf.append(f"scale={cfg['width']}:{cfg['height']}:force_original_aspect_ratio=decrease:flags=lanczos,"
              f"pad={cfg['width']}:{cfg['height']}:(ow-iw)/2:(oh-ih)/2,setsar=1")
    outs = []
    for p in src:
        out = project.dir("final") / p.name
        if force or not out.exists():
            _ff(cfg, "-i", p, "-vf", ",".join(vf), "-c:v", "libx264", "-crf", cfg["crf"], "-preset", "slow",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", out)
            print("ok:", out)
        outs.append(out)

    if cfg["concat"]:
        # um clipe por vídeo, na ordem do YAML: a seed `keep_seed` (padrão: a primeira da lista)
        from .animate import clip_path
        outs = []
        for v in project.videos:
            seed = v.get("keep_seed", v["seeds"][0])
            f = project.dir("final") / clip_path(project, v["name"], v["mode"], seed).name
            if f.exists():
                outs.append(f)
            else:
                print(f"aviso: emenda sem '{v['name']}' (falta {f.name})")
    if cfg["concat"] and len(outs) > 1:
        x, g, acc, prev = cfg["xfade"], [], 0.0, "[0:v]"
        for k in range(1, len(outs)):
            acc += _duration(cfg, outs[k - 1]) - x
            g.append(f"{prev}[{k}:v]xfade=transition=fade:duration={x}:offset={acc:.3f}[x{k}]")
            prev = f"[x{k}]"
        inputs = [a for o in outs for a in ("-i", o)]
        final = project.dir("final") / f"{project.path.stem}_completo.mp4"
        _ff(cfg, *inputs, "-filter_complex", ";".join(g), "-map", prev, "-c:v", "libx264", "-crf", cfg["crf"],
            "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", final)
        print("ok:", final)
