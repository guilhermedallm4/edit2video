"""Etapa 3: vídeos com Wan2.2-I2V-A14B a partir das imagens escolhidas.

`first` vira o 1º quadro; `last` (opcional) fixa o último quadro e o modelo inventa o movimento entre os dois.
"""
from __future__ import annotations

from .common import read_json, resolve_image, split, timed, write_json


def clip_path(project, name: str, mode: str, seed: int):
    return project.dir("clips") / f"{name}__{mode}_s{seed}.mp4"


def load_pipeline(models: dict, lightning: bool, cpu_offload: bool):
    import torch
    from diffusers import WanImageToVideoPipeline
    from huggingface_hub import hf_hub_download

    pipe = WanImageToVideoPipeline.from_pretrained(models["video"], torch_dtype=torch.bfloat16)
    if lightning:
        L = models["lightning"]
        # uma LoRA por especialista: transformer (ruído alto) e transformer_2 (ruído baixo)
        pipe.load_lora_weights(hf_hub_download(L["repo"], L["high"]), adapter_name="light_high")
        pipe.load_lora_weights(hf_hub_download(L["repo"], L["low"]), adapter_name="light_low",
                               load_into_transformer_2=True)
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe


def _size_from(image, max_area: int = 1280 * 720):
    """Mantém a proporção da imagem inicial, com área até 720p e lados múltiplos de 16."""
    r = image.width / image.height
    h = int((max_area / r) ** 0.5) // 16 * 16
    w = int(h * r) // 16 * 16
    return w, h


def run(project, only=None, part="0/1", force=False, frames=None, mode=None, seeds=None):
    import torch
    from diffusers.utils import export_to_video
    from PIL import Image

    jobs = []
    for v in [v for v in project.videos if not only or v["name"] in only]:
        v = dict(v)
        if mode:
            from .config import MODES
            v.update(mode=mode, **MODES[mode])
        if frames:
            v["frames"] = max(5, (int(frames) - 1) // 4 * 4 + 1)
        jobs += [(v, s) for s in seeds or v["seeds"]]
    # divide por tarefa (vídeo x seed), não por vídeo, para todas as GPUs trabalharem
    jobs = [(v, s) for v, s in split(jobs, part) if force or not clip_path(project, v["name"], v["mode"], s).exists()]
    if not jobs:
        print("animate: nada a fazer")
        return

    seeds_file = project.out / "seeds_videos.json"
    pipes = {}
    for v, s in jobs:
        key = v["mode"]
        if key not in pipes:
            pipes.clear()  # um modo por vez na GPU
            pipes[key] = load_pipeline(project.models, key == "lightning", v["cpu_offload"])
        pipe = pipes[key]
        first = Image.open(resolve_image(project, v["first"])).convert("RGB")
        w, h = v.get("size") or _size_from(first)
        first = first.resize((w, h), Image.LANCZOS)
        kw = {}
        if v.get("last"):
            kw["last_image"] = Image.open(resolve_image(project, v["last"])).convert("RGB").resize((w, h), Image.LANCZOS)
        out = clip_path(project, v["name"], v["mode"], s)
        with timed(project, "animate", out.name, seed=s, frames=v["frames"], size=[w, h], mode=v["mode"]):
            result = pipe(image=first, prompt=v["prompt"], negative_prompt=v["negative"], height=h, width=w,
                          num_frames=v["frames"], num_inference_steps=v["steps"], guidance_scale=v["guidance_scale"],
                          generator=torch.Generator("cuda").manual_seed(s), **kw).frames[0]
            export_to_video(result, str(out), fps=v["fps"])
        reg = read_json(seeds_file)
        reg[out.name] = {k: v.get(k) for k in ("first", "last", "prompt", "negative", "frames", "fps", "mode",
                                               "steps", "guidance_scale")} | {"seed": s, "size": [w, h]}
        write_json(seeds_file, reg)
