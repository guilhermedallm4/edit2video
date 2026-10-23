"""Etapa 1: imagens editadas a partir das referências, com Qwen-Image-Edit-2511.

Cada item de `images` recebe uma lista de referências (`refs`) que vira image 1, image 2, ... no prompt.
Uma referência também pode ser uma imagem gerada antes (a variação escolhida em selection.json).
"""
from __future__ import annotations

from .common import image_path, read_json, resolve_image, split, timed, write_json


def load_pipeline(models: dict):
    import torch
    from diffusers import QwenImageEditPlusPipeline

    extra = {}
    share = models.get("edit_share_components_from")
    if share:
        from diffusers import AutoencoderKLQwenImage
        from transformers import Qwen2_5_VLForConditionalGeneration

        extra["text_encoder"] = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            share, subfolder="text_encoder", torch_dtype=torch.bfloat16)
        extra["vae"] = AutoencoderKLQwenImage.from_pretrained(share, subfolder="vae", torch_dtype=torch.bfloat16)
    pipe = QwenImageEditPlusPipeline.from_pretrained(models["edit"], torch_dtype=torch.bfloat16, **extra)
    if models.get("edit_cpu_offload"):
        pipe.enable_model_cpu_offload()  # para GPUs com menos de ~60 GB (mais lento, usa RAM)
    else:
        pipe.to("cuda")
    return pipe


def load_ref(path, upscale_to: int):
    from PIL import Image

    im = Image.open(path).convert("RGB")
    k = max(1, upscale_to // max(im.size)) if upscale_to else 1
    return im.resize((im.width * k, im.height * k), Image.LANCZOS) if k > 1 else im


def run(project, only=None, part="0/1", force=False, variations=None):
    import torch

    specs = [s for s in project.images if not only or s["name"] in only]
    todo = []
    for s in specs:
        n = variations or s["variations"]
        todo += [(s, v) for v in range(1, n + 1)]
    # divide por tarefa (imagem x variação), não por imagem, para todas as GPUs trabalharem
    todo = [(s, v) for s, v in split(todo, part) if force or not image_path(project, s["name"], v).exists()]
    if not todo:
        print("edit: nada a fazer")
        return
    pipe = load_pipeline(project.models)
    seeds_file = project.out / "seeds_images.json"
    for s, v in todo:
        seed = s["seed_base"] + v
        refs = [load_ref(resolve_image(project, r), s["upscale_small_refs_to"]) for r in s.get("refs") or []]
        if not refs:
            raise SystemExit(f"imagem '{s['name']}': precisa de pelo menos uma referência em 'refs'")
        out = image_path(project, s["name"], v)
        with timed(project, "edit", out.name, seed=seed, refs=s.get("refs")):
            img = pipe(image=refs, prompt=s["prompt"], negative_prompt=s["negative"],
                       width=s["size"][0], height=s["size"][1], num_inference_steps=s["steps"],
                       true_cfg_scale=s["true_cfg_scale"], guidance_scale=s["guidance_scale"],
                       generator=torch.manual_seed(seed)).images[0]
            img.save(out)
        reg = read_json(seeds_file)
        reg[out.name] = {"seed": seed, "refs": s.get("refs"), "prompt": s["prompt"], "negative": s["negative"]}
        write_json(seeds_file, reg)
