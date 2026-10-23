"""Leitura e validação do arquivo de projeto (YAML)."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULTS = {
    "output_dir": "outputs",
    "models": {
        "edit": "Qwen/Qwen-Image-Edit-2511",
        # Opcional: carrega text_encoder e VAE de outro repositório com pesos idênticos (ex.: "Qwen/Qwen-Image"),
        # para não baixar 17 GB repetidos se você já tem o Qwen-Image no cache.
        "edit_share_components_from": None,
        "edit_cpu_offload": False,  # True para GPUs com menos de ~60 GB de VRAM
        "video": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "lightning": {
            "repo": "lightx2v/Wan2.2-Distill-Loras",
            "high": "wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors",
            "low": "wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors",
        },
    },
    "edit_defaults": {
        "size": [1280, 720],
        "variations": 3,
        "steps": 40,            # model card do Qwen-Image-Edit-2511
        "true_cfg_scale": 4.0,  # model card
        "guidance_scale": 1.0,  # model card
        "seed_base": 1000,
        "negative": "deformed face, extra limbs, extra fingers, text, watermark, logo, blurry",
        "upscale_small_refs_to": 900,  # amplia referências menores que isso (px no maior lado)
    },
    "video_defaults": {
        "frames": 81,           # 4k+1; 81 a 16 fps = 5,0 s
        "fps": 16,
        "mode": "lightning",    # "lightning" (4 passos, sem CFG) ou "standard" (40 passos, CFG 3,5)
        "seeds": [1],
        "negative": ("face morphing, changing identity, extra limbs, extra fingers, deformed hands, "
                     "text, watermark, scene cut, blurry"),
        "cpu_offload": False,   # True: ~38 GB de VRAM, mas ~70 GB de RAM e mais lento
    },
    "finish": {
        "fps": 24,
        "width": 1920,
        "height": 1080,
        "interpolation": "minterpolate",  # "minterpolate" ou "none"
        "crf": 16,
        "concat": False,        # emenda todos os vídeos escolhidos num só arquivo
        "xfade": 0.5,           # duração do crossfade na emenda (s)
        "ffmpeg": "ffmpeg",
    },
}
MODES = {"lightning": {"steps": 4, "guidance_scale": 1.0}, "standard": {"steps": 40, "guidance_scale": 3.5}}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


@dataclass
class Project:
    path: Path
    root: Path
    raw: dict
    references: dict = field(default_factory=dict)
    images: list = field(default_factory=list)
    videos: list = field(default_factory=list)

    @property
    def out(self) -> Path:
        return self.root / self.raw["output_dir"]

    def dir(self, name: str) -> Path:
        d = self.out / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def models(self):
        return self.raw["models"]

    @property
    def finish(self):
        return self.raw["finish"]


def _mult16(x: int) -> int:
    return max(16, int(round(x / 16)) * 16)


def load(path: str | Path) -> Project:
    path = Path(path).resolve()
    raw = _merge(DEFAULTS, yaml.safe_load(path.read_text()) or {})
    p = Project(path=path, root=path.parent, raw=raw)

    for name, ref in (raw.get("references") or {}).items():
        f = (p.root / ref).resolve()
        if not f.exists():
            raise SystemExit(f"referência '{name}': arquivo não encontrado: {f}")
        p.references[name] = f

    names = set()
    for spec in raw.get("images") or []:
        s = _merge(raw["edit_defaults"], spec)
        if "name" not in s or "prompt" not in s:
            raise SystemExit(f"cada item de 'images' precisa de 'name' e 'prompt': {spec}")
        if s["name"] in names:
            raise SystemExit(f"nome de imagem repetido: {s['name']}")
        names.add(s["name"])
        for r in s.get("refs") or []:
            if r not in p.references and r not in names - {s["name"]}:
                raise SystemExit(f"imagem '{s['name']}': ref '{r}' não é uma referência nem uma imagem anterior")
        s["size"] = [_mult16(s["size"][0]), _mult16(s["size"][1])]
        p.images.append(s)

    for spec in raw.get("videos") or []:
        v = _merge(raw["video_defaults"], spec)
        for k in ("name", "first", "prompt"):
            if k not in v:
                raise SystemExit(f"cada item de 'videos' precisa de '{k}': {spec}")
        for k in ("first", "last"):
            if v.get(k) and v[k] not in names and v[k] not in p.references:
                raise SystemExit(f"vídeo '{v['name']}': {k}='{v[k]}' não é uma imagem nem uma referência")
        if v["mode"] not in MODES:
            raise SystemExit(f"vídeo '{v['name']}': mode deve ser {list(MODES)}")
        n = int(v["frames"])
        v["frames"] = max(5, (n - 1) // 4 * 4 + 1)  # Wan exige 4k+1
        if v["frames"] != n:
            print(f"aviso: vídeo '{v['name']}': frames {n} -> {v['frames']} (precisa ser 4k+1)")
        v.update({k: v.get(k, d) for k, d in MODES[v["mode"]].items()})
        p.videos.append(v)
    return p
