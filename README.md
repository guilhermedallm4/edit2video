# edit2video

Pipeline local, dirigido por um arquivo YAML, que transforma **imagens de referência** em **imagens editadas** e depois em **vídeos curtos**, usando só modelos abertos do Hugging Face:

```
 referências (N fotos) ──► [1] edit ──► K variações por imagem ──► [2] review ──► escolha
                              Qwen-Image-Edit-2511                                │
                                                                                  ▼
                 vídeos finais ◄── [4] finish ◄── M vídeos (S seeds cada) ◄── [3] animate
                  1080p, 24 fps     ffmpeg          Wan2.2-I2V-A14B (+ Lightning)
```

- **Quantas imagens você quiser**, cada uma com as suas próprias referências (rosto, corpo, roupa, cenário…) e o seu pedido de modificação em texto.
- **Quantas variações** por imagem (seeds fixas e registradas, para refazer uma variação exata).
- Uma imagem gerada pode ser **referência de outra** (ex.: gere um retrato do personagem e use-o como base de várias cenas, mantendo a consistência).
- **Quantos vídeos quiser**, com o número de **quadros** que você escolher, a partir do 1º quadro e (opcionalmente) de um **último quadro** fixo.
- Modo **Lightning** (4 passos, ~2 min por vídeo de 5 s) ou **padrão** (40 passos, ~35 min, mais detalhe).
- Cada etapa é **retomável** (pula o que já existe), registra **tempo e seed** de cada saída e pode usar **várias GPUs** (um processo independente por GPU).

## Modelos

| Etapa | Modelo | Licença | Tamanho |
|---|---|---|---|
| Edição de imagem | [`Qwen/Qwen-Image-Edit-2511`](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) (`QwenImageEditPlusPipeline`) | Apache 2.0 | 57,7 GB (40,9 GB se reaproveitar o Qwen-Image) |
| Imagem → vídeo | [`Wan-AI/Wan2.2-I2V-A14B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) (`WanImageToVideoPipeline`) | Apache 2.0 | 126,2 GB |
| Aceleração (opcional) | [`lightx2v/Wan2.2-Distill-Loras`](https://huggingface.co/lightx2v/Wan2.2-Distill-Loras) (LoRA de 4 passos) | Apache 2.0 | 1,4 GB |
| Acabamento | `ffmpeg` (interpolação `minterpolate`, upscale lanczos, `xfade`) | — | — |

## Requisitos

- GPU NVIDIA com **≥ 80 GB** de VRAM para rodar tudo direto na GPU (testado em RTX PRO 6000 Blackwell, 96 GB).
  Com `cpu_offload` dá para usar GPUs de **~48 GB**, mais devagar e com **≥ 96 GB de RAM** (veja [Custos](#custos)).
- ~**190 GB** livres em disco para os pesos (cache do Hugging Face).
- Python ≥ 3.10, PyTorch com CUDA, `ffmpeg` no PATH (ou o caminho em `finish.ffmpeg`).

## Instalação

```bash
git clone https://github.com/guilhermedallm4/edit2video && cd edit2video
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128     # ajuste ao seu CUDA
pip install "git+https://github.com/huggingface/diffusers"               # versão com Qwen-Image-Edit-Plus e Wan2.2
pip install -e .
```

Os pesos são baixados na primeira execução (ou antes, com `hf download Qwen/Qwen-Image-Edit-2511` e `hf download Wan-AI/Wan2.2-I2V-A14B-Diffusers`).

## Início rápido (demo sem fotos de ninguém)

O demo desenha duas referências por código (um boneco de massinha e um chapéu de chef) e roda o pipeline inteiro:

```bash
python scripts/make_demo_refs.py examples/demo/refs
edit2video run examples/demo.yaml --auto            # --auto: usa a v1 sem parar na revisão
# ou, com duas GPUs:
edit2video run examples/demo.yaml --auto --gpus 0,1
```

Saídas em `examples/demo/outputs/`: `images/` (variações), `review.html`/`review.png` (folha de contato), `clips/` (vídeos brutos, 16 fps), `final/` (1080p, 24 fps), `tempos.jsonl` e `seeds_*.json`.

## O arquivo de projeto

Veja o exemplo completo em [`examples/personagem_fantasia.yaml`](examples/personagem_fantasia.yaml). O essencial:

```yaml
references:                      # nome -> arquivo (relativo ao YAML)
  rosto: refs/rosto.jpg
  corpo: refs/corpo_inteiro.jpg
  fantasia: refs/fantasia.jpg

images:                          # N imagens a gerar
  - name: retrato
    refs: [rosto, corpo, fantasia]     # viram "image 1", "image 2", "image 3" no prompt
    prompt: "Image 1 shows the man's face: keep it exactly... He wears the costume shown in image 3..."
    negative: "different person, deformed face, ..."
    size: [928, 1152]
    variations: 3
  - name: cena
    refs: [retrato, rosto]             # a variação ESCOLHIDA de "retrato" vira image 1
    prompt: "Keep exactly the same character as image 1... Change only the scene: ..."

videos:                          # M vídeos a gerar
  - name: comendo
    first: cena                        # 1º quadro
    last: null                         # opcional: último quadro fixo (outra imagem ou referência)
    prompt: "He takes a big bite of the pizza... Static camera."
    frames: 81                         # 4k+1; 81 a 16 fps = 5,0 s
    seeds: [1, 2, 3]
    mode: lightning                    # ou standard
    keep_seed: 2                       # qual seed entra na emenda final (finish.concat)
```

<details>
<summary>Todas as chaves e valores padrão</summary>

| Chave | Padrão | Descrição |
|---|---|---|
| `output_dir` | `outputs` | pasta de saída (relativa ao YAML) |
| `models.edit` | `Qwen/Qwen-Image-Edit-2511` | modelo de edição |
| `models.edit_share_components_from` | `null` | ex.: `Qwen/Qwen-Image` — reaproveita text_encoder e VAE (pesos idênticos) |
| `models.edit_cpu_offload` | `false` | `true` para GPUs com menos de ~60 GB |
| `models.video` | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | modelo de vídeo |
| `edit_defaults.size` | `[1280, 720]` | arredondado para múltiplos de 16 |
| `edit_defaults.variations` | `3` | variações por imagem |
| `edit_defaults.steps` / `true_cfg_scale` / `guidance_scale` | `40` / `4.0` / `1.0` | valores do model card |
| `edit_defaults.seed_base` | `1000` | seed = `seed_base + variação` |
| `edit_defaults.upscale_small_refs_to` | `900` | amplia referências pequenas antes de enviar |
| `video_defaults.frames` / `fps` | `81` / `16` | 5,0 s |
| `video_defaults.mode` | `lightning` | `lightning`: 4 passos, CFG 1,0 · `standard`: 40 passos, CFG 3,5 |
| `video_defaults.size` | proporção do 1º quadro, até 720p | `[largura, altura]` para fixar |
| `video_defaults.cpu_offload` | `false` | ~38 GB de VRAM, ~70 GB de RAM, ~40% mais lento |
| `finish.fps` / `width` / `height` | `24` / `1920` / `1080` | saída final (mantém proporção com barras) |
| `finish.interpolation` | `minterpolate` | ou `none` |
| `finish.concat` / `xfade` | `false` / `0.5` | emenda um clipe por vídeo com crossfade |
</details>

## Comandos

```bash
edit2video edit    projeto.yaml [--only NOMES] [--variations K] [--gpus 0,1] [--force]
edit2video review  projeto.yaml [--choose retrato=2 cena=1] [--auto]
edit2video animate projeto.yaml [--only NOMES] [--frames 81] [--mode lightning|standard] [--seeds 1 2] [--gpus 0,1]
edit2video finish  projeto.yaml
edit2video run     projeto.yaml [--auto] [--gpus 0,1]     # tudo; para na revisão se faltar escolha
```

O `run` gera as imagens em níveis: primeiro as que usam só referências; se houver imagens que dependem de outras, ele para e pede a escolha (a menos que use `--auto`).

## Como as informações chegam aos modelos

**Imagem.** As referências vão como uma **lista**, e o diffusers monta para o codificador (Qwen2.5-VL-7B) uma conversa do tipo
`Picture 1: <imagem> Picture 2: <imagem> ... {seu prompt}`. Por isso o prompt pode dizer *"image 1 shows the face… image 3 shows the costume"*: **a ordem de `refs` importa**. Cada referência é usada duas vezes: reduzida (~384²) para o modelo *entender* o conteúdo e em ~1024² como referência visual (detalhes como o rosto). O transformer (~20B) parte de ruído e o remove em 40 passos.

**Vídeo.** O 1º quadro vira um "vídeo de condição" de `frames` quadros em que só o primeiro é conhecido (e o último, se houver `last`), com uma máscara que diz ao modelo o que é fixo:

```
sem last:  [ 1º quadro | 0 | 0 | ... | 0 ]
com last:  [ 1º quadro | 0 | 0 | ... | 0 | último quadro ]
```

O VAE comprime 4× no tempo e 8× no espaço (81×720×1280 → 21×90×160 ≈ 75 mil tokens, contra ~3,6 mil de uma imagem — por isso vídeo custa tanto). O Wan2.2 A14B tem **dois especialistas de 14B**: o primeiro cuida dos passos de ruído alto (composição e movimento) e o segundo dos de ruído baixo (detalhes). A Lightning é uma LoRA para cada um, que reduz o processo a 4 passos.

## Dicas que fizeram diferença

- **Rosto.** Diga explicitamente *"keep the face of the man in image 1 exactly as it is… natural skin tone… do not paint the face… do not cover the eyes"*, e ponha no negativo `different person, changing identity`. Use foto **de frente, nítida e grande**; recorte o rosto e remova adesivos/textos. Não descreva no prompt traços que a foto não tem (ex.: "same beard"): o modelo **inventa** o que você descreve.
- **Corpo/físico.** O modelo puxa o físico da referência do personagem (ex.: super-herói musculoso). Se não for o que quer, diga o oposto no prompt e no negativo (`muscular, abs, athletic`).
- **Consistência entre cenas.** Gere primeiro um retrato aprovado e use-o como `refs: [retrato, rosto]` nas cenas seguintes ("keep exactly the same character as image 1, change only the scene").
- **Referência de personagem com cabeça própria** (máscaras, animais) às vezes faz o modelo copiar a cabeça inteira. Gere algumas seeds e descarte as que perderem o rosto.
- **Vídeo "completa a ação".** O Wan tende a executar o que a imagem sugere (um míssil na rampa é lançado, por exemplo). Descreva o movimento desejado e ponha o indesejado no negativo; `Static camera` estabiliza.
- **Primeiro e último quadro** fixos dão controle, mas o modelo tende a "chegar cedo" no último quadro e ficar parado. Para ações longas, gere em duas partes e emende.
- **`frames` precisa ser 4k+1** (ex.: 33, 49, 81). O Wan foi treinado com 81; bem mais que isso degrada.

## Custos

Medidos numa **NVIDIA RTX PRO 6000 Blackwell (96 GB)**, PyTorch 2.11 + CUDA 12.8, bf16, sem `torch.compile`:

| Etapa | Configuração | Tempo | VRAM (pico) |
|---|---|---|---|
| Carregar Qwen-Image-Edit | — | ~20–60 s | ~58 GB |
| Imagem editada, 1280×720, 40 passos | 1 referência | **~85–95 s** | ~58–61 GB |
| | 2 referências | ~90–110 s | |
| | 3 referências | ~150–175 s | |
| Carregar Wan2.2 + LoRA | — | ~30–60 s | |
| Vídeo 81 quadros (5 s), 1280×720, **Lightning** | sem offload | **~100–125 s** | ~81 GB |
| | com `cpu_offload` | ~150 s | ~39 GB (+ ~70 GB de RAM) |
| Vídeo 81 quadros, **padrão** (40 passos, CFG 3,5) | com `cpu_offload` | **~35 min** | ~38 GB |
| Acabamento ffmpeg (16→24 fps, 1080p) | CPU | ~10–25 s por clipe | — |

**Energia.** Durante a inferência a GPU ficou em ~480–500 W. Por saída, só a GPU:

| Saída | Energia aprox. |
|---|---|
| 1 imagem editada | ~0,013 kWh |
| 1 vídeo de 5 s Lightning | ~0,015 kWh |
| 1 vídeo de 5 s padrão | ~0,29 kWh |

Para estimar o custo: `custo ≈ kWh × tarifa` (ex.: a R$ 1,00/kWh, um vídeo Lightning custa ~R$ 0,02 de GPU) ou, na nuvem, `custo ≈ horas de GPU × preço/hora` de uma GPU de 80–96 GB.

**Exemplo de projeto** (3 imagens × 3 variações + 2 vídeos × 3 seeds em Lightning): ~9 × 1,7 min + 6 × 2 min ≈ **27 min de GPU (≈ 0,23 kWh)** numa GPU, ou ~14 min com duas. O mesmo em modo padrão: ~3,5 h.

**Disco:** ~185 GB de pesos (ou ~169 GB reaproveitando o Qwen-Image). As saídas são pequenas (~1 MB por imagem, ~0,5 MB por clipe bruto, 3–5 MB por clipe final).

## Limitações

- Vídeos curtos (até ~5 s por clipe); sem áudio.
- A fidelidade do rosto depende muito da foto de referência e varia entre seeds; revise sempre.
- `minterpolate` pode criar artefatos em movimentos rápidos (RIFE seria melhor, mas não está integrado).
- Duas GPUs = dois processos independentes; o modelo não é dividido entre GPUs.

## Uso responsável

- Use fotos de pessoas **somente com autorização** delas. Não crie conteúdo que se passe por registro real, que engane, difame ou sexualize alguém.
- Personagens de terceiros (ex.: de desenhos e filmes) têm direitos autorais: uso pessoal, educativo ou em uma peça escolar costuma ser tolerado; uso comercial, não.
- Indique que o material foi **gerado por IA** quando for exibido ou publicado.
- Respeite as licenças dos modelos (todas Apache 2.0 nesta configuração).

## Licença

Código sob [MIT](LICENSE). Os modelos têm as suas próprias licenças (links acima).
