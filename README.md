# ESCA LoRA Lab

**A LoRA library built into ComfyUI.** It sits on top of
[ComfyUI LoRA Manager](https://github.com/willmiao/ComfyUI-Lora-Manager), which stays your
main manager (the `/loras` page, Civitai downloads, recipes, loader nodes with trigger words).
ESCA LoRA Lab adds what large libraries need on top of it:

| Feature | What it does |
|---|---|
| **Sidebar panel** | Search, purpose facets, a "compatible with the graph's model" filter, NSFW hidden by default, one-click trigger copy. Drag a LoRA onto the canvas or press **To graph** — it lands in a LoRA Manager loader through LoRA Manager's own `update-lora-code` channel, so trigger words and widgets stay in sync |
| **Tensor-true architecture** | Detects Flux.2 Klein 9B / Klein 4B / Flux.2 Dev / Flux.1 / SDXL / SD 1.5 / Z-Image from the safetensors header (hidden width, block layout, LoKr) — never from the file name. Diffusion models are identified the same way, including GGUF |
| **Compatibility guard** | Watches the graph and warns when a LoRA does not match the loaded model. Such a LoRA silently does nothing — the guard tells you before you wait for a render |
| **Taxonomy → LoRA Manager** | Writes managed `esca:*` tags (purpose, architecture, base/distilled), a corrected `base_model` (LoRA Manager's canonical names, e.g. `Flux.2 Klein 9B`, `Flux.1 Kontext`), an `[ESCA] … [/ESCA]` notes block and trigger words where none exist. Idempotent: your own tags and notes are never touched |
| **Reversible reorganisation** | Optional: moves LoRAs into `<scope>/flux2/klein/<purpose>/`, `flux2/dev/<purpose>/`, `flux1/`, `sdxl/`, `sd15/`… through LoRA Manager (sidecars and previews move with the model), rewrites LoRA paths inside your workflow JSON files with backups, and keeps a journal for a one-call rollback |
| **Preview studio** | Uniform example images for every LoRA, rendered on *your* model with identical scenes and seeds (sliders in both directions). Starts only when you press **Start**; pause / resume / stop; the SQLite queue survives restarts; a job's key is the LoRA's SHA-256, so planning twice never duplicates work and moving files never loses it. Polite mode yields the GPU to your own prompts |

## Requirements

- ComfyUI with the new frontend (tested on ComfyUI 0.11 / frontend 1.37) and the sidebar API.
- [ComfyUI LoRA Manager](https://github.com/willmiao/ComfyUI-Lora-Manager) ≥ 1.2.6 for sync,
  reorganisation and sending LoRAs to the graph. The panel, scanner and guard work without it.
- Python dependencies are already part of ComfyUI (`aiohttp`, `Pillow`).

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/nkVas1/comfyui-esca-lora-lab.git
```

Restart ComfyUI and open the **LoRA** tab in the left sidebar. The interface follows
ComfyUI's language setting (English and Russian).

## Configuration

Settings live in `ComfyUI/user/esca-lora-lab/settings.json` (created on first start, kept
across updates):

| Key | Default | Meaning |
|---|---|---|
| `excluded_dirs` | `[]` | Folder names that are never scanned or moved (e.g. private training sets) |
| `language` | `"en"` | Language of tags and notes written into LoRA Manager (`"en"` or `"ru"`) |
| `reorg_scope` | `""` | Sub-folder of the LoRA root that the reorganisation manages (`""` = the whole root) |
| `extra_workflow_dirs` | `[]` | Workflow folders besides `user/default/workflows` whose LoRA paths are rewritten |
| `civitai_cache_dirs` | `[]` | Folders with `<sha256>.json` Civitai lookups, used when LoRA Manager has no data |
| `title_overrides` | `{}` | Civitai title prefix → category, for LoRAs whose tags mislead the rules |
| `path_overrides` | `{}` | File stem → `[category, note]`, for LoRAs unknown to Civitai |
| `lm_base_url` | `""` | LoRA Manager URL when it does not run on the same server |

Categories: `speed`, `control`, `slider`, `quality`, `style-paint`, `style-photo`,
`style-graphic`, `style-3d`, `anime`, `character`, `concept`, `nsfw`, `unknown`.

### Preview presets

`presets/previews.*.json` are examples: model files, acceleration LoRA, steps, size, scenes,
seeds and per-category rules (`by_category`, `exclude_categories`). Put your own presets
into `ComfyUI/user/esca-lora-lab/presets/` — a preset there replaces the bundled one with the
same `id`. Edit-tool LoRAs (`control`) are excluded by default: without input images their
previews say nothing.

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| GET | `/esca-lora/catalog?lang=en` | Catalog (POST `/esca-lora/catalog/rebuild` to rescan) |
| GET | `/esca-lora/model-arch?name=&kind=` | Architecture of a model file |
| GET / POST | `/esca-lora/sync` | Dry run / write to LoRA Manager (`{"confirm": true}`) |
| GET / POST | `/esca-lora/reorg`, `/reorg/apply`, `/reorg/rollback` | Plan, apply (`{"confirm": true}`), roll back (`{"run": "<id>"}`) |
| GET / POST | `/esca-lora/previews/status`, `/previews/{plan,start,pause,resume,stop,retry-failed,import-pending}` | Preview studio |

Everything runs locally against your own ComfyUI; nothing is sent anywhere else.

## Development

```bash
uv run --no-project --with pytest --with aiohttp --with pillow pytest -q
uv run --no-project --with ruff ruff check .
```

The LoRA architecture detector is a port of the author's `rhapsode` scanner.

## License

[GPL-3.0](LICENSE), like ComfyUI and ComfyUI LoRA Manager.
