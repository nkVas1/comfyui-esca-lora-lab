# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-10-26

### Added

- Header-only scanner that detects the LoRA's base architecture from tensor shapes, and the
  architecture of diffusion models (safetensors and GGUF).
- Purpose taxonomy with ordered rules, configurable hand corrections and LoRA Manager
  base-model mapping that keeps Klein base/distilled, Flux.1 Kontext and Z-Image variants.
- Incremental catalog combining the scan, LoRA Manager metadata and optional Civitai caches.
- Idempotent sync of managed `esca:*` tags, base model, notes block and trigger words into
  LoRA Manager (English or Russian).
- Reversible folder reorganisation through LoRA Manager moves, with workflow path rewriting,
  backups and a rollback journal.
- Preview studio: SQLite queue keyed by LoRA SHA-256, presets with per-category rules,
  pause/resume/stop, polite GPU sharing, import into LoRA Manager examples.
- Sidebar panel (English/Russian): catalog facets, drag-and-drop and one-click send to LoRA
  Manager loaders, graph compatibility guard, studio controls.
