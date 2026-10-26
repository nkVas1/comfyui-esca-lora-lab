"""Paths and user settings. Works inside ComfyUI and standalone (tests, CLI)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger("esca_lora_lab")

MODULE_DIR = Path(__file__).resolve().parents[1]

try:  # inside ComfyUI
    import folder_paths  # type: ignore[import-not-found]
except ImportError:  # standalone
    folder_paths = None


def data_dir() -> Path:
    """User data lives outside the module so reinstalling never loses it."""
    override = os.environ.get("ESCA_LORA_LAB_DATA")
    if override:
        base = Path(override)
    elif folder_paths is not None:
        base = Path(folder_paths.get_user_directory()) / "esca-lora-lab"
    else:
        base = MODULE_DIR / "data"
    base.mkdir(parents=True, exist_ok=True)
    return base


def lora_roots() -> list[Path]:
    if folder_paths is not None:
        return [Path(p) for p in folder_paths.get_folder_paths("loras") if Path(p).is_dir()]
    env = os.environ.get("ESCA_LORA_ROOTS", "")
    return [Path(p) for p in env.split(os.pathsep) if p]


def model_path(kind: str, name: str) -> Path | None:
    """Resolve a model file name as ComfyUI does (diffusion_models, checkpoints, unet_gguf...)."""
    if folder_paths is None:
        return None
    for folder in (kind, "diffusion_models", "checkpoints", "unet", "unet_gguf"):
        try:
            p = folder_paths.get_full_path(folder, name)
        except Exception:  # unknown folder key in this ComfyUI build
            p = None
        if p:
            return Path(p)
    return None


DEFAULT_SETTINGS: dict[str, Any] = {
    # folder names (case-insensitive) that are never scanned or moved, e.g. private training sets
    "excluded_dirs": [],
    # extra Civitai lookups by SHA-256 (files named <sha256>.json), used when LoRA Manager has no data
    "civitai_cache_dirs": [],
    # workflow folders besides ComfyUI/user/default/workflows whose LoRA paths the reorganisation rewrites
    "extra_workflow_dirs": [],
    # sub-folder of a LoRA root managed by the reorganisation ("" = the whole root)
    "reorg_scope": "",
    # LoRA Manager URL; empty = this ComfyUI server
    "lm_base_url": "",
    # language of the tags and notes written into LoRA Manager: "en" or "ru"
    "language": "en",
    # hand corrections for one's own library: Civitai title prefix -> category key
    "title_overrides": {},
    # file stem -> [category key, note]
    "path_overrides": {},
}


def settings() -> dict[str, Any]:
    path = data_dir() / "settings.json"
    merged = dict(DEFAULT_SETTINGS)
    if path.exists():
        try:
            merged.update(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("settings.json unreadable, using defaults: %s", exc)
    else:
        path.write_text(json.dumps(DEFAULT_SETTINGS, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged


def write_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)
