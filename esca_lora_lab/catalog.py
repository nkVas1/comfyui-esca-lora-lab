"""The LoRA catalog: header scan + LoRA Manager metadata + Civitai, classified.

Scanning is incremental: header results are cached by (size, mtime), so a rebuild after
adding a few files only reads those files.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from . import config, scan, taxonomy

CATALOG_FILE = "catalog.json"
SCAN_CACHE_FILE = "scan-cache.json"


def iter_loras(roots: list[Path], excluded: set[str]) -> list[tuple[Path, Path]]:
    """(root, file) pairs for every .safetensors under the roots, skipping private folders."""
    out = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d.lower() not in excluded)
            for f in sorted(filenames):
                if f.lower().endswith(".safetensors"):
                    out.append((root, Path(dirpath) / f))
    return out


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def lm_metadata(lora: Path) -> dict[str, Any]:
    """LoRA Manager sidecar (<stem>.metadata.json) when stored alongside the model."""
    return _read_json(lora.with_name(lora.stem + ".metadata.json"))


def civitai_from_lm(meta: dict[str, Any]) -> dict[str, Any]:
    c = meta.get("civitai") or {}
    model = c.get("model") or {}
    if not c.get("id") and not model:
        return {}
    return {
        "model_name": model.get("name") or meta.get("model_name") or "",
        "version_name": c.get("name") or "",
        "base_model": c.get("baseModel") or "",
        "trained_words": c.get("trainedWords") or [],
        "tags": model.get("tags") or meta.get("tags") or [],
        "nsfw": bool(model.get("nsfw")) or int(c.get("nsfwLevel") or 0) >= 4,
        "url": f"https://civitai.com/models/{c.get('modelId')}?modelVersionId={c.get('id')}" if c.get("id") else "",
        "creator": (c.get("creator") or {}).get("username", ""),
    }


def civitai_from_cache(sha256: str, cache_dirs: list[str]) -> dict[str, Any]:
    for d in cache_dirs:
        data = _read_json(Path(d) / f"{sha256}.json")
        if data.get("status") == 200:
            return {
                "model_name": data.get("model_name", ""),
                "version_name": data.get("version_name", ""),
                "base_model": data.get("base_model", ""),
                "trained_words": data.get("trained_words") or [],
                "tags": data.get("tags") or [],
                "nsfw": bool(data.get("nsfw")),
                "url": data.get("url", ""),
                "creator": data.get("creator", ""),
            }
    return {}


class Catalog:
    def __init__(self) -> None:
        self.dir = config.data_dir()
        self._entries: list[dict[str, Any]] = []
        self.built_at = 0.0

    def entries(self) -> list[dict[str, Any]]:
        if not self._entries:
            cached = _read_json(self.dir / CATALOG_FILE)
            self._entries = cached.get("entries", []) if isinstance(cached, dict) else []
            self.built_at = cached.get("built_at", 0.0) if isinstance(cached, dict) else 0.0
        return self._entries

    def build(self) -> list[dict[str, Any]]:
        cfg = config.settings()
        excluded = {d.lower() for d in cfg["excluded_dirs"]}
        cache_path = self.dir / SCAN_CACHE_FILE
        scan_cache: dict[str, Any] = _read_json(cache_path)
        fresh_cache: dict[str, Any] = {}
        entries = []
        for root, path in iter_loras(config.lora_roots(), excluded):
            key = str(path)
            size, mtime = scan.file_signature(path)
            hit = scan_cache.get(key)
            if hit and hit.get("size") == size and hit.get("mtime") == mtime:
                s = hit["scan"]
            else:
                s = scan.scan_lora(path).to_dict()
            fresh_cache[key] = {"size": size, "mtime": mtime, "scan": s}
            entries.append(self._entry(root, path, size, s, cfg))
        config.write_json_atomic(cache_path, fresh_cache)
        self._entries = entries
        self.built_at = time.time()
        config.write_json_atomic(self.dir / CATALOG_FILE, {"built_at": self.built_at, "entries": entries})
        return entries

    @staticmethod
    def _entry(root: Path, path: Path, size: int, s: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
        meta = lm_metadata(path)
        sha = (meta.get("sha256") or "").lower()
        civ = civitai_from_lm(meta) or (civitai_from_cache(sha, cfg["civitai_cache_dirs"]) if sha else {})
        rel = path.relative_to(root).as_posix()
        cls = taxonomy.classify(
            rel, title=civ.get("model_name", ""), version=civ.get("version_name", ""), tags=civ.get("tags"),
            trained_words=civ.get("trained_words"), scan_title=s.get("title", ""),
            scan_triggers=s.get("triggers"), civitai_nsfw=bool(civ.get("nsfw")),
            title_overrides=cfg.get("title_overrides"), path_overrides=cfg.get("path_overrides"),
        )
        arch = s["arch"]
        triggers = list(dict.fromkeys([*(civ.get("trained_words") or []), *(s.get("triggers") or [])]))
        civ_base = civ.get("base_model", "")
        return {
            "path": str(path),
            "root": str(root),
            "rel": rel,
            "name": path.stem,
            "title": civ.get("model_name") or meta.get("model_name") or s.get("title") or path.stem,
            "version": civ.get("version_name", ""),
            "arch": arch,
            "flux2": arch in scan.FLUX2_FAMILY,
            "category": cls.category,
            "nsfw": cls.nsfw,
            "reason": cls.reason,
            "note": cls.note,
            "triggers": [t.strip() for t in triggers if t and t.strip()][:8],
            "civitai_tags": civ.get("tags", []),
            "rank": s.get("rank"),
            "fmt": s.get("fmt", "lora"),
            "size_mb": round(size / 2**20, 1),
            "civitai_base": civ_base,
            "trained_on": ("base" if "base" in civ_base.lower() else "distilled")
            if arch.startswith("flux2-klein") and civ_base else "",
            "base_model": taxonomy.base_model_for(
                arch, civ_base, f"{s.get('base_meta', '')} {meta.get('base_model', '')} {path.stem}"),
            "trained_as": s.get("trained_as", ""),
            "sha256": sha,
            "civitai_url": civ.get("url", ""),
            "creator": civ.get("creator", ""),
            "lm_tags": meta.get("tags") or [],
            "lm_base_model": meta.get("base_model") or "",
            "has_lm_metadata": bool(meta),
            "has_preview": bool(meta.get("preview_url")),
        }
