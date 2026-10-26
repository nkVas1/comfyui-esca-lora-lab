"""HTTP bridge to ComfyUI LoRA Manager (same server, /api/lm/*).

Going through LoRA Manager's API — instead of editing its sidecars directly — keeps its
in-memory cache, previews and centralised sidecar storage consistent.
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from . import config

log = logging.getLogger("esca_lora_lab")


class LoraManagerError(RuntimeError):
    """LoRA Manager is missing or rejected a request."""


def server_url() -> str:
    override = config.settings().get("lm_base_url")
    if override:
        return str(override).rstrip("/")
    try:
        from server import PromptServer  # type: ignore[import-not-found]

        ps = PromptServer.instance
        host = getattr(ps, "address", "127.0.0.1") or "127.0.0.1"
        if host in ("0.0.0.0", "::"):
            host = "127.0.0.1"
        return f"http://{host}:{getattr(ps, 'port', 8188)}"
    except Exception:  # standalone use
        return "http://127.0.0.1:8188"


class LoraManager:
    def __init__(self, base_url: str | None = None) -> None:
        self.base = base_url or server_url()

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        timeout = aiohttp.ClientTimeout(total=kwargs.pop("timeout", 120))
        async with (
            aiohttp.ClientSession(timeout=timeout, trust_env=False) as s,
            s.request(method, self.base + path, **kwargs) as r,
        ):
            text = await r.text()
            if r.status >= 400:
                raise LoraManagerError(f"{method} {path} -> {r.status}: {text[:300]}")
            try:
                return await r.json(content_type=None)
            except Exception:
                return {"raw": text}

    async def available(self) -> bool:
        try:
            await self._request("GET", "/api/lm/loras/roots", timeout=5)
            return True
        except Exception:
            return False

    async def roots(self) -> list[str]:
        data = await self._request("GET", "/api/lm/loras/roots")
        return list(data.get("roots", [])) if isinstance(data, dict) else []

    async def save_metadata(self, file_path: str, updates: dict[str, Any]) -> dict[str, Any]:
        return await self._request("POST", "/api/lm/loras/save-metadata", json={"file_path": file_path, **updates})

    async def move_bulk(self, file_paths: list[str], target_dir: str) -> dict[str, Any]:
        return await self._request("POST", "/api/lm/loras/move_models_bulk",
                                   json={"file_paths": file_paths, "target_path": target_dir}, timeout=1800)

    async def import_examples(self, model_hash: str, file_paths: list[str]) -> dict[str, Any]:
        return await self._request("POST", "/api/lm/import-example-images",
                                   json={"model_hash": model_hash, "file_paths": file_paths}, timeout=300)

    async def rescan(self) -> None:
        await self._request("GET", "/api/lm/loras/scan", timeout=1800)
