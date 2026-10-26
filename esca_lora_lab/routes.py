"""HTTP API under /esca-lora/* (same origin as the ComfyUI page)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from aiohttp import web

from . import catalog as catalog_mod
from . import config, previews, reorg, scan, sync, taxonomy

log = logging.getLogger("esca_lora_lab")

_catalog = catalog_mod.Catalog()
_studio: previews.PreviewStudio | None = None
_model_arch_cache: dict[str, str] = {}


def entries() -> list[dict[str, Any]]:
    return _catalog.entries() or _catalog.build()


def studio() -> previews.PreviewStudio:
    global _studio
    if _studio is None:
        _studio = previews.PreviewStudio(entries)
    return _studio


def _public(e: dict[str, Any], lang: str) -> dict[str, Any]:
    keep = ("rel", "name", "title", "version", "arch", "flux2", "category", "nsfw", "note",
            "triggers", "rank", "fmt", "size_mb", "trained_on", "base_model", "trained_as", "sha256",
            "civitai_url", "creator", "has_preview")
    return {k: e.get(k) for k in keep} | {"category_label": taxonomy.BY_KEY[e["category"]].label(lang)}


async def _json(request: web.Request) -> dict[str, Any]:
    try:
        data = await request.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def register(routes: web.RouteTableDef) -> None:
    @routes.get("/esca-lora/health")
    async def health(_: web.Request) -> web.Response:
        return web.json_response({"ok": True, "version": "0.1.0", "loras": len(entries())})

    @routes.get("/esca-lora/catalog")
    async def get_catalog(request: web.Request) -> web.Response:
        lang = "ru" if request.query.get("lang", "en").lower().startswith("ru") else "en"
        return web.json_response({
            "built_at": _catalog.built_at,
            "categories": [{"key": c.key, "label": c.label(lang)} for c in taxonomy.CATEGORIES],
            "entries": [_public(e, lang) for e in entries()],
        })

    @routes.post("/esca-lora/catalog/rebuild")
    async def rebuild(_: web.Request) -> web.Response:
        built = _catalog.build()
        return web.json_response({"ok": True, "loras": len(built)})

    @routes.get("/esca-lora/model-arch")
    async def model_arch(request: web.Request) -> web.Response:
        name = request.query.get("name", "")
        kind = request.query.get("kind", "diffusion_models")
        key = f"{kind}:{name}"
        if key not in _model_arch_cache:
            path = config.model_path(kind, name)
            _model_arch_cache[key] = scan.scan_model(path) if path else "unknown"
        return web.json_response({"name": name, "arch": _model_arch_cache[key]})

    @routes.get("/esca-lora/sync")
    async def sync_plan(_: web.Request) -> web.Response:
        return web.json_response(await sync.run(entries(), apply=False))

    @routes.post("/esca-lora/sync")
    async def sync_apply(request: web.Request) -> web.Response:
        if not (await _json(request)).get("confirm"):
            return web.json_response({"error": "confirm=true required"}, status=400)
        return web.json_response(await sync.run(_catalog.build(), apply=True))

    @routes.get("/esca-lora/reorg")
    async def reorg_preview(_: web.Request) -> web.Response:
        return web.json_response(reorg.preview(entries()))

    @routes.post("/esca-lora/reorg/apply")
    async def reorg_apply(request: web.Request) -> web.Response:
        if not (await _json(request)).get("confirm"):
            return web.json_response({"error": "confirm=true required"}, status=400)
        result = await reorg.apply(_catalog.build())
        _catalog.build()
        return web.json_response(result)

    @routes.post("/esca-lora/reorg/rollback")
    async def reorg_rollback(request: web.Request) -> web.Response:
        run = (await _json(request)).get("run", "")
        if not run:
            return web.json_response({"error": "run required"}, status=400)
        result = await reorg.rollback(run)
        _catalog.build()
        return web.json_response(result)

    @routes.get("/esca-lora/previews/status")
    async def previews_status(_: web.Request) -> web.Response:
        return web.json_response(studio().status())

    @routes.post("/esca-lora/previews/{action}")
    async def previews_action(request: web.Request) -> web.Response:
        action = request.match_info["action"]
        body = await _json(request)
        s = studio()
        if action == "plan":
            return web.json_response(s.plan(body.get("presets"), bool(body.get("include_nsfw", False))))
        if action == "start":
            s.polite = bool(body.get("polite", True))
            return web.json_response(s.start())
        if action == "pause":
            return web.json_response(s.pause())
        if action == "resume":
            return web.json_response(s.resume())
        if action == "stop":
            return web.json_response(s.stop())
        if action == "retry-failed":
            return web.json_response({"retried": s.store.retry_failed(), **s.status()})
        if action == "import-pending":
            return web.json_response({"imported": await s.import_pending()})
        return web.json_response({"error": f"unknown action {action}"}, status=404)

    @routes.get("/esca-lora/previews/for/{sha}")
    async def previews_for(request: web.Request) -> web.Response:
        jobs = studio().store.for_lora(request.match_info["sha"])
        return web.json_response([{"scene": j["scene"], "strength": j["strength"],
                                   "thumb": f"/esca-lora/previews/thumb/{j['job_id']}"} for j in jobs])

    @routes.get("/esca-lora/previews/thumb/{job_id}")
    async def preview_thumb(request: web.Request) -> web.StreamResponse:
        job = studio().store.get(request.match_info["job_id"])
        if job and job.get("thumb") and Path(job["thumb"]).exists():
            return web.FileResponse(job["thumb"])
        raise web.HTTPNotFound()
