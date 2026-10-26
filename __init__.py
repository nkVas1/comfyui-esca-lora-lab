"""ESCA LoRA Lab — a LoRA library built into ComfyUI, on top of ComfyUI LoRA Manager.

Adds a sidebar panel (catalog with purpose facets, drag-and-drop into LoRA Manager loaders,
a model/LoRA compatibility guard), writes the catalog's taxonomy into LoRA Manager's
metadata, reorganises the LoRA folders reversibly and renders uniform previews on demand.
No custom nodes: the module is an HTTP API (/esca-lora/*) plus a frontend extension.
"""

import logging

try:
    from .esca_lora_lab import routes as _routes
except ImportError:  # imported as a top-level module (pytest collects the repo root as a package)
    from esca_lora_lab import routes as _routes  # type: ignore[no-redef]

log = logging.getLogger("esca_lora_lab")

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}
WEB_DIRECTORY = "./web"

try:
    from server import PromptServer  # type: ignore[import-not-found]

    _routes.register(PromptServer.instance.routes)
    log.info("ESCA LoRA Lab: routes registered under /esca-lora")
except Exception as exc:  # outside ComfyUI (tests) or a broken server object
    log.warning("ESCA LoRA Lab: routes not registered: %s", exc)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
