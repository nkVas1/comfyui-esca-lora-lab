"""Write the catalog's knowledge into LoRA Manager metadata — idempotently.

Only what this module owns is touched:
  * tags in the ``esca:`` namespace (the owner's and Civitai's tags are kept);
  * ``base_model`` when the tensors prove a different architecture than recorded;
  * a managed ``[ESCA] … [/ESCA]`` block inside notes (the rest of the notes is kept);
  * trigger words only when LoRA Manager has none.
A second run with unchanged inputs changes nothing.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

from . import catalog, config, scan, taxonomy
from .lm_bridge import LoraManager

log = logging.getLogger("esca_lora_lab")

_BLOCK = re.compile(r"\n*\[ESCA\].*?\[/ESCA\]\n*", re.S)


_TEXT = {
    "en": {"purpose": "Purpose: {}.", "incompatible": "Architecture by tensors: {} — does not work with Flux.2.",
           "trained_as": "Trained as “{}”.", "mismatch": "Civitai says {}, the tensors say {}."},
    "ru": {"purpose": "Назначение: {}.", "incompatible": "Архитектура по тензорам: {} — с Flux.2 не работает.",
           "trained_as": "Обучалась под именем «{}».", "mismatch": "Civitai указывает {}, тензоры — {}."},
}


def managed_tags(entry: dict[str, Any], lang: str = "en") -> list[str]:
    tags = [taxonomy.BY_KEY[entry["category"]].tag(lang)]
    arch_info = taxonomy.ARCH_INFO.get(entry["arch"])
    if arch_info:
        tags.append(arch_info[1])
    if entry["arch"] not in scan.FLUX2_FAMILY:
        tags.append(taxonomy.INCOMPATIBLE_TAG.get(lang, taxonomy.INCOMPATIBLE_TAG["en"]))
    if entry.get("trained_on"):
        tags.append(taxonomy.TRAINED_TAG.get(lang, taxonomy.TRAINED_TAG["en"]).format(entry["trained_on"]))
    return tags


def notes_block(entry: dict[str, Any], lang: str = "en") -> str:
    t = _TEXT.get(lang, _TEXT["en"])
    parts = [t["purpose"].format(taxonomy.BY_KEY[entry["category"]].label(lang))]
    if entry.get("note"):
        parts.append(entry["note"].rstrip(".") + ".")
    if entry["arch"] not in scan.FLUX2_FAMILY:
        parts.append(t["incompatible"].format(entry["arch"]))
    if entry.get("trained_as"):
        parts.append(t["trained_as"].format(entry["trained_as"]))
    if entry.get("civitai_base") and entry.get("base_model") and \
            not entry["civitai_base"].startswith(entry["base_model"].split("-")[0]):
        parts.append(t["mismatch"].format(entry["civitai_base"], entry["base_model"]))
    return "[ESCA] " + " ".join(parts) + " [/ESCA]"


def plan_update(entry: dict[str, Any], meta: dict[str, Any], lang: str = "en") -> dict[str, Any]:
    """Minimal LoRA Manager update for one entry; empty when already in sync."""
    upd: dict[str, Any] = {}
    current = [t for t in (meta.get("tags") or []) if isinstance(t, str)]
    kept = [t for t in current if not t.lower().startswith(taxonomy.MANAGED_PREFIX)]
    wanted = kept + managed_tags(entry, lang)
    if [t.lower() for t in wanted] != [t.lower() for t in current]:
        upd["tags"] = wanted
    target_base = entry.get("base_model") or ""
    have_base = meta.get("base_model") or ""
    # keep a more specific recorded variant of the same model (e.g. "...-base" vs ours)
    if target_base and have_base != target_base and not have_base.startswith(target_base):
        upd["base_model"] = target_base
    notes = meta.get("notes") or ""
    new_notes = (_BLOCK.sub("\n", notes).strip() + "\n\n" + notes_block(entry, lang)).strip()
    if new_notes != notes.strip():
        upd["notes"] = new_notes
    civ = meta.get("civitai") or {}
    if not civ.get("trainedWords") and entry.get("triggers"):
        upd["civitai"] = {"trainedWords": entry["triggers"][:4]}
    return upd


async def run(entries: list[dict[str, Any]], apply: bool, concurrency: int = 6) -> dict[str, Any]:
    plans = []
    lang = config.settings().get("language", "en")
    for e in entries:
        meta = catalog.lm_metadata(Path(e["path"]))
        if not meta:
            continue  # LoRA Manager has not scanned this file yet
        upd = plan_update(e, meta, lang)
        if upd:
            plans.append((e, upd))
    fields = Counter(k for _, u in plans for k in u)
    report: dict[str, Any] = {"checked": len(entries), "to_change": len(plans), "fields": dict(fields),
                              "examples": [{"file": e["rel"], "update": u} for e, u in plans[:5]],
                              "applied": 0, "errors": []}
    if not apply or not plans:
        return report
    lm = LoraManager()
    if not await lm.available():
        report["errors"].append("LoRA Manager недоступен — изменения не записаны")
        return report
    sem = asyncio.Semaphore(concurrency)

    async def one(e: dict[str, Any], u: dict[str, Any]) -> None:
        async with sem:
            try:
                await lm.save_metadata(e["path"].replace("\\", "/"), u)
                report["applied"] += 1
            except Exception as exc:  # keep going; report per file
                report["errors"].append(f"{e['rel']}: {exc}")

    await asyncio.gather(*(one(e, u) for e, u in plans))
    log.info("ESCA sync: %s of %s updates applied", report["applied"], len(plans))
    return report
