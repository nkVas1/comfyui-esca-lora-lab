"""Reversible reorganisation of the managed LoRA sub-tree by compatibility and purpose.

Layout (relative to each LoRA root):
    flux2/klein/<purpose>/   flux2/dev/<purpose>/   flux1/  sd15/  sdxl/  zimage/  other/
Moves go through LoRA Manager (it carries sidecars/previews and keeps its cache coherent);
workflow JSON files are rewritten with the *actual* new paths LoRA Manager reports, after
backing them up. Every step is journaled so ``rollback`` restores the previous state.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from . import config, scan, taxonomy
from .lm_bridge import LoraManager

log = logging.getLogger("esca_lora_lab")


def target_dir(entry: dict[str, Any], scope: str) -> Path | None:
    rel = PurePosixPath(entry["rel"])
    if scope and (not rel.parts or rel.parts[0] != scope):
        return None
    root = Path(entry["root"])
    info = taxonomy.ARCH_INFO.get(entry["arch"])
    if entry["arch"] in scan.FLUX2_FAMILY and info:
        return root / info[2] / taxonomy.BY_KEY[entry["category"]].folder
    return root / (info[2] if info else "other")


def plan(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scope = config.settings()["reorg_scope"]
    moves = []
    for e in entries:
        dst = target_dir(e, scope)
        src = Path(e["path"])
        if dst is None or src.parent.resolve() == dst.resolve():
            continue
        moves.append({"src": str(src), "dst_dir": str(dst), "root": e["root"], "name": e["name"]})
    return moves


def _loader_name(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("/", "\\")


def workflow_files() -> list[Path]:
    dirs = [Path(config.data_dir()).parent / "default" / "workflows"]
    dirs += [Path(d) for d in config.settings().get("extra_workflow_dirs", [])]
    return [p for d in dirs if d.is_dir() for p in d.rglob("*.json")]


def _rewrite(obj: Any, mapping: dict[str, str], hits: list[str]) -> Any:
    if isinstance(obj, str):
        new = mapping.get(obj.replace("/", "\\").lower())
        if new:
            hits.append(obj)
            return new
        return obj
    if isinstance(obj, list):
        return [_rewrite(x, mapping, hits) for x in obj]
    if isinstance(obj, dict):
        return {k: _rewrite(v, mapping, hits) for k, v in obj.items()}
    return obj


def references(mapping: dict[str, str]) -> dict[str, int]:
    found = {}
    for wf in workflow_files():
        try:
            data = json.loads(wf.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        hits: list[str] = []
        _rewrite(data, mapping, hits)
        if hits:
            found[str(wf)] = len(hits)
    return found


def preview(entries: list[dict[str, Any]]) -> dict[str, Any]:
    moves = plan(entries)
    by_dir: dict[str, int] = defaultdict(int)
    mapping = {}
    for m in moves:
        root, src = Path(m["root"]), Path(m["src"])
        by_dir[str(Path(m["dst_dir"]).relative_to(root)).replace("\\", "/")] += 1
        mapping[_loader_name(src, root).lower()] = _loader_name(Path(m["dst_dir"]) / src.name, root)
    return {"moves": len(moves), "by_folder": dict(sorted(by_dir.items())), "workflow_refs": references(mapping)}


async def apply(entries: list[dict[str, Any]]) -> dict[str, Any]:
    moves = plan(entries)
    logs = config.data_dir() / "reorg-logs"
    logs.mkdir(exist_ok=True)
    run = time.strftime("%Y%m%d-%H%M%S")
    journal = logs / f"{run}.jsonl"
    lm = LoraManager()
    use_lm = await lm.available()
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in moves:
        groups[m["dst_dir"]].append(m)
    mapping: dict[str, str] = {}
    moved = failed = 0
    with journal.open("a", encoding="utf-8") as j:
        for dst_dir, items in groups.items():
            results = await _move(lm if use_lm else None, [i["src"] for i in items], dst_dir)
            for item, (ok, new_path) in zip(items, results, strict=True):
                if not ok:
                    failed += 1
                    continue
                moved += 1
                root = Path(item["root"])
                mapping[_loader_name(Path(item["src"]), root).lower()] = _loader_name(Path(new_path), root)
                j.write(json.dumps({"op": "move", "src": item["src"], "dst": new_path}, ensure_ascii=False) + "\n")
        backup = logs / f"{run}-workflows"
        rewritten = 0
        for wf in workflow_files():
            text = wf.read_text(encoding="utf-8", errors="replace")
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue
            hits: list[str] = []
            new = _rewrite(data, mapping, hits)
            if not hits:
                continue
            keep = backup / f"{len(list(backup.glob('*'))) if backup.exists() else 0:03d}_{wf.name}"
            keep.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(wf, keep)
            wf.write_text(json.dumps(new, ensure_ascii=False, indent=2 if "\n " in text[:2000] else None),
                          encoding="utf-8")
            j.write(json.dumps({"op": "workflow", "path": str(wf), "backup": str(keep)}, ensure_ascii=False) + "\n")
            rewritten += 1
    log.info("ESCA reorg %s: moved %s, failed %s, workflows %s", run, moved, failed, rewritten)
    return {"run": run, "journal": str(journal), "moved": moved, "failed": failed,
            "workflows_rewritten": rewritten, "via": "LoRA Manager" if use_lm else "file system"}


async def _move(lm: LoraManager | None, sources: list[str], dst_dir: str) -> list[tuple[bool, str]]:
    if lm is not None:
        res = await lm.move_bulk([s.replace("\\", "/") for s in sources], dst_dir.replace("\\", "/"))
        by_src = {os.path.normcase(os.path.normpath(r["original_file_path"])): r for r in res.get("results", [])}
        out = []
        for s in sources:
            r = by_src.get(os.path.normcase(os.path.normpath(s)), {})
            out.append((bool(r.get("success")) and bool(r.get("new_file_path")), r.get("new_file_path") or ""))
        return out
    out = []
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    for s in sources:
        src = Path(s)
        try:
            for f in src.parent.glob(src.stem + ".*"):
                shutil.move(str(f), str(Path(dst_dir) / f.name))
            out.append((True, str(Path(dst_dir) / src.name)))
        except OSError as exc:
            log.warning("move failed %s: %s", s, exc)
            out.append((False, ""))
    return out


async def rollback(run: str) -> dict[str, Any]:
    journal = config.data_dir() / "reorg-logs" / f"{run}.jsonl"
    entries = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if line.strip()]
    lm = LoraManager()
    use_lm = await lm.available()
    back: dict[str, list[str]] = defaultdict(list)
    for e in entries:
        if e["op"] == "move" and Path(e["dst"]).exists():
            back[str(Path(e["src"]).parent)].append(e["dst"])
    restored = 0
    for dst_dir, sources in back.items():
        results = await _move(lm if use_lm else None, sources, dst_dir)
        restored += sum(ok for ok, _ in results)
    workflows = 0
    for e in entries:
        if e["op"] == "workflow":
            shutil.copy2(e["backup"], e["path"])
            workflows += 1
    for d in sorted({str(Path(e["dst"]).parent) for e in entries if e["op"] == "move"}, key=len, reverse=True):
        with contextlib.suppress(OSError):
            os.rmdir(d)  # only succeeds when the folder is empty
    return {"run": run, "restored": restored, "workflows_restored": workflows}
