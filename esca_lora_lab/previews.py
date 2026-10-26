"""Preview studio: uniform example images for every LoRA, rendered on the owner's own model.

Properties the owner asked for:
  * idempotent — a job's key is (preset, LoRA SHA-256, scene, seed, strength); planning
    again only adds what is missing, finished jobs are never redone, and moving/renaming
    a LoRA does not orphan its jobs;
  * stoppable — pause / resume / stop take effect after the current image; jobs survive
    ComfyUI restarts (SQLite), and the studio never starts by itself;
  * scalable — a prioritised queue (baselines first), polite mode that waits for the
    owner's own ComfyUI queue to drain, per-category scene/strength rules in the preset.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sqlite3
import time
from collections import deque
from pathlib import Path
from typing import Any

import aiohttp

from . import config
from .lm_bridge import LoraManager, server_url

log = logging.getLogger("esca_lora_lab")

CLIENT_ID = "esca-lora-lab"
PRESETS_DIR = config.MODULE_DIR / "presets"
MAX_ATTEMPTS = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY,
  preset TEXT NOT NULL,
  kind TEXT NOT NULL,
  lora_name TEXT, lora_rel TEXT, sha256 TEXT, category TEXT,
  scene TEXT NOT NULL, seed INTEGER NOT NULL, strength REAL NOT NULL,
  priority INTEGER NOT NULL DEFAULT 100,
  status TEXT NOT NULL DEFAULT 'pending',
  attempts INTEGER NOT NULL DEFAULT 0,
  output TEXT, thumb TEXT, lm_imported INTEGER NOT NULL DEFAULT 0,
  error TEXT, seconds REAL,
  created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_next ON jobs(status, priority, created_at);
"""


def load_presets() -> dict[str, dict[str, Any]]:
    """Bundled example presets, overridden by same-id presets in <data>/presets/."""
    presets = {}
    for folder in (PRESETS_DIR, config.data_dir() / "presets"):
        for f in sorted(folder.glob("previews.*.json")):
            data = json.loads(f.read_text(encoding="utf-8"))
            presets[data["id"]] = data
    return presets


def job_id(preset: str, sha: str, scene: str, seed: int, strength: float) -> str:
    return f"{preset}:{sha[:16] or 'baseline'}:{scene}:{seed}:{strength:+.2f}"


class JobStore:
    """SQLite-backed queue. All methods are short synchronous transactions."""

    def __init__(self, path: Path) -> None:
        self.path = path
        with self._db() as db:
            db.executescript(SCHEMA)

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def upsert(self, jobs: list[dict[str, Any]]) -> int:
        """Insert missing jobs; refresh the LoRA path of existing ones. Returns how many were added."""
        now = time.time()
        added = 0
        with self._db() as db:
            for j in jobs:
                cur = db.execute(
                    "INSERT OR IGNORE INTO jobs(job_id, preset, kind, lora_name, lora_rel, sha256, category, scene,"
                    " seed, strength, priority, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (j["job_id"], j["preset"], j["kind"], j.get("lora_name"), j.get("lora_rel"), j.get("sha256"),
                     j.get("category"), j["scene"], j["seed"], j["strength"], j["priority"], now, now))
                if cur.rowcount:
                    added += 1
                else:
                    db.execute("UPDATE jobs SET lora_rel=?, lora_name=? WHERE job_id=?",
                               (j.get("lora_rel"), j.get("lora_name"), j["job_id"]))
        return added

    def recover(self) -> None:
        with self._db() as db:
            db.execute("UPDATE jobs SET status='pending' WHERE status='running'")

    def next_job(self) -> dict[str, Any] | None:
        with self._db() as db:
            row = db.execute(
                "SELECT * FROM jobs WHERE status='pending' ORDER BY priority, created_at LIMIT 1").fetchone()
            if not row:
                return None
            db.execute("UPDATE jobs SET status='running', attempts=attempts+1, updated_at=? WHERE job_id=?",
                       (time.time(), row["job_id"]))
            return dict(row) | {"attempts": row["attempts"] + 1}

    def finish(self, jid: str, **fields: Any) -> None:
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._db() as db:
            db.execute(f"UPDATE jobs SET {cols} WHERE job_id=?", (*fields.values(), jid))

    def counts(self) -> dict[str, int]:
        with self._db() as db:
            return {r["status"]: r["n"] for r in db.execute("SELECT status, COUNT(*) n FROM jobs GROUP BY status")}

    def retry_failed(self) -> int:
        with self._db() as db:
            return db.execute("UPDATE jobs SET status='pending', attempts=0, error=NULL WHERE status='failed'").rowcount

    def recent(self, status: str, limit: int = 8) -> list[dict[str, Any]]:
        with self._db() as db:
            rows = db.execute("SELECT * FROM jobs WHERE status=? ORDER BY updated_at DESC LIMIT ?", (status, limit))
            return [dict(r) for r in rows]

    def get(self, jid: str) -> dict[str, Any] | None:
        with self._db() as db:
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (jid,)).fetchone()
            return dict(row) if row else None

    def for_lora(self, sha: str) -> list[dict[str, Any]]:
        with self._db() as db:
            rows = db.execute("SELECT * FROM jobs WHERE sha256=? AND status='done' ORDER BY scene, strength", (sha,))
            return [dict(r) for r in rows]

    def not_imported(self) -> list[dict[str, Any]]:
        with self._db() as db:
            rows = db.execute("SELECT * FROM jobs WHERE status='done' AND kind='lora' AND lm_imported=0")
            return [dict(r) for r in rows]


def plan_jobs(entries: list[dict[str, Any]], preset: dict[str, Any],
              include_nsfw: bool = False) -> list[dict[str, Any]]:
    """Every (LoRA × scene × seed × strength) the preset asks for, plus one baseline per scene/seed."""
    pid = preset["id"]
    excluded = set(preset.get("exclude_categories", []))
    jobs = []
    for scene in preset["default_scenes"]:
        for seed in preset["seeds"]:
            jobs.append({"job_id": job_id(pid, "", scene, seed, 0.0), "preset": pid, "kind": "baseline",
                         "scene": scene, "seed": seed, "strength": 0.0, "priority": 0})
    for e in entries:
        if e["arch"] != preset["model_arch"] or not e.get("sha256"):
            continue
        if e["category"] in excluded or (e["nsfw"] and not include_nsfw):
            continue
        rule = preset.get("by_category", {}).get(e["category"], {})
        priority = 10 if e["category"] == "unknown" else (20 if not e.get("has_preview") else 100)
        for scene in rule.get("scenes", preset["default_scenes"]):
            for seed in preset["seeds"]:
                for strength in rule.get("strengths", preset["default_strengths"]):
                    jobs.append({
                        "job_id": job_id(pid, e["sha256"], scene, seed, float(strength)), "preset": pid,
                        "kind": "lora", "lora_name": e["name"], "lora_rel": e["rel"], "sha256": e["sha256"],
                        "category": e["category"], "scene": scene, "seed": seed, "strength": float(strength),
                        "priority": priority,
                    })
    return jobs


def build_graph(preset: dict[str, Any], job: dict[str, Any], resolve_lora, trigger: str, prefix: str) -> dict[str, Any]:
    """ComfyUI API graph for one preview (core nodes, plus UnetLoaderGGUF for GGUF presets)."""
    unet_node = ({"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": preset["unet"]}}
                 if preset["unet"].lower().endswith(".gguf") else
                 {"class_type": "UNETLoader", "inputs": {"unet_name": preset["unet"],
                                                         "weight_dtype": preset.get("weight_dtype", "default")}})
    g: dict[str, dict[str, Any]] = {
        "unet": unet_node,
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": preset["clip"], "type": preset.get("clip_type", "flux2")}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": preset["vae"]}},
    }
    model: list[Any] = ["unet", 0]
    chain = [(resolve_lora(name), strength) for name, strength in preset.get("base_loras", [])]
    if job["kind"] == "lora" and job["strength"] != 0:
        chain.append((job["lora_rel"].replace("/", os.sep), job["strength"]))
    for i, (name, strength) in enumerate(chain):
        g[f"lora{i}"] = {"class_type": "LoraLoaderModelOnly",
                         "inputs": {"model": model, "lora_name": name, "strength_model": strength}}
        model = [f"lora{i}", 0]
    w, h = preset["width"], preset["height"]
    text = preset["scenes"][job["scene"]]
    if trigger:
        text = f"{trigger}, {text}"
    g["shift"] = {"class_type": "ModelSamplingFlux",
                  "inputs": {"model": model, "max_shift": 0.7, "base_shift": 0.46, "width": w, "height": h}}
    g["text"] = {"class_type": "CLIPTextEncode", "inputs": {"text": text, "clip": ["clip", 0]}}
    g["latent"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}}
    g["sigmas"] = {"class_type": "Flux2Scheduler", "inputs": {"steps": preset["steps"], "width": w, "height": h}}
    g["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": job["seed"]}}
    g["sampler"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": preset.get("sampler", "euler")}}
    g["guider"] = {"class_type": "BasicGuider", "inputs": {"model": ["shift", 0], "conditioning": ["text", 0]}}
    g["sample"] = {"class_type": "SamplerCustomAdvanced",
                   "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                              "sigmas": ["sigmas", 0], "latent_image": ["latent", 0]}}
    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}}
    g["save"] = {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}}
    return g


class PreviewStudio:
    def __init__(self, catalog_provider) -> None:
        self.dir = config.data_dir() / "previews"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.store = JobStore(config.data_dir() / "previews.sqlite")
        self.catalog = catalog_provider
        self.state = "idle"            # idle | running | paused | stopping
        self.current: dict[str, Any] | None = None
        self.polite = True
        self._task: asyncio.Task | None = None
        self._durations: deque[float] = deque(maxlen=20)
        self.last_error = ""

    # -- control -----------------------------------------------------------------------
    def plan(self, preset_ids: list[str] | None = None, include_nsfw: bool = False) -> dict[str, Any]:
        presets = load_presets()
        entries = self.catalog()
        added = {}
        for pid, preset in presets.items():
            if preset_ids and pid not in preset_ids:
                continue
            added[pid] = self.store.upsert(plan_jobs(entries, preset, include_nsfw))
        return {"added": added, **self.status()}

    def start(self) -> dict[str, Any]:
        if self._task and not self._task.done():
            if self.state == "paused":
                self.state = "running"
            return self.status()
        self.store.recover()
        self.state = "running"
        self._task = asyncio.get_running_loop().create_task(self._loop())
        return self.status()

    def pause(self) -> dict[str, Any]:
        if self.state == "running":
            self.state = "paused"
        return self.status()

    def resume(self) -> dict[str, Any]:
        return self.start()

    def stop(self) -> dict[str, Any]:
        if self.state in ("running", "paused"):
            self.state = "stopping"
        return self.status()

    def status(self) -> dict[str, Any]:
        counts = self.store.counts()
        pending = counts.get("pending", 0) + counts.get("running", 0)
        avg = sum(self._durations) / len(self._durations) if self._durations else None
        return {
            "state": self.state, "counts": counts, "current": self.current,
            "avg_seconds": round(avg, 1) if avg else None,
            "eta_minutes": round(pending * avg / 60) if avg else None,
            "last_error": self.last_error, "polite": self.polite,
        }

    # -- runner ------------------------------------------------------------------------
    async def _loop(self) -> None:
        presets = load_presets()
        lm = LoraManager()
        base = server_url()
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60), trust_env=False) as http:
                while self.state != "stopping":
                    if self.state == "paused":
                        await asyncio.sleep(1)
                        continue
                    if self.polite and await self._owner_busy(http, base):
                        await asyncio.sleep(3)
                        continue
                    job = self.store.next_job()
                    if not job:
                        break
                    self.current = {k: job[k] for k in ("job_id", "kind", "lora_name", "scene", "strength")}
                    t0 = time.time()
                    try:
                        out = await self._render(http, base, presets[job["preset"]], job)
                        thumb = self._thumbnail(out)
                        secs = time.time() - t0
                        self._durations.append(secs)
                        self.store.finish(job["job_id"], status="done", output=str(out), thumb=str(thumb),
                                          seconds=secs, error=None)
                    except Exception as exc:  # one bad LoRA must not stop the run
                        self.last_error = f"{job['lora_name'] or job['scene']}: {exc}"
                        log.warning("ESCA preview failed %s: %s", job["job_id"], exc)
                        status = "failed" if job["attempts"] >= MAX_ATTEMPTS else "pending"
                        self.store.finish(job["job_id"], status=status, error=str(exc)[:500])
                        continue
                    finally:
                        self.current = None
                    if job["kind"] == "lora":
                        await self._import_to_lm(lm, job["job_id"], job["sha256"], out)
        finally:
            self.state = "idle"
            self.current = None

    async def _import_to_lm(self, lm: LoraManager, jid: str, sha: str, png: Path) -> None:
        """Register a finished preview as a LoRA Manager example; a failure here never redoes the render."""
        try:
            if await lm.available():
                await lm.import_examples(sha, [str(png)])
                self.store.finish(jid, lm_imported=1)
        except Exception as exc:
            self.last_error = f"LoRA Manager import: {exc}"
            log.warning("ESCA preview import to LoRA Manager failed for %s: %s", jid, exc)

    async def import_pending(self) -> int:
        """Retry LoRA Manager import for finished previews that were not registered yet."""
        lm = LoraManager()
        done = 0
        for job in self.store.not_imported():
            await self._import_to_lm(lm, job["job_id"], job["sha256"], Path(job["output"]))
            done += 1
        return done

    async def _owner_busy(self, http: aiohttp.ClientSession, base: str) -> bool:
        """True when the ComfyUI queue holds someone else's work (we yield to the owner)."""
        try:
            async with http.get(f"{base}/queue") as r:
                q = await r.json()
        except Exception:
            return False
        for item in [*q.get("queue_running", []), *q.get("queue_pending", [])]:
            extra = item[3] if len(item) > 3 and isinstance(item[3], dict) else {}
            if extra.get("client_id") != CLIENT_ID:
                return True
        return False

    def _resolve_lora(self, name: str) -> str:
        for e in self.catalog():
            if e["name"] == name:
                return e["rel"].replace("/", os.sep)
        raise FileNotFoundError(f"base LoRA '{name}' not in catalog")

    def _trigger(self, job: dict[str, Any]) -> str:
        if job["kind"] != "lora":
            return ""
        for e in self.catalog():
            if e.get("sha256") == job["sha256"]:
                return e["triggers"][0] if e.get("triggers") else ""
        return ""

    async def _render(self, http: aiohttp.ClientSession, base: str, preset: dict[str, Any],
                      job: dict[str, Any]) -> Path:
        folder = job["sha256"][:16] if job["kind"] == "lora" else f"_baseline-{preset['id']}"
        prefix = f"esca-lora-lab/{folder}/{job['scene']}_{job['seed']}_{job['strength']:+.2f}"
        graph = build_graph(preset, job, self._resolve_lora, self._trigger(job), prefix)
        async with http.post(f"{base}/prompt", json={"prompt": graph, "client_id": CLIENT_ID}) as r:
            body = await r.json(content_type=None)
            if r.status != 200:
                raise RuntimeError(f"queue rejected: {json.dumps(body)[:400]}")
        pid = body["prompt_id"]
        deadline = time.time() + 1800
        while time.time() < deadline:
            await asyncio.sleep(1.5)
            async with http.get(f"{base}/history/{pid}") as r:
                hist = await r.json(content_type=None)
            if pid not in hist:
                continue
            entry = hist[pid]
            if entry.get("status", {}).get("status_str") == "error":
                raise RuntimeError("execution error")
            for node_out in entry.get("outputs", {}).values():
                for img in node_out.get("images", []):
                    src = Path(_output_dir()) / img.get("subfolder", "") / img["filename"]
                    dst = self.dir / folder / f"{job['scene']}_{job['seed']}_{job['strength']:+.2f}.png"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), dst)
                    return dst
            raise RuntimeError("no image in outputs")
        raise TimeoutError("render timed out")

    @staticmethod
    def _thumbnail(png: Path) -> Path:
        from PIL import Image

        thumb = png.with_suffix(".webp")
        with Image.open(png) as im:
            im.thumbnail((384, 384))
            im.save(thumb, quality=82)
        return thumb


def _output_dir() -> str:
    try:
        import folder_paths  # type: ignore[import-not-found]

        return folder_paths.get_output_directory()
    except ImportError:
        return str(config.data_dir() / "output")
