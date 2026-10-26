"""Regression tests for the parts that must never silently drift: architecture detection,
classification precedence, preview-queue idempotency and sync idempotency."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from esca_lora_lab import previews, scan, sync, taxonomy  # noqa: E402


QKV = "diffusion_model.double_blocks.0.img_attn.qkv"


def _lora_header(tmp: Path, name: str, width: int, rank: int = 16, prefix: str = QKV) -> Path:
    header = {
        f"{prefix}.lora_A.weight": {"dtype": "BF16", "shape": [rank, width], "data_offsets": [0, 0]},
        f"{prefix}.lora_B.weight": {"dtype": "BF16", "shape": [width * 3, rank], "data_offsets": [0, 0]},
        "__metadata__": {"ss_output_name": "something_else_entirely"},
    }
    raw = json.dumps(header).encode()
    path = tmp / f"{name}.safetensors"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    return path


@pytest.mark.parametrize(("width", "arch"), [(4096, "flux2-klein-9b"), (6144, "flux2-dev"), (3072, "flux1")])
def test_lora_arch_from_hidden_width(tmp_path: Path, width: int, arch: str) -> None:
    s = scan.scan_lora(_lora_header(tmp_path, f"w{width}", width))
    assert s.arch == arch
    assert s.rank == 16


def test_sd15_and_unreadable(tmp_path: Path) -> None:
    sd = _lora_header(tmp_path, "sd", 768, prefix="lora_unet_down_blocks_0_attentions_0_proj_in")
    assert scan.scan_lora(sd).arch == "sd1.5"
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"\x00" * 4)
    assert scan.scan_lora(bad).arch == "unreadable"


def test_model_arch_from_names() -> None:
    q = "double_blocks.0.img_attn.qkv.weight"
    klein = {q: [12288, 4096], "double_stream_modulation_img.lin.weight": [1, 1]}
    assert scan.detect_model_arch_from_names(klein) == "flux2-klein-9b"
    assert scan.detect_model_arch_from_names({q: [6144, 18432]}) == "flux2-dev"   # GGUF order is reversed
    assert scan.detect_model_arch_from_names({q: [9216, 3072]}) == "flux1"
    assert scan.is_compatible("flux1", "flux2-klein-9b") is False
    assert scan.is_compatible("unknown", "flux2-klein-9b") is None


@pytest.mark.parametrize(("path", "kw", "expected"), [
    ("klein/K-Slider shoot control.safetensors", {"title": "K-Slider : Shoot control"}, "slider"),
    ("klein/HyperdetailedPencil.safetensors",
     {"title": "Hyperdetailed Colored Pencil", "tags": ["niji", "style"]}, "style-paint"),
    ("klein/olympus.safetensors", {"title": "Olympus UltraReal", "tags": ["realistic"]}, "style-photo"),
    ("klein/Flux9b_FemaleNudity_Rank128.safetensors", {}, "nsfw"),
    ("klein/klein_9B_Turbo_r128.safetensors", {"title": "Klein 4B/9B Base to Turbo Lora"}, "speed"),
    ("klein/x.safetensors", {"title": "Realistic Skin & Face - Professional"}, "quality"),
])
def test_classification_precedence(path: str, kw: dict, expected: str) -> None:
    assert taxonomy.classify(path, **kw).category == expected


def _entry(sha: str, cat: str = "style-paint", arch: str = "flux2-klein-9b", nsfw: bool = False) -> dict:
    return {"name": f"lora_{sha}", "rel": f"flux2/klein/lora_{sha}.safetensors", "sha256": sha * 8, "arch": arch,
            "category": cat, "nsfw": nsfw, "has_preview": True}


PRESET = {"id": "p", "model_arch": "flux2-klein-9b", "seeds": [1], "default_scenes": ["a", "b"],
          "default_strengths": [1.0], "by_category": {"slider": {"scenes": ["a"], "strengths": [-1.5, 1.5]}},
          "exclude_categories": ["control"]}


def test_plan_rules_and_filters() -> None:
    entries = [_entry("a"), _entry("b", "slider"), _entry("c", "control"), _entry("d", nsfw=True),
               _entry("e", arch="flux1")]
    jobs = previews.plan_jobs(entries, PRESET)
    kinds = [(j["kind"], j.get("lora_name"), j["scene"], j["strength"]) for j in jobs]
    assert ("baseline", None, "a", 0.0) in kinds and ("baseline", None, "b", 0.0) in kinds
    assert sum(1 for k in kinds if k[1] == "lora_a") == 2              # two default scenes
    assert {k[3] for k in kinds if k[1] == "lora_b"} == {-1.5, 1.5}   # slider rule
    assert not any(k[1] in ("lora_c", "lora_d", "lora_e") for k in kinds)


def test_job_store_is_idempotent_and_survives_moves(tmp_path: Path) -> None:
    store = previews.JobStore(tmp_path / "q.sqlite")
    jobs = previews.plan_jobs([_entry("a")], PRESET)
    assert store.upsert(jobs) == len(jobs)
    assert store.upsert(jobs) == 0                                     # planning again adds nothing
    first = store.next_job()
    store.finish(first["job_id"], status="done")
    moved = previews.plan_jobs([{**_entry("a"), "rel": "flux2/klein/style-paint/lora_a.safetensors"}], PRESET)
    assert store.upsert(moved) == 0                                    # same SHA → same jobs
    assert store.get(first["job_id"])["status"] == "done"              # finished work is kept
    running = store.next_job()
    store.recover()                                                    # crash while running → pending again
    assert store.get(running["job_id"])["status"] == "pending"


def test_sync_update_is_idempotent() -> None:
    entry = {"category": "style-paint", "category_label": "Стиль", "arch": "flux1", "trained_on": "", "note": "",
             "trained_as": "", "civitai_base": "Flux.1 D", "base_model": "Flux.1 D", "triggers": ["tw"]}
    meta = {"tags": ["mine", "esca:old"], "base_model": "Flux.1 D", "notes": "my notes", "civitai": {}}
    upd = sync.plan_update(entry, meta)
    assert "mine" in upd["tags"] and "esca:old" not in upd["tags"]
    assert taxonomy.INCOMPATIBLE_TAG["en"] in upd["tags"]
    assert upd["notes"].startswith("my notes") and "[ESCA]" in upd["notes"]
    applied = {**meta, **{k: v for k, v in upd.items() if k != "civitai"}, "civitai": upd["civitai"]}
    assert sync.plan_update(entry, applied) == {}                     # second run changes nothing


def test_overrides_and_tag_language() -> None:
    cls = taxonomy.classify("klein/x.safetensors", title="Some Wolf LoRA", title_overrides={"Some Wolf": "concept"})
    assert cls.category == "concept" and cls.reason == "manual"
    manual = taxonomy.classify("klein/odd_name.safetensors", path_overrides={"odd_name": ["slider", "note"]})
    assert (manual.category, manual.note) == ("slider", "note")
    entry = {"category": "style-paint", "arch": "flux2-klein-9b", "trained_on": "base"}
    assert sync.managed_tags(entry, "en") == ["esca:painting", "esca:klein-9b", "esca:trained-on-base"]
    assert sync.managed_tags(entry, "ru")[0] == "esca:живопись"
