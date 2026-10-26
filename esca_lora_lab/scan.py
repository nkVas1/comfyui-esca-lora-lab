"""Header-only inspection of model files: no weights are loaded.

LoRA detection is a port of the author's ``rhapsode`` LoRA scanner: the base architecture
comes from tensor names and shapes (hidden width, block layout), never from the file name.
On top of it this module also identifies diffusion models (safetensors and GGUF) so a LoRA
can be checked against the model actually loaded in a graph.
"""

from __future__ import annotations

import collections
import json
import os
import re
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# (max double index, max single index) -> architecture, for factorised adapters (LoKr/LoHa)
_BLOCK_LAYOUT = {(7, 23): "flux2-klein-9b", (4, 19): "flux2-klein-4b", (18, 37): "flux1", (7, 47): "flux2-dev"}

_META_ARCH = {
    "flux2_klein_9b": "flux2-klein-9b", "flux_klein_9b": "flux2-klein-9b", "flux2_klein_4b": "flux2-klein-4b",
    "flux2": "flux2-dev", "flux2_dev": "flux2-dev", "flux.2": "flux2-dev", "zimage": "z-image", "z_image": "z-image",
    "flux1": "flux1", "flux": "flux1", "flux_dev": "flux1", "sdxl_base_v1-0": "sdxl", "sd_v1": "sd1.5",
    "qwen_image": "qwen-image", "wan21": "wan", "wan22": "wan", "minimax-h3": "minimax-h3",
}

FLUX2_FAMILY = ("flux2-klein-9b", "flux2-klein-4b", "flux2-dev")


@dataclass
class LoraScan:
    """What the header tells about one LoRA file."""

    arch: str
    rank: int | None = None
    fmt: str = "lora"
    blocks: str = ""
    trainer: str = ""
    base_meta: str = ""
    steps: int | None = None
    title: str = ""
    triggers: list[str] = field(default_factory=list)
    top_tags: list[str] = field(default_factory=list)
    trained_as: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_safetensors_header(path: Path) -> tuple[dict[str, dict], dict[str, str]]:
    """Return (tensor table, metadata) from a .safetensors header."""
    with path.open("rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        if n <= 0 or n > 200_000_000:
            raise ValueError(f"implausible header size {n}")
        header = json.loads(fh.read(n))
    meta = header.pop("__metadata__", None) or {}
    meta = {k: v for k, v in meta.items() if "thumbnail" not in k.lower()}  # never decode embedded images
    return header, meta


def _meta_arch(meta: dict[str, str]) -> str:
    for key in ("ss_base_model_version", "refined_architecture", "modelspec.architecture", "base_model"):
        raw = str(meta.get(key, "")).strip().lower()
        if not raw:
            continue
        raw = raw.split("/")[-1]
        for token, arch in _META_ARCH.items():
            if raw == token or raw.startswith(token):
                return arch
        if "klein" in raw and "9b" in raw:
            return "flux2-klein-9b"
        if "klein" in raw and "4b" in raw:
            return "flux2-klein-4b"
        return f"other:{raw}"
    return ""


def detect_lora_arch(tensors: dict[str, dict], meta: dict[str, str] | None = None) -> tuple[str, int | None, str, str]:
    """Return (architecture, rank, block summary, adapter format) from tensor names/shapes (+ metadata)."""
    meta = meta or {}
    joined = "\n".join(tensors)
    fmt = "lokr" if "lokr_w" in joined else ("loha" if "hada_w" in joined else "lora")
    if "policy_config" in meta or "proj_out_logweights" in joined:
        return "flux2-dev-piflow", None, "", "piflow"
    rank = None
    for k, v in tensors.items():
        if re.search(r"(lora_down|lora_A)(\.default)?\.weight$", k) and len(v["shape"]) == 2:
            rank = int(v["shape"][0])
            break
    if "lora_te1_" in joined or "lora_unet_down_blocks" in joined or "lora_unet_input_blocks" in joined:
        return ("sdxl" if "lora_te2_" in joined or "input_blocks_7" in joined else "sd1.5"), rank, "", fmt
    double = {int(m) for m in re.findall(r"double_blocks[._](\d+)", joined)}
    single = {int(m) for m in re.findall(r"single_blocks[._](\d+)", joined)}
    diff_single = {int(m) for m in re.findall(r"single_transformer_blocks[._](\d+)", joined)}
    blocks = f"double={len(double)} single={len(single) or len(diff_single)}"
    hidden = mlp_up = single_in = None
    for k, v in tensors.items():
        shape = v["shape"]
        if len(shape) != 2:
            continue
        if re.search(r"(img_attn[._]qkv|attn[._]to_q|img_attn[._]proj|linear1|attention[._]to_k)"
                     r".*(lora_down|lora_A)", k):
            hidden = hidden or int(shape[1])
        if re.search(r"img_mlp[._]0.*(lora_up|lora_B)", k):
            mlp_up = int(shape[0])
        if re.search(r"(single_transformer_blocks|single_blocks)[._]\d+[._](proj_out|linear2).*(lora_down|lora_A)", k):
            single_in = int(shape[1])
    by_meta = _meta_arch(meta)
    # Flux.2 Dev and Klein share layer names; only widths tell them apart.
    widths = collections.Counter(
        int(v["shape"][1]) for k, v in tensors.items()
        if len(v["shape"]) == 2 and re.search(r"(lora_down|lora_A)(\.default)?\.weight$", k)
        and int(v["shape"][1]) in (3072, 3840, 4096, 6144)
    )
    if widths:
        hidden = widths.most_common(1)[0][0]
    if hidden == 6144 or single_in == 24576:
        arch = "flux2-dev"
    elif hidden == 4096 or single_in == 16384:
        arch = "flux2-klein-9b"
    elif hidden == 3840 and "layers" in joined:
        arch = "z-image"
    elif hidden == 3072 or single_in in (15360, 12288 + 3072):
        arch = "flux2-klein-4b" if mlp_up == 18432 else "flux1"
    elif fmt in ("lokr", "loha") and double and single:
        arch = _BLOCK_LAYOUT.get((max(double), max(single)), by_meta or "unknown")
    else:
        arch = by_meta or "unknown"
    if arch == "unknown" and by_meta:
        arch = by_meta
    return arch, rank, blocks, fmt


def _top_tags(meta: dict[str, str], n: int = 12) -> list[str]:
    raw = meta.get("ss_tag_frequency")
    if not raw:
        return []
    try:
        freq = json.loads(raw)
    except (TypeError, ValueError):
        return []
    total: collections.Counter[str] = collections.Counter()
    for bucket in freq.values():
        total.update({k.strip(): int(v) for k, v in bucket.items()})
    return [t for t, _ in total.most_common(n)]


def _triggers(name: str, meta: dict[str, str], tags: list[str]) -> list[str]:
    found: list[str] = []
    m = re.search(r"trigger\s*[=:]\s*([^\]\)]+)", name, re.I)
    if m:
        found.append(m.group(1).strip())
    for key in ("ss_trigger", "trigger_word", "trigger", "modelspec.trigger_phrase", "activation_text"):
        if meta.get(key):
            found.append(str(meta[key]))
    if len(tags) > 3 and len(tags[0].split()) <= 3 and not tags[0].isdigit():
        found.append(tags[0])
    seen: set[str] = set()
    out = []
    for t in found:
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out[:4]


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-zа-я0-9]+", text.lower()) if len(t) >= 3 and not t.isdigit()} - {
        "lora", "flux", "flux2", "klein", "safetensors", "style", "v01", "000", "comfy", "converted"}


def _trained_as(stem: str, meta: dict[str, str]) -> str:
    """Training-time name when it shares nothing with the file name (renamed or mislabelled)."""
    trained = str(meta.get("ss_output_name") or meta.get("name") or "").strip()
    if not trained or re.match(r"^(training_\d+|my_first_lora|lora|output|last)\b", trained, re.I):
        return ""
    a, b = _tokens(stem), _tokens(trained)
    if a and b and not (a & b) and not any(x in y or y in x for x in a for y in b):
        return trained
    return ""


def scan_lora(path: Path) -> LoraScan:
    """Inspect one LoRA file; never raises — errors are reported in the result."""
    try:
        tensors, meta = read_safetensors_header(path)
    except (OSError, ValueError, struct.error, json.JSONDecodeError) as exc:
        return LoraScan(arch="unreadable", error=str(exc))
    arch, rank, blocks, fmt = detect_lora_arch(tensors, meta)
    tags = _top_tags(meta)
    steps = None
    for key in ("ss_steps", "ss_max_train_steps"):
        if str(meta.get(key, "")).isdigit():
            steps = int(meta[key])
            break
    return LoraScan(
        arch=arch, rank=rank, fmt=fmt, blocks=blocks,
        trainer="ai-toolkit" if "ai-toolkit" in str(meta.get("software", ""))
        else ("kohya" if any(k.startswith("ss_") for k in meta) else ""),
        base_meta=str(meta.get("ss_base_model_version") or meta.get("modelspec.architecture") or ""),
        steps=steps,
        title=str(meta.get("modelspec.title") or meta.get("ss_output_name") or meta.get("name") or ""),
        triggers=_triggers(path.stem, meta, tags),
        top_tags=tags,
        trained_as=_trained_as(path.stem, meta),
    )


# --- diffusion models (to check LoRA compatibility against the loaded model) -----------------

def _arch_from_hidden(hidden: int | None, has_flux2_modulation: bool) -> str:
    if hidden == 6144:
        return "flux2-dev"
    if hidden == 4096:
        return "flux2-klein-9b"
    if hidden == 3072:
        return "flux2-klein-4b" if has_flux2_modulation else "flux1"
    return "unknown"


def detect_model_arch_from_names(shapes: dict[str, list[int]]) -> str:
    """Architecture of a diffusion model from tensor names → shapes (safetensors or GGUF)."""
    names = "\n".join(shapes)
    if "input_blocks." in names or "down_blocks." in names:
        if "label_emb" in names or "add_embedding" in names:
            return "sdxl"
        return "sd1.5"
    hidden = None
    for key, shape in shapes.items():
        if re.search(r"double_blocks\.0\.img_attn\.qkv\.weight$", key) and len(shape) == 2:
            hidden = min(shape)  # [3*hidden, hidden] in torch order; reversed in GGUF
            break
    if hidden is None and "layers.0." in names and "cap_embedder" in names:
        return "z-image"
    return _arch_from_hidden(hidden, "double_stream_modulation" in names)


def _gguf_tensor_shapes(path: Path) -> dict[str, list[int]]:
    """Read tensor names and dims from a GGUF header (no tensor data)."""
    with path.open("rb") as fh:
        if fh.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        (version,) = struct.unpack("<I", fh.read(4))
        if version < 2:
            raise ValueError(f"unsupported GGUF version {version}")
        n_tensors, n_kv = struct.unpack("<QQ", fh.read(16))

        def read_str() -> str:
            (ln,) = struct.unpack("<Q", fh.read(8))
            return fh.read(ln).decode("utf-8", "replace")

        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

        def skip_value(vtype: int) -> None:
            if vtype == 8:
                read_str()
            elif vtype == 9:
                (atype,) = struct.unpack("<I", fh.read(4))
                (count,) = struct.unpack("<Q", fh.read(8))
                for _ in range(count):
                    skip_value(atype)
            else:
                fh.read(sizes[vtype])

        for _ in range(n_kv):
            read_str()
            (vtype,) = struct.unpack("<I", fh.read(4))
            skip_value(vtype)
        shapes: dict[str, list[int]] = {}
        for _ in range(n_tensors):
            name = read_str()
            (n_dims,) = struct.unpack("<I", fh.read(4))
            dims = list(struct.unpack(f"<{n_dims}Q", fh.read(8 * n_dims)))
            fh.read(4 + 8)  # ggml type + offset
            shapes[name] = dims
        return shapes


def scan_model(path: Path) -> str:
    """Architecture of a diffusion model / checkpoint file ('unknown' when unsure)."""
    try:
        if path.suffix.lower() == ".gguf":
            shapes = _gguf_tensor_shapes(path)
        else:
            tensors, _ = read_safetensors_header(path)
            shapes = {k.removeprefix("model.diffusion_model."): v["shape"] for k, v in tensors.items()}
        return detect_model_arch_from_names(shapes)
    except (OSError, ValueError, struct.error, json.JSONDecodeError, KeyError):
        return "unknown"


def is_compatible(lora_arch: str, model_arch: str) -> bool | None:
    """True/False when both are known; None when either side is unknown."""
    if lora_arch in ("unknown", "unreadable", "") or model_arch in ("unknown", ""):
        return None
    if lora_arch.startswith("other:"):
        return None
    return lora_arch == model_arch


def file_signature(path: Path) -> tuple[int, int]:
    st = os.stat(path)
    return st.st_size, st.st_mtime_ns
