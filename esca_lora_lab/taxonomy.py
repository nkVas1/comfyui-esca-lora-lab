"""Purpose taxonomy for LoRAs and the mapping to LoRA Manager's base-model names.

Classification order matters: specific purposes (speed, sliders, edit tools) win over
generic style words. Corrections for one's own library come from settings
(``title_overrides`` / ``path_overrides``), never from code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePath


@dataclass(frozen=True)
class Category:
    key: str
    label_en: str
    label_ru: str
    folder: str     # sub-folder used by the reorganisation
    tag_en: str     # managed LoRA Manager tags (namespace "esca:")
    tag_ru: str

    def label(self, lang: str) -> str:
        return self.label_ru if lang == "ru" else self.label_en

    def tag(self, lang: str) -> str:
        return self.tag_ru if lang == "ru" else self.tag_en


CATEGORIES: tuple[Category, ...] = (
    Category("speed", "Speed & distillation", "Скорость и дистилляция", "_speed", "esca:speed", "esca:скорость"),
    Category("control", "Control & editing", "Управление и редактирование", "control", "esca:control",
             "esca:управление"),
    Category("slider", "Sliders", "Слайдеры", "sliders", "esca:slider", "esca:слайдер"),
    Category("quality", "Quality & detail", "Качество и детализация", "quality", "esca:quality", "esca:качество"),
    Category("style-paint", "Style: painting & illustration", "Стиль: живопись и иллюстрация", "style-paint",
             "esca:painting", "esca:живопись"),
    Category("style-photo", "Style: photo, film, cinema", "Стиль: фото, плёнка, кино", "style-photo", "esca:photo",
             "esca:фото"),
    Category("style-graphic", "Style: graphic & design", "Стиль: графика и дизайн", "style-graphic", "esca:graphic",
             "esca:графика"),
    Category("style-3d", "Style: 3D, CG, games", "Стиль: 3D, CG, игры", "style-3d", "esca:3d", "esca:3d"),
    Category("anime", "Anime & cartoon", "Аниме и мультфильм", "anime", "esca:anime", "esca:аниме"),
    Category("character", "Characters & people", "Персонажи и люди", "character", "esca:character",
             "esca:персонажи"),
    Category("concept", "Concepts & scenes", "Концепты и сцены", "concept", "esca:concept", "esca:концепты"),
    Category("nsfw", "Adult", "Для взрослых", "nsfw", "esca:nsfw", "esca:nsfw"),
    Category("unknown", "Unclassified", "Не определено", "_unsorted", "esca:unclassified", "esca:не-определено"),
)
BY_KEY = {c.key: c for c in CATEGORIES}

# architecture -> (LoRA Manager base_model, short tag, folder)
ARCH_INFO: dict[str, tuple[str, str, str]] = {
    "flux2-klein-9b": ("Flux.2 Klein 9B", "esca:klein-9b", "flux2/klein"),
    "flux2-klein-4b": ("Flux.2 Klein 4B", "esca:klein-4b", "flux2/klein-4b"),
    "flux2-dev": ("Flux.2 D", "esca:flux2-dev", "flux2/dev"),
    "flux1": ("Flux.1 D", "esca:flux1", "flux1"),
    "sdxl": ("SDXL 1.0", "esca:sdxl", "sdxl"),
    "sd1.5": ("SD 1.5", "esca:sd15", "sd15"),
    "z-image": ("ZImageTurbo", "esca:z-image", "zimage"),
}
INCOMPATIBLE_TAG = {"en": "esca:not-for-flux2", "ru": "esca:не-для-flux2"}
TRAINED_TAG = {"en": "esca:trained-on-{}", "ru": "esca:обучена-на-{}"}
MANAGED_PREFIX = "esca:"

NSFW_WORDS = re.compile(
    r"\b(nsfw|nude|nudes|nudity|naked|porn|sex|sexy|erotic|penis|vagina|vulva|clitoris|anus|breasts?|"
    r"nipples?|cum|blowjob|deepthroat|cowgirl|missionary|ahegao|undress|genitalia|foreskin|cameltoe|"
    r"dildo|lingerie|topless|xxx|snofs|eros)\b", re.I)

RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("speed", re.compile(r"\bturbo\b|\blightning\b|\b\d\s?steps?\b|\bpi ?flow\b|\bpiid\b", re.I)),
    ("slider", re.compile(r"\bslider\b", re.I)),
    ("control", re.compile(
        r"refcontrol|controlnet|\bdepth control|lineart control|reference (depth|lineart)|\bpose\b|"
        r"head ?swap|face ?swap|\brelight|\bre light\b|multi ?angle|camera angle|\brestor|\brepair\b|upscal|"
        r"style transfer|\btransfer\b|\bblend\b|\bfusion\b|try ?on|inpaint|outpaint|\bedit\b|\bremove\b|"
        r"\bextract\b|\breplace\b|(anything|anime|cartoon|illustration) ?(2|to) ?real\b|mannequin|"
        r"seam ?fix|\bsprite\b|\bfont", re.I)),
    ("style-3d", re.compile(
        r"\b3d\b|\brender|blender|figurine|\bclay\b|\bcg\b|cgstyle|\bgame\b|video game|skyrim|elden|"
        r"fallout|baldur|botw|zelda|fromsoft|xbox|unreal", re.I)),
    ("style-paint", re.compile(
        r"\bpaint|painterly|illustrat|\bdrawing|sketch|pencil|watercolou?r|\boil\b|\bink\b|gouache|impasto|"
        r"airbrush|bryulov|caravaggio|da vinci|aenami|fairytale|gothic", re.I)),
    ("anime", re.compile(r"\banime\b|disney|cartoon|cel ?shad|\bmanga\b|\bcomic|chibi|pixar|\bniji\b|moebius", re.I)),
    ("style-graphic", re.compile(
        r"\blogo|\bposter\b|\balbum\b|book cover|mockup|\bgraphic|minimal|line ?art|typograph|sticker|"
        r"\bicon|isometric|phone case|digital asset|flat design|\bvector|newspaper", re.I)),
    ("style-photo", re.compile(
        r"\bphoto|\bfilm\b|kodak|\banalog|digicam|cctv|\bcamera\b|\bmovie|\bstill\b|instagram|snapchat|"
        r"selfie|\blomo\b|disposable|panavision|technicolor|35mm|polaroid|\bvintage\b|\bhdr\b|ultra ?real", re.I)),
    ("quality", re.compile(
        r"\bdetail|realism|realistic|quality|fixer|anatomy|\bskin\b|\bsharp|enhancer|\bdpo\b|hi ?res", re.I)),
    ("style-paint", re.compile(r"\bstyle\b|\bart\b|artist|aesthetic", re.I)),
    ("character", re.compile(
        r"\bcharacter|\bperson\b|\bwoman\b|\bman\b|\bgirl\b|\bboy\b|\bface\b|portrait|celebrity", re.I)),
    ("concept", re.compile(
        r"\bconcept|\bscene\b|\bobject|environment|hdri|panorama|fish ?eye|inflate|x ?ray|booth", re.I)),
)


def normalise(text: str) -> str:
    """Split camelCase and separators so word boundaries work on file names."""
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    return re.sub(r"[_\-.\[\]()|:,/]+", " ", text)


@dataclass(frozen=True)
class Classification:
    category: str
    nsfw: bool
    reason: str
    note: str = ""


def classify(path: str, *, title: str = "", version: str = "", tags: list[str] | None = None,
             trained_words: list[str] | None = None, scan_title: str = "", scan_triggers: list[str] | None = None,
             civitai_nsfw: bool = False, title_overrides: dict[str, str] | None = None,
             path_overrides: dict[str, list[str]] | None = None) -> Classification:
    """Purpose category for one LoRA from everything known about it."""
    stem = PurePath(path.replace("\\", "/")).stem
    text = normalise(" ".join(filter(None, [
        stem, scan_title, " ".join(scan_triggers or []), title, version, " ".join(tags or []),
        " ".join(trained_words or []),
    ])))
    nsfw = civitai_nsfw or bool(NSFW_WORDS.search(text))
    manual = (path_overrides or {}).get(stem)
    if manual and manual[0] in BY_KEY:
        cat = manual[0]
        if nsfw and cat not in ("speed", "control", "slider"):
            cat = "nsfw"
        return Classification(cat, nsfw, "manual", manual[1] if len(manual) > 1 else "")
    if nsfw:
        return Classification("nsfw", True, "tags or name")
    for prefix, cat in (title_overrides or {}).items():
        if title.startswith(prefix) and cat in BY_KEY:
            return Classification(cat, False, "manual")
    for cat, rx in RULES:
        m = rx.search(text)
        if m:
            return Classification(cat, False, f"“{m.group(0).strip()}”")
    return Classification("unknown", False, "no signal")


def base_model_for(arch: str, civitai_base: str = "", hints: str = "") -> str:
    """LoRA Manager base_model string for a tensor-proven architecture.

    Keeps variants the tensors cannot tell apart but that matter in practice: Klein
    base/distilled (from Civitai), Flux.1 Kontext and Z-Image Base/Turbo (from Civitai or
    the training metadata/file name in ``hints``).
    """
    text = f"{civitai_base} {hints}".lower()
    if arch == "flux2-klein-9b" and "base" in civitai_base.lower():
        return "Flux.2 Klein 9B-base"
    if arch == "flux2-klein-4b" and "base" in civitai_base.lower():
        return "Flux.2 Klein 4B-base"
    if arch == "flux1" and "kontext" in text:
        return "Flux.1 Kontext"
    if arch == "z-image" and "base" in text and "turbo" not in text:
        return "ZImageBase"
    return ARCH_INFO.get(arch, ("", "", ""))[0]
