"""Load subject configs: style rules, role templates, preamble, layout, examples."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
import yaml

from . import config

EXAMPLE_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
MAX_EXAMPLE_PAGES = 2  # per example file
MAX_EXAMPLE_IMAGES = 6  # per subject, keeps the prompt affordable


@dataclass
class Example:
    name: str
    images: list[Path]
    tex: str | None = None
    description: str | None = None


@dataclass
class Subject:
    key: str
    name: str
    description: str
    style_rules: str
    role_templates: dict[str, str]
    default_code_language: str | None
    preamble: str
    layout: str
    examples: list[Example] = field(default_factory=list)


def _example_images(path: Path, cache_dir: Path) -> list[Path]:
    if path.suffix.lower() in EXAMPLE_IMAGE_SUFFIXES:
        return [path]
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc):
            if i >= MAX_EXAMPLE_PAGES:
                break
            target = cache_dir / f"{path.stem}_{i}.png"
            if not target.exists() or target.stat().st_mtime < path.stat().st_mtime:
                page.get_pixmap(dpi=110, alpha=False).save(target)
            out.append(target)
    return out


def _load_examples(key: str, folder: Path) -> list[Example]:
    if not folder.is_dir():
        return []
    examples: list[Example] = []
    total = 0
    sources = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXAMPLE_IMAGE_SUFFIXES | {".pdf"})
    for src in sources:
        tex_path = src.with_suffix(".tex")
        desc_path = next((src.with_suffix(s) for s in (".md", ".txt") if src.with_suffix(s).exists()), None)
        if not tex_path.exists() and desc_path is None:
            continue  # an example needs a target: the ideal .tex or a description
        images = _example_images(src, config.DATA_DIR / "examples_cache" / key)
        images = images[: max(0, MAX_EXAMPLE_IMAGES - total)]
        if not images:
            break
        total += len(images)
        examples.append(
            Example(
                name=src.stem,
                images=images,
                tex=tex_path.read_text() if tex_path.exists() else None,
                description=desc_path.read_text() if desc_path else None,
            )
        )
    return examples


def load_subject(key: str) -> Subject:
    folder = config.SUBJECTS_DIR / key
    meta_path = folder / "subject.yaml"
    if not meta_path.exists():
        raise KeyError(f"Unknown subject '{key}'. Available: {', '.join(list_subjects())}")
    meta = yaml.safe_load(meta_path.read_text()) or {}
    base = config.SUBJECTS_DIR / "_base"
    preamble = (base / "preamble.tex").read_text()
    if (folder / "preamble.tex").exists():
        preamble += "\n" + (folder / "preamble.tex").read_text()
    layout_path = folder / "layout.tex" if (folder / "layout.tex").exists() else base / "layout.tex"
    return Subject(
        key=key,
        name=meta.get("name", key),
        description=meta.get("description", ""),
        style_rules=meta.get("style_rules", "") or "",
        role_templates=meta.get("role_templates") or {},
        default_code_language=meta.get("default_code_language"),
        preamble=preamble,
        layout=layout_path.read_text(),
        examples=_load_examples(key, folder / "examples"),
    )


def list_subjects() -> list[str]:
    return sorted(
        p.name for p in config.SUBJECTS_DIR.iterdir() if p.is_dir() and not p.name.startswith("_") and (p / "subject.yaml").exists()
    )
