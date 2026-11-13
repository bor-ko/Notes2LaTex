"""End-to-end operations shared by the CLI and the web API."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import config, dataset
from .assemble import Assembled, BlockIn, Group, assemble_document, escape_text, group_blocks
from .compile import compile_tex
from .ingest import ingest
from .segment import Box, crop, segment
from .subjects import load_subject
from .tidy import tidy
from .transcribe.base import PageInput, PageTranscription, Transcriber

# USD per million tokens, for the cost line in the CLI.
PRICES = {"claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_read": 0.20, "cache_write": 5.0}}


def estimate_cost(model: str, usage: dict) -> float | None:
    p = PRICES.get(model)
    if not p or not usage:
        return None
    return (
        usage.get("input_tokens", 0) * p["input"]
        + usage.get("output_tokens", 0) * p["output"]
        + usage.get("cache_read_input_tokens", 0) * p["cache_read"]
        + usage.get("cache_creation_input_tokens", 0) * p["cache_write"]
    ) / 1e6


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "notes"


def doc_dir(doc_id: int) -> Path:
    return config.DATA_DIR / "docs" / str(doc_id)


def output_dir(doc: dict) -> Path:
    return config.OUTPUT_DIR / doc["subject"] / f"{slugify(doc['name'])}-{doc['id']}"


# ---- import & segmentation -------------------------------------------------


def import_document(source: Path, subject: str, name: str | None = None) -> int:
    load_subject(subject)  # fail early on an unknown subject
    with dataset.connect() as conn:
        doc_id = dataset.create_document(conn, name or source.stem, subject, None)
    try:
        d = doc_dir(doc_id)
        d.mkdir(parents=True, exist_ok=True)
        stored = d / f"source{source.suffix.lower()}"
        shutil.copyfile(source, stored)
        pages = ingest(stored, d / "pages")
        with dataset.connect() as conn:
            conn.execute("UPDATE documents SET source_path = ? WHERE id = ?", (str(stored), doc_id))
            for p in pages:
                page_id = dataset.add_page(conn, doc_id, p.index, str(p.image_path), str(p.mask_path), p.width, p.height)
                _segment_page(conn, page_id)
    except Exception:
        delete_document(doc_id)
        raise
    return doc_id


def delete_document(doc_id: int) -> None:
    with dataset.connect() as conn:
        doc = dataset.get_document(conn, doc_id)
        dataset.delete_document(conn, doc_id)
    shutil.rmtree(doc_dir(doc_id), ignore_errors=True)
    if doc:
        shutil.rmtree(output_dir(doc), ignore_errors=True)


def _segment_page(conn, page_id: int) -> None:
    page = dataset.get_page(conn, page_id)
    mask = cv2.imread(page["mask_path"], cv2.IMREAD_GRAYSCALE)
    boxes = segment(mask)
    dataset.replace_blocks(conn, page_id, [(b.x0, b.y0, b.x1, b.y1) for b in boxes])
    write_crops(conn, page_id)


def write_crops(conn, page_id: int) -> None:
    page = dataset.get_page(conn, page_id)
    crops = Path(page["image_path"]).parent.parent / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    with Image.open(page["image_path"]) as img:
        for b in dataset.get_blocks(conn, page_id):
            path = crops / f"{b['id']}.png"
            crop(img, Box(b["x0"], b["y0"], b["x1"], b["y1"])).save(path)
            dataset.set_crop_path(conn, b["id"], str(path))


def _renumber(conn, page_id: int) -> None:
    for i, b in enumerate(dataset.get_blocks(conn, page_id)):
        conn.execute("UPDATE blocks SET ord = ? WHERE id = ?", (i, b["id"]))


def _join(a: str | None, b: str | None) -> str | None:
    parts = [s for s in (a, b) if s]
    return " ".join(parts) if parts else (a if a is not None else b)


def merge_with_next(block_id: int) -> None:
    """Merge a block with the one after it (segmentation split a line in two)."""
    with dataset.connect() as conn:
        a = dataset.get_block(conn, block_id)
        b = conn.execute(
            "SELECT * FROM blocks WHERE page_id = ? AND ord > ? ORDER BY ord LIMIT 1", (a["page_id"], a["ord"])
        ).fetchone()
        if b is None:
            raise ValueError("No block after this one")
        conn.execute(
            """UPDATE blocks SET x0 = ?, y0 = ?, x1 = ?, y1 = ?, latex = ?, draft_latex = ? WHERE id = ?""",
            (
                min(a["x0"], b["x0"]),
                min(a["y0"], b["y0"]),
                max(a["x1"], b["x1"]),
                max(a["y1"], b["y1"]),
                _join(a["latex"], b["latex"]) or "",
                _join(a["draft_latex"], b["draft_latex"]),
                block_id,
            ),
        )
        conn.execute("DELETE FROM blocks WHERE id = ?", (b["id"],))
        _renumber(conn, a["page_id"])
        write_crops(conn, a["page_id"])


def split_block(block_id: int, at: float = 0.5) -> None:
    """Split a block horizontally (segmentation merged two lines). `at` is the
    approximate split position as a fraction of the block height; the cut is
    moved to the emptiest row nearby."""
    with dataset.connect() as conn:
        b = dataset.get_block(conn, block_id)
        page = dataset.get_page(conn, b["page_id"])
        mask = cv2.imread(page["mask_path"], cv2.IMREAD_GRAYSCALE)[b["y0"] : b["y1"], b["x0"] : b["x1"]]
        h = b["y1"] - b["y0"]
        target = int(h * min(0.9, max(0.1, at)))
        lo, hi = max(1, target - h // 6), min(h - 1, target + h // 6)
        rows = (mask[lo:hi] > 0).sum(axis=1)
        cut = b["y0"] + lo + int(np.argmin(rows)) if len(rows) else b["y0"] + target
        conn.execute("UPDATE blocks SET y1 = ? WHERE id = ?", (cut, block_id))
        conn.execute("UPDATE blocks SET ord = ord + 1 WHERE page_id = ? AND ord > ?", (b["page_id"], b["ord"]))
        conn.execute(
            """INSERT INTO blocks (page_id, ord, x0, y0, x1, y1, role, latex, draft_role, draft_latex, source_model)
               VALUES (?, ?, ?, ?, ?, ?, ?, '', ?, '', ?)""",
            (b["page_id"], b["ord"] + 1, b["x0"], cut, b["x1"], b["y1"], b["role"], b["draft_role"], b["source_model"]),
        )
        write_crops(conn, b["page_id"])


# ---- transcription ---------------------------------------------------------


def page_input(page: dict, blocks: list[dict]) -> PageInput:
    with Image.open(page["image_path"]) as img:
        img.load()
        return PageInput(img.convert("RGB"), [Box(b["x0"], b["y0"], b["x1"], b["y1"]) for b in blocks])


def transcribe_page(page_id: int, transcriber: Transcriber) -> PageTranscription:
    with dataset.connect() as conn:
        page = dataset.get_page(conn, page_id)
        doc = dataset.get_document(conn, page["document_id"])
        blocks = dataset.get_blocks(conn, page_id)
        dataset.set_page_status(conn, page_id, "transcribing")
    subject = load_subject(doc["subject"])
    try:
        if blocks:
            result = transcriber.transcribe_page(page_input(page, blocks), subject)
        else:
            result = PageTranscription(blocks=[], model=transcriber.name)
    except Exception as e:
        with dataset.connect() as conn:
            dataset.set_page_status(conn, page_id, "error", str(e))
        raise
    result.blocks = tidy(result.blocks)
    with dataset.connect() as conn:
        dataset.save_transcription(conn, page_id, result.blocks, result.model, result.usage)
    return result


def transcribe_document(doc_id: int, transcriber: Transcriber, only_new: bool = True, on_page=None) -> list[PageTranscription]:
    with dataset.connect() as conn:
        pages = dataset.list_pages(conn, doc_id)
    results = []
    for page in pages:
        if only_new and page["status"] in ("transcribed", "reviewed"):
            continue
        r = transcribe_page(page["id"], transcriber)
        results.append(r)
        if on_page:
            on_page(page, r)
    return results


# ---- build -----------------------------------------------------------------


@dataclass
class BuildError:
    message: str
    line: int | None = None
    page_id: int | None = None
    block_ids: list[int] = field(default_factory=list)


@dataclass
class BuildResult:
    ok: bool
    tex_path: Path
    pdf_path: Path | None
    errors: list[BuildError] = field(default_factory=list)
    fixes: int = 0
    log: str = ""


def _assemble(doc: dict, draft: bool, broken: set[int] | frozenset[int] = frozenset()) -> tuple[Assembled, list[dict], Path]:
    """`broken` blocks are rendered as raw text, so one bad line cannot stop
    the whole document from compiling."""
    subject = load_subject(doc["subject"])
    out = output_dir(doc)
    figs = out / "figures"
    page_groups: list[list[Group]] = []
    with dataset.connect() as conn:
        pages = dataset.list_pages(conn, doc["id"])
        for page in pages:
            rows = dataset.get_blocks(conn, page["id"])
            by_id = {r["id"]: r for r in rows}
            groups = group_blocks(
                [
                    BlockIn(r["id"], "prose", f"\\brokenblock{{{escape_text(r['latex'] or '')}}}")
                    if r["id"] in broken
                    else BlockIn(r["id"], r["role"], r["latex"] or "", bool(r["continues_previous"]), r["code_language"])
                    for r in rows
                ]
            )
            for gi, g in enumerate(groups):
                if g.role != "diagram":
                    continue
                bs = [by_id[i] for i in g.block_ids]
                box = Box(min(b["x0"] for b in bs), min(b["y0"] for b in bs), max(b["x1"] for b in bs), max(b["y1"] for b in bs))
                figs.mkdir(parents=True, exist_ok=True)
                name = f"p{page['page_index'] + 1}_{gi}.png"
                with Image.open(page["image_path"]) as img:
                    crop(img, box).save(figs / name)
                g.image = f"figures/{name}"
                g.image_width = min(0.9, max(0.3, box.width / page["width"] * 1.1))
            page_groups.append(groups)
    return assemble_document(page_groups, subject, doc["name"], draft=draft), pages, out


def build_document(
    doc_id: int, draft: bool = True, fixer: Transcriber | None = None, max_fixes: int = 5, max_fallbacks: int = 60
) -> BuildResult:
    """Assemble and compile. A block that breaks compilation is first sent to
    `fixer` (once per block, up to `max_fixes`); if it still breaks, it is
    shown as raw text in the PDF and reported, so the rest still builds."""
    with dataset.connect() as conn:
        doc = dataset.get_document(conn, doc_id)
    if doc is None:
        raise KeyError(f"No document {doc_id}")
    subject = load_subject(doc["subject"])
    fixes = 0
    tried: set[tuple[int, ...]] = set()
    broken: set[int] = set()
    shown_raw: list[BuildError] = []
    while True:
        assembled, pages, out = _assemble(doc, draft, broken)
        out.mkdir(parents=True, exist_ok=True)
        tex_path = out / f"{slugify(doc['name'])}.tex"
        tex_path.write_text(assembled.tex)
        result = compile_tex(tex_path)
        errors: list[BuildError] = []
        lines = assembled.tex.split("\n")
        for e in result.errors:
            span = assembled.blocks_at_line(e.line) if e.line else None
            errors.append(
                BuildError(
                    message=e.message,
                    line=e.line,
                    page_id=pages[span.page_index]["id"] if span else None,
                    block_ids=span.block_ids if span else [],
                )
            )
        if result.ok:
            break
        target = next((e for e in errors if e.block_ids), None)
        if target is None:
            break  # an error outside the notes (preamble, missing package)
        key = tuple(target.block_ids)
        if fixer is not None and fixes < max_fixes and key not in tried:
            tried.add(key)
            with dataset.connect() as conn:
                rows = [dataset.get_block(conn, i) for i in target.block_ids]
            old = [r["latex"] for r in rows]
            context = lines[target.line - 1] if target.line and target.line <= len(lines) else ""
            new = fixer.fix_latex(old, rows[0]["role"], f"{target.message} (at: {context.strip()})", subject)
            if new != old:
                with dataset.connect() as conn:
                    for r, latex in zip(rows, new):
                        dataset.apply_autofix(conn, r["id"], latex)
                fixes += 1
                continue
        if set(target.block_ids) <= broken or len(broken) >= max_fallbacks:
            break
        broken.update(target.block_ids)
        shown_raw.append(target)
    reported = shown_raw if result.ok else errors + shown_raw
    return BuildResult(result.ok, tex_path, result.pdf_path if result.ok else None, reported, fixes, result.log)
