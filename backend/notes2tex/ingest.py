"""Turn tablet exports (PDF or images) into clean page images plus an ink mask.

Tablet apps export pen strokes on top of a paper template (blank, lined or
grid). The template lines would confuse segmentation and recognition, so we
remove long, thin, straight lines and keep everything else, including coloured
ink.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pymupdf

from . import config

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


@dataclass
class Page:
    index: int
    image_path: Path  # cleaned RGB page
    mask_path: Path  # binary ink mask (255 = ink)
    width: int
    height: int


def rasterize(source: Path, out_dir: Path, dpi: int = config.RENDER_DPI) -> list[Path]:
    """Render every page of a PDF (or copy a single image) to PNG files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower()
    paths: list[Path] = []
    if suffix == ".pdf":
        with pymupdf.open(source) as doc:
            for i, page in enumerate(doc):
                pix = page.get_pixmap(dpi=dpi, alpha=False)
                path = out_dir / f"raw_{i:03d}.png"
                pix.save(path)
                paths.append(path)
    elif suffix in IMAGE_SUFFIXES:
        img = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image {source}")
        path = out_dir / "raw_000.png"
        cv2.imwrite(str(path), img)
        paths.append(path)
    else:
        raise ValueError(f"Unsupported file type: {source.suffix}")
    return paths


def _long_thin(mask: np.ndarray, length: int, horizontal: bool) -> np.ndarray:
    """Pixels of `mask` on long, thin straight lines in one direction."""
    rect = cv2.getStructuringElement
    along = (15, 1) if horizontal else (1, 15)
    run = (length, 1) if horizontal else (1, length)
    across = (1, 7) if horizontal else (7, 1)
    # Bridge small gaps (compression noise, faint spots) along the line, then
    # keep only runs that cross most of the page.
    bridged = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, rect(cv2.MORPH_RECT, along))
    lines = cv2.morphologyEx(bridged, cv2.MORPH_OPEN, rect(cv2.MORPH_RECT, run))
    # A long *thick* stroke (a hand-drawn rule) is kept: subtract only the thin parts.
    lines = cv2.subtract(lines, cv2.morphologyEx(lines, cv2.MORPH_OPEN, rect(cv2.MORPH_RECT, across)))
    # Grow by a pixel across the line to catch its anti-aliased edge.
    lines = cv2.dilate(lines, rect(cv2.MORPH_RECT, (1, 3) if horizontal else (3, 1)))
    return cv2.bitwise_and(lines, mask)


def template_lines(mask: np.ndarray) -> np.ndarray:
    """Lined / grid paper: lines running across most of the page."""
    h, w = mask.shape
    return cv2.bitwise_or(_long_thin(mask, int(w * 0.6), True), _long_thin(mask, int(h * 0.6), False))


INK_DISTANCE = 40  # colour difference from the surrounding paper that counts as ink
LINE_DISTANCE = 12  # faint enough for pale template lines
# Template lines are faint (difference ~30-45 on Noteful grids) while pen ink is
# strong (~70+). Pixels on a template line above this are pen and are kept, so
# a stroke that happens to lie exactly on a grid line survives.
LINE_MAX_DISTANCE = 60


def paper_background(bgr: np.ndarray) -> np.ndarray:
    """The paper colour around every pixel: a wide median filter, so thin pen
    strokes vanish but large areas (the paper, highlighter bands) remain."""
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    return cv2.resize(cv2.medianBlur(small, 13), (w, h), interpolation=cv2.INTER_LINEAR)


def is_dark_paper(bgr: np.ndarray) -> bool:
    return float(np.median(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))) < 110


def _distance(bgr: np.ndarray, bg: np.ndarray) -> np.ndarray:
    a = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.sqrt(((a - b) ** 2).sum(axis=2))


def ink_mask(bgr: np.ndarray) -> np.ndarray:
    """Binary mask of pen ink (255) vs. paper (0), with template lines removed.

    Ink is whatever stands out from the paper around it. That works for white
    and dark (night-mode) paper alike, for every pen colour, and treats
    highlighter bands as paper while keeping the writing on top of them.
    """
    dist = _distance(bgr, paper_background(bgr))
    # The outermost pixels of a rendered page are anti-aliased page edge, not ink.
    b = 3
    dist[:b], dist[-b:], dist[:, :b], dist[:, -b:] = 0, 0, 0, 0
    mask = ((dist > INK_DISTANCE).astype(np.uint8)) * 255

    # Template lines are found where they are continuous: in the faint mask.
    # In the ink mask a line that is just about as strong as the threshold
    # shows up only in patches, too short to look like a line.
    loose = ((dist > LINE_DISTANCE).astype(np.uint8)) * 255
    faint_lines = template_lines(loose)
    # How strong this page's template lines are (most of their pixels are
    # plain line, not crossings with ink); pen is clearly stronger.
    line_px = dist[faint_lines > 0]
    limit = max(LINE_MAX_DISTANCE, 1.4 * float(np.median(line_px))) if line_px.size else LINE_MAX_DISTANCE
    weak = ((dist < limit).astype(np.uint8)) * 255
    on_template = cv2.bitwise_and(cv2.bitwise_or(template_lines(mask), faint_lines), weak)
    mask = cv2.subtract(mask, on_template)

    # Tiny specks sitting on the paper's lines (e.g. where grid lines cross)
    # are template, not writing.
    paper_lines = cv2.dilate(faint_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    area = stats[:, cv2.CC_STAT_AREA]
    on_lines = np.bincount(labels[paper_lines > 0], minlength=n)
    keep = (area >= 4) & ~((area < 40) & (on_lines >= 0.6 * area))  # specks / grid crossings
    keep[0] = False
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def clean_page(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """White page with only the ink kept, in its colours.

    On dark paper the ink is light, so its lightness is flipped (hue kept):
    white pen becomes black, orange stays orange but darker. Models, crops
    and figures then always see dark ink on white paper.
    """
    if is_dark_paper(bgr):
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = 255 - lab[:, :, 0]
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # Grow the mask slightly so anti-aliased stroke edges survive.
    grown = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    out = np.full_like(bgr, 255)
    out[grown > 0] = bgr[grown > 0]
    return out


def ingest(source: Path, out_dir: Path) -> list[Page]:
    """Rasterize a tablet export and write clean pages + ink masks to out_dir."""
    raw_paths = rasterize(source, out_dir)
    pages: list[Page] = []
    for i, raw in enumerate(raw_paths):
        bgr = cv2.imread(str(raw), cv2.IMREAD_COLOR)
        mask = ink_mask(bgr)
        clean = clean_page(bgr, mask)
        image_path = out_dir / f"page_{i:03d}.png"
        mask_path = out_dir / f"mask_{i:03d}.png"
        cv2.imwrite(str(image_path), clean)
        cv2.imwrite(str(mask_path), mask)
        raw.unlink()
        h, w = mask.shape
        pages.append(Page(i, image_path, mask_path, w, h))
    return pages
