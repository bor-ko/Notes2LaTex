"""Split a page into line-level blocks.

Each block is roughly one written line (a fraction or a superscript stays in
the same line). Line-level crops paired with their LaTeX are exactly what the
personal recognition model trains on later; grouping lines into paragraphs,
derivations or code listings happens in the transcription step.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class Box:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def width(self) -> int:
        return self.x1 - self.x0


def _runs(active: np.ndarray) -> list[tuple[int, int]]:
    """Start/end (exclusive) of consecutive True runs in a 1-D bool array."""
    padded = np.concatenate([[False], active, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def _merge_close(runs: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] <= max_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _attach_thin_bands(runs: list[tuple[int, int]], typical: float) -> list[tuple[int, int]]:
    """Join thin bands (fraction bars, underlines) to the lines they belong to.

    A fraction bar sits close to both its numerator and denominator, so it
    joins both. An underline sits close to the line above only.
    """
    out = [list(r) for r in runs]
    i = 0
    while i < len(out):
        s, e = out[i]
        if e - s >= typical * 0.45:
            i += 1
            continue
        gap_above = s - out[i - 1][1] if i > 0 else None
        gap_below = out[i + 1][0] - e if i + 1 < len(out) else None
        # Close below, or about as close below as above (a fraction bar).
        fraction_like = gap_above is not None and gap_below is not None and gap_below <= max(gap_above * 1.5, 8)
        if gap_below is not None and (gap_below <= typical * 0.6 or (fraction_like and gap_below <= typical * 0.9)):
            out[i][1] = out[i + 1][1]
            del out[i + 1]
        if i > 0 and s - out[i - 1][1] <= typical * 0.9:
            out[i - 1][1] = out[i][1]
            del out[i]
            i -= 1
        i += 1
    return [(s, e) for s, e in out]


def _peaks(sm: np.ndarray, radius: int, floor: float) -> list[int]:
    """Local maxima of a row profile, at least `radius` apart and above `floor`."""
    peaks: list[int] = []
    for i in np.flatnonzero(sm > floor):
        lo, hi = max(0, i - radius), min(len(sm), i + radius + 1)
        if sm[i] >= sm[lo:hi].max() and (not peaks or i - peaks[-1] > radius):
            peaks.append(int(i))
    return peaks


def _line_bands(ink: np.ndarray) -> list[tuple[int, int]]:
    """Vertical extents of written lines.

    Dense notes rarely leave an empty row between lines (descenders,
    brackets and margin notes bridge them), so lines are found as peaks of
    the row ink profile and cut at the deepest dip between neighbouring peaks.
    """
    h = ink.shape[0]
    prof = ink.sum(axis=1).astype(float)
    if not prof.any():
        return []
    k = max(3, h // 300) | 1
    sm = np.convolve(prof, np.ones(k) / k, mode="same")
    radius = max(6, h // 160)
    big = float(np.percentile(sm[sm > 0], 95))
    peaks = _peaks(sm, radius, floor=0.1 * big)

    cuts: list[int] = []
    for p1, p2 in zip(peaks, peaks[1:]):
        low = sm[p1 : p2 + 1] < 0.35 * min(sm[p1], sm[p2])
        if not low.any():
            continue  # two humps of the same line
        s, e = max(_runs(low), key=lambda r: r[1] - r[0])  # widest dip
        cuts.append(p1 + (s + e) // 2)

    bands: list[tuple[int, int]] = []
    bounds = [0, *cuts, h]
    for a, b in zip(bounds, bounds[1:]):
        # Inside one cut region, separate pieces with a long empty gap
        # (a stray mark far above a line).
        for s, e in _merge_close(_runs(prof[a:b] > 0), max_gap=2 * radius):
            if e - s >= 3:
                bands.append((a + s, a + e))
    return bands


def _attach_small_bands(runs: list[tuple[int, int]], ink: np.ndarray, typical: float) -> list[tuple[int, int]]:
    """Bands with little ink and height (a raised exponent, a dot, a short
    margin mark) join the nearest line if it is close."""
    if len(runs) < 2:
        return runs
    amounts = [int(ink[s:e].sum()) for s, e in runs]
    usual = float(np.median(amounts))
    out = [list(r) for r in runs]
    i = 0
    while i < len(out):
        s, e = out[i]
        amount = int(ink[s:e].sum())
        if amount >= 0.25 * usual or e - s >= 0.7 * typical:
            i += 1
            continue
        gap_above = s - out[i - 1][1] if i > 0 else None
        gap_below = out[i + 1][0] - e if i + 1 < len(out) else None
        options = [(g, j) for g, j in ((gap_above, i - 1), (gap_below, i + 1)) if g is not None and g <= 0.5 * typical]
        if not options:
            i += 1
            continue
        _, j = min(options)
        lo, hi = min(i, j), max(i, j)
        out[lo] = [min(out[lo][0], out[hi][0]), max(out[lo][1], out[hi][1])]
        del out[hi]
        i = lo
    return [(s, e) for s, e in out]


def segment(mask: np.ndarray, pad: int = 8) -> list[Box]:
    """Return line boxes in reading order (top-to-bottom, left-to-right)."""
    h, w = mask.shape
    ink = mask > 0
    runs = _line_bands(ink)
    if not runs:
        return []

    heights = np.array([e - s for s, e in runs])
    tall = heights[heights >= np.median(heights) * 0.5]
    typical = float(np.median(tall if len(tall) else heights))
    runs = _attach_thin_bands(runs, typical)
    runs = _attach_small_bands(runs, ink, typical)

    boxes: list[Box] = []
    big_gap = max(40, int(w * 0.12))
    for y0, y1 in runs:
        band = ink[y0:y1]
        col_ink = band.sum(axis=0) > 0
        cols = _merge_close(_runs(col_ink), max_gap=big_gap)
        for x0, x1 in cols:
            seg = band[:, x0:x1]
            if seg.sum() < 30:  # a stray dot
                continue
            rows = np.flatnonzero(seg.any(axis=1))
            ty0, ty1 = y0 + int(rows[0]), y0 + int(rows[-1]) + 1
            boxes.append(
                Box(
                    max(0, x0 - pad),
                    max(0, ty0 - pad),
                    min(w, x1 + pad),
                    min(h, ty1 + pad),
                )
            )
    boxes.sort(key=lambda b: (b.y0 // max(1, int(typical)), b.x0))
    return boxes


def crop(image: Image.Image, box: Box) -> Image.Image:
    return image.crop((box.x0, box.y0, box.x1, box.y1))


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_overlay(image: Image.Image, boxes: list[Box], ids: list[int] | None = None) -> Image.Image:
    """Page image with thin numbered boxes, used to ask Claude for per-block LaTeX."""
    ids = ids or list(range(1, len(boxes) + 1))
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    size = max(14, image.height // 90)
    font = _font(size)
    for bid, b in zip(ids, boxes):
        draw.rectangle((b.x0, b.y0, b.x1, b.y1), outline=(220, 30, 30), width=2)
        label = str(bid)
        tw = draw.textlength(label, font=font)
        lx = b.x0 - tw - 8 if b.x0 - tw - 8 > 0 else b.x0 + 2
        ly = b.y0
        draw.rectangle((lx - 2, ly, lx + tw + 2, ly + size + 2), fill=(220, 30, 30))
        draw.text((lx, ly), label, fill=(255, 255, 255), font=font)
    return out
