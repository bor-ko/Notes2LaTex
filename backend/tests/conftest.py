"""Shared fixtures: a synthetic tablet export (lined paper, coloured ink)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

HAND_FONTS = [
    "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
    "/System/Library/Fonts/Supplemental/Chalkboard.ttc",
]

# (text, colour). The fraction line is drawn separately.
LINES = [
    ("Kinematics", (20, 30, 120)),
    ("Def: velocity is the rate of change of position", (20, 30, 120)),
    ("v = dx/dt", (20, 30, 120)),
    ("a = dv/dt = d^2x/dt^2", (20, 30, 120)),
    ("Example: x(t) = 3t^2 + 2t", (190, 20, 20)),
]
EXPECTED_LINES = len(LINES) + 1  # + the stacked fraction


def _font(size: int) -> ImageFont.ImageFont:
    for path in HAND_FONTS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def make_page(path: Path, lined: bool = True, paper: str | None = None, line_colour=(160, 190, 230)) -> Path:
    """paper: "lined", "grid" or "blank" (default: lined if `lined` else blank)."""
    paper = paper or ("lined" if lined else "blank")
    w, h = 1654, 2339  # A4 at 200 DPI
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    if paper in ("lined", "grid"):
        step = 90 if paper == "lined" else 60
        for y in range(180, h - 100, step):
            d.line((60, y, w - 60, y), fill=line_colour, width=2)
    if paper == "grid":
        for x in range(60, w - 60, 60):
            d.line((x, 120, x, h - 100), fill=line_colour, width=2)
    font = _font(56)
    y = 200
    for text, colour in LINES:
        d.text((150, y), text, fill=colour, font=font)
        y += 180
    # A stacked fraction: numerator, bar, denominator close together.
    d.text((400, y), "x + 1", fill=(20, 30, 120), font=font)
    d.line((380, y + 78, 620, y + 78), fill=(20, 30, 120), width=5)
    d.text((440, y + 88), "2y", fill=(20, 30, 120), font=font)
    img.save(path, "PDF", resolution=200)
    return path


@pytest.fixture
def tablet_pdf(tmp_path: Path) -> Path:
    return make_page(tmp_path / "notes.pdf")


@pytest.fixture
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the app at a throwaway data/output dir."""
    from notes2tex import config

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "data" / "notes.db")
    return tmp_path
