"""Transcriber interface shared by every engine (Claude, local Ollama model,
offline placeholders, and later the personal model)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from PIL import Image, ImageDraw, ImageOps

from ..assemble import BlockIn
from ..segment import Box, _font, crop, draw_overlay
from ..subjects import Subject


class TranscriptionError(RuntimeError):
    pass


@dataclass
class PageInput:
    """A cleaned page image and its line boxes, in reading order."""

    image: Image.Image
    boxes: list[Box]

    @property
    def block_ids(self) -> list[int]:
        return list(range(1, len(self.boxes) + 1))

    def overlay(self) -> Image.Image:
        """Page with numbered boxes, for engines that read the whole page."""
        return draw_overlay(self.image, self.boxes, self.block_ids)

    def line(self, index: int, pad: int = 16) -> Image.Image:
        """One line crop with a white margin, for engines that read line by line."""
        return ImageOps.expand(crop(self.image, self.boxes[index]).convert("RGB"), border=pad, fill="white")

    def strip(self, indices: list[int], label_width: int = 60, gap: int = 14) -> Image.Image:
        """Consecutive line crops stacked into one image, numbered 1..n in red
        on the left and separated by grey rules.

        Vision models scale every small image up to a minimum size, so one
        strip of five lines costs about as much as a single line on its own.
        """
        crops = [self.line(i, pad=10) for i in indices]
        width = label_width + max(c.width for c in crops)
        height = sum(c.height for c in crops) + gap * (len(crops) - 1)
        out = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(out)
        font = _font(30)
        y = 0
        for n, c in enumerate(crops, 1):
            out.paste(c, (label_width, y))
            draw.text((8, y + c.height // 2 - 16), str(n), fill=(200, 30, 30), font=font)
            y += c.height
            if n < len(crops):
                draw.line((0, y + gap // 2, width, y + gap // 2), fill=(180, 180, 180), width=2)
                y += gap
        return out


@dataclass
class PageTranscription:
    blocks: list[BlockIn]
    model: str
    usage: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class Transcriber(Protocol):
    name: str

    def transcribe_page(self, page: PageInput, subject: Subject) -> PageTranscription: ...

    def fix_latex(self, lines: list[str], role: str, error: str, subject: Subject) -> list[str]: ...


class OfflineTranscriber:
    """Fills every block with a placeholder. Lets you use the whole pipeline
    and the review UI with no model at all."""

    name = "offline"

    def transcribe_page(self, page: PageInput, subject: Subject) -> PageTranscription:
        blocks = [BlockIn(id=i, role="prose", latex=f"\\unsure{{line {i}}}") for i in page.block_ids]
        return PageTranscription(blocks=blocks, model=self.name, warnings=["offline mode: placeholder text"])

    def fix_latex(self, lines: list[str], role: str, error: str, subject: Subject) -> list[str]:
        return lines
