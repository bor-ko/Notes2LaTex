"""Claude vision transcriber.

One request per page: the page image with numbered line boxes goes in, and a
validated JSON list of {id, role, latex, continues_previous, code_language}
comes out. The subject's style rules and example pages form a stable prefix
that is prompt-cached, so only the page image is billed at the full rate.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Literal, Optional

import anthropic
from PIL import Image
from pydantic import BaseModel

from .. import config
from ..assemble import ROLES, BlockIn
from ..subjects import Subject
from .base import PageInput, PageTranscription, TranscriptionError

Role = Literal[tuple(ROLES)]  # type: ignore[valid-type]


class BlockOut(BaseModel):
    id: int
    role: Role
    latex: str
    continues_previous: bool
    code_language: Optional[str]


class PageOut(BaseModel):
    blocks: list[BlockOut]


class FixOut(BaseModel):
    lines: list[str]


SYSTEM_TEMPLATE = """\
You transcribe handwritten study notes into LaTeX.

The user message may start with reference examples from the author's own notes. It then shows one page of notes. Every written line on that page has a thin red box with a red number label. Return exactly one entry per numbered box, in number order.

For each box:
- id: the box number.
- role: what the line is part of.
  heading / subheading: a section title. definition, theorem, lemma, proposition, corollary, proof, example, remark, note: the content of that kind of statement. equation: a single displayed formula. derivation: one line of multi-line working. prose: ordinary text. list: a bullet or numbered item. code: a line of real program code. pseudocode: a line of an algorithm in pseudocode. diagram: part of a drawing, graph or figure.
- latex: the content of that one line as LaTeX, without the surrounding environment, which comes from the role. So no \\begin{{definition}}, no \\[ \\], no align*, no lstlisting.
  - prose, list and the statement roles: text with inline math in $...$. A list line that starts a new item begins with \\item.
  - When a statement is named ("Def (Homotopy):", "Thm [Cauchy]"), start its first line with the name in brackets: [Homotopy] ...
  - equation: math only, no $ signs.
  - derivation: math for this one line, with & before the relation sign (=, \\le, \\Rightarrow, ...). No trailing \\\\.
  - code: the code line exactly as written, with its leading indentation as spaces.
  - pseudocode: algpseudocode commands (\\State, \\If{{...}}, \\EndIf, \\For{{...}}, \\EndFor, \\While{{...}}, \\EndWhile, \\Return).
  - heading, subheading: the title text only.
  - diagram: an empty string.
- continues_previous: true when this line continues the same element as the previous box (the next line of the same paragraph, definition, derivation, list, code listing, algorithm or diagram). Give it the same role as the previous box.
- code_language: the programming language for code lines, otherwise null.

Accuracy rules:
- Transcribe what is written. Do not fix the author's mathematics, fill in skipped steps, or add anything.
- Drop labels that the role already conveys, such as "Def:", "Thm.", "Lemma", "Ex:", "Proof:" (keep a name as [Name], see above).
- Leave out crossed-out writing.
- Wrap any symbol or word you cannot read with confidence in \\unsure{{...}}. Never put &, \\\\ or an unbalanced brace inside \\unsure.
- If a box holds no real content (a stray mark), give role prose and an empty latex string.
- Arrows, margin notes and small annotations belong to the line they sit next to.

Subject: {name}. {description}{language}
Style rules for this subject:
{style_rules}
"""


def language_hint(subject: Subject) -> str:
    if not subject.language:
        return ""
    return (
        f"\nThe notes are written in {subject.language} (technical terms are sometimes in English). "
        f"Transcribe words in that language with its accents and diacritics; do not translate."
    )


def has_credentials() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return (Path.home() / ".config" / "anthropic").exists()


def _image_block(img: Image.Image, max_edge: int = config.MAX_IMAGE_EDGE) -> dict:
    img = img.convert("RGB")
    scale = max_edge / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": base64.standard_b64encode(buf.getvalue()).decode()},
    }


def build_request(overlay: Image.Image, block_ids: list[int], subject: Subject) -> dict:
    """Keyword arguments for messages.parse. Pure, so it is unit-testable."""
    system_text = SYSTEM_TEMPLATE.format(
        name=subject.name,
        description=subject.description,
        language=language_hint(subject),
        style_rules=subject.style_rules.strip() or "- (none)",
    )
    system = [{"type": "text", "text": system_text}]
    content: list[dict] = []
    if subject.examples:
        content.append(
            {
                "type": "text",
                "text": "Reference examples from the author's notes in this subject, each followed by the LaTeX the author wants for it. "
                "Match their notation, macros and structure. Do not transcribe these pages.",
            }
        )
        for ex in subject.examples:
            for path in ex.images:
                with Image.open(path) as im:
                    content.append(_image_block(im, max_edge=1400))
            if ex.tex:
                content.append({"type": "text", "text": f"Target LaTeX for example '{ex.name}':\n```latex\n{ex.tex.strip()}\n```"})
            if ex.description:
                content.append({"type": "text", "text": f"How the author wants pages like '{ex.name}' structured:\n{ex.description.strip()}"})
        # Everything up to here is identical for every page of this subject.
        content[-1]["cache_control"] = {"type": "ephemeral"}
    else:
        system[-1]["cache_control"] = {"type": "ephemeral"}

    content.append(_image_block(overlay))
    content.append(
        {
            "type": "text",
            "text": f"Transcribe this page. It has {len(block_ids)} numbered boxes: {', '.join(map(str, block_ids))}.",
        }
    )
    return {
        "model": config.MODEL,
        "max_tokens": 16000,
        "system": system,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": config.EFFORT},
        "betas": ["server-side-fallback-2026-07-01"],
        "fallbacks": "default",
        "messages": [{"role": "user", "content": content}],
        "output_format": PageOut,
    }


def _check(response) -> None:
    if response.stop_reason == "refusal":
        detail = getattr(response, "stop_details", None)
        raise TranscriptionError(f"Claude declined this page ({getattr(detail, 'category', None)}).")
    if response.stop_reason == "max_tokens":
        raise TranscriptionError("Response hit max_tokens before finishing; the page may be very dense.")
    if response.parsed_output is None:
        raise TranscriptionError("Claude returned no structured output.")


class ClaudeTranscriber:
    name = "claude"

    def __init__(self, client: anthropic.Anthropic | None = None):
        if client is None and not has_credentials():
            raise TranscriptionError(
                "No Anthropic credentials found. Put ANTHROPIC_API_KEY in notes2tex/.env, or use --offline."
            )
        self.client = client or anthropic.Anthropic()

    def _parse(self, **kwargs):
        try:
            return self.client.beta.messages.parse(**kwargs)
        except anthropic.AuthenticationError as e:
            raise TranscriptionError("Anthropic API key was rejected. Check ANTHROPIC_API_KEY in .env.") from e
        except anthropic.RateLimitError as e:
            raise TranscriptionError("Rate limited by the Anthropic API. Try again in a minute.") from e
        except anthropic.APIConnectionError as e:
            raise TranscriptionError("Could not reach the Anthropic API. Check your connection.") from e
        except anthropic.APIStatusError as e:
            raise TranscriptionError(f"Anthropic API error {e.status_code}: {e.message}") from e

    def transcribe_page(self, page: PageInput, subject: Subject) -> PageTranscription:
        block_ids = page.block_ids
        response = self._parse(**build_request(page.overlay(), block_ids, subject))
        _check(response)
        by_id = {b.id: b for b in response.parsed_output.blocks}
        warnings: list[str] = []
        blocks: list[BlockIn] = []
        for bid in block_ids:
            b = by_id.get(bid)
            if b is None:
                warnings.append(f"block {bid} missing from response")
                blocks.append(BlockIn(id=bid, role="prose", latex=""))
                continue
            blocks.append(
                BlockIn(
                    id=bid,
                    role=b.role,
                    latex=b.latex,
                    continues_previous=b.continues_previous,
                    code_language=b.code_language,
                )
            )
        extra = set(by_id) - set(block_ids)
        if extra:
            warnings.append(f"ignored unknown block ids {sorted(extra)}")
        u = response.usage
        usage = {
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        }
        return PageTranscription(blocks=blocks, model=response.model, usage=usage, warnings=warnings)

    def fix_latex(self, lines: list[str], role: str, error: str, subject: Subject) -> list[str]:
        prompt = (
            f"These {len(lines)} lines are the body of a '{role}' element in LaTeX notes (the environment itself is added "
            f"separately). Compiling them failed with:\n{error}\n\nFix only what breaks compilation. Keep the content, "
            f"notation and \\unsure{{...}} markers. Return exactly {len(lines)} lines.\n\n"
            + "\n".join(f"{i + 1}: {ln}" for i, ln in enumerate(lines))
        )
        response = self._parse(
            model=config.MODEL,
            max_tokens=8000,
            thinking={"type": "adaptive"},
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=f"You repair LaTeX compile errors in transcribed notes. Subject: {subject.name}.",
            messages=[{"role": "user", "content": prompt}],
            output_format=FixOut,
        )
        _check(response)
        fixed = response.parsed_output.lines
        return fixed if len(fixed) == len(lines) else lines
