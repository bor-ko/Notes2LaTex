"""Free, local transcription with a vision model served by Ollama.

Small local models read a single line far more reliably than a whole page, so
by default every line crop is sent on its own, together with the LaTeX of the
lines just above it for context (which also lets the model decide whether the
line continues the previous element). `mode="page"` sends the numbered page
in one request like the Claude engine does, for comparison.
"""

from __future__ import annotations

import base64
import io
import time
from typing import Literal, Optional

import httpx
from PIL import Image
from pydantic import BaseModel, ValidationError

from .. import config
from ..assemble import BlockIn
from ..subjects import Subject
from .base import PageInput, PageTranscription, TranscriptionError
from .claude import SYSTEM_TEMPLATE, FixOut, PageOut, Role, language_hint

CONTEXT_LINES = 3
EXAMPLE_TEX_CHARS = 2500


class LineOut(BaseModel):
    role: Role
    latex: str
    continues_previous: bool
    code_language: Optional[str]


LINE_SYSTEM = """\
You transcribe one handwritten line from a student's study notes into LaTeX.

Return JSON with:
- role: one of heading, subheading, definition, theorem, lemma, proposition, corollary, proof, example, remark, note, equation, derivation, prose, list, code, pseudocode, diagram.
  equation = a single displayed formula; derivation = one line of multi-line working; prose = ordinary text; diagram = a drawing.
- latex: the line as LaTeX WITHOUT any surrounding environment (no \\begin{{...}}, no \\[ \\]).
  Text roles: text with inline math in $...$. equation/derivation: math only, no $ signs; for derivation put & before the relation sign (=, \\le, \\Rightarrow). code: the code exactly. heading: the title text only. diagram: empty string.
- continues_previous: true if this line continues the element of the previous line (same paragraph, derivation, list, definition, proof). Then use the same role as the previous line.
- code_language: language for code lines, otherwise null.

Rules:
- Transcribe exactly what is written. Never fix, complete or add mathematics.
- Drop labels the role already shows ("Def:", "Thm.", "Lemma", "Ex:", "Proof:"). If the statement is named ("Def (Homotopy):"), start latex with the name in brackets: [Homotopy] ...
- Wrap anything you cannot read with confidence in \\unsure{{...}} (never put & or \\\\ inside it).

Subject: {name}. {description}{language}
Style rules:
{style_rules}
{examples}"""


def _b64(img: Image.Image, max_edge: int) -> str:
    img = img.convert("RGB")
    scale = max_edge / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def _example_notes(subject: Subject) -> str:
    """Short excerpts of the author's ideal LaTeX: cheap text that teaches
    notation and macros (example images would be too slow per line)."""
    parts, budget = [], EXAMPLE_TEX_CHARS
    for ex in subject.examples:
        text = (ex.tex or ex.description or "").strip()
        if not text or budget <= 0:
            continue
        if "\\begin{document}" in text:
            text = text.split("\\begin{document}", 1)[1]
        text = text[:budget]
        budget -= len(text)
        parts.append(text)
    if not parts:
        return ""
    return "\nThe author writes LaTeX like this (match notation and macros):\n" + "\n---\n".join(parts) + "\n"


class OllamaTranscriber:
    name = "ollama"

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        mode: Literal["line", "page"] | None = None,
        client: httpx.Client | None = None,
        check: bool = True,
    ):
        self.model = model or config.LOCAL_MODEL
        self.host = (host or config.OLLAMA_HOST).rstrip("/")
        self.mode = mode or config.LOCAL_MODE
        self.client = client or httpx.Client(timeout=httpx.Timeout(300.0, connect=5.0))
        if check:
            self.ensure_ready()

    # ---- availability ------------------------------------------------------

    def installed_models(self) -> list[str]:
        try:
            r = self.client.get(f"{self.host}/api/tags")
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise TranscriptionError("Ollama is not running. Open the Ollama app (or run `ollama serve`) and try again.") from e
        return [m["name"] for m in r.json().get("models", [])]

    def ensure_ready(self) -> None:
        names = self.installed_models()
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        if wanted not in names:
            raise TranscriptionError(f"The local model {self.model} is not downloaded yet. Run: ollama pull {self.model}")

    # ---- requests ----------------------------------------------------------

    def _chat(self, system: str, user: str, images: list[str], schema: dict, num_predict: int = 1024) -> tuple[str, dict]:
        message: dict = {"role": "user", "content": user}
        if images:
            message["images"] = images
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, message],
            "format": schema,
            "stream": False,
            "keep_alive": "15m",
            "options": {"temperature": 0, "num_ctx": 16384, "num_predict": num_predict},
        }
        try:
            r = self.client.post(f"{self.host}/api/chat", json=payload)
        except httpx.HTTPError as e:
            raise TranscriptionError(f"Could not reach Ollama: {e}") from e
        if r.status_code != 200:
            raise TranscriptionError(f"Ollama error {r.status_code}: {r.text[:300]}")
        data = r.json()
        stats = {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "output_tokens": data.get("eval_count", 0),
            "seconds": round(data.get("total_duration", 0) / 1e9, 2),
        }
        return data["message"]["content"], stats

    def _ask(self, model_cls: type[BaseModel], system: str, user: str, images: list[str], num_predict: int = 1024):
        """Call with a JSON schema; one retry if the output does not validate."""
        last_err: Exception | None = None
        total = {"prompt_tokens": 0, "output_tokens": 0, "seconds": 0.0}
        for _ in range(2):
            text, stats = self._chat(system, user, images, model_cls.model_json_schema(), num_predict)
            for k in total:
                total[k] += stats[k]
            try:
                return model_cls.model_validate_json(text), total
            except ValidationError as e:
                last_err = e
        raise TranscriptionError(f"Local model returned invalid output: {last_err}")

    # ---- transcription -----------------------------------------------------

    def _system(self, template: str, subject: Subject) -> str:
        return template.format(
            name=subject.name,
            description=subject.description,
            language=language_hint(subject),
            style_rules=subject.style_rules.strip() or "- (none)",
            examples=_example_notes(subject),
        )

    def transcribe_page(self, page: PageInput, subject: Subject) -> PageTranscription:
        if self.mode == "page":
            return self._transcribe_whole_page(page, subject)
        system = self._system(LINE_SYSTEM, subject)
        blocks: list[BlockIn] = []
        warnings: list[str] = []
        usage = {"prompt_tokens": 0, "output_tokens": 0, "seconds": 0.0, "requests": 0}
        started = time.monotonic()
        for i, bid in enumerate(page.block_ids):
            context = "\n".join(f"[{b.role}] {b.latex}" for b in blocks[-CONTEXT_LINES:] if b.latex) or "(this is the first line of the page)"
            user = f"Previous lines on this page:\n{context}\n\nTranscribe the handwritten line in the image."
            try:
                out, stats = self._ask(LineOut, system, user, [_b64(page.line(i), 1600)], num_predict=768)
            except TranscriptionError as e:
                if i == 0 and ("not running" in str(e) or "Could not reach" in str(e)):
                    raise
                warnings.append(f"line {bid}: {e}")
                blocks.append(BlockIn(id=bid, role="prose", latex=""))
                continue
            for k in ("prompt_tokens", "output_tokens", "seconds"):
                usage[k] += stats[k]
            usage["requests"] += 1
            continues = out.continues_previous and bool(blocks) and blocks[-1].role == out.role
            blocks.append(BlockIn(id=bid, role=out.role, latex=out.latex, continues_previous=continues, code_language=out.code_language))
        usage["wall_seconds"] = round(time.monotonic() - started, 1)
        return PageTranscription(blocks=blocks, model=f"ollama:{self.model}", usage=usage, warnings=warnings)

    def _transcribe_whole_page(self, page: PageInput, subject: Subject) -> PageTranscription:
        system = SYSTEM_TEMPLATE.format(
            name=subject.name,
            description=subject.description,
            language=language_hint(subject),
            style_rules=subject.style_rules.strip() or "- (none)",
        ) + _example_notes(subject)
        ids = page.block_ids
        user = f"Transcribe this page. It has {len(ids)} numbered boxes: {', '.join(map(str, ids))}."
        out, usage = self._ask(PageOut, system, user, [_b64(page.overlay(), config.MAX_IMAGE_EDGE)], num_predict=8192)
        by_id = {b.id: b for b in out.blocks}
        blocks, warnings = [], []
        for bid in ids:
            b = by_id.get(bid)
            if b is None:
                warnings.append(f"block {bid} missing from response")
                blocks.append(BlockIn(id=bid, role="prose", latex=""))
            else:
                blocks.append(BlockIn(bid, b.role, b.latex, b.continues_previous, b.code_language))
        return PageTranscription(blocks=blocks, model=f"ollama:{self.model}:page", usage=usage, warnings=warnings)

    def fix_latex(self, lines: list[str], role: str, error: str, subject: Subject) -> list[str]:
        user = (
            f"These {len(lines)} lines are the body of a '{role}' element in LaTeX notes (the environment is added separately). "
            f"Compiling them failed with:\n{error}\n\nFix only what breaks compilation; keep content, notation and \\unsure{{...}} "
            f"markers. Return exactly {len(lines)} lines.\n\n" + "\n".join(f"{i + 1}: {ln}" for i, ln in enumerate(lines))
        )
        try:
            out, _ = self._ask(FixOut, "You repair LaTeX compile errors.", user, [], num_predict=2048)
        except TranscriptionError:
            return lines
        return out.lines if len(out.lines) == len(lines) else lines
