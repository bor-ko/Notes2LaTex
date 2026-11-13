"""Free, local transcription with a vision model served by Ollama.

Small local models read single lines far more reliably than a whole page, so
by default the page is sent as line crops, a few consecutive lines per request
(`LOCAL_BATCH`, default 5) stacked into one numbered strip image, together
with the LaTeX of the lines just above for context. Vision models scale every
small image up to a minimum size (~1000 tokens), so a strip of five lines
costs about what one line alone would; seeing neighbouring lines together also
helps the model read words. A batch whose answer has the wrong number of
entries is retried one line at a time.
`mode="page"` sends the numbered page in one request like the Claude engine.
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
LINE_CTX = 4096  # plenty for a batch of line crops; page mode keeps 16k


class LineOut(BaseModel):
    role: Role
    latex: str
    continues_previous: bool
    code_language: Optional[str] = None


class LinesOut(BaseModel):
    lines: list[LineOut]


# Kept short on purpose: it is re-read for every request.
LINE_SYSTEM = """\
You transcribe handwritten lines from a student's notes into LaTeX. The image shows one line, or several lines numbered in red on the left and separated by grey rules, in reading order.

For each line return:
- role: heading | subheading | definition | theorem | lemma | proposition | corollary | proof | example | remark | note | equation (one displayed formula) | derivation (one line of multi-line working) | prose | list | code | pseudocode | diagram
- latex: the line as LaTeX with no environment around it. Text roles: maths inside $...$. equation/derivation: maths only, no $; derivation: & before the relation sign. diagram: "".
- continues_previous: true if the line continues the previous line's element (same paragraph, list, derivation, proof); then use the same role.

Rules: copy exactly what is written; never correct or add maths. Drop labels such as "Def:", "Thm.", "Ex:", "Proof:"; a named statement "Def (Homotopy):" starts with [Homotopy]. Wrap unreadable parts in \\unsure{{...}}.
Subject: {name}.{language}
Style:
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
        batch: int | None = None,
    ):
        self.model = model or config.LOCAL_MODEL
        self.batch = max(1, batch or config.LOCAL_BATCH)
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

    def _chat(
        self, system: str, user: str, images: list[str], schema: dict, num_predict: int = 1024, num_ctx: int = 16384
    ) -> tuple[str, dict]:
        message: dict = {"role": "user", "content": user}
        if images:
            message["images"] = images
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, message],
            "format": schema,
            "stream": False,
            "keep_alive": "15m",
            # A smaller context window means a smaller memory reservation,
            # which matters on a Mac that is already swapping.
            "options": {"temperature": 0, "num_ctx": num_ctx, "num_predict": num_predict},
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

    def _ask(
        self, model_cls: type[BaseModel], system: str, user: str, images: list[str], num_predict: int = 1024, num_ctx: int = 16384
    ):
        """Call with a JSON schema; one retry if the output does not validate."""
        last_err: Exception | None = None
        total = {"prompt_tokens": 0, "output_tokens": 0, "seconds": 0.0}
        for _ in range(2):
            text, stats = self._chat(system, user, images, model_cls.model_json_schema(), num_predict, num_ctx)
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
        ids = page.block_ids
        for start in range(0, len(ids), self.batch):
            idx = list(range(start, min(start + self.batch, len(ids))))
            outs = self._read_lines(page, idx, blocks, system, usage, warnings)
            for i, out in zip(idx, outs):
                if out is None:
                    blocks.append(BlockIn(id=ids[i], role="prose", latex=""))
                    continue
                continues = out.continues_previous and bool(blocks) and blocks[-1].role == out.role
                blocks.append(BlockIn(ids[i], out.role, out.latex, continues, out.code_language))
        usage["wall_seconds"] = round(time.monotonic() - started, 1)
        return PageTranscription(blocks=blocks, model=f"ollama:{self.model}", usage=usage, warnings=warnings)

    def _read_lines(
        self, page: PageInput, idx: list[int], done: list[BlockIn], system: str, usage: dict, warnings: list[str]
    ) -> list[LineOut | None]:
        """Transcribe consecutive lines in one request; fall back to one
        request per line if the answer does not have one entry per image."""
        context = "\n".join(f"[{b.role}] {b.latex}" for b in done[-CONTEXT_LINES:] if b.latex) or "(start of the page)"
        n = len(idx)
        if n > 1:
            user = f"Previous lines:\n{context}\n\nThe image shows the next {n} lines, numbered 1-{n}. Return exactly {n} entries in \"lines\", in that order."
            try:
                strip = _b64(page.strip(idx), 1600)
                out, stats = self._ask(LinesOut, system, user, [strip], num_predict=200 * n, num_ctx=LINE_CTX)
                self._add(usage, stats)
                if len(out.lines) == n:
                    return list(out.lines)
                warnings.append(f"lines {idx[0] + 1}-{idx[-1] + 1}: got {len(out.lines)} answers for {n} lines, retried one by one")
            except TranscriptionError as e:
                if "not running" in str(e) or "Could not reach" in str(e):
                    raise
                warnings.append(f"lines {idx[0] + 1}-{idx[-1] + 1}: {e}; retried one by one")
        results: list[LineOut | None] = []
        for i in idx:
            img = _b64(page.line(i), 1600)
            ctx = context if not results else "\n".join(f"[{r.role}] {r.latex}" for r in results[-CONTEXT_LINES:] if r)
            user = f"Previous lines:\n{ctx or '(start of the page)'}\n\nThis image is the next line. Return exactly 1 entry in \"lines\"."
            try:
                out, stats = self._ask(LinesOut, system, user, [img], num_predict=300, num_ctx=LINE_CTX)
                self._add(usage, stats)
                results.append(out.lines[0] if out.lines else None)
                if not out.lines:
                    warnings.append(f"line {i + 1}: empty answer")
            except TranscriptionError as e:
                if "not running" in str(e) or "Could not reach" in str(e):
                    raise
                warnings.append(f"line {i + 1}: {e}")
                results.append(None)
        return results

    @staticmethod
    def _add(usage: dict, stats: dict) -> None:
        for k in ("prompt_tokens", "output_tokens", "seconds"):
            usage[k] += stats[k]
        usage["requests"] += 1

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
