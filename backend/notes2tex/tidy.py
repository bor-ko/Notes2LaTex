"""Deterministic clean-up of model output, applied to every engine's drafts.

Small local models follow formatting rules less reliably than Claude, so the
rules that can be checked mechanically are enforced here instead of hoping
the prompt is obeyed.
"""

from __future__ import annotations

import re
from dataclasses import replace

from .assemble import BlockIn

# Handwritten labels per role (English and Dutch), e.g. "Def:", "Thm 3.", "Bewijs:".
_LABELS = {
    "definition": r"def(?:inition|initie|n)?",
    "theorem": r"thm|theorem|stelling",
    "lemma": r"lemma",
    "proposition": r"prop(?:osition|ositie)?",
    "corollary": r"cor(?:ollary|ollarium)?|gevolg",
    "example": r"ex(?:ample)?|e\.g|voorbeeld|vb",
    "remark": r"rem(?:ark)?|opm(?:erking)?",
    "note": r"note|nb|n\.b",
    "proof": r"proof|pf|bewijs",
}
_LABEL_RE = {
    role: re.compile(
        rf"^\s*(?:\\textbf\{{)?(?:{words})\b\.?\s*(?:\d+(?:\.\d+)*)?\s*(?:[(\[](?P<name>[^)\]]{{1,80}})[)\]])?\s*\}}?\s*[:.]?\s*",
        re.IGNORECASE,
    )
    for role, words in _LABELS.items()
}

TEXT_ROLES = {"prose", "list", "heading", "subheading", *_LABELS}
_MATH_SIGNAL = re.compile(r"[\^_=<>]|\\(?:frac|sqrt|int|sum|prod|lim|infty|partial|cdot|times|to|mapsto|in|subset|alpha|beta|gamma|delta|epsilon|lambda|mu|pi|sigma|theta|phi|omega|dd)\b")
_WORD = re.compile(r"(?<![\\a-zA-Z])[a-zA-Z]{4,}")
_DISPLAY_WRAPPERS = [(r"\[", r"\]"), ("$$", "$$"), ("$", "$"), (r"\(", r"\)")]


def strip_label(role: str, latex: str) -> str:
    rx = _LABEL_RE.get(role)
    if not rx:
        return latex
    m = rx.match(latex)
    if not m or m.end() == 0:
        return latex
    rest = latex[m.end():]
    name = (m.group("name") or "").strip()
    # "Example" on its own would also match plain words like "Note that ...";
    # only strip when a separator or name follows, or the rest starts a sentence.
    sep = latex[: m.end()].rstrip().endswith((":", ".", ")", "]"))
    if not (sep or name):
        return latex
    return f"[{name}] {rest}" if name else rest


def _looks_like_pure_math(text: str) -> bool:
    return bool(_MATH_SIGNAL.search(text)) and not _WORD.search(re.sub(r"\\[a-zA-Z]+", "", text))


def wrap_bare_math(latex: str) -> str:
    """A text line that is entirely mathematics but has no $...$ gets one."""
    if "$" in latex or r"\(" in latex or r"\[" in latex:
        return latex
    prefix = ""
    body = latex
    m = re.match(r"^(\s*\\item\s*|\s*\[[^\]]*\]\s*)", body)
    if m:
        prefix, body = m.group(0), body[m.end():]
    if body.strip() and _looks_like_pure_math(body):
        return f"{prefix}${body.strip()}$"
    return latex


def strip_display_wrappers(latex: str) -> str:
    s = latex.strip()
    for a, b in _DISPLAY_WRAPPERS:
        if s.startswith(a) and s.endswith(b) and len(s) > len(a) + len(b):
            inner = s[len(a) : -len(b)]
            if a in ("$", "$$") and "$" in inner:
                continue
            return inner.strip()
    return latex


def tidy_block(b: BlockIn) -> BlockIn:
    latex = b.latex
    if b.role in _LABEL_RE and not b.continues_previous:
        latex = strip_label(b.role, latex)
    if b.role in TEXT_ROLES:
        latex = wrap_bare_math(latex)
    elif b.role in ("equation", "derivation"):
        latex = strip_display_wrappers(latex)
    return replace(b, latex=latex) if latex != b.latex else b


def tidy(blocks: list[BlockIn]) -> list[BlockIn]:
    return [tidy_block(b) for b in blocks]
