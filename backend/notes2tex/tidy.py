"""Deterministic clean-up of model output, applied to every engine's drafts.

Small local models follow formatting rules less reliably than Claude, so the
rules that can be checked mechanically are enforced here instead of hoping
the prompt is obeyed.
"""

from __future__ import annotations

import re
from dataclasses import replace

from .assemble import BlockIn

# Handwritten labels per role (English, Dutch, Slovenian), e.g. "Def:", "Thm 3.", "Bewijs:".
_LABELS = {
    "definition": r"def(?:inition|initie|inicija|n)?",
    "theorem": r"thm|theorem|stelling|izrek",
    "lemma": r"lemm?a",
    "proposition": r"prop(?:osition|ositie)?|trditev",
    "corollary": r"cor(?:ollary|ollarium)?|gevolg|posledica",
    "example": r"ex(?:ample)?|e\.g|voorbeeld|vb|primer|pr",
    "remark": r"rem(?:ark)?|opm(?:erking)?|opomba",
    "note": r"note|nb|n\.b|opomba",
    "proof": r"proof|pf|bewijs|dokaz",
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
# A leading \item or a statement name "[Homotopy]" (words, not "[n]") is kept
# outside any $...$ that gets added.
_PREFIX = re.compile(r"^(\s*\\item(?:\[[^\]]*\])?\s*|\s*\[(?=[^\]]*[A-Za-z\u00C0-\u017F]{3})[^\]$\\=^_{}]{3,80}\]\s*)")
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
    m = _PREFIX.match(body)
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


# ---- repairs of broken model output ---------------------------------------

# A model that writes "\to" inside JSON without escaping the backslash gets
# back a TAB followed by "o"; "\forall" becomes form-feed + "orall", and so
# on. Control characters never belong in a LaTeX line, so turn them back.
_ESCAPES = {"\t": r"\t", "\x0c": r"\f", "\x08": r"\b", "\r": r"\r", "\n": r"\n"}
_BROKEN_ESCAPE = re.compile(r"([\t\x0c\x08\r\n])(?=[A-Za-z])")


def repair_escapes(latex: str) -> str:
    return _BROKEN_ESCAPE.sub(lambda m: _ESCAPES[m.group(1)], latex)


# UTF-8 bytes read as Latin-1: "mnoÅ¾ina" -> "množina", "Ä\x8d" -> "č".
_MOJIBAKE = re.compile(r"[\u00c2-\u00f4][\u0080-\u00bf]{1,3}")


def fix_mojibake(text: str) -> str:
    def fix(m: re.Match) -> str:
        try:
            return m.group(0).encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return m.group(0)

    return _MOJIBAKE.sub(fix, text)


# ---- math in text lines ----------------------------------------------------

_TEXT_COMMANDS = {r"\text", r"\textbf", r"\textit", r"\emph", r"\texttt", r"\underline", r"\item", r"\unsure", r"\textrm"}
_ONE_LETTER_WORDS = set("aiovzsku")  # English/Slovenian one-letter words, not variables
_MATH_CHARS = re.compile(r"[\^_=<>+|]")
_FUNC_APPLICATION = re.compile(r"^[A-Za-z]'*\(")


def _split_top_level(s: str) -> list[str]:
    """Split on spaces that are not inside braces, keeping the spaces."""
    parts, cur, depth = [], "", 0
    for ch in s:
        if ch == " " and depth == 0:
            if cur:
                parts.append(cur)
            parts.append(" ")
            cur = ""
            continue
        depth += {"{": 1, "}": -1}.get(ch, 0)
        cur += ch
    if cur:
        parts.append(cur)
    return parts


def _kind(token: str) -> str:
    """'math', 'text', 'operand' (a number, \\text{...}: math if inside a
    formula, harmless outside) or 'punct'."""
    commands = re.findall(r"\\[A-Za-z]+", token)
    if any(c not in _TEXT_COMMANDS for c in commands) or _MATH_CHARS.search(token) or _FUNC_APPLICATION.match(token):
        return "math"
    bare = re.sub(r"\\[A-Za-z]+\{[^{}]*\}", "", token).strip("()[],.;:!?{}")
    if len(bare) == 1 and bare.isalpha():
        return "text" if bare in _ONE_LETTER_WORDS else "math"
    if any(ch.isalpha() for ch in bare):
        return "text"
    if commands or any(ch.isdigit() for ch in token):
        return "operand"
    return "punct"


def auto_math(latex: str) -> str:
    """Wrap the mathematical stretches of a mixed text line in $...$:
    "f je injektivna : x \\ne y \\Rightarrow ..." -> "$f$ je injektivna : $x \\ne y \\Rightarrow ...$"."""
    if "$" in latex or r"\(" in latex or r"\[" in latex or not latex.strip():
        return latex
    if latex.count("{") != latex.count("}"):
        return latex  # do not risk splitting a group
    prefix = ""
    m = _PREFIX.match(latex)
    if m:
        prefix, latex = m.group(0), latex[m.end():]
    parts = _split_top_level(latex)
    kinds = ["space" if p == " " else _kind(p) for p in parts]
    if "math" not in kinds:
        return prefix + latex
    out: list[str] = []
    i = 0
    while i < len(parts):
        if kinds[i] != "math":
            out.append(parts[i])
            i += 1
            continue
        # Extend the run over math, operands, punctuation and spaces; end it on
        # the last math token or operand (so "= 1" keeps its 1, but a
        # trailing ":" or "," stays outside).
        j, last_math = i, i
        while j < len(parts) and kinds[j] in ("math", "operand", "punct", "space"):
            if kinds[j] in ("math", "operand"):
                last_math = j
            j += 1
        out.append("$" + "".join(parts[i : last_math + 1]) + "$")
        i = last_math + 1
    return prefix + "".join(out)


# TeX commands that define macros, touch files or change parsing. They never
# belong in a transcribed line, and some (\input, \write) could read or write
# files when compiling. Replaced by their plain name.
_FORBIDDEN = re.compile(
    r"\\(def|gdef|edef|xdef|let|futurelet|newcommand|renewcommand|providecommand|DeclareRobustCommand|"
    r"input|include|InputIfFileExists|write|immediate|openout|openin|closeout|read|readline|catcode|"
    r"csname|endcsname|expandafter|special|usepackage|RequirePackage|documentclass|makeatletter|makeatother)(?![A-Za-z])"
)


def sanitize_commands(latex: str) -> str:
    return _FORBIDDEN.sub(lambda m: m.group(1), latex)


_RELATION = re.compile(r"^\s*(?:[=<>]|\\(?:le|ge|leq|geq|neq|ne|approx|equiv|sim|simeq|cong|Rightarrow|Leftarrow|Leftrightarrow|iff|implies|to|mapsto|subset|subseteq|supset|supseteq|in|propto|ll|gg|prec|succ)(?![A-Za-z]))")


def fix_spacer_ampersands(latex: str) -> str:
    """In a derivation, & marks the alignment point before a relation sign.
    Models also use it to put two formulas side by side ("f(2) = c & f(3) = b"),
    which LaTeX would glue together; there it becomes a gap instead."""
    parts = latex.split("&")
    if len(parts) == 1:
        return latex
    out = parts[0]
    for part in parts[1:]:
        out += ("&" if _RELATION.match(part) else r"\qquad ") + part
    return out


_ENUM = re.compile(r"^\s*(\d{1,2}|[a-z])[.)]\s+(?=\S)")


def enumerate_line(latex: str) -> str:
    """A text line starting with "1." or "a)" is a numbered point: keep the
    author's label on a list item (lists get wrapped automatically)."""
    m = _ENUM.match(latex)
    if not m:
        return latex
    return f"\\item[{latex[m.start(1):m.end()].strip()}] {latex[m.end():]}"


def tidy_block(b: BlockIn) -> BlockIn:
    latex = b.latex
    if b.role != "code":
        latex = sanitize_commands(fix_mojibake(repair_escapes(latex)))
    if b.role in _LABEL_RE and not b.continues_previous:
        latex = strip_label(b.role, latex)
    if b.role in TEXT_ROLES:
        if b.role not in ("heading", "subheading"):
            latex = enumerate_line(latex)
        latex = auto_math(wrap_bare_math(latex))
    elif b.role in ("equation", "derivation"):
        latex = strip_display_wrappers(latex)
        if b.role == "derivation":
            latex = fix_spacer_ampersands(latex)
    return replace(b, latex=latex) if latex != b.latex else b


def tidy(blocks: list[BlockIn]) -> list[BlockIn]:
    return [tidy_block(b) for b in blocks]
