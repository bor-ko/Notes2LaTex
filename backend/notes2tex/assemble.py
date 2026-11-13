"""Turn transcribed blocks into a full LaTeX document using a subject's templates.

Blocks are single written lines. Consecutive blocks with
``continues_previous`` form a group (a paragraph, a multi-line derivation, a
code listing) that is rendered with one role template.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

from .subjects import Subject

ROLES = [
    "heading",
    "subheading",
    "definition",
    "theorem",
    "lemma",
    "proposition",
    "corollary",
    "proof",
    "example",
    "remark",
    "note",
    "equation",
    "derivation",
    "prose",
    "list",
    "code",
    "pseudocode",
    "diagram",
]

DEFAULT_TEMPLATES = {
    "heading": "\\section{<<body>>}",
    "subheading": "\\subsection{<<body>>}",
    "definition": "\\begin{definition}<<title>>\n<<body>>\n\\end{definition}",
    "theorem": "\\begin{theorem}<<title>>\n<<body>>\n\\end{theorem}",
    "lemma": "\\begin{lemma}<<title>>\n<<body>>\n\\end{lemma}",
    "proposition": "\\begin{proposition}<<title>>\n<<body>>\n\\end{proposition}",
    "corollary": "\\begin{corollary}<<title>>\n<<body>>\n\\end{corollary}",
    "remark": "\\begin{remark}<<title>>\n<<body>>\n\\end{remark}",
    "proof": "\\begin{proof}\n<<body>>\n\\end{proof}",
    "example": "\\begin{example}<<title>>\n<<body>>\n\\end{example}",
    "note": "\\begin{note}<<title>>\n<<body>>\n\\end{note}",
    "equation": "\\[\n<<body>>\n\\]",
    "derivation": "\\begin{align*}\n<<body>>\n\\end{align*}",
    "prose": "<<body>>",
    "list": "\\begin{itemize}\n<<body>>\n\\end{itemize}",
    "code": "\\begin{lstlisting}<<lang>>\n<<body>>\n\\end{lstlisting}",
    "pseudocode": "\\begin{algorithmic}[1]\n<<body>>\n\\end{algorithmic}",
    "diagram": "\\begin{center}\n\\includegraphics[width=<<width>>\\linewidth]{<<image>>}\n\\end{center}",
}

# Statements that can carry a name, e.g. \\begin{definition}[Homotopy].
TITLED_ROLES = {"definition", "theorem", "lemma", "proposition", "corollary", "example", "remark", "note"}
# "[Homotopy] Let ..." names a statement; "[n] = {1, ..., n}" is maths.
_TITLE = re.compile(r"^\s*\[((?=[^\[\]]*[A-Za-z\u00C0-\u017F]{3})[^\[\]$\\=^_{}]{3,80})\]\s*(?![=<>]|\$\s*[=<>])")
_MANUAL_NUMBER = re.compile(r"^(?:\d+\.)+\d*\s|^\d+(?:\.\d+)*\.?\s*[A-Za-z\u00C0-\u017F]")

# How the lines of one group are joined.
JOINERS = {"derivation": " \\\\\n", "heading": " ", "subheading": " "}

# Languages the listings package knows, keyed by lower-case aliases.
LISTINGS_LANGUAGES = {
    "python": "Python",
    "py": "Python",
    "java": "Java",
    "c": "C",
    "c++": "C++",
    "cpp": "C++",
    "haskell": "Haskell",
    "sql": "SQL",
    "bash": "bash",
    "shell": "bash",
    "sh": "bash",
    "matlab": "Matlab",
    "r": "R",
    "ruby": "Ruby",
    "php": "PHP",
    "scala": "Scala",
    "prolog": "Prolog",
    "lisp": "Lisp",
    "fortran": "Fortran",
    "assembler": "Assembler",
}

_PLACEHOLDER = re.compile(r"<<(\w+)>>")


@dataclass
class BlockIn:
    id: int
    role: str
    latex: str
    continues_previous: bool = False
    code_language: str | None = None


@dataclass
class Group:
    role: str
    blocks: list[BlockIn]
    image: str | None = None  # relative figure path for diagrams
    image_width: float = 0.6

    @property
    def block_ids(self) -> list[int]:
        return [b.id for b in self.blocks]


@dataclass
class SourceSpan:
    page_index: int
    block_ids: list[int]
    start_line: int  # 1-based, inclusive
    end_line: int


@dataclass
class Assembled:
    tex: str
    spans: list[SourceSpan] = field(default_factory=list)

    def blocks_at_line(self, line: int) -> SourceSpan | None:
        for span in self.spans:
            if span.start_line <= line <= span.end_line:
                return span
        return None


def group_blocks(blocks: list[BlockIn]) -> list[Group]:
    groups: list[Group] = []
    for b in blocks:
        if b.continues_previous and groups and groups[-1].role == b.role:
            groups[-1].blocks.append(b)
        else:
            groups.append(Group(role=b.role if b.role in ROLES else "prose", blocks=[b]))
    return groups


def _fill(template: str, values: dict[str, str]) -> str:
    return _PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), template)


def _listings_option(lang: str | None) -> str:
    if not lang:
        return ""
    name = LISTINGS_LANGUAGES.get(lang.strip().lower())
    return f"[language={name}]" if name else ""


def _wrap_loose_items(lines: list[str]) -> list[str]:
    """Bullet points written inside a definition, proof, ... arrive as bare
    \\item lines; LaTeX needs a list around them."""
    if not any(ln.lstrip().startswith("\\item") for ln in lines) or any("\\begin{itemize}" in ln or "\\begin{enumerate}" in ln for ln in lines):
        return lines
    out: list[str] = []
    in_list = False
    for ln in lines:
        is_item = ln.lstrip().startswith("\\item")
        if is_item and not in_list:
            out.append("\\begin{itemize}")
            in_list = True
        elif not is_item and in_list:
            out.append("\\end{itemize}")
            in_list = False
        out.append(ln)
    if in_list:
        out.append("\\end{itemize}")
    return out


def render_group(group: Group, subject: Subject) -> str:
    template = subject.role_templates.get(group.role) or DEFAULT_TEMPLATES[group.role]
    lines = [b.latex.rstrip() for b in group.blocks if b.latex.strip()]
    if group.role == "diagram":
        if not group.image:
            return ""
        return _fill(template, {"image": group.image, "width": f"{group.image_width:.2f}"})
    if not lines:
        return ""
    if group.role in ("derivation", "equation"):
        # A trailing \\ on a line would produce an empty row when joined.
        lines = [re.sub(r"\s*\\\\\s*$", "", ln) for ln in lines]
    if group.role == "equation" and len(lines) > 1:
        # Several formulas in one display: stack them, aligned if marked with &.
        env = "align*" if any("&" in ln for ln in lines) else "gather*"
        return f"\\begin{{{env}}}\n" + " \\\\\n".join(lines) + f"\n\\end{{{env}}}"
    if group.role == "list" and not lines[0].lstrip().startswith("\\item"):
        lines[0] = "\\item " + lines[0].lstrip()
    elif group.role not in ("list", "code", "pseudocode"):
        lines = _wrap_loose_items(lines)
    if group.role in ("heading", "subheading"):
        lines = [ln.strip() for ln in lines]
        # Headings numbered by hand ("1.1. Functions") would get a second
        # number from LaTeX; keep the author's and drop LaTeX's.
        if _MANUAL_NUMBER.match(lines[0]) and group.role not in subject.role_templates:
            template = template.replace("\\section{", "\\section*{").replace("\\subsection{", "\\subsection*{")
    title = ""
    if group.role in TITLED_ROLES:
        m = _TITLE.match(lines[0])
        if m:
            title = f"[{m.group(1).strip()}]"
            lines[0] = lines[0][m.end():]
            lines = [ln for ln in lines if ln.strip()] or [""]
    if not title and lines and lines[0].lstrip().startswith("[") and group.role in TITLED_ROLES | {"proof"}:
        # LaTeX would read a leading "[n]" as the environment's optional name.
        lines[0] = "{}" + lines[0].lstrip()
    body = JOINERS.get(group.role, "\n").join(lines)
    lang = next((b.code_language for b in group.blocks if b.code_language), None) or subject.default_code_language
    return _fill(template, {"body": body, "lang": _listings_option(lang), "title": title})


def escape_text(s: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(c, c) for c in s)


def assemble_document(
    pages: list[list[Group]],
    subject: Subject,
    title: str,
    draft: bool = True,
    date: dt.date | None = None,
) -> Assembled:
    """Build the full .tex and a map from source lines back to blocks."""
    head, _, tail = subject.layout.partition("<<body>>")
    head = _fill(
        head,
        {
            "preamble": subject.preamble.rstrip(),
            "draft": "\\notesdrafttrue" if draft else "",
            "title": escape_text(title),
            "date": (date or dt.date.today()).strftime("%-d %B %Y"),
        },
    )
    out: list[str] = [head.rstrip("\n")]
    spans: list[SourceSpan] = []
    line = head.rstrip("\n").count("\n") + 1  # last line used so far

    for page_index, groups in enumerate(pages):
        if len(pages) > 1:
            out.append(f"% ---- page {page_index + 1} ----")
            line += 1
        for group in groups:
            chunk = render_group(group, subject)
            if not chunk:
                continue
            # Blank line before each group keeps paragraphs separate.
            out.append("")
            line += 1
            n = chunk.count("\n") + 1
            spans.append(SourceSpan(page_index, group.block_ids, line + 1, line + n))
            out.append(chunk)
            line += n
    out.append(tail.lstrip("\n").rstrip() if tail.strip() else "")
    return Assembled(tex="\n".join(out).rstrip() + "\n", spans=spans)
