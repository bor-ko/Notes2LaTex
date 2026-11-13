"""Accuracy metric: LaTeX token accuracy between a prediction and your final.

token accuracy = 1 - edit_distance(pred_tokens, ref_tokens) / len(ref_tokens)

Before comparing, both sides are normalised so that differences which do not
change the rendered result (spacing, \\dfrac vs \\frac, ^{2} vs ^2, \\left( vs
() do not count as errors. Each command is one token; every other character
is its own token, so a misread letter costs one error.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

_TOKEN = re.compile(r"\\[a-zA-Z]+\*?|\\.|\S")
_SPACING = re.compile(r"\\(?:[,;:!]|(?:quad|qquad|thinspace|medspace|thickspace|enspace)(?![a-zA-Z]))|\\ |~")
_SIZING = re.compile(r"\\(?:left|right|big|Big|bigg|Bigg)[lr]?(?![a-zA-Z])")
_ALIASES = {
    r"\dfrac": r"\frac",
    r"\tfrac": r"\frac",
    r"\le": r"\leq",
    r"\ge": r"\geq",
    r"\ne": r"\neq",
    r"\to": r"\rightarrow",
    r"\gets": r"\leftarrow",
    r"\lbrace": r"\{",
    r"\rbrace": r"\}",
    r"\land": r"\wedge",
    r"\lor": r"\vee",
    r"\lnot": r"\neg",
    r"\vert": "|",
    r"\mid": "|",
    r"\colon": ":",
    r"\(": "$",
    r"\)": "$",
}


def _strip_unsure(s: str) -> str:
    """\\unsure{x} -> x (balanced braces)."""
    out, i, key = [], 0, "\\unsure{"
    while True:
        j = s.find(key, i)
        if j < 0:
            out.append(s[i:])
            return "".join(out)
        out.append(s[i:j])
        k, depth = j + len(key), 1
        start = k
        while k < len(s) and depth:
            depth += {"{": 1, "}": -1}.get(s[k], 0)
            k += 1
        out.append(s[start : k - 1] if depth == 0 else s[start:])
        i = k


def tokenize(latex: str) -> list[str]:
    s = _strip_unsure(latex or "")
    s = _SPACING.sub(" ", s)
    s = _SIZING.sub("", s)
    s = s.replace("$$", "$")
    tokens = [_ALIASES.get(t, t) for t in _TOKEN.findall(s)]
    # ^{x} / _{x} with a single token inside -> ^x / _x
    out: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in ("^", "_") and i + 3 < len(tokens) and tokens[i + 1] == "{" and tokens[i + 3] == "}":
            out += [t, tokens[i + 2]]
            i += 4
            continue
        out.append(t)
        i += 1
    return out


def edit_distance(a: list[str], b: list[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def token_accuracy(pred: str, ref: str) -> float:
    p, r = tokenize(pred), tokenize(ref)
    if not r:
        return 1.0 if not p else 0.0
    return max(0.0, 1.0 - edit_distance(p, r) / len(r))


@dataclass
class Score:
    blocks: int = 0
    ref_tokens: int = 0
    errors: int = 0
    exact: int = 0
    role_correct: int = 0

    def add(self, pred: str, ref: str, role_ok: bool) -> None:
        p, r = tokenize(pred), tokenize(ref)
        d = edit_distance(p, r)
        self.blocks += 1
        self.ref_tokens += len(r)
        self.errors += min(d, max(len(r), 1))
        self.exact += d == 0
        self.role_correct += role_ok

    @property
    def accuracy(self) -> float:
        return 1.0 - self.errors / self.ref_tokens if self.ref_tokens else float("nan")

    def as_dict(self) -> dict:
        return {
            "blocks": self.blocks,
            "ref_tokens": self.ref_tokens,
            "token_accuracy": round(self.accuracy, 4) if self.ref_tokens else None,
            "exact_match": round(self.exact / self.blocks, 4) if self.blocks else None,
            "role_accuracy": round(self.role_correct / self.blocks, 4) if self.blocks else None,
        }


def report(rows: list[dict]) -> dict:
    """Aggregate (draft vs final) over reviewed blocks.

    Micro-averaged: long blocks weigh more than short ones, so the headline
    number is "share of LaTeX tokens the model got right".
    """
    overall = Score()
    groups: dict[str, dict[str, Score]] = {"subject": defaultdict(Score), "role": defaultdict(Score), "model": defaultdict(Score)}
    for r in rows:
        if r.get("draft_latex") is None:
            continue
        role_ok = r.get("draft_role") == r.get("final_role")
        for s in (
            overall,
            groups["subject"][r["subject"]],
            groups["role"][r.get("final_role") or r["role"]],
            groups["model"][r.get("source_model") or "?"],
        ):
            s.add(r["draft_latex"], r["final_latex"], role_ok)
    return {
        "overall": overall.as_dict(),
        **{k: {name: s.as_dict() for name, s in sorted(v.items())} for k, v in groups.items()},
    }
