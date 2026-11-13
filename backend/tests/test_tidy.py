import pytest

from notes2tex.assemble import BlockIn, group_blocks, render_group
from notes2tex.subjects import load_subject
from notes2tex.tidy import strip_display_wrappers, strip_label, tidy_block, wrap_bare_math


@pytest.mark.parametrize(
    "role,src,out",
    [
        ("definition", "Def: velocity is the rate of change", "velocity is the rate of change"),
        ("definition", "Def (Homotopy): Let $f$ be a map", "[Homotopy] Let $f$ be a map"),
        ("definition", "Definition 3. A group is", "A group is"),
        ("theorem", "Stelling (Cauchy). Als $f$", "[Cauchy] Als $f$"),
        ("example", "Example: $x(t) = 3t^2$", "$x(t) = 3t^2$"),
        ("example", "Ex. Take $n = 2$", "Take $n = 2$"),
        ("proof", "Proof: trivial", "trivial"),
        ("proof", "Bewijs. Triviaal", "Triviaal"),
        ("note", "Note that $x > 0$", "Note that $x > 0$"),  # a sentence, not a label
        ("definition", "Defining property of $f$", "Defining property of $f$"),
        ("prose", "Def: not a definition role", "Def: not a definition role"),
    ],
)
def test_strip_label(role, src, out):
    assert strip_label(role, src) == out


@pytest.mark.parametrize(
    "src,out",
    [
        ("x(t) = 3t^{2} + 2t", "$x(t) = 3t^{2} + 2t$"),
        (r"\item f(x) = x^2", r"\item $f(x) = x^2$"),
        ("[Name] a^2 + b^2 = c^2", "[Name] $a^2 + b^2 = c^2$"),
        ("velocity is the rate of change", "velocity is the rate of change"),  # words: leave alone
        ("$x = 1$ already", "$x = 1$ already"),
        ("Let f be continuous with f^{-1}(U) open", "Let f be continuous with f^{-1}(U) open"),  # mixed: leave for auto-fix
    ],
)
def test_wrap_bare_math(src, out):
    assert wrap_bare_math(src) == out


def test_strip_display_wrappers():
    assert strip_display_wrappers(r"\[ x = 1 \]") == "x = 1"
    assert strip_display_wrappers("$x = 1$") == "x = 1"
    assert strip_display_wrappers("$a$ and $b$") == "$a$ and $b$"


def test_tidy_only_strips_labels_at_the_start_of_an_element():
    b = tidy_block(BlockIn(2, "definition", "Def: x", continues_previous=True))
    assert b.latex == "Def: x"


def test_multi_line_equation_is_stacked():
    s = load_subject("math")
    g = group_blocks([BlockIn(1, "equation", "v = 1"), BlockIn(2, "equation", "a = 2", True)])[0]
    assert render_group(g, s) == "\\begin{gather*}\nv = 1 \\\\\na = 2\n\\end{gather*}"
