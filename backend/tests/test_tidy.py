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
    assert b.latex.startswith("Def:")  # label kept mid-element


def test_multi_line_equation_is_stacked():
    s = load_subject("math")
    g = group_blocks([BlockIn(1, "equation", "v = 1"), BlockIn(2, "equation", "a = 2", True)])[0]
    assert render_group(g, s) == "\\begin{gather*}\nv = 1 \\\\\na = 2\n\\end{gather*}"


@pytest.mark.parametrize(
    "role,src,out",
    [
        ("definition", "def. $f$ je injektivna", "$f$ je injektivna"),
        ("example", "Primer: $X \\subseteq [100]$", "$X \\subseteq [100]$"),
        ("theorem", "Izrek (Dirichlet). Če je", "[Dirichlet] Če je"),
        ("proof", "Dokaz: očitno", "očitno"),
        ("corollary", "Posledica 2. Velja", "Velja"),
    ],
)
def test_slovenian_labels(role, src, out):
    assert strip_label(role, src) == out


def test_hand_numbered_headings_are_not_numbered_twice():
    s = load_subject("math")
    numbered = render_group(group_blocks([BlockIn(1, "subheading", "1.1. Funkcije in štetje")])[0], s)
    assert numbered == r"\subsection*{1.1. Funkcije in štetje}"
    title = render_group(group_blocks([BlockIn(1, "heading", "1.OSNOVNI PROBLEMI")])[0], s)
    assert title.startswith(r"\section*{")
    plain = render_group(group_blocks([BlockIn(1, "heading", "Kombinatorika")])[0], s)
    assert plain == r"\section{Kombinatorika}"


from notes2tex.tidy import auto_math, fix_mojibake, repair_escapes  # noqa: E402


def test_repair_unescaped_backslashes():
    assert repair_escapes("f : A \to B") == r"f : A \to B"
    assert repair_escapes("\x0corall x \x08ar{y} \neq") == r"\forall x \bar{y} \neq"
    assert repair_escapes("a\tb") == r"a\tb"  # \tb is still a command-shaped string
    assert repair_escapes("x \t 1") == "x \t 1"  # tab before a space: leave it


def test_fix_mojibake():
    assert fix_mojibake("potencna mnoÅ¾ina") == "potencna množina"
    assert fix_mojibake("vrčaj ostane") == "vrčaj ostane"  # already fine
    assert fix_mojibake("Ä\x8de") == "če"


@pytest.mark.parametrize(
    "src,out",
    [
        (
            r"f je injektivna : x \ne y \Rightarrow f(x) \ne f(y)",
            r"$f$ je injektivna : $x \ne y \Rightarrow f(x) \ne f(y)$",
        ),
        (r"Interpretacija: A kroglice , B škatle \rightarrow vsako", r"Interpretacija: $A$ kroglice , $B$ škatle $\rightarrow$ vsako"),
        (r"f je bijektivna \text{če} velja 1 in 2", r"$f$ je bijektivna \text{če} velja 1 in 2"),
        (r"Lh: mn vseh podmnožic od A", r"Lh: mn vseh podmnožic od $A$"),
        (r"2^A = P(A) = \text{potenčna množica}", r"$2^A = P(A) = \text{potenčna množica}$"),
        (r"\item f(x) = 1 za vse x", r"\item $f(x) = 1$ za vse $x$"),
        ("v vrsti je 5 elementov", "v vrsti je 5 elementov"),  # Slovenian 'v' is a word
        ("already $x$ done", "already $x$ done"),
    ],
)
def test_auto_math(src, out):
    assert auto_math(src) == out


from notes2tex.tidy import enumerate_line, fix_spacer_ampersands  # noqa: E402


def test_spacer_ampersands_become_gaps():
    assert fix_spacer_ampersands("f(2) = c & f(3) = b") == r"f(2) = c \qquad  f(3) = b"
    assert fix_spacer_ampersands(r"a &= b \Rightarrow c") == r"a &= b \Rightarrow c"
    assert fix_spacer_ampersands(r"x &\le y & z = 1") == r"x &\le y \qquad  z = 1"


def test_numbered_points_become_items():
    assert enumerate_line("1. f je injektivna") == r"\item[1.] f je injektivna"
    assert enumerate_line("b) druga točka") == r"\item[b)] druga točka"
    assert enumerate_line("1.5 je število") == "1.5 je število"


def test_numbered_definition_lines_render_as_a_list():
    blocks = [
        BlockIn(1, "definition", "1. f je injektivna : x \\ne y \\Rightarrow f(x) \\ne f(y)"),
        BlockIn(2, "definition", "2. f je surjektivna", True),
    ]
    tidied = [tidy_block(b) for b in blocks]
    out = render_group(group_blocks(tidied)[0], load_subject("math"))
    assert "\\begin{itemize}\n\\item[1.] $f$ je injektivna : $x \\ne y \\Rightarrow f(x) \\ne f(y)$\n\\item[2.] $f$ je surjektivna\n\\end{itemize}" in out


def test_set_brackets_are_not_statement_names():
    s = load_subject("math")
    out = render_group(group_blocks([tidy_block(BlockIn(1, "definition", r"[n] $= \{1, \dots, n\}$"))])[0], s)
    assert out.startswith("\\begin{definition}\n{}[n] $=")  # {} stops LaTeX reading [n] as a name
    named = render_group(group_blocks([BlockIn(1, "definition", "[Potenčna množica] $2^A$")])[0], s)
    assert named.startswith("\\begin{definition}[Potenčna množica]")
