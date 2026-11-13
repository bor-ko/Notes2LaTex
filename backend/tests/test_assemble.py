from notes2tex.assemble import BlockIn, assemble_document, group_blocks, render_group
from notes2tex.subjects import load_subject


def test_grouping_follows_continues_previous_and_role():
    blocks = [
        BlockIn(1, "prose", "a"),
        BlockIn(2, "prose", "b", True),
        BlockIn(3, "derivation", "x &= 1", True),  # role differs -> new group
        BlockIn(4, "derivation", "y &= 2", True),
    ]
    groups = group_blocks(blocks)
    assert [g.block_ids for g in groups] == [[1, 2], [3, 4]]


def test_derivation_lines_joined_without_double_breaks():
    s = load_subject("math")
    g = group_blocks([BlockIn(1, "derivation", r"x &= 1 \\"), BlockIn(2, "derivation", "y &= 2", True)])[0]
    out = render_group(g, s)
    assert out == "\\begin{align*}\nx &= 1 \\\\\ny &= 2\n\\end{align*}"


def test_list_gets_item_and_code_gets_language():
    s = load_subject("cs")
    lst = render_group(group_blocks([BlockIn(1, "list", "first point")])[0], s)
    assert "\\item first point" in lst
    code = render_group(group_blocks([BlockIn(1, "code", "def f(x):"), BlockIn(2, "code", "    return x", True)])[0], s)
    assert "[language=Python]" in code  # cs default language
    assert "\n    return x\n" in code
    js = render_group(group_blocks([BlockIn(1, "code", "let x = 1", code_language="javascript")])[0], s)
    assert "language=" not in js  # listings has no JavaScript; omit rather than fail


def test_source_map_points_at_block_lines():
    s = load_subject("math")
    blocks = [BlockIn(10, "heading", "Title"), BlockIn(11, "equation", r"E = mc^2"), BlockIn(12, "prose", "")]
    doc = assemble_document([group_blocks(blocks)], s, "T & T")
    lines = doc.tex.split("\n")
    spans = {tuple(sp.block_ids): sp for sp in doc.spans}
    assert lines[spans[(10,)].start_line - 1] == r"\section{Title}"
    eq = spans[(11,)]
    assert lines[eq.start_line - 1 : eq.end_line] == [r"\[", "E = mc^2", r"\]"]
    assert (12,) not in spans  # empty blocks are skipped
    assert r"\title{T \& T}" in doc.tex
    assert r"\notesdrafttrue" in doc.tex
    assert r"\notesdrafttrue" not in assemble_document([group_blocks(blocks)], s, "T", draft=False).tex


def test_role_template_override(monkeypatch):
    s = load_subject("math")
    s.role_templates = {"heading": r"\section*{<<body>>}"}
    out = render_group(group_blocks([BlockIn(1, "heading", "Waves")])[0], s)
    assert out == r"\section*{Waves}"


def test_named_statement_moves_into_optional_argument():
    s = load_subject("math")
    out = render_group(group_blocks([BlockIn(1, "definition", "[Homotopy] Let $f$ be a map.")])[0], s)
    assert out.startswith("\\begin{definition}[Homotopy]\nLet $f$ be a map.")
    plain = render_group(group_blocks([BlockIn(1, "lemma", "$x$ is [not] special")])[0], s)
    assert plain.startswith("\\begin{lemma}\n$x$")  # brackets later in the line are content


def test_loose_items_get_a_list():
    s = load_subject("math")
    g = group_blocks(
        [
            BlockIn(1, "definition", "A map such that"),
            BlockIn(2, "definition", r"\item $H(x,0) = f(x)$", True),
            BlockIn(3, "definition", r"\item $H(x,1) = g(x)$", True),
            BlockIn(4, "definition", "We write $f \\simeq g$.", True),
        ]
    )[0]
    lines = render_group(g, s).split("\n")
    assert lines[2] == r"\begin{itemize}" and lines[5] == r"\end{itemize}"
