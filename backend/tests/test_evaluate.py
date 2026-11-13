import pytest

from notes2tex.evaluate import edit_distance, report, token_accuracy, tokenize


def test_tokenize_commands_and_chars():
    assert tokenize(r"\frac{a}{b}") == [r"\frac", "{", "a", "}", "{", "b", "}"]


@pytest.mark.parametrize(
    "a,b",
    [
        (r"\dfrac{1}{2}", r"\frac{1}{2}"),
        (r"x^{2}", r"x^2"),
        (r"\left( x \right)", r"(x)"),
        (r"a\,b", r"a b"),
        (r"\unsure{x}+1", r"x+1"),
        (r"\le", r"\leq"),
        (r"\(x\)", r"$x$"),
    ],
)
def test_rendering_neutral_differences_are_free(a, b):
    assert token_accuracy(a, b) == 1.0


def test_one_misread_symbol():
    # ref has 5 tokens: x = y + 1 ; one substitution -> 80%
    assert token_accuracy("x=y+7", "x=y+1") == pytest.approx(0.8)


def test_edit_distance():
    assert edit_distance(list("kitten"), list("sitting")) == 3


def test_report_is_micro_averaged():
    rows = [
        {"draft_latex": "abcd", "final_latex": "abcd", "draft_role": "prose", "final_role": "prose", "role": "prose", "subject": "math", "source_model": "m"},
        {"draft_latex": "ab", "final_latex": "xy", "draft_role": "prose", "final_role": "equation", "role": "equation", "subject": "math", "source_model": "m"},
    ]
    rep = report(rows)
    assert rep["overall"]["token_accuracy"] == pytest.approx(1 - 2 / 6, abs=1e-4)
    assert rep["overall"]["exact_match"] == 0.5
    assert rep["overall"]["role_accuracy"] == 0.5
    assert set(rep["role"]) == {"prose", "equation"}
