"""Local engine against a fake Ollama server (no model needed)."""

import json

import httpx
import pytest
from PIL import Image

from notes2tex.segment import Box
from notes2tex.subjects import load_subject
from notes2tex.transcribe.base import PageInput, TranscriptionError
from notes2tex.transcribe.ollama import OllamaTranscriber


def _page(n=3):
    return PageInput(Image.new("RGB", (1000, 1400), "white"), [Box(50, 50 + 120 * i, 900, 130 + 120 * i) for i in range(n)])


def fake_ollama(replies, models=("qwen3-vl:8b-instruct",), seen=None):
    """replies: list of dicts (JSON the model 'returns') or raw strings."""
    queue = list(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": m} for m in models]})
        if request.url.path == "/api/chat":
            body = json.loads(request.content)
            if seen is not None:
                seen.append(body)
            reply = queue.pop(0)
            content = reply if isinstance(reply, str) else json.dumps(reply)
            return httpx.Response(
                200,
                json={"message": {"content": content}, "prompt_eval_count": 900, "eval_count": 40, "total_duration": 2_000_000_000},
            )
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _transcriber(client, **kw):
    return OllamaTranscriber(model="qwen3-vl:8b-instruct", host="http://ollama.test", client=client, **kw)


def L(role, latex, cont=False):
    return {"role": role, "latex": latex, "continues_previous": cont}


def test_lines_are_batched_with_context():
    seen = []
    client = fake_ollama(
        [
            {"lines": [L("heading", "Kinematics"), L("derivation", "v &= \\frac{\\dd x}{\\dd t}")]},
            {"lines": [L("derivation", "a &= \\dot v", True)]},
        ],
        seen=seen,
    )
    result = _transcriber(client, batch=2).transcribe_page(_page(3), load_subject("math"))
    assert [b.role for b in result.blocks] == ["heading", "derivation", "derivation"]
    assert [b.continues_previous for b in result.blocks] == [False, False, True]
    assert len(seen) == 2 and [len(r["messages"][1]["images"]) for r in seen] == [1, 1]  # one strip per batch
    assert "numbered 1-2" in seen[0]["messages"][1]["content"]
    assert seen[0]["format"]["properties"]["lines"]["type"] == "array"  # schema-constrained output
    assert seen[0]["options"]["temperature"] == 0 and seen[0]["options"]["num_ctx"] == 4096
    assert "start of the page" in seen[0]["messages"][1]["content"]
    assert "[derivation] v &=" in seen[1]["messages"][1]["content"]  # previous lines as context
    assert "Vectors are written" in seen[0]["messages"][0]["content"]  # subject style rules
    assert "Slovenian" in seen[0]["messages"][0]["content"]  # language of the notes
    assert result.usage["requests"] == 2 and result.model == "ollama:qwen3-vl:8b-instruct"


def test_wrong_number_of_answers_falls_back_to_one_by_one():
    seen = []
    client = fake_ollama(
        [
            {"lines": [L("prose", "only one")]},  # asked for 2
            {"lines": [L("prose", "first")]},
            {"lines": [L("prose", "second", True)]},
        ],
        seen=seen,
    )
    result = _transcriber(client, batch=2).transcribe_page(_page(2), load_subject("math"))
    assert [b.latex for b in result.blocks] == ["first", "second"]
    assert len(seen) == 3  # one strip, then each line on its own
    assert any("got 1 answers for 2 lines" in w for w in result.warnings)


def test_continuation_with_a_different_role_starts_a_new_element():
    client = fake_ollama([{"lines": [L("prose", "Therefore"), L("equation", "x = 1", True)]}])
    result = _transcriber(client, batch=5).transcribe_page(_page(2), load_subject("math"))
    assert result.blocks[1].continues_previous is False


def test_invalid_json_is_retried_then_reported_per_line():
    good = {"lines": [L("prose", "ok")]}
    client = fake_ollama(["not json", good, "nope", "still not json", good])
    result = _transcriber(client, batch=1).transcribe_page(_page(3), load_subject("math"))
    assert [b.latex for b in result.blocks] == ["ok", "", "ok"]  # line 2 failed twice
    assert any(w.startswith("line 2:") for w in result.warnings)


def test_missing_model_and_server_give_clear_errors():
    with pytest.raises(TranscriptionError, match="ollama pull qwen3-vl:8b-instruct"):
        _transcriber(fake_ollama([], models=("deepseek-coder:latest",)))

    def down(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(TranscriptionError, match="Ollama is not running"):
        _transcriber(httpx.Client(transport=httpx.MockTransport(down)))


def test_page_mode_uses_numbered_overlay():
    seen = []
    client = fake_ollama(
        [{"blocks": [{"id": 1, "role": "heading", "latex": "Waves", "continues_previous": False, "code_language": None}]}],
        seen=seen,
    )
    result = _transcriber(client, mode="page").transcribe_page(_page(2), load_subject("math"))
    assert len(seen) == 1 and "2 numbered boxes" in seen[0]["messages"][1]["content"]
    assert result.blocks[0].latex == "Waves" and result.blocks[1].latex == ""
    assert any("block 2 missing" in w for w in result.warnings)


def test_strip_stacks_numbered_lines():
    page = _page(3)
    strip = page.strip([0, 1, 2])
    lines = [page.line(i, pad=10) for i in range(3)]
    assert strip.height == sum(c.height for c in lines) + 2 * 14
    assert strip.width == 60 + max(c.width for c in lines)
