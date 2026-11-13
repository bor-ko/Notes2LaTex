"""The Claude transcriber, with a fake client (no network, no API key)."""

from types import SimpleNamespace

from PIL import Image

from notes2tex.segment import Box
from notes2tex.subjects import Example, load_subject
from notes2tex.transcribe.base import PageInput
from notes2tex.transcribe.claude import BlockOut, ClaudeTranscriber, PageOut, build_request


def _page():
    return Image.new("RGB", (1654, 2339), "white")


def test_request_shape_without_examples():
    s = load_subject("math")
    s.examples = []
    req = build_request(_page(), [1, 2, 3], s)
    assert req["model"] == "claude-opus-5-5"
    assert req["thinking"] == {"type": "adaptive"}
    assert req["fallbacks"] == "default"
    assert req["output_format"] is PageOut
    assert req["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "Vectors are written" in req["system"][0]["text"]  # subject rules reach the prompt
    content = req["messages"][0]["content"]
    assert [c["type"] for c in content] == ["image", "text"]
    assert "3 numbered boxes" in content[-1]["text"]


def test_examples_form_cached_prefix(tmp_path):
    s = load_subject("math")
    img = tmp_path / "ex.png"
    Image.new("RGB", (800, 1100), "white").save(img)
    s.examples = [Example("ex", [img], tex=r"\section{X}")]
    req = build_request(_page(), [1], s)
    content = req["messages"][0]["content"]
    cached = [i for i, c in enumerate(content) if "cache_control" in c]
    assert cached == [2]  # last example block, before the page image
    assert content[3]["type"] == "image"
    assert "cache_control" not in req["system"][-1]


def test_large_pages_are_downscaled():
    s = load_subject("math")
    s.examples = []
    import base64, io

    req = build_request(Image.new("RGB", (3000, 4000), "white"), [1], s)
    data = base64.b64decode(req["messages"][0]["content"][0]["source"]["data"])
    assert max(Image.open(io.BytesIO(data)).size) == 2000


class FakeMessages:
    def __init__(self, blocks):
        self.blocks = blocks
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            stop_reason="end_turn",
            parsed_output=PageOut(blocks=self.blocks),
            model="claude-opus-5-5",
            usage=SimpleNamespace(input_tokens=1000, output_tokens=500, cache_read_input_tokens=0, cache_creation_input_tokens=0),
        )


def test_transcriber_maps_blocks_and_flags_missing():
    blocks = [
        BlockOut(id=1, role="heading", latex="Waves", continues_previous=False, code_language=None),
        BlockOut(id=3, role="equation", latex=r"\omega = 2\pi f", continues_previous=False, code_language=None),
        BlockOut(id=9, role="prose", latex="?", continues_previous=False, code_language=None),
    ]
    fake = FakeMessages(blocks)
    t = ClaudeTranscriber(client=SimpleNamespace(beta=SimpleNamespace(messages=fake)))
    s = load_subject("math")
    s.examples = []
    page = PageInput(_page(), [Box(10, 10 + 100 * i, 500, 80 + 100 * i) for i in range(3)])
    result = t.transcribe_page(page, s)
    assert [b.id for b in result.blocks] == [1, 2, 3]
    assert result.blocks[1].latex == "" and any("block 2 missing" in w for w in result.warnings)
    assert any("unknown block ids [9]" in w for w in result.warnings)
    assert result.usage["output_tokens"] == 500
