"""Offline end-to-end: import -> transcribe -> edit -> approve -> build -> eval."""

import shutil

import pytest
from conftest import EXPECTED_LINES

from notes2tex import dataset, evaluate, pipeline
from notes2tex.transcribe.base import OfflineTranscriber

needs_tectonic = pytest.mark.skipif(shutil.which("tectonic") is None, reason="tectonic not installed")


def test_import_segments_and_crops(isolated_env, tablet_pdf):
    doc_id = pipeline.import_document(tablet_pdf, "math", "Kinematics")
    with dataset.connect() as conn:
        pages = dataset.list_pages(conn, doc_id)
        blocks = dataset.get_blocks(conn, pages[0]["id"])
    assert len(pages) == 1 and len(blocks) == EXPECTED_LINES
    assert all(b["crop_path"] and b["crop_path"].endswith(".png") for b in blocks)


def test_merge_and_split_roundtrip(isolated_env, tablet_pdf):
    doc_id = pipeline.import_document(tablet_pdf, "math")
    with dataset.connect() as conn:
        page_id = dataset.list_pages(conn, doc_id)[0]["id"]
        first = dataset.get_blocks(conn, page_id)[0]
    pipeline.merge_with_next(first["id"])
    with dataset.connect() as conn:
        assert len(dataset.get_blocks(conn, page_id)) == EXPECTED_LINES - 1
    pipeline.split_block(first["id"], 0.5)
    with dataset.connect() as conn:
        blocks = dataset.get_blocks(conn, page_id)
    assert len(blocks) == EXPECTED_LINES
    assert [b["ord"] for b in blocks] == list(range(EXPECTED_LINES))
    assert blocks[0]["y1"] <= blocks[1]["y0"]


class Fixer(OfflineTranscriber):
    name = "fixer"

    def __init__(self):
        self.calls = 0

    def fix_latex(self, lines, role, error, subject):
        self.calls += 1
        return [ln.replace(r"\fraq", r"\frac") for ln in lines]


@needs_tectonic
def test_full_offline_flow_with_autofix(isolated_env, tablet_pdf):
    doc_id = pipeline.import_document(tablet_pdf, "math", "Kinematics")
    pipeline.transcribe_document(doc_id, OfflineTranscriber())
    with dataset.connect() as conn:
        page = dataset.list_pages(conn, doc_id)[0]
        assert page["status"] == "transcribed" and page["unsure_count"] == EXPECTED_LINES
        blocks = dataset.get_blocks(conn, page["id"])
        # The reviewer corrects two blocks; one has a typo that breaks LaTeX.
        dataset.update_block(conn, blocks[0]["id"], role="heading", latex="Kinematics")
        dataset.update_block(conn, blocks[-1]["id"], role="equation", latex=r"\fraq{x+1}{2y}")

    fixer = Fixer()
    result = pipeline.build_document(doc_id, fixer=fixer)
    assert result.ok, result.log
    assert result.fixes == 1 and fixer.calls == 1
    assert result.pdf_path and result.pdf_path.exists()
    assert r"\frac{x+1}{2y}" in result.tex_path.read_text()

    with dataset.connect() as conn:
        dataset.approve_page(conn, page["id"])
        rows = dataset.reviewed_blocks(conn)
    assert len(rows) == EXPECTED_LINES
    rep = evaluate.report(rows)
    assert rep["overall"]["blocks"] == EXPECTED_LINES
    assert 0 <= rep["overall"]["token_accuracy"] < 1  # placeholders were corrected


@needs_tectonic
def test_build_reports_block_of_unfixable_error(isolated_env, tablet_pdf):
    doc_id = pipeline.import_document(tablet_pdf, "math")
    pipeline.transcribe_document(doc_id, OfflineTranscriber())
    with dataset.connect() as conn:
        page = dataset.list_pages(conn, doc_id)[0]
        bad = dataset.get_blocks(conn, page["id"])[2]
        dataset.update_block(conn, bad["id"], role="equation", latex=r"\nosuchmacro x")
    result = pipeline.build_document(doc_id)  # no fixer
    assert not result.ok and result.pdf_path is None
    assert result.errors[0].block_ids == [bad["id"]]
    assert result.errors[0].page_id == page["id"]
