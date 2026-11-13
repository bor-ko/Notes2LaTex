import cv2
import numpy as np
import pytest
from conftest import EXPECTED_LINES, make_page

from notes2tex.ingest import ingest
from notes2tex.segment import segment


# Light lines (typical) and darker ones that fall inside the ink threshold,
# so the template-removal step is really exercised.
@pytest.mark.parametrize(
    "paper,colour",
    [("blank", None), ("lined", (160, 190, 230)), ("grid", (160, 190, 230)), ("lined", (110, 130, 170)), ("grid", (110, 130, 170))],
)
def test_one_box_per_written_line(tmp_path, paper, colour):
    pdf = make_page(tmp_path / "n.pdf", paper=paper, line_colour=colour or (0, 0, 0))
    page = ingest(pdf, tmp_path / "pages")[0]
    boxes = segment(cv2.imread(str(page.mask_path), cv2.IMREAD_GRAYSCALE))
    assert len(boxes) == EXPECTED_LINES
    # reading order
    assert [b.y0 for b in boxes] == sorted(b.y0 for b in boxes)


@pytest.mark.parametrize("paper", ["lined", "grid"])
def test_paper_lines_removed_but_ink_kept(tmp_path, paper):
    pdf = make_page(tmp_path / "n.pdf", paper=paper, line_colour=(110, 130, 170))
    page = ingest(pdf, tmp_path / "pages")[0]
    mask = cv2.imread(str(page.mask_path), cv2.IMREAD_GRAYSCALE) > 0
    # no row or column is mostly ink (template lines span ~90% of the page)
    assert (mask.sum(axis=1) / mask.shape[1]).max() < 0.5
    assert (mask.sum(axis=0) / mask.shape[0]).max() < 0.3
    # red ink of the example line survives in the cleaned page
    clean = cv2.imread(str(page.image_path))
    b, g, r = clean[..., 0].astype(int), clean[..., 1].astype(int), clean[..., 2].astype(int)
    assert np.count_nonzero((r > 150) & (g < 80) & (b < 80)) > 500


def test_fraction_is_one_block(tmp_path):
    pdf = make_page(tmp_path / "n.pdf")
    page = ingest(pdf, tmp_path / "pages")[0]
    last = segment(cv2.imread(str(page.mask_path), cv2.IMREAD_GRAYSCALE))[-1]
    # numerator + bar + denominator: much taller than a normal line
    assert last.height > 130


def test_fraction_bar_joins_both_sides_but_underline_only_above():
    from notes2tex.segment import _attach_thin_bands

    # numerator, bar, denominator with similar gaps -> one line
    assert _attach_thin_bands([(100, 150), (175, 180), (205, 255)], typical=50) == [(100, 255)]
    # heading, underline right below it, then the next line further down
    assert _attach_thin_bands([(100, 150), (154, 158), (196, 246)], typical=50) == [(100, 158), (196, 246)]


@pytest.mark.parametrize("colour", [(160, 190, 230), (110, 130, 170)])
def test_clean_page_has_no_grid_left_outside_the_writing(tmp_path, colour):
    pdf = make_page(tmp_path / "n.pdf", paper="grid", line_colour=colour)
    page = ingest(pdf, tmp_path / "pages")[0]
    gray = cv2.cvtColor(cv2.imread(str(page.image_path)), cv2.COLOR_BGR2GRAY)
    outside = np.ones_like(gray, bool)
    for b in segment(cv2.imread(str(page.mask_path), cv2.IMREAD_GRAYSCALE)):
        outside[b.y0 : b.y1, b.x0 : b.x1] = False
    assert int(((gray < 250) & outside).sum()) == 0  # crossings of grid lines are gone too
