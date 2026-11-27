"""Where values are printed: the frame of a displayed page and the rectangles placed on
it. pdfium is the witness (tests/boxes_help.py), never the formulas under test."""

import functools
import logging
from itertools import islice

import pytest

from invoice_recognition.recognition import boxes
from invoice_recognition.recognition import layout as lay
from invoice_recognition.recognition.boxes import PageFrame

from boxes_help import SHIFT, VARIANTS, Oracle, variant
from drawn import scan
from samples import INVOICES


@functools.cache
def pdf(name: str, kind: str = "orig") -> bytes:
    """A sample invoice displayed as `kind` (boxes_help.variant), built once per run."""
    return variant((INVOICES / f"{name}.pdf").read_bytes(), kind)


# --------------------------------------------------------------------------- #
# The frame of a page
# --------------------------------------------------------------------------- #

# A view that does not start at the origin: 100 wide, 200 tall, from (10, 20).
VIEW = (10.0, 20.0, 110.0, 220.0)


@pytest.mark.parametrize(
    ("rotation", "size", "point"),
    [
        # A user-space point (30, 50): 20 from the view's left, 50 from its right,
        # 30 above its bottom, 170 below its top.
        (0, (100.0, 200.0), (20.0, 170.0)),
        (90, (200.0, 100.0), (30.0, 20.0)),  # turned clockwise: the bottom edge is on the left
        (180, (100.0, 200.0), (80.0, 30.0)),
        (270, (200.0, 100.0), (170.0, 80.0)),  # the top edge is on the left
    ],
)
def test_frame_maps_user_space_onto_the_displayed_page(
    rotation: int, size: tuple[float, float], point: tuple[float, float]
) -> None:
    frame = PageFrame.of(VIEW, rotation)
    assert frame is not None
    assert (frame.width, frame.height) == size
    x0, y0, x1, y1 = frame.to_display(30.0, 50.0, 30.0, 50.0)
    assert (x0, y0) == (x1, y1) == point


def test_frame_turns_a_box_into_its_displayed_corners() -> None:
    frame = PageFrame.of(VIEW, 90)
    assert frame is not None
    # x from 30 to 40, y from 50 to 52: turned, x runs down the page and y across it.
    assert frame.to_display(30.0, 50.0, 40.0, 52.0) == (30.0, 20.0, 32.0, 30.0)


@pytest.mark.parametrize(
    ("view", "rotation"),
    [
        ((0.0, 0.0, 0.0, 0.0), 0),  # a crop box that misses the media box
        ((0.0, 0.0, 0.04, 100.0), 0),  # a side that rounds to 0
        ((0.0, 0.0, 100.0, 100.0), 45),
        ((0.0, 0.0, float("inf"), 100.0), 0),
        ((0.0, float("nan"), 100.0, 100.0), 90),
    ],
)
def test_no_frame_for_a_page_without_area_or_a_quarter_turn(
    view: tuple[float, float, float, float], rotation: int
) -> None:
    assert PageFrame.of(view, rotation) is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("at", range(4))
@pytest.mark.parametrize("cell", [True, False])
def test_a_box_with_a_coordinate_that_is_not_a_number_is_not_placed(
    bad: float, at: int, cell: bool
) -> None:
    frame = PageFrame.of((0.0, 0.0, 100.0, 100.0), 0)
    assert frame is not None
    box = [10.0, 10.0, 20.0, 20.0]
    box[at] = bad
    assert frame.place(*box, cell=cell) is None


def test_rectangles_are_rounded_outwards() -> None:
    frame = PageFrame.of((0.0, 0.0, 100.0, 100.0), 0)
    assert frame is not None
    # user space y 80.66 .. 87.66 is 12.34 .. 19.34 from the top
    assert frame.place(12.34, 80.66, 20.66, 87.66, cell=True) == (12.3, 12.3, 20.7, 19.4)
    # exact tenths stay where they are (0.7 * 10 is not exactly 7 in floating point)
    assert frame.place(0.7, 99.0, 2.3, 99.3, cell=True) == (0.7, 0.7, 2.3, 1.0)


def test_a_value_cut_by_the_edge_is_not_framed_a_row_is_clipped() -> None:
    frame = PageFrame.of((0.0, 0.0, 100.0, 100.0), 0)
    assert frame is not None
    # half a point out of the page is still on it, and is held to the page
    assert frame.place(-0.4, 10.0, 5.0, 20.0, cell=True) == (0.0, 80.0, 5.0, 90.0)
    assert frame.place(90.0, 10.0, 100.5, 20.0, cell=True) == (90.0, 80.0, 100.0, 90.0)
    # more than that: the value is not claimed, a row is cut to the page
    assert frame.place(-3.0, 10.0, 5.0, 20.0, cell=True) is None
    assert frame.place(-3.0, 10.0, 5.0, 20.0, cell=False) == (0.0, 80.0, 5.0, 90.0)
    assert frame.place(90.0, 95.0, 110.0, 130.0, cell=False) == (90.0, 0.0, 100.0, 5.0)
    # nothing of it left on the page, or no height at all
    assert frame.place(120.0, 10.0, 130.0, 20.0, cell=False) is None
    assert frame.place(10.0, 0.0, 20.0, 0.0, cell=False) is None


# --------------------------------------------------------------------------- #
# The frames of a PDF, against pdfium
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["s_classic", "m_grouped"])
@pytest.mark.parametrize("kind", VARIANTS)
def test_page_frames_are_the_pages_pdfium_displays(name: str, kind: str) -> None:
    data = pdf(name, kind)
    frames = boxes.page_frames(data)
    with Oracle(data) as oracle:
        assert len(frames) == len(oracle.pdf)
        for no, frame in enumerate(frames, start=1):
            assert frame is not None
            page = oracle.pdf[no - 1]
            w, h = oracle.size(no)
            assert (frame.width, frame.height) == (round(w, 1), round(h, 1))
            assert (frame.left, frame.bottom, frame.right, frame.top) == page.get_bbox()
            assert frame.rotation == page.get_rotation()
    if kind == "shift":
        assert frames[0] is not None and (frames[0].left, frames[0].bottom) == SHIFT


@pytest.mark.parametrize("kind", VARIANTS)
def test_frame_agrees_with_pdfium_on_word_corners(kind: str) -> None:
    """FPDF_PageToDevice (given no rotation of its own: pdfium adds the page's /Rotate)
    puts 200 corners of words where the frame does."""
    data = pdf("m_grouped", kind)
    frames = boxes.page_frames(data)
    words = ((tl.page, w) for page in lay.text_lines(data) for tl in page for w in tl.words)
    checked = 0
    with Oracle(data) as oracle:
        for page_no, w in islice(words, 0, None, 8):
            frame = frames[page_no - 1]
            assert frame is not None
            x0, y0, x1, y1 = frame.to_display(w.x0, w.bottom, w.x1, w.top)
            corners = [
                oracle.to_display(page_no, x, y) for x, y in ((w.x0, w.bottom), (w.x1, w.top))
            ]
            xs, ys = sorted(c[0] for c in corners), sorted(c[1] for c in corners)
            assert (x0, x1) == pytest.approx(tuple(xs), abs=0.05), (page_no, w)
            assert (y0, y1) == pytest.approx(tuple(ys), abs=0.05), (page_no, w)
            checked += 1
            if checked == 200:
                break
    assert checked == 200


def test_page_without_area_has_no_frame_and_is_logged_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    data = pdf("m_grouped", "crop_off")
    with caplog.at_level(logging.WARNING, logger=boxes.logger.name):
        frames = boxes.page_frames(data)
    assert frames == [None, None, None, None]
    assert len(caplog.records) == 1 and "[1, 2, 3, 4]" in caplog.records[0].getMessage()


def test_a_scan_has_frames() -> None:
    frames = boxes.page_frames(scan(2))
    assert [(f.width, f.height, f.rotation) if f else None for f in frames] == [
        (595.0, 842.0, 0),
        (595.0, 842.0, 0),
    ]
