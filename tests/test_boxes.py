"""Where values are printed: the frame of a displayed page and the rectangles placed on
it. pdfium is the witness (tests/boxes_help.py), never the formulas under test."""

import functools
import json
import logging
from dataclasses import dataclass
from itertools import islice

import pytest

from invoice_recognition.recognition import boxes
from invoice_recognition.recognition import layout as lay
from invoice_recognition.recognition.boxes import PageFrame
from invoice_recognition.shapes import JSONObject

from boxes_help import CROP, SHIFT, VARIANTS, Oracle, layout_of, variant
from drawn import scan
from samples import FIXTURES, INVOICES, WITH_LAYOUT, saved_layout


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


# --------------------------------------------------------------------------- #
# Line items found by code
# --------------------------------------------------------------------------- #

LAYOUT_VARIANTS = ("orig", "rot90", "rot180", "rot270", "shift", "crop")


@dataclass
class ByCode:
    data: bytes
    layout: lay.Layout
    parsed: lay.Parsed
    frames: list[PageFrame | None]
    boxes: list[boxes.LineBoxes | None]


@functools.cache
def by_code(name: str, kind: str = "orig") -> ByCode:
    """The layout road on a sample displayed as `kind`, with the recorded layout answer."""
    data = pdf(name, kind)
    pages = lay.text_lines(data)
    layout = layout_of(data, pages, saved_layout(name))
    parsed = lay.parse_document(pages, layout)
    frames = boxes.page_frames(data)
    return ByCode(data, layout, parsed, frames, boxes.from_layout(parsed, layout, frames))


def _value_of(text: str, key: str, layout: lay.Layout) -> object:
    """A value as the layout parser reads it from the text of its cell."""
    if key in ("sku", "unit"):
        return text
    if key == "line_no":
        return int(text.rstrip("."))
    return layout.number(text.replace("%", "").strip() if key == "tax_rate" else text)


def _inside(inner: boxes.Rect, outer: boxes.Rect) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


@pytest.mark.parametrize("name", WITH_LAYOUT)
@pytest.mark.parametrize("kind", LAYOUT_VARIANTS)
def test_every_cell_found_by_code_frames_its_own_text(name: str, kind: str) -> None:
    got = by_code(name, kind)
    lines = got.parsed.lines
    assert len(got.boxes) == len(lines) > 10
    framed = 0
    last = {item["page"]: i for i, item in enumerate(lines)}
    with Oracle(got.data) as oracle:
        for i, (item, found) in enumerate(zip(lines, got.boxes)):
            if found is None:
                assert kind == "crop"  # only a crop box hides whole lines
                continue
            assert tuple(found) == boxes.BOX_KEYS and found["row"] is not None
            frame = got.frames[item["page"] - 1]
            assert frame is not None
            for key, rect in found.items():
                if rect is None:
                    continue
                x0, y0, x1, y1 = rect
                assert 0 <= x0 < x1 <= frame.width and 0 <= y0 < y1 <= frame.height
                assert _inside(rect, found["row"]), (key, item)
                if key == "row":
                    continue
                text = oracle.text_under(item["page"], rect)
                if key == "description":
                    # all of it, wrapped rows included; the last line of a page may go on
                    # at the top of the next one
                    whole = " ".join(item["description"].split())
                    assert whole == text or (i == last[item["page"]] and whole.startswith(text))
                else:
                    assert _value_of(text, key, got.layout) == item[key], (key, text, item)
                framed += 1
            if kind != "crop":
                for key in ("sku", "description", "quantity", "unit_price", "amount"):
                    if item[key] not in (None, ""):
                        assert found[key] is not None, (key, item)
    assert framed > 5 * len(lines) or kind == "crop"


@pytest.mark.parametrize("name", ["s_dense", "m_grouped"])
def test_crop_box_hides_values_outside_the_displayed_page(name: str) -> None:
    """On the crop variant the page is the original one moved by the cut, and a value that
    is not wholly on what is left of it is not framed. Lines whose rows are all cut off
    have no boxes."""
    whole, cropped = by_code(name), by_code(name, "crop")
    left, _, _, top = CROP
    assert cropped.parsed.lines == whole.parsed.lines
    hidden = partly = 0
    for page_item, a, b in zip(whole.parsed.lines, whole.boxes, cropped.boxes):
        frame = cropped.frames[page_item["page"] - 1]
        assert a is not None and frame is not None
        page = (0.0, 0.0, frame.width, frame.height)
        moved = {
            k: None if r is None else (r[0] - left, r[1] - top, r[2] - left, r[3] - top)
            for k, r in a.items()
        }
        row = moved["row"]
        assert row is not None
        if row[2] <= 0 or row[3] <= 0 or row[0] >= frame.width or row[1] >= frame.height:
            assert b is None
            hidden += 1
            continue
        assert b is not None
        for key, r in moved.items():
            if key == "row" or r is None:
                assert key == "row" or b[key] is None
                continue
            near_edge = any(abs(v) < 0.6 for v in (r[0], r[1], r[2] - page[2], r[3] - page[3]))
            if near_edge:
                continue
            if _inside(r, page):
                assert b[key] == pytest.approx(r, abs=1e-6), (key, page_item)
            else:
                assert b[key] is None, (key, page_item)
                partly += 1
    assert hidden and partly


def test_tax_rate_printed_once_for_the_document_is_not_framed() -> None:
    data = pdf("s_classic")
    pages = lay.text_lines(data)
    answer = saved_layout("s_classic") | {"tax": {"in_row": False, "document_rate": 20}}
    layout = layout_of(data, pages, answer)
    parsed = lay.parse_document(pages, layout)
    found = boxes.from_layout(parsed, layout, boxes.page_frames(data))
    assert all(item["tax_rate"] == 20 for item in parsed.lines)
    assert all(b is not None and b["tax_rate"] is None and b["amount"] for b in found)


def test_a_discount_row_has_its_amount_and_description_framed() -> None:
    got = by_code("m_grouped")
    marker = got.layout.discount_marker
    assert marker
    rows = [
        (item, b)
        for item, b in zip(got.parsed.lines, got.boxes)
        if marker in item["description"].casefold()
    ]
    assert rows
    for item, b in rows:
        assert item["amount"] < 0
        assert b is not None and b["amount"] is not None and b["description"] is not None


def test_a_line_on_a_page_without_frame_has_no_boxes() -> None:
    got = by_code("m_grouped")
    frames: list[PageFrame | None] = [got.frames[0], None, *got.frames[2:]]
    found = boxes.from_layout(got.parsed, got.layout, frames)
    assert [b is None for b in found] == [item["page"] == 2 for item in got.parsed.lines]
    assert boxes.from_layout(got.parsed, got.layout, []) == [None] * len(got.parsed.lines)


SNAPSHOT = FIXTURES / "boxes" / "s_classic.layout.json"


def _snapshot(name: str) -> list[JSONObject]:
    got = by_code(name)
    return [
        {
            "sku": item["sku"],
            "page": item["page"],
            "boxes": None
            if b is None
            else {k: None if r is None else list(r) for k, r in b.items()},
        }
        for item, b in zip(got.parsed.lines, got.boxes)
    ]


# Where a word starts and ends along its line comes from pdfium's loose character boxes,
# and for a font the PDF does not embed (s_classic's Helvetica) pdfium measures them on
# a substitute that differs between platforms: on the seven samples, 7 to 10 % of the
# words end up to 0.88 point apart between macOS and Linux. Across the line the band is
# a fixed rule (layout.ASCENT, DESCENT) and comes out the same everywhere.
X_TOL_PLATFORMS = 1.0


def test_boxes_of_s_classic_are_pinned() -> None:
    """The rectangles of every line of s_classic: top and bottom to the tenth of a point,
    left and right within X_TOL_PLATFORMS."""
    got, pinned = _snapshot("s_classic"), json.loads(SNAPSHOT.read_text())
    assert [(g["sku"], g["page"]) for g in got] == [(p["sku"], p["page"]) for p in pinned]
    for g, p in zip(got, pinned, strict=True):
        assert (g["boxes"] is None) == (p["boxes"] is None)
        if g["boxes"] is None:
            continue
        assert list(g["boxes"]) == list(p["boxes"])
        for key, rect in g["boxes"].items():
            want = p["boxes"][key]
            assert (rect is None) == (want is None), (g["sku"], key)
            if rect is not None:
                assert (rect[1], rect[3]) == (want[1], want[3]), (g["sku"], key)
                assert (rect[0], rect[2]) == pytest.approx(
                    (want[0], want[2]), abs=X_TOL_PLATFORMS
                ), (g["sku"], key)
