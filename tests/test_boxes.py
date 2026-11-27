"""Where values are printed: the frame of a displayed page and the rectangles placed on
it. pdfium is the witness (tests/boxes_help.py), never the formulas under test."""

import functools
import io
import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import islice
from typing import Any

import pytest
from reportlab.pdfgen import canvas

from invoice_recognition.recognition import boxes, checks
from invoice_recognition.recognition import evidence as ev
from invoice_recognition.recognition import layout as lay
from invoice_recognition.recognition.boxes import PageFrame
from invoice_recognition.shapes import JSONObject, Line

from boxes_help import CROP, SHIFT, VARIANTS, Oracle, layout_of, variant
from drawn import scan
from fakes import make_pdf
from samples import FIXTURES, INVOICES, TEXT_LAYER, WITH_LAYOUT, saved_layout, truth_doc


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

# pdfium is asked about the rectangles of at most about this many lines of a document
# (every k-th line of a longer one): every rectangle is still checked for its place.
ORACLE_LINES = 100


def _asked(i: int, n: int) -> bool:
    return i % max(1, n // ORACLE_LINES) == 0


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
    pages = text_pages(name, kind)
    layout = layout_of(data, pages, saved_layout(name))
    parsed = lay.parse_document(pages, layout)
    frames = frames_of(name, kind)
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
                if key == "row" or not _asked(i, len(lines)):
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
    assert framed > 5 * min(len(lines), ORACLE_LINES) or kind == "crop"


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


def test_boxes_of_s_classic_are_pinned() -> None:
    """The rectangles of every line of s_classic, to the tenth of a point."""
    assert _snapshot("s_classic") == json.loads(SNAPSHOT.read_text())


# --------------------------------------------------------------------------- #
# Line items read by the model, found in the text layer afterwards
# --------------------------------------------------------------------------- #

MODEL_VARIANTS = ("orig", "rot90", "rot180", "rot270", "landscape")


def truth_lines(name: str) -> list[Line]:
    """The line items of a model that reads the document right, each with its page."""
    _, _, _, pages = truth_doc(name)
    return [row | {"page": p} for p, rows in pages.items() for row in rows]


def model_boxes(name: str, kind: str, lines: list[Line]) -> list[boxes.LineBoxes | None]:
    return boxes.locate(lines, text_pages(name, kind), frames_of(name, kind))


@functools.cache
def text_pages(name: str, kind: str) -> list[list[lay.TextLine]]:
    return lay.text_lines(pdf(name, kind))


@functools.cache
def frames_of(name: str, kind: str) -> list[PageFrame | None]:
    return boxes.page_frames(pdf(name, kind))


def _printed(text: str, key: str, item: Line) -> bool:
    """The text is a way the value is printed, compared as the presence checks do."""
    return ev.comparable(text) in boxes._targets(item)[key]


# The two largest samples are turned by 90 and 270 degrees only through the others: a
# quarter turn changes nothing for matching (see MISREAD_CASES).
READ_RIGHT_CASES = [
    (name, kind)
    for name in TEXT_LAYER
    for kind in MODEL_VARIANTS
    if kind not in ("rot90", "rot270") or name not in ("l_dense", "xl_grouped")
]


@pytest.mark.parametrize(("name", "kind"), READ_RIGHT_CASES)
def test_every_line_read_right_is_found_and_framed_on_its_printed_values(
    name: str, kind: str
) -> None:
    lines = truth_lines(name)
    found = model_boxes(name, kind, lines)
    framed = dict.fromkeys(boxes.LOCATED, 0)
    with Oracle(pdf(name, kind)) as oracle:
        for i, (item, b) in enumerate(zip(lines, found, strict=True)):
            assert b is not None, item
            assert tuple(b) == boxes.BOX_KEYS
            assert b["description"] is None and b["unit"] is None
            row = b["row"]
            assert row is not None
            for key in boxes.LOCATED:
                rect = b[key]
                if rect is None:
                    continue
                assert _inside(rect, row)
                framed[key] += 1
                if _asked(i, len(lines)):
                    text = oracle.text_under(item["page"], rect)
                    assert _printed(text, key, item), (key, text, item)
    # the values that are printed in every row of these samples are framed in nearly all
    for key in ("sku", "amount", "unit_price"):
        assert framed[key] >= len(lines) - 1, (key, framed)
    assert framed["quantity"] >= 0.95 * len(lines)


@pytest.mark.parametrize("name", TEXT_LAYER)
def test_a_landscape_page_shown_upright_is_framed_as_the_upright_page(name: str) -> None:
    """Drawn turned and shown upright by /Rotate 90, the page looks like the original: so
    must every rectangle. In user space every word of a row has the same X there; values
    matched by user-space X would frame the wrong cells."""
    lines = truth_lines(name)
    upright, turned = model_boxes(name, "orig", lines), model_boxes(name, "landscape", lines)
    for a, b in zip(upright, turned, strict=True):
        assert a is not None and b is not None
        for key in boxes.BOX_KEYS:
            ra, rb = a[key], b[key]
            assert (ra is None) == (rb is None), key
            if ra is not None and rb is not None:
                assert rb == pytest.approx(ra, abs=0.11), key


def _pairs(
    name: str, kind: str, lines: list[Line], found: list[boxes.LineBoxes | None]
) -> list[tuple[boxes.LineBoxes | None, boxes.LineBoxes | None, Line, lay.TextLine]]:
    """Lines the layout road found, paired with the same lines of `lines` by page, SKU and
    amount (in order, for rows printed twice): (layout boxes, model boxes, line, the item
    row the layout road read)."""
    code = by_code(name, kind)
    queue: dict[tuple[object, ...], list[int]] = {}
    for i, item in enumerate(lines):
        key = (item["page"], item["sku"], checks.dec(item["amount"]))
        queue.setdefault(key, []).append(i)
    out = []
    for item, b, rows in zip(code.parsed.lines, code.boxes, code.parsed.rows, strict=True):
        key = (item["page"], item["sku"], checks.dec(item["amount"]))
        if queue.get(key):
            i = queue[key].pop(0)
            out.append((b, found[i], lines[i], rows[0]))
    return out


def _row_rect(frame: PageFrame | None, row: lay.TextLine) -> boxes.Rect | None:
    assert frame is not None
    ws = row.words
    x0, y0 = min(w.x0 for w in ws), min(w.bottom for w in ws)
    return frame.place(x0, y0, max(w.x1 for w in ws), max(w.top for w in ws), cell=False)


@pytest.mark.parametrize("name", WITH_LAYOUT)
@pytest.mark.parametrize("kind", ["orig", "rot90", "rot180", "rot270"])
def test_lines_read_by_the_model_are_framed_where_code_frames_them(name: str, kind: str) -> None:
    """Every value the model road frames is the very cell the layout road frames, and its
    row is the printed row of the item the layout road read, and no other (the layout
    road's row also holds the wrapped description, which the model road does not look
    for)."""
    lines = truth_lines(name)
    pairs = _pairs(name, kind, lines, model_boxes(name, kind, lines))
    expected = len(by_code(name, kind).parsed.lines)
    assert len(pairs) == expected >= len(lines) - 1  # m_grouped turned 90: 109 of 110
    item_rows = _item_rows(name, kind)
    same = 0
    for code, model, item, row in pairs:
        assert code is not None and model is not None
        assert _rows_inside(model["row"], item_rows[item["page"]], 0.0) == [
            _row_rect(frames_of(name, kind)[item["page"] - 1], row)
        ], item
        for key in boxes.LOCATED:
            if model[key] is not None and code[key] is not None:
                assert model[key] == code[key], (key, item)
                same += 1
    assert same >= 4 * len(pairs)


def _item_rows(name: str, kind: str) -> dict[int, list[boxes.Rect | None]]:
    """The rectangles of the item rows the layout road read, by page."""
    code, frames = by_code(name, kind), frames_of(name, kind)
    out: dict[int, list[boxes.Rect | None]] = {}
    for item, rows in zip(code.parsed.lines, code.parsed.rows, strict=True):
        out.setdefault(item["page"], []).append(_row_rect(frames[item["page"] - 1], rows[0]))
    return out


def _rows_inside(
    outer: boxes.Rect | None, rows: list[boxes.Rect | None], tol: float
) -> list[boxes.Rect | None]:
    """The item rows that lie inside the rectangle (within tol)."""
    assert outer is not None
    return [
        r
        for r in rows
        if r is not None
        and r[0] >= outer[0] - tol
        and r[1] >= outer[1] - tol
        and r[2] <= outer[2] + tol
        and r[3] <= outer[3] + tol
    ]


def _misread(lines: list[Line], key: str) -> tuple[list[Line], list[int]]:
    """Every fifth line with its `key` misread; the indexes of those lines."""
    out, changed = [], []
    for i, item in enumerate(lines):
        if i % 5 == 0:
            wrong = item[key] + 1 if key == "amount" else f"{item[key]}X"
            item = item | {key: wrong}
            changed.append(i)
        out.append(item)
    return out, changed


@pytest.mark.parametrize("name", ["s_dense", "m_grouped", "l_classic"])
@pytest.mark.parametrize("key", ["amount", "sku"])
@pytest.mark.parametrize("kind", ["orig", "rot180", "landscape"])
def test_a_misread_value_is_framed_on_the_printed_cell(name: str, key: str, kind: str) -> None:
    """The row is still found by the other values, and the value read wrong is framed
    where its column is printed in that row: the cell a person must look at."""
    right = truth_lines(name)
    lines, changed = _misread(right, key)
    found = model_boxes(name, kind, lines)
    # landscape is displayed as the original page, so the layout road of the original
    # page says where each cell is
    pairs = {
        id(item): (code, model)
        for code, model, item, _ in _pairs(name, _code_kind(kind), right, found)
    }
    checked = 0
    with Oracle(pdf(name, kind)) as oracle:
        for i in changed:
            if id(right[i]) not in pairs:  # m_grouped turned 90: the line code misses
                continue
            code, model = pairs[id(right[i])]
            assert model is not None and code is not None, right[i]
            rect, want = model[key], code[key]
            assert rect is not None and want is not None, (key, right[i])
            assert rect == pytest.approx(want, abs=_tol(kind)), (key, right[i])
            text = oracle.text_under(right[i]["page"], rect)
            assert _printed(text, key, right[i]) and not _printed(text, key, lines[i])
            checked += 1
    assert checked >= len(changed) - 1
    assert all(b is not None for b in found)


def _code_kind(kind: str) -> str:
    """The variant whose layout road says where the cells of `kind` are displayed."""
    return "orig" if kind == "landscape" else kind


def _tol(kind: str) -> float:
    return 0.11 if kind == "landscape" else 0.0


# The ways a model misreads a line, as the review of the model road attacked it: every
# fifth line is changed so. None of them may frame a cell that is not the value's own.
MISREADS: dict[str, Callable[[Line], Line]] = {
    "amount digit": lambda it: it | {"amount": round(it["amount"] + 0.09, 2)},
    "unit price digit": lambda it: it | {"unit_price": round(it["unit_price"] + 0.01, 4)},
    "quantity digit": lambda it: it | {"quantity": it["quantity"] + 1},
    "tax rate digit": lambda it: it | {"tax_rate": (it["tax_rate"] or 0) + 1},
    "line number digit": lambda it: it | {"line_no": (it["line_no"] or 0) + 1},
    "sku letter": lambda it: it | {"sku": it["sku"] + "X"},
    "amount = unit price": lambda it: it | {"amount": it["unit_price"]},
    "unit price = amount": lambda it: it | {"unit_price": it["amount"]},
    "quantity = tax rate": lambda it: it | {"quantity": it["tax_rate"]},
    "quantity = unit price": lambda it: it | {"quantity": it["unit_price"]},
    "quantity and unit price swapped": lambda it: (
        it | {"quantity": it["unit_price"], "unit_price": it["quantity"]}
    ),
    "tax rate = quantity": lambda it: it | {"tax_rate": it["quantity"]},
    "sku and amount": lambda it: it | {"sku": it["sku"] + "X", "amount": it["amount"] + 1},
    "amount negated": lambda it: it | {"amount": -it["amount"]},
}


# Every family on the upright page and the page turned 180 degrees (where the displayed
# order runs against the reading order); three of them on the other variants. Matching
# works in user space along the text line, so a page turned by 90 or 270 degrees, or
# drawn turned (landscape), is matched exactly as the upright one: by construction they
# behave alike, and test_a_landscape_page_shown_upright_is_framed_as_the_upright_page
# pins that.
SAMPLED = ("amount = unit price", "quantity and unit price swapped", "sku and amount")
MISREAD_CASES = [(m, k) for m in MISREADS for k in ("orig", "rot180")] + [
    (m, k) for m in SAMPLED for k in ("rot90", "rot270", "landscape")
]


@pytest.mark.parametrize("name", ["s_dense", "m_grouped", "l_classic"])
@pytest.mark.parametrize(("misread", "kind"), MISREAD_CASES)
def test_a_misread_line_never_frames_another_cell(name: str, kind: str, misread: str) -> None:
    """Every rectangle the model road gives is the cell the layout road reads that value
    from (or no rectangle at all), however the line was misread. Rows printed alike are
    told apart by order only: such a line may take either of them (the known limit)."""
    right = truth_lines(name)
    lines = [MISREADS[misread](it) if i % 5 == 0 else it for i, it in enumerate(right)]
    found = model_boxes(name, kind, lines)
    code = by_code(name, _code_kind(kind))
    frames = frames_of(name, _code_kind(kind))
    item_rows = _item_rows(name, _code_kind(kind))
    cells: dict[tuple[object, ...], list[tuple[boxes.LineBoxes, boxes.Rect | None]]] = {}
    for item, b, rows in zip(code.parsed.lines, code.boxes, code.parsed.rows, strict=True):
        assert b is not None
        printed = (item["page"], item["sku"], checks.dec(item["amount"]))
        cells.setdefault(printed, []).append((b, _row_rect(frames[item["page"] - 1], rows[0])))
    framed = 0
    for item, b in zip(right, found, strict=True):
        twins = cells.get((item["page"], item["sku"], checks.dec(item["amount"])))
        if b is None or not twins:  # not located, or a line the layout road misses
            continue
        # the row framed holds the item row of exactly one printed line: this one, or its
        # twin printed alike
        inside = _rows_inside(b["row"], item_rows[item["page"]], _tol(kind))
        same_row = [c for c, row in twins if row in inside]
        assert len(inside) == 1 and len(same_row) == 1, (misread, item)
        for key in boxes.LOCATED:
            rect, want = b[key], same_row[0][key]
            if rect is not None:
                assert want is not None, (misread, key, item)
                assert rect == pytest.approx(want, abs=_tol(kind)), (misread, key, item)
                framed += 1
    assert sum(b is not None for b in found) == len(lines)
    assert framed >= 4 * len(lines)


def test_a_line_the_model_invented_does_not_take_the_row_of_a_real_one() -> None:
    """Before every real line, one that shares its quantity and unit price but nothing
    else: it scores two on the real row, the real line four. The real line keeps it."""
    right = truth_lines("m_grouped")
    lines: list[Line] = []
    for it in right:
        lines += [it | {"sku": "ZZ-00000", "amount": 1.23, "line_no": 999}, it]
    found = model_boxes("m_grouped", "orig", lines)
    assert found[0::2] == [None] * len(right)
    assert found[1::2] == model_boxes("m_grouped", "orig", right)


def test_a_misread_quantity_equal_to_the_number_of_a_twin_row_takes_that_row() -> None:
    """The known limit, pinned. l_classic prints FR-71160 twice on one page with the same
    values; read with its quantity one too high, the first of them matches the line number
    of the second as well, so it takes the second's row, and the second line the first."""
    right = truth_lines("l_classic")
    twins = [i for i, it in enumerate(right) if it["sku"] == "FR-71160"]
    assert len(twins) == 2 and right[twins[0]]["page"] == right[twins[1]]["page"]
    first, second = (right[i] for i in twins)
    assert first["quantity"] + 1 == second["line_no"]
    lines = list(right)
    lines[twins[0]] = first | {"quantity": first["quantity"] + 1}
    found = model_boxes("l_classic", "orig", lines)
    upright = model_boxes("l_classic", "orig", right)
    a, b = found[twins[0]], found[twins[1]]
    ua, ub = upright[twins[0]], upright[twins[1]]
    assert a is not None and b is not None and ua is not None and ub is not None
    assert (a["row"], b["row"]) == (ub["row"], ua["row"])
    assert a["row"] != b["row"]


# Rows drawn one under another in Helvetica, whose digits all have the same width: rows
# that differ only in their digits put every value at the same X, as a table does.
def _item(no: int, qty: object, price: object, amount: object, sku: str = "") -> Line:
    return {
        "line_no": no,
        "sku": sku or f"SKU-00{no}",
        "description": "Bolt",
        "quantity": qty,
        "unit": "pcs",
        "unit_price": price,
        "tax_rate": None,
        "amount": amount,
        "page": 1,
    }


def _drawn(rows: list[str], lines: list[Line]) -> tuple[list[boxes.LineBoxes | None], Oracle]:
    data = make_pdf([rows])
    return boxes.locate(lines, lay.text_lines(data), boxes.page_frames(data)), Oracle(data)


def _texts(found: boxes.LineBoxes | None, oracle: Oracle) -> dict[str, str | None]:
    assert found is not None
    return {k: None if r is None else oracle.text_under(1, r) for k, r in found.items()}


def test_a_value_printed_twice_in_its_row_is_told_apart_by_its_column() -> None:
    rows = ["1 SKU-001 Bolt 1 10.00 10.00", "2 SKU-002 Bolt 3 10.00 30.00"]
    lines = [_item(1, 1, 10, 10), _item(2, 3, 10, 30)]
    found, oracle = _drawn([*rows, "3 SKU-003 Bolt 4 10.00 40.00"], [*lines, _item(3, 4, 10, 40)])
    with oracle:
        first = _texts(found[0], oracle)
        assert first | {"row": None} == {
            "row": None,
            "line_no": "1",
            "sku": "SKU-001",
            "description": None,
            "quantity": "1",
            "unit": None,
            "unit_price": "10.00",
            "tax_rate": None,
            "amount": "10.00",
        }
        b = found[0]
        assert b is not None and b["line_no"] is not None and b["quantity"] is not None
        assert (
            b["line_no"][2] < b["quantity"][0]
        )  # the first "1" is the number, the second the quantity
    # Two rows: only one of them gives the columns of the line number and the quantity.
    found, oracle = _drawn(rows, lines)
    with oracle:
        first = _texts(found[0], oracle)
    assert first["line_no"] is None and first["quantity"] is None
    assert (first["sku"], first["unit_price"], first["amount"]) == ("SKU-001", "10.00", "10.00")


def test_a_number_inside_the_description_of_a_few_rows_is_no_column() -> None:
    """A table without line numbers; the model gives two lines the number 1, which is
    printed in their description ("vol. 1"), at the same place in both rows. Two matches
    of six rows do not make a column: the number is not framed."""
    rows = [f"SKU-00{n} Manual vol. 1 {n + 1} 10.00 {n + 1}0.00" for n in range(1, 7)]
    lines = [_item(n, n + 1, 10, (n + 1) * 10) | {"line_no": None} for n in range(1, 7)]
    lines[0] |= {"line_no": 1}
    lines[3] |= {"line_no": 1}
    found, oracle = _drawn(rows, lines)
    oracle.close()
    assert all(b is not None and b["line_no"] is None for b in found)
    assert all(b is not None and b["quantity"] is not None for b in found)


def test_numbers_printed_in_ways_no_printed_form_matches() -> None:
    """Known limits: a thousands separator printed as a space makes two words, and a minus
    printed after the number is not among the printed forms of a value. Neither is ever
    matched exactly; a column of the page frames such a cell as a whole."""
    spaced = ["1 SKU-001 Bolt 2 617,28 1 234,56"]
    others = ["2 SKU-002 Bolt 3 10,00 30,00", "3 SKU-003 Bolt 4 10,00 40,00"]
    lines = [_item(1, 2, 617.28, 1234.56), _item(2, 3, 10, 30), _item(3, 4, 10, 40)]
    found, oracle = _drawn(spaced, lines[:1])
    with oracle:
        assert _texts(found[0], oracle)["amount"] is None  # the row is found, the amount is not
    found, oracle = _drawn(spaced + others, lines)
    with oracle:
        assert _texts(found[0], oracle)["amount"] == "1 234,56"
    found, oracle = _drawn(["1 SKU-001 Bolt 1 12.00 12.00-"], [_item(1, 1, 12, -12)])
    with oracle:
        texts = _texts(found[0], oracle)
    # one row in the document: no column, only the sku and an amount printed last count
    assert texts["amount"] is None and texts["sku"] == "SKU-001"


def test_rows_printed_alike_are_taken_in_reading_order() -> None:
    """Two rows printed alike cannot be told apart: they go to the lines in the order the
    page is read, the next one after the row of the previous line first."""
    rows = ["1 SKU-001 Bolt 2 10.00 20.00", "2 SKU-002 Bolt 3 10.00 30.00"]
    rows.append(rows[0])
    same = _item(1, 2, 10, 20)
    found, oracle = _drawn(rows, [_item(2, 3, 10, 30), same, same])
    oracle.close()
    tops = [b["row"][1] if b and b["row"] else None for b in found]
    assert tops[0] is not None and tops[1] is not None and tops[2] is not None
    assert tops[2] < tops[0] < tops[1]  # 2nd printed row, then the 3rd (after it), then the 1st


def test_lines_not_printed_or_without_a_page_are_not_found() -> None:
    rows = ["1 SKU-001 Bolt 2 10.00 20.00", "2 SKU-002 Bolt 3 10.00 30.00"]
    page_of = [None, "1", 1.0, True, 0, 2, -1]
    lines = [_item(9, 7, 99.5, 696.5, "SKU-999")] + [
        _item(1, 2, 10, 20) | {"page": p} for p in page_of
    ]
    found, oracle = _drawn(rows, lines)
    oracle.close()
    assert found == [None] * len(lines)
    # only one value of the line is printed in any row: not enough to take a row
    found, oracle = _drawn(rows, [_item(1, 2, 11, 22, "SKU-777")])
    oracle.close()
    assert found == [None]


HOSTILE: list[object] = [{}, [1], "1e999999", True, None, "9" * 10_000, "x" * 10_000, -0.0]


@pytest.mark.parametrize("value", HOSTILE)
def test_values_of_any_json_type_are_taken_calmly(value: object) -> None:
    rows = ["1 SKU-001 Bolt 2 10.00 20.00", "2 SKU-002 Bolt 3 10.00 30.00"]
    lines: list[Any] = [{k: value for k in (*lay.FIELDS, "page")}, value]
    lines += [_item(1, 2, 10, 20) | {k: value} for k in (*lay.FIELDS, "page")]
    found, oracle = _drawn(rows, lines)
    oracle.close()
    assert len(found) == len(lines)


def test_a_crowded_page_is_located_quickly() -> None:
    """5,000 lines on one page of xl_grouped: the work is bounded by lines x rows."""
    lines = [item for item in truth_lines("xl_grouped") if item["page"] == 1]
    many = (lines * (5000 // len(lines) + 1))[:5000]
    pages, frames = text_pages("xl_grouped", "orig"), frames_of("xl_grouped", "orig")
    started = time.perf_counter()
    found = boxes.locate(many, pages, frames)
    # About 0.2 s on a laptop; the bound only catches work that grows as lines x rows x
    # words, not a slow machine.
    assert time.perf_counter() - started < 5.0
    assert sum(b is not None for b in found) == len(lines)  # each printed row is used once


# --------------------------------------------------------------------------- #
# A page with one line, numbers printed with spaces, rows printed alike
# --------------------------------------------------------------------------- #


def _drawn_as(
    kind: str, rows: list[str], lines: list[Line]
) -> tuple[list[boxes.LineBoxes | None], Oracle]:
    data = variant(make_pdf([rows]), kind)
    return boxes.locate(lines, lay.text_lines(data), boxes.page_frames(data)), Oracle(data)


ONE_LINE = ["Invoice 2024-11 total 20.00", "1 S-1 Bolt 2 10.00 20.00", "Total 20.00"]


@pytest.mark.parametrize("kind", ["orig", "rot180", "landscape"])
@pytest.mark.parametrize(
    ("rows", "line", "framed"),
    [
        # read right: only what stands on its own is framed, the amount printed last
        (ONE_LINE, _item(1, 2, 10, 20, "S-1"), {"sku": "S-1", "amount": "20.00"}),
        # the amount read as the unit price: not framed on the price
        (ONE_LINE, _item(1, 2, 10, 10, "S-1"), {"sku": "S-1"}),
        # the quantity read as 1: not framed on the line number
        (ONE_LINE, _item(1, 1, 10, 20, "S-1"), {"sku": "S-1", "amount": "20.00"}),
        # a tax rate of 0 read as 1: not framed on the "1" of the description
        (
            ["FR-1 Export freight zone 1 3 10.00 30.00 0%"],
            _item(1, 3, 10, 30, "FR-1") | {"line_no": None, "tax_rate": 1},
            {"sku": "FR-1", "amount": "30.00"},
        ),
    ],
)
def test_a_line_alone_in_its_document_frames_only_what_stands_on_its_own(
    kind: str, rows: list[str], line: Line, framed: dict[str, str]
) -> None:
    """No other row gives a column to check a match against: a quantity, a price, a tax
    rate or a line number could be any number of the row, and is not framed."""
    found, oracle = _drawn_as(kind, rows, [line])
    with oracle:
        texts = _texts(found[0], oracle)
    assert texts["row"] is not None
    assert {k: v for k, v in texts.items() if v is not None and k != "row"} == framed


def test_a_page_with_one_line_borrows_the_columns_of_the_document() -> None:
    """m_grouped read with only the first line of its last page, its amount read as its
    unit price: that page has one located row, the columns of page 3 check it."""
    right = truth_lines("m_grouped")
    last = next(i for i, item in enumerate(right) if item["page"] == 4)
    lines = right[:last] + [right[last] | {"amount": right[last]["unit_price"]}]
    found = model_boxes("m_grouped", "rot180", lines)
    code = next(
        b
        for b, item in zip(
            by_code("m_grouped", "rot180").boxes, by_code("m_grouped", "rot180").parsed.lines
        )
        if item["page"] == 4
    )
    b = found[-1]
    assert b is not None and code is not None
    assert {k: b[k] for k in LOCATED_KEYS} == {k: code[k] for k in LOCATED_KEYS}
    assert b["amount"] is not None and b["unit_price"] is not None


LOCATED_KEYS = ("sku", "amount", "unit_price", "quantity", "tax_rate")


def _table(rows: Sequence[tuple[str, ...]], sep: str = " ") -> bytes:
    """A table drawn by column, numbers right-aligned, as invoices print one. In a number,
    " " between digit groups is drawn as `sep`: a space, a no-break space, or "thin" - the
    groups drawn apart by 1.5 points, with no character between them."""
    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=(595, 842), invariant=True)
    pdf.setFont("Helvetica", 9)
    xs = [(40, "l"), (60, "l"), (110, "l"), (330, "r"), (420, "r"), (520, "r")]
    for n, cells in enumerate(rows):
        y = 782 - 14 * n
        for (x, align), text in zip(xs, cells, strict=True):
            if align == "l":
                pdf.drawString(x, y, text)
            elif sep != "thin":
                pdf.drawRightString(x, y, text.replace(" ", sep))
            else:  # each group on its own, the last one ending at x
                at: float = x
                for group in reversed(text.split(" ")):
                    pdf.drawRightString(at, y, group)
                    at -= pdf.stringWidth(group, "Helvetica", 9) + 1.5
    pdf.showPage()
    pdf.save()
    return buf.getvalue()


SPACED_ROWS = [
    ("1", "A-1", "Bolt", "2", "617,28", "1 234,56"),
    ("2", "A-2", "Bolt", "3", "10,00", "30,00"),
    ("3", "A-3", "Bolt", "4", "10,00", "40,00"),
    ("4", "A-4", "Bolt", "5", "1 000,00", "5 000,00"),
]
SPACED_LINES = [
    _item(1, 2, 617.28, 1234.56, "A-1"),
    _item(2, 3, 10, 30, "A-2"),
    _item(3, 4, 10, 40, "A-3"),
    _item(4, 5, 1000, 5000, "A-4"),
]


@pytest.mark.parametrize("sep", [" ", " ", "thin"])
@pytest.mark.parametrize("kind", ["orig", "rot180", "landscape"])
@pytest.mark.parametrize("misread", [False, True])
def test_a_number_printed_with_spaces_is_framed_whole(sep: str, kind: str, misread: bool) -> None:
    """ "1 234,56" is two words of the text layer. Matched or misread, the value is framed
    as the whole printed number, never as a part of it, and never with the next one."""
    lines: list[Line] = [dict(item) for item in SPACED_LINES]
    if misread:
        lines[0]["amount"], lines[3]["unit_price"] = 1234.65, 1000.5
    data = variant(_table(SPACED_ROWS, sep), kind)
    found = boxes.locate(lines, lay.text_lines(data), boxes.page_frames(data))
    with Oracle(data) as oracle:
        got = [_texts(b, oracle) for b in found]

    def chars(text: str | None) -> str:  # pdfium may give the groups in either order
        return "".join(sorted((text or "").replace(" ", "")))

    assert [(chars(t["unit_price"]), chars(t["amount"])) for t in got] == [
        (chars(r[4]), chars(r[5])) for r in SPACED_ROWS
    ]
    assert [t["sku"] for t in got] == ["A-1", "A-2", "A-3", "A-4"]


def test_rows_printed_alike_are_told_apart_by_their_line_numbers() -> None:
    """Two rows printed alike but for their number; read in the other order, each line
    still takes the row of its own number."""
    rows = [
        "1 SKU-001 Bolt 2 10.00 20.00",
        "2 SKU-002 Nut 3 10.00 30.00",
        "3 SKU-001 Bolt 2 10.00 20.00",
    ]
    first, third = _item(1, 2, 10, 20), _item(3, 2, 10, 20, "SKU-001")
    found, oracle = _drawn(rows, [third, _item(2, 3, 10, 30), first])
    with oracle:
        starts = [(_texts(b, oracle)["row"] or "")[:1] for b in found]
    assert starts == ["3", "2", "1"]


@pytest.mark.parametrize("kind", ["orig", "rot180"])
def test_two_numbers_printed_with_spaces_side_by_side_are_told_apart(kind: str) -> None:
    """Rows drawn as one string: "1 000,00 5 000,00" is four words, and the amount column
    of the other rows reaches over "000,00 5". Each value is framed as its own whole
    number; words of two numbers are never framed as one cell."""
    rows = [
        "1 A-1 Bolt 2 617,28 1 234,56",
        "2 A-2 Bolt 3 10,00 30,00",
        "3 A-3 Bolt 4 10,00 40,00",
        "4 A-4 Bolt 5 1 000,00 5 000,00",
    ]
    lines = [dict(item) for item in SPACED_LINES]
    found, oracle = _drawn_as(kind, rows, lines)
    with oracle:
        got = [_texts(b, oracle) for b in found]
    assert [(t["unit_price"], t["amount"]) for t in got] == [
        ("617,28", "1 234,56"),
        ("10,00", "30,00"),
        ("10,00", "40,00"),
        ("1 000,00", "5 000,00"),
    ]
    # misread: the amount is framed by its column, which takes two numbers' words here
    lines[3] = lines[3] | {"amount": 5000.5}
    found, oracle = _drawn_as(kind, rows, lines)
    with oracle:
        last = _texts(found[3], oracle)
    assert last["unit_price"] == "1 000,00" and last["amount"] in (None, "5 000,00")
