"""A page of a text-layer PDF shown turned reads as the upright page.

pdfium builds its text lines in display space, which includes the page's /Rotate: on an
upright page shown turned by /Rotate 90 the lines of a wrapped description came out of
order (xl_grouped: 384 of 1,250 descriptions wrong, the document still "ok"), on /Rotate
180 every row read right to left (no item row found: every page read again by the
model). evidence.upright_textpage turns the page so that its text reads upright before
the text is taken. Fake OpenRouter, no network."""

import functools
import io

import pypdfium2 as pdfium
import pytest
from reportlab.pdfgen import canvas

from invoice_recognition.recognition import boxes as bx
from invoice_recognition.recognition import evidence as ev
from invoice_recognition.recognition import layout as lay
from invoice_recognition.recognition.results import Result
from invoice_recognition.shapes import JSONObject

from boxes_help import TURNED, layout_of, sample, sample_text_lines, variant
from fakes import run
from golden_cases import Model, result_of
from samples import TEXT_LAYER, WITH_LAYOUT, saved_layout, truth_doc

PDF = "application/pdf"
ROTATED = ("rot90", "rot180", "rot270")  # the content upright, /Rotate set
# Where a page keeps its rotation, or its content is drawn turned, the two largest samples
# are left out: they add nothing a turn could break, only time.
SMALLER = tuple(n for n in TEXT_LAYER if n not in ("l_dense", "xl_grouped"))


def pdf(name: str, kind: str = "orig") -> bytes:
    return sample(name, kind)


def text_pages(name: str, kind: str = "orig") -> list[list[lay.TextLine]]:
    """Shared with test_boxes (boxes_help.sample_text_lines): not to be changed."""
    return sample_text_lines(name, kind)


@functools.cache
def texts(name: str, kind: str = "orig") -> list[str]:
    return ev.pdf_text_layer(pdf(name, kind))


@functools.cache
def parsed(name: str, kind: str = "orig") -> lay.Parsed:
    """The layout road on the sample displayed as `kind`, with the recorded layout answer."""
    data, pages = pdf(name, kind), text_pages(name, kind)
    return lay.parse_document(pages, layout_of(data, pages, saved_layout(name)))


def plain_text_lines(data: bytes, monkeypatch: pytest.MonkeyPatch) -> list[list[lay.TextLine]]:
    """text_lines as it was before upright_textpage: pdfium's text page of the page as it
    is."""
    with monkeypatch.context() as m:
        m.setattr(ev, "upright_textpage", lambda page: page.get_textpage())
        return lay.text_lines(data)


def rotations(data: bytes) -> list[int]:
    """The rotation of every page after upright_textpage has seen it."""
    doc = pdfium.PdfDocument(data)
    try:
        out = []
        for page in doc:
            ev.upright_textpage(page).close()
            out.append(page.get_rotation())
        return out
    finally:
        doc.close()


# --------------------------------------------------------------------------- #
# The text
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", TEXT_LAYER)
@pytest.mark.parametrize("kind", ROTATED)
def test_a_page_shown_turned_has_the_text_of_the_upright_page(name: str, kind: str) -> None:
    """Every word, its coordinates (user space either way), its line and the line's index;
    the evidence's text and its item rows."""
    assert text_pages(name, kind) == text_pages(name)
    assert texts(name, kind) == texts(name)
    assert ev.build(texts(name, kind), "text_layer") == ev.build(texts(name), "text_layer")


@pytest.mark.parametrize("name", SMALLER)
@pytest.mark.parametrize("kind", TURNED)
def test_content_drawn_turned_without_rotate_reads_as_the_upright_page(
    name: str, kind: str
) -> None:
    """The content itself turned, no /Rotate: the same text, lines and item rows; the
    coordinates are those of the turned drawing."""
    assert texts(name, kind) == texts(name)
    assert [[(tl.index, tl.text) for tl in p] for p in text_pages(name, kind)] == [
        [(tl.index, tl.text) for tl in p] for p in text_pages(name)
    ]


@pytest.mark.parametrize("name", WITH_LAYOUT)
@pytest.mark.parametrize("kind", ROTATED)
def test_the_layout_road_reads_the_upright_lines_on_a_page_shown_turned(
    name: str, kind: str
) -> None:
    turned, upright = parsed(name, kind), parsed(name)
    assert turned.lines == upright.lines
    assert turned.checkpoints == upright.checkpoints
    assert turned.notes == upright.notes
    assert turned.rows == upright.rows


def test_the_description_that_went_to_the_wrong_line_is_read_right() -> None:
    """xl_grouped turned by 90, page 1: SV-30179 took the description of SV-39404 after its
    own, and SV-39404 was read with none."""
    _, truth, _, _ = truth_doc("xl_grouped")
    want = {x["sku"]: x["description"] for x in truth["lines"] if x["page"] == 1}
    got = {
        x["sku"]: x["description"] for x in parsed("xl_grouped", "rot90").lines if x["page"] == 1
    }
    for sku in ("SV-30179", "SV-39404"):
        assert " ".join(got[sku].split()) == " ".join(want[sku].split()), sku


# --------------------------------------------------------------------------- #
# Pages that keep their rotation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", SMALLER)
@pytest.mark.parametrize(("kind", "rotation"), [("orig", 0), ("landscape", 90)])
def test_a_page_that_reads_upright_keeps_its_rotation_and_its_text(
    name: str, kind: str, rotation: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An upright page, and a page drawn turned and shown upright by its /Rotate."""
    data = pdf(name, kind)
    assert set(rotations(data)) == {rotation}
    assert text_pages(name, kind) == plain_text_lines(data, monkeypatch)


@pytest.mark.parametrize("name", [n for n in SMALLER if n in WITH_LAYOUT])
def test_the_layout_road_still_refuses_a_page_drawn_turned_and_shown_upright(name: str) -> None:
    """Unchanged by upright_textpage: on `landscape` every word of a row has the same
    user-space X, the layout parser finds nothing under the headings and the model reads
    the document (test_pipeline_boxes). Reading it by code is a separate change."""
    data, pages = pdf(name, "landscape"), text_pages(name, "landscape")
    with pytest.raises(lay.LayoutError):
        lay.parse_document(pages, layout_of(data, pages, saved_layout(name)))


def _with_sideways_margin(rows: list[str], margin: str) -> bytes:
    """An upright page of `rows`, and `margin` printed up the left edge in small type, in
    lines of 175 characters: legal small print, a stamp."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(595, 842), invariant=True)
    c.setFont("Helvetica", 9)
    for n, row in enumerate(rows):
        c.drawString(60, 782 - 14 * n, row)
    c.saveState()
    c.translate(20, 40)
    c.rotate(90)
    c.setFont("Helvetica", 5)
    for n in range(0, len(margin), 175):
        c.drawString(0, -6 * (n // 175), margin[n : n + 175])
    c.restoreState()
    c.showPage()
    c.save()
    return buf.getvalue()


def test_a_sideways_margin_longer_than_the_page_text_does_not_turn_the_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """More characters run up the margin than across the page, but not UPRIGHT_SHARE of
    them: the page keeps its rotation and pdfium's own text."""
    rows = [
        "ACME Ltd   Invoice INV-1   2026-11-01   Payment within 14 days",
        "No  SKU  Description  Qty  Price  Amount",
        *(f"{n}  SKU-{n:03d}  Hex bolt M{n} zinc plated  {n}  10.00  {n * 10}.00" for n in (1, 2)),
        "Subtotal 30.00   Tax 20% 6.00   Total 36.00",
        "Bank: Example Bank, IBAN XX00 0000 0000 0000, BIC EXAMPLEXX",
    ]
    margin = "Example Supplies Ltd - registered in Exampletown No 123456 - VAT XX123456789 - " * 7
    data = _with_sideways_margin(rows, margin)
    doc = pdfium.PdfDocument(data)
    try:
        page = doc[0]
        tp = page.get_textpage()
        plain = tp.get_text_range()
        votes = ev._turn_votes(tp)
        tp.close()
        # the margin outvotes the page's own text (a simple majority would turn the page),
        # but holds less than UPRIGHT_SHARE of the votes
        assert set(votes) == {0, 90}
        assert votes[0] < votes[90] < ev.UPRIGHT_SHARE * votes.total()
        tp = ev.upright_textpage(page)
        assert page.get_rotation() == 0 and tp.get_text_range() == plain
        tp.close()
    finally:
        doc.close()
    assert ev.pdf_text_layer(data) == [plain]
    assert lay.text_lines(data) == plain_text_lines(data, monkeypatch)


def _short_cells() -> bytes:
    """A page of two-character cells, each drawn on its own: pdfium inserts a space
    between every two cells, a third of the page's characters."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(595, 842), invariant=True)
    c.setFont("Helvetica", 9)
    for row in range(20):
        for col in range(16):
            c.drawString(40 + 32 * col, 780 - 14 * row, f"{(row * 16 + col) % 100:02d}")
    c.showPage()
    c.save()
    return buf.getvalue()


@pytest.mark.parametrize("kind", TURNED)
def test_short_cells_drawn_turned_turn_the_page(kind: str) -> None:
    """Only printed characters vote: the spaces pdfium inserts carry the identity matrix
    whatever the text around them, and counted, they kept this page from turning
    ({90: 640, 0: 338}, 0.65 of the votes)."""
    upright = _short_cells()
    data = variant(upright, kind)
    doc = pdfium.PdfDocument(data)
    try:
        tp = doc[0].get_textpage()
        assert set(ev._turn_votes(tp)) == {int(kind[6:])}
        tp.close()
    finally:
        doc.close()
    assert rotations(data) == [int(kind[6:])]
    assert ev.pdf_text_layer(data) == ev.pdf_text_layer(upright)


def test_a_page_without_characters_keeps_its_rotation() -> None:
    doc = pdfium.PdfDocument.new()
    try:
        page = doc.new_page(595, 842)
        page.set_rotation(90)
        tp = ev.upright_textpage(page)
        assert tp.count_chars() == 0 and page.get_rotation() == 90
        tp.close()
    finally:
        doc.close()


# --------------------------------------------------------------------------- #
# The whole flow
# --------------------------------------------------------------------------- #


@functools.cache
def flow(name: str, kind: str) -> tuple[Result, Model]:
    """process() on the sample displayed as `kind`, with a model that reads every page
    right and the recorded layout answer."""
    _, _, header, pages = truth_doc(name)
    model = Model([header], {p: [rows] for p, rows in pages.items()}, saved_layout(name))
    return run(pdf(name, kind), PDF, model), model


def _layout_prompt(model: Model) -> list[str]:
    """The text of the layout request: the text lines the model is shown."""
    (body,) = [
        b
        for b in model.bodies
        if b["messages"][0]["content"][0]["text"].startswith("You are given the text layer")
    ]
    return [part["text"] for m in body["messages"] for part in m["content"] if "text" in part]


@pytest.mark.parametrize("name", WITH_LAYOUT)
@pytest.mark.parametrize("kind", ROTATED)
def test_the_flow_on_a_page_shown_turned_is_the_flow_on_the_upright_page(
    name: str, kind: str
) -> None:
    """Status, header, lines, checkpoints, issues, notes, the requests and how each page
    was read; the layout request shows the model the same text."""
    (turned, turned_model), (upright, upright_model) = flow(name, kind), flow(name, "orig")
    assert result_of(turned.to_json()) == result_of(upright.to_json())
    assert turned.status == "ok" and turned.method == "layout" and turned.calls == 3
    assert turned_model.calls == upright_model.calls
    assert _layout_prompt(turned_model) == _layout_prompt(upright_model)


@pytest.mark.parametrize("name", ["s_classic", "m_grouped"])
@pytest.mark.parametrize("kind", TURNED)
def test_the_flow_on_content_drawn_turned_finds_every_item_row(name: str, kind: str) -> None:
    """The layout road refuses it (as `landscape`) and the model reads every page; the
    evidence checks its reading against rows that read upright, as on `landscape`. Before,
    turned by 180 or 270, pages failed their coverage check and were read again."""
    turned, _ = flow(name, kind)
    shown_upright, _ = flow(name, "landscape")
    assert turned.layout["reason"].startswith("LayoutError: ")

    def but_reason(res: Result) -> JSONObject:
        """Which heading the layout parser misses first depends on the turn."""
        out = result_of(res.to_json())
        out["layout"] = {k: v for k, v in res.layout.items() if k != "reason"}
        return out

    assert but_reason(turned) == but_reason(shown_upright)
    assert turned.status == "ok" and turned.method == "model" and turned.issues == []
    assert turned.lines == flow(name, "orig")[0].lines


# --------------------------------------------------------------------------- #
# The rectangles: the page as a viewer displays it does not change
# --------------------------------------------------------------------------- #


def _turned(rect: bx.Rect, rotation: int, width: float, height: float) -> bx.Rect:
    """A rectangle of the upright page (width x height, top-left origin, y down) on the same
    page turned clockwise by `rotation`."""
    x0, y0, x1, y1 = rect
    match rotation:
        case 90:
            return height - y1, x0, height - y0, x1
        case 180:
            return width - x1, height - y1, width - x0, height - y0
        case _:  # 270
            return y0, width - x1, y1, width - x0


@pytest.mark.parametrize("name", WITH_LAYOUT)
@pytest.mark.parametrize("kind", ROTATED)
def test_rectangles_on_a_page_shown_turned_are_the_upright_ones_turned(
    name: str, kind: str
) -> None:
    """Pinned against geometry, not against pdfium's text: the oracle test in test_boxes
    (every cell found by code frames its own text) passed on rot90 with 384 descriptions
    of xl_grouped wrong, because it reads pdfium's text on the same turned page.

    The frames keep the page's own /Rotate: the text is read with the page turned upright
    (upright_textpage), the rectangles are placed on the page as a viewer displays it. A
    change that shared one pdfium document between text_lines and page_frames would put
    the rectangles on the upright page and fail here."""
    rotation = int(kind[3:])
    turned, upright = flow(name, kind)[0], flow(name, "orig")[0]
    assert len(turned.frames) == len(upright.frames) > 0
    sizes = []
    for t, u in zip(turned.frames, upright.frames, strict=True):
        assert t is not None and u is not None and u.rotation == 0
        assert t.rotation == rotation
        swap = rotation in (90, 270)
        assert (t.width, t.height) == ((u.height, u.width) if swap else (u.width, u.height))
        sizes.append((u.width, u.height))
    assert len(turned.boxes) == len(upright.boxes) == len(turned.lines) > 10
    framed = 0
    for item, t_box, u_box in zip(upright.lines, turned.boxes, upright.boxes, strict=True):
        assert t_box is not None and u_box is not None
        width, height = sizes[item["page"] - 1]
        for key in bx.BOX_KEYS:
            got, rect = t_box[key], u_box[key]
            assert (got is None) == (rect is None), (key, item)
            if got is not None and rect is not None:
                # each page's rectangles are rounded outwards to 0.1 point on their own: a
                # side near a step lands one step apart (with the float error of the step)
                expected = _turned(rect, rotation, width, height)
                assert got == pytest.approx(expected, abs=0.11), (key, item)
                framed += 1
    assert framed > 5 * len(upright.lines)
