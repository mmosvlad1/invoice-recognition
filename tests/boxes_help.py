"""An independent witness of where text is printed on a PDF page, for the tests of the
rectangles the flow reports.

The rectangles are in points of the page as displayed: origin at its top-left corner,
y downwards. This module never maps them back by the flow's own formulas: pdfium does,
with FPDF_DeviceToPage on a device of ten units per point, so a rectangle rounded to 0.1
lands on whole device units. pdfium then says which text lies under it.
"""

import ctypes
import functools
import io
from types import TracebackType
from typing import Self

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from pypdf import PageObject, PdfWriter, Transformation
from pypdf.generic import RectangleObject

from invoice_recognition.recognition import evidence as ev
from invoice_recognition.recognition import layout as lay

from samples import INVOICES

type Rect = tuple[float, float, float, float]

UNITS = 10  # device units per point


class Oracle:
    """One PDF opened in pdfium: what text is under a rectangle of a displayed page."""

    def __init__(self, data: bytes) -> None:
        self.pdf = pdfium.PdfDocument(data)
        self._pages: dict[int, tuple[pdfium.PdfPage, pdfium.PdfTextPage]] = {}

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        for page, text in self._pages.values():
            text.close()
            page.close()
        self._pages.clear()
        self.pdf.close()

    def _page(self, page_no: int) -> tuple[pdfium.PdfPage, pdfium.PdfTextPage]:
        if page_no not in self._pages:
            page = self.pdf[page_no - 1]
            self._pages[page_no] = (page, page.get_textpage())
        return self._pages[page_no]

    def size(self, page_no: int) -> tuple[float, float]:
        """The displayed width and height pdfium reports for the page."""
        w, h = self._page(page_no)[0].get_size()
        return float(w), float(h)

    def to_page(self, page_no: int, x: float, y: float) -> tuple[float, float]:
        """A point of the displayed page (points, top-left origin) in PDF user space."""
        page = self._page(page_no)[0]
        w, h = self.size(page_no)
        px, py = ctypes.c_double(), ctypes.c_double()
        ok = pdfium_c.FPDF_DeviceToPage(
            page,
            0,
            0,
            round(w * UNITS),
            round(h * UNITS),
            0,  # pdfium adds the page's own /Rotate
            round(x * UNITS),
            round(y * UNITS),
            ctypes.byref(px),
            ctypes.byref(py),
        )
        assert ok
        return px.value, py.value

    def to_display(self, page_no: int, x: float, y: float) -> tuple[float, float]:
        """A point of user space on the displayed page, by FPDF_PageToDevice on a device
        of a hundred units per point (it returns whole units)."""
        page = self._page(page_no)[0]
        w, h = self.size(page_no)
        dx, dy = ctypes.c_int(), ctypes.c_int()
        fine = 10 * UNITS
        ok = pdfium_c.FPDF_PageToDevice(
            page,
            0,
            0,
            round(w * fine),
            round(h * fine),
            0,  # pdfium adds the page's own /Rotate
            x,
            y,
            ctypes.byref(dx),
            ctypes.byref(dy),
        )
        assert ok
        return dx.value / fine, dy.value / fine

    def text_under(self, page_no: int, rect: Rect) -> str:
        """The text pdfium finds inside the rectangle, whitespace collapsed."""
        x0, y0, x1, y1 = rect
        corners = [self.to_page(page_no, x, y) for x, y in ((x0, y0), (x1, y1))]
        xs, ys = [c[0] for c in corners], [c[1] for c in corners]
        text = self._page(page_no)[1].get_text_bounded(min(xs), min(ys), max(xs), max(ys))
        return " ".join(text.split())


def text_under(data: bytes, page_no: int, rect: Rect) -> str:
    """Oracle.text_under for one rectangle."""
    with Oracle(data) as oracle:
        return oracle.text_under(page_no, rect)


# --------------------------------------------------------------------------- #
# The same document, displayed differently
# --------------------------------------------------------------------------- #

VARIANTS = ("orig", "rot90", "rot180", "rot270", "shift", "crop", "landscape")
TURNED = ("turned90", "turned180", "turned270")
SHIFT = (100.0, 200.0)
CROP = (30.0, 300.0, 60.0, 100.0)  # cut from the left, bottom, right and top


def variant(data: bytes, kind: str) -> bytes:
    """Every page of the PDF displayed another way:

    orig      - unchanged (written again by pypdf);
    rot90, rot180, rot270 - /Rotate set, the content untouched: the page is shown turned;
    shift     - the content and the media box moved by SHIFT: user space no longer starts
                at the corner of the page;
    crop      - a crop box inside the media box (CROP): part of the text is outside the
                displayed page but stays in the text layer;
    landscape - the content turned a quarter counter-clockwise onto a landscape media box,
                and /Rotate 90 to show it upright again: on the displayed page it looks
                like orig, in user space every word of a row has the same X;
    turned90, turned180, turned270 - the content turned counter-clockwise by that many
                degrees onto a media box turned with it, no /Rotate: the page is shown
                turned, its text written along +y, -x or -y of user space;
    crop_off  - a crop box that misses the media box: the displayed page has no area.
    """
    w = PdfWriter(clone_from=io.BytesIO(data))
    for page in w.pages:
        left, bottom, right, top = (float(v) for v in page.mediabox)
        match kind:
            case "orig":
                pass
            case "rot90" | "rot180" | "rot270":
                page.rotation = int(kind[3:])
            case "shift":
                dx, dy = SHIFT
                page.add_transformation(Transformation().translate(dx, dy))
                box = RectangleObject([left + dx, bottom + dy, right + dx, top + dy])
                page.mediabox = page.cropbox = box
            case "crop":
                cl, cb, cr, ct = CROP
                page.cropbox = RectangleObject([left + cl, bottom + cb, right - cr, top - ct])
            case "landscape":
                _turn(page, 90)
                page.rotation = 90
            case "turned90" | "turned180" | "turned270":
                _turn(page, int(kind[6:]))
            case "crop_off":
                page.cropbox = RectangleObject([right + 10, top + 10, right + 50, top + 50])
            case _:
                raise ValueError(kind)
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


@functools.cache
def sample(name: str, kind: str = "orig") -> bytes:
    """A sample invoice displayed as `kind` (variant), built once per run."""
    return variant((INVOICES / f"{name}.pdf").read_bytes(), kind)


@functools.cache
def sample_text_lines(name: str, kind: str = "orig") -> list[list[lay.TextLine]]:
    """layout.text_lines of sample(name, kind), read once per run and shared by the tests
    that use it: not to be changed."""
    return lay.text_lines(sample(name, kind))


def _turn(page: PageObject, degrees: int) -> None:
    """The content of the page turned counter-clockwise by 90, 180 or 270 degrees onto a
    media box (and crop box) turned with it, its corner at the origin."""
    left, bottom, right, top = (float(v) for v in page.mediabox)
    width, height = right - left, top - bottom
    move = {90: (height, 0.0), 180: (width, height), 270: (0.0, width)}[degrees]
    page.add_transformation(
        Transformation().translate(-left, -bottom).rotate(degrees).translate(*move)
    )
    box = [0, 0, width, height] if degrees == 180 else [0, 0, height, width]
    page.mediabox = page.cropbox = RectangleObject(box)


# --------------------------------------------------------------------------- #
# The layout road, as the flow takes it
# --------------------------------------------------------------------------- #


def layout_of(data: bytes, pages: list[list[lay.TextLine]], answer: object) -> lay.Layout:
    """The model's layout answer resolved on the page the flow shows the model: the first
    one whose text has rows that look like line items."""
    evidence = ev.build(ev.pdf_text_layer(data), "text_layer")
    shown = next((p.number for p in evidence.pages if p.item_rows), 1)
    return lay.resolve(answer, pages[shown - 1])
