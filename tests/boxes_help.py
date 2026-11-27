"""An independent witness of where text is printed on a PDF page, for the tests of the
rectangles the flow reports.

The rectangles are in points of the page as displayed: origin at its top-left corner,
y downwards. This module never maps them back by the flow's own formulas: pdfium does,
with FPDF_DeviceToPage on a device of ten units per point, so a rectangle rounded to 0.1
lands on whole device units. pdfium then says which text lies under it.
"""

import ctypes
from types import TracebackType
from typing import Self

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

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
