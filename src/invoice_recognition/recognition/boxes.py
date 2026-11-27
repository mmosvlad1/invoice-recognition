"""Where on the page each read value of a line item is printed.

A rectangle is given on the page as it is displayed: the crop box (cut to the media
box), turned by /Rotate, origin at the top-left corner, y downwards, in PDF points.
That is the space of a PDF.js viewport at scale 1 (PDF.js also multiplies by
/UserUnit, which pdfium ignores; a page placing the rectangles in proportion to its
viewport is right either way).

pdfium gives the boxes of characters in PDF user space: not turned, not moved to the
crop box. PageFrame maps them onto the displayed page. Pure Python, apart from
page_frames, which opens the PDF.
"""

import logging
import math
from dataclasses import dataclass

import pypdfium2 as pdfium

logger = logging.getLogger(__name__)

type Rect = tuple[float, float, float, float]
"""[x0, y0, x1, y1]: left, top, right, bottom on the displayed page, rounded outwards
to 0.1 point and held to the page: 0 <= x0 < x1 <= width, 0 <= y0 < y1 <= height."""

EDGE_TOL = 0.5  # points: how far a value may stick out of the page and still be framed


def _down(v: float) -> float:
    # round() first: 0.7 * 10 is 7.000000000000001, which ceil() would take to 0.8.
    return math.floor(round(v * 10, 6)) / 10


def _up(v: float) -> float:
    return math.ceil(round(v * 10, 6)) / 10


@dataclass(frozen=True)
class PageFrame:
    """One PDF page as displayed. (left, bottom, right, top): the view, crop box cut to
    the media box, in user space; rotation: /Rotate, clockwise, 0 / 90 / 180 / 270;
    width, height: the displayed size, rounded to 0.1."""

    left: float
    bottom: float
    right: float
    top: float
    rotation: int
    width: float
    height: float

    @classmethod
    def of(cls, view: tuple[float, float, float, float], rotation: int) -> PageFrame | None:
        """The frame of a page with this view and rotation; None when the view has no
        area (a crop box that misses the media box) or is not made of finite numbers: no
        rectangle could be placed on it.

        pdfium reports only quarter turns; any other rotation also gives None, so that
        _point never maps a page by a turn it was not given."""
        left, bottom, right, top = (float(v) for v in view)
        if rotation not in (0, 90, 180, 270) or not all(map(math.isfinite, view)):
            return None
        w, h = round(right - left, 1), round(top - bottom, 1)
        if w <= 0 or h <= 0:
            return None
        if rotation in (90, 270):
            w, h = h, w
        return cls(left, bottom, right, top, rotation, w, h)

    def _point(self, x: float, y: float) -> tuple[float, float]:
        """A point of user space on the displayed page (PDF.js PageViewport at scale 1)."""
        match self.rotation:
            case 0:
                return x - self.left, self.top - y
            case 90:
                return y - self.bottom, x - self.left
            case 180:
                return self.right - x, y - self.bottom
            case _:  # 270
                return self.top - y, self.right - x

    def to_display(self, x0: float, bottom: float, x1: float, top: float) -> Rect:
        """A box of user space on the displayed page: (x0, y0, x1, y1), not rounded and
        not held to the page."""
        ax, ay = self._point(x0, bottom)
        bx, by = self._point(x1, top)
        return min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)

    def place(self, x0: float, bottom: float, x1: float, top: float, *, cell: bool) -> Rect | None:
        """A box of user space as a rectangle of the result, rounded outwards.

        cell=True, a value: None unless it lies on the page (EDGE_TOL), then held to it;
        a value cut by the edge of the page is not claimed. cell=False, a whole row: cut
        to the page, None when nothing of it is left. A box with a coordinate that is not
        a finite number (a broken font) is never placed."""
        if not all(map(math.isfinite, (x0, bottom, x1, top))):
            return None
        dx0, dy0, dx1, dy1 = self.to_display(x0, bottom, x1, top)
        if cell and (
            dx0 < -EDGE_TOL
            or dy0 < -EDGE_TOL
            or dx1 > self.width + EDGE_TOL
            or dy1 > self.height + EDGE_TOL
        ):
            return None
        rx0, ry0 = max(0.0, _down(dx0)), max(0.0, _down(dy0))
        rx1, ry1 = min(self.width, _up(dx1)), min(self.height, _up(dy1))
        if rx0 >= rx1 or ry0 >= ry1:
            return None
        return rx0, ry0, rx1, ry1


def page_frames(data: bytes) -> list[PageFrame | None]:
    """The frame of every page of a PDF, in order. A page without a usable frame gets
    None, and the pages that have none are logged once."""
    pdf = pdfium.PdfDocument(data)
    frames: list[PageFrame | None] = []
    try:
        for page in pdf:
            frames.append(PageFrame.of(page.get_bbox(), page.get_rotation()))
    finally:
        pdf.close()
    missing = [p for p, f in enumerate(frames, start=1) if f is None]
    if missing:
        logger.warning("pages without a usable page box, nothing is framed on them: %s", missing)
    return frames
