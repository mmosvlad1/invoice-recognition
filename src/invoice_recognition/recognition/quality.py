"""Photo quality before any model call: numbers, and the reason in words.

One sharpness number for a whole frame tells a user nothing they can act on.
This module measures separate things and names the defect:

  sheet     - is the whole sheet of paper inside the frame, or does it run out of it
  text      - where the text is, how tall a text line is in pixels, how far it is from the frame edge
  sharpness - measured on the text itself, not on the desk, the paper edge or empty paper
  light     - how bright the paper is, and whether a burnt-out spot (glare) covers part of it

`measure` returns the numbers, `reasons` turns them into a list of reasons,
`check` does both. No OCR engine and no model is needed: text is found as thin
dark strokes on a light background. (An OCR engine's word confidence was tried
as a sharpness score and ordered the photos worst of all, so it is not here.)

Everything is meant for the photo AS TAKEN, before `photo.rectify`: after
rectification the sheet fills the frame and there is no sheet edge to look at.

`pipeline.process` calls `check` on every image before the first request
(`Config.photo_quality`); the limits it passes are the `Config.photo_*` fields.
Pages of a scanned PDF are measured by the second half of this module
(`measure_pdf_pages`, `judge_page`, `ScanLimits`; `Config.scan_quality`, `Config.scan_*`):
the same text and light numbers, without the sheet-on-a-desk part.

The sharpness it decides on is `Quality.text_sharpness`, the variance of the
Laplacian inside text regions divided by the squared paper brightness. A score of
the whole frame was measured first and had three flaws: darkness reads as blur
(the variance falls with the square of brightness), an image narrower than the
working width scores too high (enlarging it by repeating pixels leaves steps that
count as sharpness), and a page with little text scores low however sharp it is.

The limits in `Limits` were measured on ten synthetic photos of two invoices
plus artificially damaged copies of them. They are starting points, not
calibrated thresholds; each one states what it rests on.
"""

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from PIL import Image

from invoice_recognition.recognition.config import MAX_IMAGE_PIXELS
from invoice_recognition.shapes import JSONObject

logger = logging.getLogger(__name__)

type Array = npt.NDArray[Any]

NORM_W = (
    1600  # every measurement is taken on a copy of this width: kernels mean the same on any photo
)
SIDES = ("top", "right", "bottom", "left")
SIDE_NAMES = {
    "top": "at the top",
    "right": "on the right",
    "bottom": "at the bottom",
    "left": "on the left",
}


# --------------------------------------------------------------------------- #
# Sheet of paper in the frame
# --------------------------------------------------------------------------- #


@dataclass
class Sheet:
    found: bool  # a large bright region (the paper) stands out against a darker background
    area: float  # its share of the frame
    # Per frame side: share of that side covered by the sheet (0 = the sheet edge is visible).
    touch: dict[str, float]
    free_corners: (
        int  # corners of the sheet outline that lie inside the frame, away from its border
    )
    # The paper, uint8 0/255, on a reduced copy of the frame.
    mask: Array | None = field(default=None, repr=False)


def _gray(img: Image.Image) -> Array:
    return np.asarray(img.convert("L"))


def find_sheet(img: Image.Image, min_contrast: float = 35.0, min_step: float = 0.25) -> Sheet:
    """The paper as the largest bright region, and how much of each frame side it covers.

    A sheet fully inside the frame touches no side. A sheet cut by the frame
    covers a stretch of that side; a photo taken so close that no desk is
    visible covers all four. `found` is False when the frame does not split
    into "paper" and "a darker background with a sharp border between them" -
    then nothing can be said about the sheet edge and every side counts as not
    visible. The sharp-border test matters: a shadow across a page that fills
    the frame also splits it into a bright and a dark half, but gradually."""
    gray = _gray(img)
    h, w = gray.shape
    scale = 800 / max(h, w)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interp)
    small = cv2.GaussianBlur(small, (7, 7), 0)
    _, raw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright, dark = small[raw > 0], small[raw == 0]
    all_sides = dict.fromkeys(SIDES, 1.0)
    sh, sw = small.shape
    # No darker background of real size: the frame is paper from edge to edge (or not a
    # sheet on a desk at all).
    if dark.size < 0.03 * small.size or bright.size == 0:
        return Sheet(False, 1.0, all_sides, 0)
    contrast = float(bright.mean()) - float(dark.mean())
    if contrast < min_contrast:
        return Sheet(False, 1.0, all_sides, 0)
    mask = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return Sheet(False, 1.0, all_sides, 0)
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c) / (sh * sw)
    if area < 0.25:  # too small to be the document; the bright thing found is something else
        return Sheet(False, float(area), all_sides, 0)
    if area > 0.97:  # paper from edge to edge: the "dark" class is the print itself, not a desk
        return Sheet(False, float(area), all_sides, 0)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [c], -1, 255, cv2.FILLED)
    # Is the outline a real paper edge? Brightness must jump across it, not drift: compare a
    # thin band just inside the outline with a thin band just outside (frame border excluded).
    k = np.ones((9, 9), np.uint8)
    band_in = (filled > 0) & (cv2.erode(filled, k) == 0)
    band_out = (filled == 0) & (cv2.dilate(filled, k) > 0)
    band_in[:4], band_in[-4:], band_in[:, :4], band_in[:, -4:] = False, False, False, False
    if band_in.sum() < 50 or band_out.sum() < 50:  # the outline runs along the frame border only
        return Sheet(False, float(area), all_sides, 0)
    step = (float(np.median(small[band_in])) - float(np.median(small[band_out]))) / contrast
    if step < min_step:
        return Sheet(False, float(area), all_sides, 0)
    strip = 3
    touch = {
        "top": float((filled[:strip].max(axis=0) > 0).mean()),
        "bottom": float((filled[-strip:].max(axis=0) > 0).mean()),
        "left": float((filled[:, :strip].max(axis=1) > 0).mean()),
        "right": float((filled[:, -strip:].max(axis=1) > 0).mean()),
    }
    quad = np.asarray(cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)).reshape(-1, 2)
    m = 0.012 * max(sh, sw)
    free = int(sum(1 for x, y in quad if m < x < sw - 1 - m and m < y < sh - 1 - m))
    # Where text may be looked for: the paper itself. Taken from the mask BEFORE closing,
    # because closing also swallows thin slivers of desk between a cut sheet and the frame
    # border, and a thin dark sliver is exactly what the text search takes for a line of text.
    paper = np.zeros_like(mask)
    rc, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(paper, [max(rc, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return Sheet(True, float(area), touch, free, paper)


# --------------------------------------------------------------------------- #
# Text: where it is, how tall, how sharp
# --------------------------------------------------------------------------- #


@dataclass
class Text:
    boxes: int  # word-like regions found
    height_px: float  # median height of a text line, in pixels of the ORIGINAL image
    height_norm: float  # the same on the normalised copy (width NORM_W)
    # Per frame side: distance from the nearest text to that side, in text heights.
    edge_margin: dict[str, float]
    # Text height / width of a letter edge, worst direction; free of size and lighting.
    edge_sharpness: float
    # Variance of the Laplacian inside text regions only, on a denoised copy 1000 px wide.
    text_laplacian: float
    contrast: float  # ink against paper inside text regions, 0-255


def _normalised(gray: Array, width: int = NORM_W) -> tuple[Array, float]:
    f = width / gray.shape[1]
    interp = cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC
    return cv2.resize(gray, None, fx=f, fy=f, interpolation=interp), f


def _otsu(values: Array) -> float:
    hist = np.bincount(values.astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.0
    w0 = np.cumsum(hist)
    m0 = np.cumsum(hist * np.arange(256))
    w1 = total - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (m0[-1] * w0 - m0 * total) ** 2 / (w0 * w1)
    between[~np.isfinite(between)] = 0
    return float(np.argmax(between))


def _word_boxes(
    norm: Array, allowed: Array | None
) -> tuple[list[tuple[int, int, int, int]], list[float], Array]:
    """Word-like dark regions on a light background: axis-aligned boxes, their true
    heights (short side of the rotated rectangle, so a tilted photo does not inflate
    them) and the ink mask. No OCR: thin dark strokes found by a black-hat filter,
    letters joined into words by a horizontal closing."""
    soft = cv2.GaussianBlur(norm, (0, 0), 1.0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    hat = cv2.morphologyEx(soft, cv2.MORPH_BLACKHAT, kernel)
    if allowed is not None:
        hat[allowed == 0] = 0
        values = hat[allowed > 0]
    else:
        values = hat
    thr = max(
        _otsu(values), 12.0
    )  # the floor keeps paper grain on a blank page from passing as ink
    ink = (hat > thr).astype(np.uint8) * 255
    words = cv2.morphologyEx(
        ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 1))
    )
    words = cv2.morphologyEx(words, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(words, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    heights: list[float] = []
    for c in contours:
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        short, long = min(rw, rh), max(rw, rh)
        if short < 5 or short > 90 or long < 1.5 * short or long > 40 * short:
            continue  # specks, ruling lines, pictures
        if cv2.contourArea(c) < 0.3 * rw * rh:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, w, h))
        heights.append(float(short))
    return boxes, heights, ink


def _laplacian_1000(gray: Array, width: int = 1000) -> Array:
    """Laplacian of a denoised copy 1000 px wide. A narrower image is enlarged with
    linear interpolation: INTER_AREA enlarges by repeating pixels, and the steps this
    leaves count as sharpness (19.1 instead of 12.2 on a photo 900 px wide)."""
    f = width / gray.shape[1]
    interp = cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR
    small = cv2.resize(gray, None, fx=f, fy=f, interpolation=interp)
    return cv2.Laplacian(cv2.medianBlur(small, 5), cv2.CV_64F)


def find_text(img: Image.Image, sheet: Sheet | None = None) -> Text:
    gray = _gray(img)
    norm, f = _normalised(gray)
    nh, nw = norm.shape
    allowed = None
    if sheet is not None and sheet.found and sheet.mask is not None:
        allowed = cv2.resize(sheet.mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        # Keep the paper edge itself out of the text search. The frame border does not erode
        # (OpenCV's default), so text that runs into the border of a cut sheet stays visible.
        allowed = cv2.erode(allowed, np.ones((31, 31), np.uint8))
    boxes, heights, _ = _word_boxes(norm, allowed)
    none = dict.fromkeys(SIDES, float("inf"))
    if len(boxes) < 5:
        return Text(len(boxes), 0.0, 0.0, none, 0.0, 0.0, 0.0)
    hn = float(np.median(heights))
    # Distance to the frame edge counts only boxes of a text-like height: a sliver of desk
    # between a cut sheet and the frame border is thin and dark too, but much lower than a line.
    typical = [b for b, hgt in zip(boxes, heights, strict=True) if 0.5 * hn <= hgt <= 4 * hn]
    xs0 = min(b[0] for b in typical)
    ys0 = min(b[1] for b in typical)
    xs1 = max(b[0] + b[2] for b in typical)
    ys1 = max(b[1] + b[3] for b in typical)
    margin = {
        "top": ys0 / hn,
        "left": xs0 / hn,
        "bottom": (nh - ys1) / hn,
        "right": (nw - xs1) / hn,
    }

    # Sharpness of the letter edges. Slope of the steepest edges divided by the ink/paper
    # contrast of the same word = 1 / (edge width in px); times the text height it says how
    # many edge widths fit into a line of text. Dividing by contrast makes it blind to dim
    # light, multiplying by height makes it blind to image size. Horizontal and vertical
    # slopes are taken separately and the worse one counts: hand shake smears one direction.
    den = cv2.medianBlur(norm, 3).astype(np.float32)
    gx = np.abs(cv2.Sobel(den, cv2.CV_32F, 1, 0, ksize=3)) / 8
    gy = np.abs(cv2.Sobel(den, cv2.CV_32F, 0, 1, ksize=3)) / 8
    sx, sy, contrasts = [], [], []
    for x, y, w, h in boxes:
        x0, y0, x1, y1 = max(x - 2, 0), max(y - 2, 0), min(x + w + 2, nw), min(y + h + 2, nh)
        patch = den[y0:y1, x0:x1]
        lo, hi = np.percentile(patch, [5, 95])
        c = float(hi - lo)
        if c < 8:
            continue
        contrasts.append(c)
        sx.append(float(np.percentile(gx[y0:y1, x0:x1], 95)) / c)
        sy.append(float(np.percentile(gy[y0:y1, x0:x1], 95)) / c)
    edge = hn * min(float(np.median(sx)), float(np.median(sy))) if sx else 0.0

    # The variance of the Laplacian, restricted to text regions.
    lap = _laplacian_1000(gray)
    region = np.zeros(lap.shape, np.uint8)
    k = lap.shape[1] / nw
    for x, y, w, h in boxes:
        top_left = (int(x * k) - 1, int(y * k) - 1)
        bottom_right = (int((x + w) * k) + 1, int((y + h) * k) + 1)
        cv2.rectangle(region, top_left, bottom_right, 255, cv2.FILLED)
    text_lap = float(lap[region > 0].var()) if (region > 0).any() else 0.0

    contrast = float(np.median(contrasts)) if contrasts else 0.0
    return Text(len(boxes), hn / f, hn, margin, float(edge), text_lap, contrast)


# --------------------------------------------------------------------------- #
# Light
# --------------------------------------------------------------------------- #


@dataclass
class Light:
    # Typical brightness of the paper, 0-255 (median over tiles of each tile's bright end).
    paper_level: float
    paper_level_low: float  # the darkest tenth of the tiles: a shadow over part of the sheet
    # Share of the sheet taken by the largest burnt-out spot that is clearly brighter than
    # the paper.
    glare: float


def find_light(img: Image.Image, sheet: Sheet | None = None, tiles: int = 12) -> Light:
    gray = _gray(img)
    h, w = gray.shape
    scale = 800 / max(h, w)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interp)
    sh, sw = small.shape
    if sheet is not None and sheet.found and sheet.mask is not None:
        inside = cv2.resize(sheet.mask, (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
    else:
        inside = np.ones((sh, sw), bool)
    levels = []
    for i in range(tiles):
        for j in range(tiles):
            ys = slice(i * sh // tiles, (i + 1) * sh // tiles)
            xs = slice(j * sw // tiles, (j + 1) * sw // tiles)
            m = inside[ys, xs]
            if m.mean() < 0.9:
                continue
            levels.append(float(np.percentile(small[ys, xs][m], 90)))
    if not levels:
        return Light(0.0, 0.0, 0.0)
    level = float(np.median(levels))
    low = float(np.percentile(levels, 10))
    # Glare: pixels at the top of the range AND well above the paper around them, as one
    # connected spot.
    soft = cv2.GaussianBlur(small, (5, 5), 0)
    burnt = ((soft >= 250) & inside & (soft.astype(np.float32) >= level + 12)).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(burnt, connectivity=8)
    biggest = int(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0
    return Light(level, low, biggest / max(int(inside.sum()), 1))


# --------------------------------------------------------------------------- #
# All numbers, and the reasons
# --------------------------------------------------------------------------- #


def _rounded(x: Any) -> Any:
    """Plain numbers for a JSON answer, rounded for reading (4 places: glare is a share)."""
    if isinstance(x, dict):
        return {k: _rounded(v) for k, v in x.items()}
    return round(float(x), 4) if isinstance(x, float) else x


@dataclass
class Quality:
    width: int
    height: int
    sheet: Sheet
    text: Text
    light: Light

    @property
    def text_sharpness(self) -> float:
        """`text.text_laplacian` divided by the squared paper brightness (1.0 = white paper
        at 255). The Laplacian is proportional to brightness, so its variance falls four
        times when the photo is twice as dark; without this a dark but crisp photo reads
        as blurred."""
        level = self.light.paper_level / 255
        return self.text.text_laplacian / (level * level) if level > 0 else 0.0

    def to_json(self) -> JSONObject:
        d = asdict(self)
        d["sheet"].pop("mask", None)
        d["text_sharpness"] = self.text_sharpness
        d["text"]["edge_margin"] = {
            k: (None if v == float("inf") else v) for k, v in self.text.edge_margin.items()
        }
        out: JSONObject = _rounded(d)
        return out


@dataclass
class Limits:
    """Where a number turns into a reason. Starting points measured on ten synthetic photos
    of two invoices and damaged copies of them, each with its basis; 0 switches a limit off."""

    # Sheet. Every photo with the whole sheet in the frame measured exactly 0; cut copies
    # 0.81-0.87; the one real photo with a clipped corner 0.017.
    sheet_touch: float = 0.005
    # Text this close (in text heights) to a frame side the sheet runs out of -> text is
    # probably lost. Copies with lost text 0-0.8, copies and photos with only blank margin
    # lost 2.0-3.3. A cut through an empty band of the page (10.4) is not caught by any value.
    text_margin_cut: float = 1.5
    # No sheet edge visible at all (shot close up): only text running into the frame border
    # rejects.
    text_margin_unseen: float = 0.5
    # Blur, on `Quality.text_sharpness`, measured on a seven-step sharpness scale of one
    # invoice photo read by the model. Reject: between the step where the model read 1 of 26
    # lines (3 of 14 on another photo) and the one where it read 12 of 26. Warn: between the
    # step with the first wrong article number (25 of 26) and the one read whole (26 of 26).
    sharpness_reject: float = 140.0
    sharpness_warn: float = 450.0
    # Light. No photo of the set is below 186, copies darkened x0.5 are at 118. Whether a dark
    # photo is read worse was NOT measured.
    paper_level: float = 150.0
    # Glare: share of the sheet. Every photo and copy without glare measured 0; the smallest
    # artificial spot 2.2%. Whether the spot covers text is not looked at.
    glare: float = 0.01
    # Text height in pixels of the original image. Off: no photo with small but sharp text
    # was read by a model, so there is nothing to put a limit on.
    text_height_px: float = 0.0
    min_boxes: int = 5


@dataclass
class Reason:
    code: str  # no_text | sheet_cut | text_at_edge | sheet_edge_unseen | blurred | dark | glare | text_small
    message: str  # for the user
    severity: str  # "reject" | "warn"
    value: float | None = None
    side: str | None = None

    def to_json(self) -> JSONObject:
        return {k: v for k, v in asdict(self).items() if v is not None}


def measure(img: Image.Image) -> Quality:
    sheet = find_sheet(img)
    return Quality(img.width, img.height, sheet, find_text(img, sheet), find_light(img, sheet))


def reasons(q: Quality, lim: Limits | None = None) -> list[Reason]:
    """Reasons for a photo measured on the frame AS TAKEN (before `photo.rectify`):
    after rectification the sheet fills the frame and the sheet check has nothing to look at."""
    lim = lim or Limits()
    out: list[Reason] = []
    t, s, light = q.text, q.sheet, q.light
    has_text = t.boxes >= lim.min_boxes
    if not has_text:
        out.append(
            Reason(
                "no_text",
                "No text was found in the photo: it is not a document, or it cannot be made out.",
                "reject",
                float(t.boxes),
            )
        )

    # The sheet and the frame.
    if s.found:
        for side in SIDES:
            if s.touch[side] <= lim.sheet_touch:
                continue
            if t.edge_margin[side] < lim.text_margin_cut:
                out.append(
                    Reason(
                        "text_at_edge",
                        f"The sheet is cut off {SIDE_NAMES[side]} and the text reaches the edge "
                        "of the frame: part of the document is not in the photo.",
                        "reject",
                        round(t.edge_margin[side], 2),
                        side,
                    )
                )
            else:
                out.append(
                    Reason(
                        "sheet_cut",
                        f"The sheet runs out of the frame {SIDE_NAMES[side]}: check that the "
                        "whole document is in the photo.",
                        "warn",
                        round(s.touch[side], 3),
                        side,
                    )
                )
    elif has_text:
        near = [side for side in SIDES if t.edge_margin[side] < lim.text_margin_unseen]
        for side in near:
            out.append(
                Reason(
                    "text_at_edge",
                    f"The text runs into the edge of the frame {SIDE_NAMES[side]}: part of the "
                    "document is not in the photo.",
                    "reject",
                    round(t.edge_margin[side], 2),
                    side,
                )
            )
        if not near:
            out.append(
                Reason(
                    "sheet_edge_unseen",
                    "No edge of the sheet is visible in the photo, so it cannot be confirmed "
                    "that the whole document is in it.",
                    "warn",
                )
            )

    if has_text:
        sharp = q.text_sharpness
        if lim.sharpness_reject and sharp < lim.sharpness_reject:
            out.append(
                Reason(
                    "blurred",
                    "The photo is blurred: the text cannot be read reliably. "
                    "Please take the photo again.",
                    "reject",
                    round(sharp, 1),
                )
            )
        elif lim.sharpness_warn and sharp < lim.sharpness_warn:
            out.append(
                Reason(
                    "blurred",
                    "The photo is not sharp: item codes (SKUs) may be misread.",
                    "warn",
                    round(sharp, 1),
                )
            )
        if lim.text_height_px and t.height_px < lim.text_height_px:
            out.append(
                Reason(
                    "text_small",
                    "The text in the photo is too small: move closer or use a higher resolution.",
                    "warn",
                    round(t.height_px, 1),
                )
            )
    if lim.paper_level and 0 < light.paper_level < lim.paper_level:
        out.append(
            Reason(
                "dark", "The photo is dark: add more light.", "warn", round(light.paper_level, 1)
            )
        )
    if light.glare > lim.glare:
        out.append(
            Reason(
                "glare",
                "There is glare on the sheet: part of the text may be washed out.",
                "warn",
                round(light.glare, 4),
            )
        )
    return out


def check(img: Image.Image, lim: Limits | None = None) -> tuple[Quality, list[Reason]]:
    """Numbers and reasons for one photo. An empty list means no defect was measured -
    not that the photo is good."""
    q = measure(img)
    return q, reasons(q, lim)


# --------------------------------------------------------------------------- #
# A page of a scanned PDF
# --------------------------------------------------------------------------- #
#
# The same numbers as for a photo, minus everything that is about a sheet lying on a desk:
# the page of a scan IS the sheet, so there is no sheet edge to look for, no text "at the
# frame edge" and no glare. What is left: is there text, how sharp is it, how dark is the
# paper.
#
# The page is measured at the resolution of its own image ("as scanned"), not rendered to a
# fixed one. Rendered to a width of 1600 px, a copy of a scan reduced to 0.3 of its
# resolution scored 666 against 700 for the original - the renderer enlarges by repeating
# pixels, and their steps count as sharpness. Measured at its own resolution the same copy
# scores 67 against 809: on this scale low resolution and blur are one defect, "the letters
# have no edges".

SCAN_MAX_W = 3200  # px; a page image wider than this is measured at this width (time, not accuracy)


@dataclass
class ScanLimits:
    """Where a number of a scanned page turns into a reason. Starting points measured on
    the 20 pages of four synthetic scans that the cheap model reads well, and on damaged
    copies of them; 0 switches a limit off."""

    # `PageQuality.text_sharpness`. The 20 good pages: 734-987 (median 830). Blurred copies of
    # one page (809 as scanned): 572 / 342 / 223 / 127 / 78 / 37 at a Gaussian radius of
    # 0.5 / 0.8 / 1.0 / 1.25 / 1.5 / 2.0 px of a 110 dpi scan; copies reduced to 0.75 / 0.6 /
    # 0.5 / 0.4 / 0.3 of the resolution: 586 / 415 / 282 / 157 / 67.
    # That page, read by the model at each level of blur: 33 rows of 33 at 809, 572, 342,
    # 223 and 127; the first wrong article number at 78; 19 of 33 at 36.5. Reject at 100
    # holds (it is also below the score at which the model invented rows on photos, 108).
    # Warn at 200: above it the model read every row, so a warning there was noise.
    sharpness_reject: float = 100.0
    sharpness_warn: float = 200.0
    # The good pages are at 243-244; copies darkened x0.7 / x0.5 / x0.3 at 170 / 122 / 73.
    # The same value as for photos; whether a dark scan is read worse was not measured.
    paper_level: float = 150.0
    # Text fainter than this share of the paper brightness on a page with no ink is not
    # text: the print of the other side of the sheet showing through (0.04-0.08 at 10-20%
    # opacity; real text 0.58-0.61, text blurred with a radius of 4 px 0.17).
    min_contrast: float = 0.10
    # A page with less than this share of clearly dark pixels is blank (blank copies and
    # show-through: 0; pages with a table: 0.03-0.10; a page with ten lines of plain text: 0.015).
    blank_ink: float = 0.001
    min_boxes: int = 5


@dataclass
class PageQuality:
    width: int  # of the image that was measured, px: the page at the resolution of its scan
    height: int
    density: float  # pixels per PDF point; x 72 = the resolution of the scan in dpi
    text: Text
    light: Light
    ink: float  # share of the page clearly darker than its paper (below 0.8 of its brightness)

    @property
    def text_sharpness(self) -> float:
        """The number the photo check decides on (`Quality.text_sharpness`), by the same code."""
        sheet = Sheet(False, 1.0, {}, 0)
        return Quality(self.width, self.height, sheet, self.text, self.light).text_sharpness

    @property
    def contrast(self) -> float:
        """Ink against paper inside the text regions, as a share of the paper brightness."""
        return self.text.contrast / self.light.paper_level if self.light.paper_level > 0 else 0.0

    def to_json(self) -> JSONObject:
        d = asdict(self) | {
            "dpi": self.density * 72,
            "text_sharpness": self.text_sharpness,
            "contrast": self.contrast,
        }
        d["text"]["edge_margin"] = {
            k: (None if v == float("inf") else v) for k, v in self.text.edge_margin.items()
        }
        out: JSONObject = _rounded(d)
        return out


def measure_page(img: Image.Image, density: float = 0.0) -> PageQuality:
    """The numbers of one page image of a scan. No sheet is looked for: text and light are
    measured over the whole page."""
    text, light = find_text(img, None), find_light(img, None)
    gray = _gray(img)
    scale = 800 / max(gray.shape)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interp)
    ink = float((small < 0.8 * light.paper_level).mean())
    return PageQuality(img.width, img.height, float(density), text, light, ink)


NO_IMAGE = "no image on the page: nothing to measure"
NOT_ONLY_A_SCAN = (
    "something visible is drawn besides the page image: the page is not measured as a scan"
)


@dataclass(frozen=True)
class TooLarge:
    """A page whose images have more pixels together than may be opened: it is not
    measured, and it is not read."""

    pixels: int
    limit: int


def scan_page_image(
    page: Any, min_cover: float = 0.85, max_pixels: int = MAX_IMAGE_PIXELS
) -> tuple[Image.Image, float] | str | TooLarge:
    """A pypdfium2 page -> (the page in gray at the resolution of its scan, pixels per
    point), or the reason why the page is not measured (NO_IMAGE, NOT_ONLY_A_SCAN), or
    TooLarge when its images hold more than max_pixels pixels together.

    The pixel sizes are read from the images' dictionaries before anything is decoded: a
    PDF of under 1 MB can declare an image that takes gigabytes once decoded, and a render
    of the page decodes every image on it. A render is capped at max_pixels as well.

    Only a page that is nothing but a scan is measured: images, and at most text that is
    not painted (the invisible layer an OCR program lays over a scan). Anything else that
    is visible - printed text, lines, a form - may be the real content over a background
    picture, and the pixels of the picture say nothing about it.

    When one upright image covers the sheet (what a scanner writes), its stored pixels are
    measured as they are. Otherwise the page is rendered at the pixel density of its
    largest image. The two differ: a render that does not land on the stored pixels one to
    one resamples them, and that alone moved the score of a page by up to 14%."""
    w, h = page.get_size()
    best = None  # (area on the page, pixels per point, the image)
    pixels = 0  # of every image on the page
    for obj in page.get_objects():
        if (
            obj.type == pdfium_c.FPDF_PAGEOBJ_TEXT
            and pdfium_c.FPDFTextObj_GetTextRenderMode(obj.raw)
            == pdfium_c.FPDF_TEXTRENDERMODE_INVISIBLE
        ):
            continue
        # An image inside a form reports its place in the form, not on the page.
        if obj.type != pdfium_c.FPDF_PAGEOBJ_IMAGE or obj.level > 0:
            return NOT_ONLY_A_SCAN
        pw, ph = obj.get_px_size()  # from the image's dictionary: nothing is decoded
        pixels += pw * ph
        if pixels > max_pixels:
            return TooLarge(pixels, max_pixels)
        left, bottom, right, top = obj.get_bounds()
        bw, bh = right - left, top - bottom
        if bw <= 0 or bh <= 0:
            continue
        if best is None or bw * bh > best[0]:
            # Long side against long side: an image placed turned by 90 degrees measures the same.
            best = (bw * bh, max(pw, ph) / max(bw, bh), obj)
    if best is None or best[1] <= 0:
        return NO_IMAGE
    area, density, obj = best
    a, b, c, d, _, _ = obj.get_matrix().get()
    upright = (
        page.get_rotation() == 0 and a > 0 and d > 0 and abs(b) < 1e-3 * a and abs(c) < 1e-3 * d
    )
    if upright and area >= min_cover * w * h:
        try:
            img: Image.Image = obj.get_bitmap(render=False).to_pil().convert("L")
            if img.width > SCAN_MAX_W:
                size = (SCAN_MAX_W, round(img.height * SCAN_MAX_W / img.width))
                img = img.resize(size, Image.Resampling.BOX)
            return img, density
        # Fault boundary: an image this build cannot hand out as it is stored: render the
        # page instead.
        except Exception:
            logger.warning("a page image could not be read as stored", exc_info=True)
    # A hair under the density: the renderer rounds the size up, and one pixel more would
    # resample every stored pixel.
    scale = min(density * (1 - 1e-6), SCAN_MAX_W / w)
    if (w * scale + 1) * (h * scale + 1) > max_pixels:  # a page far taller than it is wide
        # The sides of the same shape that hold max_pixels, less the pixel the renderer may
        # round each side up by.
        long_w, long_h = math.sqrt(max_pixels * w / h), math.sqrt(max_pixels * h / w)
        scale = min((long_w - 1) / w, (long_h - 1) / h)
    rendered: Image.Image = page.render(scale=scale).to_pil().convert("L")
    return rendered, density


def measure_pdf_pages(
    data: bytes, max_pixels: int = MAX_IMAGE_PIXELS
) -> list[PageQuality | Exception | str | TooLarge]:
    """One entry per page of the PDF: the numbers; the reason why the page is not measured
    (NO_IMAGE, NOT_ONLY_A_SCAN); TooLarge for a page whose images may not be opened; or
    the exception the measurement of that page ended with (one page that breaks is that
    page's problem)."""
    pdf = pdfium.PdfDocument(data)
    out: list[PageQuality | Exception | str | TooLarge] = []
    try:
        for page in pdf:
            try:
                got = scan_page_image(page, max_pixels=max_pixels)
                out.append(got if isinstance(got, str | TooLarge) else measure_page(*got))
            except Exception as e:  # fault boundary: one page
                logger.warning("a page of the scan could not be measured", exc_info=True)
                out.append(e)
    finally:
        pdf.close()
    return out


def judge_page(q: PageQuality, lim: ScanLimits | None = None) -> tuple[str, list[Reason]]:
    """Verdict and reasons for one page of a scan.

      empty     - no text and no ink: a blank page or the back of a sheet (reason "no_text",
                  severity "skip")
      unchecked - no lines of text were found, but the page is not blank (a picture, a stamp,
                  something this module cannot read as text): nothing to measure the
                  sharpness on (reason "no_text_lines", "info")
      reject    - text too blurred, or scanned at too low a resolution, to be read ("blurred",
                  "reject")
      warn      - doubtful: "blurred" or "dark", severity "warn"
      ok        - no defect was measured - not "the page is good"

    The reasons of a photo that are about a sheet on a desk (text_at_edge, sheet_cut,
    sheet_edge_unseen, glare) do not exist here."""
    lim = lim or ScanLimits()
    t, light = q.text, q.light
    blank = q.ink < lim.blank_ink
    has_text = t.boxes >= lim.min_boxes and not (blank and q.contrast < lim.min_contrast)
    if not has_text:
        if blank:
            reason = Reason(
                "no_text",
                "No text was found on the page: the page was not read.",
                "skip",
                float(t.boxes),
            )
            return "empty", [reason]
        reason = Reason(
            "no_text_lines",
            "No lines of text were found on the page, but it is not blank: "
            "its quality was not assessed.",
            "info",
            float(t.boxes),
        )
        return "unchecked", [reason]
    out: list[Reason] = []
    sharp = q.text_sharpness
    if lim.sharpness_reject and sharp < lim.sharpness_reject:
        out.append(
            Reason(
                "blurred",
                "The page is blurred or scanned at too low a resolution: the text cannot be "
                "read reliably. Please scan it again.",
                "reject",
                round(sharp, 1),
            )
        )
    elif lim.sharpness_warn and sharp < lim.sharpness_warn:
        out.append(
            Reason(
                "blurred",
                "The page is not sharp: the values read from it may contain errors.",
                "warn",
                round(sharp, 1),
            )
        )
    if lim.paper_level and 0 < light.paper_level < lim.paper_level:
        out.append(
            Reason(
                "dark",
                "The page is dark: the values read from it may contain errors.",
                "warn",
                round(light.paper_level, 1),
            )
        )
    verdict = "reject" if any(r.severity == "reject" for r in out) else ("warn" if out else "ok")
    return verdict, out
