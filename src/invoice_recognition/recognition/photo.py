"""A photo of an invoice before the model sees it: upright, and the sheet flattened.

Phone photos keep their rotation in EXIF, not in the pixels, and show the sheet
skewed on a desk. `load` turns the photo upright; `rectify` finds the sheet of
paper and warps it to a flat, upright page. Flattening is the only preparation
step that did not hurt the model's reading in the measurements, so it is the
only one here.
"""

import io
import logging
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageOps

from invoice_recognition.llm import ContentPart, file_part

logger = logging.getLogger(__name__)

type Array = npt.NDArray[Any]


def load(data: bytes) -> Image.Image:
    """The photo upright, as taken: what the quality check measures and what is then
    flattened."""
    img = Image.open(io.BytesIO(data))
    return ImageOps.exif_transpose(img)  # phone photos keep the rotation in EXIF


def _to_cv(img: Image.Image) -> Array:
    return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _to_pil(arr: Array) -> Image.Image:
    if arr.ndim == 2:
        return Image.fromarray(arr)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def _order(pts: Array) -> Array:
    """Corners as top-left, top-right, bottom-right, bottom-left."""
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    corners = [pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]]
    return np.array(corners, dtype=np.float32)


def find_page(img: Image.Image) -> Array | None:
    """Four corners of the sheet of paper, or None if no clear sheet is visible
    (e.g. the photo is already cropped to the page)."""
    bgr = _to_cv(img)
    h, w = bgr.shape[:2]
    scale = 800 / max(h, w)
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c) / (small.shape[0] * small.shape[1])
    if not 0.25 < area < 0.97:
        return None
    quad = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    if len(quad) != 4:
        quad = cv2.boxPoints(cv2.minAreaRect(c))
    return _order(np.asarray(quad).reshape(4, 2).astype(np.float32) / scale)


def rectify(img: Image.Image) -> Image.Image:
    """The sheet warped to a flat, upright page; the photo as it is when no sheet is found."""
    corners = find_page(img)
    if corners is None:
        return img
    tl, tr, br, bl = corners
    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if width < 50 or height < 50:
        return img
    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(corners, dst)
    flat = cv2.warpPerspective(
        _to_cv(img), m, (width, height), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255)
    )
    return _to_pil(flat)


def part(img: Image.Image) -> ContentPart:
    """The photo as a content part of a request: flattened, as a JPEG."""
    try:
        img = rectify(img)
    except Exception:  # fault boundary: a photo that cannot be flattened is sent as it is
        logger.warning("the photo could not be flattened", exc_info=True)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=92)
    return file_part(buf.getvalue(), "image/jpeg")
