"""Drawn photos and scans, and PDFs without a text layer, for the tests.

This module imports nothing from the package (and runs on older Pythons too), so
the same inputs can be built wherever a reference run needs them.
"""

import io
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from pypdf import PdfWriter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def scan(pages: int) -> bytes:
    """A PDF without a text layer: for the flow it is a scan (no evidence)."""
    w = PdfWriter()
    for _ in range(pages):
        w.add_blank_page(595, 842)
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


def scan_with_ocr_text(rows: list[dict[str, Any]]) -> bytes:
    """A scanned page (one image over the whole sheet) with a text layer laid over it by
    some OCR program, drawn as visible text."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(595, 842), invariant=True)
    c.drawImage(ImageReader(Image.new("L", (300, 424), 235)), 0, 0, width=595, height=842)
    c.setFont("Helvetica", 9)
    c.drawString(40, 800, "ACME Ltd   Invoice INV-1   SKU Description Qty Unit price Amount")
    for n, r in enumerate(rows):
        c.drawString(
            40,
            780 - 14 * n,
            f"{r['sku']} {r['description']} {r['quantity']} {r['unit_price']:.2f} {r['amount']:.2f}",
        )
    c.save()
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Drawn photos
# --------------------------------------------------------------------------- #

PHOTO_W, PHOTO_H = 600, 800
SHEET = (60, 60, 540, 740)  # x0, y0, x1, y1 of the sheet of paper in the frame
WORDS = ("INV-2026", "Hex bolt M8", "1.250,00", "20%", "324,00")


def photo(desk: int = 70, paper: int = 235, rows: int = 18) -> Image.Image:
    """A "photo" of an invoice: a light sheet with rows of words on a dark desk, in gray."""
    img = np.full((PHOTO_H, PHOTO_W), desk, np.uint8)
    x0, y0, x1, y1 = SHEET
    img[y0:y1, x0:x1] = paper
    for i in range(rows):
        for j, word in enumerate(WORDS):
            at = (x0 + 40 + j * 88, y0 + 60 + i * 30)
            cv2.putText(img, word, at, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 20, 1, cv2.LINE_AA)
    return Image.fromarray(img)


def jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Drawn scans
# --------------------------------------------------------------------------- #


def sheet(kind: str = "good") -> Image.Image:
    """One scanned page: paper from edge to edge, eighteen rows of words, 600 x 800 px."""
    img = np.full((PHOTO_H, PHOTO_W), 245, np.uint8)
    for i in range(18):
        for j, word in enumerate(WORDS):
            at = (40 + j * 105, 70 + i * 36)
            cv2.putText(img, word, at, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 20, 1, cv2.LINE_AA)
    page = Image.fromarray(img)
    if kind == "blurred":  # text sharpness about 4: far below the reject limit
        return page.filter(ImageFilter.GaussianBlur(3))
    if kind == "soft":  # about 134: between the reject limit (100) and the warn limit (200)
        return page.filter(ImageFilter.GaussianBlur(1.0))
    if kind == "dark":
        return Image.fromarray((np.asarray(page, dtype=np.float32) * 0.4).astype("uint8"))
    if kind == "blank":
        return Image.new("L", (PHOTO_W, PHOTO_H), 250)
    if kind == "back":  # the print of the other side showing through a blank back
        return Image.blend(Image.new("L", (PHOTO_W, PHOTO_H), 245), ImageOps.mirror(page), 0.1)
    assert kind == "good", kind
    return page


def scanned(*kinds: str) -> bytes:
    """A PDF whose every page is one image (one drawn sheet per kind), no text layer."""
    pages = [sheet(k) for k in kinds]
    buf = io.BytesIO()
    pages[0].save(
        buf,
        "PDF",
        save_all=True,
        append_images=pages[1:],
        resolution=72,
        quality=95,
        creationDate=time.gmtime(0),
        modDate=time.gmtime(0),
    )
    return buf.getvalue()


def scan_with_invisible_ocr(img: Image.Image) -> bytes:
    """A scanned page with the text layer an OCR program adds: text that is not painted."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(595, 842), invariant=True)
    c.drawImage(ImageReader(img), 0, 0, width=595, height=842)
    text = c.beginText(40, 800)
    text.setTextRenderMode(3)
    text.setFont("Helvetica", 9)
    for n in range(1, 6):
        text.textLine(f"SKU-00{n} Item {n} 2 10.00 20.00 ACME Ltd Invoice INV-1")
    c.drawText(text)
    c.save()
    return buf.getvalue()


def png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()
