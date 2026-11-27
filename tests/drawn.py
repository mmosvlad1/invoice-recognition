"""Drawn photos and scans, and PDFs without a text layer, for the tests.

This module imports nothing from the package (and runs on older Pythons too), so
the same inputs can be built wherever a reference run needs them.
"""

import io
import time
import zlib
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


def declared_image_scan(width: int, height: int) -> bytes:
    """A one-page PDF whose page is covered by one gray image that DECLARES width x height
    pixels but stores only a few compressed bytes. Reading its size costs nothing; decoding
    it would allocate width x height bytes, which the tests never do. Written by hand: no
    PDF library here would write an image without having its pixels."""
    data = zlib.compress(bytes(1024))
    content = b"q 595 0 0 842 0 0 cm /Im0 Do Q"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            b"/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceGray "
        b"/BitsPerComponent 8 /Filter /FlateDecode /Length %d >>\nstream\n%s\nendstream"
        % (width, height, len(data), data),
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content),
    ]
    out = bytearray(b"%PDF-1.7\n")
    offsets = []
    for n, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (n, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % at for at in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)
