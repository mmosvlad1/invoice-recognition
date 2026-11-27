"""Pick pages out of a PDF without re-rendering them (the text layer is preserved).

Each page is sent to the model as a PDF of its own, so the model reads the same
text layer and images as the original.
"""

import io

from pypdf import PdfReader, PdfWriter


def page_count(data: bytes) -> int:
    return len(PdfReader(io.BytesIO(data)).pages)


def extract_pages(data: bytes, pages: list[int]) -> bytes:
    """New PDF with the given 1-based pages, in the given order.

    A page number below 1 or past the end raises IndexError."""
    reader = PdfReader(io.BytesIO(data))
    writer = PdfWriter()
    for p in pages:
        if p < 1:  # pages[-1] would silently be the last page
            raise IndexError(f"page {p} is out of range")
        writer.add_page(reader.pages[p - 1])
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()
