"""Pick pages out of a PDF without re-rendering them (the text layer is preserved).

Each page is sent to the model as a PDF of its own, so the model reads the same
text layer and images as the original.
"""

import io

from pypdf import PdfReader, PdfWriter


def page_count(data: bytes) -> int:
    return len(PdfReader(io.BytesIO(data)).pages)


def extract_pages(data: bytes, pages: list[int]) -> bytes:
    """New PDF with the given 1-based pages, in the given order."""
    reader = PdfReader(io.BytesIO(data))
    writer = PdfWriter()
    for p in pages:
        writer.add_page(reader.pages[p - 1])
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()
