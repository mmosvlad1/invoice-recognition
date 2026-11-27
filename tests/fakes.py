"""Shared test helpers: fake model answers and generated documents. No network."""

import io
import json

from reportlab.pdfgen import canvas


def sse(text: str, cost: float | None = 0.001) -> str:
    """A streamed answer the way OpenRouter sends it: content, then usage, then [DONE].

    cost=None leaves the cost out of the usage, as some providers do.
    """
    usage = {"prompt_tokens": 100, "completion_tokens": 50} | (
        {"cost": cost} if cost is not None else {}
    )
    events = [
        {"choices": [{"delta": {"content": text}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": usage},
    ]
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"


def make_pdf(pages: list[list[str]], *, size: tuple[float, float] = (595, 842)) -> bytes:
    """A PDF with a text layer: one list of text rows per page, drawn top down.

    The output is byte-for-byte reproducible (reportlab's invariant mode), so a
    request body built from it can be compared across runs.
    """
    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=size, invariant=True)
    for rows in pages:
        pdf.setFont("Helvetica", 9)
        for n, row in enumerate(rows):
            pdf.drawString(40, size[1] - 60 - 14 * n, row)
        pdf.showPage()
    pdf.save()
    return buf.getvalue()
