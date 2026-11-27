"""Shared test helpers: fake model answers and generated documents. No network."""

import io
import json
from decimal import Decimal

from reportlab.pdfgen import canvas

from invoice_recognition.shapes import Header, Line


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


def line(no: int, qty: int = 2, price: float = 10.0, amount: float | None = None) -> Line:
    """A line item as the model returns it; amount defaults to qty x price."""
    return {
        "line_no": no,
        "sku": f"SKU-{no:03d}",
        "description": f"Item {no}",
        "quantity": qty,
        "unit": "pcs",
        "unit_price": price,
        "tax_rate": 20,
        "amount": qty * price if amount is None else amount,
    }


def header_for(lines: list[Line]) -> Header:
    """A header whose totals agree with `lines` at 20 % tax."""
    subtotal = sum(Decimal(str(item["amount"])) for item in lines)
    tax = (subtotal * Decimal("0.2")).quantize(Decimal("0.01"))
    return {
        "supplier_name": "ACME Ltd",
        "invoice_number": "INV-1",
        "currency": "EUR",
        "subtotal": float(subtotal),
        "tax_total": float(tax),
        "total": float(subtotal + tax),
    }
