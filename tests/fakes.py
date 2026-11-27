"""Shared test helpers: fake model answers, a fake model server and generated documents.

No network."""

import asyncio
import io
import json
import re
from collections import Counter
from collections.abc import Callable
from decimal import Decimal
from typing import Any

import httpx
from pypdf import PdfWriter
from reportlab.pdfgen import canvas

from invoice_recognition.recognition import pipeline
from invoice_recognition.recognition.checks import Issue
from invoice_recognition.recognition.config import Config
from invoice_recognition.recognition.results import Result
from invoice_recognition.shapes import Header, JSONObject, Line


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


# --------------------------------------------------------------------------- #
# A fake OpenRouter for the flow
# --------------------------------------------------------------------------- #

type Answer = list[Any] | dict[str, Any] | str | httpx.Response | BaseException
"""What the fake server answers: a list of rows (sent as {"rows": [...]}), any other
JSON object, a raw SSE body, a ready response, or an exception to raise."""


class Server:
    """A fake OpenRouter that tells the requests of the flow apart by their prompt.

    header: answer to the header request. pages: {page: [answer of attempt 1, attempt 2,
    ...]}, the last one repeats. layout: answer to the layout request (text-layer PDFs).
    calls counts the requests: "header", "layout", and the page number of each page.
    """

    def __init__(
        self, header: Answer, pages: dict[int, list[Answer]], layout: Answer | None = None
    ) -> None:
        self.header, self.pages, self.layout = header, pages, layout
        self.calls: Counter[int | str] = Counter()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        prompt = body["messages"][0]["content"][0]["text"]
        answer: Answer | None
        if prompt.startswith("You read ONE page"):
            m = _PAGE_OF.search(prompt)
            assert m is not None
            page = int(m.group(1))
            answers = self.pages[page]
            answer = answers[min(self.calls[page], len(answers) - 1)]
            self.calls[page] += 1
        elif prompt.startswith("You are given the text layer"):
            # The layout request carries no file.
            assert [c["type"] for c in body["messages"][0]["content"]] == ["text"]
            answer = self.layout
            self.calls["layout"] += 1
        else:
            answer = self.header
            self.calls["header"] += 1
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, httpx.Response):
            return answer
        if isinstance(answer, str):
            return httpx.Response(200, text=answer)
        payload = {"rows": answer} if isinstance(answer, list) else answer
        return httpx.Response(200, text=sse(json.dumps(payload)))


_PAGE_OF = re.compile(r"page (\d+) of \d+")

type Handler = Callable[[httpx.Request], httpx.Response]


def run(data: bytes, mime: str, server: Handler, cfg: Config | None = None) -> Result:
    """process() on one file, with every request answered by `server`."""

    async def go() -> Result:
        async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
            return await pipeline.process(data, mime, client, cfg)

    return asyncio.run(go())


def recording(server: Handler) -> tuple[Handler, list[JSONObject]]:
    """The server, and the parsed bodies of the requests it got, in order."""
    bodies: list[JSONObject] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return server(request)

    return handler, bodies


def never(request: httpx.Request) -> httpx.Response:
    raise AssertionError("no request may be sent")


def scan(pages: int) -> bytes:
    """A PDF without a text layer: for the flow it is a scan (no evidence)."""
    w = PdfWriter()
    for _ in range(pages):
        w.add_blank_page(595, 842)
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


def codes(issues: list[Issue]) -> list[str]:
    return [i.check for i in issues]
