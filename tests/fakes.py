"""Shared test helpers: fake model answers, a fake model server and generated documents.

No network."""

import asyncio
import io
import json
import re
from collections import Counter
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from decimal import Decimal
from typing import Any

import anyio
import httpx
from fastapi import FastAPI
from pydantic import SecretStr
from reportlab.pdfgen import canvas

from invoice_recognition.api.app import create_app
from invoice_recognition.api.jobs import FINAL
from invoice_recognition.api.settings import Settings
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
        kind = kind_of(request)
        answer: Answer | None
        if isinstance(kind, int):
            answers = self.pages[kind]
            answer = answers[min(self.calls[kind], len(answers) - 1)]
        elif kind == "layout":
            # The layout request carries no file.
            content = json.loads(request.content)["messages"][0]["content"]
            assert [c["type"] for c in content] == ["text"]
            answer = self.layout
        else:
            answer = self.header
        self.calls[kind] += 1
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, httpx.Response):
            return answer
        if isinstance(answer, str):
            return httpx.Response(200, text=answer)
        payload = {"rows": answer} if isinstance(answer, list) else answer
        return httpx.Response(200, text=sse(json.dumps(payload)))


_PAGE_OF = re.compile(r"page (\d+) of \d+")


def kind_of(request: httpx.Request) -> int | str:
    """Which request of the flow this is: "header", "layout", or the number of the page."""
    prompt = json.loads(request.content)["messages"][0]["content"][0]["text"]
    if prompt.startswith("You read ONE page"):
        m = _PAGE_OF.search(prompt)
        assert m is not None
        return int(m.group(1))
    if prompt.startswith("You are given the text layer"):
        return "layout"
    return "header"


type Handler = Callable[[httpx.Request], httpx.Response]


class SlowServer:
    """A fake server that answers asynchronously, each request after its own sleep, and
    counts the requests in flight. The answer itself comes from `server`.

    sleeps: {kind: [seconds before answering the 1st request of that kind, the 2nd, ...]},
    kind as kind_of() gives it; the last entry repeats. A kind not listed waits for no
    time, but still lets the event loop run once, as a real connection would.

    in_flight: requests in flight now, from their arrival until the client closes the
    answer (it has read the stream to the end, or given up on it); most / pages_most: the
    most at any time, of all requests / of page requests. A limit released before the
    answer is read therefore shows here. arrived: the kind of every request, in order.
    cancelled: the kinds of the requests the client gave up on while they were waited
    for (the sleep was cancelled)."""

    def __init__(
        self, server: Handler, sleeps: Mapping[int | str, Sequence[float]] | None = None
    ) -> None:
        self.server, self.sleeps = server, dict(sleeps or {})
        self.calls: Counter[int | str] = Counter()
        self.arrived: list[int | str] = []
        self.cancelled: list[int | str] = []
        self.in_flight = self.most = self.pages_in_flight = self.pages_most = 0

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        kind = kind_of(request)
        self.arrived.append(kind)
        delays = self.sleeps.get(kind, [0.0])
        delay = delays[min(self.calls[kind], len(delays) - 1)]
        self.calls[kind] += 1
        page = isinstance(kind, int)
        self.in_flight += 1
        self.pages_in_flight += page
        self.most = max(self.most, self.in_flight)
        self.pages_most = max(self.pages_most, self.pages_in_flight)

        def done() -> None:
            self.in_flight -= 1
            self.pages_in_flight -= page

        try:
            await asyncio.sleep(delay)
            answer = self.server(request)
            body = await answer.aread()
        except BaseException as e:
            if isinstance(e, asyncio.CancelledError):
                self.cancelled.append(kind)
            done()
            raise
        return httpx.Response(answer.status_code, stream=_Closing(body, done))


class _Closing(httpx.AsyncByteStream):
    """An answer that calls `on_close` once, when the client closes it."""

    def __init__(self, body: bytes, on_close: Callable[[], None]) -> None:
        self.body, self.on_close, self.closed = body, on_close, False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.body

    async def aclose(self) -> None:
        if not self.closed:
            self.closed = True
            self.on_close()


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


def codes(issues: list[Issue]) -> list[str]:
    return [i.check for i in issues]


# --------------------------------------------------------------------------- #
# The API
# --------------------------------------------------------------------------- #


# Any: each setting has its own type; Settings validates what it is given.
def api_settings(**overrides: Any) -> Settings:
    """Settings for a test app: every field given (its default unless overridden) and no
    .env read, so a developer's environment cannot leak in. The key is a dummy; no
    request leaves the test."""
    given = {name: f.default for name, f in Settings.model_fields.items() if not f.is_required()}
    given["openrouter_api_key"] = SecretStr("test-key")
    return Settings(_env_file=None, **(given | overrides))


@asynccontextmanager
async def api_client(
    app: FastAPI, *, raise_app_exceptions: bool = True
) -> AsyncIterator[httpx.AsyncClient]:
    """A client that sends its requests straight to `app`, in the test's own event loop.

    raise_app_exceptions=False: an exception the app does not handle comes back as the
    500 a server would send, instead of being raised in the test."""
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=raise_app_exceptions)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@asynccontextmanager
async def serve(
    server: Handler | SlowServer, settings: Settings | None = None
) -> AsyncIterator[tuple[httpx.AsyncClient, FastAPI]]:
    """The app started as a server starts it (its lifespan: the reading service runs),
    with every model request answered by `server`, and a client for it. The app stops on
    the way out."""
    app = create_app(settings or api_settings(), transport=httpx.MockTransport(server))
    async with app.router.lifespan_context(app), api_client(app) as client:
        yield client, app


async def wait_final(client: httpx.AsyncClient, job_id: str) -> JSONObject:
    """Poll the job until its status is final; its last answer."""
    with anyio.fail_after(5):
        while True:
            answer = await client.get(f"/invoices/{job_id}")
            assert answer.status_code == 200, answer.text
            body: JSONObject = answer.json()
            if body["status"] in FINAL:
                return body
            await anyio.sleep(0.01)
