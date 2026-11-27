"""Try the page without a key: the real app, with every model request answered by a fake
model that knows the page's four samples. Free and offline; nothing leaves the machine.

    uv run python tests/demo_server.py [--port 8012] [--delay 0.4] [--access-code CODE]

then open http://127.0.0.1:8012/ and enter the access code (`try-the-demo-page-0001`
unless given). The fake reads each sample as a careful model would: the header from the
generated invoice's truth, its rows page by page, and the layout of a text-layer PDF as
the cheap model really described it. The misprinted sample is read as printed (one
amount 344.34 instead of 688.68). Any other file gets 503 from the "provider", so it
ends as failed / model_unavailable. Each answer comes after `--delay` seconds, so the
progress can be watched.

The fake knows which document a request belongs to from a context variable that the
reading service sets around each job (`DemoService`); tasks the flow starts copy it.
"""

import argparse
import asyncio
import hashlib
from contextvars import ContextVar

import httpx
import uvicorn
from fastapi import FastAPI
from pydantic import SecretStr

from invoice_recognition.api.app import create_app
from invoice_recognition.api.jobs import Job
from invoice_recognition.api.page import STATIC_DIR
from invoice_recognition.api.service import Service

from fakes import Answer, Server, api_settings
from samples import saved_layout, truth_doc

SAMPLES = STATIC_DIR / "samples"
SMALL = SAMPLES / "invoice-small-misprint.pdf"
"""m_grouped.pdf with line 58's amount (page 2) printed 344.34 instead of 688.68."""
ACCESS_CODE = "try-the-demo-page-0001"

# The amount of line 58 (page 2) of the small sample, as generated and as misprinted.
PRINTED, MISPRINTED = 688.68, 344.34

DOCUMENT: ContextVar[str | None] = ContextVar("document", default=None)
"""The sha256 of the document being read in this task."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def reader(name: str, *, layout: bool) -> Server:
    """A model that reads the generated invoice `name` right: its header, each page's
    rows and, for a text-layer PDF, the layout as the cheap model described it."""
    _, _, header, pages = truth_doc(name)
    answers: dict[int, list[Answer]] = {p: [rows] for p, rows in pages.items()}
    return Server(header, answers, saved_layout(name) if layout else None)


def misprint_reader() -> Server:
    """A model that reads the small sample as printed: line 58 with 344.34."""
    server = reader("m_grouped", layout=True)
    rows = server.pages[2][0]
    assert isinstance(rows, list)
    server.pages[2] = [[r | {"amount": MISPRINTED} if r["amount"] == PRINTED else r for r in rows]]
    return server


def readers() -> dict[str, Server]:
    """The sha256 of each sample file -> the fake model's answers to its requests."""
    by_file = {
        SMALL.name: misprint_reader(),
        "invoice-large.pdf": reader("xl_grouped", layout=True),
        "invoice-scan-simulated.pdf": reader("m_grouped", layout=False),
        "invoice-photo-simulated.jpg": reader("s_classic", layout=False),
    }
    return {digest((SAMPLES / name).read_bytes()): server for name, server in by_file.items()}


class FakeModel:
    """The transport handler: the answer of the sample being read, after `delay`
    seconds; 503 for a document the demo does not know."""

    def __init__(self, delay: float) -> None:
        self.delay, self.readers = delay, readers()

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(self.delay)
        reader = self.readers.get(DOCUMENT.get() or "")
        if reader is None:
            return httpx.Response(
                503, json={"error": {"message": "the demo reads only its samples"}}
            )
        return reader(request)


class DemoService(Service):
    """The reading service, telling the fake model which document each request is for."""

    async def read(self, job: Job) -> None:
        token = DOCUMENT.set(digest(job.data))
        try:
            await super().read(job)
        finally:
            DOCUMENT.reset(token)


def demo_app(*, delay: float, access_code: str) -> FastAPI:
    """The app on the fake model, reading through DemoService."""
    settings = api_settings(access_code=SecretStr(access_code))
    return create_app(
        settings, transport=httpx.MockTransport(FakeModel(delay)), service_class=DemoService
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--delay", type=float, default=0.4, help="seconds before each answer")
    parser.add_argument("--access-code", default=ACCESS_CODE, help="at least 16 characters")
    args = parser.parse_args()
    app = demo_app(delay=args.delay, access_code=args.access_code)
    print(f"Demo page: http://127.0.0.1:{args.port}/  (access code: {args.access_code})")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
