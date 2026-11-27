"""Limits on the model requests in flight: one per document (Config.concurrency) and one
shared by every document given the same semaphore. Fake OpenRouter, no network."""

import asyncio

import httpx

from invoice_recognition.recognition import pipeline as pl
from invoice_recognition.recognition.config import Config
from invoice_recognition.recognition.results import Result

from drawn import scan
from fakes import Answer, Server, SlowServer, header_for, line

NO_MEASURE = Config(scan_quality=False)  # blank drawn pages: nothing to measure


def server_for(pages: int) -> Server:
    rows = {p: [line(p)] for p in range(1, pages + 1)}
    answers: dict[int, list[Answer]] = {p: [r] for p, r in rows.items()}
    return Server(header_for([item for r in rows.values() for item in r]), answers)


def test_documents_sharing_a_limit_never_exceed_it_together() -> None:
    server = SlowServer(server_for(10), {"header": [0.02], **{p: [0.02] for p in range(1, 11)}})

    async def go() -> list[Result]:
        slots = asyncio.Semaphore(3)
        async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
            return list(
                await asyncio.gather(
                    *(
                        pl.process(
                            scan(10), "application/pdf", client, NO_MEASURE, request_slots=slots
                        )
                        for _ in range(2)
                    )
                )
            )

    results = asyncio.run(go())
    assert server.most == 3  # reached, never passed
    assert len(server.arrived) == 2 * 11  # header + 10 pages, each document
    assert all(r.status == "ok" and len(r.lines) == 10 for r in results)


def test_without_a_shared_limit_a_document_keeps_its_own() -> None:
    # Today's behaviour: up to Config.concurrency (12) page requests, and the header
    # besides them.
    server = SlowServer(server_for(20), {"header": [0.2], **{p: [0.05] for p in range(1, 21)}})

    async def go() -> Result:
        async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
            return await pl.process(scan(20), "application/pdf", client, NO_MEASURE)

    res = asyncio.run(go())
    assert server.pages_most == 12 and server.most == 13
    assert res.status == "ok" and len(res.lines) == 20
