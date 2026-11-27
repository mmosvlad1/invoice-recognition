"""The reading service starts and stops with the app; GET /invoices/{id} reports a job.
Fake OpenRouter, no network."""

import threading

import anyio
import pytest
from fastapi import FastAPI

from invoice_recognition.api import messages
from invoice_recognition.api.app import create_app
from invoice_recognition.api.jobs import Job
from invoice_recognition.api.service import Service
from invoice_recognition.recognition import pipeline as pl
from invoice_recognition.recognition.config import Config

import drawn
from fakes import Server, SlowServer, api_client, api_settings, header_for, line, serve, wait_final

PDF = "application/pdf"
SLOW = [30.0]


def server() -> Server:
    return Server(header_for([line(1), line(2)]), {1: [[line(1)]], 2: [[line(2)]]})


def scan_job() -> Job:
    return Job("scan-0042.pdf", "pdf", PDF, drawn.scan(2), pages_total=2)


def service_of(app: FastAPI) -> Service:
    service = app.state.service
    assert isinstance(service, Service)
    return service


@pytest.mark.anyio
async def test_a_submitted_job_is_reported_at_once_and_then_with_its_result() -> None:
    # One reader, busy with a first job for a moment: the second one waits in the queue.
    slow_first = SlowServer(server(), {"header": [0.3, 0.0]})
    async with serve(slow_first, api_settings(reading_at_once=1)) as (client, app):
        service_of(app).submit(scan_job())
        job = scan_job()
        service_of(app).submit(job)
        answer = await client.get(f"/invoices/{job.id}")
        assert answer.status_code == 200
        assert answer.headers["cache-control"] == "no-store"
        assert answer.headers["content-type"] == "application/json"
        first = answer.json()
        final = await wait_final(client, job.id)

    assert first == {
        "id": job.id,
        "filename": "scan-0042.pdf",
        "kind": "pdf",
        "size_bytes": len(job.data),
        "status": "queued",
        "created_at": first["created_at"],
        "started_at": None,
        "finished_at": None,
        "progress": {"pages_total": 2, "pages_done": 0},
        "error": None,
        "result": None,
    }
    assert first["created_at"].endswith("Z")
    assert final["status"] == "ok" and final["error"] is None
    assert final["progress"] == {"pages_total": 2, "pages_done": 2}
    assert final["started_at"] is not None and final["finished_at"] is not None
    result = final["result"]
    assert list(result) == [
        "header",
        "lines",
        "issues",
        "notes",
        "checkpoints",
        "pages",
        "details",
    ]
    assert [(item["index"], item["sku"]) for item in result["lines"]] == [
        (0, "SKU-001"),
        (1, "SKU-002"),
    ]
    assert result["details"]["calls"] == 3  # the header and two pages


@pytest.mark.anyio
async def test_an_unknown_id_is_not_found() -> None:
    async with serve(server()) as (client, _):
        answer = await client.get("/invoices/0123456789abcdef0123456789abcdef")
    assert answer.status_code == 404
    assert answer.headers["content-type"] == "application/problem+json"
    assert answer.json() == {
        "status": 404,
        "title": "Not Found",
        "code": "not_found",
        "detail": messages.JOB_NOT_FOUND,
    }


@pytest.mark.anyio
async def test_status_is_answered_while_a_document_is_prepared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release, entered = threading.Event(), threading.Event()
    real = pl.prepare

    def held(data: bytes, mime: str, cfg: Config) -> pl.Prepared:
        entered.set()
        assert release.wait(2), "the test never released the lane"
        return real(data, mime, cfg)

    monkeypatch.setattr(pl, "prepare", held)
    async with serve(server()) as (client, app):
        job = scan_job()
        service_of(app).submit(job)
        try:
            with anyio.fail_after(1):
                while not entered.is_set():
                    await anyio.sleep(0.005)
                answer = await client.get(f"/invoices/{job.id}")
            assert answer.json()["status"] == "processing"
        finally:
            release.set()
        assert (await wait_final(client, job.id))["status"] == "ok"


@pytest.mark.anyio
async def test_stopping_the_app_cancels_the_workers_and_closes_the_client() -> None:
    held = SlowServer(server(), {"header": SLOW, 1: SLOW, 2: SLOW})
    async with serve(held) as (_client, app):
        service = service_of(app)
        workers = list(service._workers)
        assert len(workers) == 3 and not service.client.is_closed
        service.submit(scan_job())
        with anyio.fail_after(2):
            while held.in_flight < 3:
                await anyio.sleep(0.005)

    assert all(task.done() for task in workers)
    assert service.client.is_closed and held.in_flight == 0


@pytest.mark.anyio
async def test_without_the_lifespan_a_route_that_needs_the_service_fails_loudly() -> None:
    app = create_app(api_settings())
    async with api_client(app, raise_app_exceptions=False) as client:
        answer = await client.get("/invoices/x")
    assert answer.status_code == 500 and answer.json()["code"] == "internal_error"
