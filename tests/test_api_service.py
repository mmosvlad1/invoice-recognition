"""The reading service: jobs are read in the background by a few workers, one at a time
each, and every job ends with a final status. Fake OpenRouter, no network."""

import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import anyio
import httpx
import pytest
from PIL import ImageFilter

from invoice_recognition.api import service as sv
from invoice_recognition.api import views
from invoice_recognition.api.jobs import Job
from invoice_recognition.api.service import Service
from invoice_recognition.recognition import pipeline as pl
from invoice_recognition.recognition.config import FAST
from invoice_recognition.recognition.results import Result

import drawn
from fakes import Answer, Handler, Server, SlowServer, api_settings, header_for, line, never
from samples import saved_layout, truth_doc

PDF = "application/pdf"
SLOW = [30.0]  # far beyond any test: such a request is always cancelled, never waited for


@asynccontextmanager
async def running(server: Handler | SlowServer, **settings: Any) -> AsyncIterator[Service]:
    """A started service whose model requests go to `server`; stopped on the way out."""
    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        service = Service(api_settings(**settings), client)
        service.start()
        try:
            yield service
        finally:
            await service.stop()


async def until(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    with anyio.fail_after(timeout):
        while not condition():
            await anyio.sleep(0.005)


def scan_job(pages: int = 2) -> Job:
    """A PDF without a text layer whose blank pages are not measured: read page by page."""
    return Job("scan.pdf", "pdf", PDF, drawn.scan(pages), pages_total=pages)


def scan_server(pages: int = 2) -> Server:
    rows = [line(p) for p in range(1, pages + 1)]
    return Server(header_for(rows), {p: [[line(p)]] for p in range(1, pages + 1)})


@pytest.mark.anyio
async def test_a_job_goes_from_queued_through_processing_to_ok() -> None:
    server = SlowServer(scan_server(), {"header": [0.2]})
    async with running(server) as service:
        job = scan_job()
        service.submit(job)
        assert (job.status, job.started_at) == ("queued", None)
        assert service.store.get(job.id) is job

        await until(lambda: job.status == "processing")
        assert job.started_at is not None and job.finished_at is None
        await until(lambda: job.finished)

    assert job.status == "ok" and job.error is None
    assert job.pages_done == job.pages_total == 2
    assert job.result is not None and job.view is not None
    assert [item.sku for item in job.view.lines] == ["SKU-001", "SKU-002"]
    assert job.finished_at is not None and job.started_at <= job.finished_at


@pytest.mark.anyio
async def test_a_document_with_an_issue_ends_as_needs_review_without_an_error() -> None:
    wrong = line(2, amount=99.0)  # 2 x 10 is not 99
    server = Server(header_for([line(1), wrong]), {1: [[line(1)]], 2: [[wrong]]})
    async with running(server) as service:
        job = scan_job()
        service.submit(job)
        await until(lambda: job.finished)

    assert job.status == "needs_review" and job.error is None
    assert job.view is not None and "line_math" in [i.code for i in job.view.issues]


@pytest.mark.anyio
async def test_the_flow_gets_the_settings_the_lane_and_the_shared_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}
    real = pl.process

    async def spy(*args: Any, **kwargs: Any) -> Result:
        seen.update(cfg=args[3], **kwargs)
        return await real(*args, **kwargs)

    monkeypatch.setattr(pl, "process", spy)
    settings = {
        "page_model": "p/m",
        "header_model": "h/m",
        "layout_model": "l/m",
        "deadline_s": 99.0,
        "model_requests_in_flight": 7,
    }
    async with running(scan_server(), **settings) as service:
        job = scan_job()
        service.submit(job)
        await until(lambda: job.finished)

    cfg = seen["cfg"]
    assert cfg.page_models == ("p/m", "p/m") and cfg.header_model == "h/m"
    assert (cfg.layout_model, cfg.deadline_s) == ("l/m", 99.0)
    assert cfg.layout_witness and cfg.use_layout  # the witness is not configurable
    assert seen["lane"] is service.lane and seen["request_slots"] is service.request_slots
    assert service.request_slots._value == 7  # the semaphore's own count, nothing else shows it


def test_the_pixel_cap_of_the_upload_checks_holds_for_the_pages_of_a_scan() -> None:
    assert sv.config(api_settings(max_image_pixels=1234)).max_image_pixels == 1234
    assert sv.config(api_settings()).max_image_pixels == 50_000_000


def test_the_default_models_are_the_cheap_one() -> None:
    cfg = sv.config(api_settings())
    assert cfg.page_models == (FAST, FAST) and cfg.header_model == cfg.layout_model == FAST
    assert cfg.deadline_s == 240


@pytest.mark.anyio
async def test_with_one_reader_the_second_job_waits_in_the_queue() -> None:
    held = SlowServer(scan_server(), {"header": SLOW})
    async with running(held, reading_at_once=1) as service:
        first, second = scan_job(), scan_job()
        service.submit(first)
        service.submit(second)
        await until(lambda: held.calls["header"] == 1)
        await anyio.sleep(0.1)
        assert (first.status, second.status) == ("processing", "queued")
        assert second.started_at is None and second.pages_done == 0


@pytest.mark.anyio
async def test_a_flow_that_raises_fails_its_job_and_the_next_job_runs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    real = pl.process
    calls = 0

    async def breaks_once(*args: Any, **kwargs: Any) -> Result:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("a bug in the flow")
        return await real(*args, **kwargs)

    monkeypatch.setattr(pl, "process", breaks_once)
    async with running(scan_server(), reading_at_once=1) as service:
        broken, fine = scan_job(), scan_job()
        service.submit(broken)
        service.submit(fine)
        await until(lambda: fine.finished)

    assert broken.status == "failed" and broken.finished_at is not None
    assert broken.error is not None and broken.error.code == "internal_error"
    assert (broken.result, broken.view) == (None, None)
    assert fine.status == "ok"
    [record] = [r for r in caplog.records if r.name == sv.__name__]
    assert record.levelno == logging.ERROR and broken.id in record.getMessage()
    assert record.exc_info is not None and "a bug in the flow" in str(record.exc_info[1])


@pytest.mark.anyio
async def test_a_result_the_view_cannot_show_fails_its_job_and_the_next_job_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = views.result_view
    calls = 0

    def breaks_once(result: Result) -> views.ResultView:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("a value the view cannot hold")
        return real(result)

    monkeypatch.setattr(views, "result_view", breaks_once)
    async with running(scan_server(), reading_at_once=1) as service:
        broken, fine = scan_job(), scan_job()
        service.submit(broken)
        service.submit(fine)
        await until(lambda: fine.finished)

    assert broken.status == "failed" and broken.error is not None
    assert broken.error.code == "internal_error" and broken.result is None
    assert fine.status == "ok" and fine.view is not None


@pytest.mark.anyio
async def test_a_blank_page_of_a_scan_counts_as_done() -> None:
    data = drawn.scanned("good", "blank")
    server = Server(header_for([line(1)]), {1: [[line(1)]]})
    async with running(server) as service:
        job = Job("scan.pdf", "pdf", PDF, data, pages_total=2)
        service.submit(job)
        await until(lambda: job.finished)

    assert job.pages_done == 2
    assert server.calls[2] == 0  # the blank page was not sent
    assert job.view is not None and [p.method for p in job.view.pages] == ["model", "skipped"]


@pytest.mark.anyio
async def test_a_rejected_photo_stays_at_no_page_done() -> None:
    data = drawn.jpeg(drawn.photo().filter(ImageFilter.GaussianBlur(3)))
    async with running(never) as service:
        job = Job("photo.jpg", "image", "image/jpeg", data, pages_total=1)
        service.submit(job)
        await until(lambda: job.finished)

    assert (job.status, job.pages_done, job.pages_total) == ("failed", 0, 1)
    assert job.error is not None and job.error.code == "photo_unreadable"


@pytest.mark.anyio
async def test_the_deadline_fails_the_job_and_keeps_what_was_read() -> None:
    data, _, header, pages = truth_doc("m_grouped")
    answers: dict[int, list[Answer]] = {p: [rows] for p, rows in pages.items()}
    server = SlowServer(Server(header, answers, saved_layout("m_grouped")), {"layout": SLOW})
    async with running(server, deadline_s=1.5) as service:
        job = Job("m.pdf", "pdf", PDF, data, pages_total=4)
        service.submit(job)
        await until(lambda: job.finished)

    assert job.status == "failed"
    assert job.error is not None and job.error.code == "deadline_exceeded"
    assert job.view is not None
    assert job.view.lines and {item.page for item in job.view.lines} == {1}  # the witness page
    assert "deadline_exceeded" in [i.code for i in job.view.issues]
    assert job.pages_done == 4  # an unread page is settled too: it will not be read


@pytest.mark.anyio
async def test_progress_never_goes_back_or_past_the_total(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []
    real = pl.process

    async def announces_twice(*args: Any, **kwargs: Any) -> Result:
        data, mime, client, cfg, on_page = args

        def twice(page: Any) -> None:
            on_page(page)
            on_page(page)
            seen.append(job.pages_done)

        return await real(data, mime, client, cfg, twice, **kwargs)

    monkeypatch.setattr(pl, "process", announces_twice)
    async with running(scan_server(3)) as service:
        job = scan_job(3)
        service.submit(job)
        await until(lambda: job.finished)

    assert sorted(seen) == seen == [1, 2, 3]


@pytest.mark.anyio
async def test_progress_stops_at_the_total_when_more_pages_are_announced() -> None:
    # The flow reads three pages of a job that was told it has one: progress stays at 1 of 1.
    async with running(scan_server(3)) as service:
        job = Job("scan.pdf", "pdf", PDF, drawn.scan(3), pages_total=1)
        service.submit(job)
        await until(lambda: job.finished)
    assert job.status == "ok" and job.pages_done == 1


@pytest.mark.anyio
async def test_stop_shuts_the_lane_down() -> None:
    async with running(scan_server()) as service:
        lane = service.lane
        assert lane.submit(int).result(timeout=2) == 0
    with pytest.raises(RuntimeError, match="shutdown"):
        lane.submit(int)


@pytest.mark.anyio
async def test_stop_cancels_a_job_in_flight_promptly_and_leaves_no_task() -> None:
    held = SlowServer(scan_server(), {"header": SLOW, 1: SLOW, 2: SLOW})
    async with httpx.AsyncClient(transport=httpx.MockTransport(held)) as client:
        service = Service(api_settings(), client)
        service.start()
        workers = list(service._workers)
        job = scan_job()
        service.submit(job)
        await until(lambda: held.in_flight == 3)

        started = time.monotonic()
        await service.stop()
        took = time.monotonic() - started

    assert took < 2
    assert all(t.done() for t in workers) and not service._workers
    assert held.in_flight == 0 and sorted(map(str, held.cancelled)) == ["1", "2", "header"]
    assert job.status == "processing"  # the process is ending: the job is not recorded as read
