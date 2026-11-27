"""POST /invoices: the cheap checks of an upload, the size limit, the cap on documents
waiting, and the job it creates. Fake OpenRouter, no network.

The refusals themselves (which file is unreadable and why) are tested on the check in
test_api_uploads.py; here they are seen through the route."""

import io
import threading
from collections.abc import AsyncIterator
from typing import Any

import anyio
import httpx
import pytest
from pypdf import PdfReader, PdfWriter

from invoice_recognition.api import messages, uploads
from invoice_recognition.api.uploads import UploadLimits

import drawn
from fakes import (
    Server,
    SlowServer,
    api_settings,
    header_for,
    line,
    make_pdf,
    never,
    serve,
    wait_final,
)

PROBLEM = "application/problem+json"
MB = 1024 * 1024
LIMIT = 10 * MB  # the default
SLOW = [30.0]
BOUNDARY = "invoice-test-boundary"


def server(pages: int = 2) -> Server:
    rows = [line(p) for p in range(1, pages + 1)]
    return Server(header_for(rows), {p: [[line(p)]] for p in range(1, pages + 1)})


def held() -> SlowServer:
    """Every request waits far longer than any test: the documents stay unfinished."""
    return SlowServer(server(), {"header": SLOW, 1: SLOW, 2: SLOW})


async def post(
    client: httpx.AsyncClient, data: bytes, name: str = "invoice.pdf", mime: str = "x/y"
) -> httpx.Response:
    """Upload one file; the declared type is deliberately meaningless: content decides."""
    return await client.post("/invoices", files={"file": (name, data, mime)})


def problem(answer: httpx.Response, status: int, code: str) -> dict[str, Any]:
    assert answer.status_code == status, answer.text
    assert answer.headers["content-type"] == PROBLEM
    body: dict[str, Any] = answer.json()
    assert (body["status"], body["code"]) == (status, code)
    return body


def form(data: bytes, name: str = "big.pdf") -> bytes:
    """A multipart body with one file part, written out by hand."""
    head = (
        f"--{BOUNDARY}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{name}"\r\n'
        "Content-Type: application/pdf\r\n\r\n"
    )
    return head.encode() + data + f"\r\n--{BOUNDARY}--\r\n".encode()


FORM = {"content-type": f"multipart/form-data; boundary={BOUNDARY}"}


# --------------------------------------------------------------------------- #
# Accepted
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_an_upload_is_accepted_and_read_in_the_background() -> None:
    data = drawn.scan(2)
    async with serve(server()) as (client, _):
        answer = await post(client, data, "Invoice 42 «draft».pdf")
        assert answer.status_code == 202, answer.text
        body = answer.json()
        assert answer.headers["location"] == f"/invoices/{body['id']}"
        assert answer.headers["content-type"] == "application/json"
        final = await wait_final(client, body["id"])

    assert body["status"] == "queued" and body["result"] is None
    assert (body["filename"], body["kind"], body["size_bytes"]) == (
        "Invoice 42 «draft».pdf",
        "pdf",
        len(data),
    )
    assert body["progress"] == {"pages_total": 2, "pages_done": 0}
    assert final["status"] == "ok"
    assert [(item["index"], item["sku"]) for item in final["result"]["lines"]] == [
        (0, "SKU-001"),
        (1, "SKU-002"),
    ]


@pytest.mark.anyio
async def test_the_type_comes_from_the_content_not_the_name_or_the_declared_type() -> None:
    async with serve(never) as (client, _):
        photo = await post(client, drawn.png(drawn.photo()), "scan.pdf", "application/pdf")
        text = await post(client, b"%PDF? no, plain text", "invoice.pdf", "application/pdf")
    assert photo.status_code == 202 and photo.json()["kind"] == "image"
    problem(text, 415, "unsupported_file")


@pytest.mark.anyio
async def test_a_jpeg_cut_short_is_accepted_and_its_job_fails_as_unreadable() -> None:
    data = drawn.jpeg(drawn.photo())
    async with serve(never) as (client, _):
        answer = await post(client, data[: len(data) // 2], "photo.jpg")
        assert answer.status_code == 202, answer.text
        final = await wait_final(client, answer.json()["id"])
    assert final["status"] == "failed"
    assert final["error"]["code"] == "unreadable_file"
    assert final["progress"] == {"pages_total": 1, "pages_done": 0}


@pytest.mark.anyio
async def test_the_upload_check_runs_on_the_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    threads: list[str] = []
    real = uploads.inspect

    def spy(data: bytes, limits: UploadLimits) -> uploads.Upload:
        threads.append(threading.current_thread().name)
        return real(data, limits)

    monkeypatch.setattr(uploads, "inspect", spy)
    async with serve(server()) as (client, app):
        answer = await post(client, drawn.scan(2))
        assert answer.status_code == 202
        await wait_final(client, answer.json()["id"])
        lane = app.state.service.lane
    [name] = threads
    assert name.startswith("lane") and name != threading.current_thread().name
    assert lane._max_workers == 1  # the one lane, shared with the flow


# --------------------------------------------------------------------------- #
# Size
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_a_file_of_exactly_the_limit_is_checked_one_byte_more_is_refused() -> None:
    exact = b"%PDF-" + b"0" * (LIMIT - 5)
    async with serve(never) as (client, _):
        at_limit = await post(client, exact)
        over = await post(client, exact + b"0")
    # Not refused for its size: checked, and found not to be a PDF that opens.
    body = problem(at_limit, 422, "unreadable_file")
    assert body["detail"] == messages.PDF_DAMAGED
    body = problem(over, 413, "file_too_large")
    assert body == {
        "status": 413,
        "title": "Content Too Large",
        "code": "file_too_large",
        "detail": "The file is larger than the limit of 10 MB.",
    }


@pytest.mark.anyio
async def test_a_body_over_the_cap_with_a_declared_length_is_refused_before_it_is_read() -> None:
    body = form(b"%PDF-" + b"0" * (LIMIT + 64 * 1024))
    async with serve(never) as (client, _):
        answer = await client.post("/invoices", content=body, headers=FORM)
    # The framework answers it itself, before any handler: plain text (see the contract).
    assert answer.status_code == 413
    assert answer.headers["content-type"].startswith("text/plain")
    assert answer.text == "Content Too Large"


@pytest.mark.anyio
async def test_a_body_over_the_cap_sent_in_chunks_is_refused_as_a_problem() -> None:
    body = form(b"%PDF-" + b"0" * (LIMIT + 64 * 1024))

    async def chunks() -> AsyncIterator[bytes]:
        for at in range(0, len(body), MB):
            yield body[at : at + MB]

    async with serve(never) as (client, _):
        answer = await client.post("/invoices", content=chunks(), headers=FORM)
    assert "content-length" not in answer.request.headers
    problem(answer, 413, "file_too_large")


@pytest.mark.anyio
async def test_the_size_limit_comes_from_the_settings() -> None:
    async with serve(never, api_settings(max_file_bytes=1000)) as (client, _):
        answer = await post(client, drawn.png(drawn.photo()))
    body = problem(answer, 413, "file_too_large")
    assert body["detail"] == "The file is larger than the limit of 1,000 bytes."


# --------------------------------------------------------------------------- #
# The file
# --------------------------------------------------------------------------- #


def encrypted(user_password: str) -> bytes:
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(make_pdf([["Invoice"]]))))
    writer.encrypt(user_password=user_password, owner_password="owner", algorithm="AES-256")
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def no_pages() -> bytes:
    buf = io.BytesIO()
    PdfWriter().write(buf)
    return buf.getvalue()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("data", "code", "detail"),
    [
        (b"%PDF-1.7\nnothing else", "unreadable_file", messages.PDF_DAMAGED),
        (encrypted("user"), "unreadable_file", messages.PDF_NEEDS_PASSWORD),
        (no_pages(), "unreadable_file", messages.PDF_NO_PAGES),
        (b"\x89PNG\r\n\x1a\ngarbage", "unreadable_file", messages.IMAGE_DAMAGED),
        (
            drawn.scan(51),
            "too_many_pages",
            "The PDF has 51 pages; the limit is 50. Split the file and upload the parts.",
        ),
    ],
    ids=["damaged", "password", "no-pages", "damaged-image", "too-many-pages"],
)
async def test_a_file_that_cannot_be_read_is_refused(data: bytes, code: str, detail: str) -> None:
    async with serve(never) as (client, app):
        answer = await post(client, data)
        jobs = app.state.service.store.newest_first()
    assert problem(answer, 422, code)["detail"] == detail
    assert jobs == []  # no job is created for a refused file


@pytest.mark.anyio
async def test_an_image_over_the_pixel_limit_is_refused() -> None:
    async with serve(never, api_settings(max_image_pixels=1000)) as (client, _):
        answer = await post(client, drawn.png(drawn.photo()))
    body = problem(answer, 422, "unreadable_file")
    assert body["detail"] == messages.IMAGE_TOO_LARGE.format(limit="1,000 pixels")


@pytest.mark.anyio
async def test_a_pdf_page_limit_comes_from_the_settings() -> None:
    async with serve(never, api_settings(max_pages=2)) as (client, _):
        answer = await post(client, drawn.scan(3))
    problem(answer, 422, "too_many_pages")


@pytest.mark.anyio
async def test_anything_but_pdf_jpeg_png_is_unsupported() -> None:
    async with serve(never) as (client, _):
        answer = await post(client, b"Invoice 42\nTotal 100.00\n", "invoice.txt", "text/plain")
    assert problem(answer, 415, "unsupported_file")["detail"] == messages.UNSUPPORTED_FILE


# --------------------------------------------------------------------------- #
# The form
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_two_files_are_a_malformed_request() -> None:
    files = [("file", ("a.pdf", drawn.scan(1))), ("file", ("b.pdf", drawn.scan(1)))]
    async with serve(never) as (client, _):
        answer = await client.post("/invoices", files=files)
    assert problem(answer, 400, "invalid_request")["detail"] == messages.BAD_REQUEST


@pytest.mark.anyio
async def test_a_field_besides_the_file_is_a_malformed_request() -> None:
    async with serve(never) as (client, _):
        answer = await client.post(
            "/invoices", files={"file": ("a.pdf", drawn.scan(1))}, data={"note": "x"}
        )
    problem(answer, 400, "invalid_request")


@pytest.mark.anyio
async def test_no_file_part_is_an_invalid_request_that_names_the_field() -> None:
    async with serve(never) as (client, _):
        answer = await client.post("/invoices")
    body = problem(answer, 422, "invalid_request")
    assert [e["loc"] for e in body["errors"]] == [["body", "file"]]


# --------------------------------------------------------------------------- #
# The cap on documents waiting
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_one_document_over_the_cap_is_refused_with_retry_after() -> None:
    settings = api_settings(max_documents=2, reading_at_once=1)
    async with serve(held(), settings) as (client, app):
        first = await post(client, drawn.scan(2))
        second = await post(client, drawn.scan(2))
        third = await post(client, drawn.scan(2))
        statuses = [j.status for j in app.state.service.store.newest_first()]
    assert (first.status_code, second.status_code) == (202, 202)
    body = problem(third, 503, "too_many_documents")
    assert third.headers["retry-after"] == "30"
    assert body["detail"] == (
        "The server is busy with other documents: at most 2 can wait or be read at once. "
        "Please try again shortly."
    )
    assert statuses == ["queued", "processing"]


@pytest.mark.anyio
async def test_the_cap_is_checked_again_when_the_job_is_added(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two uploads pass the early check together; the one whose file check ends second
    finds no room left and is refused, though the early check let it in."""
    entered, release = threading.Event(), threading.Event()
    real = uploads.inspect

    def slow_first(data: bytes, limits: UploadLimits) -> uploads.Upload:
        if not entered.is_set():
            entered.set()
            assert release.wait(2), "the test never released the check"
        return real(data, limits)

    monkeypatch.setattr(uploads, "inspect", slow_first)
    async with serve(held(), api_settings(max_documents=1)) as (client, app):
        store = app.state.service.store
        early: list[bool] = []  # what the room check answered, in order
        has_room = store.has_room

        def counted() -> bool:
            early.append(has_room())
            return early[-1]

        monkeypatch.setattr(store, "has_room", counted)
        answers: list[httpx.Response] = []

        async def upload() -> None:
            answers.append(await post(client, drawn.scan(2)))

        async with anyio.create_task_group() as tg:
            tg.start_soon(upload)
            tg.start_soon(upload)
            # Both early checks found room; the first file's check is held on the lane.
            with anyio.fail_after(2):
                while len(early) < 2 or not entered.is_set():
                    await anyio.sleep(0.005)
            assert early == [True, True]
            release.set()
        jobs = app.state.service.store.newest_first()
    assert sorted(a.status_code for a in answers) == [202, 503]
    assert len(jobs) == 1


# --------------------------------------------------------------------------- #
# The documentation
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_the_upload_is_documented_with_its_refusals() -> None:
    async with serve(never) as (client, _):
        schema = (await client.get("/openapi.json")).json()
    operation = schema["paths"]["/invoices"]["post"]
    responses = operation["responses"]
    assert {"202", "400", "413", "415", "422", "503"} <= set(responses)
    for status in ("400", "413", "415", "422", "503"):
        assert list(responses[status]["content"]) == [PROBLEM]
    assert "Location" in responses["202"]["headers"]
    body = operation["requestBody"]["content"]
    assert list(body) == ["multipart/form-data"]
