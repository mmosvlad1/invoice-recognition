"""GET /invoices (every job held) and GET /invoices/{id}/file (the uploaded bytes).
Fake OpenRouter, no network."""

from typing import Any

import httpx
import pytest

from invoice_recognition.api import messages
from invoice_recognition.api.app import content_disposition

import drawn
from fakes import Server, SlowServer, api_settings, header_for, line, serve, wait_final

SLOW = [30.0]


def server() -> Server:
    return Server(header_for([line(1), line(2)]), {1: [[line(1)]], 2: [[line(2)]]})


async def upload(client: httpx.AsyncClient, data: bytes, name: str) -> str:
    answer = await client.post("/invoices", files={"file": (name, data)})
    assert answer.status_code == 202, answer.text
    job_id: str = answer.json()["id"]
    return job_id


async def items(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    answer = await client.get("/invoices")
    assert answer.status_code == 200
    assert answer.headers["cache-control"] == "no-store"
    assert list(answer.json()) == ["items"]
    found: list[dict[str, Any]] = answer.json()["items"]
    return found


# --------------------------------------------------------------------------- #
# The list
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_the_list_is_newest_first_without_results_with_counts_once_finished() -> None:
    # The first document is read at once; the second waits behind a held third.
    slow = SlowServer(server(), {"header": [0.0, 30.0]})
    async with serve(slow, api_settings(reading_at_once=1)) as (client, _):
        first = await upload(client, drawn.scan(2), "first.pdf")
        await wait_final(client, first)
        held = await upload(client, drawn.scan(2), "held.pdf")
        waiting = await upload(client, drawn.png(drawn.photo()), "waiting.png")
        listed = await items(client)

    assert [i["id"] for i in listed] == [waiting, held, first]
    assert [i["status"] for i in listed] == ["queued", "processing", "ok"]
    assert all("result" not in i for i in listed)
    assert [(i["line_count"], i["issue_count"]) for i in listed] == [
        (None, None),
        (None, None),
        (2, 0),
    ]
    assert listed[0] == {
        "id": waiting,
        "filename": "waiting.png",
        "kind": "image",
        "size_bytes": len(drawn.png(drawn.photo())),
        "status": "queued",
        "created_at": listed[0]["created_at"],
        "started_at": None,
        "finished_at": None,
        "progress": {"pages_total": 1, "pages_done": 0},
        "error": None,
        "line_count": None,
        "issue_count": None,
    }


@pytest.mark.anyio
async def test_an_empty_list() -> None:
    async with serve(server()) as (client, _):
        assert await items(client) == []


@pytest.mark.anyio
async def test_a_job_dropped_by_retention_is_gone_from_the_list_and_both_addresses() -> None:
    async with serve(server(), api_settings(max_documents=1, jobs_kept=1)) as (client, _):
        old = await upload(client, drawn.scan(2), "old.pdf")
        await wait_final(client, old)
        new = await upload(client, drawn.scan(2), "new.pdf")
        listed = await items(client)
        status = await client.get(f"/invoices/{old}")
        file = await client.get(f"/invoices/{old}/file")

    assert [i["id"] for i in listed] == [new]
    for answer in (status, file):
        assert answer.status_code == 404
        assert answer.headers["content-type"] == "application/problem+json"
        assert answer.json()["code"] == "not_found"
        assert answer.json()["detail"] == messages.JOB_NOT_FOUND


# --------------------------------------------------------------------------- #
# The file
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("data", "mime"),
    [
        (drawn.scan(2), "application/pdf"),
        (drawn.jpeg(drawn.photo()), "image/jpeg"),
        (drawn.png(drawn.photo()), "image/png"),
    ],
    ids=["pdf", "jpeg", "png"],
)
async def test_the_file_comes_back_as_it_was_uploaded(data: bytes, mime: str) -> None:
    async with serve(SlowServer(server(), {"header": SLOW, 1: SLOW, 2: SLOW})) as (client, _):
        job_id = await client.post("/invoices", files={"file": ("doc", data, "text/plain")})
        answer = await client.get(f"/invoices/{job_id.json()['id']}/file")

    assert answer.status_code == 200
    assert answer.content == data
    assert answer.headers["content-type"] == mime  # decided at upload, not as declared
    assert answer.headers["content-length"] == str(len(data))
    assert answer.headers["content-disposition"] == 'inline; filename="doc"'
    assert answer.headers["x-content-type-options"] == "nosniff"
    assert answer.headers["cache-control"] == "private, max-age=31536000, immutable"
    assert "accept-ranges" not in answer.headers and "content-encoding" not in answer.headers


@pytest.mark.anyio
async def test_a_range_request_gets_the_whole_file() -> None:
    data = drawn.scan(2)
    async with serve(SlowServer(server(), {"header": SLOW, 1: SLOW, 2: SLOW})) as (client, _):
        job_id = await upload(client, data, "a.pdf")
        answer = await client.get(f"/invoices/{job_id}/file", headers={"Range": "bytes=0-9"})
    assert answer.status_code == 200 and answer.content == data


@pytest.mark.anyio
async def test_the_file_of_an_unknown_id_is_not_found() -> None:
    async with serve(server()) as (client, _):
        answer = await client.get("/invoices/0123456789abcdef0123456789abcdef/file")
    assert answer.status_code == 404 and answer.json()["code"] == "not_found"


@pytest.mark.anyio
async def test_a_name_that_is_not_plain_ascii_is_sent_encoded() -> None:
    async with serve(SlowServer(server(), {"header": SLOW, 1: SLOW, 2: SLOW})) as (client, _):
        job_id = await upload(client, drawn.scan(2), "Рахунок №7.pdf")
        answer = await client.get(f"/invoices/{job_id}/file")
    expected = (
        "inline; filename*=utf-8''%D0%A0%D0%B0%D1%85%D1%83%D0%BD%D0%BE%D0%BA%20%E2%84%967.pdf"
    )
    assert answer.headers["content-disposition"] == expected


@pytest.mark.anyio
async def test_a_spoofing_name_is_not_shown_or_sent_as_sent() -> None:
    async with serve(SlowServer(server(), {"header": SLOW, 1: SLOW, 2: SLOW})) as (client, _):
        job_id = await upload(client, drawn.scan(2), "../../etc/\u202efdp.exe")
        shown = (await client.get(f"/invoices/{job_id}")).json()["filename"]
        listed = (await items(client))[0]["filename"]
        header = (await client.get(f"/invoices/{job_id}/file")).headers["content-disposition"]
    assert shown == listed == "fdp.exe"
    assert header == 'inline; filename="fdp.exe"'


@pytest.mark.parametrize(
    ("name", "header"),
    [
        ("invoice-42.pdf", 'inline; filename="invoice-42.pdf"'),
        ("my invoice.pdf", "inline; filename*=utf-8''my%20invoice.pdf"),
        ('a"b\\c.pdf', 'inline; filename="abc.pdf"'),
        ("evil\r\nSet-Cookie: x=1.pdf", "inline; filename*=utf-8''evilSet-Cookie%3A%20x%3D1.pdf"),
        ("tab\there\x7f\x85.pdf", 'inline; filename="tabhere.pdf"'),
        ("façade.pdf", "inline; filename*=utf-8''fa%C3%A7ade.pdf"),
        ('"\\\r\n', "inline"),
        ("", "inline"),
    ],
)
def test_content_disposition(name: str, header: str) -> None:
    assert content_disposition(name) == header
    header.encode("latin-1")  # always a valid header value
