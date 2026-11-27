"""Decisions on the issues of a read document, through the API: PUT and DELETE
/invoices/{id}/issues/{issue_id}/resolution, the review in GET /invoices/{id} and the
open count in the list. Fake OpenRouter, no network."""

import json
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from pydantic import SecretStr

from invoice_recognition.api import app as app_module
from invoice_recognition.api import messages
from invoice_recognition.api.jobs import Job
from invoice_recognition.api.service import Service
from invoice_recognition.api.views import result_view
from invoice_recognition.recognition.checks import Issue
from invoice_recognition.recognition.results import PageResult, Result
from invoice_recognition.shapes import JSONObject

import drawn
from demo_server import MISPRINTED, SMALL, misprint_reader
from fakes import Server, SlowServer, api_settings, header_for, line, serve, wait_final

CODE = "s3cret-Code_42-long-enough"
RIGHT = {"Authorization": f"Bearer {CODE}"}

# --------------------------------------------------------------------------- #
# A generated invoice with one amount misprinted
# --------------------------------------------------------------------------- #

# The small sample of the page: line 58 of m_grouped (page 2) is 18 x 38.26 = 688.68,
# printed as 344.34, with the same six characters.


def misprinted() -> bytes:
    return SMALL.read_bytes()


# What the flow finds on it: the line, a page card, and three issues of the document's
# arithmetic that follow from the line's amount.
LINE_MATH, COVERAGE, SUBTOTAL, TAX, SECTION = range(5)
SUGGESTIONS = [
    {"set": {"line.quantity": 9}, "source": "arithmetic"},
    {"set": {"line.unit_price": 19.13}, "source": "arithmetic"},
    {"set": {"line.amount": 688.68}, "source": "arithmetic", "agrees_with": "header.subtotal"},
]
QUANTITY, UNIT_PRICE, AMOUNT = range(3)


async def upload(client: httpx.AsyncClient, data: bytes, headers: dict[str, str]) -> str:
    answer = await client.post("/invoices", files={"file": ("m.pdf", data)}, headers=headers)
    assert answer.status_code == 202, answer.text
    job_id: str = answer.json()["id"]
    return job_id


async def read(client: httpx.AsyncClient) -> str:
    """Upload the misprinted invoice, wait until it is read; its id."""
    job_id = await upload(client, misprinted(), {})
    await wait_final(client, job_id)
    return job_id


def path(job_id: str, issue_id: int | str) -> str:
    return f"/invoices/{job_id}/issues/{issue_id}/resolution"


ACCEPT_AMOUNT = {"action": "accept", "suggestion": AMOUNT}
KEEP = {"action": "keep"}


def states(review: JSONObject) -> list[tuple[str, list[int]]]:
    return [(i["state"], i["settled_by"]) for i in review["issues"]]


def problem(answer: httpx.Response, status: int, code: str) -> JSONObject:
    assert answer.status_code == status, answer.text
    assert answer.headers["content-type"] == "application/problem+json"
    body: JSONObject = answer.json()
    assert (body["status"], body["code"]) == (status, code)
    return body


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[datetime]]:
    """The times decisions are taken at, one per decision request, in order."""
    times = [datetime(2026, 11, 27, 15, 2, 11 + n, tzinfo=UTC) for n in range(20)]
    taken = iter(times)
    monkeypatch.setattr(app_module, "now", lambda: next(taken))
    yield times


# --------------------------------------------------------------------------- #
# Deciding
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_the_misprinted_invoice_has_one_line_and_what_follows_from_it() -> None:
    async with serve(misprint_reader()) as (client, _):
        job_id = await read(client)
        body = (await client.get(f"/invoices/{job_id}")).json()
        listed = (await client.get("/invoices")).json()["items"]
    assert body["status"] == "needs_review"
    assert [p["method"] for p in body["result"]["pages"]] == ["layout"] * 4
    issues = body["result"]["issues"]
    assert [(i["code"], i["follows_from"]) for i in issues] == [
        ("line_math", []),
        ("coverage", []),
        ("doc_sums", [LINE_MATH]),
        ("tax", [LINE_MATH]),
        ("checkpoints", [LINE_MATH]),
    ]
    assert issues[LINE_MATH]["suggestions"] == SUGGESTIONS
    assert body["result"]["lines"][issues[LINE_MATH]["line_index"]]["amount"] == MISPRINTED
    assert body["review"] == {
        "open_count": 5,
        "issues": [
            {
                "id": n,
                "state": "open",
                "suggestion": None,
                "set": None,
                "decided_at": None,
                "settled_by": [],
            }
            for n in range(5)
        ],
    }
    assert (listed[0]["issue_count"], listed[0]["open_issue_count"]) == (5, 5)


@pytest.mark.anyio
async def test_accepting_the_amount_settles_what_follows_and_changes_nothing_read(
    clock: list[datetime],
) -> None:
    async with serve(misprint_reader()) as (client, _):
        job_id = await read(client)
        before = (await client.get(f"/invoices/{job_id}")).json()
        answer = await client.put(path(job_id, LINE_MATH), json=ACCEPT_AMOUNT)
        after = (await client.get(f"/invoices/{job_id}")).json()
        listed = (await client.get("/invoices")).json()["items"]

    assert answer.status_code == 200, answer.text
    assert answer.headers["cache-control"] == "no-store"
    assert answer.json() == after["review"]
    assert after["review"]["open_count"] == 1
    assert after["review"]["issues"][LINE_MATH] == {
        "id": LINE_MATH,
        "state": "accepted",
        "suggestion": AMOUNT,
        "set": {"line.amount": 688.68},
        "decided_at": "2026-11-27T15:02:11Z",
        "settled_by": [],
    }
    assert states(after["review"]) == [
        ("accepted", []),
        ("open", []),
        ("settled", [LINE_MATH]),
        ("settled", [LINE_MATH]),
        ("settled", [LINE_MATH]),
    ]
    # The reading's verdict stays: values as read, the same issues, the same status.
    for key in ("result", "status", "error", "finished_at"):
        assert after[key] == before[key], key
    assert (listed[0]["status"], listed[0]["issue_count"]) == ("needs_review", 5)
    assert listed[0]["open_issue_count"] == 1


@pytest.mark.anyio
@pytest.mark.parametrize(
    "decision",
    [
        {"action": "accept", "suggestion": QUANTITY},
        {"action": "accept", "suggestion": UNIT_PRICE},
        KEEP,
    ],
    ids=["quantity", "unit-price", "keep"],
)
async def test_anything_but_the_amount_settles_nothing(decision: JSONObject) -> None:
    async with serve(misprint_reader()) as (client, _):
        job_id = await read(client)
        answer = await client.put(path(job_id, LINE_MATH), json=decision)
    assert answer.status_code == 200, answer.text
    assert [s for s, _ in states(answer.json())][SUBTOTAL:] == ["open"] * 3
    assert answer.json()["open_count"] == 4


@pytest.mark.anyio
async def test_undo_reopens_and_undoing_nothing_is_fine() -> None:
    async with serve(misprint_reader()) as (client, _):
        job_id = await read(client)
        await client.put(path(job_id, LINE_MATH), json=ACCEPT_AMOUNT)
        undone = await client.delete(path(job_id, LINE_MATH))
        again = await client.delete(path(job_id, LINE_MATH))
        never_decided = await client.delete(path(job_id, COVERAGE))
    assert undone.status_code == again.status_code == never_decided.status_code == 200
    assert undone.headers["cache-control"] == "no-store"
    assert states(undone.json()) == [("open", [])] * 5
    assert again.json() == never_decided.json() == undone.json()


@pytest.mark.anyio
async def test_the_same_decision_twice_changes_nothing_and_another_replaces_it(
    clock: list[datetime],
) -> None:
    async with serve(misprint_reader()) as (client, _):
        job_id = await read(client)
        first = await client.put(path(job_id, LINE_MATH), json=ACCEPT_AMOUNT)
        second = await client.put(path(job_id, LINE_MATH), json=ACCEPT_AMOUNT)
        other = await client.put(path(job_id, LINE_MATH), json=KEEP)
    assert first.json() == second.json()  # decided_at too: the first time is kept
    assert second.json()["issues"][LINE_MATH]["decided_at"] == "2026-11-27T15:02:11Z"
    kept = other.json()["issues"][LINE_MATH]
    assert (kept["state"], kept["set"], kept["decided_at"]) == (
        "kept",
        None,
        "2026-11-27T15:02:13Z",
    )


@pytest.mark.anyio
async def test_every_issue_checked() -> None:
    async with serve(misprint_reader()) as (client, _):
        job_id = await read(client)
        await client.put(path(job_id, COVERAGE), json=KEEP)  # "mark as checked"
        answer = await client.put(path(job_id, LINE_MATH), json=ACCEPT_AMOUNT)
        listed = (await client.get("/invoices")).json()["items"]
    assert answer.json()["open_count"] == 0
    assert (listed[0]["status"], listed[0]["open_issue_count"]) == ("needs_review", 0)


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_an_unknown_document_or_issue_is_not_found() -> None:
    async with serve(misprint_reader()) as (client, _):
        job_id = await read(client)
        answers = {
            "job-put": await client.put(path("0" * 32, 0), json=KEEP),
            "job-delete": await client.delete(path("0" * 32, 0)),
            "issue-put": await client.put(path(job_id, 5), json=KEEP),
            "issue-delete": await client.delete(path(job_id, 5)),
            "negative": await client.put(path(job_id, -1), json=KEEP),
        }
        not_an_id = await client.put(path(job_id, "x"), json=KEEP)
    for name in ("job-put", "job-delete"):
        assert problem(answers[name], 404, "not_found")["detail"] == messages.JOB_NOT_FOUND
    for name in ("issue-put", "issue-delete", "negative"):
        body = problem(answers[name], 404, "issue_not_found")
        assert body["detail"] == messages.ISSUE_NOT_FOUND
    problem(not_an_id, 422, "invalid_request")


@pytest.mark.anyio
async def test_nothing_to_accept_is_an_invalid_resolution() -> None:
    async with serve(misprint_reader()) as (client, _):
        job_id = await read(client)
        none = await client.put(path(job_id, COVERAGE), json={"action": "accept", "suggestion": 0})
        past = await client.put(path(job_id, LINE_MATH), json={"action": "accept", "suggestion": 3})
        review = (await client.get(f"/invoices/{job_id}")).json()["review"]
    assert problem(none, 422, "invalid_resolution")["detail"] == messages.NO_SUGGESTIONS
    assert "no number 3" in problem(past, 422, "invalid_resolution")["detail"]
    assert review["open_count"] == 5


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body",
    [
        {"action": "accept"},
        {"action": "accept", "suggestion": "2"},
        {"action": "accept", "suggestion": 2.0},
        {"action": "accept", "suggestion": True},
        {"action": "accept", "suggestion": -1},
        {"action": "keep", "suggestion": 0},
        {"action": "keep", "why": "looked"},
        {"action": "reject"},
        {},
        [KEEP],
        "keep",
    ],
)
async def test_a_body_off_the_schema_is_an_invalid_request(body: Any) -> None:
    async with serve(misprint_reader()) as (client, _):
        job_id = await read(client)
        answer = await client.put(path(job_id, LINE_MATH), json=body)
        review = (await client.get(f"/invoices/{job_id}")).json()["review"]
    assert problem(answer, 422, "invalid_request")["errors"]
    assert review["open_count"] == 5


@pytest.mark.anyio
async def test_a_body_that_is_not_json_is_an_invalid_request() -> None:
    async with serve(misprint_reader()) as (client, _):
        job_id = await read(client)
        headers = {"Content-Type": "application/json"}
        answers = [
            await client.put(path(job_id, LINE_MATH), content=b"keep", headers=headers),
            await client.put(path(job_id, LINE_MATH)),
        ]
    for answer in answers:
        problem(answer, 422, "invalid_request")


@pytest.mark.anyio
async def test_a_body_over_1_kib_is_refused() -> None:
    body = json.dumps(KEEP | {"pad": "x" * 1024}).encode()
    async with serve(misprint_reader()) as (client, _):
        job_id = await read(client)
        answer = await client.put(
            path(job_id, COVERAGE), content=body, headers={"Content-Type": "application/json"}
        )
        review = (await client.get(f"/invoices/{job_id}")).json()["review"]
    assert answer.status_code == 413
    assert review["open_count"] == 5


@pytest.mark.anyio
async def test_a_body_over_1_kib_sent_in_chunks_is_a_body_too_large_problem() -> None:
    body = json.dumps(KEEP | {"pad": "x" * 2048}).encode()

    async def chunks() -> AsyncIterator[bytes]:
        for at in range(0, len(body), 512):
            yield body[at : at + 512]

    async with serve(misprint_reader()) as (client, _):
        job_id = await read(client)
        answer = await client.put(
            path(job_id, COVERAGE), content=chunks(), headers={"Content-Type": "application/json"}
        )
        review = (await client.get(f"/invoices/{job_id}")).json()["review"]
    assert "content-length" not in answer.request.headers
    body_problem = problem(answer, 413, "body_too_large")
    assert body_problem["detail"] == messages.BODY_TOO_LARGE
    assert review["open_count"] == 5


def held() -> SlowServer:
    return SlowServer(misprint_reader(), {"header": [30.0]})


@pytest.mark.anyio
async def test_a_document_not_yet_read_cannot_be_decided() -> None:
    async with serve(held(), api_settings(reading_at_once=1)) as (client, _):
        processing = await upload(client, misprinted(), {})
        queued = await upload(client, misprinted(), {})
        answers = [
            await client.put(path(job_id, 0), json=KEEP) for job_id in (processing, queued)
        ] + [await client.delete(path(job_id, 0)) for job_id in (processing, queued)]
        jobs = [(await client.get(f"/invoices/{j}")).json() for j in (processing, queued)]
    assert [j["status"] for j in jobs] == ["processing", "queued"]
    assert [j["review"] for j in jobs] == [None, None]
    for answer in answers:
        body = problem(answer, 409, "not_reviewable")
        assert body["detail"] == messages.NOT_REVIEWABLE["unfinished"]


@pytest.mark.anyio
async def test_a_failed_document_cannot_be_decided() -> None:
    down = httpx.Response(503, text="no")
    async with serve(Server(down, {1: [down]})) as (client, _):
        job_id = await upload(client, drawn.scan(1), {})
        body = await wait_final(client, job_id)
        answer = await client.put(path(job_id, 0), json=KEEP)
        listed = (await client.get("/invoices")).json()["items"]
    assert body["status"] == "failed" and body["result"]["issues"]
    assert body["review"] is None and listed[0]["open_issue_count"] is None
    assert problem(answer, 409, "not_reviewable")["detail"] == messages.NOT_REVIEWABLE["failed"]


@pytest.mark.anyio
async def test_a_document_dropped_by_retention_is_not_found() -> None:
    settings = api_settings(max_documents=1, jobs_kept=1)
    async with serve(misprint_reader(), settings) as (client, _):
        first = await read(client)
        await client.put(path(first, LINE_MATH), json=ACCEPT_AMOUNT)
        await upload(client, misprinted(), {})  # the store keeps one: the first is dropped
        put = await client.put(path(first, LINE_MATH), json=KEEP)
        delete = await client.delete(path(first, LINE_MATH))
    problem(put, 404, "not_found")
    problem(delete, 404, "not_found")


# --------------------------------------------------------------------------- #
# Two accepted values for one cell
# --------------------------------------------------------------------------- #


def disagreeing(amount: Any) -> Job:
    """A read document whose line 1 has a line_math issue and a second reader's
    disagreement on the same amount (no flow on this branch raises the second yet)."""
    return built(
        [
            Issue(
                "line_math",
                "2 x 10.0 = 20.00, but amount is 99.0",
                line=1,
                page=1,
                fields=["line.unit_price", "line.amount"],
                suggestions=[
                    {"set": {"line.unit_price": 49.5}, "source": "arithmetic"},
                    {"set": {"line.amount": 20}, "source": "arithmetic"},
                ],
            ),
            Issue(
                "second_reading",
                "the second reader read the line differently",
                line=1,
                page=1,
                severity="warning",
                fields=["line.sku", "line.amount"],
                suggestions=[
                    {
                        "set": {"line.sku": "SKU-0O2", "line.amount": amount},
                        "source": "second_reading",
                    }
                ],
            ),
        ]
    )


def built(issues: list[Issue]) -> Job:
    """A document read as needs_review with these issues, its result built by hand."""
    lines = [line(1) | {"page": 1}, line(2, amount=99.0) | {"page": 1}]
    result = Result(
        status="needs_review",
        header=header_for([line(1), line(2)]),
        lines=lines,
        checkpoints=[],
        issues=issues,
        pages=[PageResult(1, lines)],
        evidence_source=None,
        duration_s=1.0,
        calls=2,
        cost_usd=0.002,
        prompt_tokens=200,
        completion_tokens=100,
    )
    job = Job("built.pdf", "pdf", "application/pdf", b"%PDF-1.7 ...", pages_total=1)
    job.result, job.view, job.status = result, result_view(result), "needs_review"
    return job


def store(app: FastAPI, job: Job) -> str:
    service = app.state.service
    assert isinstance(service, Service)
    service.store.add(job)
    return job.id


@pytest.mark.anyio
async def test_two_accepted_values_for_one_cell_conflict() -> None:
    async with serve(misprint_reader()) as (client, app):
        job_id = store(app, disagreeing(21.0))
        await client.put(path(job_id, 0), json={"action": "accept", "suggestion": 1})
        answer = await client.put(path(job_id, 1), json={"action": "accept", "suggestion": 0})
        review = (await client.get(f"/invoices/{job_id}")).json()["review"]
    body = problem(answer, 409, "resolution_conflict")
    assert body["detail"] == messages.resolution_conflict(0, "line.amount", 20, 21.0)
    assert [i["state"] for i in review["issues"]] == ["accepted", "open"]


@pytest.mark.anyio
async def test_accepting_another_item_of_the_same_issue_replaces_it() -> None:
    # Two items of one issue set the same cell to 20 and 21: no conflict with itself.
    both = Issue(
        "second_reading",
        "two readings of the amount",
        line=1,
        suggestions=[
            {"set": {"line.amount": 20}, "source": "arithmetic"},
            {"set": {"line.amount": 21}, "source": "arithmetic"},
        ],
    )
    async with serve(misprint_reader()) as (client, app):
        job_id = store(app, built([both]))
        first = await client.put(path(job_id, 0), json={"action": "accept", "suggestion": 0})
        second = await client.put(path(job_id, 0), json={"action": "accept", "suggestion": 1})
    assert first.status_code == second.status_code == 200, second.text
    decided = second.json()["issues"][0]
    assert (decided["state"], decided["suggestion"], decided["set"]) == (
        "accepted",
        1,
        {"line.amount": 21},
    )


@pytest.mark.anyio
async def test_an_item_code_that_reads_like_a_number_is_compared_as_text() -> None:
    def sku(value: str) -> Issue:
        return Issue(
            "second_reading",
            "the second reader read the item code differently",
            line=1,
            suggestions=[{"set": {"line.sku": value}, "source": "second_reading"}],
        )

    async with serve(misprint_reader()) as (client, app):
        job_id = store(app, built([sku("0012"), sku("12")]))
        await client.put(path(job_id, 0), json={"action": "accept", "suggestion": 0})
        answer = await client.put(path(job_id, 1), json={"action": "accept", "suggestion": 0})
    problem(answer, 409, "resolution_conflict")


@pytest.mark.anyio
@pytest.mark.parametrize("ids", [[-1], [99]])
async def test_an_id_outside_the_issue_list_leaves_the_issue_open(ids: list[int]) -> None:
    line_math = Issue(
        "line_math",
        "2 x 10.0 = 20.00, but amount is 99.0",
        line=1,
        suggestions=[{"set": {"line.amount": 20}, "source": "arithmetic"}],
    )
    issues = [line_math, Issue("doc_sums", "lines add up", follows_from=ids)]
    async with serve(misprint_reader()) as (client, app):
        job_id = store(app, built(issues))
        put = await client.put(path(job_id, 0), json={"action": "accept", "suggestion": 0})
        job = await client.get(f"/invoices/{job_id}")
        listed = await client.get("/invoices")
    assert put.status_code == job.status_code == listed.status_code == 200
    assert [i["state"] for i in job.json()["review"]["issues"]] == ["accepted", "open"]
    assert listed.json()["items"][0]["open_issue_count"] == 1


@pytest.mark.anyio
async def test_the_same_amount_written_differently_is_no_conflict() -> None:
    async with serve(misprint_reader()) as (client, app):
        job_id = store(app, disagreeing("20.00"))
        await client.put(path(job_id, 0), json={"action": "accept", "suggestion": 1})
        answer = await client.put(path(job_id, 1), json={"action": "accept", "suggestion": 0})
    assert answer.status_code == 200, answer.text
    assert [i["set"] for i in answer.json()["issues"]] == [
        {"line.amount": 20},
        {"line.sku": "SKU-0O2", "line.amount": "20.00"},
    ]


# --------------------------------------------------------------------------- #
# The access code and the documentation
# --------------------------------------------------------------------------- #


SENT = json.dumps(KEEP).encode()


class Pulled:
    """A JSON body that records whether the server read any of it."""

    def __init__(self) -> None:
        self.pulled = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.pulled = True
        yield SENT


@pytest.mark.anyio
@pytest.mark.parametrize("method", ["PUT", "DELETE"])
async def test_without_the_code_a_decision_is_refused_before_its_body_is_read(
    method: str,
) -> None:
    async with serve(misprint_reader(), api_settings(access_code=SecretStr(CODE))) as (client, _):
        job_id = await upload(client, misprinted(), RIGHT)
        await wait_final(client, job_id, RIGHT)
        body = Pulled()
        headers = {"Content-Type": "application/json", "Content-Length": str(len(SENT))}
        answer = await client.request(method, path(job_id, COVERAGE), content=body, headers=headers)
        allowed = await client.request(method, path(job_id, COVERAGE), json=KEEP, headers=RIGHT)
        review = (await client.get(f"/invoices/{job_id}", headers=RIGHT)).json()["review"]
    problem(answer, 401, "unauthorized")
    assert not body.pulled
    assert allowed.status_code == 200
    assert review["issues"][COVERAGE]["state"] == ("kept" if method == "PUT" else "open")


@pytest.mark.anyio
async def test_the_documentation_describes_both_operations_and_their_problems() -> None:
    async with serve(misprint_reader()) as (client, _):
        schema: JSONObject = (await client.get("/openapi.json")).json()
    operations = schema["paths"]["/invoices/{job_id}/issues/{issue_id}/resolution"]
    assert set(operations) == {"put", "delete"}
    put, delete = operations["put"], operations["delete"]
    assert {"200", "401", "404", "409", "413", "422"} <= set(put["responses"])
    assert {"200", "401", "404", "409"} <= set(delete["responses"])
    for operation in (put, delete):
        assert operation["security"] == [{"HTTPBearer": []}]
        ok = operation["responses"]["200"]["content"]["application/json"]["schema"]
        assert ok == {"$ref": "#/components/schemas/ReviewView"}
        for status in ("404", "409"):
            content = operation["responses"][status]["content"]
            assert set(content) == {"application/problem+json"}
    for code in ("not_found", "issue_not_found", "not_reviewable"):
        assert code in json.dumps(put["responses"]) and code in json.dumps(delete["responses"])
    for code in ("resolution_conflict", "invalid_resolution", "invalid_request", "body_too_large"):
        assert code in json.dumps(put["responses"])
    assert "issue_id" in put["responses"]["422"]["description"]
    assert "invalid_request" in delete["responses"]["422"]["description"]
    body = put["requestBody"]["content"]["application/json"]["schema"]
    assert body["discriminator"]["propertyName"] == "action"
    assert set(body["discriminator"]["mapping"]) == {"accept", "keep"}
    accept = schema["components"]["schemas"]["AcceptRequest"]
    assert accept["additionalProperties"] is False
    assert accept["properties"]["suggestion"]["minimum"] == 0
