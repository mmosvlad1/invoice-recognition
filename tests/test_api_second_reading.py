"""The second reading of scans as the service runs it: off by default, on when
SECOND_READING_MODEL names a model. Through the API, on a drawn scan; fake OpenRouter, no
network."""

import pytest

from invoice_recognition.api import service as sv
from invoice_recognition.recognition.config import FAST, OTHER
from invoice_recognition.shapes import JSONObject

import drawn
from fakes import (
    Answer,
    Handler,
    Server,
    api_settings,
    header_for,
    line,
    recording,
    serve,
    wait_final,
)

SCAN = drawn.scanned("good", "good")  # two pages measured as readable: both read twice


def server(second: dict[int, list[Answer]]) -> Server:
    first: dict[int, list[Answer]] = {1: [[line(1)]], 2: [[line(2)]]}
    return Server(header_for([line(1), line(2)]), first, models={OTHER: Server({}, second)})


async def read(handler: Handler, **settings: object) -> JSONObject:
    async with serve(handler, api_settings(**settings)) as (client, _):
        answer = await client.post(
            "/invoices", files={"file": ("scan.pdf", SCAN, "application/pdf")}
        )
        assert answer.status_code == 202, answer.text
        return await wait_final(client, answer.json()["id"])


ON = {"second_reading_model": OTHER}


def test_the_service_reads_scans_once_unless_told_otherwise() -> None:
    assert api_settings().second_reading_model == ""
    assert sv.config(api_settings()).second_reading_model is None
    assert sv.config(api_settings(second_reading_model="  ")).second_reading_model is None
    assert sv.config(api_settings(**ON)).second_reading_model == OTHER == "openai/gpt-6-luna"


@pytest.mark.anyio
async def test_a_disagreement_comes_back_as_an_issue_with_a_suggestion() -> None:
    second: dict[int, list[Answer]] = {1: [[line(1)]], 2: [[line(2) | {"sku": "SKU-0O2"}]]}
    job = await read(server(second), **ON)
    result = job["result"]
    assert job["status"] == "needs_review"
    [issue] = result["issues"]
    assert (issue["code"], issue["severity"], issue["line_index"], issue["fields"]) == (
        "second_reading",
        "warning",
        1,
        ["line.sku"],
    )
    assert issue["suggestions"] == [{"set": {"line.sku": "SKU-0O2"}, "source": "second_reading"}]
    assert issue["message"].startswith(
        "Position 2 on page 2 was read differently by a second model"
    )
    assert result["lines"][1]["sku"] == "SKU-002"  # the first reading is kept
    attempts = result["details"]["attempts"]
    assert [(a["what"], a["page"], a["model"]) for a in attempts if a["what"] != "header"] == [
        ("page", 1, FAST),
        ("second_reading", 1, OTHER),
        ("page", 2, FAST),
        ("second_reading", 2, OTHER),
    ]


@pytest.mark.anyio
async def test_by_default_no_second_request_is_sent() -> None:
    handler, bodies = recording(server({}))
    job = await read(handler)
    assert job["status"] == "ok"
    assert {model for model in (b["model"] for b in bodies)} == {FAST}
    attempts = job["result"]["details"]["attempts"]
    assert "second_reading" not in [a["what"] for a in attempts]


def _accept(issue: JSONObject, cell: str) -> JSONObject:
    """The accept request for the issue's suggestion that sets `cell`."""
    [k] = [k for k, s in enumerate(issue["suggestions"]) if cell in s["set"]]
    return {"action": "accept", "suggestion": k}


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("second_amount", "second_status"), [(20, 200), (25, 409)], ids=["same", "different"]
)
async def test_two_issues_suggesting_one_cell_can_both_be_accepted_only_if_they_agree(
    second_amount: int, second_status: int
) -> None:
    """A real result: the first reader reads an amount of 99 for 2 x 10 (line_math
    suggests 20), the second reader reads `second_amount`. Accepting both suggestions for
    that cell works when they are the same value and is refused when they are not."""
    first: dict[int, list[Answer]] = {1: [[line(1, amount=99.0)]]}
    second: dict[int, list[Answer]] = {1: [[line(1, amount=second_amount)]]}
    server = Server(header_for([line(1)]), first, models={OTHER: Server({}, second)})
    scan = drawn.scanned("good")
    async with serve(server, api_settings(**ON)) as (client, _):
        answer = await client.post(
            "/invoices", files={"file": ("scan.pdf", scan, "application/pdf")}
        )
        job_id = answer.json()["id"]
        issues = (await wait_final(client, job_id))["result"]["issues"]
        [arithmetic] = [i for i in issues if i["code"] == "line_math"]
        [read_twice] = [i for i in issues if i["code"] == "second_reading"]
        assert read_twice["suggestions"][0]["set"]["line.amount"] == second_amount

        def resolution(issue: JSONObject) -> str:
            return f"/invoices/{job_id}/issues/{issue['id']}/resolution"

        accepted = await client.put(resolution(arithmetic), json=_accept(arithmetic, "line.amount"))
        assert accepted.status_code == 200, accepted.text
        again = await client.put(resolution(read_twice), json=_accept(read_twice, "line.amount"))
    assert again.status_code == second_status, again.text
    if second_status == 409:
        assert again.json()["code"] == "resolution_conflict"
