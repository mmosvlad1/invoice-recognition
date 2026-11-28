"""The backup model as the service runs it: on by default (FALLBACK_MODEL), off when the
setting is empty. Through the API, on a drawn scan; fake OpenRouter, no network."""

import httpx
import pytest

from invoice_recognition.api import messages
from invoice_recognition.api import service as sv
from invoice_recognition.api.settings import Settings
from invoice_recognition.recognition.config import FAST, OTHER
from invoice_recognition.shapes import JSONObject

import drawn
from fakes import Answer, Handler, Server, api_settings, header_for, line, serve, wait_final

SCAN = drawn.scanned("good", "good")
DOWN = httpx.Response(503, text="no provider")


async def read(handler: Handler, **settings: object) -> JSONObject:
    async with serve(handler, api_settings(**settings)) as (client, _):
        answer = await client.post(
            "/invoices", files={"file": ("scan.pdf", SCAN, "application/pdf")}
        )
        assert answer.status_code == 202, answer.text
        return await wait_final(client, answer.json()["id"])


def outage(backup: Handler) -> Server:
    """The first model's provider answers nothing but 503."""
    return Server(DOWN, {1: [DOWN], 2: [DOWN]}, models={OTHER: backup})


def healthy() -> Server:
    pages: dict[int, list[Answer]] = {1: [[line(1)]], 2: [[line(2)]]}
    return Server(header_for([line(1), line(2)]), pages)


def test_the_service_falls_back_unless_told_otherwise() -> None:
    assert Settings.model_fields["fallback_model"].default == OTHER == "openai/gpt-6-luna"
    assert sv.config(api_settings(fallback_model=OTHER)).fallback_model == OTHER
    assert sv.config(api_settings(fallback_model=f"  {OTHER}\n")).fallback_model == OTHER
    assert sv.config(api_settings(fallback_model="  ")).fallback_model is None
    # The tests' settings switch it off (fakes.api_settings).
    assert sv.config(api_settings()).fallback_model is None


@pytest.mark.anyio
async def test_an_outage_of_the_first_provider_is_read_by_the_backup() -> None:
    job = await read(outage(healthy()), fallback_model=OTHER)
    assert job["status"] == "ok" and job["error"] is None
    result = job["result"]
    assert [item["sku"] for item in result["lines"]] == ["SKU-001", "SKU-002"]
    attempts = [
        (a["what"], a["page"], a["model"], a["error"]) for a in result["details"]["attempts"]
    ]
    failed = "HTTP 503: no provider"
    # This fake answers at once, so the header fails before a page is sent: from then on
    # the first model is down for the document, and the pages go to the backup alone.
    assert attempts == [
        ("header", None, FAST, failed),
        ("header", None, OTHER, None),
        ("page", 1, OTHER, None),
        ("page", 2, OTHER, None),
    ]
    [note] = [n for n in result["notes"] if n["kind"] == "fallback_model"]
    assert note["text"] == (
        f"The usual reading model was not available, so a backup model, {OTHER}, read the "
        "header and pages 1–2. What it read went through the same checks."
    )


@pytest.mark.anyio
async def test_both_providers_down_is_model_unavailable() -> None:
    backup = Server(DOWN, {1: [DOWN], 2: [DOWN]})
    job = await read(outage(backup), fallback_model=OTHER)
    assert job["status"] == "failed"
    assert job["error"] == {
        "code": "model_unavailable",
        "message": "No model could be reached to read this invoice: every request failed. "
        "Please try again later; if this keeps happening, tell the person who runs this "
        "service.",
    }
    assert job["error"]["message"] == messages.JOB_ERRORS["model_unavailable"]
    models = [a["model"] for a in job["result"]["details"]["attempts"]]
    # The first model once; then the backup alone, once per attempt (the header's two and
    # each page's two): no request goes back and forth between the two.
    assert models.count(FAST) == 1 and models.count(OTHER) == 6


@pytest.mark.anyio
async def test_switched_off_an_outage_fails_the_document() -> None:
    backup = healthy()
    job = await read(outage(backup), fallback_model="")
    assert job["status"] == "failed" and job["error"]["code"] == "model_unavailable"
    assert backup.calls == {}
    assert {a["model"] for a in job["result"]["details"]["attempts"]} == {FAST}
