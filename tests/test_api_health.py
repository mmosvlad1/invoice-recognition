"""The health check and the generated documentation."""

import pytest

from invoice_recognition.api.app import create_app

from fakes import api_client, api_settings


@pytest.mark.anyio
async def test_health_answers_ok() -> None:
    async with api_client(create_app(api_settings())) as client:
        answer = await client.get("/health")
    assert answer.status_code == 200
    assert answer.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_the_documentation_is_served() -> None:
    async with api_client(create_app(api_settings())) as client:
        docs = await client.get("/docs")
        schema = (await client.get("/openapi.json")).json()
    assert docs.status_code == 200 and "swagger" in docs.text.lower()
    assert schema["info"]["title"] == "Invoice recognition"
    assert list(schema["paths"]) == ["/health", "/invoices/{job_id}"]


@pytest.mark.anyio
async def test_health_answers_head_too() -> None:
    async with api_client(create_app(api_settings())) as client:
        answer = await client.head("/health")
    assert answer.status_code == 200 and answer.content == b""


@pytest.mark.anyio
async def test_a_trailing_slash_is_not_redirected() -> None:
    async with api_client(create_app(api_settings())) as client:
        answer = await client.get("/health/")
    assert answer.status_code == 404 and "location" not in answer.headers
