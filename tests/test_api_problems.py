"""Every error the API answers has one shape: RFC 9457 problem details."""

from typing import Any

import pytest
from fastapi import FastAPI
from starlette.exceptions import HTTPException

from invoice_recognition.api import messages
from invoice_recognition.api.app import create_app
from invoice_recognition.api.problems import ApiError

from fakes import api_client, api_settings

PROBLEM = "application/problem+json"


def app_with_test_routes() -> FastAPI:
    app = create_app(api_settings())

    @app.get("/test/number/{n}")
    async def number(n: int) -> dict[str, int]:
        return {"n": n}

    @app.get("/test/busy")
    async def busy() -> dict[str, str]:
        raise ApiError(
            503,
            "too_many_documents",
            "Ten documents are already waiting or being read. Try again in a minute.",
            headers={"Retry-After": "30"},
            extra={"limit": 10},
        )

    @app.get("/test/framework/{status}")
    async def framework(status: int) -> dict[str, str]:
        raise HTTPException(status, headers={"X-Kept": "yes"})

    @app.get("/test/broken")
    async def broken() -> dict[str, str]:
        raise RuntimeError("a bug, with secret-ish details")

    return app


async def get(path: str, method: str = "GET", **kw: Any) -> tuple[int, dict[str, str], Any]:
    async with api_client(app_with_test_routes(), raise_app_exceptions=False) as client:
        answer = await client.request(method, path, **kw)
    return answer.status_code, dict(answer.headers), answer.json()


@pytest.mark.anyio
async def test_an_unknown_route_is_a_problem_not_found() -> None:
    status, headers, body = await get("/nowhere")
    assert status == 404 and headers["content-type"] == PROBLEM
    assert body == {
        "status": 404,
        "title": "Not Found",
        "code": "not_found",
        "detail": messages.NOT_FOUND,
    }


@pytest.mark.anyio
async def test_a_wrong_method_is_method_not_allowed_and_keeps_allow() -> None:
    status, headers, body = await get("/health", "DELETE")
    assert status == 405 and headers["content-type"] == PROBLEM
    assert headers["allow"] == "GET"
    assert body == {
        "status": 405,
        "title": "Method Not Allowed",
        "code": "method_not_allowed",
        "detail": "This address does not accept DELETE requests.",
    }


@pytest.mark.anyio
async def test_a_request_off_the_schema_lists_the_fields_without_their_values() -> None:
    status, headers, body = await get("/test/number/not-a-number")
    assert status == 422 and headers["content-type"] == PROBLEM
    assert body["title"] == "Unprocessable Content"
    assert body["code"] == "invalid_request" and body["detail"] == messages.INVALID_REQUEST
    assert body["errors"] == [
        {
            "loc": ["path", "n"],
            "msg": "Input should be a valid integer, unable to parse string as an integer",
            "type": "int_parsing",
        }
    ]
    assert "not-a-number" not in str(body)


@pytest.mark.anyio
async def test_an_api_error_keeps_its_headers_and_extra_keys() -> None:
    status, headers, body = await get("/test/busy")
    assert status == 503 and headers["content-type"] == PROBLEM
    assert headers["retry-after"] == "30"
    assert body == {
        "status": 503,
        "title": "Service Unavailable",
        "code": "too_many_documents",
        "detail": "Ten documents are already waiting or being read. Try again in a minute.",
        "limit": 10,
    }


@pytest.mark.anyio
async def test_an_unhandled_error_is_a_logged_internal_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    status, headers, body = await get("/test/broken")
    assert status == 500 and headers["content-type"] == PROBLEM
    assert body == {
        "status": 500,
        "title": "Internal Server Error",
        "code": "internal_error",
        "detail": messages.INTERNAL_ERROR,
    }
    assert "secret-ish" not in str(body)
    [record] = [r for r in caplog.records if r.name == "invoice_recognition.api.problems"]
    assert record.exc_info is not None and record.exc_info[0] is RuntimeError


@pytest.mark.anyio
async def test_a_working_route_is_not_touched() -> None:
    status, headers, body = await get("/test/number/7")
    assert (status, headers["content-type"], body) == (200, "application/json", {"n": 7})


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status", "code", "detail"),
    [
        (400, "invalid_request", messages.BAD_REQUEST),
        (401, "unauthorized", messages.UNAUTHORIZED),
        # Not the upload: its body is no file (the upload's own 413 is in test_api_upload).
        (413, "body_too_large", messages.BODY_TOO_LARGE),
        (418, "http_error", messages.HTTP_ERROR),
    ],
)
async def test_an_error_the_framework_raises_gets_a_code_by_its_status(
    status: int, code: str, detail: str
) -> None:
    got, headers, body = await get(f"/test/framework/{status}")
    assert got == status and headers["content-type"] == PROBLEM
    assert headers["x-kept"] == "yes"
    assert body["code"] == code and body["detail"] == detail


@pytest.mark.anyio
async def test_a_status_without_a_body_gets_none() -> None:
    async with api_client(app_with_test_routes()) as client:
        answer = await client.get("/test/framework/304")
    assert answer.status_code == 304 and answer.content == b""
    assert answer.headers["x-kept"] == "yes"


@pytest.mark.anyio
async def test_a_trailing_slash_is_an_unknown_route_not_a_redirect() -> None:
    status, headers, body = await get("/health/")
    assert status == 404 and headers["content-type"] == PROBLEM
    assert body["code"] == "not_found" and "location" not in headers


def test_the_documentation_describes_errors_as_problem_details() -> None:
    schema = app_with_test_routes().openapi()
    answers = schema["paths"]["/test/number/{n}"]["get"]["responses"]
    assert "422" not in answers  # the framework's own 422 is not the shape sent
    problem = {"schema": {"$ref": "#/components/schemas/Problem"}}
    for status in ("4XX", "5XX"):
        assert answers[status]["content"] == {"application/problem+json": problem}
    assert answers["200"]["content"].keys() == {"application/json"}
    assert "HTTPValidationError" not in schema["components"]["schemas"]
    assert schema["components"]["schemas"]["Problem"]["required"] == [
        "status",
        "title",
        "code",
        "detail",
    ]
    errors = schema["components"]["schemas"]["Problem"]["properties"]["errors"]
    assert errors["type"] == "array" and "default" not in errors  # absent, never null
    assert "anyOf" not in errors
    health = schema["paths"]["/health"]
    assert list(health) == ["get"] and "4XX" in health["get"]["responses"]
