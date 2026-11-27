"""The access code: every /invoices request carries it; without it the answer is 401,
given before the request's body is read. Fake OpenRouter, no network."""

import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from invoice_recognition.api import access, messages
from invoice_recognition.api.app import create_app
from invoice_recognition.api.settings import Settings, load_settings

import drawn
from fakes import Server, SlowServer, api_client, api_settings, header_for, line, serve

CODE = "s3cret-Code_42-long-enough"
RIGHT = {"Authorization": f"Bearer {CODE}"}
SLOW = [30.0]


def held() -> SlowServer:
    rows = [line(1), line(2)]
    server = Server(header_for(rows), {1: [[line(1)]], 2: [[line(2)]]})
    return SlowServer(server, {"header": SLOW, 1: SLOW, 2: SLOW})


def guarded() -> Settings:
    return api_settings(access_code=SecretStr(CODE))


def refused(answer: httpx.Response) -> None:
    assert answer.status_code == 401, answer.text
    assert answer.headers["content-type"] == "application/problem+json"
    assert answer.headers["www-authenticate"] == "Bearer"
    assert answer.json() == {
        "status": 401,
        "title": "Unauthorized",
        "code": "unauthorized",
        "detail": messages.UNAUTHORIZED,
    }


async def every_route(client: httpx.AsyncClient, headers: dict[str, str]) -> list[httpx.Response]:
    """One request to each of the four routes, with the given headers. The id and the file
    are those of a job uploaded with the right code."""
    job = await client.post("/invoices", files={"file": ("a.pdf", drawn.scan(2))}, headers=RIGHT)
    assert job.status_code == 202
    job_id = job.json()["id"]
    return [
        await client.post("/invoices", files={"file": ("b.pdf", drawn.scan(2))}, headers=headers),
        await client.get("/invoices", headers=headers),
        await client.get(f"/invoices/{job_id}", headers=headers),
        await client.get(f"/invoices/{job_id}/file", headers=headers),
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong"},
        {"Authorization": f"Bearer {CODE}x"},
        {"Authorization": f"Basic {CODE}"},
        {"Authorization": CODE},
        {"Authorization": "Bearer"},
    ],
    ids=["missing", "wrong", "longer", "basic", "no-scheme", "no-code"],
)
async def test_a_request_without_the_right_code_is_refused_on_every_route(
    headers: dict[str, str],
) -> None:
    async with serve(held(), guarded()) as (client, _):
        answers = await every_route(client, headers)
    for answer in answers:
        refused(answer)


@pytest.mark.anyio
@pytest.mark.parametrize("scheme", ["Bearer", "bearer", "BEARER"])
async def test_with_the_right_code_every_route_answers(scheme: str) -> None:
    async with serve(held(), guarded()) as (client, _):
        answers = await every_route(client, {"Authorization": f"{scheme} {CODE}"})
    assert [a.status_code for a in answers] == [202, 200, 200, 200]


@pytest.mark.anyio
async def test_health_and_the_documentation_stay_open() -> None:
    open_paths = ("/health", "/docs", "/docs/oauth2-redirect", "/openapi.json")
    async with serve(held(), guarded()) as (client, _):
        answers = [await client.get(path) for path in open_paths]
        head = await client.head("/health")
        redoc = await client.get("/redoc")
    assert [a.status_code for a in answers] == [200, 200, 200, 200]
    assert head.status_code == 200
    assert redoc.status_code == 404  # one documentation page is enough


@pytest.mark.anyio
async def test_a_path_that_only_starts_like_the_routes_is_not_found_not_refused() -> None:
    async with serve(held(), guarded()) as (client, _):
        answer = await client.get("/invoicesX")
    assert answer.status_code == 404 and answer.json()["code"] == "not_found"


@pytest.mark.anyio
async def test_the_routes_under_a_root_path_are_guarded_too() -> None:
    app = create_app(guarded(), transport=httpx.MockTransport(held()))
    transport = httpx.ASGITransport(app=app, root_path="/api")
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        without = await client.get("/api/invoices")
        right = await client.get("/api/invoices", headers=RIGHT)
    refused(without)
    assert right.status_code == 200  # the router does serve it there


class Body:
    """A request body that records whether the server pulled any of it."""

    def __init__(self, size: int) -> None:
        self.size, self.pulled = size, False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.pulled = True
        boundary = "b0undary"
        head = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.pdf"'
            "\r\n\r\n%PDF-"
        )
        yield head.encode()
        for _ in range(self.size // (1024 * 1024)):
            yield b"0" * (1024 * 1024)
        yield f"\r\n--{boundary}--\r\n".encode()

    @property
    def headers(self) -> dict[str, str]:
        return {"Content-Type": "multipart/form-data; boundary=b0undary"}


@pytest.mark.anyio
@pytest.mark.parametrize("size", [1024 * 1024, 20 * 1024 * 1024], ids=["small", "over-cap"])
async def test_an_upload_without_the_code_is_refused_before_its_body_is_read(size: int) -> None:
    body = Body(size)
    # A declared length too: an over-cap body would otherwise be refused by the size limit.
    headers = body.headers | {"Content-Length": str(size + 200)}
    async with serve(held(), guarded()) as (client, app):
        answer = await client.post("/invoices", content=body, headers=headers)
        jobs = app.state.service.store.newest_first()
    refused(answer)
    assert not body.pulled and jobs == []


@pytest.mark.anyio
async def test_the_same_upload_with_the_code_is_read() -> None:
    body = Body(1024 * 1024)
    async with serve(held(), guarded()) as (client, _):
        answer = await client.post("/invoices", content=body, headers=body.headers | RIGHT)
    assert body.pulled
    assert answer.status_code == 422  # read and checked: zeros are not a PDF that opens


@pytest.mark.anyio
async def test_the_documentation_offers_the_code_on_the_invoice_routes_only() -> None:
    async with serve(held(), guarded()) as (client, _):
        schema: dict[str, Any] = (await client.get("/openapi.json")).json()
    assert schema["components"]["securitySchemes"] == {
        "HTTPBearer": {
            "type": "http",
            "scheme": "bearer",
            "description": "The access code (ACCESS_CODE), sent as Authorization: Bearer <code>.",
        }
    }
    secured = {
        (path, method)
        for path, operations in schema["paths"].items()
        for method, operation in operations.items()
        if operation.get("security") == [{"HTTPBearer": []}]
    }
    assert secured == {
        ("/invoices", "get"),
        ("/invoices", "post"),
        ("/invoices/{job_id}", "get"),
        ("/invoices/{job_id}/file", "get"),
        ("/invoices/{job_id}/issues/{issue_id}/resolution", "put"),
        ("/invoices/{job_id}/issues/{issue_id}/resolution", "delete"),
    }
    assert "401" in schema["paths"]["/invoices"]["get"]["responses"]


@pytest.mark.anyio
async def test_an_empty_code_leaves_the_routes_open_and_warns_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        async with serve(held()) as (client, _):
            answer = await client.get("/invoices")
    assert answer.status_code == 200
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    text = (
        "ACCESS_CODE is not set: the /invoices routes are open to anyone who can reach this "
        "server. Set it for anything reachable from outside."
    )
    assert [r.getMessage() for r in warnings] == [text]


@pytest.mark.anyio
async def test_a_set_code_logs_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        async with serve(held(), guarded()):
            pass
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.anyio
async def test_without_the_lifespan_the_check_still_applies() -> None:
    async with api_client(create_app(guarded())) as client:
        refused(await client.get("/invoices"))


# --------------------------------------------------------------------------- #
# The check itself
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("header", "ok"),
    [
        (f"Bearer {CODE}", True),
        (f"bearer {CODE}", True),
        (f"  Bearer   {CODE}  ", True),
        (None, False),
        ("", False),
        ("Bearer", False),
        ("Bearer ", False),
        (f"Bearer {CODE[:-1]}", False),
        (f"Bearer {CODE}{CODE}", False),
        (f"Basic {CODE}", False),
        (f"Bearer{CODE}", False),
        ("Bearer ключ", False),
        (f"Bearer {CODE.upper()}", False),
    ],
)
def test_check(header: str | None, ok: bool) -> None:
    assert access.check(header, CODE) is ok


@pytest.mark.parametrize(
    ("path", "root", "route"),
    [
        ("/invoices", "", "/invoices"),
        ("/api/invoices", "/api", "/invoices"),
        ("/invoices", "/api", "/invoices"),
        ("/apix/invoices", "/api", "/apix/invoices"),
        ("/api", "/api", ""),
    ],
)
def test_route_path_follows_the_router(path: str, root: str, route: str) -> None:
    assert access.route_path({"type": "http", "path": path, "root_path": root}) == route


@pytest.mark.parametrize(
    ("path", "guarded_"),
    [
        ("/invoices", True),
        ("/invoices/", True),
        ("/invoices/abc/file", True),
        ("/invoicesX", False),
        ("/health", False),
        ("/docs", False),
        ("/", False),
    ],
)
def test_protected(path: str, guarded_: bool) -> None:
    assert access.protected(path) is guarded_


def test_the_middleware_needs_a_code() -> None:
    async def app(scope: Any, receive: Any, send: Any) -> None:
        raise AssertionError

    with pytest.raises(ValueError, match="no access code"):
        access.AccessMiddleware(app, "")


# --------------------------------------------------------------------------- #
# The setting
# --------------------------------------------------------------------------- #


def test_the_code_is_stripped_and_empty_by_default() -> None:
    assert api_settings().access_code.get_secret_value() == ""
    padded = SecretStr(f"  {CODE}\n")
    assert api_settings(access_code=padded).access_code.get_secret_value() == CODE


@pytest.mark.parametrize(
    "code",
    [
        "two words and more than enough",
        "ключ-ключ-ключ-ключ-ключ",
        "tab\tx-long-enough-to-pass",
        "short-code-15ch",
    ],
)
def test_a_code_no_client_could_send_is_refused_without_showing_it(code: str) -> None:
    with pytest.raises(ValidationError) as refusal:
        api_settings(access_code=SecretStr(code))
    assert code not in str(refusal.value)
    [error] = refusal.value.errors()
    assert error["loc"] == ("access_code",) and "at least 16 characters" in error["msg"]


def test_a_code_of_16_characters_is_enough() -> None:
    code = "a" * 15 + "b"
    assert api_settings(access_code=SecretStr(code)).access_code.get_secret_value() == code


def test_a_short_code_stops_the_start_naming_the_setting_and_the_rule(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("ACCESS_CODE", "tiny-secret")
    monkeypatch.chdir(tmp_path)  # no .env
    with pytest.raises(SystemExit) as stop:
        load_settings()
    text = str(stop.value.code)
    assert "ACCESS_CODE is not valid: it must be at least 16 characters long" in text
    assert "tiny-secret" not in text and not capsys.readouterr().out
