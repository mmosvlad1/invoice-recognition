"""The page at / and its files under /static: served open, with their own security headers,
the types pinned and the cache rules; and static checks of the page's own sources."""

import re
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from invoice_recognition.api.app import create_app
from invoice_recognition.api.page import CSP, STATIC_DIR, TYPES, VENDOR, WEB_DIR, headers_for

from fakes import api_client, api_settings

PAGE_HEADERS = {
    "content-security-policy": CSP,
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
}

SHIPPED = sorted(p for p in STATIC_DIR.rglob("*") if p.is_file())
OWN = [WEB_DIR / "index.html", *(p for p in SHIPPED if not p.is_relative_to(VENDOR))]
OWN_TEXT = [p for p in OWN if p.suffix in (".html", ".css", ".svg", ".mjs")]
SCRIPTS = sorted((STATIC_DIR / "js").glob("*.mjs"))


def url(path: Path) -> str:
    return "/static/" + path.relative_to(STATIC_DIR).as_posix()


def has_page_headers(answer: httpx.Response) -> None:
    for name, value in PAGE_HEADERS.items():
        assert answer.headers[name] == value, name


def test_the_policy_allows_only_the_pages_own_files() -> None:
    assert CSP == (
        "default-src 'none'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self'; "
        "img-src 'self' blob:; font-src 'self'; connect-src 'self'; worker-src 'self'; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    )


@pytest.mark.anyio
@pytest.mark.parametrize("method", ["GET", "HEAD"])
async def test_the_page_is_served_at_the_root(method: str) -> None:
    async with api_client(create_app(api_settings())) as client:
        answer = await client.request(method, "/")
    assert answer.status_code == 200
    assert answer.headers["content-type"] == "text/html; charset=utf-8"
    assert answer.headers["cache-control"] == "no-cache"
    has_page_headers(answer)
    expected = b"" if method == "HEAD" else (WEB_DIR / "index.html").read_bytes()
    assert answer.content == expected


@pytest.mark.anyio
async def test_every_shipped_file_is_served_with_its_type_and_cache_rule() -> None:
    async with api_client(create_app(api_settings())) as client:
        for path in SHIPPED:
            answer = await client.get(url(path))
            assert answer.status_code == 200, path
            assert answer.headers["content-type"] == TYPES[path.suffix], path
            vendored = path.is_relative_to(VENDOR)
            cache = "public, max-age=31536000, immutable" if vendored else "no-cache"
            assert answer.headers["cache-control"] == cache, path
            has_page_headers(answer)
            assert answer.content == path.read_bytes()


@pytest.mark.anyio
@pytest.mark.parametrize("path", ["/", "/static/styles.css"])
async def test_an_unchanged_file_is_answered_not_modified(path: str) -> None:
    async with api_client(create_app(api_settings())) as client:
        first = await client.get(path)
        again = await client.get(path, headers={"If-None-Match": first.headers["etag"]})
    assert again.status_code == 304
    assert again.content == b""
    assert again.headers["cache-control"] == "no-cache"
    assert "content-type" not in again.headers
    has_page_headers(again)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "path",
    [
        "/static/nope.css",
        "/static/%2e%2e/index.html",
        "/static/js/%2e%2e/%2e%2e/index.html",
        "/static",
        "/static/",
        "/nope",
    ],
)
async def test_nothing_else_is_served(path: str) -> None:
    async with api_client(create_app(api_settings())) as client:
        answer = await client.get(path)
    assert answer.status_code == 404, path
    assert answer.headers["content-type"] == "application/problem+json"
    assert answer.json()["code"] == "not_found"


def test_the_type_goes_by_the_suffix_in_any_case() -> None:
    assert headers_for(Path("APP.MJS"))["Content-Type"] == "text/javascript; charset=utf-8"
    assert headers_for(Path("x.unknown"))["Content-Type"] == "application/octet-stream"


@pytest.mark.anyio
async def test_the_files_accept_only_reading() -> None:
    async with api_client(create_app(api_settings())) as client:
        static = await client.post("/static/styles.css")
        root = await client.post("/")
    assert (static.status_code, static.json()["code"]) == (405, "method_not_allowed")
    assert (root.status_code, root.json()["code"]) == (405, "method_not_allowed")


@pytest.mark.anyio
async def test_the_page_is_open_while_the_invoices_need_the_code() -> None:
    app = create_app(api_settings(access_code=SecretStr("s3cret-Code_42-long-enough")))
    async with api_client(app) as client:
        page = await client.get("/")
        script = await client.get("/static/js/app.mjs")
        invoices = await client.get("/invoices")
    assert (page.status_code, script.status_code, invoices.status_code) == (200, 200, 401)


@pytest.mark.anyio
async def test_the_documentation_keeps_its_own_rules() -> None:
    async with api_client(create_app(api_settings())) as client:
        docs = await client.get("/docs")
        schema = (await client.get("/openapi.json")).json()
    assert docs.status_code == 200
    assert "content-security-policy" not in docs.headers
    assert "/" not in schema["paths"]


# --------------------------------------------------------------------------- #
# The page's own sources
# --------------------------------------------------------------------------- #


def test_every_shipped_file_has_a_pinned_type() -> None:
    assert SHIPPED
    assert {p.suffix for p in SHIPPED} <= set(TYPES)


def test_every_file_the_page_names_is_shipped() -> None:
    page = (WEB_DIR / "index.html").read_text()
    named = re.findall(r'(?:src|href)="(/[^"]*)"', page)
    assert "/static/js/app.mjs" in named
    for name in named:
        assert (STATIC_DIR / name.removeprefix("/static/")).is_file(), name
    for script in SCRIPTS:
        for target in re.findall(
            r'^import .* from "(\.[^"]+)";$', script.read_text(), re.MULTILINE
        ):
            assert (script.parent / target).is_file(), (script.name, target)


def test_the_page_loads_nothing_from_elsewhere() -> None:
    for path in OWN_TEXT:
        addresses = re.findall(r"https?://[^\s\"'`)]+", path.read_text())
        assert set(addresses) <= {"http://www.w3.org/2000/svg"}, path


def test_the_page_has_no_inline_script_or_style() -> None:
    page = (WEB_DIR / "index.html").read_text()
    assert re.findall(r"<script\b[^>]*>", page) == [
        '<script type="module" src="/static/js/app.mjs">'
    ]
    assert not re.search(r"\sstyle=", page)
    assert not re.search(r"\son[a-z]+=", page)
    for script in SCRIPTS:
        text = script.read_text()
        assert 'style="' not in text, script.name
        assert not re.search(r"\son[a-z]+=", text), script.name


def test_markup_is_written_into_the_document_only_by_put() -> None:
    sinks = re.compile(r"\b(?:innerHTML|outerHTML|insertAdjacentHTML)\b")
    for script in SCRIPTS:
        text = script.read_text()
        if script.name != "html.mjs":
            assert not sinks.search(text), script.name
            continue
        put = re.search(r"^export function put\(.*?^}$", text, re.MULTILINE | re.DOTALL)
        assert put is not None
        assert len(sinks.findall(text)) == len(sinks.findall(put.group())) == 1


def test_every_attribute_in_the_scripts_markup_is_quoted() -> None:
    # html`` escapes quotes, so a value is safe in a quoted attribute only: unquoted,
    # "x onmouseover=alert(1)" would add an attribute.
    for script in SCRIPTS:
        assert "=${" not in script.read_text(), script.name


def test_the_samples_say_what_they_are() -> None:
    page = (WEB_DIR / "index.html").read_text()
    assert (
        "Generated test invoices; the scan and the photo are simulated from them; the small "
        "PDF has one amount misprinted on purpose."
    ) in page
    for label in ("Scan (simulated)", "Photo (simulated)", "one misprinted amount"):
        assert label in page
