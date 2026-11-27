"""The page: `index.html` at `/` and its files under `/static/`, served by the API itself.

Both are open (the access code guards only /invoices...): the page asks for the code and
sends it with every request it makes. Every answer here carries the page's own
Content-Security-Policy, so that nothing but our own files can run in it, even if a file
name or a value read from an invoice slipped into its markup. /docs (Swagger UI, which
loads its script from a CDN) gets no policy: it is set only on the page's answers.

Our own files may change with every release: `no-cache` (the browser asks again and gets a
304 while the file is the same). The vendored PDF.js lives under a versioned path and
never changes there: cached for a year.
"""

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from starlette.types import Scope

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
STATIC_DIR = WEB_DIR / "static"

DIRECTIVES = (
    "default-src 'none'",
    # 'wasm-unsafe-eval' lets PDF.js's worker compile its image decoder (OpenJPEG, for
    # JPX images in scans); it allows WebAssembly only, never eval of JavaScript.
    "script-src 'self' 'wasm-unsafe-eval'",
    "style-src 'self'",
    # blob: the uploaded photo, shown from the bytes the page fetched with the code.
    "img-src 'self' blob:",
    "font-src 'self'",
    "connect-src 'self'",
    "worker-src 'self'",
    "base-uri 'none'",
    "form-action 'none'",
    "frame-ancestors 'none'",
)
CSP = "; ".join(DIRECTIVES)

PAGE_HEADERS = {
    "Content-Security-Policy": CSP,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}

# The types are pinned, not guessed: `mimetypes` also reads /etc/mime.types on Linux, and a
# module script is refused under nosniff unless it is served as JavaScript. "" is a file
# without a suffix (the vendored LICENSE).
TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".txt": "text/plain; charset=utf-8",
    "": "text/plain; charset=utf-8",
}

VENDOR = STATIC_DIR / "vendor"
REVALIDATE = "no-cache"
IMMUTABLE = "public, max-age=31536000, immutable"


def headers_for(path: Path) -> dict[str, str]:
    """The headers of a page file: its pinned type, how long it may be cached, the page's
    security headers."""
    cache = IMMUTABLE if path.is_relative_to(VENDOR) else REVALIDATE
    return {
        "Content-Type": TYPES.get(path.suffix.lower(), "application/octet-stream"),
        "Cache-Control": cache,
    } | PAGE_HEADERS


class PageFiles(StaticFiles):
    """StaticFiles with the page's headers on every file it serves (a 304 included)."""

    def file_response(
        self,
        full_path: str | os.PathLike[str],
        stat_result: os.stat_result,
        scope: Scope,
        status_code: int = 200,
    ) -> Response:
        response = super().file_response(full_path, stat_result, scope, status_code)
        headers = headers_for(Path(full_path))
        if response.status_code == 304:  # no body, so no type
            del headers["Content-Type"]
        response.headers.update(headers)
        return response


def mount_page(app: FastAPI) -> None:
    """`GET` and `HEAD /` -> index.html; `/static/...` -> the page's files. No catch-all:
    an unknown path keeps its 404 problem."""

    files = PageFiles(directory=STATIC_DIR)
    index = WEB_DIR / "index.html"

    @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
    async def page(request: Request) -> Response:
        # Through the same file response as /static: the same headers, and a 304 when the
        # browser's copy is current.
        return files.file_response(index, index.stat(), request.scope)

    app.mount("/static", files, name="static")
