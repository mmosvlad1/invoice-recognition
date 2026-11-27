"""The access code: one shared secret that every request to /invoices... carries as
`Authorization: Bearer <code>`.

The check is a plain ASGI middleware, not a route dependency, because the framework
reads and parses an upload's body before any dependency runs: a stranger could make the
server receive a 10 MB file, or learn the limits from its refusals. Here a request
without the right code is answered 401 from its headers alone; its body is never read.

The code keeps strangers and scanners out. It does not tell users apart: everyone who
has it sees every document.
"""

import secrets

from starlette.types import ASGIApp, Receive, Scope, Send

from invoice_recognition.api import messages
from invoice_recognition.api.problems import problem_response

PREFIX = "/invoices"


def check(authorization: str | None, code: str) -> bool:
    """Whether the Authorization header carries the code: the scheme Bearer (in any case),
    then the code. Compared in constant time, as bytes, so text that is not ASCII is
    simply a wrong code."""
    if authorization is None:
        return False
    scheme, _, token = authorization.strip().partition(" ")
    if scheme.lower() != "bearer":
        return False
    return secrets.compare_digest(token.strip().encode("utf-8"), code.encode("utf-8"))


def route_path(scope: Scope) -> str:
    """The path the router matches against: the request's path without the root path the
    app is served under (`--root-path`), the rule of Starlette's router. Checking the raw
    path instead would leave /root/invoices open while the router serves it."""
    path: str = scope["path"]
    root: str = scope.get("root_path", "")
    if root and path.startswith(root) and (path == root or path[len(root)] == "/"):
        return path[len(root) :]
    return path


def protected(path: str) -> bool:
    """/invoices and everything under it; not /invoicesX, not /health or /docs."""
    return path == PREFIX or path.startswith(PREFIX + "/")


def _authorization(scope: Scope) -> str | None:
    for name, value in scope["headers"]:
        if name == b"authorization":
            # As Starlette decodes a header: every byte is a character.
            return bytes(value).decode("latin-1")
    return None


class AccessMiddleware:
    """Answers 401 unauthorized to a request for /invoices... without the right code,
    before the app sees it. Added only when a code is set."""

    def __init__(self, app: ASGIApp, code: str) -> None:
        if not code:
            raise ValueError("no access code: leave the middleware out instead")
        self.app, self.code = app, code

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] == "http"
            and protected(route_path(scope))
            and not check(_authorization(scope), self.code)
        ):
            refusal = problem_response(
                401, "unauthorized", messages.UNAUTHORIZED, {"WWW-Authenticate": "Bearer"}
            )
            await refusal(scope, receive, send)  # sends; never reads the body
            return
        await self.app(scope, receive, send)
