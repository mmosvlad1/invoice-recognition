"""One shape for every error the API answers: RFC 9457 problem details.

    {"status": 422, "title": "Unprocessable Content", "code": "too_many_pages",
     "detail": "The PDF has 62 pages; the limit is 50."}

`title` is the phrase of the HTTP status (RFC 9457 without a `type`, which means
"about:blank"); `code` (stable, for programs) and `detail` (a sentence for a person)
carry the meaning. Errors of the request's shape add `errors`: one entry per field.

The handlers cover the API's own errors (`ApiError`), the errors the framework raises
(unknown route, wrong method, a body it cannot parse), a request that does not match
the schema, and anything else that escapes a route (500, logged).

The one exception: a body whose declared Content-Length is over the size limit is
refused by the framework before any handler runs, as plain text ("Content Too Large",
413). Clients rely on the status code there.
"""

import logging
from collections.abc import Callable, Coroutine, Mapping
from http import HTTPStatus
from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema
from starlette.exceptions import HTTPException

from invoice_recognition.api import messages

logger = logging.getLogger(__name__)

MEDIA_TYPE = "application/problem+json"


class FieldError(BaseModel):
    loc: list[str | int] = Field(description='Where in the request, e.g. ["path", "id"].')
    msg: str
    type: str


def _absent_when_none(schema: dict[str, Any]) -> None:
    # A field left out when it is None (exclude_none) has no default in what is sent.
    schema.pop("default", None)


class Problem(BaseModel):
    """An error answer (RFC 9457). Extra keys may follow `detail` for some codes."""

    model_config = ConfigDict(extra="allow")

    status: int = Field(description="The HTTP status code, repeated.")
    title: str = Field(description="The phrase of the HTTP status, e.g. Not Found.")
    code: str = Field(description="Stable machine-readable reason, e.g. too_many_pages.")
    detail: str = Field(description="What went wrong, for a person.")
    # Never sent as null: left out unless there are errors (problem_response drops
    # None), so the schema says "optional array", not "nullable".
    errors: list[FieldError] | SkipJsonSchema[None] = Field(
        default=None,
        description="With code invalid_request: the fields at fault; absent otherwise.",
        json_schema_extra=_absent_when_none,
    )


class ApiError(Exception):
    """An error the API answers on purpose: raise it from a route or a dependency."""

    def __init__(
        self,
        status: int,
        code: str,
        detail: str,
        *,
        headers: Mapping[str, str] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(detail)
        self.status, self.code, self.detail = status, code, detail
        self.headers = dict(headers or {})
        self.extra = dict(extra or {})


def problem_response(
    status: int,
    code: str,
    detail: str,
    headers: Mapping[str, str] | None = None,
    *,
    extra: Mapping[str, Any] | None = None,
    errors: list[FieldError] | None = None,
) -> Response:
    """The answer for an error, in the one shape."""
    if status < 200 or status in (204, 205, 304):  # a status that must not carry a body
        return Response(status_code=status, headers=headers)
    problem = Problem(
        status=status,
        title=HTTPStatus(status).phrase,
        code=code,
        detail=detail,
        errors=errors,
        **dict(extra or {}),
    )
    return JSONResponse(
        problem.model_dump(mode="json", exclude_none=True),
        status_code=status,
        headers=headers,
        media_type=MEDIA_TYPE,
    )


type Handler = Callable[[Request, Any], Coroutine[Any, Any, Response]]


async def _api_error(request: Request, exc: ApiError) -> Response:
    return problem_response(exc.status, exc.code, exc.detail, exc.headers, extra=exc.extra)


def _http_error(max_file_bytes: int) -> Handler:
    # The errors the framework raises itself, by status: (code, detail).
    known = {
        400: ("invalid_request", messages.BAD_REQUEST),
        401: ("unauthorized", messages.UNAUTHORIZED),
        404: ("not_found", messages.NOT_FOUND),
        405: ("method_not_allowed", messages.METHOD_NOT_ALLOWED),
        # A body over the size limit, sent in chunks (with a Content-Length it never
        # gets here: see the module's docstring).
        413: (
            "file_too_large",
            messages.FILE_TOO_LARGE.format(limit=messages.size(max_file_bytes)),
        ),
    }

    async def handle(request: Request, exc: HTTPException) -> Response:
        code, detail = known.get(exc.status_code, ("http_error", messages.HTTP_ERROR))
        # The Allow header of a 405 and any other header the framework set are kept.
        return problem_response(
            exc.status_code, code, detail.format(method=request.method), exc.headers
        )

    return handle


async def _invalid_request(request: Request, exc: RequestValidationError) -> Response:
    # The value that was sent is left out: it can be large, or a secret.
    errors = [
        FieldError(loc=list(e.get("loc", ())), msg=str(e.get("msg", "")), type=str(e.get("type")))
        for e in exc.errors()
    ]
    return problem_response(422, "invalid_request", messages.INVALID_REQUEST, errors=errors)


async def _internal_error(request: Request, exc: Exception) -> Response:
    # Runs in the outermost middleware; the server still logs the exception as well.
    logger.error("unhandled error in %s %s", request.method, request.url.path, exc_info=exc)
    return problem_response(500, "internal_error", messages.INTERNAL_ERROR)


def handlers(*, max_file_bytes: int) -> dict[int | type[Exception], Handler]:
    """The exception handlers of the app (FastAPI(exception_handlers=...)).
    max_file_bytes: the size limit of an uploaded file, named when a body is too large."""
    return {
        ApiError: _api_error,
        HTTPException: _http_error(max_file_bytes),
        RequestValidationError: _invalid_request,
        Exception: _internal_error,
    }


# --------------------------------------------------------------------------- #
# The documentation (OpenAPI)
# --------------------------------------------------------------------------- #

_ANY_ERROR = {"model": Problem, "description": "Problem details (RFC 9457)."}

# Every route can answer an error of the client (an unknown method, a request off the
# schema) and of the server, in the one shape. Declared for every route, they also
# replace the framework's own 422 description, which is not the shape sent.
RESPONSES: dict[int | str, dict[str, Any]] = {"4XX": _ANY_ERROR, "5XX": _ANY_ERROR}


def document(schema: dict[str, Any]) -> dict[str, Any]:
    """The generated OpenAPI schema with the error answers under their real media type,
    application/problem+json (the framework lists a model's schema as application/json).
    Changes the schema in place and returns it."""
    for operations in schema.get("paths", {}).values():
        for operation in operations.values():
            for status, answer in operation.get("responses", {}).items():
                content = answer.get("content", {})
                if status[0] in "45" and "application/json" in content:
                    content[MEDIA_TYPE] = content.pop("application/json")
    return schema
