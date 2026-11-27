"""The FastAPI application: the routes, and the reading service that lives as long as
the app does."""

import asyncio
import logging
import unicodedata
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import version
from typing import Annotated, Any, Literal
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Depends, FastAPI, File, Request, Response, UploadFile
from fastapi.security import HTTPBearer
from pydantic import BaseModel

from invoice_recognition.api import messages, problems, review, uploads, views
from invoice_recognition.api.access import AccessMiddleware
from invoice_recognition.api.jobs import Job, StoreFull, now
from invoice_recognition.api.problems import ApiError, Problem
from invoice_recognition.api.service import Service
from invoice_recognition.api.settings import Settings
from invoice_recognition.llm import openrouter_client

logger = logging.getLogger(__name__)

# Declares the access code in the documentation, so that /docs has an "Authorize" button.
# Its value is not used: the code is checked by AccessMiddleware before the body is read.
BEARER = HTTPBearer(
    auto_error=False,
    description="The access code (ACCESS_CODE), sent as Authorization: Bearer <code>.",
)

# Room in an upload's body for the multipart envelope around the file (boundaries, the
# part's headers with the file name): the size limit is on the file, MAX_FILE_BYTES.
ENVELOPE = 64 * 1024

# When a refused upload may be tried again, in seconds.
RETRY_AFTER = 30

# The bytes of a file never change for an id: a browser may keep them, for itself only.
IMMUTABLE = "private, max-age=31536000, immutable"

# What the status of a job is now must never be answered from a cache.
NO_STORE = "no-store"

# Where documents are uploaded (POST): the one route whose body is a file.
UPLOAD_PATH = "/invoices"

# The body of a decision is a few dozen bytes; anything near this is not one.
MAX_DECISION_BYTES = 1024


class _App(FastAPI):
    def openapi(self) -> dict[str, Any]:
        """The generated schema, with error answers under their real media type."""
        return problems.document(super().openapi())


class Health(BaseModel):
    status: Literal["ok"] = "ok"


def get_service(request: Request) -> Service:
    """The reading service of the app, started by its lifespan."""
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise RuntimeError("the reading service is not running: the app was not started")
    if not isinstance(service, Service):
        raise TypeError(f"app.state.service is a {type(service).__name__}, not a Service")
    return service


ServiceDep = Annotated[Service, Depends(get_service)]


def _problem(description: str) -> dict[str, Any]:
    return {"model": Problem, "description": description}


def content_disposition(filename: str) -> str:
    """Content-Disposition for showing a file in the browser under the name it was uploaded
    with. The name is untrusted: quotes, backslashes and control characters are removed;
    a name that is not plain URL-safe ASCII is sent percent-encoded as UTF-8 (filename*),
    as Starlette's FileResponse does. No name left: no filename at all."""
    name = "".join(c for c in filename if c not in '"\\' and unicodedata.category(c) != "Cc")
    if not name:
        return "inline"
    encoded = quote(name)
    if encoded == name:
        return f'inline; filename="{name}"'
    return f"inline; filename*=utf-8''{encoded}"


def _job(service: Service, job_id: str) -> Job:
    """The job with this id; 404 not_found when none is kept (never kept, or dropped by
    retention)."""
    job = service.store.get(job_id)
    if job is None:
        raise ApiError(404, "not_found", messages.JOB_NOT_FOUND)
    return job


def _review(job: Job) -> views.ReviewView:
    """The review of a job a decision was just taken on (so it can be reviewed)."""
    reviewed = views.review_view(job)
    if reviewed is None:  # review.decide and review.undo refuse such a job first
        raise RuntimeError(f"job {job.id} has no review after a decision")
    return reviewed


_NOT_FOUND = _problem(
    "No document with this id is kept (not_found), or it has no issue with this id "
    "(issue_not_found)."
)
_NOT_REVIEWABLE = "The document is not finished, or it failed (not_reviewable)."


def _invoices(settings: Settings) -> APIRouter:
    router = APIRouter(
        prefix="/invoices",
        tags=["invoices"],
        dependencies=[Depends(BEARER)],
        responses={401: _problem("The access code is missing or wrong (unauthorized).")},
    )
    limits = uploads.UploadLimits(
        max_file_bytes=settings.max_file_bytes,
        max_pages=settings.max_pages,
        max_image_pixels=settings.max_image_pixels,
    )

    def busy() -> ApiError:
        return ApiError(
            503,
            "too_many_documents",
            messages.TOO_MANY_DOCUMENTS.format(limit=settings.max_documents),
            headers={"Retry-After": str(RETRY_AFTER)},
        )

    @router.post(
        "",
        status_code=202,
        summary="Upload a document",
        # A body larger than this is refused while it arrives, before the route runs.
        max_body_size=settings.max_file_bytes + ENVELOPE,
        # One part, the file: a second file or any other field is a malformed request.
        form_limits={"max_files": 1, "max_fields": 0},
        responses={
            202: {
                "description": "Accepted: the document waits to be read. The body is the job.",
                "headers": {
                    "Location": {
                        "description": "Where to follow the job: /invoices/{id} (relative).",
                        "schema": {"type": "string"},
                    }
                },
            },
            400: _problem("Not one file part, or a form that cannot be read (invalid_request)."),
            413: _problem(
                "The file is larger than the limit (file_too_large). When the request declares "
                "its length, the refusal comes before the body is read, as plain text."
            ),
            415: _problem("Not a PDF, JPEG or PNG, judged by its content (unsupported_file)."),
            422: _problem(
                "The file does not open, needs a password, has no pages or too many pixels "
                "(unreadable_file); a PDF with too many pages (too_many_pages); no file part "
                "(invalid_request)."
            ),
            503: _problem(
                "As many documents as allowed are waiting or being read (too_many_documents); "
                "Retry-After says when to try again."
            ),
        },
    )
    async def upload(
        file: Annotated[UploadFile, File(description="The invoice: a PDF, a JPEG or a PNG.")],
        service: ServiceDep,
        response: Response,
    ) -> views.JobView:
        """Upload an invoice to be read. Only cheap checks run here; the reading happens in
        the background: follow the job at the address in Location."""
        if not service.store.has_room():  # before any work on the file
            raise busy()
        # One byte more than allowed is enough to know the file is too large. A file
        # over 1 MB is spooled to disk by the form parser; it is read here, once.
        data = await file.read(limits.max_file_bytes + 1)
        if len(data) > limits.max_file_bytes:
            raise ApiError(
                413,
                "file_too_large",
                messages.FILE_TOO_LARGE.format(limit=messages.size(limits.max_file_bytes)),
            )
        # pypdf and Pillow: on the lane, the one thread for local work on files.
        loop = asyncio.get_running_loop()
        checked = await loop.run_in_executor(service.lane, uploads.inspect, data, limits)
        # From here on no await: the check of the room and the adding of the job cannot be
        # split by another upload.
        job = Job(file.filename or "", checked.kind, checked.mime, data, checked.pages_total)
        try:
            service.submit(job)
        except StoreFull:
            raise busy() from None
        response.headers["Location"] = f"/invoices/{job.id}"
        return views.job_view(job)

    @router.get("", summary="Every document held")
    async def list_invoices(service: ServiceDep, response: Response) -> views.JobList:
        """Every document the server still holds, newest first, without results; a
        finished one has its counts of lines and issues. The page polls this list."""
        response.headers["Cache-Control"] = NO_STORE
        return views.JobList(items=[views.job_item(j) for j in service.store.newest_first()])

    @router.get(
        "/{job_id}",
        summary="Status and result of a document",
        responses={404: _problem("No document with this id is kept (not_found).")},
    )
    async def get_invoice(job_id: str, service: ServiceDep, response: Response) -> views.JobView:
        """The status of a document, its progress and, once it is finished, the result.

        Final statuses: ok, needs_review, failed. Poll until one of them comes back."""
        job = _job(service, job_id)
        response.headers["Cache-Control"] = NO_STORE
        return views.job_view(job)

    @router.get(
        "/{job_id}/file",
        summary="The uploaded file",
        response_class=Response,
        responses={
            200: {
                "description": "The file as it was uploaded.",
                "content": {mime: {} for mime in ("application/pdf", "image/jpeg", "image/png")},
            },
            404: _problem("No document with this id is kept (not_found)."),
        },
    )
    async def get_file(job_id: str, service: ServiceDep) -> Response:
        """The original bytes, with the type decided at upload, to show in the browser. No
        range requests: the file is sent whole."""
        job = _job(service, job_id)
        # Starlette's Response sets Content-Length; no compression middleware is added.
        return Response(
            job.data,
            media_type=job.mime,
            headers={
                "Content-Disposition": content_disposition(job.filename),
                # Shown as the type decided at upload, never as what a browser guesses.
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": IMMUTABLE,
            },
        )

    @router.put(
        "/{job_id}/issues/{issue_id}/resolution",
        summary="Decide on an issue",
        max_body_size=MAX_DECISION_BYTES,
        responses={
            404: _NOT_FOUND,
            409: _problem(
                f"{_NOT_REVIEWABLE} Or the item would set a cell that another accepted "
                "decision set to a different value (resolution_conflict); a numeric cell "
                "(quantity, unit price, tax rate, amount, subtotal, tax, total) compares "
                "values as numbers, every other cell exactly."
            ),
            413: _problem(
                "The body is larger than 1 KiB (body_too_large). When the request declares "
                "its length, the refusal comes before the body is read, as plain text."
            ),
            422: _problem(
                "The body is not one of the two shapes, or issue_id is not an integer "
                "(invalid_request; checked before the document is looked up); accept on "
                "an issue without suggestions, or with no item of that number "
                "(invalid_resolution)."
            ),
        },
    )
    async def decide(
        job_id: str,
        issue_id: int,
        decision: views.ResolutionRequest,
        service: ServiceDep,
        response: Response,
    ) -> views.ReviewView:
        """Accept one of the issue's suggested values (`{"action": "accept", "suggestion": k}`:
        every cell of item k's `set`) or keep the values as read (`{"action": "keep"}`, also
        "mark as checked"). The same request again changes nothing; a different one replaces
        the decision. The answer is the document's review.

        Decisions never change the result: values stay as read, issues and status as found,
        and no check is run again. A cell's final value is the `set` of its accepted
        decision, else as read; cells never collide (409)."""
        job = _job(service, job_id)
        # No await from here on: the decision is checked and stored in one step.
        review.decide(job, issue_id, decision, now())
        response.headers["Cache-Control"] = NO_STORE
        return _review(job)

    @router.delete(
        "/{job_id}/issues/{issue_id}/resolution",
        summary="Take a decision back",
        responses={
            404: _NOT_FOUND,
            409: _problem(_NOT_REVIEWABLE),
            422: _problem("issue_id is not an integer (invalid_request)."),
        },
    )
    async def undo(
        job_id: str, issue_id: int, service: ServiceDep, response: Response
    ) -> views.ReviewView:
        """Take back the decision on an issue (undo); the issue is open again, and so is
        any issue it settled. With no decision taken, nothing changes. The answer is the
        document's review."""
        job = _job(service, job_id)
        review.undo(job, issue_id)
        response.headers["Cache-Control"] = NO_STORE
        return _review(job)

    return router


def create_app(settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    """The API with the given settings.

    transport replaces the network for the requests to the model provider (tests pass an
    `httpx.MockTransport`); None in production."""

    code = settings.access_code.get_secret_value()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # One client for every model request of the process, and the workers that send
        # them; both end with the app. Cancelled workers are awaited before the client is
        # closed, so no request is left using it.
        if not code:
            logger.warning(
                "ACCESS_CODE is not set: the /invoices routes are open to anyone who can "
                "reach this server. Set it for anything reachable from outside."
            )
        client = openrouter_client(
            settings.openrouter_api_key.get_secret_value(), transport=transport
        )
        service = Service(settings, client)
        service.start()
        app.state.service = service
        try:
            yield
        finally:
            await service.stop()
            await client.aclose()

    app = _App(
        title="Invoice recognition",
        summary="Reads invoices (PDF, scan or photo) and checks what was read.",
        version=version("invoice-recognition"),
        lifespan=lifespan,
        # One documentation page (/docs, Swagger UI) is enough.
        redoc_url=None,
        # "/health/" is not "/health": the automatic redirect would name an http:// address,
        # wrong behind an HTTPS tunnel.
        redirect_slashes=False,
        exception_handlers=problems.handlers(
            max_file_bytes=settings.max_file_bytes, upload_path=UPLOAD_PATH
        ),
        responses=problems.RESPONSES,
    )

    @app.get("/health", tags=["service"])
    async def health() -> Health:
        """The process is up and answers requests."""
        return Health()

    # HEAD too, for the health checkers that send it; documented once, as GET.
    app.head("/health", include_in_schema=False)(health)

    app.include_router(_invoices(settings))
    if code:
        # Outside the router, where the body limit and the form parsing are: it answers
        # before anything reads the body.
        app.add_middleware(AccessMiddleware, code=code)
    return app
