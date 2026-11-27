"""The FastAPI application: the routes, and the reading service that lives as long as
the app does."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import version
from typing import Annotated, Any, Literal

import httpx
from fastapi import APIRouter, Depends, FastAPI, File, Request, Response, UploadFile
from pydantic import BaseModel

from invoice_recognition.api import messages, problems, uploads, views
from invoice_recognition.api.jobs import Job, StoreFull
from invoice_recognition.api.problems import ApiError, Problem
from invoice_recognition.api.service import Service
from invoice_recognition.api.settings import Settings
from invoice_recognition.llm import openrouter_client

# Room in an upload's body for the multipart envelope around the file (boundaries, the
# part's headers with the file name): the size limit is on the file, MAX_FILE_BYTES.
ENVELOPE = 64 * 1024

# When a refused upload may be tried again, in seconds.
RETRY_AFTER = 30

# What the status of a job is now must never be answered from a cache.
NO_STORE = "no-store"


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


def _invoices(settings: Settings) -> APIRouter:
    router = APIRouter(prefix="/invoices", tags=["invoices"])
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

    @router.get(
        "/{job_id}",
        summary="Status and result of a document",
        responses={404: _problem("No document with this id is kept (not_found).")},
    )
    async def get_invoice(job_id: str, service: ServiceDep, response: Response) -> views.JobView:
        """The status of a document, its progress and, once it is finished, the result.

        Final statuses: ok, needs_review, failed. Poll until one of them comes back."""
        job = service.store.get(job_id)
        if job is None:
            raise ApiError(404, "not_found", messages.JOB_NOT_FOUND)
        response.headers["Cache-Control"] = NO_STORE
        return views.job_view(job)

    return router


def create_app(settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    """The API with the given settings.

    transport replaces the network for the requests to the model provider (tests pass an
    `httpx.MockTransport`); None in production."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # One client for every model request of the process, and the workers that send
        # them; both end with the app. Cancelled workers are awaited before the client is
        # closed, so no request is left using it.
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
        # "/health/" is not "/health": the automatic redirect would name an http:// address,
        # wrong behind an HTTPS tunnel.
        redirect_slashes=False,
        exception_handlers=problems.handlers(max_file_bytes=settings.max_file_bytes),
        responses=problems.RESPONSES,
    )

    @app.get("/health", tags=["service"])
    async def health() -> Health:
        """The process is up and answers requests."""
        return Health()

    # HEAD too, for the health checkers that send it; documented once, as GET.
    app.head("/health", include_in_schema=False)(health)

    app.include_router(_invoices(settings))
    return app
