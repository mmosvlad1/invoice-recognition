"""The FastAPI application: the routes, and the reading service that lives as long as
the app does."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import version
from typing import Annotated, Any, Literal

import httpx
from fastapi import APIRouter, Depends, FastAPI, Request, Response
from pydantic import BaseModel

from invoice_recognition.api import messages, problems, views
from invoice_recognition.api.problems import ApiError, Problem
from invoice_recognition.api.service import Service
from invoice_recognition.api.settings import Settings
from invoice_recognition.llm import openrouter_client

# The largest file an upload may carry (the multipart envelope not counted). It becomes
# a setting with the upload route.
MAX_FILE_BYTES = 10 * 1024 * 1024

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


def _invoices() -> APIRouter:
    router = APIRouter(prefix="/invoices", tags=["invoices"])

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
        exception_handlers=problems.handlers(max_file_bytes=MAX_FILE_BYTES),
        responses=problems.RESPONSES,
    )

    @app.get("/health", tags=["service"])
    async def health() -> Health:
        """The process is up and answers requests."""
        return Health()

    # HEAD too, for the health checkers that send it; documented once, as GET.
    app.head("/health", include_in_schema=False)(health)

    app.include_router(_invoices())
    return app
