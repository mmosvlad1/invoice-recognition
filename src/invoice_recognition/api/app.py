"""The FastAPI application."""

from importlib.metadata import version
from typing import Literal

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

from invoice_recognition.api.settings import Settings


class Health(BaseModel):
    status: Literal["ok"] = "ok"


def create_app(settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    """The API with the given settings.

    transport replaces the network for the requests to the model provider (tests pass an
    `httpx.MockTransport`); None in production. The reading service that sends those
    requests is started with the app in a later step; until then neither the key nor the
    transport is used."""
    app = FastAPI(
        title="Invoice recognition",
        summary="Reads invoices (PDF, scan or photo) and checks what was read.",
        version=version("invoice-recognition"),
        # "/health/" is not "/health": the automatic redirect would name an http:// address,
        # wrong behind an HTTPS tunnel.
        redirect_slashes=False,
    )

    @app.get("/health", tags=["service"])
    async def health() -> Health:
        """The process is up and answers requests."""
        return Health()

    # HEAD too, for the health checkers that send it; documented once, as GET.
    app.head("/health", include_in_schema=False)(health)

    return app
