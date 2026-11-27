"""Async tests (the API's) run on asyncio only, the loop the app is served on.

The anyio pytest plugin runs a test marked `@pytest.mark.anyio` once per backend this
fixture names. The flow's own tests stay plain functions that call asyncio.run()."""

import pytest


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
