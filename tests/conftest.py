"""Async tests (the API's) run on both event loops the app can be served on.

`uvicorn` (what `fastapi run`, `fastapi dev` and the container start) serves the app on
uvloop when it is installed, as it is with the locked dependencies on macOS and Linux and
in the image; otherwise on asyncio's own loop. The two differ where timing matters: a
timer fires up to one tick of the clock early on asyncio (1 ms in Docker Desktop's Linux)
and up to about 2 ms early on uvloop. So every test marked `@pytest.mark.anyio` runs once
on each (the anyio pytest plugin runs it once per value of this fixture). The flow's own
tests stay plain functions that call asyncio.run(); the deadline's run on both loops too
(tests/test_pipeline_deadline.py)."""

from importlib.util import find_spec
from typing import Any

import pytest

LOOPS: list[tuple[str, dict[str, Any]]] = [("asyncio", {})]
if find_spec("uvloop") is not None:
    LOOPS.append(("asyncio", {"use_uvloop": True}))


@pytest.fixture(params=LOOPS, ids=lambda loop: "uvloop" if loop[1] else "asyncio")
def anyio_backend(request: pytest.FixtureRequest) -> tuple[str, dict[str, Any]]:
    backend: tuple[str, dict[str, Any]] = request.param
    return backend
