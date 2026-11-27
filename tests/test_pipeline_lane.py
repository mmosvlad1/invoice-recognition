"""The local work of the flow on a lane of its own: the event loop stays free while a file
is opened and measured. Fake OpenRouter, no network."""

import asyncio
import dataclasses
import os
import sys
import threading
from collections.abc import Iterator
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from types import BuiltinFunctionType, FrameType
from typing import Any

import cv2
import httpx
import numpy
import PIL
import pypdf
import pypdfium2
import pytest

from invoice_recognition.recognition import pipeline as pl
from invoice_recognition.recognition.config import Config
from invoice_recognition.recognition.results import Result

from drawn import jpeg, photo, scanned
from fakes import Answer, Handler, Server, header_for, line
from samples import saved_layout, truth_doc

HEAVY = (pypdfium2, pypdf, PIL, cv2, numpy)
"""The libraries of the local work. None of them may run on the event loop's thread."""

HEAVY_DIRS = {str(Path(m.__file__).parent) + os.sep: m.__name__ for m in HEAVY if m.__file__}
# OpenCV's functions are built-ins with neither a module nor an object they are bound to:
# they are told by identity.
CV2_FUNCTIONS = frozenset(f for f in vars(cv2).values() if isinstance(f, BuiltinFunctionType))


def _library_of_builtin(f: Any) -> str | None:
    if f in CV2_FUNCTIONS:
        return "cv2"
    module = getattr(f, "__module__", None) or type(getattr(f, "__self__", None)).__module__
    name = module.split(".")[0]
    return name if name in HEAVY_DIRS.values() else None


@contextmanager
def heavy_calls_on_this_thread() -> Iterator[list[tuple[str, str]]]:
    """Record every call into a HEAVY library made on this thread while the block runs, as
    (library, function): Python functions by the file they are defined in, built-in (C)
    functions by their module, the module of the object they are bound to, or (OpenCV) by
    identity. sys.setprofile is per thread, so work on other threads is not seen.

    Blind spots: the profiler reports neither a call through ctypes (pypdfium2.raw, called
    directly) nor a numpy ufunc (np.add, np.maximum ...), as neither is a built-in function.
    Such a call is caught only through the Python functions of the library around it;
    test_prepared_holds_only_plain_data covers the rest from the other side: nothing the
    loop gets from prepare() is an object of those libraries."""
    seen: list[tuple[str, str]] = []

    def hook(frame: FrameType, event: str, arg: Any) -> None:
        if event == "call":
            code = frame.f_code
            for folder, library in HEAVY_DIRS.items():
                if code.co_filename.startswith(folder):
                    seen.append((library, code.co_name))
        elif event == "c_call" and (library := _library_of_builtin(arg)) is not None:
            seen.append((library, getattr(arg, "__name__", "?")))

    sys.setprofile(hook)
    try:
        yield seen
    finally:
        sys.setprofile(None)


def run(
    data: bytes, mime: str, server: Handler, cfg: Config | None = None, **kwargs: Any
) -> Result:
    async def go() -> Result:
        async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
            return await pl.process(data, mime, client, cfg, **kwargs)

    return asyncio.run(go())


def text_layer() -> tuple[bytes, str, Server]:
    """A PDF with a text layer, read on the layout road."""
    data, _, header, pages = truth_doc("s_classic")
    answers: dict[int, list[Answer]] = {p: [rows] for p, rows in pages.items()}
    return data, "application/pdf", Server(header, answers, saved_layout("s_classic"))


def scan() -> tuple[bytes, str, Server]:
    """A scan of two pages, both measured before any request; the second is flagged."""
    rows = {1: [line(1)], 2: [line(2)]}
    answers: dict[int, list[Answer]] = {p: [r] for p, r in rows.items()}
    return (
        scanned("good", "soft"),
        "application/pdf",
        Server(header_for(rows[1] + rows[2]), answers),
    )


def a_photo() -> tuple[bytes, str, Server]:
    rows = [line(1), line(2)]
    return jpeg(photo()), "image/jpeg", Server(header_for(rows), {1: [rows]})


INPUTS = {"text_layer": text_layer, "scan": scan, "photo": a_photo}
# What each input calls when it is prepared on the loop's thread, as the check sees it.
EXPECTED = {
    "text_layer": {"pypdf", "pypdfium2"},
    "scan": {"pypdf", "pypdfium2", "PIL", "numpy", "cv2"},
    "photo": {"PIL", "numpy", "cv2"},
}


@pytest.mark.parametrize("kind", INPUTS)
def test_nothing_heavy_runs_on_the_loop_thread(kind: str) -> None:
    data, mime, server = INPUTS[kind]()
    with ThreadPoolExecutor(1) as lane, heavy_calls_on_this_thread() as seen:
        res = run(data, mime, server, lane=lane)

    assert seen == []
    assert res.status in ("ok", "needs_review") and res.lines  # the file was really read
    assert res.method == ("layout" if kind == "text_layer" else "model")
    if kind == "scan":
        assert res.quality["verdict"] == "warn"  # the pages were measured (on the lane)


@pytest.mark.parametrize("kind", INPUTS)
def test_the_check_sees_heavy_work_done_inline(kind: str) -> None:
    # The same run without a lane: the check above would catch the local work, in every
    # library it uses.
    data, mime, server = INPUTS[kind]()
    with heavy_calls_on_this_thread() as seen:
        run(data, mime, server)

    assert {library for library, _ in seen} == EXPECTED[kind]


def test_other_tasks_run_while_a_file_is_prepared(monkeypatch: pytest.MonkeyPatch) -> None:
    data, mime, server = scan()
    entered, release = threading.Event(), threading.Event()
    waits: list[bool] = []
    real = pl.prepare

    def blocking(data: bytes, mime: str, cfg: Config) -> pl.Prepared:
        entered.set()
        # Timed, so that a regression fails instead of hanging: if prepare() blocked the
        # loop, nothing could set the event while it waits.
        waits.append(release.wait(2))
        return real(data, mime, cfg)

    monkeypatch.setattr(pl, "prepare", blocking)

    async def go() -> Result:
        async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
            with ThreadPoolExecutor(1) as lane:
                task = asyncio.create_task(pl.process(data, mime, client, lane=lane))
                async with asyncio.timeout(2):
                    while not entered.is_set():
                        await asyncio.sleep(0.005)
                release.set()  # run by this task, on the loop, while prepare() is blocked
                return await task

    res = asyncio.run(go())
    assert waits == [True]
    assert res.lines


def test_without_a_lane_the_file_is_prepared_on_the_calling_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, mime, server = scan()
    threads: list[threading.Thread] = []
    real = pl.prepare

    def recording(data: bytes, mime: str, cfg: Config) -> pl.Prepared:
        threads.append(threading.current_thread())
        return real(data, mime, cfg)

    monkeypatch.setattr(pl, "prepare", recording)
    run(data, mime, server)
    lane: Executor
    with ThreadPoolExecutor(1, thread_name_prefix="local-work") as lane:
        run(data, mime, server, lane=lane)

    assert threads[0] is threading.current_thread()
    assert threads[1].name.startswith("local-work")


PLAIN = (str, bytes, int, float, bool, type(None), Decimal)


def foreign(value: Any, path: str = "prepared") -> list[str]:
    """Every value inside `value` that is not plain data: builtins, Decimal, or a dataclass
    of this package holding plain data. Returns their paths and types."""
    if isinstance(value, PLAIN):
        return []
    if isinstance(value, dict):
        return [
            f for k, v in value.items() for f in foreign(k, path) + foreign(v, f"{path}[{k!r}]")
        ]
    if isinstance(value, list | tuple):
        return [f for i, v in enumerate(value) for f in foreign(v, f"{path}[{i}]")]
    kind = type(value)
    if dataclasses.is_dataclass(value) and kind.__module__.startswith("invoice_recognition."):
        return [
            f
            for field in dataclasses.fields(value)
            for f in foreign(getattr(value, field.name), f"{path}.{field.name}")
        ]
    return [f"{path}: {kind.__module__}.{kind.__qualname__}"]


@pytest.mark.parametrize("kind", INPUTS)
def test_prepared_holds_only_plain_data(kind: str) -> None:
    # The reading on the loop gets only Prepared: no document, page, image or array of the
    # heavy libraries can travel with it and be used there.
    data, mime, _ = INPUTS[kind]()
    prep = pl.prepare(data, mime, Config())

    assert prep.failure is None and prep.parts
    if kind == "text_layer":
        assert prep.evidence is not None and prep.text_pages is not None
    assert foreign(prep) == []
    assert foreign({"x": [numpy.zeros(1)]}) == ["prepared['x'][0]: numpy.ndarray"]  # it sees one
