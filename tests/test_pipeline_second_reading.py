"""Scanned pages read a second time by a model of another provider, in the flow. Fake
OpenRouter, no network.

The second reader is told apart by the model named in the request: it is asked with the
same prompt as the first reader of a one-page scan, so the prompt cannot tell them apart.
Blank drawn PDFs (scan()) carry no image: nothing is measured, every page is read."""

import asyncio
import json
import re
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import httpx
import pytest

from invoice_recognition.llm import PDF_NATIVE
from invoice_recognition.recognition import pipeline as pl
from invoice_recognition.recognition import second_reading as sr
from invoice_recognition.recognition.checks import Issue
from invoice_recognition.recognition.config import FAST, OTHER, Config
from invoice_recognition.recognition.prompts import EXTRACT_PAGE
from invoice_recognition.recognition.results import PageResult, Result
from invoice_recognition.shapes import JSONObject, Line

from drawn import scan
from fakes import (
    Answer,
    Handler,
    Server,
    SlowServer,
    header_for,
    line,
    model_of,
    never,
    recording,
    run,
    sse,
)
from golden_cases import CASES, GOLDEN, Case, Model, request_of


def on(cfg: Config | None = None) -> Config:
    """The configuration with the second reader switched on."""
    return replace(cfg or Config(), second_reading_model=OTHER)


CUT = re.compile(r"timeout after \d+s")


def rows(pages: int) -> dict[int, list[Answer]]:
    """One line item per page, numbered by its page."""
    return {p: [[line(p)]] for p in range(1, pages + 1)}


def server_for(pages: int, second: dict[int, list[Answer]] | None = None) -> Server:
    """The first reader answers every page with its line item; the second reader answers
    with `second` (by default the same)."""
    first = rows(pages)
    every = [line(p) for p in range(1, pages + 1)]
    return Server(header_for(every), first, models={OTHER: Server({}, second or first)})


def raw(server: Handler) -> tuple[Handler, list[httpx.Request]]:
    """The server, and every request it got, in order, as sent."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return server(request)

    return handler, seen


def to(model: str, requests: list[httpx.Request]) -> list[bytes]:
    return [r.content for r in requests if model_of(r) == model]


def filename(body: JSONObject) -> str:
    name: str = body["messages"][0]["content"][1]["file"]["filename"]
    return name


def second_readings(res: Result) -> dict[int, str | None]:
    """The pages read a second time, with the error of that request."""
    return {p.page: p.second_reading.error for p in res.pages if p.second_reading is not None}


def missing(res: Result) -> list[JSONObject]:
    return [n for n in res.notes if n["kind"] == "second_reading_missing"]


def run_async(
    data: bytes,
    mime: str,
    server: Handler | SlowServer,
    cfg: Config,
    *,
    slots: asyncio.Semaphore | None = None,
    on_page: Callable[[PageResult], None] | None = None,
) -> tuple[Result, float]:
    """process() with a server that may answer asynchronously; also how long it took."""

    async def go() -> tuple[Result, float]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
            started = time.monotonic()
            res = await pl.process(data, mime, client, cfg, on_page, request_slots=slots)
            return res, time.monotonic() - started

    return asyncio.run(go())


# --------------------------------------------------------------------------- #
# The request
# --------------------------------------------------------------------------- #


def test_every_page_of_a_scan_is_read_once_more_with_the_plain_page_prompt() -> None:
    handler, bodies = recording(server_for(3))
    res = run(scan(3), "application/pdf", handler, on())

    first = {filename(b): b for b in bodies if b["model"] == FAST and b["max_tokens"] == 16_000}
    second = [b for b in bodies if b["model"] == OTHER]
    assert sorted(filename(b) for b in second) == ["page1.pdf", "page2.pdf", "page3.pdf"]
    for body in second:
        p = int(filename(body)[4:-4])
        gemini = first[f"page{p}.pdf"]
        assert body == {
            "model": OTHER,
            "stream": True,
            "usage": {"include": True},
            "max_tokens": 24_000,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        # Not the "carry_in" prompt the first reader of a scan gets.
                        {"type": "text", "text": EXTRACT_PAGE.format(page=p, total=3)},
                        gemini["messages"][0]["content"][1],  # the same file part
                    ],
                }
            ],
            "response_format": {"type": "json_object"},
            "plugins": PDF_NATIVE,
        }
        assert gemini["messages"][0]["content"][0]["text"] != EXTRACT_PAGE.format(page=p, total=3)
    assert res.status == "ok" and res.issues == [] and res.notes == []
    assert second_readings(res) == {1: None, 2: None, 3: None}
    assert [p.second_reading.model for p in res.pages if p.second_reading] == [OTHER] * 3
    assert [p.second_reading.lines for p in res.pages if p.second_reading] == [1, 1, 1]
    # Counted in the document's sums: the header, three pages, three second readings.
    assert res.calls == 7 and res.cost_usd == pytest.approx(0.007)
    assert (res.prompt_tokens, res.completion_tokens) == (700, 350)


def test_switched_off_by_default_nothing_more_is_sent() -> None:
    assert Config().second_reading_model is None
    handler, bodies = recording(server_for(2))
    res = run(scan(2), "application/pdf", handler)
    assert [b["model"] for b in bodies] == [FAST] * 3
    assert all(p.second_reading is None for p in res.pages)
    assert all("second_reading" not in p for p in res.to_json()["pages"])


def test_the_json_of_a_page_read_twice_has_its_request() -> None:
    res = run(scan(1), "application/pdf", server_for(1), on())
    (page,) = res.to_json()["pages"]
    assert page["second_reading"] == {
        "model": OTHER,
        "duration_s": page["second_reading"]["duration_s"],
        "cost_usd": 0.001,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "lines": 1,
        "issues": 0,
        "error": None,
        "cost_known": True,
    }
    assert len(page["attempts"]) == 1  # not one of the attempts at the page


def by_name(name: str) -> Case:
    return next(c for c in CASES if c.name == name)


def agreeing(model: Model) -> Model:
    """A second reader that answers every page as the scenario's model last answers it."""
    return Model([{}], {p: [answers[-1]] for p, answers in model.pages.items()})


@pytest.mark.parametrize("served", [False, True], ids=["plain", "served"])
@pytest.mark.parametrize("name", ["scan_retries", "scan_failures", "scan_ocr_overlay"])
def test_the_first_readers_requests_do_not_change(name: str, served: bool) -> None:
    """With retries, a failed request, a cut answer and a scan with OCR text over it: the
    requests to the first reader are those of the same run without the second reader,
    byte for byte and in the same order."""
    case = by_name(name)
    data, mime = case.data()

    def go(second: bool) -> tuple[Result, list[httpx.Request]]:
        model = case.model()
        model.models = {OTHER: agreeing(model)}
        handler, seen = raw(model)
        cfg = Config(**case.config, deadline_s=240.0) if served else Config(**case.config)
        cfg = on(cfg) if second else cfg
        if not served:
            return run(data, mime, handler, cfg), seen
        return run_async(data, mime, handler, cfg, slots=asyncio.Semaphore(24))[0], seen

    res_off, without = go(False)
    res_on, with_it = go(True)
    assert to(FAST, with_it) == to(FAST, without) and to(FAST, without)
    assert len(to(OTHER, with_it)) == len([p for p in res_on.pages if p.method != "skipped"])
    assert to(OTHER, without) == []
    assert res_on.status == res_off.status and res_on.lines == res_off.lines
    # The readers agree: nothing is added.
    assert [i.to_json() for i in res_on.issues] == [i.to_json() for i in res_off.issues]


def test_photos_and_pdfs_with_a_text_layer_are_never_read_twice() -> None:
    for case in (by_name(n) for n in ("photo_soft", "layout_confirmed", "layout_swapped")):
        data, mime = case.data()
        model = case.model()
        model.models = {OTHER: never}
        res = run(data, mime, model, on(Config(**case.config)))
        assert all(p.second_reading is None for p in res.pages), case.name
    # Every page read by the model with the text layer as evidence: after a failed
    # layout, and with the layout road switched off.
    for case in (by_name(n) for n in ("layout_failed", "text_layer_carry")):
        data, mime = case.data()
        model = case.model()
        model.models = {OTHER: never}
        res = run(data, mime, model, on(Config(**case.config)))
        assert res.evidence_source == "text_layer" and res.layout["pages_model"] > 0
        assert all(p.second_reading is None for p in res.pages), case.name


def test_only_pages_sent_without_a_quality_warning_are_read_twice() -> None:
    """A drawn scan: a good page, one too blurred to send, a blank one and a soft one that
    is read with a warning. Only the good page is read twice, and the first reader's
    requests are exactly the pinned ones."""
    case = by_name("scan_quality")
    data, mime = case.data()
    model = case.model()
    other, bodies = recording(agreeing(model))
    model.models = {OTHER: other}
    res = run(data, mime, model, on(Config(**case.config)))
    assert [filename(b) for b in bodies] == ["page1.pdf"]
    assert second_readings(res) == {1: None}
    golden = json.loads((GOLDEN / "scan_quality.json").read_text())
    assert [request_of(b) for b in model.bodies] == golden["requests"]
    assert res.status == "needs_review" and missing(res) == []
    # Without the measurement every page is sent, and read twice.
    res = run(data, mime, server_for(4), on(Config(scan_quality=False)))
    assert second_readings(res) == {1: None, 2: None, 3: None, 4: None}


# --------------------------------------------------------------------------- #
# A second reading that fails changes nothing but a note
# --------------------------------------------------------------------------- #


def stream(text: str, finish: str = "stop") -> str:
    events = [
        {"choices": [{"delta": {"content": text}}]},
        {
            "choices": [{"delta": {}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.001},
        },
    ]
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"


FAILURES: dict[str, tuple[Answer, str | None, bool]] = {
    # answer of the second reader to page 1: its error, whether its cost is known
    "http_503": (httpx.Response(503, text="busy"), "HTTP 503: busy", False),
    "error_event": (
        'data: {"error": {"message": "overloaded"}}\n\n',
        '{"message": "overloaded"}',
        False,
    ),
    "transport": (httpx.ConnectError("refused"), "ConnectError: refused", False),
    "not_json": (sse("the page shows an invoice"), None, True),
    "cut_at_the_token_limit": (stream(json.dumps({"rows": [line(1)]}), "length"), None, True),
    "rows_not_a_list": ({"rows": "none"}, None, True),
    "huge_number": (sse('{"rows": [{"sku": "A", "amount": ' + "9" * 5_000 + "}]}"), None, True),
    "deep_nesting": (sse('{"rows": ' + "[" * 1_000_000 + "]" * 1_000_000 + "}"), None, True),
}

NOT_READ = (
    "Page 1 was not read a second time: the other model did not answer, or not in time. "
    "Nothing else checks the item codes (SKUs) on this page."
)
UNUSABLE = (
    "Page 1 has no usable second reading: the other model's answer was broken or had no "
    "line items. Nothing else checks the item codes (SKUs) on this page."
)


@pytest.mark.parametrize("name", list(FAILURES))
def test_a_second_reading_that_cannot_be_used_leaves_a_note(name: str) -> None:
    answer, error, cost_known = FAILURES[name]
    server = server_for(2, {1: [answer], 2: [[line(2)]]})
    plain = run(scan(2), "application/pdf", server_for(2))
    res = run(scan(2), "application/pdf", server, on())

    assert (res.status, res.issues, res.lines) == ("ok", [], plain.lines)
    assert missing(res) == [
        {
            "page": None,
            "kind": "second_reading_missing",
            "text": NOT_READ if error else UNUSABLE,
            "source": "second_reading",
            "pages": [1],
        }
    ]
    assert second_readings(res) == {1: error, 2: None}
    first = res.pages[0].second_reading
    assert first is not None and (first.cost_known, first.lines) == (cost_known, 0)
    assert res.calls == plain.calls + 2
    assert res.cost_unknown_calls == (0 if cost_known else 1)
    assert res.cost_usd == pytest.approx(plain.cost_usd + 0.001 * (1 + cost_known))


def test_a_second_reading_slower_than_its_timeout_is_given_up() -> None:
    slow = SlowServer(server_for(3), models={OTHER: {1: [30.0], 3: [30.0]}})
    res, took = run_async(
        scan(3), "application/pdf", slow, replace(on(), second_reading_timeout_s=0.3)
    )
    assert took < 2
    errors = second_readings(res)
    assert errors[2] is None and CUT.fullmatch(errors[1] or "") and CUT.fullmatch(errors[3] or "")
    assert res.status == "ok" and res.issues == []
    (note,) = missing(res)
    assert note["pages"] == [1, 3]
    assert note["text"] == (
        "Pages 1 and 3 were not read a second time: the other model did not answer, or not in "
        "time. Nothing else checks the item codes (SKUs) on these pages."
    )
    assert res.cost_unknown_calls == 2


def test_a_second_reading_that_breaks_does_not_fail_the_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def broken(*args: Any) -> None:
        raise ValueError("unexpected")

    monkeypatch.setattr(pl, "read_second", broken)
    res = run(scan(2), "application/pdf", server_for(2), on())
    assert res.status == "ok" and res.issues == [] and len(res.lines) == 2
    assert second_readings(res) == {}
    assert [n["pages"] for n in missing(res)] == [[1, 2]]


# --------------------------------------------------------------------------- #
# Time, limits and progress
# --------------------------------------------------------------------------- #


def test_a_second_reading_cut_or_stopped_by_the_deadline_is_no_deadline_issue() -> None:
    """The first reader answers at once; the second reader's page 1 would take 30 s and
    holds the document's only slot for second readings, so page 2's never starts."""
    slow = SlowServer(server_for(2), models={OTHER: {1: [30.0]}})
    cfg = on(Config(deadline_s=1.0, concurrency=1))
    res, took = run_async(scan(2), "application/pdf", slow, cfg)

    assert took < 3
    assert res.status == "ok" and res.issues == []  # no deadline_exceeded
    assert slow.calls_of[OTHER] == {1: 1}  # page 2 was not sent
    cut = res.pages[0].second_reading
    assert cut is not None and CUT.fullmatch(cut.error or "") and not cut.cost_known
    assert res.pages[1].second_reading is None
    assert "second_reading" not in res.to_json()["pages"][1]
    assert [n["pages"] for n in missing(res)] == [[1, 2]]
    assert res.cost_unknown_calls == 1 and res.calls == 4


def test_second_readings_take_shared_slots() -> None:
    slow = SlowServer(
        server_for(6),
        {p: [0.05] for p in range(1, 7)},
        models={OTHER: {p: [0.05] for p in range(1, 7)}},
    )
    res, _ = run_async(scan(6), "application/pdf", slow, on(), slots=asyncio.Semaphore(3))
    assert len(second_readings(res)) == 6
    assert slow.most == 3


def test_second_readings_have_a_limit_of_their_own_in_a_document() -> None:
    """Each within Config.concurrency, and not in the pages' limit: a retry of the first
    reader does not wait behind them."""
    sleeps: dict[int | str, list[float]] = {p: [0.05] for p in range(1, 7)}
    slow = SlowServer(server_for(6), sleeps, models={OTHER: sleeps})
    res, _ = run_async(scan(6), "application/pdf", slow, on(Config(concurrency=2)))
    assert len(second_readings(res)) == 6
    assert slow.pages_most == 2 and slow.most_of[OTHER] == 2
    assert slow.most >= 4


def test_a_page_is_announced_without_waiting_for_its_second_reading() -> None:
    announced: list[float] = []
    started = time.monotonic()
    slow = SlowServer(server_for(3), models={OTHER: {p: [0.8] for p in range(1, 4)}})
    res, took = run_async(
        scan(3),
        "application/pdf",
        slow,
        on(),
        on_page=lambda pr: announced.append(time.monotonic() - started),
    )
    assert len(announced) == 3 and max(announced) < 0.5 and took >= 0.8
    assert len(second_readings(res)) == 3


# --------------------------------------------------------------------------- #
# Shared slots: the first readers go first
# --------------------------------------------------------------------------- #


def retried(pages: int) -> Server:
    """Every page's first attempt has a wrong amount and is read again; the second reader
    answers every page right."""
    first: dict[int, list[Answer]] = {
        p: [[line(p, amount=1.0)], [line(p)]] for p in range(1, pages + 1)
    }
    every = [line(p) for p in range(1, pages + 1)]
    return Server(header_for(every), first, models={OTHER: Server({}, rows(pages))})


def same_reading(a: Result, b: Result) -> None:
    """What the first reader returned is the same: lines, attempts, issues (but those of
    the second reading), status."""
    assert (a.status, a.lines) == (b.status, b.lines)
    assert [len(p.attempts) for p in a.pages] == [len(p.attempts) for p in b.pages]
    mine = [i.to_json() for i in a.issues if i.check != "second_reading"]
    assert mine == [i.to_json() for i in b.issues if i.check != "second_reading"]


def test_a_retry_never_waits_behind_a_second_reading_for_a_shared_slot() -> None:
    """Two shared slots, four pages that each need a second attempt, a deadline: the
    second readings (3 s each) must not take a slot a retry needs."""

    def go(cfg: Config) -> tuple[Result, SlowServer]:
        sleeps: dict[int | str, list[float]] = {p: [0.3] for p in range(1, 5)}
        slow = SlowServer(retried(4), sleeps, models={OTHER: {p: [3.0] for p in range(1, 5)}})
        res, _ = run_async(scan(4), "application/pdf", slow, cfg, slots=asyncio.Semaphore(2))
        return res, slow

    plain, slow_off = go(Config(deadline_s=2.0))
    twice, slow_on = go(on(Config(deadline_s=2.0)))
    assert plain.status == "ok" and [len(p.attempts) for p in plain.pages] == [2] * 4
    same_reading(twice, plain)
    assert slow_on.arrived == slow_off.arrived  # the first reader's requests, in order
    assert "deadline_exceeded" not in [i.check for i in twice.issues]
    # None of them could end before the deadline: each is missing, nothing else.
    assert [n["pages"] for n in missing(twice)] == [[1, 2, 3, 4]]


def test_documents_sharing_slots_are_read_as_fast_with_second_readings() -> None:
    """Three documents of ten pages started together, every page read twice by the first
    reader, 16 shared slots: the first readers' requests go out when they would without
    the second reader, and the last page is settled then too. The first document's
    second readings must not take the slots the other two start with."""

    async def go(second: bool) -> tuple[list[Result], float, SlowServer, float]:
        sleeps: dict[int | str, list[float]] = {p: [0.2] for p in range(1, 11)}
        slow = SlowServer(retried(10), sleeps, models={OTHER: {p: [3.0] for p in range(1, 11)}})
        slots, announced = asyncio.Semaphore(16), []
        cfg = Config(scan_quality=False, deadline_s=1.5)
        cfg = on(cfg) if second else cfg
        async with httpx.AsyncClient(transport=httpx.MockTransport(slow)) as client:
            started = time.monotonic()
            results = await asyncio.gather(
                *(
                    pl.process(
                        scan(10),
                        "application/pdf",
                        client,
                        cfg,
                        lambda pr: announced.append(time.monotonic() - started),
                        request_slots=slots,
                    )
                    for _ in range(3)
                )
            )
        return list(results), max(announced), slow, started

    plain, last_off, slow_off, start_off = asyncio.run(go(False))
    twice, last_on, slow_on, start_on = asyncio.run(go(True))
    for a, b in zip(twice, plain, strict=True):
        same_reading(a, b)
    assert [r.status for r in twice] == ["ok"] * 3
    assert slow_on.arrived == slow_off.arrived
    late = [
        (on_at - start_on) - (off_at - start_off)
        for on_at, off_at in zip(slow_on.arrived_at, slow_off.arrived_at, strict=True)
    ]
    assert max(late) < 0.1  # no request of a first reader went out later
    assert last_on == pytest.approx(last_off, abs=0.1)


def test_a_second_reading_that_never_gets_a_slot_is_only_missing() -> None:
    """A one-page scan shares two slots with a document whose pages want them all the
    time (it reads one page at a time, so the next always waits): the second reading is
    never sent before the deadline, and the document is otherwise as without it."""

    async def go() -> tuple[Result, float, SlowServer]:
        busy: dict[int, list[Answer]] = {p: [[line(p)]] for p in range(1, 9)}
        server = Server(
            header_for([line(1)]),
            {1: [[line(1)]]},
            models={
                OTHER: Server({}, rows(1)),
                "busy/model": Server(header_for([line(p) for p in range(1, 9)]), busy),
            },
        )
        slow = SlowServer(
            server, models={"busy/model": {p: [0.25] for p in range(1, 9)}, OTHER: {}}
        )
        slots = asyncio.Semaphore(2)
        other = Config(page_models=("busy/model",), header_model="busy/model", concurrency=1)
        async with httpx.AsyncClient(transport=httpx.MockTransport(slow)) as client:
            started = time.monotonic()

            async def mine() -> tuple[Result, float]:
                cfg = on(Config(deadline_s=1.0))
                res = await pl.process(scan(1), "application/pdf", client, cfg, request_slots=slots)
                return res, time.monotonic() - started

            (res, took), _ = await asyncio.gather(
                mine(),
                pl.process(scan(8), "application/pdf", client, other, request_slots=slots),
            )
        return res, took, slow

    res, took, slow = asyncio.run(go())
    assert took < 1.5
    assert slow.calls_of[OTHER] == {}  # never sent
    assert res.status == "ok" and res.issues == []
    assert res.pages[0].second_reading is None
    assert [n["text"] for n in missing(res)] == [NOT_READ]


# --------------------------------------------------------------------------- #
# The gate on its own: when an optional request may take a shared slot
# --------------------------------------------------------------------------- #


def test_optional_requests_hold_at_most_half_of_the_shared_slots() -> None:
    async def go() -> list[bool]:
        slots = asyncio.Semaphore(5)  # half, rounded down: 2
        gates = [pl.Gate(slots) for _ in range(4)]
        entering = [asyncio.create_task(g.enter(None, optional=True)) for g in gates]
        await asyncio.sleep(0.05)
        taken = [t.done() for t in entering]
        gates[0].leave(None, optional=True)  # one gives its slot back: one more may go
        await asyncio.sleep(0.05)
        taken += [t.done() for t in entering]
        for t in entering:
            t.cancel()
        return taken

    assert asyncio.run(go()) == [True, True, False, False, True, True, True, False]


def test_an_ordinary_request_that_gives_up_wakes_every_optional_one_that_can_go() -> None:
    """An ordinary request waits for its document's own limit, so optional requests wait
    although slots are free. When it gives up at its deadline no slot is given back, and
    yet all of them that fit must go at once."""

    async def go() -> tuple[bool, list[bool]]:
        slots = asyncio.Semaphore(6)  # optional requests may hold 3
        own = asyncio.Semaphore(1)
        await own.acquire()  # the document's limit is taken: its request waits
        ordinary = pl.Gate(slots, time.monotonic() + 0.2)
        waiting = asyncio.create_task(ordinary.enter(own))
        await asyncio.sleep(0)
        optional = [
            asyncio.create_task(pl.Gate(slots, time.monotonic() + 5).enter(None, optional=True))
            for _ in range(3)
        ]
        await asyncio.sleep(0.1)
        before = any(t.done() for t in optional)
        with pytest.raises(pl.DeadlinePassed):
            await waiting
        await asyncio.sleep(0.05)
        after = [t.done() for t in optional]
        for t in optional:
            t.cancel()
        return before, after

    assert asyncio.run(go()) == (False, [True, True, True])


# --------------------------------------------------------------------------- #
# A document that starts while others are being read twice
# --------------------------------------------------------------------------- #


def late_scan_of(pages: int) -> Server:
    first: dict[int, list[Answer]] = {p: [[line(p)]] for p in range(1, pages + 1)}
    every = [line(p) for p in range(1, pages + 1)]
    return Server(header_for(every), first, models={OTHER: Server({}, first)})


async def read_late(kind: str, delay: float, second: bool, slots: asyncio.Semaphore) -> list[float]:
    """One document started after `delay`: the times, from its start, at which its first
    reader's requests arrived and its last page was settled."""
    await asyncio.sleep(delay)
    request = 0.2  # every first-reader request; a second reading takes 0.5 s
    if kind == "scan":
        sleeps: dict[int | str, list[float]] = {"header": [request]}
        sleeps |= {p: [request] for p in range(1, 11)}
        slow = SlowServer(
            late_scan_of(10), sleeps, models={OTHER: {p: [0.5] for p in range(1, 11)}}
        )
        data, mime, cfg = scan(10), "application/pdf", Config(scan_quality=False)
    else:
        case = by_name("layout_confirmed")
        data, mime = case.data()
        sleeps = {"header": [request], "layout": [request]}
        sleeps |= {p: [request] for p in range(1, 30)}
        slow = SlowServer(case.model(), sleeps)
        cfg = Config(**case.config)
    cfg = replace(on(cfg) if second else cfg, deadline_s=10.0)
    settled: list[float] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(slow)) as client:
        started = time.monotonic()
        await pl.process(
            data,
            mime,
            client,
            cfg,
            lambda pr: settled.append(time.monotonic() - started),
            request_slots=slots,
        )
    return [at - started for at in slow.arrived_at] + [max(settled)]


@pytest.mark.parametrize(
    ("early", "late"),
    [(2, "scan"), (3, "scan"), (3, "pdf"), (2, "pdf")],
    ids=["2-scans-then-a-scan", "3-scans-then-a-scan", "3-scans-then-a-pdf", "2-scans-then-a-pdf"],
)
def test_a_document_that_starts_late_finds_slots_second_readings_cannot_take(
    early: int, late: str
) -> None:
    """Scans read twice start together and share 24 slots; another document starts once
    their first readers are done and their second readings are under way. Its first
    reader goes as fast as with the feature off: half of the slots are always out of the
    second readings' reach."""

    async def go(second: bool) -> list[float]:
        slots = asyncio.Semaphore(24)
        *_, timings = await asyncio.gather(
            *(read_late("scan", 0.0, second, slots) for _ in range(early)),
            read_late(late, 0.45, second, slots),
        )
        return timings

    def apart() -> float:
        off, with_second = asyncio.run(go(False)), asyncio.run(go(True))
        assert len(with_second) == len(off)
        return max(abs(a - b) for a, b in zip(with_second, off, strict=True))

    # These are wall-clock times of two separate runs, so one pause of the machine while
    # the late document is read moves them apart (seen once on a CI runner). The closest
    # of up to three measurements is judged. That makes the cases with three early scans
    # lenient: with the cap removed they come out close in some measurements; the case
    # of two scans and a late scan is apart in every one.
    closest = apart()
    for _ in range(2):
        if closest < 0.1:
            break
        closest = min(closest, apart())
    assert closest < 0.1


# What a disagreement becomes
# --------------------------------------------------------------------------- #


def with_sku(item: Line, sku: str) -> Line:
    return item | {"sku": sku}


def test_a_row_the_readers_disagree_on_is_flagged_and_keeps_the_first_reading() -> None:
    second: dict[int, list[Answer]] = {1: [[line(1)]], 2: [[with_sku(line(2), "SKU-0O2")]]}
    plain = run(scan(2), "application/pdf", server_for(2))
    res = run(scan(2), "application/pdf", server_for(2, second), on())

    assert plain.status == "ok"
    assert res.status == "needs_review" and res.lines == plain.lines
    (issue,) = res.issues
    assert issue.to_json() == {
        "check": "second_reading",
        "message": "Position 2 on page 2: a second reading by another model gives SKU SKU-0O2 "
        "instead of SKU-002. One of the two readings is wrong; please check this line.",
        "line": 1,
        "page": 2,
        "severity": "warning",
        "line_no": 2,
        "fields": ["line.sku"],
        "suggestions": [{"set": {"line.sku": "SKU-0O2"}, "source": "second_reading"}],
    }
    # Not an issue of the page: the page as the first reader left it.
    assert [p.issues for p in res.pages] == [[], []]
    reads = [p.second_reading for p in res.pages]
    assert [(a.issues, a.lines) for a in reads if a is not None] == [(0, 1), (1, 1)]
    assert res.notes == []


def test_issues_point_at_their_line_in_the_whole_document() -> None:
    first: dict[int, list[Answer]] = {
        1: [[line(1), line(2)]],
        2: [[line(3), line(4), line(5)]],
        3: [[line(6)]],
    }
    second: dict[int, list[Answer]] = {
        1: [[line(1), line(2, qty=3)]],
        2: [[line(3), line(4), with_sku(line(5), "SKU-500")]],
        3: [[line(6)]],
    }
    every = [line(n) for n in range(1, 7)]
    server = Server(header_for(every), first, models={OTHER: Server({}, second)})
    res = run(scan(3), "application/pdf", server, on())
    assert [(i.page, i.line, i.line_no, i.fields) for i in res.issues] == [
        (1, 1, 2, ["line.quantity", "line.amount"]),
        (2, 4, 5, ["line.sku"]),
    ]
    assert all(res.lines[i.line]["line_no"] == i.line_no for i in res.issues if i.line is not None)


def test_the_kept_attempt_is_compared_not_the_last_one() -> None:
    """Both attempts at page 1 fail their checks; the first fails less and is kept. The
    second reader reads the page as that kept attempt does: no disagreement."""
    kept = [line(1), line(2, amount=99.0)]  # one wrong line
    worse = [line(1, amount=1.0), line(2, amount=99.0)]  # two wrong lines
    every = [line(1), line(2)]
    server = Server(header_for(every), {1: [kept, worse]}, models={OTHER: Server({}, {1: [kept]})})
    res = run(scan(1), "application/pdf", server, on())
    assert len(res.pages[0].attempts) == 2 and res.lines[0]["amount"] == 20.0
    assert "second_reading" not in [i.check for i in res.issues]
    assert res.pages[0].second_reading is not None and missing(res) == []


def test_a_page_the_first_reader_could_not_read_is_not_compared() -> None:
    every = [line(1), line(2)]
    first: dict[int, list[Answer]] = {1: [[line(1)]], 2: [httpx.Response(500, text="down")]}
    server = Server(header_for(every), first, models={OTHER: Server({}, rows(2))})
    res = run(scan(2), "application/pdf", server, on())
    assert "second_reading" not in [i.check for i in res.issues]
    assert second_readings(res) == {1: None, 2: None}  # both answered
    assert missing(res) == [
        {
            "page": None,
            "kind": "second_reading_missing",
            # The page has its own issue saying it was not read: no word about its codes.
            "text": "Page 2 was read a second time, but the two readings could not be compared.",
            "source": "second_reading",
            "pages": [2],
        }
    ]


def test_unanswered_and_uncompared_pages_share_one_note() -> None:
    every = [line(p) for p in range(1, 5)]
    first: dict[int, list[Answer]] = rows(4) | {4: [httpx.Response(500, text="down")]}
    second: dict[int, list[Answer]] = rows(4) | {1: [httpx.Response(503)], 2: ["data: [DONE]\n\n"]}
    server = Server(header_for(every), first, models={OTHER: Server({}, second)})
    res = run(scan(4), "application/pdf", server, on())
    (note,) = missing(res)
    assert note["pages"] == [1, 2, 4]
    assert note["text"] == (
        "Page 1 was not read a second time: the other model did not answer, or not in time. "
        "Page 2 has no usable second reading: the other model's answer was broken or had no "
        "line items. Page 4 was read a second time, but the two readings could not be "
        "compared. Nothing else checks the item codes (SKUs) on pages 1–2."
    )


def test_a_description_joined_across_a_page_break_raises_nothing() -> None:
    """The first reader returns the end of a cut description with page 2 and the flow joins
    it to the last line of page 1; the second reader leaves that description cut."""
    first: dict[int, list[Answer]] = {
        1: [{"carry_in": "", "rows": [line(1), line(2) | {"description": "Hex bolt M8"}]}],
        2: [{"carry_in": "zinc plated", "rows": [line(3)]}],
    }
    second: dict[int, list[Answer]] = {
        1: [[line(1), line(2) | {"description": "Hex bolt M8"}]],
        2: [[line(3)]],
    }
    every = [line(1), line(2), line(3)]
    server = Server(header_for(every), first, models={OTHER: Server({}, second)})
    res = run(scan(2), "application/pdf", server, on())
    assert res.lines[1]["description"] == "Hex bolt M8 zinc plated"
    assert res.status == "ok" and res.issues == []


def test_the_issues_of_the_first_reading_keep_their_positions() -> None:
    """A scenario with retries, a cut answer and issues of every kind, read once without and
    once with a second reader that disagrees on page 2: the issues of the first reading are
    the same, in the same places; the second reading's come after them."""
    case = by_name("scan_retries")
    data, mime = case.data()

    def go(cfg: Config) -> Result:
        model = case.model()
        other = agreeing(model)
        p2 = model.pages[2][-1]
        other.pages[2] = [p2 | {"rows": [with_sku(p2["rows"][0], "SKU-0O3"), *p2["rows"][1:]]}]
        model.models = {OTHER: other}
        return run(data, mime, model, cfg)

    plain, twice = go(Config(**case.config)), go(on(Config(**case.config)))
    before = [i.to_json() for i in plain.issues]
    assert before and [i.to_json() for i in twice.issues[: len(before)]] == before
    (added,) = twice.issues[len(before) :]
    assert (added.check, added.page, added.line_no, added.fields) == (
        "second_reading",
        2,
        3,
        ["line.sku"],
    )
    assert twice.lines[added.line or 0]["sku"] == "SKU-003"
    assert (plain.status, twice.status) == ("needs_review", "needs_review")


def test_a_comparison_that_breaks_leaves_a_note(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(first: list[Line], second: list[Line], page: int) -> list[Issue]:
        raise ValueError("unexpected")

    monkeypatch.setattr(sr, "compare_page", broken)
    second: dict[int, list[Answer]] = {1: [[with_sku(line(1), "X")]], 2: [[line(2)]]}
    res = run(scan(2), "application/pdf", server_for(2, second), on())
    assert res.status == "ok" and res.issues == [] and len(res.lines) == 2
    assert [n["pages"] for n in missing(res)] == [[1, 2]]
    assert second_readings(res) == {1: None, 2: None}  # the requests are still counted
    assert res.calls == 5


def test_an_empty_second_reading_of_a_page_with_lines_is_only_missing() -> None:
    res = run(scan(1), "application/pdf", server_for(1, {1: [{"rows": []}]}), on())
    assert res.status == "ok" and res.issues == []
    assert [n["text"] for n in missing(res)] == [UNUSABLE]
    assert res.pages[0].second_reading is not None


def test_a_page_kept_from_the_second_readers_model_is_not_compared_with_it() -> None:
    """The first reader's own retry is by the same model as the second reader: two
    readings by one model are not two readers."""
    first: dict[int, list[Answer]] = {1: [[line(1, amount=1.0)]]}
    same_model = Server({}, {1: [[with_sku(line(1), "SKU-0O1")]]})
    server = Server(header_for([line(1)]), first, models={OTHER: same_model})
    res = run(scan(1), "application/pdf", server, on(Config(page_models=(FAST, OTHER))))
    assert [a.model for a in res.pages[0].attempts] == [FAST, OTHER]
    assert res.pages[0].model == OTHER and res.lines[0]["sku"] == "SKU-0O1"
    assert "second_reading" not in [i.check for i in res.issues]
    assert [n["pages"] for n in missing(res)] == [[1]]


def test_the_second_reading_is_repaired_before_it_is_compared() -> None:
    """The second reader reads the thousands separator of a quantity as a decimal point
    (6.6 for 6,600); the printed amount proves the repair, as on the first reader's side."""
    first = [line(1, qty=6600, price=0.5)]
    second: dict[int, list[Answer]] = {1: [[line(1, qty=6600, price=0.5) | {"quantity": 6.6}]]}
    server = Server(header_for(first), {1: [first]}, models={OTHER: Server({}, second)})
    res = run(scan(1), "application/pdf", server, on())
    assert res.status == "ok" and res.issues == []


def test_second_reading_issues_come_after_the_deadline_issue() -> None:
    """The header's second attempt is cut by the deadline, and the second reader disagrees
    on a SKU: its issue is the last one, after deadline_exceeded."""
    rows_1 = [line(1), line(2)]
    header = header_for(rows_1) | {"total": None}  # asks for a second attempt
    second: dict[int, list[Answer]] = {1: [[line(1), with_sku(line(2), "SKU-0O2")]]}
    server = Server(header, {1: [rows_1]}, models={OTHER: Server({}, second)})
    slow = SlowServer(server, {"header": [0.0, 30.0]})
    res, took = run_async(scan(1), "application/pdf", slow, on(Config(deadline_s=1.0)))
    assert took < 3
    assert [i.check for i in res.issues][-2:] == ["deadline_exceeded", "second_reading"]


def test_any_second_answer_keeps_the_result_strict_json() -> None:
    hostile = {
        "line_no": 1,
        "sku": ["SKU-001"],
        "quantity": float("nan"),
        "unit_price": float("inf"),
        "tax_rate": True,
        "amount": "9" * 400,
    }
    second: dict[int, list[Answer]] = {
        1: [sse('{"rows": [' + json.dumps(hostile) + "]}")],  # NaN, Infinity: JSON allows
    }
    res = run(scan(1), "application/pdf", server_for(1, second), on())
    (issue,) = res.issues
    assert issue.suggestions == []
    json.dumps(res.to_json(), allow_nan=False)
