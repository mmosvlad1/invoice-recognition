"""The overall deadline of a document: no request starts after it, a request in flight is
cut short at it, and what was read is kept. Fake OpenRouter, no network.

No test races local work: the inputs are blank drawn pages that are not measured, or
small PDFs with a text layer (local work of at most a few tens of milliseconds), and a
deadline of 0.5 s is held against requests that would take 30 s. A request that would sleep 30 s is always cut or cancelled, never waited for."""

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator

import httpx
import pytest

from invoice_recognition.recognition import pipeline as pl
from invoice_recognition.recognition.config import Config
from invoice_recognition.recognition.results import Result

from drawn import scan
from fakes import Answer, Handler, Server, SlowServer, header_for, kind_of, line
from samples import saved_layout, truth_doc

DEADLINE = 0.5
SLOW = [30.0]  # far beyond the deadline
CUT = re.compile(r"timeout after \d+s")


def cfg(concurrency: int = 12) -> Config:
    """Blank drawn pages, not measured, and the deadline."""
    return Config(scan_quality=False, deadline_s=DEADLINE, concurrency=concurrency)


def run(data: bytes, server: Handler | SlowServer, config: Config) -> tuple[Result, float]:
    """process() on a PDF; also returns how long it took."""

    async def go() -> tuple[Result, float]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
            started = time.monotonic()
            res = await pl.process(data, "application/pdf", client, config)
            return res, time.monotonic() - started

    return asyncio.run(go())


def rows_by_page(pages: int) -> dict[int, list[Answer]]:
    return {p: [[line(p)]] for p in range(1, pages + 1)}


def page_issues(res: Result, page: int, check: str) -> list[str]:
    return [i.message for i in res.issues if i.page == page and i.check == check]


def not_read(page: int) -> str:
    return f"Page {page} was not read: the time limit for the document passed."


def deadline_message(res: Result) -> str:
    [issue] = [i for i in res.issues if i.check == "deadline_exceeded"]
    assert issue.severity == "error"
    return issue.message


START = "Reading stopped at the time limit of 0.5 s for one document."


def test_a_slow_page_is_cut_and_the_others_are_kept() -> None:
    server = SlowServer(Server(header_for([line(1), line(2), line(3)]), rows_by_page(3)), {2: SLOW})
    res, took = run(scan(3), server, cfg())

    assert took < 2
    assert res.status == "failed"
    assert page_issues(res, 2, "page_read") == [not_read(2)]  # the timeout is not repeated
    assert deadline_message(res) == f"{START} Page 2 was not read."
    assert [item["sku"] for item in res.lines] == ["SKU-001", "SKU-003"]  # pages 1 and 3 read
    p2 = res.pages[1]
    assert p2.deadline_unread and not res.pages[0].deadline_unread
    assert len(p2.attempts) == 1 and server.calls[2] == 1  # cut: no second attempt
    assert p2.attempts[0].error is not None and CUT.fullmatch(p2.attempts[0].error)
    assert p2.attempts[0].cost_known is False and server.cancelled == [2]
    assert res.cost_unknown_calls == 1
    attempts = res.header_attempts + [a for p in res.pages for a in p.attempts]
    paid = [a.cost_usd for a in attempts if a.cost_known]
    assert len(paid) == 3 and res.cost_usd == pytest.approx(sum(paid)) and res.cost_usd > 0


def test_no_attempt_is_started_after_the_deadline() -> None:
    # One page request at a time. Both pages fail their checks on the first attempt, so
    # each needs a second one. Page 1's second attempt is slow and holds the document's
    # only request slot past the deadline; page 2's second attempt waits for it and is
    # never sent.
    bad: dict[int, list[Answer]] = {p: [[line(p, amount=99.0)]] for p in (1, 2)}
    server = SlowServer(Server(header_for([line(1), line(2)]), bad), {1: [0.0, *SLOW]})
    res, took = run(scan(2), server, cfg(concurrency=1))

    assert took < 2
    assert server.calls[2] == 1  # no second request for page 2
    p1, p2 = res.pages
    assert len(p2.attempts) == 1 and p2.attempts[0].error is None
    assert p2.lines and not p2.deadline_unread  # page 2 keeps its first attempt
    assert len(p1.attempts) == 2 and p1.attempts[1].error is not None  # cut, first kept
    assert p1.lines and not p1.deadline_unread
    assert not any(p.stronger_model for p in res.pages)
    assert [n["kind"] for n in res.notes if n["kind"] == "stronger_model_not_called"] == []
    assert deadline_message(res) == (
        f"{START} Every page was read; a repeated read of a page or of the header was stopped."
    )
    assert res.status == "needs_review"  # every page was read


def test_a_cut_second_header_attempt_keeps_the_first_and_does_not_fail() -> None:
    rows = [line(1), line(2)]
    first = header_for(rows) | {"total": None}  # asks for a second attempt
    server = SlowServer(Server(first, {1: [rows]}), {"header": [0.0, *SLOW]})
    res, took = run(scan(1), server, cfg())

    assert took < 2
    assert server.calls["header"] == 2 and server.cancelled == ["header"]
    assert res.header == first
    assert [a.error is None for a in res.header_attempts] == [True, False]
    assert res.lines == [item | {"page": 1} for item in rows]
    assert deadline_message(res) == (
        f"{START} Every page was read; a repeated read of a page or of the header was stopped."
    )
    assert res.status == "needs_review"


def test_a_header_never_answered_fails_the_document_and_says_so() -> None:
    rows = [line(1), line(2)]
    server = SlowServer(Server(header_for(rows), {1: [rows]}), {"header": SLOW})
    res, took = run(scan(1), server, cfg())

    assert took < 2
    assert res.lines and not res.pages[0].deadline_unread  # the page was read
    assert res.header == {} and len(res.header_attempts) == 1
    assert deadline_message(res) == (
        f"{START} The header (supplier, invoice number, totals) was not read."
    )
    assert res.status == "failed"


def test_waiting_for_a_shared_slot_ends_at_the_deadline_and_spares_the_holder() -> None:
    # Document B (no deadline) holds the only shared slot with a slow header request.
    # Document A waits for that slot, gives up at its own deadline, and B's request goes on.
    holder = SlowServer(Server(header_for([line(1)]), rows_by_page(1)), {"header": SLOW})
    waiter = SlowServer(Server(header_for([line(1), line(2)]), rows_by_page(2)))

    async def go() -> tuple[Result, float, int, list[int | str]]:
        slots = asyncio.Semaphore(1)
        async with (
            httpx.AsyncClient(transport=httpx.MockTransport(holder)) as b,
            httpx.AsyncClient(transport=httpx.MockTransport(waiter)) as a,
        ):
            held = asyncio.create_task(
                pl.process(
                    scan(1), "application/pdf", b, Config(scan_quality=False), request_slots=slots
                )
            )
            async with asyncio.timeout(2):
                while holder.in_flight == 0:
                    await asyncio.sleep(0.005)
            started = time.monotonic()
            async with asyncio.timeout(5):  # a regression fails here instead of hanging
                res = await pl.process(scan(2), "application/pdf", a, cfg(), request_slots=slots)
            took = time.monotonic() - started
            in_flight, cancelled = holder.in_flight, list(holder.cancelled)
            held.cancel()
            with pytest.raises(asyncio.CancelledError):
                await held
            return res, took, in_flight, cancelled

    res, took, in_flight, cancelled = asyncio.run(go())
    assert took < 2
    assert waiter.arrived == []  # A sent nothing
    assert in_flight == 1 and cancelled == []  # B's request was still running, untouched
    assert holder.cancelled == ["header"]  # cancelled only by the test, afterwards
    assert res.status == "failed" and res.header == {} and res.calls == 0
    assert all(p.deadline_unread for p in res.pages)
    assert deadline_message(res) == (
        f"{START} Pages 1–2 were not read. "
        "The header (supplier, invoice number, totals) was not read."
    )


class _Stalls(httpx.AsyncByteStream):
    """A whole answer with its token counts but no cost, then nothing: the stream never
    ends."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        events = [
            {"choices": [{"delta": {"content": json.dumps({"rows": [line(1)]})}}]},
            {"usage": {"prompt_tokens": 100, "completion_tokens": 50}},
        ]
        for e in events:
            yield f"data: {json.dumps(e)}\n\n".encode()
        await asyncio.sleep(30)
        yield b"data: [DONE]\n\n"


def test_a_request_cut_mid_stream_keeps_its_attempt_and_tokens() -> None:
    answers = Server(header_for([line(1)]), rows_by_page(1))

    def server(request: httpx.Request) -> httpx.Response:
        if isinstance(kind_of(request), int):
            return httpx.Response(200, stream=_Stalls())
        return answers(request)

    res, took = run(scan(1), server, cfg())

    assert took < 2
    [attempt] = res.pages[0].attempts  # the attempt is kept: the call was not cancelled
    assert attempt.error is not None and CUT.fullmatch(attempt.error)
    assert (attempt.prompt_tokens, attempt.completion_tokens) == (100, 50)
    assert attempt.cost_known is False and res.cost_unknown_calls == 1
    assert res.prompt_tokens == 200  # the header and the cut page
    assert res.status == "failed"
    # The rows of the cut answer are kept, and the page says they may be incomplete.
    assert res.lines == [line(1) | {"page": 1}]
    assert page_issues(res, 1, "page_read") == [
        (
            "Page 1 was not read to the end: the time limit for the document passed. "
            "The 1 line item read before that may be incomplete."
        )
    ]


def test_a_layout_answer_after_the_deadline_leaves_the_other_pages_unread() -> None:
    data, _, header, pages = truth_doc("m_grouped")
    answers: dict[int, list[Answer]] = {p: [rows] for p, rows in pages.items()}
    server = SlowServer(Server(header, answers, saved_layout("m_grouped")), {"layout": SLOW})
    res, took = run(data, server, Config(deadline_s=DEADLINE))

    assert took < 2
    assert dict(server.calls) == {"header": 1, "layout": 1, 1: 1}  # no fallback page asked
    assert res.layout["status"] == "failed" and len(res.layout_attempts) == 1
    assert res.layout_attempts[0].cost_known is False
    assert [p.page for p in res.pages if p.deadline_unread] == [2, 3, 4]
    assert res.pages[0].lines and not res.pages[0].deadline_unread  # the witness page
    assert deadline_message(res) == f"{START} Pages 2–4 were not read."
    assert res.status == "failed"


def test_a_witness_read_cut_by_the_deadline_leaves_the_rows_unconfirmed() -> None:
    data, truth, header, pages = truth_doc("s_classic")
    answers: dict[int, list[Answer]] = {p: [rows] for p, rows in pages.items()}
    server = SlowServer(Server(header, answers, saved_layout("s_classic")), {1: SLOW})
    res, took = run(data, server, Config(deadline_s=DEADLINE))

    assert took < 2
    assert res.method == "layout" and len(res.lines) == len(truth["lines"])  # found by code
    assert res.layout["status"] == "unverified"
    assert page_issues(res, 1, "layout_unverified") == [
        (
            "the check read of page 1 was stopped by the time limit for the document, so the "
            "rows found by code were not confirmed"
        )
    ]
    assert deadline_message(res) == (
        f"{START} Every page was read; a repeated read of a page or of the header was stopped."
    )
    assert res.status == "needs_review"


@pytest.mark.parametrize(
    ("unread", "header_read", "end"),
    [
        ([2], True, "Page 2 was not read."),
        (list(range(3, 51)), True, "Pages 3–50 were not read."),
        ([12, 2, 5, 6, 7, 8, 9], True, "Pages 2, 5–9 and 12 were not read."),
        (
            [1, 3],
            False,
            (
                "Pages 1 and 3 were not read. The header (supplier, invoice number, "
                "totals) was not read."
            ),
        ),
    ],
)
def test_deadline_text_names_what_was_not_read(
    unread: list[int], header_read: bool, end: str
) -> None:
    assert pl.deadline_text(0.5, unread, header_read) == f"{START} {end}"


def test_gate_after_the_deadline_takes_no_slot() -> None:
    async def go() -> tuple[bool, bool]:
        own, shared = asyncio.Semaphore(1), asyncio.Semaphore(1)
        gate = pl.Gate(shared, deadline=time.monotonic() - 1)
        with pytest.raises(pl.DeadlinePassed):
            await gate.enter(own)
        return gate.hit, own.locked() or shared.locked()

    assert asyncio.run(go()) == (True, False)


def test_gate_without_a_deadline_waits_and_gives_back() -> None:
    async def go() -> tuple[bool, bool, bool]:
        own, shared = asyncio.Semaphore(1), asyncio.Semaphore(1)
        gate = pl.Gate(shared)
        await gate.enter(own)
        taken = own.locked() and shared.locked()
        gate.leave(own)
        return taken, own.locked() or shared.locked(), gate.hit

    assert asyncio.run(go()) == (True, False, False)


def test_gate_gives_up_waiting_for_a_shared_slot_at_the_deadline() -> None:
    async def go() -> tuple[float, bool, bool]:
        own, shared = asyncio.Semaphore(1), asyncio.Semaphore(0)  # no shared slot free
        gate = pl.Gate(shared, deadline=time.monotonic() + 0.1)
        started = time.monotonic()
        with pytest.raises(pl.DeadlinePassed):
            await gate.enter(own)
        return time.monotonic() - started, own.locked(), gate.hit

    waited, own_locked, hit = asyncio.run(go())
    assert 0.05 < waited < 1 and not own_locked and hit


def test_gate_cancelled_while_waiting_for_a_shared_slot_gives_back_its_own() -> None:
    async def go() -> bool:
        own, shared = asyncio.Semaphore(1), asyncio.Semaphore(0)
        waiting = asyncio.create_task(pl.Gate(shared).enter(own))
        async with asyncio.timeout(2):
            while not own.locked():  # it holds its own limit and waits for the shared one
                await asyncio.sleep(0)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        return own.locked()

    assert asyncio.run(go()) is False


class _PassesWhileWaiting(pl.Gate):
    """A gate whose deadline passes right after its slots were taken."""

    def __init__(self, slots: asyncio.Semaphore) -> None:
        super().__init__(slots)
        self.asked = 0

    def passed(self) -> bool:
        self.asked += 1
        return self.asked > 1


def test_gate_whose_deadline_passes_after_taking_the_slots_gives_them_back() -> None:
    async def go() -> tuple[bool, bool]:
        own, shared = asyncio.Semaphore(1), asyncio.Semaphore(1)
        gate = _PassesWhileWaiting(shared)
        with pytest.raises(pl.DeadlinePassed):
            await gate.enter(own)
        return own.locked() or shared.locked(), gate.hit

    assert asyncio.run(go()) == (False, True)
