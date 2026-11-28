"""A backup model of another provider reads a request again when the provider of the first
model fails (Config.fallback_model), in the flow. Fake OpenRouter, no network.

The first model and the backup are fake servers of their own, told apart by the model
named in the request: each records what it got, so a test asserts on the requests that
reached the backup. Nothing is asserted by raising inside a server: the flow turns any
exception there into a failed request - which would itself send the request to the backup.
"""

import asyncio
import json
import time
from collections import Counter
from collections.abc import Callable, Coroutine, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from importlib.util import find_spec
from typing import Any

import httpx
import pytest
from test_golden import comparable, comparable_requests, compare

from invoice_recognition.api.views import job_error, result_view
from invoice_recognition.llm import PDF_NATIVE
from invoice_recognition.recognition import pipeline as pl
from invoice_recognition.recognition.config import FAST, OTHER, Config
from invoice_recognition.recognition.prompts import EXTRACT_PAGE
from invoice_recognition.recognition.results import Result
from invoice_recognition.shapes import JSONObject

from drawn import scan, scanned
from fakes import Answer, Handler, SlowServer, header_for, kind_of, line, model_of, sse
from golden_cases import CASES, GOLDEN, Case, Model, request_of, result_of, sha256
from samples import saved_layout, truth_doc

type Key = int | str

# Every test here runs on asyncio's own loop and on uvloop, which the app is served on when
# it is installed (tests/conftest.py): their timers fire early by different amounts, and
# the order in which waiting tasks run is each loop's own.
type LoopFactory = Callable[[], asyncio.AbstractEventLoop] | None
LOOPS: dict[str, LoopFactory] = {"asyncio": None}
if find_spec("uvloop") is not None:
    import uvloop

    LOOPS["uvloop"] = uvloop.new_event_loop
_loop: list[LoopFactory] = [None]


@pytest.fixture(autouse=True, params=sorted(LOOPS))
def event_loop_kind(request: pytest.FixtureRequest) -> str:
    _loop[0] = LOOPS[request.param]
    name: str = request.param
    return name


def on_loop[T](coro: Coroutine[Any, Any, T]) -> T:
    """asyncio.run() on the loop of this run of the test."""
    return asyncio.run(coro, loop_factory=_loop[0])


def run(data: bytes, mime: str, server: Handler | SlowServer, cfg: Config) -> Result:
    """process() on one file, every request answered by `server`, on this run's loop."""

    async def go() -> Result:
        async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
            return await pl.process(data, mime, client, cfg)

    return on_loop(go())


def on(cfg: Config | None = None, **change: Any) -> Config:
    """The configuration with the backup model switched on (and any other change)."""
    return replace(cfg or Config(), fallback_model=OTHER, **change)


def events(*items: JSONObject) -> str:
    """A raw stream of these events, ended as OpenRouter ends it."""
    return "".join(f"data: {json.dumps(e)}\n\n" for e in items) + "data: [DONE]\n\n"


def content(text: str, finish: str | None = None) -> JSONObject:
    choice: JSONObject = {"delta": {"content": text}}
    if finish is not None:
        choice["finish_reason"] = finish
    return {"choices": [choice]}


def finish(reason: str) -> JSONObject:
    return {"choices": [{"delta": {}, "finish_reason": reason}]}


def respond(answer: Answer) -> httpx.Response:
    """As fakes.Server answers: an exception is raised, a response returned, a string sent
    as the stream, a list of rows or an object as a whole answer."""
    if isinstance(answer, BaseException):
        raise answer
    if isinstance(answer, httpx.Response):
        return answer
    if isinstance(answer, str):
        return httpx.Response(200, text=answer)
    payload = {"rows": answer} if isinstance(answer, list) else answer
    return httpx.Response(200, text=sse(json.dumps(payload)))


class Script:
    """A fake model: answers each kind of request (fakes.kind_of) with its answers in
    turn, the last one repeating, and keeps every body it got, in order."""

    def __init__(self, answers: Mapping[Key, list[Answer]]) -> None:
        self.answers = dict(answers)
        self.calls: Counter[Key] = Counter()
        self.bodies: list[JSONObject] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        kind = kind_of(request)
        answers = self.answers[kind]
        answer = answers[min(self.calls[kind], len(answers) - 1)]
        self.calls[kind] += 1
        return respond(answer)


def route(first: Handler, backup: Handler) -> Handler:
    """The requests to the backup model to `backup`, every other one to `first`."""
    return lambda request: (backup if model_of(request) == OTHER else first)(request)


def wire(first: Handler, backup: Handler) -> SlowServer:
    """route() answering each request after one turn of the event loop, as a connection
    would: every request a document starts with is sent before the first answer comes."""
    return SlowServer(route(first, backup))


def run_with(
    data: bytes,
    handler: Callable[[httpx.Request], Coroutine[None, None, httpx.Response]],
    cfg: Config,
) -> Result:
    """process() on a PDF, every request answered by an asynchronous handler."""

    async def go() -> Result:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await pl.process(data, "application/pdf", client, cfg)

    return on_loop(go())


def key_of(body: JSONObject) -> Key:
    return kind_of(httpx.Request("POST", "http://x", content=json.dumps(body).encode()))


def as_backup(body: JSONObject) -> JSONObject:
    """The body the backup must get for this request of the first model: the same but the
    model, its token limit and, for a PDF, the plugin that sends the file itself."""
    assert "plugins" not in body
    files = [p for p in body["messages"][0]["content"] if p["type"] == "file"]
    return (
        body | {"model": OTHER, "max_tokens": 24_000} | ({"plugins": PDF_NATIVE} if files else {})
    )


def note_text(parts: str) -> str:
    return (
        f"The usual reading model was not available, so a backup model, {OTHER}, read {parts}. "
        "What it read went through the same checks."
    )


def fallback_notes(res: Result) -> list[JSONObject]:
    return [n for n in res.notes if n["kind"] == "fallback_model"]


def same_result(a: Result, b: Result) -> None:
    assert result_of(a.to_json()) == result_of(b.to_json())


# --------------------------------------------------------------------------- #
# What fails at the provider, and what does not
# --------------------------------------------------------------------------- #

ROW_START = '{"rows": [' + json.dumps(line(1)) + ", "

TRIGGERS: dict[str, Answer] = {
    **{
        f"HTTP {s}": httpx.Response(s, text="upstream error")
        for s in (408, 429, 500, 502, 503, 504)
    },
    "an error event after a row": events(
        content(ROW_START), {"error": {"code": 502, "message": "provider went away"}}
    ),
    "an error event without a number": events({"error": {"code": "server_error"}}),
    "the stream's own error marker": events(content(ROW_START), finish("error")),
    "a timeout": httpx.ReadTimeout("read timed out"),
    "a refused connection": httpx.ConnectError("connection refused"),
    "an exception": RuntimeError("stream blew up"),
    "nothing at all": "data: [DONE]\n\n",
    "only whitespace": events(content("  \n", "stop")),
    "an answer cut short on the way": events(content(ROW_START)),
    "nothing readable": "data: {broken\n\ndata: [DONE]\n\n",
}

NON_TRIGGERS: dict[str, Answer] = {
    **{f"HTTP {s}": httpx.Response(s, text="refused") for s in (400, 401, 402, 403, 404, 413, 422)},
    "an error event about the request": events({"error": {"code": 400, "message": "bad"}}),
    "nothing, at the token limit": events(finish("length")),
    "nothing, filtered": events(finish("content_filter")),
    "broken JSON": events(content(ROW_START, "stop")),
    "an answer cut at the token limit": events(content(ROW_START, "length")),
}


# --------------------------------------------------------------------------- #
# The kinds of request, each with the answers of a model that reads it right
# --------------------------------------------------------------------------- #


@dataclass
class Kind:
    name: str
    data: Callable[[], tuple[bytes, str]]
    good: dict[Key, list[Answer]]
    target: Key  # the request that fails at the first model
    cfg: Config = field(default_factory=Config)
    # What the result keeps from the backup's answer to the target.
    kept: Callable[[Result], None] = lambda res: None

    def with_first(self, answer: Answer) -> dict[Key, list[Answer]]:
        """The good answers, the target's first one replaced."""
        return self.good | {self.target: [answer, *self.good[self.target]]}


def page_kept(page: int) -> Callable[[Result], None]:
    def check(res: Result) -> None:
        pr = res.pages[page - 1]
        assert (pr.model, pr.by_backup, pr.method) == (OTHER, True, "model")
        assert [a.model for a in pr.attempts] == [FAST, OTHER]
        assert pr.attempts[1].error is None
        assert [i.check for i in pr.issues] == []
        [note] = fallback_notes(res)
        assert note == {
            "page": None,
            "kind": "fallback_model",
            "text": note_text(f"page {page}"),
            "source": "fallback",
            "model": OTHER,
            "pages": [page],
            "header": False,
            "layout": False,
        }

    return check


def witness_kept(res: Result) -> None:
    """The witness read by the backup confirms the rows found by code, which are kept: the
    backup's rows are not, so no note."""
    assert res.layout["status"] == "confirmed"
    pr = res.pages[0]
    assert (pr.method, pr.by_backup) == ("layout", False)
    assert [a.model for a in pr.attempts] == [FAST, OTHER]
    assert fallback_notes(res) == []


def scan_kind(pages: int, target: int) -> Kind:
    every = [line(p) for p in range(1, pages + 1)]
    good: dict[Key, list[Answer]] = {"header": [header_for(every)]}
    good |= {p: [[line(p)]] for p in range(1, pages + 1)}
    name = "a page of a scan" + (" (with the carry prompt)" if pages > 1 else "")
    return Kind(
        name, lambda: (scan(pages), "application/pdf"), good, target, kept=page_kept(target)
    )


def digital(
    name: str, **change: Any
) -> tuple[Callable[[], tuple[bytes, str]], dict[Key, list[Answer]]]:
    data, _, header, pages = truth_doc(name)
    good: dict[Key, list[Answer]] = {"header": [header], "layout": [saved_layout(name)]}
    good |= {p: [rows] for p, rows in pages.items()}
    return (lambda: (data, "application/pdf")), good


PAGE_KINDS = [
    scan_kind(2, 2),
    scan_kind(1, 1),
    Kind(
        "a page of a PDF with a text layer",
        *digital("s_classic"),
        1,
        Config(use_layout=False),
        page_kept(1),
    ),
    Kind("the witness page", *digital("s_classic"), 1, kept=witness_kept),
]


def header_kept(res: Result) -> None:
    assert [a.model for a in res.header_attempts] == [FAST, OTHER]
    assert res.header and res.status == "ok"
    [note] = fallback_notes(res)
    assert (note["text"], note["pages"], note["header"], note["layout"]) == (
        note_text("the header"),
        [],
        True,
        False,
    )


def layout_kept(res: Result) -> None:
    assert [a.model for a in res.layout_attempts] == [FAST, OTHER]
    assert res.layout["status"] == "confirmed" and res.method == "layout"
    [note] = fallback_notes(res)
    assert (note["text"], note["pages"], note["header"], note["layout"]) == (
        note_text("the layout of the item table"),
        [],
        False,
        True,
    )


KINDS = [
    *PAGE_KINDS,
    replace(scan_kind(2, 1), name="the header of a scan", target="header", kept=header_kept),
    Kind(
        "the header of a PDF with a text layer", *digital("s_classic"), "header", kept=header_kept
    ),
    Kind("the layout of the item table", *digital("s_classic"), "layout", kept=layout_kept),
]


def read(
    kind: Kind,
    first_answer: Answer,
    cfg: Config | None = None,
    backup_answers: Mapping[Key, list[Answer]] | None = None,
) -> tuple[Result, Script, Script]:
    first = Script(kind.with_first(first_answer))
    backup = Script(backup_answers if backup_answers is not None else kind.good)
    data, mime = kind.data()
    res = run(data, mime, wire(first, backup), cfg or on(kind.cfg))
    return res, first, backup


def check_fell_back(kind: Kind, res: Result, first: Script, backup: Script) -> None:
    [sent] = backup.bodies
    assert key_of(sent) == kind.target
    original = next(b for b in first.bodies if key_of(b) == kind.target)
    assert sent == as_backup(original)
    kind.kept(res)


def check_no_fallback(kind: Kind, answer: Answer) -> None:
    """Nothing reaches the backup, and the result and the requests are those of the flow
    without one."""
    res, first, backup = read(kind, answer)
    assert backup.bodies == []
    plain, first_plain, _ = read(kind, answer, kind.cfg)
    assert first.bodies == first_plain.bodies
    same_result(res, plain)


@pytest.mark.parametrize("answer", TRIGGERS.values(), ids=TRIGGERS.keys())
@pytest.mark.parametrize("kind", KINDS, ids=[k.name for k in KINDS])
def test_a_request_whose_provider_failed_goes_to_the_backup(kind: Kind, answer: Answer) -> None:
    check_fell_back(kind, *read(kind, answer))


@pytest.mark.parametrize("answer", NON_TRIGGERS.values(), ids=NON_TRIGGERS.keys())
@pytest.mark.parametrize("kind", KINDS, ids=[k.name for k in KINDS])
def test_a_request_that_failed_for_any_other_reason_does_not(kind: Kind, answer: Answer) -> None:
    check_no_fallback(kind, answer)


def whole_without_finish(kind: Kind) -> str:
    [answer] = kind.good[kind.target]
    payload = {"rows": answer} if isinstance(answer, list) else answer
    return events(content(json.dumps(payload)))


@pytest.mark.parametrize("kind", KINDS, ids=[k.name for k in KINDS])
def test_a_whole_answer_without_a_finish_reason_is_kept(kind: Kind) -> None:
    check_no_fallback(kind, whole_without_finish(kind))


@pytest.mark.parametrize("kind", PAGE_KINDS, ids=[k.name for k in PAGE_KINDS])
def test_a_page_that_fails_its_checks_is_read_again_by_its_own_model(kind: Kind) -> None:
    check_no_fallback(kind, [line(1, amount=99.0)])


def test_a_page_slower_than_its_timeout_is_read_by_the_backup() -> None:
    kind = scan_kind(1, 1)
    first, backup = Script(kind.good), Script(kind.good)
    slow = SlowServer(route(first, backup), {1: [5.0, 0.0]}, models={OTHER: {}})
    res = run(scan(1), "application/pdf", slow, on(page_timeout_s=0.2))
    assert [b["model"] for b in backup.bodies] == [OTHER]
    assert res.pages[0].attempts[0].error == "timeout after 0s"
    page_kept(1)(res)


def test_a_page_cut_by_the_deadline_is_not_sent_to_the_backup() -> None:
    kind = scan_kind(1, 1)
    first, backup = Script(kind.good), Script(kind.good)
    slow = SlowServer(route(first, backup), {1: [5.0]}, models={OTHER: {}})
    res = run(scan(1), "application/pdf", slow, on(deadline_s=0.5))
    assert backup.bodies == []
    assert "deadline_exceeded" in [i.check for i in res.issues]
    assert res.pages[0].deadline_unread


def test_a_request_cut_by_the_deadline_is_not_a_failure_of_its_provider() -> None:
    """Its timeout was the time left until the deadline, and it ran out: the call says it
    was cut, and that is ours, not the provider's - nothing goes to the backup and the
    model is not marked down, even with a clock that does not yet show the deadline as
    passed (a timer that fired early)."""
    asked: list[str] = []
    never = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        asked.append(model_of(request))
        await never.wait()  # no answer: the call gives up at the deadline
        return DOWN

    class LaggingGate(pl.Gate):
        def passed(self) -> bool:  # the clock 0.3 s behind the loop's timers
            late = self.deadline is not None and time.monotonic() - 0.3 >= self.deadline
            return self.hit or late

    async def go() -> tuple[list[pl.Sent], pl.Gate]:
        gate = LaggingGate(deadline=time.monotonic() + 0.2)
        backup = pl.Backup(OTHER, 24_000, 90.0)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            sent = await pl._ask(client, FAST, "p", None, 100, 90.0, gate=gate, backup=backup)
        return sent, gate

    sent, gate = on_loop(go())
    assert [(s.cut, s.backup, s.res.failure) for s in sent] == [(True, False, "timeout")]
    assert asked == [FAST] and gate.hit and gate.down == set()


# --------------------------------------------------------------------------- #
# Attempts at a page: how the backup's answers compose with the next attempt
# --------------------------------------------------------------------------- #

DOWN = httpx.Response(503, text="no provider")


def one_page(first: list[Answer], backup: list[Answer]) -> tuple[Result, Script, Script]:
    kind = scan_kind(1, 1)
    first_model = Script(kind.good | {1: first})
    backup_model = Script(kind.good | {1: backup})
    res = run(scan(1), "application/pdf", wire(first_model, backup_model), on())
    return res, first_model, backup_model


def test_a_backup_answer_that_passes_ends_the_page() -> None:
    res, first, backup = one_page([DOWN], [[line(1)]])
    assert (first.calls[1], backup.calls[1]) == (1, 1)
    assert res.status == "ok" and res.lines == [line(1) | {"page": 1}]


def test_a_backup_answer_with_issues_is_read_again_by_the_backup() -> None:
    """The first model is down for the document: the next attempt goes straight to the
    backup, and the better answer is kept."""
    res, first, backup = one_page([DOWN], [[line(1, amount=99.0)], [line(1)]])
    assert (first.calls[1], backup.calls[1]) == (1, 2)
    pr = res.pages[0]
    assert [a.model for a in pr.attempts] == [FAST, OTHER, OTHER]
    assert pr.model == OTHER and pr.by_backup and res.status == "ok"


def test_answers_that_still_fail_name_every_read() -> None:
    res, _, _ = one_page([""], [[line(1, amount=99.0)]])
    pr = res.pages[0]
    assert [a.model for a in pr.attempts] == [FAST, OTHER, OTHER]
    [note] = [n for n in res.notes if n["kind"] == "stronger_model_not_called"]
    assert note["text"].startswith("Page 1 still has problems after 3 reads.")
    assert [n["pages"] for n in fallback_notes(res)] == [[1]]


def test_both_providers_down_send_each_request_at_most_once_more() -> None:
    """No ping-pong: the first model is asked once, then only the backup."""
    res, first, backup = one_page([DOWN], [DOWN])
    assert (first.calls[1], backup.calls[1]) == (1, 2)
    assert [a.error for a in res.pages[0].attempts] == ["HTTP 503: no provider"] * 3
    assert res.status == "failed" and fallback_notes(res) == []
    assert not res.pages[0].stronger_model  # no request got an answer


def test_rows_saved_from_a_broken_answer_beat_a_backup_that_failed() -> None:
    rows = [line(1), line(2), line(3)]
    broken = events(
        content('{"rows": [' + ", ".join(json.dumps(r) for r in rows) + ", "),
        {"error": {"code": 502, "message": "provider went away"}},
    )
    res, _, _ = one_page([broken], [DOWN])
    pr = res.pages[0]
    assert len(pr.lines) == 3 and (pr.model, pr.by_backup) == (FAST, False)
    assert fallback_notes(res) == []


def test_a_failed_backup_answer_kept_for_its_rows_is_not_read_by_the_backup() -> None:
    """The backup's answer broke after two rows: they beat the first model's nothing and
    are kept, but the backup's reading failed - the page has its page_read issue, and
    nothing says the backup read it."""
    rows = [line(1), line(2)]
    broken = events(
        content('{"rows": [' + ", ".join(json.dumps(r) for r in rows) + ", "),
        {"error": {"code": 502, "message": "provider went away"}},
    )
    res, _, _ = one_page([DOWN], [broken, DOWN])
    pr = res.pages[0]
    assert len(pr.lines) == 2 and (pr.model, pr.by_backup) == (OTHER, False)
    assert "page_read" in [i.check for i in pr.issues]
    assert fallback_notes(res) == []


def test_a_page_the_backup_answered_but_the_first_models_answer_won_is_not_named() -> None:
    """The first model's stream gives the whole page, then ends with its own error marker:
    its provider failed, so the backup is asked too. The backup's answer has issues; the
    first model's clean answer is kept, and the note does not claim the page."""
    marked = events(content(json.dumps({"rows": [line(1)]}), "error"))
    res, first, backup = one_page([marked], [[line(1, amount=99.0)]])
    assert (first.calls[1], backup.calls[1]) == (1, 1)
    pr = res.pages[0]
    assert [a.model for a in pr.attempts] == [FAST, OTHER]
    assert (pr.model, pr.by_backup) == (FAST, False)
    assert res.status == "ok" and fallback_notes(res) == []


def test_a_page_the_backup_answered_but_another_reading_won_is_not_named() -> None:
    """The backup's answer has issues; the next attempt, by a second model the
    configuration names, reads the page cleanly and is kept: the note does not claim the
    page for the backup."""
    kind = scan_kind(2, 1)
    first = Script(kind.with_first(DOWN))
    backup = Script(kind.good | {1: [[line(1, amount=99.0)]]})
    cfg = on(page_models=(FAST, "vendor/second"))
    res = run(scan(2), "application/pdf", wire(first, backup), cfg)
    pr = res.pages[0]
    assert [a.model for a in pr.attempts] == [FAST, OTHER, "vendor/second"]
    assert (pr.model, pr.by_backup) == ("vendor/second", False)
    assert res.status == "ok" and fallback_notes(res) == []


def test_a_header_from_the_backup_without_a_total_is_read_again() -> None:
    """Each answer in turn, as before; the kept header is the best one read."""
    kind = replace(scan_kind(1, 1), target="header")
    good = kind.good["header"][0]
    assert isinstance(good, dict)
    res, first, backup = read(
        kind, DOWN, backup_answers=kind.good | {"header": [good | {"total": None}, good]}
    )
    assert (first.calls["header"], backup.calls["header"]) == (1, 2)
    assert [a.model for a in res.header_attempts] == [FAST, OTHER, OTHER]
    assert res.header == good and res.status == "ok"


def test_a_header_that_both_providers_fail_to_read_again_keeps_the_first_answer() -> None:
    """The first answer lacks the total; the second iteration fails at the first model's
    provider and at the backup's: the partial header read first is kept, not thrown away,
    and nothing says the backup read it."""
    kind = replace(scan_kind(1, 1), target="header")
    good = kind.good["header"][0]
    assert isinstance(good, dict)
    partial = good | {"total": None}
    first = Script(kind.good | {"header": [partial, DOWN]})
    backup = Script(kind.good | {"header": [DOWN]})
    res = run(scan(1), "application/pdf", wire(first, backup), on())
    assert [a.model for a in res.header_attempts] == [FAST, FAST, OTHER]
    assert res.header == partial and fallback_notes(res) == []


def test_a_header_kept_from_a_backup_answer_that_failed_is_not_the_backups() -> None:
    """The backup's stream broke after the whole header: kept, since nothing else was
    read, but the backup's reading failed - no note says it read the header."""
    kind = replace(scan_kind(1, 1), target="header")
    good = kind.good["header"][0]
    broken = events(content(json.dumps(good)), {"error": {"code": 502, "message": "gone"}})
    res, _, _ = read(kind, DOWN, backup_answers=kind.good | {"header": [broken]})
    assert res.header == good and [a.model for a in res.header_attempts] == [FAST, OTHER]
    assert fallback_notes(res) == []


def test_a_whole_header_is_kept_whatever_the_backup_answers_after_it() -> None:
    """The first model's stream gives the whole header, then ends with its own error
    marker: its provider failed, so the backup is asked too and answers a header without
    the total. The whole header read first is kept and not asked again, and nothing says
    the backup read it."""
    kind = replace(scan_kind(1, 1), target="header")
    good = kind.good["header"][0]
    assert isinstance(good, dict)
    marked = events(content(json.dumps(good), "error"))
    res, first, backup = read(
        kind, marked, backup_answers=kind.good | {"header": [good | {"total": None}]}
    )
    assert (first.calls["header"], backup.calls["header"]) == (1, 1)
    assert [a.model for a in res.header_attempts] == [FAST, OTHER]
    assert res.header == good and res.header["total"] is not None and res.status == "ok"
    assert not any(n["header"] for n in fallback_notes(res))


def test_a_layout_neither_model_can_give_fails_with_the_backups_reason() -> None:
    kind = Kind("layout", *digital("s_classic"), "layout")
    res, _, _ = read(kind, DOWN, backup_answers=kind.good | {"layout": [sse("{oops")]})
    assert [a.model for a in res.layout_attempts] == [FAST, OTHER]
    assert [a.issues for a in res.layout_attempts] == [1, 1]
    assert res.layout["status"] == "failed"
    assert res.layout["reason"] == "answer is not valid JSON (finish=stop)"
    # The one page is the witness, read by the first model while the layout was asked.
    assert res.method == "model" and fallback_notes(res) == []


def test_a_layout_answer_the_backup_failed_to_give_is_not_the_backups() -> None:
    """ask_layout itself: neither answer can be used, so the answer is no one's."""
    backup_answer = httpx.Response(200, text=sse("{oops"))

    async def go() -> tuple[JSONObject | None, str | None, bool]:
        handler = route(lambda _: DOWN, lambda _: backup_answer)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            backup = pl.Backup(OTHER, 24_000, 90.0)
            answer, _, why, by_backup = await pl.ask_layout(client, "p", Config(), backup=backup)
        return answer, why, by_backup

    assert on_loop(go()) == (None, "answer is not valid JSON (finish=stop)", False)


def test_one_note_names_everything_the_backup_read() -> None:
    every = [line(p) for p in range(1, 6)]
    good: dict[Key, list[Answer]] = {"header": [header_for(every)]}
    good |= {p: [[line(p)]] for p in range(1, 6)}
    first = Script(good | {"header": [DOWN], 2: [DOWN], 4: [DOWN], 5: [DOWN]})
    backup = Script(good)
    res = run(scan(5), "application/pdf", wire(first, backup), on())
    [note] = fallback_notes(res)
    assert note["text"] == note_text("the header and pages 2 and 4–5")
    assert (note["pages"], note["header"]) == ([2, 4, 5], True)
    assert res.notes[-1] is note


def test_both_providers_down_bound_the_requests_and_claim_nothing() -> None:
    """A three-page scan, both providers answering 503 at once: each request goes to the
    first model once, then to the backup alone - the header and each page at most three
    requests - and since the backup read nothing, no note says it did."""
    good = scan_kind(3, 1).good
    first, backup = Script({k: [DOWN] for k in good}), Script({k: [DOWN] for k in good})
    res = run(scan(3), "application/pdf", wire(first, backup), on())
    assert first.calls == Counter({"header": 1, 1: 1, 2: 1, 3: 1})
    assert backup.calls == Counter({"header": 2, 1: 2, 2: 2, 3: 2})
    assert res.status == "failed" and res.header == {} and fallback_notes(res) == []
    error = job_error("pdf", res)
    assert error is not None and error.code == "model_unavailable"


def test_both_providers_down_ask_the_layout_twice() -> None:
    kind = Kind("layout", *digital("s_classic"), "layout")
    res, first, backup = read(kind, DOWN, backup_answers=kind.good | {"layout": [DOWN]})
    assert (first.calls["layout"], backup.calls["layout"]) == (1, 1)
    assert res.layout["status"] == "failed" and fallback_notes(res) == []


# --------------------------------------------------------------------------- #
# A model that failed is down for the rest of the document, and only for it
# --------------------------------------------------------------------------- #


def test_after_the_first_wave_a_document_asks_the_backup_alone() -> None:
    """Six pages sent together: each first request fails once, then every next attempt
    goes straight to the backup."""
    good = scan_kind(6, 1).good
    failing: dict[Key, list[Answer]] = {p: [DOWN] for p in range(1, 7)}
    first = Script(failing | {"header": good["header"]})
    backup = Script(good | {p: [[line(p, amount=99.0)], [line(p)]] for p in range(1, 7)})
    slow = SlowServer(route(first, backup), models={OTHER: {}})
    res = run(scan(6), "application/pdf", slow, on())
    assert slow.calls == Counter({"header": 1} | {p: 1 for p in range(1, 7)})
    assert backup.calls == Counter({p: 2 for p in range(1, 7)})
    assert res.lines == [line(p) | {"page": p} for p in range(1, 7)]
    assert [n["pages"] for n in fallback_notes(res)] == [[1, 2, 3, 4, 5, 6]]


def test_the_next_document_starts_with_its_model_up() -> None:
    kind = scan_kind(2, 1)
    failed_once = Script(kind.good | {1: [DOWN, [line(1)]]})
    backup = Script(kind.good)

    async def go() -> list[Result]:
        handler = wire(failed_once, backup)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return [await pl.process(scan(2), "application/pdf", client, on()) for _ in range(2)]

    _, second_doc = on_loop(go())
    assert [key_of(b) for b in backup.bodies] == [1]  # the first document's page 1 only
    assert fallback_notes(second_doc) == [] and second_doc.status == "ok"
    assert [p.model for p in second_doc.pages] == [FAST, FAST]


def test_documents_read_together_do_not_share_a_model_that_is_down() -> None:
    """One document's first model fails while the other is being read: the other's next
    attempts still go to its own model. The second document's first answers wait until
    the first document has sent its backup request, so its retries come after it."""
    a_fell_back = asyncio.Event()
    good_a = scan_kind(2, 1).good
    good_b = scan_kind(3, 1).good
    backup = Script(good_a)
    b_first = Script(good_b | {p: [[line(p, amount=99.0)], [line(p)]] for p in range(1, 4)})

    async def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["messages"][0]["content"][0]["text"]
        if model_of(request) == OTHER:
            a_fell_back.set()
            return backup(request)
        if "of 2" in prompt:  # the first document's pages
            return DOWN
        if "of 3" in prompt and b_first.calls[kind_of(request)] == 0:
            await a_fell_back.wait()
        return b_first(request)

    async def go() -> tuple[Result, Result]:
        slots = asyncio.Semaphore(24)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            a, b = await asyncio.gather(
                pl.process(scan(2), "application/pdf", client, on(), request_slots=slots),
                pl.process(scan(3), "application/pdf", client, on(), request_slots=slots),
            )
        return a, b

    a, b = on_loop(go())
    assert [n["pages"] for n in fallback_notes(a)] == [[1, 2]]
    assert sorted(key_of(x) for x in backup.bodies) == [1, 2]  # a's pages
    # Both headers are answered here (their prompt does not name the document).
    assert b_first.calls == Counter({"header": 2, 1: 2, 2: 2, 3: 2})
    assert b.status == "ok" and fallback_notes(b) == []


# --------------------------------------------------------------------------- #
# Pages of a scan with a quality warning never go to the backup
# --------------------------------------------------------------------------- #


def quality_scan() -> tuple[bytes, dict[Key, list[Answer]]]:
    """Pages: good, blurred (not sent), blank (not sent), soft (a warning)."""
    p1, p4 = [line(1), line(2)], [line(3)]
    good: dict[Key, list[Answer]] = {"header": [header_for(p1 + p4)], 1: [p1], 4: [p4]}
    return scanned("good", "blurred", "blank", "soft"), good


def test_a_page_with_a_quality_warning_stays_with_its_own_model() -> None:
    data, good = quality_scan()
    first = Script({k: [DOWN] for k in good})
    backup = Script(good)
    res = run(data, "application/pdf", wire(first, backup), on())
    assert sorted(key_of(b) for b in backup.bodies if key_of(b) != "header") == [1]
    assert first.calls[4] == 2  # both attempts, by its own model although it is down
    by_page = {p.page: p for p in res.pages}
    assert by_page[1].by_backup and by_page[1].lines
    assert [i.check for i in by_page[4].issues] == ["page_read", "scan_blurred"]
    assert [n["pages"] for n in fallback_notes(res)] == [[1]]


def test_the_backup_reads_exactly_the_pages_a_second_reader_reads() -> None:
    data, good = quality_scan()
    first = Script({k: [DOWN] for k in good})
    backup_and_second = Script(good)
    cfg = on(second_reading_model=OTHER)
    run(data, "application/pdf", wire(first, backup_and_second), cfg)
    bodies = backup_and_second.bodies
    plain = {key_of(b) for b in bodies if _plain_prompt(b)}
    other = {key_of(b) for b in bodies if not _plain_prompt(b) and key_of(b) != "header"}
    assert plain == other == {1}


def test_a_failure_on_a_page_with_a_warning_marks_the_model_down() -> None:
    """Page 4 (a warning) fails at its provider; page 1's first answer (with issues) comes
    only once page 4 is being read again, so the model is down by then: page 1's retry
    goes straight to the backup."""
    data, good = quality_scan()
    first = Script(good | {1: [[line(1), line(2, amount=99.0)]], 4: [DOWN, good[4][0]]})
    backup = Script(good)
    page_4_again = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if model_of(request) != OTHER and kind_of(request) == 4 and first.calls[4] == 1:
            page_4_again.set()
        if model_of(request) != OTHER and kind_of(request) == 1 and first.calls[1] == 0:
            await page_4_again.wait()
        return route(first, backup)(request)

    res = run_with(data, handler, on())
    assert (first.calls[1], first.calls[4]) == (1, 2)
    assert [key_of(b) for b in backup.bodies if key_of(b) != "header"] == [1]
    assert res.pages[0].by_backup


def _plain_prompt(body: JSONObject) -> bool:
    text: str = body["messages"][0]["content"][0]["text"]
    return any(text == EXTRACT_PAGE.format(page=p, total=4) for p in range(1, 5))


# --------------------------------------------------------------------------- #
# The deadline, the backup's own timeout and the time it needs
# --------------------------------------------------------------------------- #


def test_a_backup_slower_than_the_time_left_is_cut_at_the_deadline() -> None:
    kind = scan_kind(1, 1)
    first, backup = Script(kind.with_first(DOWN)), Script(kind.good)
    slow = SlowServer(route(first, backup), models={OTHER: {1: [5.0]}})
    started = time.monotonic()
    res = run(scan(1), "application/pdf", slow, on(deadline_s=0.5))
    assert time.monotonic() - started < 2
    [primary, cut] = res.pages[0].attempts
    assert (primary.model, cut.model, cut.cost_known) == (FAST, OTHER, False)
    assert "deadline_exceeded" in [i.check for i in res.issues] and res.status == "failed"


def test_the_backup_has_its_own_timeout() -> None:
    kind = scan_kind(1, 1)
    first, backup = Script(kind.with_first(DOWN)), Script(kind.good)
    slow = SlowServer(route(first, backup), models={OTHER: {1: [5.0, 0.0]}})
    res = run(scan(1), "application/pdf", slow, on(page_timeout_s=60, fallback_timeout_s=0.2))
    assert [a.error for a in res.pages[0].attempts] == [
        "HTTP 503: no provider",
        "timeout after 0s",  # the backup's 0.2 s, not the page's 60 s
        None,  # the next attempt, straight to the backup
    ]
    assert res.pages[0].by_backup and "deadline_exceeded" not in [i.check for i in res.issues]


def test_a_backup_that_waits_for_a_slot_past_the_deadline_is_not_sent() -> None:
    """The one shared slot goes to another document's request the moment the failed one
    gives it back; the deadline passes while the backup waits behind it."""
    in_flight, answer_now = asyncio.Event(), asyncio.Event()
    kind = scan_kind(1, 1)
    backup = Script(kind.good)

    async def handler(request: httpx.Request) -> httpx.Response:
        if model_of(request) == OTHER:
            return backup(request)
        if kind_of(request) == "header":
            return respond(kind.good["header"][0])
        in_flight.set()
        await answer_now.wait()
        return DOWN

    async def go() -> tuple[Result, float, pl._Waiting, asyncio.Semaphore]:
        slots = asyncio.Semaphore(1)
        cfg = on(deadline_s=0.5)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            started = time.monotonic()
            reading = asyncio.create_task(
                pl.process(scan(1), "application/pdf", client, cfg, request_slots=slots)
            )
            await in_flight.wait()
            other = asyncio.create_task(slots.acquire())  # queued for the slot
            await asyncio.sleep(0)
            answer_now.set()
            res = await reading
            took = time.monotonic() - started
            await other
            slots.release()
        return res, took, pl._waiting(slots), slots

    res, took, waiting, slots = on_loop(go())
    assert took < 1.0
    assert backup.bodies == []
    assert [a.model for a in res.pages[0].attempts] == [FAST]
    assert "deadline_exceeded" in [i.check for i in res.issues]
    assert (waiting.ordinary, waiting.held, waiting.starting) == (0, 0, 0)
    assert slots._value == 1


@pytest.mark.parametrize("deadline_s", [10.0, None])
def test_a_model_that_is_down_is_not_asked_again_with_or_without_a_deadline(
    deadline_s: float | None,
) -> None:
    """With a deadline or without one, the page's next attempt goes to the backup alone: it
    is bounded by the deadline like any request, never sent back to the model that is down."""
    kind = scan_kind(1, 1)
    first = Script(kind.with_first(DOWN))
    backup = Script(kind.good | {1: [[line(1, amount=99.0)], [line(1)]]})
    res = run(scan(1), "application/pdf", wire(first, backup), on(deadline_s=deadline_s))
    assert (first.calls[1], backup.calls[1]) == (1, 2)
    assert res.pages[0].by_backup and res.status == "ok"


# --------------------------------------------------------------------------- #
# With the second reading: a page read by the backup is not "confirmed" by the backup
# --------------------------------------------------------------------------- #


def test_a_page_read_by_the_backup_is_not_compared_with_the_same_model() -> None:
    kind = scan_kind(2, 1)
    first = Script(kind.with_first(DOWN))
    other = Script(kind.good)
    res = run(scan(2), "application/pdf", wire(first, other), on(second_reading_model=OTHER))
    assert res.pages[0].model == OTHER and res.pages[1].model == FAST
    assert [p.second_reading is not None for p in res.pages] == [True, True]
    [missing] = [n for n in res.notes if n["kind"] == "second_reading_missing"]
    assert missing["pages"] == [1]
    assert missing["text"].startswith(
        "Page 1 was read a second time, but the two readings could not be compared."
    )
    assert [n["kind"] for n in res.notes][-1] == "fallback_model"


def test_a_second_reading_that_fails_is_never_read_again() -> None:
    kind = scan_kind(2, 1)
    first = Script(kind.good)
    other = Script({k: [DOWN] for k in kind.good})
    res = run(scan(2), "application/pdf", wire(first, other), on(second_reading_model=OTHER))
    assert sorted(key_of(b) for b in other.bodies) == [1, 2]  # one second reading each
    assert all(_plain_prompt_of(b, 2) for b in other.bodies)
    assert first.calls == Counter({"header": 1, 1: 1, 2: 1})
    assert res.status == "ok" and fallback_notes(res) == []


def _plain_prompt_of(body: JSONObject, total: int) -> bool:
    text: str = body["messages"][0]["content"][0]["text"]
    return any(text == EXTRACT_PAGE.format(page=p, total=total) for p in range(1, total + 1))


def arrivals(
    kind: Kind, first: Script, other: Script, total: int
) -> tuple[Callable[[httpx.Request], Coroutine[None, None, httpx.Response]], list[str]]:
    """A handler that notes who each request is for, in the order they arrive: "first <kind>",
    "backup <kind>" or "second <kind>" (a second reading: the plain page prompt), and
    answers after one turn of the event loop."""
    arrived: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if model_of(request) != OTHER:
            who = "first"
        else:
            who = "second" if _plain_prompt_of(body, total) else "backup"
        arrived.append(f"{who} {kind_of(request)}")
        await asyncio.sleep(0)
        return route(first, other)(request)

    return handler, arrived


def read_with_one_slot(
    pages: int,
    handler: Callable[[httpx.Request], Coroutine[None, None, httpx.Response]],
    cfg: Config,
) -> Result:
    """process() with one shared request slot; the gate's counters are all back to nothing
    and the slot is free at the end."""

    async def go() -> tuple[Result, pl._Waiting, asyncio.Semaphore]:
        slots = asyncio.Semaphore(1)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            res = await pl.process(scan(pages), "application/pdf", client, cfg, request_slots=slots)
        return res, pl._waiting(slots), slots

    res, waiting, slots = on_loop(go())
    assert (waiting.ordinary, waiting.held, waiting.starting) == (0, 0, 0)
    assert slots._value == 1
    return res


def test_the_backup_takes_its_slot_before_any_waiting_second_reading() -> None:
    """One shared slot; the second readings wait for it from the start. Page 2, the last
    first reading, fails at its provider when nothing else wants the slot: the backup asks
    for it in the same step, as an ordinary request, before a second reading can look."""
    kind = scan_kind(2, 2)
    first, other = Script(kind.with_first(DOWN)), Script(kind.good)
    handler, arrived = arrivals(kind, first, other, 2)
    res = read_with_one_slot(2, handler, on(second_reading_model=OTHER))
    assert arrived[:4] == ["first header", "first 1", "first 2", "backup 2"], arrived
    assert sorted(arrived[4:]) == ["second 1", "second 2"]
    assert res.pages[1].by_backup


def test_the_backup_waits_its_turn_among_first_readers_not_after_them() -> None:
    """One shared slot. The header fails at its provider while the three pages (read by a
    model of their own, which stays up) wait for the slot; each page is then read again.
    The backup's header request queues as an ordinary request: after the pages already
    waiting, before their retries."""
    kind = scan_kind(3, 1)
    answers = kind.good | {"header": [DOWN]}
    answers |= {p: [[line(p, amount=99.0)], [line(p)]] for p in (1, 2, 3)}
    first, other = Script(answers), Script(kind.good)
    handler, arrived = arrivals(kind, first, other, 3)
    res = read_with_one_slot(3, handler, on(page_models=("vendor/pages", "vendor/pages")))
    assert arrived == [
        "first header",
        "first 1",
        "first 2",
        "first 3",
        "backup header",
        "first 1",
        "first 2",
        "first 3",
    ]
    assert res.status == "ok" and res.header_attempts[-1].model == OTHER


# --------------------------------------------------------------------------- #
# A hostile answer of the backup goes through the same parsing as the first model's
# --------------------------------------------------------------------------- #


HOSTILE = (
    '{"carry_in": 7, "rows": [1, "x", null, '
    '{"line_no": 1, "sku": ["SKU-001"], "quantity": NaN, "unit_price": Infinity, '
    '"tax_rate": true, "amount": "' + "9" * 400 + '"}, {"line_no": 2, "sku": "SKU-002"}]}'
)


def test_a_hostile_backup_answer_is_read_as_the_first_models_would_be() -> None:
    kind = scan_kind(2, 1)
    answer = events(content(HOSTILE, "stop"))
    once = Config(page_models=(FAST,))  # the hostile answer is the one kept
    by_backup, _, _ = read(kind, DOWN, on(once), kind.good | {1: [answer]})
    by_first, _, _ = read(kind, answer, once)
    assert by_backup.pages[0].by_backup and by_backup.lines == by_first.lines
    assert [i.to_json() for i in by_backup.pages[0].issues] == [
        i.to_json() for i in by_first.pages[0].issues
    ]
    seen, expected = (json.loads(result_view(r).model_dump_json()) for r in (by_backup, by_first))
    assert (seen["lines"], seen["issues"]) == (expected["lines"], expected["issues"])
    json.dumps(seen, allow_nan=False)  # what the API serves is strict JSON
    # An integer of 5,000 digits: the parser refuses the answer, as the first model's.
    huge = events(content('{"rows": [{"sku": "A", "amount": ' + "9" * 5000 + "}]}", "stop"))
    res, _, backup = read(kind, DOWN, on(once), kind.good | {1: [huge]})
    assert len(backup.bodies) == 1 and res.pages[0].attempts[1].error is None
    json.dumps(result_view(res).model_dump(mode="json"), allow_nan=False)


def test_switched_off_nothing_reaches_the_backup() -> None:
    assert Config().fallback_model is None
    kind = scan_kind(2, 1)
    res, _, backup = read(kind, DOWN, kind.cfg)
    assert backup.bodies == [] and res.pages[0].by_backup is False


def test_a_page_read_again_by_a_configured_second_model_is_not_the_backups() -> None:
    """The same model as the backup, but as the next attempt the configuration asks for -
    not because a provider failed: no note."""
    kind = scan_kind(1, 1)
    first = Script(kind.with_first([line(1, amount=99.0)]))
    res = run(scan(1), "application/pdf", first, on(page_models=(FAST, OTHER)))
    assert [a.model for a in res.pages[0].attempts] == [FAST, OTHER]
    assert res.pages[0].model == OTHER and not res.pages[0].by_backup
    assert fallback_notes(res) == []


def test_a_backup_that_is_the_first_model_sends_nothing_more() -> None:
    kind = scan_kind(1, 1)
    first = Script(kind.with_first(DOWN))
    res = run(scan(1), "application/pdf", first, Config(fallback_model=FAST))
    assert first.calls[1] == 2 and [a.model for a in res.pages[0].attempts] == [FAST, FAST]
    # Both are the page's own attempts, as without a backup: no backup request among them.
    assert {b["max_tokens"] for b in first.bodies if key_of(b) == 1} == {16_000}
    assert not res.pages[0].by_backup and fallback_notes(res) == []


# --------------------------------------------------------------------------- #
# The pinned scenarios: unchanged while the first provider answers
# --------------------------------------------------------------------------- #

HEALTHY = [c for c in CASES if c.name not in ("scan_failures", "layout_failed")]


def golden_run(
    case: Case, served: bool, backup: Handler, first: Model | None = None
) -> tuple[Result, Model]:
    """process() on a scenario with the backup on, its model as the first one."""
    data, mime = case.data()
    model = first or case.model()
    model.models[OTHER] = backup
    cfg = Config(**case.config, fallback_model=OTHER)

    async def go() -> Result:
        async with httpx.AsyncClient(transport=httpx.MockTransport(model)) as client:
            if not served:
                return await pl.process(data, mime, client, cfg)
            with ThreadPoolExecutor(1) as lane:
                return await pl.process(
                    data,
                    mime,
                    client,
                    replace(cfg, deadline_s=240.0),
                    lane=lane,
                    request_slots=asyncio.Semaphore(24),
                )

    return on_loop(go()), model


def golden(case: Case) -> JSONObject:
    pinned: JSONObject = json.loads((GOLDEN / f"{case.name}.json").read_text())
    return pinned


@pytest.mark.parametrize("served", [False, True], ids=["plain", "served"])
@pytest.mark.parametrize("case", HEALTHY, ids=[c.name for c in HEALTHY])
def test_with_the_backup_on_a_scenario_is_read_as_pinned(case: Case, served: bool) -> None:
    """No request of these scenarios fails at its provider: the backup is never asked, and
    the result and every request are those of the golden file."""
    asked = Script({})
    res, model = golden_run(case, served, asked)
    assert asked.bodies == []
    pinned = golden(case)
    data, _ = case.data()
    exact = sha256(data) == pinned["input_sha256"]
    differences = compare(comparable(result_of(res.to_json())), comparable(pinned["result"]))
    sent = comparable_requests([request_of(b) for b in model.bodies], exact)
    differences += compare(sent, comparable_requests(pinned["requests"], exact))
    assert differences == []


def shifted(case: Case) -> Model:
    """The scenario's model answering each request with the answer its first model gave to
    that request's next attempt in the golden file: the backup reads what it would have."""
    model = case.model()
    model.calls = Counter({k: 1 for k in [*model.pages, "header", "layout"]})
    return model


def what_a_person_checks(result: JSONObject) -> JSONObject:
    keys = ("status", "header", "lines", "checkpoints", "issues", "notes")
    out = {k: comparable(result)[k] for k in keys}
    out["notes"] = [n for n in out["notes"] if n["kind"] != "fallback_model"]
    for note in out["notes"]:
        if note["kind"] == "stronger_model_not_called":  # it counts the reads
            note.pop("text")
    return out


@pytest.mark.parametrize("served", [False, True], ids=["plain", "served"])
@pytest.mark.parametrize("name", ["scan_failures", "layout_failed"])
def test_with_the_backup_on_a_failed_provider_reads_what_the_golden_file_did(
    name: str, served: bool
) -> None:
    """The first model fails here: its requests up to the failure are the golden file's,
    every later one goes to the backup, and - with the backup answering what the first
    model answered then - a person sees the same result, but the note."""
    case = next(c for c in CASES if c.name == name)
    backup = shifted(case)
    res, model = golden_run(case, served, backup)
    pinned = golden(case)
    data, _ = case.data()
    exact = sha256(data) == pinned["input_sha256"]
    sent = comparable_requests([request_of(b) for b in model.bodies], exact)
    expected = comparable_requests(pinned["requests"], exact)
    assert compare(sent, expected[: len(sent)]) == []
    assert len(sent) < len(expected)  # the first model was not asked everything
    asked = comparable_requests([request_of(b) for b in backup.bodies], exact)
    # Each backup request repeats one of the golden file's: the same prompt and file.
    assert asked and all(r["model"] == OTHER for r in asked)
    assert all(r["messages"] in [e["messages"] for e in expected] for r in asked)
    seen = what_a_person_checks(result_of(res.to_json()))
    assert compare(seen, what_a_person_checks(pinned["result"])) == []
    assert [n["model"] for n in fallback_notes(res)] == [OTHER]


def test_a_layout_the_backup_gives_keeps_the_rows_found_by_code() -> None:
    case = next(c for c in CASES if c.name == "layout_failed")
    _, _, header, pages = truth_doc("s_classic")
    backup = Model([header], {p: [rows] for p, rows in pages.items()}, saved_layout("s_classic"))
    res, _ = golden_run(case, False, backup)
    assert res.layout["status"] == "confirmed" and res.method == "layout"
    [note] = fallback_notes(res)
    assert note["text"] == note_text("the layout of the item table")


def test_switched_off_a_scenario_that_fails_never_asks_the_backup() -> None:
    asked = Script({})
    every: dict[Key, list[Answer]] = {k: [DOWN] for k in ("header", "layout", 1)}
    model = Script(every)
    res = run(*CASES[4].data(), route(model, asked), Config())
    assert asked.bodies == [] and res.status == "failed"
