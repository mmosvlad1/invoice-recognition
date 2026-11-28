"""Page-by-page invoice extraction.

Instead of one request for the whole document:
  - the header and totals are read from the first and last page,
  - every page is read by its own request, all in parallel,
  - every page is checked on its own (arithmetic; text layer when there is one)
    and re-read if it fails - only that page,
  - pages are concatenated in order and the document is checked as a whole.

The unit of work, of failure and of retry is a page, not a document.

A PDF with a text layer takes a shorter road (layout.py): the model describes
the layout of the table once, code reads the rows of every page from the text
layer, and the model reads one page the usual way as a witness. Pages are
checked exactly as above; a page that fails is read by the model, and so is
the whole document if the layout cannot be used or the witness disagrees.

A line item cut by a page break (carryover.py): in a PDF of more than one page the
answer of a page read by the model has one more field, "carry_in", for the end of a
description printed at the top of the page. When all pages are read, code appends it
to the last line item of the previous page. What was joined, and what found no row to
go to, is in Result.notes; it raises no issue.

A photo (any file that is not a PDF) is measured before the first request
(quality.py): a cut sheet, blur, darkness, glare. A photo that cannot be read ends as
"failed" with the reason and no request; a doubtful one is read and carries the
reason as an issue.

Where each value is printed (boxes.py): for a PDF with its own text layer, every line
item gets the rectangles of its printed values on its page, as the page is displayed -
from the cells the layout parser cut, or, for a page read by the model, found in the text
layer afterwards. Scans and photos get the page only. This runs on the lane after the
reading, and never changes a value, an issue, a status or a request: a failure there
costs the rectangles, nothing else.

A scanned PDF (no text layer, or page images with somebody's OCR text over them) is
measured page by page before the first request, by the same numbers minus what is about
a sheet on a desk (quality.judge_page). A page without text is not sent and leaves a
note; a page too blurred to read is not sent and leaves the issue "scan_blurred" with its
number, the other pages are read as usual; a doubtful page is read and flagged
"scan_<code>". When no page is left to read, the document ends as "failed" with no
request.

A second reading of a scan (second_reading.py), when Config.second_reading_model names a
model: every page of a scan sent without a quality warning is also read once by that
model, next to the first reading. Where the two disagree on a line item, the item gets an
issue "second_reading" (a warning, with the other model's values as a suggestion); the
lines stay those of the first reading. A second reading yields every shared request slot
to the first readers (Gate.enter) and never counts against the deadline; one that cannot
be used leaves one note and changes nothing else.
"""

import asyncio
import logging
import time
import weakref
from collections import deque
from collections.abc import Callable
from concurrent.futures import Executor
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dataclass_fields
from decimal import Decimal
from typing import Any, cast, get_args

import httpx
from PIL import Image

from invoice_recognition.llm import (
    PDF_NATIVE,
    ContentPart,
    LLMResult,
    complete,
    file_part,
    parse_json,
    salvage_lines,
)
from invoice_recognition.recognition import boxes as bx
from invoice_recognition.recognition import carryover as co
from invoice_recognition.recognition import checks, photo
from invoice_recognition.recognition import evidence as ev
from invoice_recognition.recognition import issue_values as iv
from invoice_recognition.recognition import layout as lay
from invoice_recognition.recognition import quality as ql
from invoice_recognition.recognition import second_reading as sr
from invoice_recognition.recognition.checks import Issue
from invoice_recognition.recognition.config import Config
from invoice_recognition.recognition.pdf import extract_pages, page_count
from invoice_recognition.recognition.prompts import EXTRACT_HEADER, EXTRACT_PAGE
from invoice_recognition.recognition.results import Attempt, PageResult, Result
from invoice_recognition.shapes import Checkpoint, Doc, Header, JSONObject, Line

logger = logging.getLogger(__name__)

LINE_KEYS = (
    "line_no",
    "sku",
    "description",
    "quantity",
    "unit",
    "unit_price",
    "tax_rate",
    "amount",
)


# --------------------------------------------------------------------------- #
# One page
# --------------------------------------------------------------------------- #


def repair(lines: list[Line]) -> int:
    """Fix the one error that can be fixed without a model: a quantity whose
    thousands separator was read as a decimal point ("8.200" -> 8.2). The
    printed amount proves which reading is right."""
    fixed = 0
    for item in lines:
        q = checks.dec(item.get("quantity"))
        p = checks.dec(item.get("unit_price"))
        a = checks.dec(item.get("amount"))
        if q is None or p is None or a is None or abs(ev.money(q * p) - a) <= ev.CENT:
            continue
        if abs(ev.money(q * 1000 * p) - a) <= ev.CENT:
            item["quantity"] = checks.plain(q * 1000)
            item["repaired"] = "quantity_x1000"
            fixed += 1
    return fixed


def rows_to_page(rows: list[Any], page: int) -> tuple[list[Line], list[Checkpoint]]:
    """Split the model's ordered rows into line items and control totals. The
    position of a total is derived here, from the order - never asked from the model."""
    lines: list[Line] = []
    totals: list[Checkpoint] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if "total_label" in row:
            totals.append(
                {
                    "label": row.get("total_label"),
                    "amount": row.get("amount"),
                    "after_line": len(lines),
                    "page": page,
                }
            )
        elif row.get("sku") is not None or row.get("amount") is not None:
            lines.append({k: row.get(k) for k in LINE_KEYS} | {"page": page})
    return lines, totals


def check_page(
    lines: list[Line], totals: list[Checkpoint], page: int, evidence: ev.Evidence | None
) -> list[Issue]:
    doc: Doc = {"lines": lines, "checkpoints": totals}
    issues = checks.check_line_math(doc) + checks.check_checkpoints(doc, partial=True)
    if evidence is not None:
        only = ev.Evidence(evidence.source, evidence.locale, evidence.page(page))
        issues += checks.check_grounding(doc, only) + checks.check_coverage(doc, only)
    for i in issues:
        i.page = page
        if i.line is not None and 0 <= i.line < len(lines):
            no = lines[i.line].get("line_no")
            if isinstance(no, int) and not isinstance(no, bool):
                i.line_no = no
    return issues


class DeadlinePassed(Exception):
    """The overall deadline of the document passed before a request could be sent."""


class _Waiting:
    """Who wants the request slots shared by every document given the same semaphore.

    ordinary: requests of a first reader that want a slot and hold none yet - waiting for
    a slot, or for their document's own limit. starting: documents whose first requests
    exist as tasks that have not all run yet, so have not asked for their slots yet.
    optional: the optional requests (second readings) waiting for a slot. held: the slots
    optional requests hold now; cap: the most they may hold at once - half of the slots
    (at least one), so that a first-reader request that comes late always finds slots no
    optional request can take."""

    def __init__(self, size: int) -> None:
        self.ordinary = 0
        self.starting = 0
        self.optional: deque[asyncio.Future[None]] = deque()
        self.held = 0
        self.cap = max(1, size // 2)

    def free_for_optional(self, slots: asyncio.Semaphore) -> bool:
        return (
            self.ordinary == 0
            and self.starting == 0
            and self.held < self.cap
            and not slots.locked()
        )

    def wake(self) -> None:
        """Let every optional request look again whether it may take a slot."""
        while self.optional:
            waiter = self.optional.popleft()
            if not waiter.done():
                waiter.set_result(None)

    def started(self) -> None:
        self.starting -= 1
        if self.starting == 0:
            self.wake()


# One entry per semaphore of shared slots, for as long as somebody holds the semaphore:
# the documents that share the slots share this too.
_WAITING: weakref.WeakKeyDictionary[asyncio.Semaphore, _Waiting] = weakref.WeakKeyDictionary()


def _waiting(slots: asyncio.Semaphore) -> _Waiting:
    found = _WAITING.get(slots)
    if found is None:
        # asyncio.Semaphore does not tell its size: it is read from its private counter,
        # once, when the flow first sees the semaphore. Only the flow takes its slots, so
        # all are free then; were some taken, the cap would only come out smaller.
        found = _WAITING[slots] = _Waiting(slots._value)
    return found


def _starting(gate: Gate) -> None:
    """Call right after the requests a document starts with are created as tasks, before
    any of them ran: optional requests wait until each of them has asked for its slot.
    The tasks run in the order they were created, so a callback scheduled now runs after
    the first step of every one of them."""
    if gate.slots is not None:
        waiting = _waiting(gate.slots)
        waiting.starting += 1
        asyncio.get_running_loop().call_soon(waiting.started)


async def _optional_slot(slots: asyncio.Semaphore, waiting: _Waiting) -> None:
    """Take a shared slot for an optional request without making an ordinary request of
    any document sharing the slots wait for it: only while no ordinary request wants one
    (waiting for a slot, or for its document's own limit), no document is starting, and
    optional requests hold fewer than half of the slots.

    So an ordinary request never waits for an optional one that has not been sent. The
    requests a document starts with, and those of every document started in the same
    step of the event loop, are counted before an optional request can look. A first
    reader's retry asks for its slot in the step in which its own attempt gives one back,
    before any optional request can run.

    An ordinary request sent after a pause in which nothing wanted a slot - a document
    whose local work ends later, or the pages a text-layer document reads after its layout
    answer - finds at least half of the slots out of reach of optional requests. It waits
    for an optional request already sent (at most that request's timeout) only when it
    needs more than those, and the other first readers hold the rest."""
    while not waiting.free_for_optional(slots):
        waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        waiting.optional.append(waiter)
        await waiter
    await slots.acquire()  # free, and nobody waits for it: taken at once
    waiting.held += 1


@dataclass
class Gate:
    """What every model request of one process() call passes before it is sent.

    slots: the limit on requests in flight shared by every document being read (None:
    no shared limit). Each document also has its own limit (Config.concurrency) on its
    page requests; that one is passed to enter() and leave() as `own`.

    deadline: the overall deadline of the document on the time.monotonic() clock (None:
    none). hit: a request was not sent, or was cut short, because it passed."""

    slots: asyncio.Semaphore | None = None
    deadline: float | None = None
    hit: bool = False

    def remaining(self) -> float | None:
        return None if self.deadline is None else self.deadline - time.monotonic()

    def passed(self) -> bool:
        return self.deadline is not None and time.monotonic() >= self.deadline

    def cut(self, res: LLMResult) -> bool:
        """The request ended in an error when the deadline had passed: it was cut short by
        it. Told by the clock, right after the request returns, never by the error text."""
        if res.error is not None and self.passed():
            self.hit = True
            return True
        return False

    async def enter(self, own: asyncio.Semaphore | None, optional: bool = False) -> None:
        """Wait for the document's own limit, then for a shared slot: always in this
        order, so that no two requests hold one each and wait for the other.

        optional: a request nothing depends on (a second reading). It takes a shared slot
        only when no ordinary request wants one, and optional requests hold fewer than
        half of the slots (_optional_slot). Without shared slots there is nothing to
        yield.

        Only these waits are bounded by the deadline, never the request itself: a request
        that is sent runs to its own timeout (which _call shortens to the deadline), so its
        attempt, tokens and cost are kept. Raises DeadlinePassed, holding nothing, when the
        deadline has passed or passes while waiting."""
        held: list[asyncio.Semaphore] = []
        waiting = None if self.slots is None else _waiting(self.slots)
        ordinary = waiting is not None and not optional
        if waiting is not None and ordinary:
            waiting.ordinary += 1
        try:
            if self.passed():
                raise DeadlinePassed
            async with asyncio.timeout(self.remaining()):
                if own is not None:
                    await own.acquire()
                    held.append(own)
                if self.slots is not None and waiting is not None:
                    if ordinary:
                        await self.slots.acquire()
                    else:
                        await _optional_slot(self.slots, waiting)
                    held.append(self.slots)
            if self.passed():
                raise DeadlinePassed
        except TimeoutError, DeadlinePassed:
            self._give_back(held, waiting, optional)
            self.hit = True
            raise DeadlinePassed from None
        except BaseException:  # cancelled while waiting: give back what was taken
            self._give_back(held, waiting, optional)
            raise
        finally:
            if waiting is not None and ordinary:
                waiting.ordinary -= 1
                if waiting.ordinary == 0 and not held:
                    # It gave up its place: a slot may be free with no release to come.
                    waiting.wake()

    def _give_back(
        self, held: list[asyncio.Semaphore], waiting: _Waiting | None, optional: bool
    ) -> None:
        for sem in held:
            sem.release()
        if waiting is not None and self.slots in held:
            if optional:
                waiting.held -= 1
            waiting.wake()
        held.clear()

    def leave(self, own: asyncio.Semaphore | None, optional: bool = False) -> None:
        for sem in (self.slots, own):
            if sem is not None:
                sem.release()
        if self.slots is not None:
            waiting = _waiting(self.slots)
            if optional:
                waiting.held -= 1
            waiting.wake()


async def _call(
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
    part: ContentPart | None,
    max_tokens: int,
    timeout_s: float,
    *,
    gate: Gate | None = None,
    own: asyncio.Semaphore | None = None,
    plugins: list[JSONObject] | None = None,
    optional: bool = False,
) -> LLMResult:
    """complete() that never raises: whatever goes wrong inside one request is
    that request's error, not the document's. part=None: a text-only request.
    plugins: OpenRouter plugins of the request (None: none, the body as before).
    optional: a request that yields every shared slot to the others (Gate.enter).

    Every wait before a request happens here: for `own`, the limit of the document on its
    page requests (None for the header and the layout), and for a shared slot of the
    gate. Raises DeadlinePassed when the request cannot be sent before the deadline; a
    request that is sent ends at the deadline at the latest."""
    if gate is None:
        gate = Gate()
    await gate.enter(own, optional)
    try:
        left = gate.remaining()
        return await complete(
            client,
            model,
            prompt,
            [part] if part else [],
            max_tokens=max_tokens,
            timeout_s=timeout_s if left is None else min(timeout_s, left),
            plugins=plugins,
        )
    except Exception as e:  # fault boundary: one request
        logger.warning("request to %s failed", model, exc_info=True)
        return LLMResult(model=model, error=f"{type(e).__name__}: {e}")
    finally:
        gate.leave(own, optional)


def _attempt(res: LLMResult, lines: int, issues: int) -> Attempt:
    return Attempt(
        res.model,
        res.duration_s,
        res.cost_usd or 0.0,
        res.prompt_tokens,
        res.completion_tokens,
        lines,
        issues,
        res.error,
        cost_known=res.cost_usd is not None,
    )


async def read_page(
    client: httpx.AsyncClient,
    part: ContentPart,
    page: int,
    total: int,
    evidence: ev.Evidence | None,
    cfg: Config,
    sem: asyncio.Semaphore,
    carry: bool = False,
    *,
    gate: Gate | None = None,
) -> PageResult:
    """Read one page, once per entry of cfg.page_models until an attempt passes its
    checks, and keep the best attempt. sem: the document's limit on page requests in
    flight; gate: the shared one and the deadline (None: neither).

    At the deadline the attempts stop. A page with no attempt that ran to its end (none
    was sent, or every one was cut short) gets a "page_read" issue that says so and
    PageResult.deadline_unread.

    Issues of the returned page index its own lines (0-based within the page);
    process() shifts them to document positions.

    carry=True: the page is asked with the "carry_in" field (carryover.py);
    PageResult.carry_in is the field of the attempt that is kept. carry=False:
    EXTRACT_PAGE, to the byte."""
    best: PageResult | None = None
    best_key: tuple[bool, int, int] = (True, 0, 0)
    attempts: list[Attempt] = []
    counts: list[int] = []  # lines per attempt, for the attempts that were read cleanly
    prompt = co.carry_prompt(page, total) if carry else EXTRACT_PAGE.format(page=page, total=total)
    if gate is None:
        gate = Gate()
    stopped = completed = False
    for model in cfg.page_models:
        try:
            res = await _call(
                client,
                model,
                prompt,
                part,
                cfg.page_max_tokens,
                cfg.page_timeout_s,
                gate=gate,
                own=sem,
            )
        except DeadlinePassed:
            stopped = True
            break  # no stronger-model note: the page was stopped by the deadline
        cut = gate.cut(res)
        completed = completed or not cut
        parsed = parse_json(res.text)
        rows = (parsed or {}).get("rows")
        tail = ""
        if rows is None:  # broken or cut-off answer: keep whatever complete rows it has
            rows = salvage_lines(res.text)
        elif carry and parsed is not None and isinstance(parsed.get("carry_in"), str):
            # Not taken from a salvaged answer; anything but a string is no tail.
            tail = co.carry_text(parsed["carry_in"])
        rows = rows if isinstance(rows, list) else []
        lines, totals = rows_to_page(rows, page)
        fixed = repair(lines)
        issues = check_page(lines, totals, page, evidence)
        failed = bool(res.error) or parsed is None
        if failed:
            why = res.error or f"answer is not valid JSON (finish={res.finish_reason})"
            issues.append(Issue("page_read", why, page=page, form="failed"))
        else:
            counts.append(len(lines))
            # Without a text layer nothing else notices a page that came back empty
            # (with one, check_coverage does). A page with only control totals is not empty.
            if not lines and not totals and evidence is None:
                issues.append(
                    Issue(
                        "page_empty",
                        "the answer has no line items and no control totals",
                        page=page,
                    )
                )
        dropped = len(rows) - len(lines) - len(totals)
        if dropped:
            issues.append(
                Issue(
                    "rows_dropped",
                    f"{dropped} row(s) of the answer have neither sku nor amount and were left out",
                    page=page,
                    values=iv.RowsDropped(dropped),
                )
            )
        attempts.append(_attempt(res, len(lines), len(issues)))
        cand = PageResult(page, lines, totals, issues, [], fixed, carry_in=tail, model=model)
        # A clean read beats a failed one, then fewer issues, then more lines; on a tie the
        # earlier attempt stays. Counting issues alone let an emptier answer win.
        key = (failed, len(issues), -len(lines))
        if best is None or key < best_key:
            best, best_key = cand, key
        if not best.issues or cut:  # cut: the deadline passed, no attempt can follow
            break
    else:
        # Every attempt was made and the best one still fails its checks: this is where a
        # stronger model would read the page again (process() notes it; none is called).
        # Not when no request got an answer: the page_read issue says the page could not be
        # read, and another model is not the remedy. Not when the attempts already name
        # more than one model: escalation is configured.
        if best is not None:
            answered = any(a.error is None for a in attempts)
            best.stronger_model = answered and len(set(cfg.page_models)) == 1
    if best is None:
        if not stopped:
            raise ValueError("no attempt was made: Config.page_models is empty")
        best = PageResult(page)  # the deadline passed before the first attempt
    if not completed:
        # One page_read issue for the page, this one: the error of each cut attempt (its
        # timeout, shortened to the deadline) stays in its Attempt.
        best.issues = [i for i in best.issues if i.check != "page_read"]
        best.issues.append(deadline_unread_issue(page, len(best.lines)))
        best.deadline_unread = True
    best.attempts = attempts
    if evidence is None and len(set(counts)) > 1:
        # No text layer to say which attempt is complete: keep the page flagged.
        best.issues.append(
            Issue(
                "attempts_disagree",
                "attempts returned different numbers of lines: " + " and ".join(map(str, counts)),
                page=page,
                values=iv.AttemptCounts(tuple(counts)),
            )
        )
    return best


def _other_model_extra(part: ContentPart) -> list[JSONObject] | None:
    """The plugins of a request to a model of another provider: a PDF must reach it as the
    file, not as the text of OpenRouter's OCR of it. None for an image."""
    return PDF_NATIVE if part.get("type") == "file" else None


async def read_second(
    client: httpx.AsyncClient,
    model: str,
    part: ContentPart,
    page: int,
    total: int,
    cfg: Config,
    own: asyncio.Semaphore,
    gate: Gate,
) -> tuple[Attempt | None, list[Line] | None]:
    """Read one page of a scan once more, by `model`, for comparison with the first
    reading (second_reading.py). One request, no retry: EXTRACT_PAGE, to the byte (never
    the "carry_in" prompt), the same file part as the first reader got.

    Returns the attempt (None: not sent, the deadline passed first) and the lines as the
    flow keeps a page - line items only, the quantity repair applied - or None when the
    answer cannot be compared: an error, a timeout, an answer that is not valid JSON or
    was cut at the token limit (its missing rows would all look like disagreements), or
    no list of rows. The lines of a broken answer are never salvaged.

    own and gate: the document's limit on second readings and a gate of their own that
    shares the document's slots and deadline. The request is optional: it takes a shared
    slot only when no request of a first reader wants one (Gate.enter). A second reading
    stopped by the deadline is never the document's deadline_exceeded: nothing reads that
    gate's hit."""
    try:
        res = await _call(
            client,
            model,
            EXTRACT_PAGE.format(page=page, total=total),
            part,
            cfg.second_reading_max_tokens,
            cfg.second_reading_timeout_s,
            gate=gate,
            own=own,
            plugins=_other_model_extra(part),
            optional=True,
        )
    except DeadlinePassed:
        return None, None
    answer = None if res.error or res.finish_reason == "length" else parse_json(res.text)
    rows = (answer or {}).get("rows")
    if not isinstance(rows, list):
        return _attempt(res, 0, 0), None
    lines, _ = rows_to_page(rows, page)
    repair(lines)
    return _attempt(res, len(lines), 0), lines


def _pages(pages: list[int]) -> tuple[str, bool]:
    """("Page 3", True) or ("Pages 2 and 4–5", False)."""
    if len(pages) == 1:
        return f"Page {pages[0]}", True
    return f"Pages {page_ranges(pages)}", False


def second_reading_note(
    unanswered: list[int],
    unusable: list[int],
    uncompared: list[int] | None = None,
    first_unread: list[int] | None = None,
) -> JSONObject:
    """The one note for the pages of a scan that were to be read a second time and were
    not compared with it. unanswered: the request failed, timed out or was not sent before
    the deadline. unusable: an answer came, but not one that can be compared (broken, cut
    at the token limit, or without line items while the first reading has some).
    uncompared: a usable one came, but the page could not be compared - its first reading
    failed (first_unread, a subset), it was itself read by the second reader's model, or
    the comparison broke. Their status is what it would be without the second reading."""
    uncompared, first_unread = uncompared or [], first_unread or []
    text = []
    if unanswered:
        pages, one = _pages(unanswered)
        text.append(
            f"{pages} {'was' if one else 'were'} not read a second time: the other model did "
            "not answer, or not in time."
        )
    if unusable:
        pages, one = _pages(unusable)
        text.append(
            f"{pages} {'has' if one else 'have'} no usable second reading: the other model's "
            "answer was broken or had no line items."
        )
    if uncompared:
        pages, one = _pages(uncompared)
        text.append(
            f"{pages} {'was' if one else 'were'} read a second time, but the two readings "
            "could not be compared."
        )
    every = sorted({*unanswered, *unusable, *uncompared})
    # A page the first reader could not read already has its own issue that says so.
    unchecked = [p for p in every if p not in first_unread]
    if unchecked == every:
        which = "this page" if len(every) == 1 else "these pages"
        text.append(f"Nothing else checks the item codes (SKUs) on {which}.")
    elif unchecked:
        pages, _ = _pages(unchecked)
        text.append(f"Nothing else checks the item codes (SKUs) on {pages[0].lower()}{pages[1:]}.")
    return {
        "page": None,
        "kind": "second_reading_missing",
        "text": " ".join(text),
        "source": "second_reading",
        "pages": every,
    }


def compare_second(
    pages: list[PageResult],
    second_pages: list[int],
    seconds: list[tuple[Attempt | None, list[Line] | None]],
    model: str | None,
) -> tuple[list[Issue], JSONObject | None]:
    """Compare every page read a second time with the page as the result keeps it (the
    attempt read_page chose, with any carried text joined). Sets PageResult.second_reading
    and, on its attempt, the number of issues it raised. Returns the issues, `line`
    indexing Result.lines (pages: every page of the document, in order), and the note for
    the pages not compared (None: every one was). model: the second reader's model - a
    page whose kept attempt is by that model is not compared with itself."""
    first: dict[int, int] = {}  # index of the first line of each page within the document
    done = 0
    for pr in pages:
        first[pr.page] = done
        done += len(pr.lines)
    issues: list[Issue] = []
    unanswered: list[int] = []
    unusable: list[int] = []
    uncompared: list[int] = []
    first_unread: list[int] = []
    for p, (attempt, read) in zip(second_pages, seconds, strict=True):
        pr = pages[p - 1]
        pr.second_reading = attempt
        if read is None or (not read and pr.lines):  # nothing to hold the page against
            answered = attempt is not None and attempt.error is None
            (unusable if answered else unanswered).append(p)
            continue
        if any(i.check == "page_read" for i in pr.issues):  # nothing reliable to compare
            uncompared.append(p)
            first_unread.append(p)
            continue
        if pr.model is not None and pr.model == model:  # one model twice is not two readers
            uncompared.append(p)
            continue
        try:
            found = sr.compare_page(pr.lines, read, p)
        except Exception:  # fault boundary: a bug here costs this comparison, not the document
            logger.warning("the two readings of page %d could not be compared", p, exc_info=True)
            uncompared.append(p)
            continue
        for issue in found:
            if issue.line is not None:
                issue.line += first[p]
        if attempt is not None:
            attempt.issues = len(found)
        issues += found
    missed = unanswered or unusable or uncompared
    note = second_reading_note(unanswered, unusable, uncompared, first_unread) if missed else None
    return issues, note


def deadline_unread_issue(page: int, lines: int) -> Issue:
    """The page_read issue of a page no attempt at which ran to its end; lines: the line
    items kept from a cut attempt."""
    text = deadline_unread_text(page, lines)
    if not lines:
        return Issue("page_read", text, page=page, form="not_read")
    return Issue("page_read", text, page=page, values=iv.CutByDeadline(lines))


def deadline_unread_text(page: int, lines: int) -> str:
    """The message of the page_read issue of a page no attempt at which ran to its end;
    lines: the line items kept from a cut attempt."""
    if not lines:
        return f"Page {page} was not read: the time limit for the document passed."
    items = "line item" if lines == 1 else "line items"
    return (
        f"Page {page} was not read to the end: the time limit for the document passed. "
        f"The {lines} {items} read before that may be incomplete."
    )


def stronger_model_note(pr: PageResult) -> JSONObject:
    """The note for a page that still fails its checks after every attempt. Reading it once
    more with a stronger model is where the flow would go next; that model is not called,
    to save cost, so the page is left to the person who reviews the result."""
    n = len(pr.attempts)
    return {
        "page": pr.page,
        "kind": "stronger_model_not_called",
        "text": f"Page {pr.page} still has problems after {n} read{'s' if n != 1 else ''}. "
        "A stronger, more expensive model could read it again; it is not used here, to save "
        "cost. Please check this page.",
        "source": "retry",
    }


# --------------------------------------------------------------------------- #
# Text-layer PDF: layout from the model, rows by code
# --------------------------------------------------------------------------- #

# Parser notes that do not touch any extracted value: reported in Result.notes, never as issues.
HARMLESS_NOTES = ("unclassified_amount", "total_without_amount", "joined_across_pages")

# The fields a parser note that becomes an issue is about. A note names a text row of the
# page, not a line item, so the issue has no line and these are columns. Words outside
# every column ("stray_words") belong to no known cell.
NOTE_FIELDS = {"no_tax_rate": ["line.tax_rate"], "text_under_item": ["line.description"]}

# Quantity and unit price, the two columns arithmetic cannot tell apart (q x p = p x q).
CROSSED_COLUMNS = ["line.quantity", "line.unit_price"]


async def ask_layout(
    client: httpx.AsyncClient, prompt: str, cfg: Config, *, gate: Gate | None = None
) -> tuple[JSONObject | None, list[Attempt], str | None]:
    """One text-only request. Returns (answer, attempts, why it cannot be used)."""
    if gate is None:
        gate = Gate()
    try:
        res = await _call(
            client, cfg.layout_model, prompt, None, cfg.layout_max_tokens, 90, gate=gate
        )
    except DeadlinePassed:
        return None, [], "the overall deadline passed"
    gate.cut(res)
    raw = None if res.error else parse_json(res.text)
    error = res.error or (
        None if raw is not None else f"answer is not valid JSON (finish={res.finish_reason})"
    )
    return raw, [_attempt(res, 0, 1 if error else 0)], error


def layout_pages(
    parsed: lay.Parsed, n: int, evidence: ev.Evidence
) -> tuple[dict[int, PageResult], list[JSONObject]]:
    """Rows found by code -> one PageResult per page, checked like a page read by
    the model. Also returns the harmless notes of the parser."""
    by_page: dict[int, list[Line]] = {p: [] for p in range(1, n + 1)}
    totals: dict[int, list[Checkpoint]] = {p: [] for p in range(1, n + 1)}
    first: dict[int, int] = {}  # index of the first line of each page within the document
    for idx, item in enumerate(parsed.lines):
        first.setdefault(item["page"], idx)
        by_page[item["page"]].append(
            {k: (checks.plain(v) if isinstance(v, Decimal) else v) for k, v in item.items()}
        )
    done = 0
    for p in range(1, n + 1):  # a page without lines starts where the previous one ended
        first.setdefault(p, done)
        done = first[p] + len(by_page[p])
    for c in parsed.checkpoints:
        totals[c["page"]].append(
            {
                "label": c["label"],
                "amount": checks.plain(c["amount"]),
                "after_line": c["after_line"] - first[c["page"]],
                "page": c["page"],
            }
        )
    pages = {}
    for p in range(1, n + 1):
        try:
            issues = check_page(by_page[p], totals[p], p, evidence)
        # Fault boundary: a bug here is this page's problem; the model reads it.
        except Exception as e:
            logger.warning("checks of page %d found by code failed", p, exc_info=True)
            issues = [Issue("page_read", f"{type(e).__name__}: {e}", page=p, form="failed")]
        issues += [
            _layout_row_issue(note, p)
            for note in parsed.notes
            if note["page"] == p and note["kind"] not in HARMLESS_NOTES
        ]
        pages[p] = PageResult(p, by_page[p], totals[p], issues, method="layout")
    return pages, [note for note in parsed.notes if note["kind"] in HARMLESS_NOTES]


def _layout_row_issue(note: JSONObject, page: int) -> Issue:
    """A note of the parser about a row it placed, as an issue of the row's page."""
    kind, row = note["kind"], str(note["text"])
    return Issue(
        "layout_row",
        f"{kind}: {row}",
        page=page,
        fields=list(NOTE_FIELDS.get(kind, [])),
        # A kind of note without a sentence keeps this message.
        values=iv.LayoutRow(kind, row) if kind in get_args(iv.LayoutRowKind) else None,
    )


def compare_witness(code: list[Line], model: list[Line]) -> JSONObject:
    """Rows of one page found by code against the same page read by the model.

    Arithmetic cannot tell quantity from unit price (q x p = p x q), so a layout
    answer with the two columns swapped passes every other check. The model
    reading the page is a second opinion on exactly that. Verdict:
      confirmed  - more than half of the rows agree in sku, quantity, price, amount
      swapped    - in more than half, quantity and price are crossed
      mismatch   - neither: the two readings disagree in some other way
      unverified - the model returned no rows to compare with
    Half is the threshold because the outcomes are two far-apart clusters: a
    right layout disagrees only where the model misread a row (a few percent in
    the measurements), a wrong one disagrees on every row whose quantity differs
    from its price. The middle keeps the widest margin from both."""
    free = list(range(len(model)))
    same = crossed = 0
    for c in code:
        sku = " ".join(str(c.get("sku") or "").upper().split())
        j = next(
            (j for j in free if " ".join(str(model[j].get("sku") or "").upper().split()) == sku),
            None,
        )
        if j is None:
            continue
        free.remove(j)
        cq, cp, ca = (checks.dec(c.get(k)) for k in ("quantity", "unit_price", "amount"))
        mq, mp, ma = (checks.dec(model[j].get(k)) for k in ("quantity", "unit_price", "amount"))
        if cq is None or cp is None or ca is None or ca != ma:
            continue
        if (cq, cp) == (mq, mp):
            same += 1
        elif (cq, cp) == (mp, mq):
            crossed += 1
    n = max(len(code), len(model))
    if not model:
        verdict = "unverified"
    elif same * 2 > n:
        verdict = "confirmed"
    elif crossed * 2 > n:
        verdict = "swapped"
    else:
        verdict = "mismatch"
    return {"status": verdict, "compared": n, "same": same, "crossed": crossed}


def _page_key(pr: PageResult) -> tuple[bool, int, int]:
    """The order of read_page between attempts: a clean read, then fewer issues, then more lines."""
    return (any(i.check == "page_read" for i in pr.issues), len(pr.issues), -len(pr.lines))


# --------------------------------------------------------------------------- #
# A description that goes on at the top of the next page
# --------------------------------------------------------------------------- #


def join_carried(pages: list[PageResult]) -> list[JSONObject]:
    """Append every PageResult.carry_in to the description of the last line item of the
    previous page, in place, by the rules of carryover.join_carry: nothing is touched
    when that page ends with a control total, has no rows, was not read, or there is no
    previous page. Returns the notes for Result.notes ("line" indexes Result.lines).

    A page that kept a "page_read" issue owns nothing: its answer failed or was cut off,
    so its last row is not known to be the last row printed. A page found by code already
    has the text when the layout parser joined it across the break: it is not appended
    twice."""
    entries: list[JSONObject] = [
        {
            "page": pr.page,
            "rows": co.flow_rows(pr.lines, pr.totals),
            "carry_in": pr.carry_in,
            "unread": any(i.check == "page_read" for i in pr.issues),
            "skipped": pr.method == "skipped",
            "joined_by_code": pr.method == "layout",
        }
        for pr in pages
    ]
    joined, found = co.join_carry(entries)
    rows = {p["page"]: p["rows"] for p in joined}
    by_page = {pr.page: pr for pr in pages}
    first, done = {}, 0  # index of the first line of each page within the document
    for pr in pages:
        first[pr.page] = done
        done += len(pr.lines)
    notes: list[JSONObject] = []
    for f in found:
        note: JSONObject = {
            "page": f["page"],
            "kind": f["kind"],
            "text": f["text"],
            "source": "carry_in",
        }
        if "to_page" in f:
            # The owner is the last row of that page and a line item: owner.lines[-1].
            owner = by_page[f["to_page"]]
            if f["kind"] == co.JOINED:
                owner.lines[-1]["description"] = rows[f["to_page"]][f["row"]]["description"]
            note |= {
                "to_page": f["to_page"],
                "line": first[f["to_page"]] + len(owner.lines) - 1,
                "line_no": f["line_no"],
                "sku": f["sku"],
            }
        else:
            note["reason"] = f["reason"]
        notes.append(note)
    return notes


# --------------------------------------------------------------------------- #
# Photo quality
# --------------------------------------------------------------------------- #


def _photo_issue(r: ql.Reason, lim: ql.Limits) -> Issue:
    """A reason of quality.py as an issue. The text for the user is Reason.message
    (Result.quality); this one follows the other issues: with the number and the limit it
    was held against."""
    side = f" ({r.side} side)" if r.side else ""
    reject = r.severity == "reject"
    values: iv.IssueValues | None = None
    match r.code:
        case "no_text":
            text = (
                f"No text was found in the photo ({r.value:.0f} word-like regions, "
                f"at least {lim.min_boxes} needed)."
                if r.value is not None
                else ""
            )
        case "text_at_edge":
            text = (
                f"Text reaches the edge of the frame{side}, {r.value} text heights from it: "
                "part of the document is outside the photo."
            )
            values = iv.FrameSide(r.side)
        case "sheet_cut":
            text = f"The sheet runs out of the frame{side}; no text is near that edge."
            values = iv.FrameSide(r.side)
        case "sheet_edge_unseen":
            text = (
                "No edge of the sheet is visible, so nothing confirms that the whole document "
                "is in the photo."
            )
        case "blurred":
            limit = lim.sharpness_reject if reject else lim.sharpness_warn
            text = f"Text sharpness {r.value} is below {limit:g}." + (
                ""
                if reject
                else " Numbers are usually still read; item codes (SKUs) may be misread."
            )
            values = _measured(r, limit)
        case "dark":
            text = f"Paper brightness {r.value} of 255 is below {lim.paper_level:g}."
            values = _measured(r, lim.paper_level)
        case "glare":
            share = r.value or 0
            text = f"Glare covers {share * 100:.1f}% of the sheet."
            values = iv.Glare(share)
        case "text_small":
            text = f"Text lines are {r.value} px tall, below {lim.text_height_px:g}."
            values = _measured(r, lim.text_height_px)
        case _:
            text = ""
    text = text or r.code
    if reject:
        text += " No request was sent."
    return Issue(
        f"photo_{r.code}",
        text,
        page=1,
        severity="error" if reject else "warning",
        values=values,
    )


def _measured(r: ql.Reason, limit: float) -> iv.Measured | None:
    """The number a reason measured and its limit; none without a number (a reason that
    measures one always has it)."""
    return None if r.value is None else iv.Measured(r.value, limit)


def _photo_quality(img: Image.Image, cfg: Config) -> tuple[JSONObject, list[Issue]]:
    """Measure the photo as taken. Returns Result.quality and the issues for its reasons.
    Never raises: a photo the check cannot measure is read the way every photo was read
    before the check existed."""
    try:
        names = {f.name for f in dataclass_fields(ql.Limits)}
        lim = ql.Limits(
            **{
                k[6:]: v
                for k, v in asdict(cfg).items()
                if k.startswith("photo_") and k[6:] in names
            }
        )
        q, why = ql.check(img, lim)
        verdict = (
            "reject" if any(r.severity == "reject" for r in why) else ("warn" if why else "ok")
        )
        info = {
            "verdict": verdict,
            "reasons": [r.to_json() for r in why],
            "measures": q.to_json(),
            "limits": asdict(lim),
        }
        return info, [_photo_issue(r, lim) for r in why]
    except Exception as e:  # fault boundary: the measurement itself broke
        logger.warning("the photo could not be measured", exc_info=True)
        return {"verdict": "unchecked", "error": f"{type(e).__name__}: {e}", "reasons": []}, []


def _scan_issue(r: ql.Reason, page: int, lim: ql.ScanLimits) -> Issue:
    """A reject or warn reason of a scanned page as an issue, with the number and its limit."""
    reject = r.severity == "reject"
    values: iv.Measured | None = None
    match r.code:
        case "blurred":
            limit = lim.sharpness_reject if reject else lim.sharpness_warn
            text = f"Text sharpness {r.value} of the page is below {limit:g}."
            values = _measured(r, limit)
        case "dark":
            text = f"Paper brightness {r.value} of 255 is below {lim.paper_level:g}."
            values = _measured(r, lim.paper_level)
        case _:
            text = r.code
    if reject:
        text += " The page was not sent to the model."
    return Issue(
        f"scan_{r.code}",
        text,
        page=page,
        severity="error" if reject else "warning",
        values=values,
    )


def _too_large_issue(q: ql.TooLarge, page: int) -> Issue:
    return Issue(
        "scan_too_large",
        f"The images of the page have {q.pixels:,} pixels, more than the {q.limit:,} that may "
        "be opened. The page was not sent to the model.",
        page=page,
        severity="error",
        values=iv.TooLarge(q.pixels, q.limit),
    )


def _scan_quality(
    data: bytes, n: int, cfg: Config
) -> tuple[JSONObject, dict[int, str], dict[int, list[Issue]], list[JSONObject]]:
    """Measure every page of a scanned PDF of n pages. Returns (Result.quality, pages that
    are not sent {page: "empty" | "reject"}, issues by page, notes for Result.notes).
    Never raises: a page the check cannot measure is read the way every page was read
    before the check existed."""
    try:
        names = {f.name for f in dataclass_fields(ql.ScanLimits)}
        lim = ql.ScanLimits(
            **{k[5:]: v for k, v in asdict(cfg).items() if k.startswith("scan_") and k[5:] in names}
        )
        measured = ql.measure_pdf_pages(data, cfg.max_image_pixels)
        # The renderer and the splitter disagree about the pages: no verdict can be tied to
        # a page.
        if len(measured) != n:
            raise ValueError(f"{len(measured)} pages measured, the document has {n}")
    except Exception as e:  # fault boundary: the measurement itself broke
        logger.warning("the pages of the scan could not be measured", exc_info=True)
        error = f"{type(e).__name__}: {e}"
        return {"verdict": "unchecked", "error": error, "reasons": [], "pages": []}, {}, {}, []
    if all(q == ql.NO_IMAGE for q in measured):  # no page carries an image: not a scan of paper
        return {}, {}, {}, []
    pages: list[JSONObject] = []
    skipped: dict[int, str] = {}
    issues: dict[int, list[Issue]] = {}
    notes: list[JSONObject] = []
    for p, q in enumerate(measured, start=1):
        if isinstance(q, ql.TooLarge):  # opening it could exhaust the memory: not sent either
            reason = ql.Reason(
                "too_large",
                "The page holds an image too large to open: the page was not read. "
                "Save the scan at a lower resolution.",
                "reject",
                float(q.pixels),
            )
            pages.append({"page": p, "verdict": "reject", "reasons": [reason.to_json()]})
            issues[p] = [_too_large_issue(q, p)]
            skipped[p] = "reject"
            continue
        if isinstance(q, str):  # not a page this check can judge: read the way it always was
            pages.append({"page": p, "verdict": "unchecked", "reasons": [], "note": q})
            continue
        if isinstance(q, Exception):  # the measurement of this page broke (and logged it)
            error = f"{type(q).__name__}: {q}"
            pages.append({"page": p, "verdict": "unchecked", "reasons": [], "error": error})
            continue
        try:
            verdict, why = ql.judge_page(q, lim)
            info = {
                "page": p,
                "verdict": verdict,
                "reasons": [r.to_json() for r in why],
                "measures": q.to_json(),
            }
            found = [_scan_issue(r, p, lim) for r in why if r.severity in ("reject", "warn")]
        except Exception as e:  # fault boundary: one page
            logger.warning("page %d of the scan could not be judged", p, exc_info=True)
            error = f"{type(e).__name__}: {e}"
            pages.append({"page": p, "verdict": "unchecked", "reasons": [], "error": error})
            continue
        pages.append(info)
        if found:
            issues[p] = found
        if verdict in ("empty", "reject"):
            skipped[p] = verdict
        if verdict == "empty":
            notes.append(
                {
                    "page": p,
                    "kind": "page_without_text",
                    "text": f"No text was found on page {p}, so it was not read.",
                    "source": "scan_quality",
                }
            )
    kinds = {i["verdict"] for i in pages}
    if len(skipped) == len(pages):
        verdict = "reject"
    elif kinds & {"reject", "warn"}:
        verdict = "warn"
    else:
        verdict = "unchecked" if kinds == {"unchecked"} else "ok"
    quality = {
        "verdict": verdict,
        "reasons": [r | {"page": i["page"]} for i in pages for r in i["reasons"]],
        "pages": pages,
        "limits": asdict(lim),
    }
    return quality, skipped, issues, notes


# --------------------------------------------------------------------------- #
# Where each value is printed
# --------------------------------------------------------------------------- #


def _page_frames(data: bytes) -> list[bx.PageFrame | None]:
    """boxes.page_frames, [] when the page boxes cannot be read: nothing is framed then."""
    try:
        return bx.page_frames(data)
    except Exception:  # fault boundary: the rectangles are extra output
        logger.warning("page boxes could not be read; nothing will be framed", exc_info=True)
        return []


def _layout_boxes(
    parsed: lay.Parsed, layout: lay.Layout, frames: list[bx.PageFrame | None]
) -> dict[int, list[bx.LineBoxes | None]]:
    """The rectangles of the line items found by code, by page in the order of each page's
    lines; {} when they cannot be worked out."""
    try:
        found = bx.from_layout(parsed, layout, frames)
        by_page: dict[int, list[bx.LineBoxes | None]] = {}
        for item, b in zip(parsed.lines, found, strict=True):
            by_page.setdefault(item["page"], []).append(b)
    except Exception:  # fault boundary: the rectangles are extra output
        logger.warning("cells of the rows found by code could not be framed", exc_info=True)
        return {}
    return by_page


def _located(pages: list[PageResult], prep: Prepared) -> dict[int, list[bx.LineBoxes | None]]:
    """The lines of the pages the model read, looked up in the text layer, by page. All
    such pages at once: a page that holds one line checks it against the columns of the
    others. If that breaks, page by page: a page that breaks has no entry."""
    model = [pr for pr in pages if pr.method == "model" and pr.lines]
    if not model or prep.evidence is None or not prep.text_pages:
        return {}
    text_pages, frames = prep.text_pages, prep.frames
    try:
        found = bx.locate([item for pr in model for item in pr.lines], text_pages, frames)
        out: dict[int, list[bx.LineBoxes | None]] = {}
        for pr in model:
            out[pr.page], found = found[: len(pr.lines)], found[len(pr.lines) :]
        return out
    except Exception:  # fault boundary: all pages; tried again one by one
        logger.warning("lines could not be located together; page by page", exc_info=True)
    out = {}
    for pr in model:
        try:
            out[pr.page] = bx.locate(pr.lines, text_pages, frames)
        except Exception:  # fault boundary: one page
            logger.warning("lines of page %d could not be located", pr.page, exc_info=True)
    return out


def _page_boxes(
    pr: PageResult,
    code: dict[int, list[bx.LineBoxes | None]],
    located: dict[int, list[bx.LineBoxes | None]],
) -> list[bx.LineBoxes | None]:
    """The rectangles of the lines of one page: of its cells for a page found by code,
    looked up in the text layer for a page the model read, None for anything else."""
    found: list[bx.LineBoxes | None] = [None] * len(pr.lines)
    if pr.method == "layout" and pr.page in code:
        found = code[pr.page]
    elif pr.method == "model" and pr.page in located:
        found = located[pr.page]
    if len(found) != len(pr.lines):
        raise ValueError(f"page {pr.page}: {len(found)} boxes for {len(pr.lines)} lines")
    return found


def _line_boxes(
    pages: list[PageResult], prep: Prepared, by_code: tuple[lay.Parsed, lay.Layout] | None
) -> list[bx.LineBoxes | None]:
    """Where each line of the document is printed, parallel to the lines of `pages` taken
    in order (see _page_boxes); whatever cannot be worked out gets None - a page that
    breaks costs its own rectangles only. by_code: what the layout road read, when its
    pages were kept. Pure Python, bounded by lines x rows of a page: run on the lane."""
    total = sum(len(pr.lines) for pr in pages)
    try:
        code = _layout_boxes(*by_code, prep.frames) if by_code is not None else {}
        located = _located(pages, prep)
        out: list[bx.LineBoxes | None] = []
        for pr in pages:
            try:
                out += _page_boxes(pr, code, located)
            except Exception:  # fault boundary: one page
                logger.warning("lines of page %d could not be framed", pr.page, exc_info=True)
                out += [None] * len(pr.lines)
    except Exception:  # fault boundary: the rectangles are extra output
        logger.warning("lines could not be located on their pages", exc_info=True)
        return [None] * total
    return out


async def _frame_lines(
    pages: list[PageResult],
    prep: Prepared,
    by_code: tuple[lay.Parsed, lay.Layout] | None,
    lane: Executor | None,
    gate: Gate,
) -> list[bx.LineBoxes | None]:
    """_line_boxes on the lane, as prepare() runs (None: on the calling thread). Bounded
    by the document's deadline: when it has passed, or passes while the lane is busy or
    working, the document is kept as read and its lines are left without rectangles.

    Work already started on the lane cannot be cancelled: when the deadline passes during
    locating, that work still runs to its end and holds the lane meanwhile, so the next
    document's local work waits for it (up to 1.5 s in the review's worst case, a page of
    6,000 text lines with 3,000 lines read)."""
    total = sum(len(pr.lines) for pr in pages)
    if gate.passed():
        logger.warning("the time limit passed before the lines could be located")
        return [None] * total
    if lane is None:
        return _line_boxes(pages, prep, by_code)
    work = asyncio.get_running_loop().run_in_executor(lane, _line_boxes, pages, prep, by_code)
    try:
        return await asyncio.wait_for(work, gate.remaining())
    except TimeoutError:
        logger.warning("the time limit passed while the lines were being located")
        return [None] * total


# --------------------------------------------------------------------------- #
# Whole document
# --------------------------------------------------------------------------- #


async def read_header(
    client: httpx.AsyncClient, part: ContentPart, cfg: Config, *, gate: Gate | None = None
) -> tuple[Header, list[Attempt], bool]:
    """Read the header, a second time when the first answer lacks the total or the
    supplier. Returns (header, attempts, whether an attempt ran to its end).

    At the deadline the attempts stop and the header of the last attempt that ran to its
    end is kept ({} when there is none). A cut attempt's answer is never used: a header
    cannot be salvaged from a part of it, as page rows can."""
    if gate is None:
        gate = Gate()
    attempts: list[Attempt] = []
    header: Header = {}
    answered = False
    for _ in range(2):
        try:
            res = await _call(client, cfg.header_model, EXTRACT_HEADER, part, 4_000, 90, gate=gate)
        except DeadlinePassed:
            break
        cut = gate.cut(res)
        answer = parse_json(res.text) or {}
        attempts.append(_attempt(res, 0, 0 if answer else 1))
        if cut:
            break
        header, answered = answer, True
        if header.get("total") is not None and header.get("supplier_name"):
            break
    return header, attempts, answered


def page_ranges(pages: list[int]) -> str:
    """[2, 5, 6, 7, 12] -> "2, 5–7 and 12"."""
    return checks.ranges(pages)


def deadline_issue(deadline_s: float, unread: list[int], header_read: bool) -> Issue:
    """The issue "deadline_exceeded": the pages and the header not read."""
    return Issue(
        "deadline_exceeded",
        deadline_text(deadline_s, unread, header_read),
        values=iv.DeadlineStop(deadline_s, tuple(unread), header_read),
    )


def deadline_text(deadline_s: float | None, unread: list[int], header_read: bool) -> str:
    """The message of the issue "deadline_exceeded": the pages and the header not read."""
    text = [f"Reading stopped at the time limit of {deadline_s:g} s for one document."]
    if len(unread) == 1:
        text.append(f"Page {unread[0]} was not read.")
    elif unread:
        text.append(f"Pages {page_ranges(unread)} were not read.")
    if not header_read:
        text.append("The header (supplier, invoice number, totals) was not read.")
    if not unread and header_read:
        text.append("Every page was read; a repeated read of a page or of the header was stopped.")
    return " ".join(text)


def link_issues(pages: list[PageResult], doc: Doc, issues: list[Issue], document: range) -> None:
    """Once every issue exists, tie the issues that one wrong line amount causes to the
    line_math issue of that line. Ids are positions in `issues` (Result.issues); `document`
    holds those of the document's own checks, `pages` the pages in order.

    The lines looked at are those whose line_math issue names the amount: they get
    quantity x unit price for it. A line whose issue names only its quantity (read 1000
    times too small) keeps its amount, which is the right one.

    - follows_from: an issue of the document's arithmetic (the lines against the subtotal,
      the tax by rate, a control total, on a page or in the document) that passes with
      those amounts lists the line_math issues it needs (checks.explained_by_lines).
    - agrees_with: a line's suggested amount says so when it makes the lines add up to the
      subtotal and the amount as read does not, with the other lines in doubt either as
      read or each at quantity x unit price. A suggested quantity or unit price leaves the
      amount as it is and never does.

    Changes no verdict, message or order. Never raises: what cannot be worked out is left
    out. Its time grows with the length of the document, not with its square."""
    try:
        ids = {
            i.line: n for n, i in enumerate(issues) if i.check == "line_math" and i.line is not None
        }
        amounts: dict[int, Decimal] = {}
        for at, n in ids.items():
            q = checks.dec(doc["lines"][at].get("quantity"))
            p = checks.dec(doc["lines"][at].get("unit_price"))
            if "line.amount" in issues[n].fields and q is not None and p is not None:
                amounts[at] = ev.money(q * p)
        position = {id(i): n for n, i in enumerate(issues)}
        follows: dict[int, list[int]] = {}

        def match(found: list[tuple[Issue, list[int]]], scope: list[int], offset: int) -> None:
            # A check raises one issue per message, so the message names the issue it raised.
            by_text: dict[tuple[str, str], list[int]] = {}
            for n in scope:
                by_text.setdefault((issues[n].check, issues[n].message), []).append(n)
            for issue, needed in found:
                for n in by_text.get((issue.check, issue.message), []):
                    follows[n] = sorted(ids[at + offset] for at in needed)

        if amounts:
            match(checks.explained_by_lines(doc, amounts), list(document), 0)
            start = 0
            for pr in pages:
                end = start + len(pr.lines)
                local = {at - start: amounts[at] for at in range(start, end) if at in amounts}
                if local:
                    scope = [position[id(i)] for i in pr.issues if id(i) in position]
                    part: Doc = {"lines": pr.lines, "checkpoints": pr.totals}
                    match(checks.explained_by_lines(part, local, partial=True), scope, start)
                start = end
        agree = [
            s
            for at in checks.agree_with_subtotal(doc, amounts)
            for s in issues[ids[at]].suggestions
            if "line.amount" in s["set"]
        ]
    except Exception:  # fault boundary: a bug here costs the links, not the document
        logger.warning("issues could not be linked to the line they follow from", exc_info=True)
        return
    for n, follows_from in follows.items():
        issues[n].follows_from = follows_from
    for s in agree:
        s["agrees_with"] = "header.subtotal"


@dataclass
class Prepared:
    """A file made ready for the model: everything process() works out before its first
    request. prepare() builds it from the bytes, with all the heavy local work (pypdf,
    pdfium, Pillow, OpenCV); the reading that follows gets only this, never the bytes or
    an image."""

    mime: str
    n: int = 0  # pages
    evidence: ev.Evidence | None = None  # the text layer, when the document has its own
    # The same text layer with coordinates: set whenever evidence is (unless reading it
    # failed), for the layout road and for locating what the model read.
    text_pages: list[list[lay.TextLine]] | None = None
    # How each page of a PDF is displayed, one per page ([] for a photo, or when the page
    # boxes could not be read; None for a page without a usable box).
    frames: list[bx.PageFrame | None] = field(default_factory=list)
    layout_info: JSONObject = field(default_factory=dict)
    quality: JSONObject = field(default_factory=dict)
    photo_issues: list[Issue] = field(default_factory=list)
    # Pages of a scan that are not sent to the model: "empty" | "reject".
    skipped: dict[int, str] = field(default_factory=dict)
    scan_issues: dict[int, list[Issue]] = field(default_factory=dict)  # by page
    scan_notes: list[JSONObject] = field(default_factory=list)
    parts: list[ContentPart] = field(default_factory=list)  # one per page
    header_part: ContentPart | None = None
    # Nothing is to be sent: (issues, notes) of the failed result - a damaged file, a
    # rejected photo, a scan with no page left to read.
    failure: tuple[list[Issue], list[JSONObject]] | None = None

    @property
    def is_pdf(self) -> bool:
        return self.mime == "application/pdf"


def prepare(data: bytes, mime: str, cfg: Config) -> Prepared:
    """Open the file, measure it and cut it into the parts sent to the model. Never
    raises for a bad file: Prepared.failure says why nothing can be sent."""
    n = 0
    evidence: ev.Evidence | None = None
    text_pages: list[list[lay.TextLine]] | None = None
    frames: list[bx.PageFrame | None] = []
    layout_info: JSONObject = {"status": "off", "reason": "not a PDF"}
    quality: JSONObject = {}
    photo_issues: list[Issue] = []
    # Pages of a scan that are not sent to the model: "empty" | "reject".
    skipped: dict[int, str] = {}
    scan_issues: dict[int, list[Issue]] = {}  # by page
    scan_notes: list[JSONObject] = []
    parts: list[ContentPart] = []
    header_part: ContentPart | None = None
    try:
        if mime != "application/pdf":
            # Any other file is a photo: one page, read as an image (photo.py).
            n = 1
            img = photo.load(data)
            if cfg.photo_quality:
                # Before any request: on an unreadable photo the model does not answer
                # "nothing", it invents rows.
                quality, photo_issues = _photo_quality(img, cfg)
            if quality.get("verdict") != "reject":
                parts = [photo.part(img)]
                header_part = parts[0]
        else:
            n = page_count(data)
            frames = _page_frames(data)
            texts = ev.pdf_text_layer(data)
            layout_info["reason"] = "no text layer"
            if ev.has_text_layer(texts):  # hard checks only against a text layer
                try:
                    scanned = lay.scanned_pages(data)
                # Fault boundary: cannot tell, so read it as if the check did not exist.
                except Exception:
                    logger.warning("could not look for page images", exc_info=True)
                    scanned = []
                if scanned:  # a scan with somebody's OCR text over it is still a scan
                    layout_info["reason"] = (
                        f"page image with text over it (pages {scanned[:5]}): treated as a scan"
                    )
                else:
                    evidence = ev.build(texts, "text_layer")
                    layout_info["reason"] = "disabled"
                    try:
                        text_pages = lay.text_lines(data)
                    # Fault boundary: the layout road is an optimisation, and so is
                    # framing what the model read; without them the model reads every page
                    # and nothing is framed.
                    except Exception as e:
                        logger.warning("no text layer with coordinates", exc_info=True)
                        if cfg.use_layout:
                            layout_info["reason"] = (
                                "text layer with coordinates is not available: "
                                f"{type(e).__name__}: {e}"
                            )
            if evidence is None and cfg.scan_quality:
                # Before any request, as for a photo: a blank page costs a request for
                # nothing, an unreadable one gets rows invented.
                quality, skipped, scan_issues, scan_notes = _scan_quality(data, n, cfg)
            parts = [
                file_part(extract_pages(data, [p]), mime, f"page{p}.pdf") for p in range(1, n + 1)
            ]
            # The header is read from the first and the last page that go to the model:
            # [1, n] when none is left out.
            sent = [p for p in range(1, n + 1) if p not in skipped]
            if sent:
                header_pages = [sent[0], sent[-1]] if len(sent) > 1 else sent
                header_part = file_part(extract_pages(data, header_pages), mime, "header.pdf")
    except Exception as e:  # fault boundary: a damaged or unsupported file
        logger.warning("file cannot be read", exc_info=True)
        failure = [Issue("unreadable_file", f"{type(e).__name__}: {e}")]
        return Prepared(mime, quality=quality, failure=(failure, []))
    if quality.get("verdict") == "reject":
        if mime != "application/pdf":
            return Prepared(mime, quality=quality, failure=(photo_issues, []))
        # A scan with no page left to read: every page that has text is too blurred, or no
        # page has text at all.
        unread = [i for p in sorted(scan_issues) for i in scan_issues[p]]
        if not any(v == "reject" for v in skipped.values()):
            unread.append(
                Issue(
                    "scan_no_text",
                    "No page of the document has text. No request was sent.",
                    severity="error",
                )
            )
        return Prepared(mime, quality=quality, failure=(unread, scan_notes))
    return Prepared(
        mime,
        n=n,
        evidence=evidence,
        text_pages=text_pages,
        frames=frames,
        layout_info=layout_info,
        quality=quality,
        photo_issues=photo_issues,
        skipped=skipped,
        scan_issues=scan_issues,
        scan_notes=scan_notes,
        parts=parts,
        header_part=header_part,
    )


async def process(
    data: bytes,
    mime: str,
    client: httpx.AsyncClient,
    cfg: Config | None = None,
    on_page: Callable[[PageResult], None] | None = None,
    *,
    lane: Executor | None = None,
    request_slots: asyncio.Semaphore | None = None,
) -> Result:
    """Read one invoice file. Never raises for a bad file or a bad answer: those end
    up in Result.issues and Result.status.

    on_page is called once per page, in its final form, as soon as it is settled.

    lane runs the local work on the file (prepare: pypdf, pdfium, Pillow, OpenCV; and, once
    read, locating the lines in the text layer), so the event loop stays free while a
    document is opened, measured and framed. It must have one
    thread and be the only one that calls those libraries: pdfium must never be called
    from two threads at once. None: the work runs on the calling thread, never on the
    loop's default executor (a pool of several threads).

    request_slots limits the model requests in flight of every document that is given the
    same semaphore; each document also keeps its own limit on page requests
    (Config.concurrency). None: only that one.

    Config.deadline_s bounds the whole call, counted from its start: the wait for the lane
    and the local work count. When it passes, no request is started, a request in flight
    ends as a failed attempt (its cost is then unknown) and what was read is returned with
    the issue "deadline_exceeded". The status is "failed" when a page has no attempt that
    ran to its end (that page has a "page_read" issue saying so); otherwise it follows the
    usual rule, so a document whose every page was read is at worst "needs_review"."""
    cfg = cfg or Config()
    started = time.monotonic()
    deadline = None if cfg.deadline_s is None else started + cfg.deadline_s
    gate = Gate(request_slots, deadline)
    if lane is None:
        prep = prepare(data, mime, cfg)
    else:
        prep = await asyncio.get_running_loop().run_in_executor(lane, prepare, data, mime, cfg)
    if prep.failure is not None:
        # Nothing was sent to a model: a result the caller can show, not a crash.
        issues, notes = prep.failure
        return Result(
            status="failed",
            header={},
            lines=[],
            checkpoints=[],
            issues=issues,
            pages=[],
            evidence_source=None,
            duration_s=time.monotonic() - started,
            calls=0,
            cost_usd=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            quality=prep.quality,
            notes=notes,
        )
    return await _read(prep, client, cfg, on_page, started, gate, lane)


async def _read(
    prep: Prepared,
    client: httpx.AsyncClient,
    cfg: Config,
    on_page: Callable[[PageResult], None] | None,
    started: float,
    gate: Gate,
    lane: Executor | None = None,
) -> Result:
    """Send the parts of a prepared file to the model, check what comes back and build the
    result. Pure Python on extracted text and model answers: no file is opened here."""
    sem = asyncio.Semaphore(cfg.concurrency)
    n, evidence, text_pages, layout_info = prep.n, prep.evidence, prep.text_pages, prep.layout_info
    quality, photo_issues, skipped = prep.quality, prep.photo_issues, prep.skipped
    scan_issues, scan_notes, parts = prep.scan_issues, prep.scan_notes, prep.parts
    if prep.header_part is None:  # prepare() leaves it out only with a failure
        raise ValueError("no page of the document is sent to the model")
    header_part = prep.header_part

    # The "carry_in" field is asked for only where a previous page can exist. A photo and
    # a one-page PDF keep the request they always had, to the byte.
    carry = cfg.carry_in and prep.is_pdf and n > 1
    # A second reading by another model: only a PDF read without a text layer (a scan),
    # only the pages sent to the model, and not a page with a quality warning - it already
    # goes to a person, and the second reader was never measured on such pages.
    second_pages: list[int] = []
    second_model = cfg.second_reading_model
    if second_model is not None and prep.is_pdf and evidence is None:
        doubtful = set(scan_issues) - set(skipped)
        second_pages = [p for p in range(1, n + 1) if p not in skipped and p not in doubtful]
    # Second readings have a limit of their own, so a retry of the first reader never
    # waits behind them for the document's; they yield every shared slot to the first
    # readers (Gate.enter). They share the deadline, but not its verdict.
    second_sem = asyncio.Semaphore(cfg.concurrency)
    second_gate = Gate(gate.slots, gate.deadline)

    async def second(p: int) -> tuple[Attempt | None, list[Line] | None]:
        if second_model is None:  # second_pages is empty then
            return None, None
        try:
            return await read_second(
                client, second_model, parts[p - 1], p, n, cfg, second_sem, second_gate
            )
        except Exception:  # fault boundary: a second reading never fails the document
            logger.warning("page %d could not be read a second time", p, exc_info=True)
            return None, None

    async def one(p: int, announce: bool = True, with_carry: bool | None = None) -> PageResult:
        if p in skipped:  # a page of a scan that is blank or too blurred to read: no request
            r = PageResult(p, issues=list(scan_issues.get(p, [])), method="skipped")
        else:
            try:
                ask = carry if with_carry is None else with_carry
                r = await read_page(client, parts[p - 1], p, n, evidence, cfg, sem, ask, gate=gate)
            except Exception as e:  # fault boundary: whatever breaks while reading one page
                logger.warning("page %d could not be read", p, exc_info=True)
                issue = Issue("page_read", f"{type(e).__name__}: {e}", page=p, form="failed")
                r = PageResult(p, issues=[issue])
            # A doubtful page of a scan: read as usual, flagged with its reason.
            r.issues += scan_issues.get(p, [])
        if on_page and announce:
            on_page(r)
        return r

    layout_attempts: list[Attempt] = []
    layout_issues: list[Issue] = []
    notes: list[JSONObject] = list(scan_notes)
    read_by_code: tuple[lay.Parsed, lay.Layout] | None = None  # framed after reading
    seconds: list[tuple[Attempt | None, list[Line] | None]] = []
    if not cfg.use_layout or text_pages is None or evidence is None:
        # The second readings are tasks of their own, next to the pages: a page is settled
        # (and on_page called) without waiting for its second reading.
        reading = asyncio.gather(
            read_header(client, header_part, cfg, gate=gate),
            *(one(p) for p in range(1, n + 1)),
            *(second(p) for p in second_pages),
        )
        _starting(gate)
        first, *rest = await reading
        # gather() of awaitables of different types gives a list of their union.
        header, header_attempts, header_read = cast(tuple[Header, list[Attempt], bool], first)
        pages = cast(list[PageResult], rest[:n])
        seconds = cast(list[tuple[Attempt | None, list[Line] | None]], rest[n:])
    else:
        # The page shown to the layout model and read by the witness: the first one with
        # rows that look like items.
        wp = next((p.number for p in evidence.pages if p.item_rows), 1)

        async def witness() -> PageResult | None:
            # The witness request is not changed: EXTRACT_PAGE. It is compared with the rows
            # found by code, nothing more.
            return await one(wp, announce=False, with_carry=False) if cfg.layout_witness else None

        beginning = asyncio.gather(
            read_header(client, header_part, cfg, gate=gate),
            ask_layout(client, lay.layout_request(text_pages, wp), cfg, gate=gate),
            witness(),
        )
        _starting(gate)
        (
            (header, header_attempts, header_read),
            (raw, layout_attempts, error),
            seen,
        ) = await beginning
        by_code: dict[int, PageResult] | None = None
        if error is None:
            try:
                layout = lay.resolve(raw, text_pages[wp - 1])
                parsed = lay.parse_document(text_pages, layout)
                by_code, notes = layout_pages(parsed, n, evidence)
            except lay.LayoutError as e:  # the answer does not fit the page
                error = f"{type(e).__name__}: {e}"
            except Exception as e:  # fault boundary: a bug takes the same way out
                logger.warning("rows could not be read by the layout", exc_info=True)
                error = f"{type(e).__name__}: {e}"
            else:
                # Framed at the end, on the lane, outside the try above: a failure to frame
                # the cells must never send the document to the fallback road.
                read_by_code = parsed, layout
        if by_code is None:
            layout_info = {"status": "failed", "reason": error}
        elif seen is None:
            layout_info = {"status": "unchecked", "reason": "witness page is switched off"}
        else:
            layout_info = compare_witness(by_code[wp].lines, seen.lines)
        layout_info["witness_page"] = wp if cfg.layout_witness else None

        if by_code is None or layout_info["status"] in ("failed", "swapped", "mismatch"):
            # Fallback for the whole document: every page by the model; the witness page is
            # already read.
            notes = []
            reading_rest = asyncio.gather(
                *(one(p) for p in range(1, n + 1) if seen is None or p != wp)
            )
            _starting(gate)
            others = await reading_rest
            pages = sorted([*others, *([seen] if seen else [])], key=lambda r: r.page)
            if seen and on_page:
                on_page(seen)
            if layout_info["status"] != "failed":
                what = (
                    "quantity and unit price are crossed"
                    if layout_info["status"] == "swapped"
                    else "the rows differ"
                )
                same, compared, crossed = (
                    layout_info["same"],
                    layout_info["compared"],
                    layout_info["crossed"],
                )
                layout_issues.append(
                    Issue(
                        f"layout_{layout_info['status']}",
                        page=wp,
                        message=(
                            f"rows found by code and rows read by the model disagree on page "
                            f"{wp}: {what} ({same} of {compared} agree, {crossed} crossed); the "
                            "document was read page by page by the model"
                        ),
                        # Crossed columns: one issue for the document, about the two
                        # columns (no line), never one per row.
                        fields=list(CROSSED_COLUMNS) if layout_info["status"] == "swapped" else [],
                        values=iv.RowsCompared(same, compared, crossed),
                    )
                )
        else:
            if layout_info["status"] == "unverified":
                stopped = seen is not None and seen.deadline_unread
                why = (
                    f"the check read of page {wp} was stopped by the time limit for the "
                    "document, so the rows found by code were not confirmed"
                    if stopped
                    else f"the model returned no rows for page {wp}, so nothing confirms "
                    "which column is quantity and which is unit price"
                )
                layout_issues.append(
                    Issue(
                        "layout_unverified",
                        page=wp,
                        message=why,
                        fields=list(CROSSED_COLUMNS),
                        form="deadline" if stopped else "no_rows",
                    )
                )
            found = by_code

            async def settle(p: int) -> PageResult:
                """A page found by code stays unless it fails its checks; then the model
                reads it and the better one is kept."""
                mine = found[p]
                theirs = seen if seen is not None and p == wp else None
                if mine.issues and theirs is None:
                    theirs = await one(p, announce=False)
                if theirs is not None and mine.issues and _page_key(theirs) < _page_key(mine):
                    chosen = theirs
                else:
                    chosen = mine
                    chosen.attempts = theirs.attempts if theirs is not None else []
                    # The rows found by code are kept because the model's re-read was no
                    # better: the page still fails, as the model's attempts did.
                    chosen.stronger_model = (
                        bool(mine.issues) and theirs is not None and theirs.stronger_model
                    )
                if on_page:
                    on_page(chosen)
                return chosen

            settling = asyncio.gather(*(settle(p) for p in range(1, n + 1)))
            _starting(gate)
            pages = list(await settling)
            kept = {p.page for p in pages if p.method == "layout"}
            notes = [note for note in notes if note["page"] in kept]
    layout_info |= {
        "pages_layout": sum(1 for p in pages if p.method == "layout"),
        "pages_model": sum(1 for p in pages if p.method == "model"),
    }
    if carry:
        # Every page is read and its best attempt chosen; the lines are not concatenated
        # or checked as a document yet.
        try:
            notes = notes + join_carried(pages)
        except Exception as e:  # fault boundary: a bug here costs the tails, not the document
            logger.warning("descriptions carried over a page break were not joined", exc_info=True)
            failure = f"{type(e).__name__}: {e}"
            notes = notes + [
                {"page": None, "kind": "carry_join_failed", "text": failure, "source": "carry_in"}
            ]
    notes += [stronger_model_note(p) for p in pages if p.stronger_model]
    # After the carried text is joined: each page as the result keeps it.
    second_issues, missing = compare_second(pages, second_pages, seconds, second_model)
    if missing is not None:
        notes.append(missing)

    lines: list[Line] = []
    checkpoints: list[Checkpoint] = []
    issues: list[Issue] = []
    for pr in pages:
        offset = len(lines)
        lines += pr.lines
        checkpoints += [t | {"after_line": t["after_line"] + offset} for t in pr.totals]
        # Page issues index the page's own lines; in the Result every issue indexes Result.lines.
        for i in pr.issues:
            if i.line is not None:
                i.line += offset
        issues += pr.issues
    doc: Doc = {"header": header, "lines": lines, "checkpoints": checkpoints}
    start = len(issues)
    issues += (
        checks.check_doc_sums(doc)
        + checks.check_tax(doc)
        + checks.check_numbering(doc)
        + checks.check_checkpoints(doc)
    )
    document = range(start, len(issues))
    if evidence is not None:
        issues += checks.check_header_grounding(doc, evidence)

    issues += layout_issues + photo_issues
    unread = [p.page for p in pages if p.deadline_unread]
    if gate.hit and cfg.deadline_s is not None:  # the gate is hit only with a deadline
        issues.append(deadline_issue(cfg.deadline_s, unread, header_read))
    # Last: every other issue keeps its position (its id) with or without a second reading.
    issues += second_issues
    link_issues(pages, doc, issues, document)

    code_read = read_by_code if any(p.method == "layout" for p in pages) else None
    line_boxes = await _frame_lines(pages, prep, code_read, lane, gate)

    all_attempts = header_attempts + layout_attempts + [a for p in pages for a in p.attempts]
    all_attempts += [p.second_reading for p in pages if p.second_reading is not None]
    if unread or not lines or not header:
        status = "failed"
    else:
        status = "needs_review" if issues else "ok"
    return Result(
        status=status,
        header=header,
        lines=lines,
        checkpoints=checkpoints,
        issues=issues,
        pages=list(pages),
        evidence_source=evidence.source if evidence else None,
        duration_s=time.monotonic() - started,
        calls=len(all_attempts),
        cost_usd=sum(a.cost_usd for a in all_attempts),
        prompt_tokens=sum(a.prompt_tokens for a in all_attempts),
        completion_tokens=sum(a.completion_tokens for a in all_attempts),
        header_attempts=header_attempts,
        cost_unknown_calls=sum(1 for a in all_attempts if not a.cost_known),
        layout=layout_info,
        layout_attempts=layout_attempts,
        notes=notes,
        quality=quality,
        boxes=line_boxes,
        frames=prep.frames,
        text_layer=evidence is not None,
    )
