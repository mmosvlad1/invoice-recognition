"""A person's decisions on the issues of a read document, kept with its job.

For each issue a person either accepts one of its suggested values (`accept`, item `k`
of the issue's `suggestions`, with every cell of its `set`) or keeps the values as read
(`keep`; for an issue without suggestions it means "checked"). A decision lives with the
job, in memory, and is lost with it. It never changes what was read: the result, its
issues and the job's status stay the reading's verdict, and no check is run again. A
cell's final value is the `set` of its accepted decision, else as read; cells never
collide, because accepting a value for a cell that another accepted decision set to a
different value is refused (409 resolution_conflict).

One state is derived on every read, never stored: an issue that follows from others (a
subtotal broken by a wrong line amount) is `settled` once every issue it follows from is
accepted with the arithmetic amount, the assumption under which the flow linked them.
Taking one of those decisions back reopens it.

Issue ids are positions in the job's view, which is built once and never changes, so an
id and a suggestion's number always mean the same thing. Like the store, this is used
only from the event loop's thread; a decision is validated and written with no await in
between, so each request is atomic.
"""

import copy
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from pydantic import JsonValue

from invoice_recognition.api import messages
from invoice_recognition.api.problems import ApiError
from invoice_recognition.recognition.checks import dec

if TYPE_CHECKING:  # jobs.py and views.py import this module: at run time it needs neither
    from invoice_recognition.api.jobs import Job
    from invoice_recognition.api.views import AcceptRequest, IssueView, KeepRequest, ResultView

logger = logging.getLogger(__name__)

type Action = Literal["accept", "keep"]
type State = Literal["open", "accepted", "kept", "settled"]

# The statuses whose issues a person can decide: the document was read.
REVIEWABLE = ("ok", "needs_review")

# What an accepted item must be for the issues that follow from it to be settled: the
# flow linked them assuming each line's amount is its quantity x unit price.
SETTLING_SOURCE = "arithmetic"
SETTLING_FIELD = "line.amount"

# The cells that hold a number: two values for one of them are compared as numbers.
# Every other cell (an item code, a description, an invoice number) is text, where
# "0012" and "12" are different values.
NUMERIC_CELLS = frozenset(
    {
        "line.quantity",
        "line.unit_price",
        "line.amount",
        "line.tax_rate",
        "header.subtotal",
        "header.tax_total",
        "header.total",
    }
)


@dataclass(frozen=True)
class Decision:
    action: Action
    suggestion: int | None  # the item accepted; None for keep
    set: dict[str, JsonValue] | None  # a copy of that item's cells; None for keep
    decided_at: datetime


@dataclass(frozen=True)
class IssueState:
    """Where the review of one issue stands."""

    id: int
    state: State
    suggestion: int | None
    set: dict[str, JsonValue] | None
    decided_at: datetime | None
    settled_by: list[int]  # with state "settled": the accepted issues it follows from


def _view(job: Job) -> ResultView | None:
    """The job's read result when a person can decide its issues, else None."""
    return job.view if job.status in REVIEWABLE else None


def states(job: Job) -> list[IssueState] | None:
    """One state per issue, in id order; None unless the job is finished as ok or
    needs_review."""
    view = _view(job)
    if view is None:
        return None
    issues, decisions = view.issues, job.decisions
    out = []
    for issue in issues:
        own = decisions.get(issue.id)
        if own is not None:  # an own decision always wins over settled
            state: State = "accepted" if own.action == "accept" else "kept"
            cells = copy.deepcopy(own.set)
            out.append(IssueState(issue.id, state, own.suggestion, cells, own.decided_at, []))
            continue
        parents = issue.follows_from
        # An id outside the list (never sent by the flow) settles nothing.
        settled = bool(parents) and all(
            0 <= p < len(issues) and _settles(issues[p], decisions.get(p)) for p in parents
        )
        state = "settled" if settled else "open"
        by = list(dict.fromkeys(parents)) if settled else []  # each id once, in order
        out.append(IssueState(issue.id, state, None, None, None, by))
    return out


def _settles(parent: IssueView, decision: Decision | None) -> bool:
    """Whether the decision on `parent` settles the issues that follow from it: it
    accepts an arithmetic item that sets the line's amount."""
    if decision is None or decision.action != "accept" or decision.suggestion is None:
        return False
    item = parent.suggestions[decision.suggestion]
    cells = item.get("set")
    return (
        item.get("source") == SETTLING_SOURCE
        and isinstance(cells, dict)
        and SETTLING_FIELD in cells
    )


def decide(job: Job, issue_id: int, request: AcceptRequest | KeepRequest, at: datetime) -> None:
    """Store a person's decision on one issue, decided at `at`. The same decision again
    changes nothing (its time is kept); a different one replaces it. Raises ApiError:
    409 not_reviewable, 404 issue_not_found, 422 invalid_resolution (no item to accept),
    409 resolution_conflict (a cell another accepted decision set to another value)."""
    issues = _issues(job, issue_id)
    issue = issues[issue_id]
    if request.action == "keep":
        decision = Decision("keep", None, None, at)
    else:
        k, count = request.suggestion, len(issue.suggestions)
        if count == 0:
            raise ApiError(422, "invalid_resolution", messages.NO_SUGGESTIONS)
        if k >= count:
            detail = messages.NO_SUCH_SUGGESTION.format(
                count=count, values="value" if count == 1 else "values", k=k
            )
            raise ApiError(422, "invalid_resolution", detail)
        decision = Decision("accept", k, _cells(issue.suggestions[k]), at)
    stored = job.decisions.get(issue_id)
    if stored is not None and stored.suggestion == decision.suggestion:
        return  # the same action: only accept names an item, and keep never does
    if decision.set:
        _refuse_a_collision(job, issues, issue, decision.set)
    job.decisions[issue_id] = decision
    what = "keep" if decision.suggestion is None else f"accept {decision.suggestion}"
    logger.info("decision on job %s issue %d: %s", job.id, issue_id, what)


def undo(job: Job, issue_id: int) -> None:
    """Take back the decision on one issue; with none stored, nothing changes. Raises
    ApiError: 409 not_reviewable, 404 issue_not_found."""
    _issues(job, issue_id)
    if job.decisions.pop(issue_id, None) is not None:
        logger.info("decision on job %s issue %d: undone", job.id, issue_id)


def _issues(job: Job, issue_id: int) -> list[IssueView]:
    """The job's issues, once it is known that a person can decide `issue_id`."""
    view = _view(job)
    if view is None:
        reason = "failed" if job.status == "failed" else "unfinished"
        raise ApiError(409, "not_reviewable", messages.NOT_REVIEWABLE[reason])
    if not 0 <= issue_id < len(view.issues):
        raise ApiError(404, "issue_not_found", messages.ISSUE_NOT_FOUND)
    return view.issues


def _cells(item: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """A copy of the cells an item sets. The flow always gives a `set` object; anything
    else sets no cell."""
    cells = item.get("set")
    return copy.deepcopy(cells) if isinstance(cells, dict) else {}


type Cell = tuple[int | None, str]
"""(line index, "line.x") for a cell of a line; (None, "header.x") for the header."""


def _cell(issue: IssueView, field: str) -> Cell:
    return (issue.line_index if field.startswith("line.") else None, field)


def _refuse_a_collision(
    job: Job, issues: list[IssueView], issue: IssueView, cells: dict[str, JsonValue]
) -> None:
    """Raises 409 resolution_conflict when one of `cells` (to be set by `issue`) is a cell
    that another issue's accepted decision set to a different value."""
    wanted = {_cell(issue, f): (f, v) for f, v in cells.items()}
    for other_id, other in sorted(job.decisions.items()):
        if other_id == issue.id or other.set is None:
            continue
        for field, theirs in other.set.items():
            mine = wanted.get(_cell(issues[other_id], field))
            if mine is not None and not same(field, mine[1], theirs):
                detail = messages.resolution_conflict(other_id, field, theirs, mine[1])
                raise ApiError(409, "resolution_conflict", detail)


def same(field: str, a: JsonValue, b: JsonValue) -> bool:
    """Whether two values for the cell `field` are the same. A numeric cell compares them
    as numbers when both are numbers or numeric strings (688.68 and "688.680"); any other
    value, and every other cell, as exactly equal JSON: an item code "0012" is not "12",
    1 is not true, and "SG-1" is not "sg-1"."""
    if field in NUMERIC_CELLS:
        x, y = dec(a), dec(b)
        if x is not None and y is not None:
            return x == y
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
