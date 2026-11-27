"""What the API answers about a document: response models, and the functions that
build them from what the flow returned.

Values a model read (the header, the fields of a line, a control total) are what it
returned, not cleaned: usually a number or a string such as "8.200", sometimes null
or, in a bad answer, any other JSON. They are typed "any JSON value" and are never
coerced. Objects that pass keys through (a line, the header, a note) allow extra keys,
whose values are JSON values too. Every field is typed strictly.

Everything the flow returned goes through one sanitiser, `_json`, before any model
sees it, so that an odd value never throws a read document away: NaN and infinity
become null; numpy scalars and Decimals become plain numbers; keys become strings;
nesting deeper than 32 levels becomes its JSON text; any other object its text; and
text that cannot be written as UTF-8 (a lone surrogate, which the JSON of a model's
answer can carry as "\\ud83d") has each such character replaced by "?" (`_utf8`, also
used for every other text that enters a view). The numbers of an attempt (time, cost,
tokens, counts) are what a provider reported, so they are read tolerantly (`_number`):
a numeric string as its number, a count sent as a float rounded, anything else as not
reported. The totals are summed from the attempts' views, after that.

A view is built once, when the job finishes, and validated then: what still does not
fit fails there, inside the worker's fault boundary, and never makes a later status
request fail; a view that was built always serialises. Every field is sent, so the
serialisation schema marks each as required.
"""

import json
import logging
import math
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypedDict

import numpy as np
from pydantic import AwareDatetime, BaseModel, BeforeValidator, ConfigDict, Field, JsonValue

from invoice_recognition.api import messages
from invoice_recognition.recognition.checks import Issue, dec
from invoice_recognition.recognition.results import Result
from invoice_recognition.shapes import JSONObject, Line

if TYPE_CHECKING:  # jobs.py holds views; at run time only the annotations name them
    from invoice_recognition.api.jobs import Job, Kind, Status

logger = logging.getLogger(__name__)

AS_READ = (
    'As read: usually a number or a string such as "8.200", or null when not read; '
    "in a bad answer any JSON value. Never coerced."
)


class _View(BaseModel):
    # The API's own fields are what the flow computed: no silent conversion. A field
    # with a default is still always sent: required in the serialisation schema.
    model_config = ConfigDict(strict=True, json_schema_serialization_defaults_required=True)


class _Open(_View):
    """Keys the flow or the model added beyond the documented ones are passed through,
    validated as JSON values."""

    model_config = ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)


def _count(value: Any) -> Any:
    """A count sent as a float (12.0, or 12.4 by a provider that averages) as an int."""
    if isinstance(value, float) and math.isfinite(value):
        return round(value)
    return value


type Count = Annotated[int, BeforeValidator(_count)]


def _read(description: str = AS_READ) -> Any:
    return Field(default=None, description=description)


class HeaderView(_Open):
    supplier_name: JsonValue = _read()
    supplier_tax_id: JsonValue = _read()
    invoice_number: JsonValue = _read()
    invoice_date: JsonValue = _read()
    due_date: JsonValue = _read()
    po_number: JsonValue = _read()
    currency: JsonValue = _read()
    subtotal: JsonValue = _read()
    tax_total: JsonValue = _read()
    total: JsonValue = _read()
    tax_breakdown: JsonValue = _read(
        "As read: usually a list of {rate, base, tax}; any JSON value in a bad answer."
    )


class LineView(_Open):
    index: int = Field(description="Position of the line in the document, from 0.")
    page: int | None = Field(default=None, description="The page the line is printed on.")
    line_no: JsonValue = _read("The position number printed in front of the line, as read.")
    sku: JsonValue = _read()
    description: JsonValue = _read()
    quantity: JsonValue = _read()
    unit: JsonValue = _read()
    unit_price: JsonValue = _read()
    tax_rate: JsonValue = _read()
    amount: JsonValue = _read()


class IssueView(_View):
    id: int = Field(description="Position of the issue in the list; stable once finished.")
    code: str = Field(description="Which check found it, e.g. line_math.")
    severity: str = Field(description="error | warning.")
    message: str
    line_index: int | None = Field(
        description="The index of the line it is about; null for the document or a page."
    )
    line_no: int | None = Field(description="The position number of that line, if printed.")
    page: int | None
    fields: list[str] = Field(
        default_factory=list,
        description="What is in doubt, as qualified names: line.<key> (with line_index the "
        "cell, without it the column) or header.<key>; [] for the page or the file.",
    )
    suggestions: list[dict[str, JsonValue]] = Field(
        default_factory=list,
        description='Values that would settle the issue, each {"set": {field: value, ...}, '
        '"source": "arithmetic" | "second_reading", ...}; accepting one writes all its cells.',
    )
    follows_from: list[int] = Field(
        default_factory=list, description="Ids of the issues that explain this one."
    )


class NoteView(_Open):
    """Something a person does not need to act on. Other keys depend on the kind."""

    kind: str
    page: int | None = None
    text: str


class CheckpointView(_Open):
    """A control total printed among the lines (a page subtotal, "carried forward")."""

    label: JsonValue = _read()
    amount: JsonValue = _read()
    after_line: int = Field(description="How many line items precede it.")
    page: int | None = None


class PageView(_View):
    page: int
    method: str = Field(description="layout (rows found by code) | model | skipped.")
    line_count: int
    repaired_count: int = Field(description="Lines on the page whose quantity code corrected.")


class LayoutView(_Open):
    """How the layout road went; other keys (reason, counts) pass through."""

    status: str = Field(
        description="confirmed | swapped | mismatch | unverified | unchecked | failed | off. "
        "Treat an unknown value as text."
    )


class AttemptView(_View):
    what: Literal["header", "layout", "page", "second_reading"]
    page: int | None
    model: str
    duration_s: float
    cost_usd: float | None = Field(description="null when the provider reported no cost.")
    cost_known: bool
    prompt_tokens: Count
    completion_tokens: Count
    line_count: Count
    issue_count: Count
    error: str | None


class DetailsView(_View):
    method: str = Field(description="layout | model | mixed.")
    layout: LayoutView | None = Field(
        description="null when the file was refused before anything was sent."
    )
    evidence_source: str | None
    duration_s: float
    cost_usd: float = Field(description="Sum over the attempts whose cost was reported.")
    cost_unknown_calls: Count
    calls: Count
    prompt_tokens: Count
    completion_tokens: Count
    attempts: list[AttemptView]
    quality: dict[str, JsonValue] = Field(
        description="Measurements of the photo or of the scan pages; {} when none."
    )


class ResultView(_View):
    header: HeaderView
    lines: list[LineView]
    issues: list[IssueView]
    notes: list[NoteView]
    checkpoints: list[CheckpointView]
    pages: list[PageView]
    details: DetailsView


# --------------------------------------------------------------------------- #
# Building the view
# --------------------------------------------------------------------------- #


# Deeper than this, a value is sent as its JSON text: no invoice field nests so deep, and
# the JSON value type of the models stops at 254 levels.
MAX_DEPTH = 32


def _json(value: Any, depth: int = 0) -> Any:
    """What the flow returned, as JSON-safe data: NaN and infinity -> None, numpy scalars
    -> Python ones, Decimal -> int or float (as the flow writes numbers), a tuple -> a
    list, a key -> a string, anything nested deeper than MAX_DEPTH -> its JSON text,
    any other object -> its text, text -> text that can be written as UTF-8 (_utf8). A
    JSON value that can be written comes out as it went in."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        return _utf8(value)
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Decimal):
        if not value.is_finite():
            return None
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, dict | list | tuple):
        if depth >= MAX_DEPTH:
            return _text(value)
        if isinstance(value, dict):
            return {_key(k): _json(v, depth + 1) for k, v in value.items()}
        return [_json(v, depth + 1) for v in value]
    return _utf8(str(value))


def _utf8(text: str) -> str:
    """The text with every character UTF-8 cannot write (a lone surrogate: JSON can
    carry one as "\\ud83d", and Python then holds it) replaced by "?". Without this, a
    view holding such text would be built but could not be serialised."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return text.encode("utf-8", "replace").decode("utf-8")
    return text


def _text(value: Any) -> str | None:
    try:
        return json.dumps(value, default=str)
    except ValueError, RecursionError:  # a circular or absurdly deep value
        return None


def _key(key: Any) -> str:
    if isinstance(key, str):
        return _utf8(key)
    # As JSON writes a key: true, null, 1.5.
    if key is None or isinstance(key, bool | int | float):
        return json.dumps(key)
    return _utf8(str(key))


def _line(index: int, line: Line) -> JSONObject:
    # "index" is the API's: a key of the same name in the line does not replace it.
    clean: JSONObject = _json(line)
    return {"index": index} | {k: v for k, v in clean.items() if k != "index"}


def _issue(id_: int, issue: Issue) -> JSONObject:
    # From the issue's JSON: keys an issue does not carry (yet) take their defaults.
    raw: JSONObject = _json(issue.to_json())
    return {
        "id": id_,
        "code": raw["check"],
        "severity": raw.get("severity", "error"),
        "message": _utf8(messages.issue_text(issue)),
        "line_index": raw.get("line"),
        "line_no": raw.get("line_no"),
        "page": raw.get("page"),
        "fields": raw.get("fields", []),
        "suggestions": raw.get("suggestions", []),
        "follows_from": raw.get("follows_from", []),
    }


def _note(note: JSONObject) -> JSONObject:
    clean: JSONObject = _json(note)
    out = {"page": None} | {k: v for k, v in clean.items() if k != "line"}
    if "line" in clean:
        out["line_index"] = clean["line"]
    return out


def _plain(d: Decimal) -> str:
    return f"{d.normalize():f}"


def _repair_notes(lines: list[Line]) -> list[JSONObject]:
    """One note per line whose quantity the flow corrected ("8.200" read as 8.2)."""
    notes = []
    for index, line in enumerate(lines):
        quantity = dec(line.get("quantity"))
        if line.get("repaired") != "quantity_x1000" or quantity is None:
            continue
        text = messages.NOTE_TEXTS["quantity_repaired"].format(
            read=f"{quantity / 1000:.3f}", corrected=_plain(quantity)
        )
        notes.append(
            {
                "kind": "quantity_repaired",
                "page": line.get("page"),
                "text": text,
                "line_index": index,
                "line_no": _json(line.get("line_no")),
            }
        )
    return notes


def _number(value: Any) -> int | float | None:
    """A number as a provider reported it, read tolerantly: a number as it is, a numeric
    string ("12", "0.001") as its number; anything else (text, a bool, a list, null,
    NaN) None, not reported. A count from a string comes as a float: Count rounds it."""
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return value if math.isfinite(value) else None


def _attempt(what: str, page: int | None, a: JSONObject) -> AttemptView:
    # A count or a time not reported is 0, as the flow takes an unreported token count;
    # a cost not reported is unknown.
    cost = _number(a.get("cost_usd")) if a.get("cost_known", True) else None
    return AttemptView.model_validate(
        {
            "what": what,
            "page": page,
            "model": a.get("model"),
            "duration_s": _number(a.get("duration_s")) or 0.0,
            "cost_usd": cost,
            "cost_known": cost is not None,
            "prompt_tokens": _number(a.get("prompt_tokens")) or 0,
            "completion_tokens": _number(a.get("completion_tokens")) or 0,
            "line_count": _number(a.get("lines")) or 0,
            "issue_count": _number(a.get("issues")) or 0,
            "error": a.get("error"),
        }
    )


def _attempts(flow: JSONObject) -> list[AttemptView]:
    """Every request of the document, from the result's JSON: the header, the layout, each
    page's attempts and, when a page was read a second time by another model, that one
    request under the page's key "second_reading"."""
    out = [_attempt("header", None, a) for a in flow["header_attempts"]]
    out += [_attempt("layout", None, a) for a in flow["layout_attempts"]]
    for p in flow["pages"]:
        out += [_attempt("page", p["page"], a) for a in p["attempts"]]
        if p.get("second_reading") is not None:
            out.append(_attempt("second_reading", p["page"], p["second_reading"]))
    return out


def result_view(result: Result) -> ResultView:
    """The result of a finished job as the API shows it. Raises (pydantic's
    ValidationError) when the result holds something that is not JSON."""
    attempts = _attempts(_json(result.to_json()))
    view = {
        "header": _json(result.header),
        "lines": [_line(i, line) for i, line in enumerate(result.lines)],
        "issues": [_issue(k, issue) for k, issue in enumerate(result.issues)],
        "notes": [_note(n) for n in result.notes] + _repair_notes(result.lines),
        "checkpoints": [_json(c) for c in result.checkpoints],
        "pages": [
            {
                "page": p.page,
                "method": _utf8(p.method),
                "line_count": len(p.lines),
                "repaired_count": p.repaired,
            }
            for p in result.pages
        ],
        "details": {
            "method": _utf8(result.method),
            "layout": _json(result.layout) if result.layout else None,
            "evidence_source": _json(result.evidence_source),
            "duration_s": result.duration_s,
            # The totals are those of the attempts listed, as validated, so the two never
            # disagree, whichever requests the flow counts in its own totals.
            "cost_usd": sum(a.cost_usd or 0.0 for a in attempts),
            "cost_unknown_calls": sum(1 for a in attempts if not a.cost_known),
            "calls": len(attempts),
            "prompt_tokens": sum(a.prompt_tokens for a in attempts),
            "completion_tokens": sum(a.completion_tokens for a in attempts),
            "attempts": attempts,
            "quality": _json(result.quality),
        },
    }
    return ResultView.model_validate(view)


# --------------------------------------------------------------------------- #
# Why a job failed
# --------------------------------------------------------------------------- #


class JobError(_View):
    """The one reason a job failed, for a person; part of a successful answer."""

    code: str = Field(
        description="photo_unreadable | scan_unreadable | unreadable_file | model_unavailable "
        "| no_line_items | no_header | deadline_exceeded | internal_error."
    )
    message: str


def _error(code: str, message: str | None = None) -> JobError:
    # The flow's own message (a photo's reason) is text from the flow: made writable.
    return JobError(code=code, message=_utf8(message or messages.JOB_ERRORS[code]))


def _photo_reason(result: Result) -> str | None:
    """The message for a person of the first reason the photo was rejected for."""
    reasons = result.quality.get("reasons")
    for reason in reasons if isinstance(reasons, list) else []:
        if isinstance(reason, dict) and reason.get("severity") == "reject":
            message = reason.get("message")
            return message if isinstance(message, str) and message else None
    return None


def _scan_reason(result: Result) -> str:
    """Which pages of a scan were too blurred, which held an image too large to open and
    which had no text."""
    pages = result.quality.get("pages")
    verdicts = [p for p in pages if isinstance(p, dict)] if isinstance(pages, list) else []

    def too_large(p: JSONObject) -> bool:
        reasons = p.get("reasons")
        codes = (
            [r.get("code") for r in reasons if isinstance(r, dict)]
            if isinstance(reasons, list)
            else []
        )
        return "too_large" in codes

    rejected = [p for p in verdicts if p.get("verdict") == "reject"]
    blurred = [p["page"] for p in rejected if not too_large(p)]
    large = [p["page"] for p in rejected if too_large(p)]
    blank = [p["page"] for p in verdicts if p.get("verdict") == "empty"]
    return messages.scan_unreadable(blurred, blank, large)


def _all_failed(attempts: list[AttemptView]) -> bool:
    return bool(attempts) and all(a.error is not None for a in attempts)


def job_error(kind: str, result: Result) -> JobError | None:
    """Why a "failed" result failed; None for "ok" and "needs_review" (also when the
    deadline cut only a repeated read). kind: "pdf" | "image". The first rule that
    matches wins:

    1. the overall deadline left something unread -> deadline_exceeded;
    2. no request was sent: the file did not open -> unreadable_file; a photo was
       rejected -> photo_unreadable, with the message of the flow's first reason; no
       page of a scan was sent -> scan_unreadable, naming the pages too blurred and the
       pages without text;
    3. every request ended in an error, or no line item was read and every request
       that could bring one (layout, pages) ended in an error -> model_unavailable.
       Only the first reader's requests count: a second reader that answers while the
       first model is down must not turn "unavailable" into "no line items";
    4. no line items -> no_line_items; 5. no header -> no_header;
    6. anything else is a gap in these rules -> internal_error, logged. The job keeps
       its result, which shows what was read; a null result is only for a job whose
       reading raised."""
    if result.status != "failed":
        return None
    codes = [i.check for i in result.issues]
    if "deadline_exceeded" in codes:
        return _error("deadline_exceeded")
    attempts = [a for a in _attempts(_json(result.to_json())) if a.what != "second_reading"]
    if not attempts:
        if "unreadable_file" in codes:
            return _error("unreadable_file")
        errors = [i.check for i in result.issues if i.severity == "error"]
        if kind == "image" and any(c.startswith("photo_") for c in errors):
            return _error("photo_unreadable", _photo_reason(result))
        if kind == "pdf" and any(c.startswith("scan_") for c in errors):
            return _error("scan_unreadable", _scan_reason(result))
    reading = [a for a in attempts if a.what != "header"]
    if _all_failed(attempts) or (not result.lines and _all_failed(reading)):
        return _error("model_unavailable")
    if not result.lines:
        return _error("no_line_items")
    if not result.header:
        return _error("no_header")
    logger.error("a failed result matches no known reason; issues: %s", codes)
    return _error("internal_error")


# --------------------------------------------------------------------------- #
# A job
# --------------------------------------------------------------------------- #


class Progress(_View):
    pages_total: int = Field(description="Known at upload; 1 for an image.")
    pages_done: int = Field(
        description="Pages whose reading is settled, skipped ones included. A hint for a "
        "progress bar, not a sign that the job is finished: only a final status is."
    )


class JobItem(_View):
    """A job in the list: everything but the result."""

    id: str
    filename: str = Field(description="As the client sent it: untrusted text, never HTML.")
    kind: Literal["pdf", "image"]
    size_bytes: int
    status: Literal["queued", "processing", "ok", "needs_review", "failed"] = Field(
        description="Final statuses: ok, needs_review, failed."
    )
    created_at: AwareDatetime
    started_at: AwareDatetime | None
    finished_at: AwareDatetime | None
    progress: Progress
    error: JobError | None = Field(description="Only for failed: why.")
    line_count: int | None = Field(description="Once finished; null without a result.")
    issue_count: int | None = Field(description="Once finished; null without a result.")


class JobView(_View):
    """A job with its result (GET /invoices/{id})."""

    id: str
    filename: str = Field(description="As the client sent it: untrusted text, never HTML.")
    kind: Literal["pdf", "image"]
    size_bytes: int
    status: Literal["queued", "processing", "ok", "needs_review", "failed"] = Field(
        description="Final statuses: ok, needs_review, failed."
    )
    created_at: AwareDatetime
    started_at: AwareDatetime | None
    finished_at: AwareDatetime | None
    progress: Progress
    error: JobError | None = Field(description="Only for failed: why.")
    result: ResultView | None = Field(
        description="null until finished, and for a job whose reading raised."
    )


class _Common(TypedDict):
    """The fields JobView and JobItem share, typed so that the models check them."""

    id: str
    filename: str
    kind: Kind
    size_bytes: int
    status: Status
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    progress: Progress
    error: JobError | None


def _common(job: Job) -> _Common:
    return {
        "id": job.id,
        "filename": job.filename,
        "kind": job.kind,
        "size_bytes": job.size_bytes,
        "status": job.status,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "progress": Progress(pages_total=job.pages_total, pages_done=job.pages_done),
        "error": job.error if job.finished else None,
    }


def job_view(job: Job) -> JobView:
    return JobView(**_common(job), result=job.view if job.finished else None)


def job_item(job: Job) -> JobItem:
    view = job.view if job.finished else None
    return JobItem(
        **_common(job),
        line_count=None if view is None else len(view.lines),
        issue_count=None if view is None else len(view.issues),
    )
