"""A person's decisions on the issues of a read document: states, the one derived state
(settled), conflicts between accepted values, and what a decision never changes."""

import json
import logging
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import JsonValue, TypeAdapter, ValidationError

from invoice_recognition.api import messages, review
from invoice_recognition.api.jobs import Job, Status
from invoice_recognition.api.problems import ApiError
from invoice_recognition.api.views import (
    AcceptRequest,
    KeepRequest,
    ResolutionRequest,
    job_item,
    job_view,
    result_view,
)
from invoice_recognition.recognition.checks import Issue
from invoice_recognition.recognition.results import PageResult, Result
from invoice_recognition.shapes import JSONObject

from fakes import header_for, line

T0 = datetime(2026, 11, 27, 15, 2, 11, tzinfo=UTC)
T1 = T0 + timedelta(minutes=5)

KEEP = KeepRequest(action="keep")


def accept(k: int) -> AcceptRequest:
    return AcceptRequest(action="accept", suggestion=k)


def arithmetic(field: str, value: JsonValue, **extra: JsonValue) -> JSONObject:
    return {"set": {field: value}, "source": "arithmetic"} | extra


# Shaped like the issues of an invoice with one amount misprinted (344.34 for 688.68):
# a line_math issue with three items, a page card, and three issues of the document's
# arithmetic that follow from the line's amount. Messages are not looked at here.
LINE_MATH = Issue(
    "line_math",
    "9 x 19.13 = 172.17, but amount is 344.34",
    line=1,
    page=1,
    line_no=2,
    fields=["line.quantity", "line.unit_price", "line.amount"],
    suggestions=[
        arithmetic("line.quantity", 9),
        arithmetic("line.unit_price", 19.13),
        arithmetic("line.amount", 688.68, agrees_with="header.subtotal"),
    ],
)
COVERAGE = Issue("coverage", "page 1: 2 item rows printed, 3 extracted", page=1)
SUBTOTAL = Issue("doc_sums", "lines add up", fields=["header.subtotal"], follows_from=[0])
TAX = Issue("tax", "tax does not add up", fields=["header.tax_total"], follows_from=[0])
SECTION = Issue("checkpoints", "section subtotal", page=1, follows_from=[0])
SAMPLE = [LINE_MATH, COVERAGE, SUBTOTAL, TAX, SECTION]


def finished(issues: list[Issue], status: Status = "needs_review") -> Job:
    """A finished job whose read result carries these issues."""
    lines = [line(1) | {"page": 1}, line(2) | {"page": 1}, line(3) | {"page": 1}]
    result = Result(
        status="failed" if status == "failed" else ("ok" if not issues else "needs_review"),
        header=header_for(lines),
        lines=lines,
        checkpoints=[],
        issues=issues,
        pages=[PageResult(1, lines)],
        evidence_source=None,
        duration_s=1.0,
        calls=2,
        cost_usd=0.002,
        prompt_tokens=200,
        completion_tokens=100,
    )
    job = Job("invoice.pdf", "pdf", "application/pdf", b"%PDF-1.7 ...", pages_total=1)
    job.result, job.view, job.status = result, result_view(result), status
    return job


def table(job: Job) -> list[tuple[str, list[int]]]:
    """Each issue's (state, settled_by)."""
    found = review.states(job)
    assert found is not None
    return [(s.state, s.settled_by) for s in found]


def refused(status: int, code: str, call: object) -> ApiError:
    assert callable(call)
    with pytest.raises(ApiError) as caught:
        call()
    assert (caught.value.status, caught.value.code) == (status, code)
    return caught.value


# --------------------------------------------------------------------------- #
# States
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("status", ["queued", "processing", "failed"])
def test_a_document_that_was_not_read_has_no_review(status: Status) -> None:
    job = finished(SAMPLE, status)  # a view, even: only the status counts
    assert review.states(job) is None


def test_every_issue_starts_open_in_id_order() -> None:
    found = review.states(finished(SAMPLE))
    assert found is not None
    assert [(s.id, s.state, s.suggestion, s.set, s.decided_at) for s in found] == [
        (n, "open", None, None, None) for n in range(5)
    ]


def test_a_document_without_issues_has_an_empty_review() -> None:
    assert review.states(finished([], "ok")) == []


def test_accepting_stores_the_item_and_its_cells_keeping_stores_neither() -> None:
    job = finished(SAMPLE)
    review.decide(job, 0, accept(2), T0)
    review.decide(job, 1, KEEP, T1)  # no suggestions: "mark as checked"
    found = review.states(job)
    assert found is not None
    first, second = found[:2]
    assert (first.state, first.suggestion, first.set, first.decided_at) == (
        "accepted",
        2,
        {"line.amount": 688.68},
        T0,
    )
    assert (second.state, second.suggestion, second.set, second.decided_at) == (
        "kept",
        None,
        None,
        T1,
    )


# --------------------------------------------------------------------------- #
# Settled: the one derived state
# --------------------------------------------------------------------------- #


def test_accepting_the_arithmetic_amount_settles_what_follows_from_it() -> None:
    job = finished(SAMPLE)
    review.decide(job, 0, accept(2), T0)
    assert table(job) == [
        ("accepted", []),
        ("open", []),  # the page card follows from nothing
        ("settled", [0]),
        ("settled", [0]),
        ("settled", [0]),
    ]


@pytest.mark.parametrize("k", [0, 1], ids=["quantity", "unit-price"])
def test_accepting_the_quantity_or_the_price_settles_nothing(k: int) -> None:
    # The amount stays as read: the subtotal still does not add up.
    job = finished(SAMPLE)
    review.decide(job, 0, accept(k), T0)
    assert [s for s, _ in table(job)] == ["accepted", "open", "open", "open", "open"]


def test_keeping_the_amount_as_read_settles_nothing() -> None:
    job = finished(SAMPLE)
    review.decide(job, 0, KEEP, T0)
    assert [s for s, _ in table(job)] == ["kept", "open", "open", "open", "open"]


def test_an_amount_from_another_source_settles_nothing() -> None:
    # The flow linked the issues assuming quantity x unit price; a second reader's amount
    # is another number.
    other = Issue(
        "line_math",
        "9 x 19.13 = 172.17, but amount is 344.34",
        line=1,
        suggestions=[{"set": {"line.amount": 688.68}, "source": "second_reading"}],
    )
    job = finished([other, SUBTOTAL])
    review.decide(job, 0, accept(0), T0)
    assert [s for s, _ in table(job)] == ["accepted", "open"]


def test_an_own_decision_wins_over_settled() -> None:
    job = finished(SAMPLE)
    review.decide(job, 3, KEEP, T0)
    review.decide(job, 0, accept(2), T1)
    assert [s for s, _ in table(job)] == ["accepted", "open", "settled", "kept", "settled"]
    review.undo(job, 3)
    assert table(job)[3] == ("settled", [0])


def test_taking_the_amount_back_reopens_what_it_settled() -> None:
    job = finished(SAMPLE)
    review.decide(job, 0, accept(2), T0)
    review.undo(job, 0)
    assert [s for s, _ in table(job)] == ["open"] * 5
    review.decide(job, 0, accept(2), T1)  # and again
    assert table(job)[2:] == [("settled", [0])] * 3
    review.decide(job, 0, accept(0), T1)  # replaced by the quantity: open again
    assert [s for s, _ in table(job)][2:] == ["open"] * 3


TWO_WRONG = [
    Issue("line_math", "a", line=0, suggestions=[arithmetic("line.amount", 20)]),
    Issue(
        "line_math",
        "b",
        line=1,
        suggestions=[arithmetic("line.unit_price", 49.5), arithmetic("line.amount", 20)],
    ),
    Issue("doc_sums", "lines add up", follows_from=[0, 1]),
]


def test_an_issue_is_settled_only_when_every_issue_it_follows_from_is() -> None:
    job = finished(TWO_WRONG)
    review.decide(job, 0, accept(0), T0)
    assert table(job)[2] == ("open", [])
    review.decide(job, 1, accept(1), T0)
    assert table(job)[2] == ("settled", [0, 1])
    review.decide(job, 1, KEEP, T1)  # one parent kept: no longer
    assert table(job)[2] == ("open", [])
    review.decide(job, 1, accept(0), T1)  # one parent's price: no longer
    assert table(job)[2] == ("open", [])


@pytest.mark.parametrize("ids", [[-1], [99], [0, 99]])
def test_an_id_outside_the_issue_list_settles_nothing(ids: list[int]) -> None:
    job = finished([LINE_MATH, Issue("doc_sums", "lines add up", follows_from=ids)])
    review.decide(job, 0, accept(2), T0)
    assert table(job) == [("accepted", []), ("open", [])]
    assert job_view(job).review is not None and job_item(job).open_issue_count == 1


def test_an_issue_named_twice_settles_once() -> None:
    job = finished([LINE_MATH, Issue("doc_sums", "lines add up", follows_from=[0, 0])])
    review.decide(job, 0, accept(2), T0)
    assert table(job)[1] == ("settled", [0])


def test_the_open_count_counts_open_issues_only() -> None:
    job = finished(SAMPLE)
    reviewed = job_view(job).review
    assert reviewed is not None and reviewed.open_count == 5
    review.decide(job, 0, accept(2), T0)
    reviewed = job_view(job).review
    assert reviewed is not None and reviewed.open_count == 1  # settled is not open
    review.decide(job, 1, KEEP, T0)
    assert job_item(job).open_issue_count == 0


# --------------------------------------------------------------------------- #
# Conflicts between accepted values
# --------------------------------------------------------------------------- #


def second_reading(cells: dict[str, JsonValue], at: int = 1) -> Issue:
    return Issue(
        "second_reading",
        "the second reader read the line differently",
        line=at,
        severity="warning",
        fields=list(cells),
        suggestions=[{"set": cells, "source": "second_reading"}],
    )


def test_two_accepted_values_for_one_cell_conflict() -> None:
    job = finished([LINE_MATH, second_reading({"line.sku": "SKU-2", "line.amount": 700})])
    review.decide(job, 0, accept(2), T0)
    error = refused(409, "resolution_conflict", lambda: review.decide(job, 1, accept(0), T1))
    assert error.detail == messages.resolution_conflict(0, "line.amount", 688.68, 700)
    assert "Issue 0" in error.detail and "688.68" in error.detail and "700" in error.detail
    assert 1 not in job.decisions  # nothing written
    # The other way round too.
    review.undo(job, 0)
    review.decide(job, 1, accept(0), T0)
    refused(409, "resolution_conflict", lambda: review.decide(job, 0, accept(2), T1))


@pytest.mark.parametrize("value", [688.68, "688.680", "6.8868E2"])
def test_the_same_number_written_differently_does_not_conflict(value: JsonValue) -> None:
    job = finished([LINE_MATH, second_reading({"line.amount": value})])
    review.decide(job, 0, accept(2), T0)
    review.decide(job, 1, accept(0), T1)
    assert [s for s, _ in table(job)] == ["accepted", "accepted"]


def test_text_is_compared_exactly() -> None:
    first = second_reading({"line.sku": "SG-54066"})
    job = finished([first, second_reading({"line.sku": "sg-54066"})])
    review.decide(job, 0, accept(0), T0)
    refused(409, "resolution_conflict", lambda: review.decide(job, 1, accept(0), T1))
    job = finished([first, second_reading({"line.sku": "SG-54066"})])
    review.decide(job, 0, accept(0), T0)
    review.decide(job, 1, accept(0), T1)


@pytest.mark.parametrize("field", ["line.sku", "line.description", "line.unit"])
def test_a_text_cell_that_reads_like_a_number_is_compared_as_text(field: str) -> None:
    # An item code "0012" is not "12", though both read as the number 12.
    job = finished([second_reading({field: "0012"}), second_reading({field: "12"})])
    review.decide(job, 0, accept(0), T0)
    refused(409, "resolution_conflict", lambda: review.decide(job, 1, accept(0), T1))


@pytest.mark.parametrize(
    ("field", "theirs", "ours"),
    [
        ("line.quantity", "0012", 12),
        ("line.unit_price", "19.130", 19.13),
        ("line.tax_rate", "20%", 20),
        ("line.amount", 688.68, "688.680"),
    ],
)
def test_a_numeric_cell_of_a_line_is_compared_as_a_number(
    field: str, theirs: JsonValue, ours: JsonValue
) -> None:
    job = finished([second_reading({field: theirs}), second_reading({field: ours})])
    review.decide(job, 0, accept(0), T0)
    review.decide(job, 1, accept(0), T1)


@pytest.mark.parametrize(
    ("field", "equal"),
    [
        ("header.subtotal", True),
        ("header.tax_total", True),
        ("header.total", True),
        ("header.invoice_number", False),
        ("header.po_number", False),
    ],
)
def test_the_numeric_cells_of_the_header_only_are_compared_as_numbers(
    field: str, equal: bool
) -> None:
    job = finished([second_reading({field: "0012"}), second_reading({field: 12.0}, at=2)])
    review.decide(job, 0, accept(0), T0)
    if equal:
        review.decide(job, 1, accept(0), T1)
    else:
        refused(409, "resolution_conflict", lambda: review.decide(job, 1, accept(0), T1))


def test_the_same_column_of_two_lines_is_two_cells() -> None:
    job = finished([LINE_MATH, second_reading({"line.amount": 1.0}, at=2)])
    review.decide(job, 0, accept(2), T0)
    review.decide(job, 1, accept(0), T1)


def test_a_header_cell_is_one_cell_whichever_line_names_it() -> None:
    job = finished([second_reading({"header.total": 10}), second_reading({"header.total": 11}, 2)])
    review.decide(job, 0, accept(0), T0)
    error = refused(409, "resolution_conflict", lambda: review.decide(job, 1, accept(0), T1))
    assert "the total to 10" in error.detail


def test_a_kept_value_never_conflicts() -> None:
    job = finished([LINE_MATH, second_reading({"line.amount": 700})])
    review.decide(job, 0, KEEP, T0)
    review.decide(job, 1, accept(0), T1)


def test_another_item_of_the_same_issue_replaces_its_decision() -> None:
    job = finished(SAMPLE)
    review.decide(job, 0, accept(2), T0)
    review.decide(job, 0, accept(0), T1)
    assert job.decisions[0] == review.Decision("accept", 0, {"line.quantity": 9}, T1)


def test_an_issue_never_conflicts_with_its_own_decision() -> None:
    # Two items of one issue for the same cell: taking the second replaces the first.
    both = Issue(
        "second_reading",
        "two readings of the amount",
        line=1,
        suggestions=[arithmetic("line.amount", 20), arithmetic("line.amount", 21)],
    )
    job = finished([both])
    review.decide(job, 0, accept(0), T0)
    review.decide(job, 0, accept(1), T1)
    found = review.states(job)
    assert found is not None
    assert (found[0].state, found[0].suggestion, found[0].set) == (
        "accepted",
        1,
        {"line.amount": 21},
    )


@pytest.mark.parametrize(
    ("field", "a", "b", "equal"),
    [
        ("line.amount", 688.68, "688.680", True),
        ("line.quantity", 9, 9.0, True),
        ("line.tax_rate", "20%", 20, True),
        ("line.amount", 1, True, False),  # a number is not a boolean
        ("line.amount", 0, None, False),
        ("line.amount", None, None, True),
        ("line.amount", "n/a", "N/A", False),  # not numbers: compared exactly
        ("line.sku", "SG-1", "SG-1", True),
        ("line.sku", "SG-1", "SG-1 ", False),
        ("line.sku", "0012", "12", False),
        ("line.sku", 12, "12", False),
        ("line.description", {"a": 1, "b": 2}, {"b": 2, "a": 1}, True),
        ("line.description", [1, 2], [2, 1], False),
    ],
)
def test_same(field: str, a: JsonValue, b: JsonValue, equal: bool) -> None:
    assert review.same(field, a, b) is equal
    assert review.same(field, b, a) is equal


# --------------------------------------------------------------------------- #
# Idempotency, refusals, and what a decision never changes
# --------------------------------------------------------------------------- #


def test_the_same_decision_twice_keeps_its_time() -> None:
    job = finished(SAMPLE)
    review.decide(job, 0, accept(2), T0)
    review.decide(job, 0, accept(2), T1)
    review.decide(job, 1, KEEP, T0)
    review.decide(job, 1, KEEP, T1)
    assert (job.decisions[0].decided_at, job.decisions[1].decided_at) == (T0, T0)


def test_a_different_decision_replaces_the_stored_one() -> None:
    job = finished(SAMPLE)
    review.decide(job, 0, KEEP, T0)
    review.decide(job, 0, accept(1), T1)
    assert job.decisions[0] == review.Decision("accept", 1, {"line.unit_price": 19.13}, T1)


def test_taking_back_nothing_changes_nothing() -> None:
    job = finished(SAMPLE)
    review.decide(job, 1, KEEP, T0)
    review.undo(job, 0)
    assert list(job.decisions) == [1]


@pytest.mark.parametrize(
    ("status", "reason"),
    [("queued", "unfinished"), ("processing", "unfinished"), ("failed", "failed")],
)
def test_a_document_that_was_not_read_cannot_be_decided(status: Status, reason: str) -> None:
    job = finished(SAMPLE, status)
    error = refused(409, "not_reviewable", lambda: review.decide(job, 0, KEEP, T0))
    assert error.detail == messages.NOT_REVIEWABLE[reason]
    refused(409, "not_reviewable", lambda: review.undo(job, 0))
    assert job.decisions == {}


@pytest.mark.parametrize("issue_id", [5, 99, -1])
def test_an_unknown_issue_is_not_found(issue_id: int) -> None:
    job = finished(SAMPLE)
    refused(404, "issue_not_found", lambda: review.decide(job, issue_id, KEEP, T0))
    refused(404, "issue_not_found", lambda: review.undo(job, issue_id))
    refused(404, "issue_not_found", lambda: review.decide(finished([], "ok"), 0, KEEP, T0))


def test_nothing_to_accept_is_an_invalid_resolution() -> None:
    job = finished(SAMPLE)
    error = refused(422, "invalid_resolution", lambda: review.decide(job, 1, accept(0), T0))
    assert error.detail == messages.NO_SUGGESTIONS
    error = refused(422, "invalid_resolution", lambda: review.decide(job, 0, accept(3), T0))
    assert (
        error.detail == "This issue has 3 suggested values, numbered from 0: there is no number 3."
    )
    assert job.decisions == {}


def test_a_decision_never_changes_the_result_or_the_status() -> None:
    job = finished(SAMPLE)
    assert job.view is not None
    before = (job.view.model_dump_json(), job.status, job_item(job).issue_count, job.result)
    review.decide(job, 0, accept(2), T0)
    review.decide(job, 1, KEEP, T0)
    found = review.states(job)
    assert found is not None and found[0].set is not None
    found[0].set["line.amount"] = 1  # what a reader of the review does with it stays there
    after = (job.view.model_dump_json(), job.status, job_item(job).issue_count, job.result)
    assert after == before
    assert job.decisions[0].set == {"line.amount": 688.68}
    assert job.view.issues[0].suggestions[2]["set"] == {"line.amount": 688.68}
    shown = json.loads(job_view(job).model_dump_json())
    assert shown["result"]["lines"][1]["amount"] == line(2)["amount"]  # as read


def test_each_decision_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    job = finished(SAMPLE)
    with caplog.at_level(logging.INFO, logger="invoice_recognition.api.review"):
        review.decide(job, 0, accept(2), T0)
        review.decide(job, 0, accept(2), T1)  # nothing changes, nothing logged
        review.decide(job, 0, KEEP, T1)
        review.undo(job, 0)
        review.undo(job, 0)  # nothing to take back
    assert [r.getMessage() for r in caplog.records] == [
        f"decision on job {job.id} issue 0: accept 2",
        f"decision on job {job.id} issue 0: keep",
        f"decision on job {job.id} issue 0: undone",
    ]


# --------------------------------------------------------------------------- #
# The request
# --------------------------------------------------------------------------- #

REQUEST = TypeAdapter(ResolutionRequest)


def test_the_two_requests() -> None:
    assert REQUEST.validate_json('{"action": "accept", "suggestion": 2}') == accept(2)
    assert REQUEST.validate_json('{"action": "keep"}') == KEEP


@pytest.mark.parametrize(
    "body",
    [
        {"action": "accept"},
        {"action": "accept", "suggestion": "1"},
        {"action": "accept", "suggestion": 1.0},
        {"action": "accept", "suggestion": True},
        {"action": "accept", "suggestion": -1},
        {"action": "accept", "suggestion": None},
        {"action": "keep", "suggestion": 0},
        {"action": "keep", "note": "looked"},
        {"action": "reject"},
        {"action": "Keep"},
        {},
        [],
    ],
)
def test_anything_else_is_refused(body: JsonValue) -> None:
    with pytest.raises(ValidationError):
        REQUEST.validate_json(json.dumps(body))
    with pytest.raises(ValidationError):
        REQUEST.validate_python(body)
