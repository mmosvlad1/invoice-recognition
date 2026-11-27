"""The one reason a failed job gives, derived from what the flow returned."""

import json
from dataclasses import asdict, fields

import httpx
import pytest
from PIL import ImageFilter

from invoice_recognition.api import messages
from invoice_recognition.api.views import JobError, job_error, result_view
from invoice_recognition.recognition.checks import Issue
from invoice_recognition.recognition.results import Attempt, PageResult, Result
from invoice_recognition.shapes import JSONObject

import drawn
from fakes import Answer, Server, header_for, line, never, run


def attempt(error: str | None = None) -> Attempt:
    return Attempt("m", 1.0, 0.001, 100, 50, 1, 0, error=error)


def failed(
    *,
    issues: list[Issue] | None = None,
    pages: list[PageResult] | None = None,
    header_attempts: list[Attempt] | None = None,
    with_lines: bool = False,
    with_header: bool = False,
    status: str = "failed",
) -> Result:
    lines = [line(1) | {"page": 1}] if with_lines else []
    return Result(
        status=status,
        header=header_for(lines) if with_header else {},
        lines=lines,
        checkpoints=[],
        issues=issues or [],
        pages=pages or [],
        evidence_source=None,
        duration_s=1.0,
        calls=0,
        cost_usd=0.0,
        prompt_tokens=0,
        completion_tokens=0,
        header_attempts=header_attempts or [],
    )


def code(kind: str, result: Result) -> str | None:
    error = job_error(kind, result)
    return None if error is None else error.code


ANSWERED = [PageResult(1, attempts=[attempt()])]


def test_ok_and_needs_review_have_no_error() -> None:
    assert job_error("pdf", failed(status="ok", with_lines=True, with_header=True)) is None
    cut_retry = failed(
        status="needs_review",
        issues=[Issue("deadline_exceeded", "Reading stopped at the time limit.")],
        pages=ANSWERED,
        with_lines=True,
        with_header=True,
    )
    assert job_error("pdf", cut_retry) is None


def test_the_deadline_comes_first() -> None:
    res = failed(
        issues=[Issue("page_read", "not read", page=2), Issue("deadline_exceeded", "stopped")],
        pages=[PageResult(1, attempts=[attempt("timed out")])],
    )
    assert job_error("pdf", res) == JobError(
        code="deadline_exceeded", message=messages.JOB_ERRORS["deadline_exceeded"]
    )


def test_a_file_that_does_not_open_is_unreadable() -> None:
    res = run(b"%PDF-1.7 broken", "application/pdf", never)
    assert code("pdf", res) == "unreadable_file"
    cut = drawn.jpeg(drawn.photo())
    assert code("image", run(cut[: len(cut) // 3], "image/jpeg", never)) == "unreadable_file"


def test_a_rejected_photo_gives_the_flows_message_for_a_person() -> None:
    blurred = drawn.photo().filter(ImageFilter.GaussianBlur(3))
    res = run(drawn.jpeg(blurred), "image/jpeg", never)
    error = job_error("image", res)
    assert error is not None and error.code == "photo_unreadable"
    first = next(r for r in res.quality["reasons"] if r["severity"] == "reject")
    assert error.message == first["message"]
    assert error.message.startswith("The photo is blurred")
    assert not any(ch.isdigit() for ch in error.message)  # the numbers stay in the issues


def test_a_rejected_photo_without_a_reason_gets_a_plain_sentence() -> None:
    res = failed(issues=[Issue("photo_blurred", "Text sharpness 50 is below 140.", page=1)])
    assert job_error("image", res) == JobError(
        code="photo_unreadable", message="The photo cannot be read. Please take the photo again."
    )


def test_a_reason_whose_text_cannot_be_utf8_still_serialises() -> None:
    res = failed(issues=[Issue("photo_blurred", "blurred", page=1)])
    res.quality = {"reasons": [{"severity": "reject", "message": "Blurred \ud800 photo."}]}
    error = job_error("image", res)
    assert error is not None and error.message == "Blurred ? photo."
    assert json.loads(error.model_dump_json())["message"] == "Blurred ? photo."


@pytest.mark.parametrize(
    ("kinds", "message"),
    [
        (("blurred",), "No page of the scan could be read: page 1 is too blurred."),
        (
            ("blurred", "blurred", "blurred"),
            "No page of the scan could be read: pages 1–3 are too blurred.",
        ),
        (("blank", "blank"), "The scan has no text on any page."),
        (
            ("blurred", "blank", "blurred"),
            (
                "No page of the scan could be read: pages 1 and 3 are too blurred "
                "and page 2 has no text."
            ),
        ),
        (
            ("blank", "blurred", "blank"),
            (
                "No page of the scan could be read: page 2 is too blurred "
                "and pages 1 and 3 have no text."
            ),
        ),
    ],
)
def test_a_scan_with_nothing_worth_sending_names_its_pages(
    kinds: tuple[str, ...], message: str
) -> None:
    res = run(drawn.scanned(*kinds), "application/pdf", never)
    assert job_error("pdf", res) == JobError(code="scan_unreadable", message=message)


def test_a_scan_whose_pages_are_too_large_to_open_says_so() -> None:
    res = run(drawn.declared_image_scan(30000, 30000), "application/pdf", never)
    assert job_error("pdf", res) == JobError(
        code="scan_unreadable",
        message="No page of the scan could be read: page 1 holds an image too large to open.",
    )


def test_every_request_failing_is_model_unavailable() -> None:
    res = failed(
        header_attempts=[attempt("HTTP 401"), attempt("HTTP 401")],
        pages=[PageResult(1, attempts=[attempt("HTTP 401"), attempt("HTTP 401")])],
    )
    assert code("pdf", res) == "model_unavailable"


HEADER = header_for([line(1)])


def test_a_header_answered_while_every_page_request_failed_is_model_unavailable() -> None:
    pages: dict[int, list[Answer]] = {p: [httpx.Response(500, text="down")] for p in (1, 2)}
    res = run(drawn.scan(2), "application/pdf", Server(HEADER, pages))
    assert res.status == "failed" and res.header and not res.lines
    assert all(a.error for p in res.pages for a in p.attempts)
    assert job_error("pdf", res) == JobError(
        code="model_unavailable", message=messages.JOB_ERRORS["model_unavailable"]
    )


def test_pages_that_answered_with_no_rows_are_no_line_items() -> None:
    pages: dict[int, list[Answer]] = {p: [[]] for p in (1, 2)}
    res = run(drawn.scan(2), "application/pdf", Server(HEADER, pages))
    assert res.status == "failed" and not res.lines
    assert job_error("pdf", res) == JobError(
        code="no_line_items",
        message="The model answered, but no line items were found in the document.",
    )


class ReadTwice(Result):
    """A scan whose first model failed everywhere while the second reader answered, as
    the flow puts it in a page's JSON."""

    def to_json(self) -> JSONObject:
        out = super().to_json()
        out["pages"][0]["second_reading"] = asdict(attempt())
        return out


def test_a_second_reader_that_answers_does_not_hide_an_unavailable_model() -> None:
    plain = failed(
        header_attempts=[attempt("HTTP 503")],
        pages=[PageResult(1, attempts=[attempt("HTTP 503")])],
    )
    twice = ReadTwice(**{f.name: getattr(plain, f.name) for f in fields(Result)})
    # The view lists the second reading; the reason leaves it out.
    attempts = result_view(twice).details.attempts
    assert [(a.what, a.error is None) for a in attempts] == [
        ("header", False),
        ("page", False),
        ("second_reading", True),
    ]
    assert code("pdf", twice) == "model_unavailable"


def test_no_lines_then_no_header() -> None:
    assert code("pdf", failed(pages=ANSWERED)) == "no_line_items"
    assert code("pdf", failed(pages=ANSWERED, with_lines=True)) == "no_header"


def test_nothing_sent_without_a_known_reason_is_not_model_unavailable() -> None:
    # No request at all is not "every request failed": the guard on an empty list.
    assert code("pdf", failed()) == "no_line_items"
    assert code("image", failed()) == "no_line_items"


def test_a_failed_result_no_rule_explains_is_an_internal_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    res = failed(pages=ANSWERED, with_lines=True, with_header=True)
    assert code("pdf", res) == "internal_error"
    assert any("matches no known reason" in r.getMessage() for r in caplog.records)


def test_every_reason_has_a_sentence() -> None:
    for text in messages.JOB_ERRORS.values():
        assert text[0].isupper() and text.endswith(".")


def test_a_photo_that_was_read_is_not_a_photo_problem() -> None:
    rows = [line(1)]
    res = run(drawn.jpeg(drawn.photo()), "image/jpeg", Server(header_for(rows), {1: [rows]}))
    assert res.status in ("ok", "needs_review") and job_error("image", res) is None
