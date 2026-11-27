"""The page-by-page flow against a fake OpenRouter (httpx.MockTransport). No network, no key."""

import json

import httpx
import pytest

from invoice_recognition.recognition import pipeline as pl
from invoice_recognition.recognition.checks import Issue
from invoice_recognition.recognition.config import Config
from invoice_recognition.recognition.evidence import Evidence
from invoice_recognition.recognition.prompts import EXTRACT_HEADER, EXTRACT_PAGE
from invoice_recognition.shapes import Checkpoint, Line

from fakes import Server, codes, header_for, line, make_pdf, never, recording, run, scan, sse

# --------------------------------------------------------------------------- #
# Which attempt wins
# --------------------------------------------------------------------------- #


def test_empty_second_attempt_does_not_beat_first() -> None:
    first = [line(1), line(2, amount=99.0), line(3)]  # wrong arithmetic in one line: a retry
    res = run(scan(1), "application/pdf", Server(header_for(first), {1: [first, []]}))
    assert len(res.lines) == 3
    assert [a.lines for a in res.pages[0].attempts] == [3, 0]
    assert "line_math" in codes(res.issues)
    assert res.status == "needs_review"


def test_failed_second_attempt_does_not_beat_first_and_is_not_a_disagreement() -> None:
    first = [line(1), line(2, amount=99.0), line(3)]
    server = Server(header_for(first), {1: [first, httpx.Response(500, text="boom")]})
    res = run(scan(1), "application/pdf", server)
    assert len(res.lines) == 3
    # A request that failed returned no count to compare.
    assert "attempts_disagree" not in codes(res.issues)


def test_attempts_with_different_line_counts_on_a_scan_are_flagged() -> None:
    first = [line(1), line(2, amount=99.0), line(3)]
    second = [line(1), line(3)]  # the doubtful row is simply gone: no issues of its own
    res = run(scan(1), "application/pdf", Server(header_for(second), {1: [first, second]}))
    assert len(res.lines) == 2  # fewer issues still wins ...
    flagged = [i for i in res.issues if i.check == "attempts_disagree"]
    assert len(flagged) == 1 and flagged[0].page == 1  # ... but the page stays flagged
    assert "3" in flagged[0].message and "2" in flagged[0].message
    assert res.status == "needs_review"


def test_same_issue_count_prefers_the_attempt_with_more_lines() -> None:
    first = [line(1), line(2, amount=99.0)]
    second = [line(1), line(2, amount=99.0), line(3)]
    res = run(scan(1), "application/pdf", Server(header_for(second), {1: [first, second]}))
    assert len(res.lines) == 3


def test_page_with_only_totals_is_not_empty() -> None:
    p1 = [line(1), line(2), line(3)]
    p2 = [{"total_label": "Subtotal", "amount": 60.0}]
    res = run(scan(2), "application/pdf", Server(header_for(p1), {1: [p1], 2: [p2]}))
    assert "page_empty" not in codes(res.issues)
    assert res.checkpoints == [{"label": "Subtotal", "amount": 60.0, "after_line": 3, "page": 2}]
    assert res.status == "ok" and res.issues == []
    assert len(res.pages[1].attempts) == 1  # nothing to retry


def test_empty_page_of_a_scan_is_flagged() -> None:
    p1 = [line(1), line(2), line(3)]
    res = run(scan(2), "application/pdf", Server(header_for(p1), {1: [p1], 2: [[]]}))
    empty = [i for i in res.issues if i.check == "page_empty"]
    assert len(empty) == 1 and empty[0].page == 2
    assert len(res.pages[1].attempts) == 2  # an empty answer is retried
    assert "attempts_disagree" not in codes(res.issues)  # both attempts agree: 0 lines
    assert res.status == "needs_review"


def test_rows_without_sku_and_amount_are_counted() -> None:
    rows = [line(1), {"description": "Section A"}, line(2), "garbage"]
    res = run(scan(1), "application/pdf", Server(header_for([line(1), line(2)]), {1: [rows]}))
    dropped = [i for i in res.issues if i.check == "rows_dropped"]
    assert len(res.lines) == 2
    assert len(dropped) == 1 and dropped[0].message.startswith("2 row(s)") and dropped[0].page == 1


# --------------------------------------------------------------------------- #
# Where an issue points
# --------------------------------------------------------------------------- #


def test_issue_line_indexes_result_lines_on_later_pages() -> None:
    p1 = [line(1), line(2), line(3)]
    p2 = [line(4), line(5, amount=77.0)]
    res = run(scan(2), "application/pdf", Server(header_for(p1 + p2), {1: [p1], 2: [p2]}))
    (issue,) = [i for i in res.issues if i.check == "line_math"]
    assert issue.page == 2 and issue.line == 4
    assert res.lines[issue.line]["sku"] == "SKU-005"
    assert issue.line_no == 5
    assert issue.to_json()["line"] == 4 and issue.to_json()["line_no"] == 5
    # the per-page copy in the result uses the same coordinates
    assert res.to_json()["pages"][1]["issues"][0]["line"] == 4


# --------------------------------------------------------------------------- #
# One page failing
# --------------------------------------------------------------------------- #


def test_exception_on_one_page_does_not_fail_the_document() -> None:
    p1 = [line(1), line(2)]
    server = Server(header_for(p1), {1: [p1], 2: [RuntimeError("stream blew up")]})
    res = run(scan(2), "application/pdf", server)
    assert len(res.lines) == 2
    failed = [i for i in res.issues if i.check == "page_read"]
    assert len(failed) == 1 and failed[0].page == 2 and "stream blew up" in failed[0].message
    assert res.status == "needs_review"


def test_malformed_stream_on_one_page_does_not_fail_the_document() -> None:
    p1 = [line(1), line(2)]
    broken = 'data: {"choices": [{"delta": {"content": "{\\"ro\n\ndata: [DONE]\n\n'
    res = run(scan(2), "application/pdf", Server(header_for(p1), {1: [p1], 2: [broken]}))
    failed = [i for i in res.issues if i.check == "page_read"]
    assert len(res.lines) == 2
    assert len(failed) == 1 and failed[0].page == 2 and "malformed stream" in failed[0].message


def test_bug_inside_page_processing_does_not_fail_the_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = pl.check_page

    def flaky(
        lines: list[Line], totals: list[Checkpoint], page: int, evidence: Evidence | None
    ) -> list[Issue]:
        if page == 2:
            raise ValueError("unexpected")
        return real(lines, totals, page, evidence)

    monkeypatch.setattr(pl, "check_page", flaky)
    p1 = [line(1), line(2)]
    res = run(scan(2), "application/pdf", Server(header_for(p1), {1: [p1], 2: [[line(3)]]}))
    assert len(res.lines) == 2
    assert [(i.check, i.page) for i in res.issues if i.check == "page_read"] == [("page_read", 2)]


def test_cut_off_answer_keeps_its_complete_rows_and_is_retried() -> None:
    cut = sse('{"rows": [' + json.dumps(line(1)) + ', {"line_no": 2, "sku": "SK')
    p1 = [line(1), line(2)]
    server = Server(header_for(p1), {1: [cut, httpx.Response(500, text="boom")]})
    res = run(scan(1), "application/pdf", server)
    assert [item["sku"] for item in res.lines] == ["SKU-001"]  # the salvaged row of attempt 1
    (read,) = [i for i in res.issues if i.check == "page_read"]
    assert read.message == "answer is not valid JSON (finish=stop)"
    assert server.calls[1] == 2


# --------------------------------------------------------------------------- #
# Cost that is not known; header attempts in the JSON
# --------------------------------------------------------------------------- #


def test_unknown_cost_is_visible() -> None:
    first = httpx.Response(500, text="upstream error")
    second = [line(1), line(2)]
    res = run(scan(1), "application/pdf", Server(header_for(second), {1: [first, second]}))
    a1, a2 = res.pages[0].attempts
    assert (a1.cost_known, a1.cost_usd) == (False, 0.0) and a1.error
    assert (a2.cost_known, a2.cost_usd) == (True, 0.001)
    assert res.cost_unknown_calls == 1
    assert res.cost_usd == pytest.approx(0.002)  # header + the page attempt that reported a cost
    assert res.status == "ok"
    out = res.to_json()
    assert out["cost_unknown_calls"] == 1
    assert [a["cost_known"] for a in out["pages"][0]["attempts"]] == [False, True]
    assert len(out["header_attempts"]) == 1 and out["header_attempts"][0]["cost_known"] is True
    json.dumps(out)


# --------------------------------------------------------------------------- #
# The header
# --------------------------------------------------------------------------- #


def test_header_without_a_total_is_asked_again() -> None:
    rows = [line(1)]
    answers = iter([header_for(rows) | {"total": None}, header_for(rows)])
    pages = Server({}, {1: [rows]})

    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["messages"][0]["content"][0]["text"] == EXTRACT_HEADER:
            return httpx.Response(200, text=sse(json.dumps(next(answers))))
        return pages(request)

    res = run(scan(1), "application/pdf", handler)
    assert len(res.header_attempts) == 2 and res.header == header_for(rows)
    assert res.status == "ok" and res.calls == 3


def test_header_is_read_from_the_first_and_the_last_page() -> None:
    handler, bodies = recording(Server(header_for([line(1)]), {p: [[line(p)]] for p in (1, 2, 3)}))
    run(scan(3), "application/pdf", handler)
    files = {b["messages"][0]["content"][1]["file"]["filename"] for b in bodies}
    assert files == {"header.pdf", "page1.pdf", "page2.pdf", "page3.pdf"}
    pages = [b["messages"][0]["content"][0]["text"] for b in bodies if b is not bodies[0]]
    assert pages == [EXTRACT_PAGE.format(page=p, total=3) for p in (1, 2, 3)]
    assert bodies[0]["messages"][0]["content"][0]["text"] == EXTRACT_HEADER


def test_no_header_or_no_lines_fails_the_document() -> None:
    rows = [line(1)]
    no_header = run(scan(1), "application/pdf", Server("data: [DONE]\n\n", {1: [rows]}))
    assert no_header.status == "failed" and no_header.header == {} and len(no_header.lines) == 1
    no_lines = run(scan(1), "application/pdf", Server(header_for(rows), {1: [[]]}))
    assert no_lines.status == "failed" and no_lines.lines == []


# --------------------------------------------------------------------------- #
# A text layer
# --------------------------------------------------------------------------- #


def test_page_is_checked_against_its_text_layer() -> None:
    pdf = make_pdf([["SKU-001 Item 1 2 10.00 20.00", "SKU-002 Item 2 2 10.00 20.00"]])
    rows = [line(1), line(2, qty=3, amount=30.0)]  # misread: 3 x 10 = 30 is not printed
    res = run(pdf, "application/pdf", Server(header_for(rows), {1: [rows]}))
    assert res.evidence_source == "text_layer"
    found = {
        (i.check, i.line) for i in res.issues if i.check in ("grounding_value", "grounding_row")
    }
    assert found == {("grounding_value", 1)}
    assert len(res.pages[0].attempts) == 2  # the page failed its checks and was read again


# --------------------------------------------------------------------------- #
# Files that cannot be read
# --------------------------------------------------------------------------- #


def test_damaged_pdf_gives_failed_result_not_exception() -> None:
    res = run(b"%PDF-1.4 truncated", "application/pdf", never)
    assert res.status == "failed" and codes(res.issues) == ["unreadable_file"]
    assert res.calls == 0 and res.pages == []
    json.dumps(res.to_json())


# --------------------------------------------------------------------------- #
# The parts on their own
# --------------------------------------------------------------------------- #


def test_repair_multiplies_a_quantity_read_with_the_wrong_separator() -> None:
    lines: list[Line] = [
        {"quantity": 8.2, "unit_price": 1.5, "amount": 12300.0},  # "8.200" read as 8.2
        {"quantity": 2, "unit_price": 1.5, "amount": 3.0},  # right
        {"quantity": 2, "unit_price": 1.5, "amount": 7.0},  # wrong, and not by x1000
        {"quantity": None, "unit_price": 1.5, "amount": 3.0},
    ]
    assert pl.repair(lines) == 1
    assert lines[0] == {
        "quantity": 8200,
        "unit_price": 1.5,
        "amount": 12300.0,
        "repaired": "quantity_x1000",
    }
    assert [item["quantity"] for item in lines[1:]] == [2, 2, None]


def test_rows_to_page_places_a_total_after_the_lines_above_it() -> None:
    rows = [
        {"total_label": "Carried forward", "amount": 5},
        line(1) | {"extra": "dropped"},
        {"sku": None, "amount": None},
        line(2),
        {"total_label": "Subtotal", "amount": 45},
        None,
    ]
    lines, totals = pl.rows_to_page(rows, 3)
    assert lines == [line(1) | {"page": 3}, line(2) | {"page": 3}]
    assert totals == [
        {"label": "Carried forward", "amount": 5, "after_line": 0, "page": 3},
        {"label": "Subtotal", "amount": 45, "after_line": 2, "page": 3},
    ]


def test_check_page_names_the_printed_position_number() -> None:
    lines = [line(7) | {"page": 1}, line(8, amount=1.0) | {"page": 1}]
    (issue,) = pl.check_page(lines, [], 4, None)
    assert (issue.check, issue.line, issue.page, issue.line_no) == ("line_math", 1, 4, 8)


def test_empty_list_of_page_models_is_a_page_error_not_a_crash() -> None:
    rows = [line(1)]
    res = run(scan(1), "application/pdf", Server(header_for(rows), {}), Config(page_models=()))
    (issue,) = [i for i in res.issues if i.check == "page_read"]
    assert "page_models is empty" in issue.message and res.status == "failed"


def test_quantity_repair_is_kept_in_the_result() -> None:
    rows = [line(1, qty=2, price=1.5, amount=3000.0)]
    res = run(scan(1), "application/pdf", Server(header_for(rows), {1: [rows]}))
    assert res.lines[0]["quantity"] == 2000 and res.lines[0]["repaired"] == "quantity_x1000"
    assert res.pages[0].repaired == 1 and res.to_json()["pages"][0]["repaired"] == 1
    assert res.status == "ok"


def test_header_total_must_match_the_lines() -> None:
    rows = [line(1), line(2)]
    header = header_for(rows) | {"subtotal": 41.0}
    res = run(scan(1), "application/pdf", Server(header, {1: [rows]}))
    assert "doc_sums" in codes(res.issues) and res.status == "needs_review"
