"""The issue messages a person reads: one text for every code the flow can raise, with the
values involved. Each text is checked on an issue the flow's own code produced, so a
change of the flow's wording shows here instead of silently falling back to it."""

import ast
import inspect
import json
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import get_args

import httpx
import pytest

from invoice_recognition.api import messages
from invoice_recognition.api.views import result_view
from invoice_recognition.recognition import checks
from invoice_recognition.recognition import evidence as ev
from invoice_recognition.recognition import layout as lay
from invoice_recognition.recognition import pipeline as pl
from invoice_recognition.recognition import quality as ql
from invoice_recognition.recognition import second_reading as sr
from invoice_recognition.recognition.checks import Issue
from invoice_recognition.recognition.config import Config
from invoice_recognition.recognition.issue_values import (
    HeaderNotPrinted,
    IssueValues,
    LayoutRow,
    LayoutRowKind,
    SubtotalOff,
)
from invoice_recognition.recognition.results import Result
from invoice_recognition.shapes import Doc, JSONObject, Line

import drawn
from fakes import Answer, Server, SlowServer, header_for, line, never, run, sse
from samples import saved_layout, truth_doc

RECOGNITION = Path(pl.__file__).parent


# --------------------------------------------------------------------------- #
# Every code the flow can raise has a text, and nothing else has one
# --------------------------------------------------------------------------- #


def _calls(tree: ast.AST, name: str) -> Iterator[ast.Call]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name:
            yield node


def _calls_of(tree: ast.AST, method: str) -> Iterator[ast.Call]:
    """Calls of a method by its dotted name: "notes.append"."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == method:
            yield node


def _first_string(call: ast.Call) -> str | None:
    if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
        return call.args[0].value
    return None


def _reason_codes(function: str, severities: tuple[str, ...]) -> set[str]:
    """The codes of the Reason(...) a function of quality.py builds with those severities."""
    tree = ast.parse(inspect.getsource(getattr(ql, function)))
    found = set()
    for call in _calls(tree, "Reason"):
        severity = call.args[2]
        assert isinstance(severity, ast.Constant), ast.unparse(call)
        if severity.value in severities:
            code = _first_string(call)
            assert code is not None, ast.unparse(call)
            found.add(code)
    return found


def emitted_codes() -> set[str]:
    """Re-derived from the source of the flow: every Issue("<code>", ...) it builds, the
    photo and scan reasons that become issues photo_<code> / scan_<code>, and the two
    layout statuses that become layout_<status>."""
    literal, dynamic = set(), []
    for path in sorted(RECOGNITION.glob("*.py")):
        for call in _calls(ast.parse(path.read_text()), "Issue"):
            code = _first_string(call)
            if code is not None:
                literal.add(code)
            elif call.args:  # an f-string: listed below, checked to be the known three
                dynamic.append(ast.unparse(call.args[0]))
    assert sorted(dynamic) == [
        "f\"layout_{layout_info['status']}\"",
        "f'photo_{r.code}'",
        "f'scan_{r.code}'",
    ]
    photo = _reason_codes("reasons", ("reject", "warn"))
    # judge_page's "skip" and "info" reasons become notes, never issues (pipeline._scan_quality).
    scan = _reason_codes("judge_page", ("reject", "warn"))
    # Only a status that sends the document page by page to the model becomes an issue;
    # "failed" does not (pipeline._read).
    layout = {"layout_swapped", "layout_mismatch"}
    return literal | {f"photo_{c}" for c in photo} | {f"scan_{c}" for c in scan} | layout


def worded_codes() -> set[str]:
    """The codes that have a sentence: a sentence without values, or a shape of values
    that names the code."""
    plain = {code for code, _ in messages._PLAIN}
    return plain | {code for shape in get_args(IssueValues.__value__) for code in shape.codes}


def test_every_code_the_flow_raises_has_a_text_and_no_other_code_does() -> None:
    codes = emitted_codes()
    assert len(codes) >= 30  # the derivation found the flow's codes, not an empty set
    assert worded_codes() == codes


def test_no_text_is_read_from_the_flows_message() -> None:
    source = Path(messages.__file__).read_text()
    tree = ast.parse(source)
    imported = {
        a.name for node in ast.walk(tree) if isinstance(node, ast.Import) for a in node.names
    }
    imported |= {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not imported & {"re", "regex"}
    # One read of an issue's message: the fault boundary, which keeps it as it is.
    reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "message"
    ]
    assert [ast.unparse(node) for node in reads] == ["issue.message"]


# --------------------------------------------------------------------------- #
# Issues as the flow's code produces them
# --------------------------------------------------------------------------- #


def one(issues: list[Issue], code: str) -> Issue:
    [found] = [i for i in issues if i.check == code]
    return found


# Words of the code base that a person checking an invoice would not know.
JARGON = ("grounding", "witness", "layout", "sku", "evidence", "token", "extracted", "text layer")


def text(issue: Issue, line: Line | None = None) -> str:
    """The issue's text for a person; fails when the flow's form was not recognised (the
    flow's own message came back) or when the text uses the code base's words."""
    worded = messages.issue_text(issue, line)
    assert worded != issue.message, f"the flow's form is not recognised: {issue.message!r}"
    assert not [w for w in JARGON if w in worded.lower()], worded
    # Values or a form, never both; values only of a shape made for the issue's code.
    assert issue.form is None or issue.values is None
    assert issue.values is None or issue.check in type(issue.values).codes
    return worded


def doc(lines: list[Line], **header: object) -> Doc:
    return {"header": header, "lines": lines, "checkpoints": []}


def test_document_totals() -> None:
    nothing = checks.check_doc_sums(doc([]))
    assert [text(i) for i in nothing] == [
        "The supplier's name was not read. Check who issued the invoice.",
        "No line items were read, so the totals cannot be checked against them.",
    ]
    [no_totals] = checks.check_doc_sums(doc([line(1)], supplier_name="ACME"))
    assert text(no_totals) == (
        "The subtotal, tax total and total were not read, so the line items cannot be "
        "checked against the totals. Check them on the invoice."
    )
    [no_tax] = checks.check_doc_sums(
        doc([line(1)], supplier_name="ACME", subtotal=20.0, total=24.0)
    )
    assert no_tax.fields == ["header.tax_total"]  # which one: the issue says
    assert text(no_tax) == (
        "The tax total was not read, so the line items cannot be checked against the totals. "
        "Check it on the invoice."
    )
    lines = [line(1), line(2, amount=25.0), {"sku": "X", "amount": None}]
    found = checks.check_doc_sums(
        doc(lines, supplier_name="ACME", subtotal=40.0, tax_total=8.0, total=50.0)
    )
    assert [text(i) for i in found] == [
        (
            "Some line items have no amount, so they cannot be checked against the subtotal. "
            "Check the amounts of the lines."
        ),
        (
            "The line amounts add up to 45.0, but the invoice subtotal is 40.0 (a difference of "
            "5.0). A line may be missing, read twice or have a wrong amount: check the line "
            "amounts against the invoice."
        ),
        (
            "The subtotal 40.0 plus the tax 8.0 makes 48.0, but the invoice total is 50.0. "
            "Check these three figures on the invoice."
        ),
    ]


def test_line_arithmetic() -> None:
    lines: list[Line] = [
        {"quantity": 12, "unit_price": 6.18, "amount": 71.16},
        {"quantity": "8.200", "unit_price": 10, "amount": 82000},
        {"quantity": 2, "unit_price": None, "amount": 20},
    ]
    found = checks.check_line_math(doc(lines))
    assert [text(i, lines[i.line or 0]) for i in found] == [
        (
            "12 × 6.18 = 74.16, but the amount read is 71.16. One of the three values was "
            "misread: check the quantity, the unit price and the amount of this line."
        ),
        (
            "8.200 × 10 = 82.00, but the amount read is 82000. One of the three values was "
            "misread: check the quantity, the unit price and the amount of this line. The "
            "quantity may be 1000 times too small: a thousands separator may have been read as "
            "a decimal point."
        ),
        (
            "The quantity, the unit price or the amount of this line is missing or is not a "
            "number, so the line cannot be checked. Check it on the invoice."
        ),
    ]


def test_tax() -> None:
    lines = [line(1) | {"tax_rate": 20}, line(2) | {"tax_rate": 10}]
    breakdown = [{"rate": 20, "base": 25.0, "tax": 5.0}]
    found = checks.check_tax(doc(lines, tax_breakdown=breakdown))
    assert [text(i) for i in found] == [
        (
            "Lines worth 20.0 carry 10% tax, but the invoice's tax summary has no 10% rate. "
            "Check the tax rates of these lines and the summary."
        ),
        (
            "The lines taxed at 20% add up to 20.0, but the invoice's tax summary gives 25.0 "
            "for that rate. Check the tax rates of the lines and the summary."
        ),
    ]
    [total] = checks.check_tax(doc([line(1)], tax_total=9.0))
    assert text(total) == (
        "The tax worked out from the lines is 4.00, but the invoice's tax total is 9.0. "
        "Check the tax rates of the lines and the tax total."
    )


def test_a_tax_row_whose_rate_is_not_a_number() -> None:
    breakdown = [{"rate": 20, "base": 20.0, "tax": 4.0}, {"rate": "n/a", "base": 12.5}]
    [issue] = checks.check_tax(doc([line(1)], tax_breakdown=breakdown))
    assert issue.message == "lines at None% add up to 0, invoice says 12.5"
    assert text(issue) == (
        "The invoice's tax summary has a row whose rate could not be read as a number (base "
        "12.5). Check the tax rates in the summary."
    )


def test_numbering() -> None:
    twice = checks.check_numbering(doc([line(1), line(2), line(2), line(3), line(3)]))
    assert text(one(twice, "numbering")) == (
        "Position numbers 2 and 3 appear more than once. Lines may have been read twice: "
        "check the lines with these numbers."
    )
    [once] = checks.check_numbering(doc([line(1), line(1), line(2)]))
    assert text(once) == (
        "Position number 1 appears more than once. A line may have been read twice: check "
        "the lines with this number."
    )
    [gap] = checks.check_numbering(doc([line(1), line(3)]))
    assert text(gap) == (
        "Position number 2 is missing from the sequence. A line may have been skipped: "
        "check the invoice around it."
    )
    [gaps] = checks.check_numbering(doc([line(1), line(4), line(7)]))
    assert text(gaps) == (
        "4 position numbers are missing from the sequence: 2, 3, 5 and 6. Lines may have been "
        "skipped: check the invoice around them."
    )
    [many] = checks.check_numbering(doc([line(1), line(13)]))  # 11 missing, 10 listed
    assert text(many) == (
        "11 position numbers are missing from the sequence: for example 2, 3, 4, 5, 6, 7, 8, "
        "9, 10 and 11. Lines may have been skipped: check the invoice around them."
    )


def test_more_than_ten_repeated_numbers() -> None:
    [issue] = checks.check_numbering(doc([line(n) for n in range(1, 13) for _ in (1, 2)]))
    # The flow lists the first ten of the twelve.
    assert text(issue) == (
        "12 position numbers appear more than once, for example 1, 2, 3, 4, 5, 6, 7, 8, 9 and "
        "10. Lines may have been read twice: check the lines with these numbers."
    )


def test_control_totals() -> None:
    d = doc([line(1), line(2)]) | {
        "checkpoints": [{"label": 'Carried "forward"', "amount": 50.0, "after_line": 2}]
    }
    [issue] = checks.check_checkpoints(d)
    assert text(issue) == (
        'Line items 1–2 add up to 40.0, but the printed "Carried "forward"" is 50.0. A line '
        "in this range may be missing, read twice or misread: check them."
    )


def test_a_control_total_whose_label_has_a_line_break() -> None:
    d = doc([line(1), line(2)]) | {
        "checkpoints": [{"label": "Carried\nforward", "amount": 50.0, "after_line": 2}]
    }
    [issue] = checks.check_checkpoints(d)
    assert text(issue) == (
        'Line items 1–2 add up to 40.0, but the printed "Carried\nforward" is 50.0. A line '
        "in this range may be missing, read twice or misread: check them."
    )


PAGE = "ACME Ltd Invoice INV-7\nAB-1001 Item 2 10.00 20.00\nAB-1002 Item 3 5.00 15.00\nTotal 35.00"


def test_values_not_printed() -> None:
    evidence = ev.build([PAGE], "text_layer")
    lines = [
        {"sku": "AB-1001", "quantity": 2, "unit_price": 10, "amount": 21, "page": 1},
        {"sku": "AB-1009", "quantity": 3, "unit_price": 5, "amount": 15, "page": 1},
        {"sku": "AB-1002", "quantity": 2, "unit_price": 10, "amount": 20, "page": 1},
    ]
    found = checks.check_grounding(doc(lines), evidence)
    assert [(i.check, text(i, lines[i.line or 0])) for i in found] == [
        (
            "grounding_value",
            (
                "This line's amount (21) is not printed on the page. The model may have "
                "misread it: check it on the invoice."
            ),
        ),
        (
            "grounding_value",
            (
                "This line's item code (AB-1009) is not printed on the page. The model may "
                "have misread it: check it on the invoice."
            ),
        ),
        (
            "grounding_row",
            (
                "Each value of this line is printed on the page, but not together in one row: "
                "the line may mix values from different rows. Check it on the invoice."
            ),
        ),
    ]
    two = [{"sku": "AB-404", "quantity": 2, "unit_price": 10, "amount": 99, "page": 1}]
    [issue] = checks.check_grounding(doc(two), evidence)
    assert issue.fields == ["line.sku", "line.amount"]  # which cells: the issue says
    assert text(issue, two[0]) == (
        "This line's item code (AB-404) and amount (99) are not printed on the page. The "
        "model may have misread them: check them on the invoice."
    )
    assert text(issue) == (
        "This line's item code and amount are not printed on the page. The model may have "
        "misread them: check them on the invoice."
    )
    empty = [{"sku": "AB-1001", "quantity": "1.000", "unit_price": 7, "amount": 20, "page": 1}]
    [issue] = checks.check_grounding(doc(empty), evidence)
    assert text(issue, empty[0] | {"unit_price": None}) == (
        "This line's quantity (1.000) and unit price (empty) are not printed on the page. The "
        "model may have misread them: check them on the invoice."
    )


def test_header_values_not_printed() -> None:
    evidence = ev.build([PAGE], "text_layer")
    header = {"total": 36.0, "invoice_number": "INV-8", "po_number": "PO-1"}
    found = checks.check_header_grounding(doc([], **header), evidence)
    assert [text(i) for i in found] == [
        (
            "The total read as 36.0 is not printed anywhere in the document. It may have been "
            "misread: check it on the invoice."
        ),
        (
            "The invoice number read as INV-8 is not printed anywhere in the document. It may "
            "have been misread: check it on the invoice."
        ),
        (
            "The purchase order number read as PO-1 is not printed anywhere in the document. It "
            "may have been misread: check it on the invoice."
        ),
    ]


def test_an_identifier_with_a_tab_or_a_line_break() -> None:
    evidence = ev.build([PAGE], "text_layer")
    [tab] = checks.check_header_grounding(doc([], po_number="PO\t1"), evidence)
    assert text(tab) == (
        "The purchase order number read as PO\t1 is not printed anywhere in the document. It "
        "may have been misread: check it on the invoice."
    )
    [broken] = checks.check_header_grounding(doc([], po_number="PO\n1"), evidence)
    assert text(broken) == (
        "The purchase order number read as PO\n1 is not printed anywhere in the document. It "
        "may have been misread: check it on the invoice."
    )


# Text the model read, as it may come: with a line break, with a lone surrogate (JSON
# allows one; the view replaces it), very long.
HOSTILE = ["A\nB", "A\ud800B", "A" * 200_000]


@pytest.mark.parametrize("label", HOSTILE, ids=["line-break", "surrogate", "long"])
def test_a_control_total_label_is_quoted_as_printed(label: str) -> None:
    d = doc([line(1), line(2)]) | {
        "checkpoints": [{"label": label, "amount": 50.0, "after_line": 2}]
    }
    [issue] = checks.check_checkpoints(d)
    assert text(issue) == (
        f'Line items 1–2 add up to 40.0, but the printed "{label}" is 50.0. A line in this '
        "range may be missing, read twice or misread: check them."
    )


@pytest.mark.parametrize("value", HOSTILE, ids=["line-break", "surrogate", "long"])
def test_an_identifier_is_quoted_as_compared(value: str) -> None:
    evidence = ev.build([PAGE], "text_layer")
    [issue] = checks.check_header_grounding(doc([], invoice_number=value), evidence)
    assert text(issue) == (
        f"The invoice number read as {value.upper()} is not printed anywhere in the document. "
        "It may have been misread: check it on the invoice."
    )


def test_a_lone_surrogate_reaches_the_view_as_a_question_mark() -> None:
    d = doc([line(1), line(2)]) | {
        "checkpoints": [{"label": "A\ud800B", "amount": 50.0, "after_line": 2}]
    }
    res = Result(
        status="needs_review",
        header=header_for(d["lines"]),
        lines=d["lines"],
        checkpoints=[],
        issues=checks.check_checkpoints(d),
        pages=[],
        evidence_source="none",
        duration_s=1.0,
        calls=1,
        cost_usd=0.0,
        prompt_tokens=0,
        completion_tokens=0,
    )
    [issue] = json.loads(result_view(res).model_dump_json())["issues"]
    assert '"A?B"' in issue["message"]


def test_line_items_missing_from_a_page() -> None:
    evidence = ev.build([PAGE, PAGE], "text_layer")
    found = checks.check_coverage(doc([line(1) | {"page": 1}]), evidence)
    assert [text(i) for i in found] == [
        (
            "Page 1 seems to have 2 line items, but 1 was read. A line may be missing or read "
            "twice: check the page."
        ),
        (
            "Page 2 seems to have 2 line items, but 0 were read. A line may be missing or read "
            "twice: check the page."
        ),
    ]
    [whole] = checks.check_coverage(doc([line(1)]), evidence)
    assert text(whole) == (
        "The document seems to have 4 line items, but 1 was read. A line may be missing or "
        "read twice: check the line items."
    )
    one_row = ev.build(["ACME Ltd\nAB-1001 Item 2 10.00 20.00\nTotal 20.00"], "text_layer")
    found = checks.check_coverage(doc([line(1) | {"page": 1}, line(2) | {"page": 1}]), one_row)
    assert [text(i) for i in found] == [
        (
            "Page 1 seems to have 1 line item, but 2 were read. A line may be missing or read "
            "twice: check the page."
        )
    ]


def test_a_page_of_a_pdf_whose_reading_has_no_lines() -> None:
    # The model returned no rows for page 2 of a PDF with a text layer: the page's own
    # checks count its printed rows against none, and the issue is the page's.
    [issue] = pl.check_page([], [], 2, ev.build([PAGE, PAGE], "text_layer"))
    assert issue.page == 2
    assert text(issue) == (
        "The document seems to have 2 line items, but 0 were read. A line may be missing or "
        "read twice: check the line items."
    )


# --------------------------------------------------------------------------- #
# Pages read by the model (fake server)
# --------------------------------------------------------------------------- #


def scan_result(pages: dict[int, list[Answer]], header: list[Line] | None = None) -> Result:
    rows = header if header is not None else [line(1)]
    return run(drawn.scan(len(pages)), "application/pdf", Server(header_for(rows), pages))


def test_a_page_with_nothing_read() -> None:
    res = scan_result({1: [[line(1)]], 2: [[]]})
    assert text(one(res.issues, "page_empty")) == (
        "No line items were read from page 2. If the page has line items, they are missing: "
        "check it."
    )


def test_rows_left_out() -> None:
    res = scan_result({1: [[line(1), {"description": "Section A"}]]})
    assert text(one(res.issues, "rows_dropped")) == (
        "1 row read from page 1 had neither an item code nor an amount and was left out. If it "
        "is a line item, it is missing: check the page."
    )
    res = scan_result({1: [[line(1), {"description": "A"}, "junk"]]})
    assert text(one(res.issues, "rows_dropped")) == (
        "2 rows read from page 1 had neither an item code nor an amount and were left out. "
        "If any of them is a line item, it is missing: check the page."
    )


def test_two_readings_that_disagree() -> None:
    first = [line(1), line(2, amount=99.0), line(3)]
    second = [line(1), line(3)]
    res = scan_result({1: [first, second]}, header=second)
    assert text(one(res.issues, "attempts_disagree")) == (
        "Page 1 was read twice, and the readings found different numbers of line items (3 "
        "and 2). A line may be missing or read twice: check the page."
    )


PAGE_NOT_READ_IN_FULL = (
    "could not be read in full: the model's answer was broken or every request failed. Line "
    "items from this page may be missing: check the page."
)


def test_a_page_that_could_not_be_read() -> None:
    res = scan_result({1: [[line(1)]], 2: [httpx.Response(500, text="boom")]})
    assert text(one(res.issues, "page_read")) == f"Page 2 {PAGE_NOT_READ_IN_FULL}"


def test_a_page_whose_answer_was_cut_off_does_not_claim_its_lines_are_missing() -> None:
    rows = [line(1), line(2), line(3)]
    cut = '{"rows": [' + ", ".join(json.dumps(r) for r in rows) + ', {"sku": "SKU-004", "desc'
    res = scan_result({1: [sse(cut)]}, header=rows)
    [issue] = [i for i in res.issues if i.check == "page_read"]
    view = json.loads(result_view(res).model_dump_json())
    assert [item["sku"] for item in view["lines"]] == ["SKU-001", "SKU-002", "SKU-003"]
    worded = text(issue)
    assert worded == f"Page 1 {PAGE_NOT_READ_IN_FULL}"
    assert "are missing" not in worded and "by hand" not in worded


def test_a_page_whose_reading_broke(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: object, **kwargs: object) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr(pl, "read_page", broken)
    res = scan_result({1: [[line(1)]]})
    assert text(one(res.issues, "page_read")) == f"Page 1 {PAGE_NOT_READ_IN_FULL}"


def test_a_page_found_by_code_whose_checks_broke(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: object, **kwargs: object) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr(pl, "check_page", broken)
    pages, _ = pl.layout_pages(lay.Parsed([], [], []), 1, ev.build([PAGE], "text_layer"))
    assert text(one(pages[1].issues, "page_read")) == f"Page 1 {PAGE_NOT_READ_IN_FULL}"


def test_pages_cut_by_the_deadline() -> None:
    unread = pl.deadline_unread_issue(3, 0)
    assert text(unread) == (
        "Page 3 was not read: reading the document took longer than allowed. Its line items "
        "are missing; upload the document again to read it in full."
    )
    cut = pl.deadline_unread_issue(3, 1)
    assert text(cut) == (
        "Page 3 was not read to the end: reading the document took longer than allowed. The "
        "1 line item read from it may be incomplete: check the page."
    )
    many = pl.deadline_unread_issue(3, 4)
    assert "The 4 line items read from it may be incomplete" in text(many)


@pytest.mark.parametrize(
    ("unread", "header_read", "worded"),
    [
        (
            [2, 3, 4],
            True,
            (
                "Reading was stopped after 240 seconds, the time allowed for one document. Pages "
                "2–4 were not read."
            ),
        ),
        (
            [1],
            False,
            (
                "Reading was stopped after 240 seconds, the time allowed for one document. Page 1 "
                "was not read. The header (supplier, invoice number, totals) was not read."
            ),
        ),
        (
            [],
            True,
            (
                "Reading was stopped after 240 seconds, the time allowed for one document. Every "
                "page was read; only a second reading of a page or of the header was cut short."
            ),
        ),
        (
            [],
            False,
            (
                "Reading was stopped after 240 seconds, the time allowed for one document. The "
                "header (supplier, invoice number, totals) was not read."
            ),
        ),
        (
            [2, 5],
            False,
            (
                "Reading was stopped after 240 seconds, the time allowed for one document. Pages "
                "2 and 5 were not read. The header (supplier, invoice number, totals) was not "
                "read."
            ),
        ),
    ],
)
def test_the_deadline(unread: list[int], header_read: bool, worded: str) -> None:
    assert text(pl.deadline_issue(240.0, unread, header_read)) == worded


def test_a_deadline_of_a_million_seconds() -> None:
    assert text(pl.deadline_issue(1e6, [2], True)) == (
        "Reading was stopped after 1e+06 seconds, the time allowed for one document. Page 2 "
        "was not read."
    )


# --------------------------------------------------------------------------- #
# A PDF whose rows were taken from its text
# --------------------------------------------------------------------------- #


def layout_result(name: str, pages: dict[int, list[Answer]], layout: Answer) -> Result:
    data, _, header, truth_pages = truth_doc(name)
    answers: dict[int, list[Answer]] = {p: [rows] for p, rows in truth_pages.items()}
    return run(data, "application/pdf", Server(header, answers | pages, layout))


def swapped(layout: JSONObject) -> JSONObject:
    """The layout with the quantity and unit price columns exchanged."""
    flip = {"quantity": "unit_price", "unit_price": "quantity"}
    return layout | {
        "columns": [c | {"field": flip.get(c["field"], c["field"])} for c in layout["columns"]]
    }


def test_crossed_columns() -> None:
    res = layout_result("l_classic", {}, swapped(saved_layout("l_classic")))
    issue = one(res.issues, "layout_swapped")
    assert text(issue) == (
        f"On page 1, the line items taken from the text of the PDF and those read by the "
        f"model disagree: quantity and unit price look swapped ({res.layout['crossed']} of "
        f"{res.layout['compared']} lines). The model's reading of every page was used "
        "instead: check the quantities and unit prices."
    )


def test_rows_that_differ() -> None:
    _, _, _, pages = truth_doc("l_classic")
    wrong = [r | {"quantity": r["quantity"] + 1} for r in pages[1]]
    res = layout_result("l_classic", {1: [wrong]}, saved_layout("l_classic"))
    assert text(one(res.issues, "layout_mismatch")) == (
        f"On page 1, the line items taken from the text of the PDF and those read by the "
        f"model differ (only {res.layout['same']} of {res.layout['compared']} lines agree). "
        "The model's reading of every page was used instead: check the line items."
    )


def test_rows_not_confirmed() -> None:
    failing: dict[int, list[Answer]] = {1: [httpx.Response(500, text="boom")]}
    res = layout_result("l_classic", failing, saved_layout("l_classic"))
    assert text(one(res.issues, "layout_unverified")) == (
        "The line items were taken from the text of the PDF, but the control reading of page "
        "1 returned no line items, so nothing confirms which column is the quantity and which "
        "the unit price. Check that they are not swapped."
    )
    data, _, header, pages = truth_doc("s_classic")
    answers: dict[int, list[Answer]] = {p: [rows] for p, rows in pages.items()}
    slow = SlowServer(Server(header, answers, saved_layout("s_classic")), {1: [30.0]})
    stopped = run(data, "application/pdf", slow, Config(deadline_s=0.5))
    worded = text(one(stopped.issues, "layout_unverified"))
    assert "the control reading of page 1 was stopped by the time limit, so" in worded


@pytest.mark.parametrize(
    ("kind", "worded"),
    [
        (
            "stray_words",
            (
                "On page 1, the row of a line item also holds words outside the table's columns: "
                '"AB-1001 Item 2 10.00 20.00 urgent". Check that this line was read in full.'
            ),
        ),
        (
            "no_tax_rate",
            (
                'On page 1, no tax rate was found in the row "AB-1001 Item 2 10.00 20.00 urgent", '
                "though the table has a tax rate column. Check the tax rate of this line."
            ),
        ),
        (
            "text_under_item",
            (
                'On page 1, the text "AB-1001 Item 2 10.00 20.00 urgent" under a line item was not '
                "added to it. Check whether it belongs to the description of that line."
            ),
        ),
    ],
)
def test_rows_the_layout_could_not_place(kind: str, worded: str) -> None:
    row = "AB-1001 Item 2 10.00 20.00 urgent"
    parsed = lay.Parsed([], [], [{"page": 1, "row": 3, "kind": kind, "text": row}])
    pages, _ = pl.layout_pages(parsed, 1, ev.build([PAGE], "text_layer"))
    assert text(one(pages[1].issues, "layout_row")) == worded


def test_every_kind_of_row_the_parser_notes_has_a_sentence_or_is_harmless() -> None:
    # Every note layout.py makes: notes.append({..., "kind": "<kind>", ...}).
    kinds = set()
    for call in _calls_of(ast.parse(inspect.getsource(lay)), "notes.append"):
        [note] = call.args
        assert isinstance(note, ast.Dict), ast.unparse(call)
        named = dict(zip(note.keys, note.values, strict=True))
        [kind] = [v for k, v in named.items() if isinstance(k, ast.Constant) and k.value == "kind"]
        assert isinstance(kind, ast.Constant), ast.unparse(call)
        kinds.add(kind.value)
    assert set(pl.HARMLESS_NOTES) < kinds
    assert kinds - set(pl.HARMLESS_NOTES) == set(get_args(LayoutRowKind))


def layout_row(kind: str, row: str) -> Issue:
    parsed = lay.Parsed([], [], [{"page": 1, "row": 3, "kind": kind, "text": row}])
    pages, _ = pl.layout_pages(parsed, 1, ev.build([PAGE], "text_layer"))
    return one(pages[1].issues, "layout_row")


@pytest.mark.parametrize("row", HOSTILE, ids=["line-break", "surrogate", "long"])
def test_a_row_is_quoted_as_printed(row: str) -> None:
    assert text(layout_row("stray_words", row)) == (
        "On page 1, the row of a line item also holds words outside the table's columns: "
        f'"{row}". Check that this line was read in full.'
    )


def test_a_kind_of_row_without_a_sentence_keeps_the_flows_message() -> None:
    issue = layout_row("a_new_kind", "x y")
    assert messages.issue_text(issue) == "a_new_kind: x y"


# --------------------------------------------------------------------------- #
# The file, the photo, the scan
# --------------------------------------------------------------------------- #


def test_a_file_that_does_not_open() -> None:
    res = run(b"%PDF-1.7 broken", "application/pdf", never)
    assert text(one(res.issues, "unreadable_file")) == messages.JOB_ERRORS["unreadable_file"]


def test_a_scan_with_no_text_at_all() -> None:
    res = run(drawn.scanned("blank", "blank"), "application/pdf", never)
    assert text(one(res.issues, "scan_no_text")) == (
        "No page of the scan has any text, so nothing was read. Check that this is the right file."
    )


def test_a_page_read_twice_by_two_models() -> None:
    def item(no: int, **change: object) -> Line:
        return line(no) | {"sku": f"HB-{no:05d}"} | change

    [row] = sr.compare_page(
        [item(1), item(2)], [item(1), item(2, sku="HB-0O002", quantity=3, amount=30)], 4
    )
    assert text(row) == (
        "Position 2 on page 4 was read differently by a second model: item code HB-0O002 "
        "instead of HB-00002, quantity 3 instead of 2 and amount 30 instead of 20. One of the "
        "two readings is wrong: check this line on the invoice."
    )
    [empty] = sr.compare_page([item(1)], [item(1, sku=None)], 2)
    assert "no item code where the result has HB-00001" in text(empty)
    unnumbered = [item(n, line_no=None) for n in (1, 2, 3)]
    [column] = sr.compare_page(unnumbered, [i | {"tax_rate": None} for i in unnumbered], 1)
    assert text(column) == (
        "On page 1, a second model reads the tax rate differently on 3 of 3 line items (for "
        "example at line item 1: no tax rate where the result has 20%). The two readings write "
        "this column differently, or one of them misread it: check the tax rate column on this "
        "page."
    )
    [codes] = sr.compare_page(
        [item(n) for n in (1, 2, 3)], [item(n, sku=f"hb {n:05d}") for n in (1, 2, 3)], 1
    )
    assert text(codes) == (
        "On page 1, a second model reads the item code differently on 3 of 3 line items (for "
        "example at position 1: item code hb 00001 instead of HB-00001). The two readings "
        "write this column differently, or one of them misread it: check the item code column "
        "on this page."
    )
    [unmatched] = sr.compare_page([item(1), item(2), item(7)], [item(1), item(2), item(4)], 3)
    assert text(unmatched) == (
        "On page 3, a second model finds as many line items as the result (3), but not at the "
        "same positions. Position 4 is not in the result. Position 7 of the result was not "
        "found by the second reading. Check this page on the invoice."
    )
    [counts] = sr.compare_page(unnumbered, unnumbered[:1], 5)
    assert text(counts) == (
        "On page 5, a second model finds 1 line item where the result has 3. The lines could "
        "not be matched one to one, so they were not compared. Check this page on the invoice."
    )


def test_item_codes_that_hold_the_word_sku() -> None:
    def item(no: int, sku: str) -> Line:
        return line(no) | {"sku": sku}

    # The values are quoted as read, "SKU" in them too (text() would take it for jargon).
    [inside] = sr.compare_page([item(1, "HB-00001")], [item(1, "Piano SKU 5")], 2)
    assert messages.issue_text(inside) == (
        "Position 1 on page 2 was read differently by a second model: item code Piano SKU 5 "
        "instead of HB-00001. One of the two readings is wrong: check this line on the invoice."
    )
    [listed] = sr.compare_page([item(1, "PACK, SKU 7")], [item(1, "X")], 2)
    assert messages.issue_text(listed) == (
        "Position 1 on page 2 was read differently by a second model: item code X instead of "
        "PACK, SKU 7. One of the two readings is wrong: check this line on the invoice."
    )


@pytest.mark.parametrize(
    ("sku", "quoted"),
    [
        ("HB\n00009", "HB 00009"),  # whitespace inside is collapsed, as the flow compares it
        ("HB\ud80009", "HB\ud80009"),
        ("H" * 200_000, "H" * 39 + "…"),  # the flow quotes 40 characters of a code
    ],
    ids=["line-break", "surrogate", "long"],
)
def test_an_item_code_is_quoted_as_the_flow_quotes_it(sku: str, quoted: str) -> None:
    [issue] = sr.compare_page([line(1) | {"sku": "HB-1"}], [line(1) | {"sku": sku}], 2)
    assert text(issue) == (
        f"Position 1 on page 2 was read differently by a second model: item code {quoted} "
        "instead of HB-1. One of the two readings is wrong: check this line on the invoice."
    )


def test_a_scan_page_too_large_to_open() -> None:
    res = run(drawn.declared_image_scan(30000, 30000), "application/pdf", never)
    assert text(one(res.issues, "scan_too_large")) == (
        "Page 1 holds an image too large to open (900,000,000 pixels; at most 50,000,000 can "
        "be opened), so it was not read. Save the scan at a lower resolution and upload it "
        "again."
    )


LIMITS = ql.Limits()


@pytest.mark.parametrize(
    ("reason", "worded"),
    [
        (
            ql.Reason("blurred", "", "reject", 120.0),
            (
                "The photo is too blurred to read: its sharpness is 120.0, and at least 140 is "
                "needed. Nothing was read; please take the photo again, holding the phone still."
            ),
        ),
        (
            ql.Reason("blurred", "", "warn", 300.0),
            (
                "The photo is not quite sharp (sharpness 300.0; 450 or more is safe). Amounts are "
                "usually read correctly, but item codes may be misread: check them."
            ),
        ),
        (
            ql.Reason("no_text", "", "reject", 3.0),
            (
                "No text was found in the photo, so nothing was read. Check that the photo shows "
                "the invoice and is close enough to read."
            ),
        ),
        (
            ql.Reason("no_text", "", "reject"),
            (
                "No text was found in the photo, so nothing was read. Check that the photo shows "
                "the invoice and is close enough to read."
            ),
        ),
        (
            ql.Reason("text_at_edge", "", "reject", 0.4, "left"),
            (
                "Text runs into the left edge of the photo: part of the invoice is cut off. "
                "Nothing was read; please take the photo again with the whole sheet in the frame."
            ),
        ),
        (
            ql.Reason("text_at_edge", "", "warn", 1.2, "bottom"),
            (
                "Text comes close to the bottom edge of the photo: part of the invoice may be cut "
                "off. Check that no line or total is missing."
            ),
        ),
        (
            ql.Reason("sheet_cut", "", "warn", None, "top"),
            (
                "The sheet runs out of the photo past the top edge. No text is near that edge, but "
                "check that nothing of the invoice is missing."
            ),
        ),
        (
            ql.Reason("text_at_edge", "", "reject", 0.4),
            (
                "Text runs into an edge of the photo: part of the invoice is cut off. Nothing was "
                "read; please take the photo again with the whole sheet in the frame."
            ),
        ),
        (
            ql.Reason("sheet_cut", "", "warn", 0.2),
            (
                "The sheet runs out of the photo past an edge. No text is near that edge, but "
                "check that nothing of the invoice is missing."
            ),
        ),
        (
            ql.Reason("sheet_edge_unseen", "", "warn"),
            (
                "No edge of the sheet is visible in the photo, so nothing confirms that the whole "
                "invoice is in it. Check that no line or total is cut off."
            ),
        ),
        (
            ql.Reason("dark", "", "warn", 120.5),
            (
                "The photo is dark (the paper's brightness is 120.5 of 255; below 150 counts as "
                "dark). Values read from it may contain errors: check them."
            ),
        ),
        (
            ql.Reason("glare", "", "warn", 0.034),
            (
                "Glare covers 3.4% of the sheet, so some text may be washed out. Check the values "
                "in the bright part of the photo."
            ),
        ),
        (
            ql.Reason("text_small", "", "warn", 9.5),
            (
                f"The text is small in the photo (its lines are 9.5 pixels tall; "
                f"{LIMITS.text_height_px:g} or more is safe). Values may be misread: check them, or "
                "take the photo again from closer."
            ),
        ),
    ],
    # "hand-built": a reason quality.reasons() does not produce (text_at_edge is always a
    # reject, no_text always has a count, and both side reasons always have a side), kept
    # for the wording's other branches.
    ids=[
        "blurred-reject",
        "blurred-warn",
        "no_text",
        "no_text-unmeasured-hand-built",
        "text_at_edge-reject",
        "text_at_edge-warn-hand-built",
        "sheet_cut",
        "text_at_edge-no-side-hand-built",
        "sheet_cut-no-side-hand-built",
        "sheet_edge_unseen",
        "dark",
        "glare",
        "text_small",
    ],
)
def test_photo(reason: ql.Reason, worded: str) -> None:
    assert text(pl._photo_issue(reason, LIMITS)) == worded


@pytest.mark.parametrize(
    ("reason", "worded"),
    [
        (
            ql.Reason("blurred", "", "reject", 40.2),
            (
                "Page 2 is too blurred to read (sharpness 40.2; at least 100 is needed), so it was "
                "not read. Scan it again at a higher resolution."
            ),
        ),
        (
            ql.Reason("blurred", "", "warn", 150.0),
            (
                "Page 2 is not quite sharp (sharpness 150.0; 200 or more is safe). Values read from "
                "it may contain errors: check them."
            ),
        ),
        (
            ql.Reason("dark", "", "warn", 101.0),
            (
                "Page 2 is dark (the paper's brightness is 101.0 of 255; below 150 counts as dark). "
                "Values read from it may contain errors: check them."
            ),
        ),
    ],
    ids=["blurred-reject", "blurred-warn", "dark"],
)
def test_scan_page(reason: ql.Reason, worded: str) -> None:
    assert text(pl._scan_issue(reason, 2, ql.ScanLimits())) == worded


@pytest.mark.parametrize(
    "issue",
    [
        pl._photo_issue(ql.Reason("sheet_cut", "", "reject", 0.2, "top"), LIMITS),
        pl._photo_issue(ql.Reason("dark", "", "reject", 120.5), LIMITS),
        pl._photo_issue(ql.Reason("glare", "", "reject", 0.034), LIMITS),
        pl._photo_issue(ql.Reason("text_small", "", "reject", 9.5), LIMITS),
        pl._scan_issue(ql.Reason("dark", "", "reject", 101.0), 2, ql.ScanLimits()),
    ],
    ids=["sheet_cut", "dark", "glare", "text_small", "scan-dark"],
)
def test_a_warning_raised_as_a_reject_keeps_the_flows_message(
    issue: Issue, caplog: pytest.LogCaptureFixture
) -> None:
    # Hand-built: quality.py raises these only as warnings, and their sentences say values
    # were read; as a reject nothing was, so the person reads the flow's own words.
    assert messages.issue_text(issue) == issue.message
    assert "could not be worded" in caplog.text


def test_the_photo_and_scan_texts_cover_every_reason_measured() -> None:
    photo = {f"photo_{c}" for c in _reason_codes("reasons", ("reject", "warn"))}
    scan = {f"scan_{c}" for c in _reason_codes("judge_page", ("reject", "warn"))}
    tested = {"photo_" + c for c in ("blurred", "no_text", "text_at_edge", "sheet_cut")}
    tested |= {"photo_sheet_edge_unseen", "photo_dark", "photo_glare", "photo_text_small"}
    assert photo == tested and scan == {"scan_blurred", "scan_dark"}


# --------------------------------------------------------------------------- #
# Through the view
# --------------------------------------------------------------------------- #


def test_the_view_words_an_issue_with_its_line() -> None:
    lines = [{"sku": "AB-1001", "quantity": 2, "unit_price": 10, "amount": 21.0, "page": 1}]
    issues = checks.check_grounding(doc(lines), ev.build([PAGE], "text_layer"))
    res = Result(
        status="needs_review",
        header=header_for(lines),
        lines=lines,
        checkpoints=[],
        issues=issues,
        pages=[],
        evidence_source="text_layer",
        duration_s=1.0,
        calls=1,
        cost_usd=0.0,
        prompt_tokens=0,
        completion_tokens=0,
    )
    [issue] = json.loads(result_view(res).model_dump_json())["issues"]
    assert issue["message"] == (
        "This line's amount (21.0) is not printed on the page. The model may have misread it: "
        "check it on the invoice."
    )


# Hand-built issues: the flow never builds these; they pin the fault boundary.


def test_an_issue_without_a_sentence_keeps_the_flows_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Neither values nor a form for a check whose sentences need one, and an unknown code.
    issue = Issue("tax", "some new way the flow says it")
    assert messages.issue_text(issue) == "some new way the flow says it"
    assert messages.issue_text(Issue("a_new_code", "as written")) == "as written"
    assert caplog.text.count("could not be worded") == 2


def test_a_wording_that_breaks_keeps_the_flows_message(caplog: pytest.LogCaptureFixture) -> None:
    # A header value whose issue names no field of the header.
    issue = Issue("grounding_header", "as written", values=HeaderNotPrinted("INV-1"))
    assert messages.issue_text(issue) == issue.message
    assert "could not be worded" in caplog.text


@pytest.mark.parametrize(
    "issue",
    [
        Issue("tax", "as written", values=SubtotalOff(Decimal(1), Decimal(2), Decimal(-1))),
        Issue(
            "doc_sums",
            "as written",
            form="totals_missing",
            values=SubtotalOff(Decimal(1), Decimal(2), Decimal(-1)),
        ),
    ],
    ids=["shape-of-another-code", "form-and-values"],
)
def test_values_that_do_not_fit_the_issue_keep_the_flows_message(
    issue: Issue, caplog: pytest.LogCaptureFixture
) -> None:
    # Hand-built: a programming error at a call site is never worded as another sentence.
    assert messages.issue_text(issue) == "as written"
    assert "could not be worded" in caplog.text


def test_a_row_without_a_page_still_reads_as_a_sentence() -> None:
    # Hand-built: layout_pages always gives a row its page.
    issue = Issue("layout_row", "stray_words: x y", values=LayoutRow("stray_words", "x y"))
    assert text(issue).startswith("The row of a line item also holds words")
