"""The fields named by the issues the flow raises itself, beside those of the checks.

An issue about a page or the file names no field ([]); the issues of the layout road
name columns (no line). Fake OpenRouter, no network."""

import asyncio
from decimal import Decimal

import httpx
import pytest

from invoice_recognition.recognition import evidence as ev
from invoice_recognition.recognition import layout as lay
from invoice_recognition.recognition import pipeline as pl
from invoice_recognition.recognition import quality as ql
from invoice_recognition.recognition.checks import Issue
from invoice_recognition.recognition.config import Config
from invoice_recognition.recognition.results import Result
from invoice_recognition.shapes import JSONObject

from drawn import scan, scanned
from fakes import Answer, Server, SlowServer, header_for, line, never, run
from golden_cases import CASES
from samples import saved_layout, truth_doc

COLUMNS = ["line.quantity", "line.unit_price"]


def of(res: Result, check: str) -> list[Issue]:
    return [i for i in res.issues if i.check == check]


def fields_of(res: Result, check: str) -> list[list[str]]:
    return [i.fields for i in of(res, check)]


def answers(pages: dict[int, list[JSONObject]]) -> dict[int, list[Answer]]:
    return {p: [rows] for p, rows in pages.items()}


# --------------------------------------------------------------------------- #
# A page read by the model
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "answer",
    [httpx.Response(500, text="boom"), "not an answer"],
    ids=["request-failed", "not-json"],
)
def test_page_that_could_not_be_read_names_no_field(answer: Answer) -> None:
    rows = [line(1)]
    res = run(scan(1), "application/pdf", Server(header_for(rows), {1: [answer]}))
    assert fields_of(res, "page_read") == [[]]
    assert res.to_json()["issues"][0]["fields"] == []


def test_page_whose_reading_broke_names_no_field(monkeypatch: pytest.MonkeyPatch) -> None:
    async def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("a bug")

    monkeypatch.setattr(pl, "read_page", broken)
    rows = [line(1)]
    res = run(scan(1), "application/pdf", Server(header_for(rows), {1: [rows]}))
    assert [(i.message, i.fields) for i in of(res, "page_read")] == [("RuntimeError: a bug", [])]


def test_empty_page_names_no_field() -> None:
    p1 = [line(1)]
    res = run(scan(2), "application/pdf", Server(header_for(p1), {1: [p1], 2: [[]]}))
    assert fields_of(res, "page_empty") == [[]]


def test_dropped_rows_name_no_field() -> None:
    rows = [line(1), {"description": "Section A"}]
    res = run(scan(1), "application/pdf", Server(header_for([line(1)]), {1: [rows]}))
    assert fields_of(res, "rows_dropped") == [[]]


def test_attempts_that_disagree_name_no_field() -> None:
    first = [line(1), line(2, amount=99.0), line(3)]
    second = [line(1), line(3)]
    res = run(scan(1), "application/pdf", Server(header_for(second), {1: [first, second]}))
    assert fields_of(res, "attempts_disagree") == [[]]


# --------------------------------------------------------------------------- #
# The layout road
# --------------------------------------------------------------------------- #


def test_layout_notes_name_the_column_and_have_no_line() -> None:
    notes = [
        {"page": 1, "row": 3, "kind": "no_tax_rate", "text": "A 2 10.00 20.00"},
        {"page": 1, "row": 4, "kind": "stray_words", "text": "A 2 10.00 20.00 x"},
        {"page": 1, "row": 5, "kind": "text_under_item", "text": "more words"},
        {"page": 1, "row": 6, "kind": "unclassified_amount", "text": "Subtotal 20.00"},
    ]
    parsed = lay.Parsed([], [], notes)
    pages, harmless = pl.layout_pages(parsed, 1, ev.build(["Invoice INV-1"], "text_layer"))
    assert [(i.check, i.line, i.fields) for i in pages[1].issues] == [
        ("layout_row", None, ["line.tax_rate"]),
        ("layout_row", None, []),  # which cell the words belong to is not known
        ("layout_row", None, ["line.description"]),
    ]
    assert [n["kind"] for n in harmless] == ["unclassified_amount"]  # a note, not an issue


def test_checks_of_a_page_found_by_code_that_broke_name_no_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*args: object) -> list[Issue]:
        raise RuntimeError("a bug")

    monkeypatch.setattr(pl, "check_page", broken)
    parsed = lay.Parsed([{"sku": "A", "amount": Decimal(1), "page": 1}], [], [])
    pages, _ = pl.layout_pages(parsed, 1, ev.build(["A 1"], "text_layer"))
    assert [(i.check, i.fields) for i in pages[1].issues] == [("page_read", [])]


def swapped(layout: JSONObject) -> JSONObject:
    flip = {"quantity": "unit_price", "unit_price": "quantity"}
    columns = [c | {"field": flip.get(c["field"], c["field"])} for c in layout["columns"]]
    return layout | {"columns": columns}


def test_crossed_columns_are_one_issue_about_the_two_columns() -> None:
    data, _, header, pages = truth_doc("l_classic")
    server = Server(header, answers(pages), swapped(saved_layout("l_classic")))
    res = run(data, "application/pdf", server)
    assert res.layout["status"] == "swapped"
    (issue,) = [i for i in res.issues if i.check.startswith("layout_")]
    assert (issue.check, issue.line, issue.page, issue.fields) == (
        "layout_swapped",
        None,
        1,
        COLUMNS,
    )
    # No issue of any row names the two columns: the document was read by the model.
    assert not [i for i in res.issues if i.line is not None and set(COLUMNS) & set(i.fields)]
    assert res.to_json()["issues"][-1]["fields"] == COLUMNS


def test_rows_that_differ_in_another_way_name_no_field() -> None:
    data, _, header, pages = truth_doc("l_classic")
    wrong = [r | {"quantity": r["quantity"] + 1} for r in pages[1]]
    server = Server(header, answers(pages) | {1: [wrong]}, saved_layout("l_classic"))
    res = run(data, "application/pdf", server)
    assert fields_of(res, "layout_mismatch") == [[]]


def test_unconfirmed_layout_names_the_two_columns() -> None:
    data, _, header, pages = truth_doc("l_classic")
    failing = answers(pages) | {1: [httpx.Response(500, text="boom")]}
    res = run(data, "application/pdf", Server(header, failing, saved_layout("l_classic")))
    assert [(i.line, i.fields) for i in of(res, "layout_unverified")] == [(None, COLUMNS)]


# --------------------------------------------------------------------------- #
# The overall deadline
# --------------------------------------------------------------------------- #

DEADLINE = 0.5
SLOW = [30.0]  # far beyond the deadline


def run_slow(data: bytes, server: SlowServer, cfg: Config) -> Result:
    async def go() -> Result:
        async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
            return await pl.process(data, "application/pdf", client, cfg)

    return asyncio.run(go())


def test_page_not_read_at_the_deadline_and_the_deadline_name_no_field() -> None:
    rows: dict[int, list[Answer]] = {p: [[line(p)]] for p in (1, 2)}
    server = SlowServer(Server(header_for([line(1), line(2)]), rows), {2: SLOW})
    res = run_slow(scan(2), server, Config(scan_quality=False, deadline_s=DEADLINE))
    (unread,) = of(res, "page_read")
    assert unread.message.startswith("Page 2 was not read") and unread.fields == []
    assert fields_of(res, "deadline_exceeded") == [[]]


def test_witness_cut_by_the_deadline_names_the_two_columns() -> None:
    data, _, header, pages = truth_doc("s_classic")
    server = SlowServer(Server(header, answers(pages), saved_layout("s_classic")), {1: SLOW})
    res = run_slow(data, server, Config(deadline_s=DEADLINE))
    (issue,) = of(res, "layout_unverified")
    assert "stopped by the time limit" in issue.message
    assert (issue.line, issue.fields) == (None, COLUMNS)


# --------------------------------------------------------------------------- #
# Photo and scan quality, a file that cannot be read
# --------------------------------------------------------------------------- #

PHOTO_CODES = [
    "no_text",
    "text_at_edge",
    "sheet_cut",
    "sheet_edge_unseen",
    "blurred",
    "dark",
    "glare",
    "text_small",
]


@pytest.mark.parametrize("severity", ["reject", "warn"])
@pytest.mark.parametrize("code", PHOTO_CODES)
def test_photo_quality_issue_names_no_field(code: str, severity: str) -> None:
    issue = pl._photo_issue(ql.Reason(code, "m", severity, 1.0, "right"), ql.Limits())
    assert (issue.check, issue.fields) == (f"photo_{code}", [])


@pytest.mark.parametrize("severity", ["reject", "warn"])
@pytest.mark.parametrize("code", ["blurred", "dark"])
def test_scan_quality_issue_names_no_field(code: str, severity: str) -> None:
    issue = pl._scan_issue(ql.Reason(code, "m", severity, 1.0), 2, ql.ScanLimits())
    assert (issue.check, issue.page, issue.fields) == (f"scan_{code}", 2, [])


def test_scan_without_text_names_no_field() -> None:
    res = run(scanned("blank", "blank"), "application/pdf", never)
    assert fields_of(res, "scan_no_text") == [[]]


def test_unreadable_file_names_no_field() -> None:
    res = run(b"%PDF-1.4 truncated", "application/pdf", never)
    assert fields_of(res, "unreadable_file") == [[]]


# --------------------------------------------------------------------------- #
# End to end: a scan whose page is read again
# --------------------------------------------------------------------------- #


def run_case(name: str) -> Result:
    (case,) = [c for c in CASES if c.name == name]
    data, mime = case.data()

    async def go() -> Result:
        async with httpx.AsyncClient(transport=httpx.MockTransport(case.model())) as client:
            return await pl.process(data, mime, client, Config(**case.config))

    return asyncio.run(go())


def test_fields_of_a_scan_read_again_end_to_end() -> None:
    res = run_case("scan_retries")
    out = res.to_json()
    assert [(i["check"], i["line"], i["fields"]) for i in out["issues"]] == [
        ("line_math", 1, ["line.quantity", "line.unit_price", "line.amount"]),
        ("rows_dropped", None, []),
        ("doc_sums", None, ["header.subtotal", "line.amount"]),
        ("tax", None, ["header.tax_total", "line.tax_rate"]),
    ]
    # The copy of a page's issues in the result names the same fields.
    assert [[i["fields"] for i in p["issues"]] for p in out["pages"]] == [
        [["line.quantity", "line.unit_price", "line.amount"]],
        [[]],
        [],
    ]
