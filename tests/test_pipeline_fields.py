"""The fields named by the issues the flow raises itself, beside those of the checks.

An issue about a page or the file names no field ([]); the issues of the layout road
name columns (no line). Fake OpenRouter, no network."""

import asyncio
import time
from collections import Counter
from decimal import Decimal
from typing import Any

import httpx
import pytest

from invoice_recognition.recognition import checks
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


def test_one_wrong_amount_of_a_scan_read_again_end_to_end() -> None:
    # Line 2 reads 2 x 10.0 = 99.0; the header was printed for 20.0.
    out = run_case("scan_retries").to_json()
    line_math, rows_dropped, doc_sums, tax = out["issues"]
    # No quantity: 99.0 / 10.0 = 9.9 is no quantity for one read whole.
    assert line_math["suggestions"] == [
        {"set": {"line.unit_price": 49.5}, "source": "arithmetic"},
        {"set": {"line.amount": 20}, "source": "arithmetic", "agrees_with": "header.subtotal"},
    ]
    # The subtotal and the tax both pass with 20.0 for the amount: they follow from it.
    assert doc_sums["follows_from"] == [0] and tax["follows_from"] == [0]
    assert "follows_from" not in line_math and "follows_from" not in rows_dropped
    assert "suggestions" not in doc_sums and "suggestions" not in rows_dropped
    assert out["pages"][0]["issues"] == [line_math]  # the page shows the same issue


# --------------------------------------------------------------------------- #
# A wrong line amount and the issues it causes
# --------------------------------------------------------------------------- #

WRONG = [line(1), line(2, amount=99.0)]  # 2 x 10.0 = 20.0, read as 99.0


def read(rows: list[JSONObject], header: JSONObject) -> Result:
    return run(scan(1), "application/pdf", Server(header, {1: [rows]}))


def agreeing(res: Result) -> list[JSONObject]:
    (issue,) = of(res, "line_math")
    return [s["set"] for s in issue.suggestions if "agrees_with" in s]


def test_the_alternative_that_makes_the_subtotal_add_up_says_so() -> None:
    res = read(WRONG, header_for([line(1), line(2)]))  # subtotal 40.0
    assert agreeing(res) == [{"line.amount": 20}]
    (issue,) = of(res, "line_math")
    assert [s.get("agrees_with") for s in issue.suggestions] == [None, "header.subtotal"]


def test_no_alternative_agrees_when_none_makes_the_subtotal_add_up() -> None:
    res = read(WRONG, header_for([line(1), line(2)]) | {"subtotal": 50.0})
    assert len(of(res, "line_math")[0].suggestions) == 2 and agreeing(res) == []
    # The subtotal is wrong for another reason: it follows from no line.
    (doc_sums,) = [i for i in of(res, "doc_sums") if i.message.startswith("lines add up")]
    assert doc_sums.follows_from == []
    assert "follows_from" not in doc_sums.to_json()


@pytest.mark.parametrize("quantity", [2, 2.5], ids=["price", "quantity-and-price"])
def test_a_suggested_quantity_or_price_never_agrees(quantity: float) -> None:
    # The subtotal was printed for the amount as read: a quantity or a price that fits it
    # leaves the lines adding up, but agrees with nothing - it does not change the amount.
    rows = [line(1), line(2, amount=99.0) | {"quantity": quantity}]
    res = read(rows, header_for(rows))
    assert [i.check for i in res.issues] == ["line_math"]
    (issue,) = of(res, "line_math")
    assert [next(iter(s["set"])) for s in issue.suggestions][-2:] == [
        "line.unit_price",
        "line.amount",
    ]
    assert len(issue.suggestions) == (3 if quantity == 2.5 else 2)
    assert agreeing(res) == []


def test_control_totals_on_the_page_and_in_the_document_follow_from_the_line() -> None:
    rows = [
        line(1),
        {"total_label": "Section A", "amount": 20.0},
        line(2, amount=99.0),
        {"total_label": "Section B", "amount": 20.0},
        line(3),
    ]
    res = read(rows, header_for([line(1), line(2), line(3)]))
    section_b = 'lines 2-2 add up to 99.0, but "Section B" says 20.0'
    assert [(n, i.check, i.message, i.follows_from) for n, i in enumerate(res.issues)] == [
        (0, "line_math", "2 x 10.0 = 20.00, but amount is 99.0", []),
        (1, "checkpoints", section_b, [0]),  # found on the page
        (2, "doc_sums", "lines add up to 139.0, invoice subtotal is 60.0 (difference 79.0)", [0]),
        (3, "tax", "tax computed from lines is 27.80, invoice says 12.0", [0]),
        (4, "checkpoints", section_b, [0]),  # found in the document
    ]


def test_two_wrong_lines_both_explain_the_subtotal() -> None:
    rows = [line(1, amount=21.0), line(2, amount=99.0), line(3)]
    res = read(rows, header_for([line(1), line(2), line(3)]))
    ids = [n for n, i in enumerate(res.issues) if i.check == "line_math"]
    assert ids == [0, 1]
    assert [(i.check, i.follows_from) for i in res.issues[2:]] == [("doc_sums", ids), ("tax", ids)]
    # Neither line alone makes the subtotal add up, but each does with the other at
    # quantity x unit price: both amounts agree.
    assert [[s.get("agrees_with") for s in res.issues[n].suggestions] for n in ids] == [
        [None, "header.subtotal"],  # 21.0 / 2 = 10.50 is a price; 21.0 / 10.0 no quantity
        [None, "header.subtotal"],
    ]


def test_an_amount_agrees_when_another_line_is_wrong_in_its_quantity() -> None:
    # 2 x 10.0, 3 x 5.0 and 4 x 2.5 add up to 45.0; read as 2 x 10.0 = 21.0, with the
    # quantity of line 2 misread as 4, and 4 x 2.5 = 10.0.
    rows = [
        line(1, amount=21.0),
        line(2, qty=4, price=5.0, amount=15.0),
        line(3, qty=4, price=2.5),
    ]
    res = read(rows, header_for([line(1), line(2, qty=3, price=5.0), line(3, qty=4, price=2.5)]))
    first, second = of(res, "line_math")
    # 20.0 for line 1 makes the lines add up with line 2 as read; 20.0 for line 2 does not.
    assert [s.get("agrees_with") for s in first.suggestions] == [None, "header.subtotal"]
    assert second.suggestions == [
        {"set": {"line.quantity": 3}, "source": "arithmetic"},
        {"set": {"line.unit_price": 3.75}, "source": "arithmetic"},
        {"set": {"line.amount": 20}, "source": "arithmetic"},
    ]


def test_control_totals_on_two_pages_with_the_same_message_stay_on_their_page() -> None:
    page = [
        {"total_label": "Section", "amount": 20.0},
        line(1, amount=99.0),
        {"total_label": "Section", "amount": 20.0},
    ]
    header = header_for([line(1), line(1)])
    res = run(scan(2), "application/pdf", Server(header, {1: [page], 2: [page]}))
    on_pages = [
        (n, i) for n, i in enumerate(res.issues) if i in res.pages[0].issues + res.pages[1].issues
    ]
    section = 'lines 1-1 add up to 99.0, but "Section" says 20.0'
    assert [(n, i.check, i.page, i.message, i.follows_from) for n, i in on_pages] == [
        (0, "line_math", 1, "2 x 10.0 = 20.00, but amount is 99.0", []),
        (1, "checkpoints", 1, section, [0]),
        (2, "line_math", 2, "2 x 10.0 = 20.00, but amount is 99.0", []),
        (3, "checkpoints", 2, section, [2]),
    ]


def test_a_missing_amount_is_offered_and_agrees() -> None:
    res = read([line(1), line(2) | {"amount": None}], header_for([line(1), line(2)]))
    assert [(i.check, i.follows_from) for i in res.issues] == [
        ("line_math", []),
        ("doc_sums", []),  # some lines have no amount
        ("doc_sums", [0]),  # the lines against the subtotal
        ("tax", [0]),
    ]
    assert agreeing(res) == [{"line.amount": 20}]


def test_a_control_total_on_page_2_follows_from_the_line_by_its_position_in_the_result() -> None:
    page_1 = [
        line(1, amount=21.0),
        {"description": "Section A"},  # dropped: one more issue on page 1
        line(2),
        {"total_label": "Carried forward", "amount": 40.0},
    ]
    page_2 = [
        {"total_label": "Carried forward", "amount": 40.0},
        line(3, amount=99.0),
        line(4),
        {"total_label": "Carried forward", "amount": 80.0},
    ]
    header = header_for([line(n) for n in range(1, 5)])
    res = run(scan(2), "application/pdf", Server(header, {1: [page_1], 2: [page_2]}))
    assert [(n, i.check, i.page, i.follows_from) for n, i in enumerate(res.issues)] == [
        (0, "line_math", 1, []),
        (1, "rows_dropped", 1, []),
        (2, "line_math", 2, []),
        (3, "checkpoints", 2, [2]),  # found on page 2, where its line is the first
        (4, "doc_sums", None, [0, 2]),
        (5, "tax", None, [0, 2]),
        (6, "checkpoints", 1, [0]),
        (7, "checkpoints", 2, [2]),
    ]


def linked(lines: list[JSONObject], header: JSONObject) -> list[Issue]:
    """The issues of a document of `lines` and `header` as the flow raises them, linked."""
    doc: JSONObject = {"header": header, "lines": lines, "checkpoints": []}
    issues = checks.check_line_math(doc)
    start = len(issues)
    issues += checks.check_doc_sums(doc) + checks.check_tax(doc) + checks.check_checkpoints(doc)
    pl.link_issues([], doc, issues, range(start, len(issues)))
    return issues


def test_a_quantity_read_1000_times_too_small_keeps_its_amount() -> None:
    # 0.002 x 1000.0 = 2.00, but 2000.0 is printed: the quantity is in doubt, not the
    # amount, which stays as read when the other line gets 2 x 10.0 = 20.0 for 99.0.
    thousand = line(1, price=1000.0, amount=2000.0)
    issues = linked(
        [thousand | {"quantity": 0.002}, line(2, amount=99.0)], header_for([thousand, line(2)])
    )
    assert [(i.check, i.follows_from) for i in issues] == [
        ("line_math", []),
        ("line_math", []),
        ("doc_sums", [1]),
        ("tax", [1]),
    ]
    assert issues[0].fields == ["line.quantity"] and "line.amount" in issues[1].fields
    assert issues[0].suggestions == [{"set": {"line.quantity": 2}, "source": "arithmetic"}]
    # 2000.0 + 20.0 is the subtotal.
    assert issues[1].suggestions[-1] == {
        "set": {"line.amount": 20},
        "source": "arithmetic",
        "agrees_with": "header.subtotal",
    }


def test_linking_that_breaks_leaves_the_issues_as_they_are(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("a bug")

    monkeypatch.setattr(checks, "explained_by_lines", broken)
    res = read(WRONG, header_for([line(1), line(2)]))
    assert [i.check for i in res.issues] == ["line_math", "doc_sums", "tax"]
    assert all(i.follows_from == [] for i in res.issues) and agreeing(res) == []
    assert res.status == "needs_review"


# --------------------------------------------------------------------------- #
# A long document misread throughout
# --------------------------------------------------------------------------- #


def misread_throughout(wrong: int) -> tuple[JSONObject, dict[int, list[Answer]]]:
    """The answers of a model reading the scan of xl_grouped (1,250 lines on 34 pages)
    with the amount of the first `wrong` lines one more than printed (quantity and unit
    price right), and a running total at the top (from page 2) and the bottom of every
    page: a check of the document and one of the page for each page with such a line."""
    _, _, header, pages = truth_doc("xl_grouped")
    answers: dict[int, list[Answer]] = {}
    running, n = Decimal(0), 0
    for p, rows in pages.items():
        out: list[JSONObject] = []
        if p > 1:
            out.append({"total_label": "Carried forward", "amount": float(running)})
        for row in rows:
            running += Decimal(str(row["amount"]))
            if n < wrong:
                row = row | {"amount": float(Decimal(str(row["amount"])) + 1)}
            out.append(row)
            n += 1
        out.append({"total_label": "Carried forward", "amount": float(running)})
        answers[p] = [out]
    return header, answers


def timed_links(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The time each linking pass of the flow takes, from now on."""
    spent: list[float] = []
    real = pl.link_issues

    def timed(*args: Any, **kwargs: Any) -> None:
        started = time.perf_counter()
        real(*args, **kwargs)
        spent.append(time.perf_counter() - started)

    monkeypatch.setattr(pl, "link_issues", timed)
    return spent


# xl_grouped as a document has 1,250 lines, 67 control totals and 3 tax rows: this many
# wrong lines are still tested one by one.
TESTED = checks.TEST_BUDGET // (1250 + 67 + 3)


@pytest.mark.parametrize("wrong", [TESTED, TESTED + 1, 1250], ids=["tested", "untested", "every"])
def test_linking_a_long_document_misread_throughout_takes_little_time(
    monkeypatch: pytest.MonkeyPatch, wrong: int
) -> None:
    spent = timed_links(monkeypatch)
    header, answers = misread_throughout(wrong)
    res = run(scan(len(answers)), "application/pdf", Server(header, answers))
    # Measured on a laptop: 0.05 s with every line wrong (49 s when each issue re-ran its
    # check once per line), 0.01 s with lines tested one by one.
    assert spent[0] < 1.0

    ids = [n for n, i in enumerate(res.issues) if i.check == "line_math"]
    assert len(ids) == wrong
    assert [res.issues[n].line for n in ids] == list(range(wrong))  # the first lines
    rate = {n: res.lines[at]["tax_rate"] for at, n in enumerate(ids)}
    pages = {res.issues[n].page for n in ids}
    others = [i for i in res.issues if i.check != "line_math"]
    assert Counter(i.check for i in others) == {
        "doc_sums": 1,
        "tax": 3,
        "checkpoints": 2 * len(pages) - 1,  # page 1 has no total at its top
    }
    for issue in others:
        if issue.check == "doc_sums":
            assert issue.follows_from == ids
        elif issue.check == "tax":  # "lines at 20% add up to ..."
            r = int(issue.message.split()[2].rstrip("%"))
            assert issue.follows_from == [n for n in ids if rate[n] == r]
        else:  # the page's own lines, on the page and in the document
            assert issue.follows_from == [n for n in ids if res.issues[n].page == issue.page]
    # With every other line at quantity x unit price, each suggested amount makes the
    # lines add up to the subtotal.
    assert all(res.issues[n].suggestions[-1].get("agrees_with") for n in ids)


def one_long_page(
    n: int, failing_totals: bool, rates: bool
) -> tuple[JSONObject, dict[int, list[Answer]]]:
    """One page of `n` lines of 2 x 10.0, every 25th read as 99.0, with a control total
    after every two lines (failing whatever the amounts, or not); with `rates`, every
    line has a tax rate of its own, printed with a wrong base."""
    rows: list[JSONObject] = []
    for i in range(n):
        item = line(i + 1, amount=99.0 if i % 25 == 0 else None)
        if rates:
            item["tax_rate"] = f"{1 + i / 100:.2f}"
        rows.append(item)
        if i % 2 == 1:
            rows.append({"total_label": "Section", "amount": 41.0 if failing_totals else 40.0})
    header = header_for([line(i + 1) for i in range(n)])
    if rates:
        header["tax_breakdown"] = [
            {"rate": f"{1 + i / 100:.2f}", "base": 21.0, "tax": 0} for i in range(n)
        ]
    return header, {1: [rows]}


@pytest.mark.parametrize(
    ("failing_totals", "rates"),
    [(False, False), (True, False), (True, True)],
    ids=["control-totals", "failing-control-totals", "failing-tax-rates"],
)
def test_linking_one_long_page_takes_little_time(
    monkeypatch: pytest.MonkeyPatch, failing_totals: bool, rates: bool
) -> None:
    spent = timed_links(monkeypatch)
    header, answers = one_long_page(1250, failing_totals, rates)
    res = run(scan(1), "application/pdf", Server(header, answers))
    # Measured on a laptop: 0.01-0.02 s; 0.13 s, 0.14 s and 0.08 s when up to 50 wrong
    # lines were tested one by one.
    assert spent[0] < 1.0
    ids = [n for n, i in enumerate(res.issues) if i.check == "line_math"]
    assert len(ids) == 50
    (doc_sums,) = of(res, "doc_sums")
    assert doc_sums.follows_from == ids
    sections = of(res, "checkpoints")
    if failing_totals:  # they fail with the amounts at quantity x unit price too
        assert all(i.follows_from == [] for i in sections)
    else:  # the page's, where the first total is only a base, then the document's
        assert [i.follows_from for i in sections] == [[n] for n in ids[1:] + ids]
